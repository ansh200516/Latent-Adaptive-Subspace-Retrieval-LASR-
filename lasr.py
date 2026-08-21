"""LASR: Latent Adaptive Subspace Retrieval.

Online loop from the README:
  1. ANN (FAISS HNSW) retrieves a K-candidate pool.
  2. SVD (or query-weighted PCA) builds a query-adaptive low-rank
     thematic subspace from the high-dimensional candidate embeddings.
  3. Candidates are rescored by thematic fit in that subspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_scores(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    q = l2_normalize(np.asarray(query, dtype=np.float64).reshape(1, -1))[0]
    e = l2_normalize(np.asarray(corpus, dtype=np.float64))
    return e @ q


def fit_thematic_subspace(
    embeddings: np.ndarray,
    n_components: int = 3,
    weights: Optional[np.ndarray] = None,
    mean_centering: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """SVD thematic subspace. Optional row weights = query-cosine (weighted PCA)."""
    E = np.asarray(embeddings, dtype=np.float64)
    if E.ndim == 1:
        E = E.reshape(1, -1)
    k, d = E.shape
    p = max(1, min(n_components, k, d))

    if k == 1:
        v = l2_normalize(E[0])
        return v.reshape(1, -1), np.array([1.0]), 1.0

    if weights is None:
        mu = E.mean(axis=0) if mean_centering else 0.0
        X = E - mu
    else:
        w = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-6, None)
        w = w / w.sum()
        mu = (w[:, None] * E).sum(axis=0) if mean_centering else 0.0
        X = (E - mu) * np.sqrt(w)[:, None]

    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    energy = float((S[:p] ** 2).sum() / (np.maximum((S ** 2).sum(), 1e-12)))
    return Vt[:p], S[:p].astype(np.float64), energy


def centrality_scores(
    embeddings: np.ndarray,
    components: np.ndarray,
    singular_values: np.ndarray,
) -> np.ndarray:
    """README score: || e_i Vp Σp ||_2^2."""
    E = np.asarray(embeddings, dtype=np.float64)
    Vp = np.asarray(components, dtype=np.float64)
    s = np.asarray(singular_values, dtype=np.float64).reshape(-1)
    if Vp.ndim == 1:
        Vp = Vp.reshape(1, -1)
    proj = E @ Vp.T
    weighted = proj * s.reshape(1, -1)
    return np.sum(weighted ** 2, axis=1)


def vec_subspace_similarity(
    vector: np.ndarray,
    components: np.ndarray,
    singular_values: Optional[np.ndarray] = None,
    weighted: bool = True,
) -> float:
    """Alignment of a vector with a subspace (from the project's EL scorer)."""
    v = np.asarray(vector, dtype=np.float64).reshape(-1)
    Vp = np.asarray(components, dtype=np.float64)
    if Vp.ndim == 1:
        return float(np.abs(cosine_scores(v, Vp.reshape(1, -1))[0]))
    if weighted and singular_values is not None:
        s = np.asarray(singular_values, dtype=np.float64).reshape(-1)
        mat = (v @ (Vp.T * s)) / (np.sum(s) + 1e-12)
    else:
        mat = v @ Vp.T
    denom = np.linalg.norm(v)
    if denom == 0:
        return 0.0
    return float(np.sqrt(np.sum(mat ** 2)) / denom)


def fit_labeled_subspaces(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_components: int = 1,
    skip_label: int = 0,
    mean_centering: bool = False,
) -> dict:
    """One SVD subspace per discrete label (offline type model)."""
    models = {}
    for lab in sorted(set(labels.tolist())):
        if lab == skip_label:
            continue
        idx = np.where(labels == lab)[0]
        if len(idx) == 0:
            continue
        comps, singular, energy = fit_thematic_subspace(
            embeddings[idx],
            n_components=min(n_components, len(idx)),
            mean_centering=mean_centering,
        )
        centroid = l2_normalize(embeddings[idx].astype(np.float64).mean(axis=0))
        models[int(lab)] = {
            "components": comps,
            "singular_values": singular,
            "energy": energy,
            "centroid": centroid,
            "indices": idx,
        }
    return models


def type_subspace_scores(query: np.ndarray, models: dict, weighted: bool = True) -> dict:
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    return {
        lab: vec_subspace_similarity(
            q, m["components"], m["singular_values"], weighted=weighted
        )
        for lab, m in models.items()
    }


def route_by_type_scores(
    query: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    score_by_type: dict,
) -> np.ndarray:
    """Rank corpus by (type score, then within-type cosine)."""
    cos = cosine_scores(query, embeddings)
    type_fit = np.array([score_by_type.get(int(lab), -1.0) for lab in labels], dtype=np.float64)
    keys = np.stack([type_fit, cos], axis=1)
    return np.lexsort((-keys[:, 1], -keys[:, 0]))


def route_by_type_subspace(
    query: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    models: dict,
    weighted: bool = True,
) -> np.ndarray:
    return route_by_type_scores(
        query, embeddings, labels, type_subspace_scores(query, models, weighted=weighted)
    )


def route_by_type_centroid(
    query: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    models: dict,
) -> np.ndarray:
    q = l2_normalize(np.asarray(query, dtype=np.float64).reshape(1, -1))[0]
    scores = {lab: float(q @ m["centroid"]) for lab, m in models.items()}
    return route_by_type_scores(query, embeddings, labels, scores)


def subspace_cosine_scores(
    query: np.ndarray,
    embeddings: np.ndarray,
    components: np.ndarray,
    singular_values: Optional[np.ndarray] = None,
    weighted: bool = True,
) -> np.ndarray:
    """Cosine(query, candidate) after projection onto the thematic subspace."""
    E = np.asarray(embeddings, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64).reshape(1, -1)
    Vp = np.asarray(components, dtype=np.float64)
    if Vp.ndim == 1:
        Vp = Vp.reshape(1, -1)
    scale = np.ones(Vp.shape[0], dtype=np.float64)
    if weighted and singular_values is not None:
        s = np.asarray(singular_values, dtype=np.float64).reshape(-1)
        scale = s / (np.sum(s) + 1e-12)
    q_p = l2_normalize((q @ Vp.T) * scale.reshape(1, -1))
    e_p = l2_normalize((E @ Vp.T) * scale.reshape(1, -1))
    return (e_p @ q_p.T).reshape(-1)


@dataclass
class LASRResult:
    indices: np.ndarray
    scores: np.ndarray
    energy_retained: float
    candidate_indices: np.ndarray
    cosine_candidate_scores: np.ndarray


class LASRRetriever:
    """ANN shortlist + SVD/WPCA thematic rerank."""

    def __init__(
        self,
        embeddings: np.ndarray,
        index=None,
        k: int = 15,
        n_components: int = 3,
        score: str = "subspace_cosine",
        weighted_pca: bool = True,
        mean_centering: bool = True,
    ):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.index = index
        self.k = k
        self.n_components = n_components
        self.score = score
        self.weighted_pca = weighted_pca
        self.mean_centering = mean_centering

    def _shortlist(self, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if self.index is not None:
            import faiss

            qn = q.copy()
            faiss.normalize_L2(qn)
            sims, idx = self.index.search(qn, k)
            return idx[0], sims[0]
        sims = cosine_scores(q[0], self.embeddings)
        idx = np.argsort(-sims)[:k]
        return idx, sims[idx]

    def retrieve(self, query_vec: np.ndarray, k: Optional[int] = None) -> LASRResult:
        pool_k = min(k or self.k, len(self.embeddings))
        cand_idx, cand_cos = self._shortlist(query_vec, pool_k)
        valid = cand_idx >= 0
        cand_idx = cand_idx[valid]
        cand_cos = np.asarray(cand_cos[valid], dtype=np.float64)
        E = self.embeddings[cand_idx].astype(np.float64)
        q = np.asarray(query_vec, dtype=np.float64).reshape(-1)

        weights = cand_cos if self.weighted_pca else None
        components, singular, energy = fit_thematic_subspace(
            E,
            n_components=self.n_components,
            weights=weights,
            mean_centering=self.mean_centering,
        )

        if self.score == "centrality":
            scores = centrality_scores(E, components, singular)
        elif self.score == "hybrid":
            scores = subspace_cosine_scores(q, E, components, singular, weighted=True)
            scores = 0.5 * l2_normalize(scores.reshape(1, -1)).ravel() + 0.5 * l2_normalize(
                cand_cos.reshape(1, -1)
            ).ravel()
        else:
            scores = subspace_cosine_scores(q, E, components, singular, weighted=True)

        order = np.argsort(-scores)
        return LASRResult(
            indices=cand_idx[order],
            scores=scores[order],
            energy_retained=energy,
            candidate_indices=cand_idx,
            cosine_candidate_scores=cand_cos,
        )

    def rank_all(self, query_vec: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """Return corpus-length ranks (1-based); unretrieved items get pool_k+1."""
        result = self.retrieve(query_vec, k=k)
        ranks = np.full(len(self.embeddings), fill_value=len(result.indices) + 1, dtype=np.int32)
        for r, idx in enumerate(result.indices, start=1):
            ranks[int(idx)] = r
        return ranks
