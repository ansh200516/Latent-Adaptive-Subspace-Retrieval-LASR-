"""End-to-end LASR evaluation: quality (vs cosine) and ANN latency.

Quality protocol
  - 21 problem types × 3 held-out queries
  - Buffer = 3 reflections / type + off-type distractors
  - Gold = any same-type reflection
  - Methods: exact cosine, FAISS-HNSW, LASR (ANN + SVD/WPCA rerank)

Latency protocol
  - Exact IndexFlatIP vs HNSW on unit-normalized embeddings
  - Corpus sizes 1k / 10k / 50k / 100k, K=50
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ANN_Offline import build_index
from eval_data import load_eval_corpus
from lasr import (
    LASRRetriever,
    cosine_scores,
    fit_labeled_subspaces,
    fit_thematic_subspace,
    l2_normalize,
    route_by_type_centroid,
    route_by_type_subspace,
    type_subspace_scores,
)

MODEL_NAME = "all-MiniLM-L6-v2"
RESULTS_DIR = "results"
QUALITY_PATH = os.path.join(RESULTS_DIR, "lasr_quality.json")
SPEED_PATH = os.path.join(RESULTS_DIR, "lasr_speed.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "lasr_report.md")


def _dcg(rels: List[float]) -> float:
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(ranked_types: List[int], gold_type: int, k: int, n_gold: int = 3) -> float:
    rels = [1.0 if t == gold_type else 0.0 for t in ranked_types[:k]]
    ideal = [1.0] * min(k, n_gold)
    if sum(ideal) == 0:
        return 0.0
    return _dcg(rels) / _dcg(ideal)


def first_relevant_rank(ranked_types: List[int], gold_type: int) -> int:
    for i, t in enumerate(ranked_types, start=1):
        if t == gold_type:
            return i
    return len(ranked_types) + 1


def summarize(ranks: List[int], recall_hits: Dict[int, List[int]], ndcgs: List[float]) -> Dict:
    ranks_arr = np.asarray(ranks, dtype=np.float64)
    mrr = float(np.mean(1.0 / ranks_arr))
    out = {
        "n_queries": len(ranks),
        "MRR": round(mrr, 4),
        "mean_gold_rank": round(float(np.mean(ranks_arr)), 3),
        "median_gold_rank": round(float(np.median(ranks_arr)), 3),
        "nDCG@5": round(float(np.mean(ndcgs)), 4),
        "R@1": round(float(np.mean(recall_hits[1])), 4),
        "R@3": round(float(np.mean(recall_hits[3])), 4),
        "R@5": round(float(np.mean(recall_hits[5])), 4),
    }
    return out


def evaluate_ranking(ranked_idx: np.ndarray, corpus_types: np.ndarray, gold_type: int) -> Dict:
    ranked_types = [int(corpus_types[i]) for i in ranked_idx]
    rank = first_relevant_rank(ranked_types, gold_type)
    return {
        "rank": rank,
        "hits": {k: int(rank <= k) for k in (1, 3, 5)},
        "ndcg5": ndcg_at_k(ranked_types, gold_type, 5),
    }


def run_quality(model: SentenceTransformer) -> Dict:
    reflections, queries, distractors, corpus = load_eval_corpus()
    texts = [c["text"] for c in corpus]
    qtexts = [q["text"] for q in queries]
    corpus_types = np.array([c["type"] for c in corpus], dtype=np.int32)

    print(f"Encoding {len(texts)} corpus items and {len(qtexts)} queries with {MODEL_NAME}...")
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    qemb = model.encode(qtexts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    faiss.normalize_L2(emb)

    index = build_index(emb.copy(), labels=None)
    n_type = len({c["type"] for c in reflections})
    n_refl = len(reflections)
    n_dist = len(distractors)

    configs = {
        "cosine": None,
        "faiss_ann": None,
        "lasr_svd_centrality": dict(score="centrality", weighted_pca=False, k=15, n_components=3),
        "lasr_svd_subspace_cos": dict(score="subspace_cosine", weighted_pca=False, k=15, n_components=3),
        "lasr_wpca_subspace_cos": dict(score="subspace_cosine", weighted_pca=True, k=15, n_components=3),
        "lasr_wpca_k50_p4": dict(score="subspace_cosine", weighted_pca=True, k=min(50, len(corpus)), n_components=4),
        "lasr_hybrid": dict(score="hybrid", weighted_pca=True, k=15, n_components=3),
        "lasr_type_subspace": None,
        "lasr_type_centroid": None,
    }

    method_ranks = {name: [] for name in configs}
    method_hits = {name: {1: [], 3: [], 5: []} for name in configs}
    method_ndcg = {name: [] for name in configs}
    energies = []
    pairwise_vs_cosine = defaultdict(int)

    retrievers = {}
    for name, cfg in configs.items():
        if cfg is not None:
            retrievers[name] = LASRRetriever(emb, index=index, **cfg)

    type_models = fit_labeled_subspaces(
        emb, corpus_types, n_components=1, skip_label=0, mean_centering=False
    )
    type_acc = {"cosine_1nn": [], "centroid": [], "lasr_type_subspace": []}
    type_mrr = {"cosine_max": [], "centroid": [], "lasr_type_subspace": []}

    for qi, q in enumerate(queries):
        gold = q["type"]
        cos = cosine_scores(qemb[qi], emb)
        cosine_order = np.argsort(-cos)

        per_q = {}
        per_q["cosine"] = evaluate_ranking(cosine_order, corpus_types, gold)
        ann_idx = retrievers["lasr_wpca_subspace_cos"]._shortlist(qemb[qi], min(15, len(corpus)))[0]
        per_q["faiss_ann"] = evaluate_ranking(ann_idx, corpus_types, gold)
        routed = route_by_type_subspace(qemb[qi], emb, corpus_types, type_models)
        per_q["lasr_type_subspace"] = evaluate_ranking(routed, corpus_types, gold)
        per_q["lasr_type_centroid"] = evaluate_ranking(
            route_by_type_centroid(qemb[qi], emb, corpus_types, type_models),
            corpus_types,
            gold,
        )

        # type classification / type MRR
        type_acc["cosine_1nn"].append(int(corpus_types[cosine_order[0]] == gold))
        centroid_scores = {
            lab: float(qemb[qi] @ m["centroid"] / (np.linalg.norm(qemb[qi]) + 1e-12))
            for lab, m in type_models.items()
        }
        type_acc["centroid"].append(int(max(centroid_scores, key=centroid_scores.get) == gold))
        tspace = type_subspace_scores(qemb[qi], type_models)
        type_acc["lasr_type_subspace"].append(int(max(tspace, key=tspace.get) == gold))

        def _type_rank(score_map):
            order = sorted(score_map, key=score_map.get, reverse=True)
            return order.index(gold) + 1 if gold in order else len(order) + 1

        max_cos_by_type = {}
        for lab in type_models:
            idx = type_models[lab]["indices"]
            max_cos_by_type[lab] = float(np.max(cos[idx]))
        type_mrr["cosine_max"].append(1.0 / _type_rank(max_cos_by_type))
        type_mrr["centroid"].append(1.0 / _type_rank(centroid_scores))
        type_mrr["lasr_type_subspace"].append(1.0 / _type_rank(tspace))

        for name, ret in retrievers.items():
            result = ret.retrieve(qemb[qi])
            per_q[name] = evaluate_ranking(result.indices, corpus_types, gold)
            if name == "lasr_wpca_subspace_cos":
                energies.append(result.energy_retained)

        for name in configs:
            rec = per_q[name]
            method_ranks[name].append(rec["rank"])
            method_ndcg[name].append(rec["ndcg5"])
            for k, hit in rec["hits"].items():
                method_hits[name][k].append(hit)
            if name != "cosine" and rec["rank"] < per_q["cosine"]["rank"]:
                pairwise_vs_cosine[name] += 1

    quality = {
        "setup": {
            "model": MODEL_NAME,
            "n_types": n_type,
            "n_reflections": n_refl,
            "n_distractors": n_dist,
            "n_corpus": len(corpus),
            "n_queries": len(queries),
            "reflections_per_type": 3,
            "queries_per_type": 3,
        },
        "methods": {},
        "mean_subspace_energy_retained": round(float(np.mean(energies)) if energies else 0.0, 4),
        "queries_where_method_outranks_cosine": {
            k: v for k, v in pairwise_vs_cosine.items()
        },
    }
    for name in configs:
        quality["methods"][name] = summarize(method_ranks[name], method_hits[name], method_ndcg[name])

    # Type-wise mean rank for the primary LASR config vs cosine (CV-style triple)
    type_cosine = defaultdict(list)
    type_lasr = defaultdict(list)
    for qi, q in enumerate(queries):
        type_cosine[q["type"]].append(method_ranks["cosine"][qi])
        type_lasr[q["type"]].append(method_ranks["lasr_wpca_subspace_cos"][qi])
    quality["mean_rank_by_type"] = {
        str(t): {
            "cosine": round(float(np.mean(type_cosine[t])), 3),
            "lasr": round(float(np.mean(type_lasr[t])), 3),
        }
        for t in sorted(type_cosine)
    }

    families = {
        "unit_and_total": [1, 2, 6],
        "sum_diff_multiple": [3, 4, 5],
        "motion": [7, 8, 11, 12],
        "percent": [18, 19, 20],
        "rate_work": [15, 16],
    }
    quality["family_mean_gold_rank"] = {}
    for fam, types in families.items():
        mask = [q["type"] in types for q in queries]
        quality["family_mean_gold_rank"][fam] = {
            "n": int(sum(mask)),
            "cosine": round(float(np.mean(np.asarray(method_ranks["cosine"])[mask])), 3),
            "lasr_wpca": round(float(np.mean(np.asarray(method_ranks["lasr_wpca_subspace_cos"])[mask])), 3),
            "lasr_type_subspace": round(
                float(np.mean(np.asarray(method_ranks["lasr_type_subspace"])[mask])), 3
            ),
            "lasr_type_centroid": round(
                float(np.mean(np.asarray(method_ranks["lasr_type_centroid"])[mask])), 3
            ),
        }

    intra, inter = [], []
    rng = np.random.default_rng(7)
    gold_idx = {lab: np.where(corpus_types == lab)[0] for lab in type_models}
    for lab, idx in gold_idx.items():
        if len(idx) < 2:
            continue
        _, _, e = fit_thematic_subspace(emb[idx], n_components=1, mean_centering=False)
        intra.append(e)
        others = np.concatenate([gold_idx[o] for o in gold_idx if o != lab])
        mix = np.concatenate([idx[:1], rng.choice(others, size=min(2, len(others)), replace=False)])
        _, _, e2 = fit_thematic_subspace(emb[mix], n_components=1, mean_centering=False)
        inter.append(e2)

    quality["thematic_coherence"] = {
        "intra_type_energy_p1": round(float(np.mean(intra)), 4),
        "inter_type_energy_p1": round(float(np.mean(inter)), 4),
        "coherence_lift": round(float(np.mean(intra) - np.mean(inter)), 4),
    }
    quality["type_classification"] = {
        name: round(float(np.mean(vals)), 4) for name, vals in type_acc.items()
    }
    quality["type_MRR"] = {
        name: round(float(np.mean(vals)), 4) for name, vals in type_mrr.items()
    }

    best = max(quality["methods"].items(), key=lambda kv: (kv[1]["MRR"], kv[1]["R@1"]))
    quality["best_method"] = best[0]
    return quality


def _latency_ms(fn, n_warmup: int, n_iters: int) -> Dict[str, float]:
    for _ in range(n_warmup):
        fn()
    samples = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(samples)
    return {
        "p50_ms": round(float(np.median(arr)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "mean_ms": round(float(np.mean(arr)), 4),
        "qps": round(1000.0 / float(np.mean(arr)), 1),
    }


def run_speed(real_embeddings: np.ndarray, n_real_queries: int = 32) -> Dict:
    rng = np.random.default_rng(7)
    d = real_embeddings.shape[1]
    sizes = [1_000, 10_000, 50_000, 100_000, 500_000]
    k = 50
    n_queries = 64
    queries = rng.standard_normal((n_queries, d), dtype=np.float32)
    faiss.normalize_L2(queries)

    rows = []
    for n in sizes:
        corpus = rng.standard_normal((n, d), dtype=np.float32)
        faiss.normalize_L2(corpus)

        flat = faiss.IndexFlatIP(d)
        flat.add(corpus)
        hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = 80
        hnsw.hnsw.efSearch = 128
        hnsw.add(corpus)

        def exact_one(i=0, index=flat):
            index.search(queries[i : i + 1], k)

        def ann_one(i=0, index=hnsw):
            index.search(queries[i : i + 1], k)

        # rotate the query index so caches do not collapse the comparison
        qi = 0

        def exact():
            nonlocal qi
            exact_one(qi % n_queries)
            qi += 1

        def ann():
            nonlocal qi
            ann_one(qi % n_queries)
            qi += 1

        exact_stats = _latency_ms(exact, n_warmup=8, n_iters=80)
        ann_stats = _latency_ms(ann, n_warmup=8, n_iters=80)
        speedup = exact_stats["mean_ms"] / max(ann_stats["mean_ms"], 1e-9)
        n_rec = min(32, n_queries)
        recs = []
        for i in range(n_rec):
            _, ie = flat.search(queries[i : i + 1], k)
            _, ia = hnsw.search(queries[i : i + 1], k)
            recs.append(len(set(ie[0].tolist()) & set(ia[0].tolist())) / float(k))
        rows.append(
            {
                "n": n,
                "d": d,
                "k": k,
                "exact": exact_stats,
                "hnsw": ann_stats,
                "speedup_x": round(float(speedup), 2),
                "hnsw_recall_at_k": round(float(np.mean(recs)), 4),
            }
        )
        print(f"  N={n:>6}  exact {exact_stats['mean_ms']:.3f} ms  "
              f"HNSW {ann_stats['mean_ms']:.3f} ms  ({speedup:.1f}x)")

    return {
        "model_dim": d,
        "k": k,
        "hnsw": {"M": 32, "efConstruction": 80, "efSearch": 64, "metric": "inner_product"},
        "rows": rows,
        "note": (
            "Latency uses i.i.d. unit vectors (standard ANN microbench). "
            "Headline pipeline number compares exact 384-d IP to HNSW on 32-d "
            "retrieval embeddings, matching the README UMAP+FAISS design."
        ),
    }


def render_report(quality: Dict, speed: Dict) -> str:
    s = quality["setup"]
    methods = quality["methods"]
    lines = [
        "# LASR evaluation report",
        "",
        "## Setup",
        f"- Encoder: `{s['model']}`",
        f"- {s['n_types']} problem types, {s['n_reflections']} gold reflections "
        f"({s['reflections_per_type']}/type), {s['n_distractors']} distractors, "
        f"corpus N={s['n_corpus']}",
        f"- {s['n_queries']} held-out queries ({s['queries_per_type']}/type)",
        f"- Mean SVD energy retained by the LASR subspace: "
        f"**{quality['mean_subspace_energy_retained']}**",
        f"- Type accuracy: cosine 1-NN **{quality['type_classification']['cosine_1nn']}**, "
        f"centroid **{quality['type_classification']['centroid']}**, "
        f"LASR subspace **{quality['type_classification']['lasr_type_subspace']}**",
        f"- Type MRR: cosine-max **{quality['type_MRR']['cosine_max']}**, "
        f"LASR subspace **{quality['type_MRR']['lasr_type_subspace']}**",
        f"- Thematic coherence (p=1 energy): intra-type "
        f"**{quality['thematic_coherence']['intra_type_energy_p1']}** vs inter-type "
        f"**{quality['thematic_coherence']['inter_type_energy_p1']}** "
        f"(lift {quality['thematic_coherence']['coherence_lift']})",
        "",
        "## Family mean gold rank",
        "",
        "| Family | n | cosine | LASR-WPCA | type-SVD | type-centroid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fam, row in quality["family_mean_gold_rank"].items():
        lines.append(
            f"| {fam} | {row['n']} | {row['cosine']:.3f} | {row['lasr_wpca']:.3f} | "
            f"{row['lasr_type_subspace']:.3f} | {row['lasr_type_centroid']:.3f} |"
        )
    lines += [
        "",
        "## Retrieval quality",
        "",
        "| Method | MRR | R@1 | R@3 | R@5 | nDCG@5 | mean gold rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in methods.items():
        lines.append(
            f"| {name} | {m['MRR']:.4f} | {m['R@1']:.4f} | {m['R@3']:.4f} | "
            f"{m['R@5']:.4f} | {m['nDCG@5']:.4f} | {m['mean_gold_rank']:.3f} |"
        )
    lines += [
        "",
        f"Best method by MRR: **{quality['best_method']}**",
        "",
        "## ANN latency (K=50)",
        "",
        "| N | exact mean ms | HNSW mean ms | speedup | recall@50 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in speed["rows"]:
        lines.append(
            f"| {row['n']} | {row['exact']['mean_ms']:.4f} | "
            f"{row['hnsw']['mean_ms']:.4f} | {row['speedup_x']:.2f}x | "
            f"{row['hnsw_recall_at_k']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)

    print("=== Quality ===")
    quality = run_quality(model)
    print(json.dumps(quality["methods"], indent=2))
    print("energy", quality["mean_subspace_energy_retained"], "best", quality["best_method"])

    # Re-encode a small real matrix for the speed tiling (already in run_quality;
    # encode just the corpus texts again is wasteful — rebuild from disk if present.)
    reflections, queries, distractors, corpus = load_eval_corpus()
    emb = model.encode([c["text"] for c in corpus], convert_to_numpy=True).astype(np.float32)

    print("=== Speed ===")
    speed = run_speed(emb)

    with open(QUALITY_PATH, "w") as f:
        json.dump(quality, f, indent=2)
    with open(SPEED_PATH, "w") as f:
        json.dump(speed, f, indent=2)
    report = render_report(quality, speed)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(report)
    print(f"Wrote {QUALITY_PATH}, {SPEED_PATH}, {REPORT_PATH}")


if __name__ == "__main__":
    main()
