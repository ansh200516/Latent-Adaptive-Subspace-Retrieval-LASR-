# LASR evaluation report

Measured with `all-MiniLM-L6-v2` on a labeled thought-template buffer
(21 MetaBuffer-Math types × 3 reflections + 50 distractors = 113 items)
and 63 held-out queries (3 / type). Gold = any same-type reflection.

## Headline

| Metric | Cosine 1-NN | LASR type-SVD (p=1) | LASR type-centroid |
|---|---:|---:|---:|
| R@1 | 0.8254 | 0.8730 | **0.8889** |
| MRR | 0.8884 | 0.8956 | **0.9069** |
| nDCG@5 | 0.7866 | 0.8974 | **0.9011** |
| Type accuracy | 0.8254 | 0.8730 | **0.8889** |
| Type MRR | 0.9137 | 0.9212 | **0.9286** |

Type-centroid routing is rank-1 uncentered SVD of each type's buffer
(the leading singular vector ≈ the type mean). It lifts R@1 by **+6.4 pp**
and nDCG@5 by **+14.5% relative** vs ambient cosine.

Same-type embeddings concentrate **0.734** of energy in PC1 vs **0.552**
for mixed-type triples (coherence lift **+0.182**). Dynamic rank-3
candidate-pool SVD retains **0.450** of shortlist energy.

Candidate-pool rerank (centrality / subspace-cosine / hybrid) did **not**
beat cosine on this buffer — those rows are in `lasr_quality.json`.

## ANN latency (K=50)

| Setting | N | exact ms | HNSW ms | speedup | recall@50 |
|---|---:|---:|---:|---:|---:|
| 32-d vs 32-d | 100k | 0.185 | 0.089 | 2.1× | 0.889 |
| 32-d vs 32-d | 500k | 1.144 | 0.264 | 4.3× | 0.786 |
| **384-d exact vs 32-d HNSW** | **500k** | **15.92** | **0.192** | **82.8×** | (reduced-d index) |

The 82.8× figure is the README pipeline: 32-d ANN candidate generation
vs a full 384-d exact inner-product scan.

## Family mean gold rank

| Family | n | cosine | type-SVD | type-centroid |
|---|---:|---:|---:|---:|
| unit_and_total | 9 | 1.778 | 3.000 | 2.333 |
| sum_diff_multiple | 9 | 3.444 | 4.000 | 4.000 |
| motion | 12 | 1.083 | 1.000 | 1.000 |
| percent | 9 | 1.000 | 1.000 | 1.000 |
| rate_work | 6 | 1.333 | 1.000 | 1.000 |
