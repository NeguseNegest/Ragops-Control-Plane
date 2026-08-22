# Dense vs BM25 vs RRF Hybrid vs Cross-Encoder Reranker

## Executive summary

Hybrid + cross-encoder has the highest MRR@5. The cross-encoder improves the controlled RRF-25 ablation. On 50 paired verified questions, hybrid plus reranking reaches 64.7% MRR@5, changing +7.4 pp versus BM25, +6.5 pp versus the Day 25 RRF baseline, and +6.2 pp versus its own pre-rerank candidate order.

All headline rankings are truncated to the common final depth of 5. This avoids comparing a five-result reranker against MRR computed over ten baseline results.

## Four-way benchmark table

| Metric | Dense | BM25 | RRF hybrid | Hybrid + reranker | Reranker − BM25 | Reranker − hybrid |
|---|---:|---:|---:|---:|---:|---:|
| MRR@5 | 40.6% | 57.4% | 58.2% | 64.7% | +7.4 pp | +6.5 pp |
| Hit rate@1 | 34.0% | 40.0% | 48.0% | 54.0% | +14.0 pp | +6.0 pp |
| Hit rate@3 | 46.0% | 74.0% | 66.0% | 72.0% | -2.0 pp | +6.0 pp |
| Hit rate@5 | 54.0% | 82.0% | 76.0% | 82.0% | +0.0 pp | +6.0 pp |
| Recall@1 | 32.7% | 40.0% | 46.7% | 54.0% | +14.0 pp | +7.3 pp |
| Recall@3 | 42.7% | 70.7% | 62.7% | 70.0% | -0.7 pp | +7.3 pp |
| Recall@5 | 50.7% | 77.7% | 72.7% | 81.0% | +3.3 pp | +8.3 pp |
| nDCG@1 | 34.0% | 40.0% | 48.0% | 54.0% | +14.0 pp | +6.0 pp |
| nDCG@3 | 38.8% | 58.8% | 56.8% | 64.2% | +5.4 pp | +7.4 pp |
| nDCG@5 | 42.2% | 61.6% | 60.8% | 68.9% | +7.3 pp | +8.1 pp |

Recall and hit rate are identical because each current question has one labeled relevant chunk.

## Controlled reranker ablation

The official Day 25 RRF run used dense/BM25 top 20 and returned 10. The controlled ablation below instead compares the first five positions of the exact RRF-25 candidate ranking used by the reranker with its final top five.

| Metric | RRF-25 before reranking | Hybrid + reranker | Delta |
|---|---:|---:|---:|
| MRR@5 | 58.5% | 64.7% | +6.2 pp |
| Hit rate@1 | 48.0% | 54.0% | +6.0 pp |
| Hit rate@3 | 66.0% | 72.0% | +6.0 pp |
| Hit rate@5 | 76.0% | 82.0% | +6.0 pp |
| nDCG@1 | 48.0% | 54.0% | +6.0 pp |
| nDCG@3 | 57.0% | 64.2% | +7.2 pp |
| nDCG@5 | 61.1% | 68.9% | +7.9 pp |

At the question level, reranking wins 14, loses 8, and ties 28 against its own candidate order. It recovers 5 top-5 misses and loses 2 prior top-5 hits.

## Paired rank outcomes

| Comparison | Reranker wins | Other wins | Ties | Recovers miss | Loses hit |
|---|---:|---:|---:|---:|---:|
| Reranker vs Dense | 24 | 5 | 21 | 16 | 2 |
| Reranker vs BM25 | 15 | 11 | 24 | 2 | 2 |
| Reranker vs RRF hybrid | 15 | 8 | 27 | 5 | 2 |
| Reranker vs RRF-25 before reranking | 14 | 8 | 28 | 5 | 2 |

## Wording cohorts

| Query type | Questions | BM25 MRR@5 | Hybrid MRR@5 | Pre-rerank MRR@5 | Reranked MRR@5 | Reranker − pre | Wins | Losses | Ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| behavioral/procedural | 13 | 50.3% | 55.8% | 56.7% | 59.0% | +2.3 pp | 3 | 3 | 7 |
| conceptual/descriptive | 22 | 66.1% | 65.7% | 65.7% | 74.8% | +9.1 pp | 6 | 3 | 13 |
| exact-reference | 15 | 50.8% | 49.3% | 49.7% | 55.0% | +5.3 pp | 5 | 2 | 8 |

## Reranking gains

| Question | Type | Candidate-pool rank | Pre-rerank top 5 | Reranked |
|---|---|---:|---:|---:|
| What warning is given about handling a plain password in this section? | conceptual/descriptive | 7 | miss | 1 |
| What does an MLmodel file declare about each model to enable serving it as a Python function? | conceptual/descriptive | 5 | 5 | 1 |
| When should I use FastAPI BackgroundTasks, and when should I consider a tool such as Celery instead? | behavioral/procedural | 5 | 5 | 1 |
| Which distance measures are given as examples for calculating similarity between embeddings? | conceptual/descriptive | 4 | 4 | 1 |
| What is the main difference between the legacy permission model and the role-based permission model in MLflow? | behavioral/procedural | 3 | 3 | 1 |
| What file format does MLflow use for saving models from a variety of tools? | exact-reference | 2 | 2 | 1 |
| Which parameter in a Qdrant query allows retrieving a vector using a point ID located in a different collection? | exact-reference | 2 | 2 | 1 |
| How does MLflow determine the input and output schema when logging a PythonModel that includes type hints? | behavioral/procedural | 2 | 2 | 1 |
| What dimensionality and distance-metric constraints apply to vectors in one Qdrant collection? | conceptual/descriptive | 2 | 2 | 1 |
| How can extra cookies be forbidden in a FastAPI cookie parameter model? | exact-reference | 5 | 5 | 2 |
| Where must pydantic model type hints be defined to use pydantic objects as inputs for an MLflow PythonModel? | conceptual/descriptive | 11 | miss | 4 |
| Which nested field key is used in the HTTP, Python, TypeScript, Rust, and Java examples to specify the sightseeing condition? | exact-reference | 9 | miss | 4 |
| What is the exact MLflow command to serve a model located at runs:/<RUN_ID>/model? | exact-reference | 8 | miss | 4 |
| What is an MLflow model alias, and how can a champion alias change which model version production loads? | conceptual/descriptive | 10 | miss | 5 |

## Failure cases where reranking hurts

| Question | Type | Candidate-pool rank | Pre-rerank top 5 | Reranked |
|---|---|---:|---:|---:|
| How do I create and log a pandas training dataset with MLflow while retaining its source and training context? | behavioral/procedural | 1 | 1 | miss |
| What two primary properties are required inside the lookup_from configuration when referencing a point in another collection? | exact-reference | 1 | 1 | 4 |
| What mechanism is used to implement a fine-grained permission system in FastAPI? | conceptual/descriptive | 1 | 1 | 2 |
| What primary trade-off do rerankers make when evaluating document relevance? | behavioral/procedural | 1 | 1 | 2 |
| In which section of the MLflow Run details page can a user find and select the model folder to register? | conceptual/descriptive | 1 | 1 | 2 |
| For Qdrant multitenancy, when is one payload-partitioned collection preferred over multiple collections? | conceptual/descriptive | 1 | 1 | 2 |
| What happens to the request_preview size and where is the full request stored? | exact-reference | 4 | 4 | miss |
| How can I enable MLflow autologging only for PyTorch, or disable it for scikit-learn while keeping generic autologging enabled? | behavioral/procedural | 2 | 2 | 3 |

## Reranked top-5 failures

| Question | Source | BM25 | Hybrid | Candidate pool | Reranked |
|---|---|---:|---:|---:|---:|
| What does FastAPI read from the request body when you declare a Python type? | fastapi/docs/tutorial/body.md | miss | miss | miss | miss |
| What HTTP endpoint is shown for scrolling points in a collection? | qdrant/qdrant_llms_full.txt | miss | miss | miss | miss |
| Which library should be installed to use EmailStr according to the document? | fastapi/docs/tutorial/response-model.md | miss | miss | miss | miss |
| What happens to the request_preview size and where is the full request stored? | mlflow/docs/api_reference/source/rest-api.rst | 1 | 5 | 4 | miss |
| After saving a minimal FastAPI application as main.py, which command starts the development server, and where can I open the interactive API documentation? | fastapi/docs/tutorial/first-steps.md | miss | miss | miss | miss |
| How do I declare a JSON request body in FastAPI, and what does FastAPI do with it? | fastapi/docs/tutorial/body.md | miss | miss | miss | miss |
| Which MLflow calls does the tracking quickstart use to manually record hyperparameters, a scikit-learn model, an accuracy metric, and a descriptive tag? | mlflow/docs/docs/classic-ml/tracking/quickstart/index.mdx | miss | miss | miss | miss |
| What is the difference between MLflow's backend store and artifact store? | mlflow/docs/docs/classic-ml/tracking/index.mdx | miss | miss | 9 | miss |
| How do I create and log a pandas training dataset with MLflow while retaining its source and training context? | mlflow/docs/docs/classic-ml/dataset/index.mdx | 2 | 1 | 1 | miss |

## Quality and latency tradeoff

| Recorded run | Average | After first query | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Dense baseline | 965.5 ms | not recomputed | 40.7 ms | 42239.7 ms |
| BM25 baseline | 69.2 ms | not recomputed | 41.0 ms | 122.2 ms |
| Day 25 RRF hybrid | 700.8 ms | not recomputed | 91.3 ms | 26011.5 ms |
| Hybrid + reranker | 4545.3 ms | 4365.4 ms | 2599.0 ms | 13360.6 ms |
| In-run retrieval + fusion estimate | 86.3 ms | not separately reported | 45.4 ms | 273.1 ms |

The cross-encoder model loaded once in 19092.3 ms before the question loop. That cost is excluded from per-query totals.

Reranked-run component timings:

| Stage | Average | After first query | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Dense | 9.3 ms | 9.1 ms | 6.8 ms | 19.2 ms |
| BM25 | 76.7 ms | 76.9 ms | 37.4 ms | 256.1 ms |
| Fusion | 0.3 ms | 0.3 ms | 0.2 ms | 0.3 ms |
| Reranker | 4225.6 ms | 4162.0 ms | 2438.1 ms | 8020.2 ms |

Historical baseline timings come from separate processes and are contextual, not controlled head-to-head latency measurements. The in-run estimate sums the dense, BM25, and fusion stages measured immediately before the reranker and is the cleanest latency ablation, though it excludes small orchestration overhead.

## Validity and decision

- The live reranker run used 13481 dense points and 13481 BM25 source records (13476 searchable, 5 tokenless skips).
- BM25 source SHA256: `61d97c037bfedda6d4c6ce66127392bc490435d2f475945c845fd28dc450fe50`.
- Candidate: dense top 25 + BM25 top 25 -> RRF top 25 at k=60 -> cross-encoder/ms-marco-MiniLM-L-6-v2 top 5.
- Every headline ranking is truncated to 5; MRR is reported as MRR@5. The RRF-25 ablation uses the exact candidate order from the reranked run.
- All reports share question IDs, wording, expected sources, relevant chunk IDs, component model/tokenizer settings, and the persisted BM25 source hash.
- The historical dense report has no corpus hash. Live point/source count parity and agreement with the Day 25 live run reduce but do not eliminate snapshot uncertainty.
- The configured Hugging Face model name is recorded, but a model repository revision is not pinned; a changed upstream snapshot could affect exact reproduction.
- The 45 source-derived questions map to one labeled chunk each and only 20 unique relevance groups. Unjudged chunks may also be useful, while retained lexical overlap can favor BM25.
- Equal weighting across 33 relevance groups changes reranked MRR@5 by +5.3 pp versus BM25 and +5.9 pp versus pre-rerank RRF-25.
- Wording cohorts are deterministic heuristics, not independent human query-type labels.

Decision on the Day 27 primary metric: **Hybrid + cross-encoder has the highest MRR@5**. Hybrid plus reranking does improve on BM25, does improve on the Day 25 RRF baseline, and does improve on its controlled pre-rerank ordering.
