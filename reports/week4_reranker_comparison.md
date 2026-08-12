# Dense vs BM25 vs RRF Hybrid vs Cross-Encoder Reranker

## Executive summary

Hybrid + cross-encoder has the highest MRR@5. The cross-encoder improves the controlled RRF-25 ablation. On 45 paired verified questions, hybrid plus reranking reaches 68.9% MRR@5, changing +7.4 pp versus BM25, +12.5 pp versus the Day 25 RRF baseline, and +12.4 pp versus its own pre-rerank candidate order.

All headline rankings are truncated to the common final depth of 5. This avoids comparing a five-result reranker against MRR computed over ten baseline results.

## Four-way benchmark table

| Metric | Dense | BM25 | RRF hybrid | Hybrid + reranker | Reranker − BM25 | Reranker − hybrid |
|---|---:|---:|---:|---:|---:|---:|
| MRR@5 | 31.6% | 61.5% | 56.4% | 68.9% | +7.4 pp | +12.5 pp |
| Hit rate@1 | 26.7% | 46.7% | 46.7% | 57.8% | +11.1 pp | +11.1 pp |
| Hit rate@3 | 31.1% | 75.6% | 64.4% | 75.6% | +0.0 pp | +11.1 pp |
| Hit rate@5 | 44.4% | 84.4% | 75.6% | 84.4% | +0.0 pp | +8.9 pp |
| Recall@1 | 26.7% | 46.7% | 46.7% | 57.8% | +11.1 pp | +11.1 pp |
| Recall@3 | 31.1% | 75.6% | 64.4% | 75.6% | +0.0 pp | +11.1 pp |
| Recall@5 | 44.4% | 84.4% | 75.6% | 84.4% | +0.0 pp | +8.9 pp |
| nDCG@1 | 26.7% | 46.7% | 46.7% | 57.8% | +11.1 pp | +11.1 pp |
| nDCG@3 | 29.2% | 63.7% | 56.7% | 69.0% | +5.3 pp | +12.3 pp |
| nDCG@5 | 34.7% | 67.3% | 61.1% | 72.8% | +5.6 pp | +11.7 pp |

Recall and hit rate are identical because each current question has one labeled relevant chunk.

## Controlled reranker ablation

The official Day 25 RRF run used dense/BM25 top 20 and returned 10. The controlled ablation below instead compares the first five positions of the exact RRF-25 candidate ranking used by the reranker with its final top five.

| Metric | RRF-25 before reranking | Hybrid + reranker | Delta |
|---|---:|---:|---:|
| MRR@5 | 56.4% | 68.9% | +12.4 pp |
| Hit rate@1 | 46.7% | 57.8% | +11.1 pp |
| Hit rate@3 | 64.4% | 75.6% | +11.1 pp |
| Hit rate@5 | 73.3% | 84.4% | +11.1 pp |
| nDCG@1 | 46.7% | 57.8% | +11.1 pp |
| nDCG@3 | 57.0% | 69.0% | +12.0 pp |
| nDCG@5 | 60.6% | 72.8% | +12.2 pp |

At the question level, reranking wins 16, loses 5, and ties 24 against its own candidate order. It recovers 6 top-5 misses and loses 1 prior top-5 hits.

## Paired rank outcomes

| Comparison | Reranker wins | Other wins | Ties | Recovers miss | Loses hit |
|---|---:|---:|---:|---:|---:|
| Reranker vs Dense | 26 | 1 | 18 | 18 | 0 |
| Reranker vs BM25 | 14 | 10 | 21 | 1 | 1 |
| Reranker vs RRF hybrid | 17 | 5 | 23 | 5 | 1 |
| Reranker vs RRF-25 before reranking | 16 | 5 | 24 | 6 | 1 |

## Wording cohorts

| Query type | Questions | BM25 MRR@5 | Hybrid MRR@5 | Pre-rerank MRR@5 | Reranked MRR@5 | Reranker − pre | Wins | Losses | Ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| behavioral/procedural | 8 | 62.9% | 67.1% | 66.7% | 87.5% | +20.8 pp | 3 | 1 | 4 |
| conceptual/descriptive | 23 | 74.1% | 63.6% | 63.6% | 77.2% | +13.6 pp | 8 | 2 | 13 |
| exact-reference | 14 | 40.1% | 38.6% | 38.9% | 44.6% | +5.7 pp | 5 | 2 | 7 |

## Reranking gains

| Question | Type | Candidate-pool rank | Pre-rerank top 5 | Reranked |
|---|---|---:|---:|---:|
| What happens if the submitted data is invalid? | behavioral/procedural | 6 | miss | 1 |
| What warning is given about handling a plain password in this section? | conceptual/descriptive | 7 | miss | 1 |
| What does an MLmodel file declare about each model to enable serving it as a Python function? | conceptual/descriptive | 5 | 5 | 1 |
| Which distance measures are given as examples for calculating similarity between embeddings? | conceptual/descriptive | 4 | 4 | 1 |
| What is the main difference between the legacy permission model and the role-based permission model in MLflow? | behavioral/procedural | 3 | 3 | 1 |
| What file format does MLflow use for saving models from a variety of tools? | exact-reference | 2 | 2 | 1 |
| What intermediate representation is needed to implement similarity learning quickly? | conceptual/descriptive | 2 | 2 | 1 |
| Which parameter in a Qdrant query allows retrieving a vector using a point ID located in a different collection? | exact-reference | 2 | 2 | 1 |
| What operation is used to quantify the similarity between the query and document vectors? | conceptual/descriptive | 9 | miss | 2 |
| How does MLflow determine the input and output schema when logging a PythonModel that includes type hints? | behavioral/procedural | 2 | 2 | 1 |
| How can extra cookies be forbidden in a FastAPI cookie parameter model? | exact-reference | 5 | 5 | 2 |
| Where must pydantic model type hints be defined to use pydantic objects as inputs for an MLflow PythonModel? | conceptual/descriptive | 11 | miss | 4 |
| Which nested field key is used in the HTTP, Python, TypeScript, Rust, and Java examples to specify the sightseeing condition? | exact-reference | 9 | miss | 4 |
| What is the exact MLflow command to serve a model located at runs:/<RUN_ID>/model? | exact-reference | 8 | miss | 4 |
| What schemas are generated for your model and where can you use them? | conceptual/descriptive | 3 | 3 | 2 |
| What kind of interface does UploadFile expose? | conceptual/descriptive | 3 | 3 | 2 |

## Failure cases where reranking hurts

| Question | Type | Candidate-pool rank | Pre-rerank top 5 | Reranked |
|---|---|---:|---:|---:|
| What two primary properties are required inside the lookup_from configuration when referencing a point in another collection? | exact-reference | 1 | 1 | 4 |
| What mechanism is used to implement a fine-grained permission system in FastAPI? | conceptual/descriptive | 1 | 1 | 2 |
| What primary trade-off do rerankers make when evaluating document relevance? | behavioral/procedural | 1 | 1 | 2 |
| In which section of the MLflow Run details page can a user find and select the model folder to register? | conceptual/descriptive | 1 | 1 | 2 |
| What happens to the request_preview size and where is the full request stored? | exact-reference | 4 | 4 | miss |

## Reranked top-5 failures

| Question | Source | BM25 | Hybrid | Candidate pool | Reranked |
|---|---|---:|---:|---:|---:|
| What does FastAPI read from the request body when you declare a Python type? | fastapi/docs/tutorial/body.md | miss | miss | miss | miss |
| In the Go client initialization, what host and port are configured? | qdrant/qdrant_llms_full.txt | miss | miss | miss | miss |
| What HTTP endpoint is shown for scrolling points in a collection? | qdrant/qdrant_llms_full.txt | miss | miss | miss | miss |
| Which library should be installed to use EmailStr according to the document? | fastapi/docs/tutorial/response-model.md | miss | miss | miss | miss |
| What is the data type of request_time? | mlflow/docs/api_reference/source/rest-api.rst | miss | miss | 22 | miss |
| What happens to the request_preview size and where is the full request stored? | mlflow/docs/api_reference/source/rest-api.rst | 1 | 5 | 4 | miss |
| What is one alternative to class-based decision making? | qdrant/qdrant_llms_full.txt | miss | miss | miss | miss |

## Quality and latency tradeoff

| Recorded run | Average | After first query | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Dense baseline | 679.9 ms | not recomputed | 36.7 ms | 24009.5 ms |
| BM25 baseline | 87.9 ms | not recomputed | 42.1 ms | 294.4 ms |
| Day 25 RRF hybrid | 837.4 ms | not recomputed | 80.9 ms | 29892.1 ms |
| Hybrid + reranker | 4681.6 ms | 4476.4 ms | 2744.0 ms | 13710.6 ms |
| In-run retrieval + fusion estimate | 333.7 ms | not separately reported | 88.7 ms | 6163.2 ms |

The cross-encoder model loaded once in 28693.4 ms before the question loop. That cost is excluded from per-query totals.

Reranked-run component timings:

| Stage | Average | After first query | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Dense | 261.2 ms | 128.6 ms | 38.0 ms | 6095.2 ms |
| BM25 | 72.2 ms | 72.4 ms | 39.0 ms | 137.3 ms |
| Fusion | 0.3 ms | 0.3 ms | 0.2 ms | 0.6 ms |
| Reranker | 4347.6 ms | 4274.9 ms | 2620.0 ms | 7609.7 ms |

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
- Equal weighting across 20 relevance groups changes reranked MRR@5 by +8.8 pp versus BM25 and +13.1 pp versus pre-rerank RRF-25.
- Wording cohorts are deterministic heuristics, not independent human query-type labels.

Decision on the Day 27 primary metric: **Hybrid + cross-encoder has the highest MRR@5**. Hybrid plus reranking does improve on BM25, does improve on the Day 25 RRF baseline, and does improve on its controlled pre-rerank ordering.
