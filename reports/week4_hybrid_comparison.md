# Dense vs BM25 vs RRF Hybrid Benchmark

## Executive summary

Hybrid improves on dense retrieval but does not improve on the stronger BM25 baseline by MRR. On 45 paired verified questions, MRR is 33.6% dense, 61.9% BM25, and 57.7% hybrid. Hybrid changes MRR by +24.1 pp versus dense and -4.2 pp versus BM25.

## Benchmark table

| Metric | Dense | BM25 | RRF hybrid | Hybrid − dense | Hybrid − BM25 |
|---|---:|---:|---:|---:|---:|
| MRR | 33.6% | 61.9% | 57.7% | +24.1 pp | -4.2 pp |
| Hit rate@1 | 26.7% | 46.7% | 46.7% | +20.0 pp | +0.0 pp |
| Hit rate@3 | 31.1% | 75.6% | 64.4% | +33.3 pp | -11.1 pp |
| Hit rate@5 | 44.4% | 84.4% | 75.6% | +31.1 pp | -8.9 pp |
| Hit rate@10 | 60.0% | 86.7% | 84.4% | +24.4 pp | -2.2 pp |
| Recall@1 | 26.7% | 46.7% | 46.7% | +20.0 pp | +0.0 pp |
| Recall@3 | 31.1% | 75.6% | 64.4% | +33.3 pp | -11.1 pp |
| Recall@5 | 44.4% | 84.4% | 75.6% | +31.1 pp | -8.9 pp |
| Recall@10 | 60.0% | 86.7% | 84.4% | +24.4 pp | -2.2 pp |
| nDCG@1 | 26.7% | 46.7% | 46.7% | +20.0 pp | +0.0 pp |
| nDCG@3 | 29.2% | 63.7% | 56.7% | +27.5 pp | -7.0 pp |
| nDCG@5 | 34.7% | 67.3% | 61.1% | +26.4 pp | -6.2 pp |
| nDCG@10 | 39.6% | 68.1% | 64.0% | +24.4 pp | -4.0 pp |

Recall and hit rate are identical because each current question has one labeled relevant chunk.

## Paired rank outcomes

| Comparison | Hybrid wins | Other wins | Ties | Hybrid recovers miss | Hybrid loses hit |
|---|---:|---:|---:|---:|---:|
| Hybrid vs dense | 26 | 1 | 18 | 11 | 0 |
| Hybrid vs BM25 | 10 | 14 | 21 | 0 | 1 |

Against the better component rank on each individual question, hybrid wins 4, loses 15, and ties 26.

## Wording cohorts

| Query type | Questions | Dense MRR | BM25 MRR | Hybrid MRR | Hybrid − BM25 | Hybrid wins | BM25 wins | Ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| behavioral/procedural | 8 | 53.6% | 62.9% | 67.1% | +4.2 pp | 3 | 3 | 2 |
| conceptual/descriptive | 23 | 34.4% | 74.8% | 64.9% | -9.9 pp | 4 | 8 | 11 |
| exact-reference | 14 | 20.9% | 40.1% | 40.4% | +0.2 pp | 3 | 3 | 8 |

## Hybrid gains over BM25

| Question | Type | Dense | BM25 | Hybrid |
|---|---|---:|---:|---:|
| Which specific third-party packages are mentioned as easily usable with FastAPI for security handling? | conceptual/descriptive | 1 | 6 | 1 |
| What trade-off occurs when opting for a lower number of vector chunks? | behavioral/procedural | 1 | 5 | 1 |
| Which MLflow flavor can log scikit-learn models as artifacts for serving? | conceptual/descriptive | 1 | 5 | 1 |
| What two primary properties are required inside the lookup_from configuration when referencing a point in another collection? | exact-reference | 8 | 2 | 1 |
| Which methods can be used to serve static frontend applications in FastAPI? | conceptual/descriptive | 1 | 2 | 1 |
| What initial methods can be used to retrieve a subset of documents before applying a reranker? | conceptual/descriptive | 4 | 2 | 1 |
| What occurs if an explicit signature parameter is provided when logging a PythonModel with type hints? | exact-reference | 1 | 2 | 1 |
| In what order does FastAPI evaluate incoming requests when serving static frontend files alongside API routes? | behavioral/procedural | 6 | 2 | 1 |
| What file format does MLflow use for saving models from a variety of tools? | exact-reference | 9 | 3 | 2 |
| How does MLflow determine the input and output schema when logging a PythonModel that includes type hints? | behavioral/procedural | 1 | 3 | 2 |

## Hybrid regressions versus BM25

| Question | Type | Dense | BM25 | Hybrid |
|---|---|---:|---:|---:|
| Where must pydantic model type hints be defined to use pydantic objects as inputs for an MLflow PythonModel? | conceptual/descriptive | miss | 1 | miss |
| What operation is used to quantify the similarity between the query and document vectors? | conceptual/descriptive | miss | 1 | 6 |
| What does an MLmodel file declare about each model to enable serving it as a Python function? | conceptual/descriptive | miss | 1 | 5 |
| What happens to the request_preview size and where is the full request stored? | exact-reference | miss | 1 | 5 |
| Which distance measures are given as examples for calculating similarity between embeddings? | conceptual/descriptive | miss | 1 | 4 |
| What schemas are generated for your model and where can you use them? | conceptual/descriptive | miss | 1 | 3 |
| What occurs when a user chooses the "Create New Model" option in the Model dropdown menu? | behavioral/procedural | miss | 1 | 3 |
| What intermediate representation is needed to implement similarity learning quickly? | conceptual/descriptive | 7 | 1 | 2 |
| What happens if the submitted data is invalid? | behavioral/procedural | miss | 2 | 5 |
| What is the exact MLflow command to serve a model located at runs:/<RUN_ID>/model? | exact-reference | miss | 3 | 8 |
| What warning is given about handling a plain password in this section? | conceptual/descriptive | miss | 3 | 7 |
| What is the main difference between the legacy permission model and the role-based permission model in MLflow? | behavioral/procedural | 8 | 2 | 3 |
| What kind of interface does UploadFile expose? | conceptual/descriptive | 5 | 2 | 3 |
| Which nested field key is used in the HTTP, Python, TypeScript, Rust, and Java examples to specify the sightseeing condition? | exact-reference | miss | 4 | 8 |

## Hybrid top-10 failures

| Question | Source | Dense | BM25 | Hybrid |
|---|---|---:|---:|---:|
| What does FastAPI read from the request body when you declare a Python type? | fastapi/docs/tutorial/body.md | miss | miss | miss |
| In the Go client initialization, what host and port are configured? | qdrant/qdrant_llms_full.txt | miss | miss | miss |
| What HTTP endpoint is shown for scrolling points in a collection? | qdrant/qdrant_llms_full.txt | miss | miss | miss |
| Which library should be installed to use EmailStr according to the document? | fastapi/docs/tutorial/response-model.md | miss | miss | miss |
| What is the data type of request_time? | mlflow/docs/api_reference/source/rest-api.rst | miss | miss | miss |
| What is one alternative to class-based decision making? | qdrant/qdrant_llms_full.txt | miss | miss | miss |
| Where must pydantic model type hints be defined to use pydantic objects as inputs for an MLflow PythonModel? | mlflow/docs/docs/classic-ml/model/python_model.mdx | miss | 1 | miss |

## Latency

| Recorded run | Average | After first query | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Dense baseline | 679.9 ms | 149.6 ms | 36.7 ms | 24009.5 ms |
| BM25 baseline | 87.9 ms | 88.1 ms | 42.1 ms | 294.4 ms |
| RRF hybrid | 837.4 ms | 177.1 ms | 80.9 ms | 29892.1 ms |

Hybrid component timings from the same run:

| Stage | Average | Minimum | Maximum |
|---|---:|---:|---:|
| Dense | 772.5 ms | 36.3 ms | 29810.8 ms |
| BM25 | 64.7 ms | 35.0 ms | 109.2 ms |
| Fusion | 0.2 ms | 0.1 ms | 0.2 ms |

The historical dense and BM25 latency rows come from separate runs and are not controlled head-to-head latency measurements. Hybrid total latency is end-to-end sequential dense retrieval, BM25 scoring, and fusion; its component table is internally comparable. First-query model warm-up can dominate averages.

## Validity and decision

- The live hybrid run used 13481 dense points and the BM25 source artifact contains 13481 records (13476 searchable, 5 tokenless skips).
- BM25 source SHA256: `61d97c037bfedda6d4c6ce66127392bc490435d2f475945c845fd28dc450fe50`.
- All three reports use identical question IDs, question text, expected sources, relevant chunk IDs, and metric cutoffs.
- The historical dense report does not persist a corpus hash; live dense/BM25 record-count parity reduces but does not eliminate snapshot uncertainty.
- The 45 source-derived questions map to only 20 labeled chunks, have one relevance judgment each, and retain lexical overlap that can favor BM25.
- Giving each of the 20 unique relevance groups equal weight still leaves hybrid MRR -3.2 pp versus BM25; hybrid wins 5 groups, BM25 wins 11, and 4 tie.
- Wording cohorts are deterministic heuristics, not independent human query-type labels.
- Unweighted RRF rewards cross-retriever consensus. On this lexically aligned set, that can demote strong BM25-only evidence below weaker chunks appearing in both lists.

Decision on the Day 25 primary metric: **BM25 has the highest MRR**. Hybrid does not improve on BM25 and does improve on dense retrieval.
