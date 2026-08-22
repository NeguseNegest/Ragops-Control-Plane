# Dense vs BM25 vs RRF Hybrid Benchmark

## Executive summary

Hybrid improves on both recorded component baselines by MRR. On 50 paired verified questions, MRR is 42.3% dense, 57.7% BM25, and 59.5% hybrid. Hybrid changes MRR by +17.3 pp versus dense and +1.8 pp versus BM25.

## Benchmark table

| Metric | Dense | BM25 | RRF hybrid | Hybrid − dense | Hybrid − BM25 |
|---|---:|---:|---:|---:|---:|
| MRR | 42.3% | 57.7% | 59.5% | +17.3 pp | +1.8 pp |
| Hit rate@1 | 34.0% | 40.0% | 48.0% | +14.0 pp | +8.0 pp |
| Hit rate@3 | 46.0% | 74.0% | 66.0% | +20.0 pp | -8.0 pp |
| Hit rate@5 | 54.0% | 82.0% | 76.0% | +22.0 pp | -6.0 pp |
| Hit rate@10 | 68.0% | 84.0% | 86.0% | +18.0 pp | +2.0 pp |
| Recall@1 | 32.7% | 40.0% | 46.7% | +14.0 pp | +6.7 pp |
| Recall@3 | 42.7% | 70.7% | 62.7% | +20.0 pp | -8.0 pp |
| Recall@5 | 50.7% | 77.7% | 72.7% | +22.0 pp | -5.0 pp |
| Recall@10 | 64.7% | 82.7% | 82.7% | +18.0 pp | +0.0 pp |
| nDCG@1 | 34.0% | 40.0% | 48.0% | +14.0 pp | +8.0 pp |
| nDCG@3 | 38.8% | 58.8% | 56.8% | +17.9 pp | -2.0 pp |
| nDCG@5 | 42.2% | 61.6% | 60.8% | +18.6 pp | -0.8 pp |
| nDCG@10 | 46.7% | 63.6% | 64.2% | +17.5 pp | +0.6 pp |

Recall and hit rate are identical because each current question has one labeled relevant chunk.

## Paired rank outcomes

| Comparison | Hybrid wins | Other wins | Ties | Hybrid recovers miss | Hybrid loses hit |
|---|---:|---:|---:|---:|---:|
| Hybrid vs dense | 24 | 3 | 23 | 9 | 0 |
| Hybrid vs BM25 | 14 | 10 | 26 | 2 | 1 |

Against the better component rank on each individual question, hybrid wins 5, loses 13, and ties 32.

## Wording cohorts

| Query type | Questions | Dense MRR | BM25 MRR | Hybrid MRR | Hybrid − BM25 | Hybrid wins | BM25 wins | Ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| behavioral/procedural | 13 | 55.7% | 50.3% | 56.7% | +6.5 pp | 6 | 2 | 5 |
| conceptual/descriptive | 22 | 40.8% | 66.8% | 67.0% | +0.2 pp | 5 | 5 | 12 |
| exact-reference | 15 | 32.8% | 50.8% | 51.0% | +0.2 pp | 3 | 3 | 9 |

## Hybrid gains over BM25

| Question | Type | Dense | BM25 | Hybrid |
|---|---|---:|---:|---:|
| Which specific third-party packages are mentioned as easily usable with FastAPI for security handling? | conceptual/descriptive | 1 | 6 | 1 |
| Which MLflow flavor can log scikit-learn models as artifacts for serving? | conceptual/descriptive | 1 | 5 | 1 |
| What two primary properties are required inside the lookup_from configuration when referencing a point in another collection? | exact-reference | 8 | 2 | 1 |
| Which methods can be used to serve static frontend applications in FastAPI? | conceptual/descriptive | 1 | 2 | 1 |
| What initial methods can be used to retrieve a subset of documents before applying a reranker? | conceptual/descriptive | 4 | 2 | 1 |
| What occurs if an explicit signature parameter is provided when logging a PythonModel with type hints? | exact-reference | 1 | 2 | 1 |
| In what order does FastAPI evaluate incoming requests when serving static frontend files alongside API routes? | behavioral/procedural | 6 | 2 | 1 |
| For Qdrant multitenancy, when is one payload-partitioned collection preferred over multiple collections? | conceptual/descriptive | 1 | 2 | 1 |
| How do I create and log a pandas training dataset with MLflow while retaining its source and training context? | behavioral/procedural | 1 | 2 | 1 |
| When should I use FastAPI BackgroundTasks, and when should I consider a tool such as Celery instead? | behavioral/procedural | 1 | miss | 4 |
| What file format does MLflow use for saving models from a variety of tools? | exact-reference | 9 | 3 | 2 |
| How does MLflow determine the input and output schema when logging a PythonModel that includes type hints? | behavioral/procedural | 1 | 3 | 2 |
| How should a callable dependency be passed to FastAPI's Depends, and who invokes it? | behavioral/procedural | 9 | 5 | 3 |
| What is the difference between MLflow's backend store and artifact store? | behavioral/procedural | 3 | miss | 8 |

## Hybrid regressions versus BM25

| Question | Type | Dense | BM25 | Hybrid |
|---|---|---:|---:|---:|
| Where must pydantic model type hints be defined to use pydantic objects as inputs for an MLflow PythonModel? | conceptual/descriptive | miss | 1 | miss |
| What does an MLmodel file declare about each model to enable serving it as a Python function? | conceptual/descriptive | miss | 1 | 5 |
| What happens to the request_preview size and where is the full request stored? | exact-reference | miss | 1 | 5 |
| Which distance measures are given as examples for calculating similarity between embeddings? | conceptual/descriptive | miss | 1 | 4 |
| What occurs when a user chooses the "Create New Model" option in the Model dropdown menu? | behavioral/procedural | miss | 1 | 3 |
| What is an MLflow model alias, and how can a champion alias change which model version production loads? | conceptual/descriptive | miss | 2 | 7 |
| What is the exact MLflow command to serve a model located at runs:/<RUN_ID>/model? | exact-reference | miss | 3 | 8 |
| What warning is given about handling a plain password in this section? | conceptual/descriptive | miss | 3 | 7 |
| What is the main difference between the legacy permission model and the role-based permission model in MLflow? | behavioral/procedural | 8 | 2 | 3 |
| Which nested field key is used in the HTTP, Python, TypeScript, Rust, and Java examples to specify the sightseeing condition? | exact-reference | miss | 4 | 8 |

## Hybrid top-10 failures

| Question | Source | Dense | BM25 | Hybrid |
|---|---|---:|---:|---:|
| What does FastAPI read from the request body when you declare a Python type? | fastapi/docs/tutorial/body.md | miss | miss | miss |
| What HTTP endpoint is shown for scrolling points in a collection? | qdrant/qdrant_llms_full.txt | miss | miss | miss |
| Which library should be installed to use EmailStr according to the document? | fastapi/docs/tutorial/response-model.md | miss | miss | miss |
| Where must pydantic model type hints be defined to use pydantic objects as inputs for an MLflow PythonModel? | mlflow/docs/docs/classic-ml/model/python_model.mdx | miss | 1 | miss |
| After saving a minimal FastAPI application as main.py, which command starts the development server, and where can I open the interactive API documentation? | fastapi/docs/tutorial/first-steps.md | miss | miss | miss |
| How do I declare a JSON request body in FastAPI, and what does FastAPI do with it? | fastapi/docs/tutorial/body.md | miss | miss | miss |
| Which MLflow calls does the tracking quickstart use to manually record hyperparameters, a scikit-learn model, an accuracy metric, and a descriptive tag? | mlflow/docs/docs/classic-ml/tracking/quickstart/index.mdx | miss | miss | miss |

## Latency

| Recorded run | Average | After first query | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Dense baseline | 965.5 ms | 123.1 ms | 40.7 ms | 42239.7 ms |
| BM25 baseline | 69.2 ms | 69.3 ms | 41.0 ms | 122.2 ms |
| RRF hybrid | 700.8 ms | 184.2 ms | 91.3 ms | 26011.5 ms |

Hybrid component timings from the same run:

| Stage | Average | Minimum | Maximum |
|---|---:|---:|---:|
| Dense | 9.9 ms | 6.0 ms | 20.7 ms |
| BM25 | 68.9 ms | 38.3 ms | 110.5 ms |
| Fusion | 0.2 ms | 0.1 ms | 0.2 ms |

The historical dense and BM25 latency rows come from separate runs and are not controlled head-to-head latency measurements. Hybrid total latency is end-to-end sequential dense retrieval, BM25 scoring, and fusion; its component table is internally comparable. First-query model warm-up can dominate averages.

## Validity and decision

- The live hybrid run used 13481 dense points and the BM25 source artifact contains 13481 records (13476 searchable, 5 tokenless skips).
- BM25 source SHA256: `61d97c037bfedda6d4c6ce66127392bc490435d2f475945c845fd28dc450fe50`.
- All three reports use identical question IDs, question text, expected sources, relevant chunk IDs, and metric cutoffs.
- The historical dense report does not persist a corpus hash; live dense/BM25 record-count parity reduces but does not eliminate snapshot uncertainty.
- The 45 source-derived questions map to only 20 labeled chunks, have one relevance judgment each, and retain lexical overlap that can favor BM25.
- Giving each of the 33 unique relevance groups equal weight still leaves hybrid MRR +0.4 pp versus BM25; hybrid wins 9 groups, BM25 wins 9, and 15 tie.
- Wording cohorts are deterministic heuristics, not independent human query-type labels.
- Unweighted RRF rewards cross-retriever consensus. On this lexically aligned set, that can demote strong BM25-only evidence below weaker chunks appearing in both lists.

Decision on the Day 25 primary metric: **RRF hybrid has the highest MRR**. Hybrid does improve on BM25 and does improve on dense retrieval.
