# Dense vs BM25 Retrieval Baseline

## Executive summary

BM25 leads on MRR for this 50-question verified label set (57.7% BM25 vs 42.3% dense; +15.4 pp). BM25 wins 25 paired questions, dense wins 9, and 16 tie.

## Aggregate retrieval quality

| Metric | Dense | BM25 | BM25 − dense |
|---|---:|---:|---:|
| MRR | 42.3% | 57.7% | +15.4 pp |
| Hit rate@1 | 34.0% | 40.0% | +6.0 pp |
| Hit rate@3 | 46.0% | 74.0% | +28.0 pp |
| Hit rate@5 | 54.0% | 82.0% | +28.0 pp |
| Hit rate@10 | 68.0% | 84.0% | +16.0 pp |
| Recall@1 | 32.7% | 40.0% | +7.3 pp |
| Recall@3 | 42.7% | 70.7% | +28.0 pp |
| Recall@5 | 50.7% | 77.7% | +27.0 pp |
| Recall@10 | 64.7% | 82.7% | +18.0 pp |
| nDCG@1 | 34.0% | 40.0% | +6.0 pp |
| nDCG@3 | 38.8% | 58.8% | +19.9 pp |
| nDCG@5 | 42.2% | 61.6% | +19.5 pp |
| nDCG@10 | 46.7% | 63.6% | +16.9 pp |

Recall and hit rate are identical here because each verified question has one relevant chunk.

## Paired wins and query types

Query types are deterministic wording cohorts, not hand-retrofitted judgments: exact references include identifiers, endpoints, commands, fields, and parameters; behavioral/procedural queries ask how something behaves; all others are conceptual/descriptive.
BM25 recovers 10 dense top-k misses; dense recovers 2 BM25 top-k misses.

| Query type | Questions | BM25 wins | Dense wins | Ties | Dense MRR | BM25 MRR | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| behavioral/procedural | 13 | 4 | 4 | 5 | 55.7% | 50.3% | -5.4 pp |
| conceptual/descriptive | 22 | 13 | 4 | 5 | 40.8% | 66.8% | +26.0 pp |
| exact-reference | 15 | 8 | 1 | 6 | 32.8% | 50.8% | +18.0 pp |

### Questions where BM25 wins

| Question | Type | Dense rank | BM25 rank |
|---|---|---:|---:|
| What happens to the request_preview size and where is the full request stored? | exact-reference | miss | 1 |
| What occurs when a user chooses the "Create New Model" option in the Model dropdown menu? | behavioral/procedural | miss | 1 |
| What mechanism is used to implement a fine-grained permission system in FastAPI? | conceptual/descriptive | miss | 1 |
| Which distance measures are given as examples for calculating similarity between embeddings? | conceptual/descriptive | miss | 1 |
| Where must pydantic model type hints be defined to use pydantic objects as inputs for an MLflow PythonModel? | conceptual/descriptive | miss | 1 |
| What does an MLmodel file declare about each model to enable serving it as a Python function? | conceptual/descriptive | miss | 1 |
| What options are available in the Model dropdown menu on the model registration form? | conceptual/descriptive | 9 | 1 |
| Which standards can be used to establish a secure authentication setup in FastAPI according to the tutorial recap? | conceptual/descriptive | 5 | 1 |
| In which section of the MLflow Run details page can a user find and select the model folder to register? | conceptual/descriptive | 4 | 1 |
| What HTTP method and path are used to observe the cluster state after synchronization? | exact-reference | 3 | 1 |
| What is an MLflow model alias, and how can a champion alias change which model version production loads? | conceptual/descriptive | miss | 2 |
| What Python object does UploadFile expose that can be passed to libraries expecting a file-like object? | conceptual/descriptive | 2 | 1 |
| What is the main difference between the legacy permission model and the role-based permission model in MLflow? | behavioral/procedural | 8 | 2 |
| What two primary properties are required inside the lookup_from configuration when referencing a point in another collection? | exact-reference | 8 | 2 |
| In what order does FastAPI evaluate incoming requests when serving static frontend files alongside API routes? | behavioral/procedural | 6 | 2 |
| What is the exact MLflow command to serve a model located at runs:/<RUN_ID>/model? | exact-reference | miss | 3 |
| What warning is given about handling a plain password in this section? | conceptual/descriptive | miss | 3 |
| Which parameter in a Qdrant query allows retrieving a vector using a point ID located in a different collection? | exact-reference | 4 | 2 |
| What initial methods can be used to retrieve a subset of documents before applying a reranker? | conceptual/descriptive | 4 | 2 |
| Which nested field key is used in the HTTP, Python, TypeScript, Rust, and Java examples to specify the sightseeing condition? | exact-reference | miss | 4 |
| What file format does MLflow use for saving models from a variety of tools? | exact-reference | 9 | 3 |
| Can I use a wildcard CORS origin in FastAPI when requests include cookies or Authorization headers? | conceptual/descriptive | 3 | 2 |
| What dimensionality and distance-metric constraints apply to vectors in one Qdrant collection? | conceptual/descriptive | 3 | 2 |
| How can extra cookies be forbidden in a FastAPI cookie parameter model? | exact-reference | 10 | 5 |
| How should a callable dependency be passed to FastAPI's Depends, and who invokes it? | behavioral/procedural | 9 | 5 |

### Questions where dense retrieval wins

| Question | Type | Dense rank | BM25 rank |
|---|---|---:|---:|
| When should I use FastAPI BackgroundTasks, and when should I consider a tool such as Celery instead? | behavioral/procedural | 1 | miss |
| Which specific third-party packages are mentioned as easily usable with FastAPI for security handling? | conceptual/descriptive | 1 | 6 |
| Which MLflow flavor can log scikit-learn models as artifacts for serving? | conceptual/descriptive | 1 | 5 |
| How does MLflow determine the input and output schema when logging a PythonModel that includes type hints? | behavioral/procedural | 1 | 3 |
| How do I create and log a pandas training dataset with MLflow while retaining its source and training context? | behavioral/procedural | 1 | 2 |
| For Qdrant multitenancy, when is one payload-partitioned collection preferred over multiple collections? | conceptual/descriptive | 1 | 2 |
| What occurs if an explicit signature parameter is provided when logging a PythonModel with type hints? | exact-reference | 1 | 2 |
| Which methods can be used to serve static frontend applications in FastAPI? | conceptual/descriptive | 1 | 2 |
| What is the difference between MLflow's backend store and artifact store? | behavioral/procedural | 3 | miss |

## Latency context

| Retriever | Average | Minimum | Maximum |
|---|---:|---:|---:|
| Dense | 965.5 ms | 40.7 ms | 42239.7 ms |
| BM25 | 69.2 ms | 41.0 ms | 122.2 ms |

These timings are diagnostic, not a controlled benchmark. Dense latency includes query embedding and Qdrant search, with its recorded first-query model warm-up; BM25 latency measures in-memory scoring after the persisted index is loaded.

## Reproduction and limits

- Both runs use the same 50 verified questions and cutoffs [1, 3, 5, 10].
- Per-question wins compare the rank of the first relevant chunk; two misses or equal ranks are ties.
- The verified set is intentionally small and source-balanced enough for iteration, not statistical proof of production superiority.
- Questions were generated from and verified against exact source chunks, so retained source vocabulary can favor lexical retrieval.
- Use the machine-readable comparison artifact for downstream hybrid-retrieval experiments.
