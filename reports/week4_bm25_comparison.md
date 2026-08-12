# Dense vs BM25 Retrieval Baseline

## Executive summary

BM25 leads on MRR for this 45-question verified label set (61.9% BM25 vs 33.6% dense; +28.3 pp). BM25 wins 27 paired questions, dense wins 6, and 12 tie.

## Aggregate retrieval quality

| Metric | Dense | BM25 | BM25 − dense |
|---|---:|---:|---:|
| MRR | 33.6% | 61.9% | +28.3 pp |
| Hit rate@1 | 26.7% | 46.7% | +20.0 pp |
| Hit rate@3 | 31.1% | 75.6% | +44.4 pp |
| Hit rate@5 | 44.4% | 84.4% | +40.0 pp |
| Hit rate@10 | 60.0% | 86.7% | +26.7 pp |
| Recall@1 | 26.7% | 46.7% | +20.0 pp |
| Recall@3 | 31.1% | 75.6% | +44.4 pp |
| Recall@5 | 44.4% | 84.4% | +40.0 pp |
| Recall@10 | 60.0% | 86.7% | +26.7 pp |
| nDCG@1 | 26.7% | 46.7% | +20.0 pp |
| nDCG@3 | 29.2% | 63.7% | +34.5 pp |
| nDCG@5 | 34.7% | 67.3% | +32.5 pp |
| nDCG@10 | 39.6% | 68.1% | +28.4 pp |

Recall and hit rate are identical here because each verified question has one relevant chunk.

## Paired wins and query types

Query types are deterministic wording cohorts, not hand-retrofitted judgments: exact references include identifiers, endpoints, commands, fields, and parameters; behavioral/procedural queries ask how something behaves; all others are conceptual/descriptive.
BM25 recovers 12 dense top-k misses; dense recovers 0 BM25 top-k misses.

| Query type | Questions | BM25 wins | Dense wins | Ties | Dense MRR | BM25 MRR | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| behavioral/procedural | 8 | 4 | 2 | 2 | 53.6% | 62.9% | +9.3 pp |
| conceptual/descriptive | 23 | 15 | 3 | 5 | 34.4% | 74.8% | +40.4 pp |
| exact-reference | 14 | 8 | 1 | 5 | 20.9% | 40.1% | +19.3 pp |

### Questions where BM25 wins

| Question | Type | Dense rank | BM25 rank |
|---|---|---:|---:|
| What happens to the request_preview size and where is the full request stored? | exact-reference | miss | 1 |
| What operation is used to quantify the similarity between the query and document vectors? | conceptual/descriptive | miss | 1 |
| What occurs when a user chooses the "Create New Model" option in the Model dropdown menu? | behavioral/procedural | miss | 1 |
| What mechanism is used to implement a fine-grained permission system in FastAPI? | conceptual/descriptive | miss | 1 |
| What schemas are generated for your model and where can you use them? | conceptual/descriptive | miss | 1 |
| Which distance measures are given as examples for calculating similarity between embeddings? | conceptual/descriptive | miss | 1 |
| Where must pydantic model type hints be defined to use pydantic objects as inputs for an MLflow PythonModel? | conceptual/descriptive | miss | 1 |
| What does an MLmodel file declare about each model to enable serving it as a Python function? | conceptual/descriptive | miss | 1 |
| What options are available in the Model dropdown menu on the model registration form? | conceptual/descriptive | 9 | 1 |
| What intermediate representation is needed to implement similarity learning quickly? | conceptual/descriptive | 7 | 1 |
| Which standards can be used to establish a secure authentication setup in FastAPI according to the tutorial recap? | conceptual/descriptive | 5 | 1 |
| In which section of the MLflow Run details page can a user find and select the model folder to register? | conceptual/descriptive | 4 | 1 |
| What does the API return in the response when a browser creates a user with a password using the same input/output model in this example? | conceptual/descriptive | 4 | 1 |
| What HTTP method and path are used to observe the cluster state after synchronization? | exact-reference | 3 | 1 |
| What Python object does UploadFile expose that can be passed to libraries expecting a file-like object? | conceptual/descriptive | 2 | 1 |
| What happens if the submitted data is invalid? | behavioral/procedural | miss | 2 |
| What is the main difference between the legacy permission model and the role-based permission model in MLflow? | behavioral/procedural | 8 | 2 |
| What two primary properties are required inside the lookup_from configuration when referencing a point in another collection? | exact-reference | 8 | 2 |
| In what order does FastAPI evaluate incoming requests when serving static frontend files alongside API routes? | behavioral/procedural | 6 | 2 |
| What is the exact MLflow command to serve a model located at runs:/<RUN_ID>/model? | exact-reference | miss | 3 |
| What warning is given about handling a plain password in this section? | conceptual/descriptive | miss | 3 |
| What kind of interface does UploadFile expose? | conceptual/descriptive | 5 | 2 |
| Which parameter in a Qdrant query allows retrieving a vector using a point ID located in a different collection? | exact-reference | 4 | 2 |
| What initial methods can be used to retrieve a subset of documents before applying a reranker? | conceptual/descriptive | 4 | 2 |
| Which nested field key is used in the HTTP, Python, TypeScript, Rust, and Java examples to specify the sightseeing condition? | exact-reference | miss | 4 |
| What file format does MLflow use for saving models from a variety of tools? | exact-reference | 9 | 3 |
| How can extra cookies be forbidden in a FastAPI cookie parameter model? | exact-reference | 10 | 5 |

### Questions where dense retrieval wins

| Question | Type | Dense rank | BM25 rank |
|---|---|---:|---:|
| Which specific third-party packages are mentioned as easily usable with FastAPI for security handling? | conceptual/descriptive | 1 | 6 |
| Which MLflow flavor can log scikit-learn models as artifacts for serving? | conceptual/descriptive | 1 | 5 |
| What trade-off occurs when opting for a lower number of vector chunks? | behavioral/procedural | 1 | 5 |
| How does MLflow determine the input and output schema when logging a PythonModel that includes type hints? | behavioral/procedural | 1 | 3 |
| What occurs if an explicit signature parameter is provided when logging a PythonModel with type hints? | exact-reference | 1 | 2 |
| Which methods can be used to serve static frontend applications in FastAPI? | conceptual/descriptive | 1 | 2 |

## Latency context

| Retriever | Average | Minimum | Maximum |
|---|---:|---:|---:|
| Dense | 679.9 ms | 36.7 ms | 24009.5 ms |
| BM25 | 87.9 ms | 42.1 ms | 294.4 ms |

These timings are diagnostic, not a controlled benchmark. Dense latency includes query embedding and Qdrant search, with its recorded first-query model warm-up; BM25 latency measures in-memory scoring after the persisted index is loaded.

## Reproduction and limits

- Both runs use the same 45 verified questions and cutoffs [1, 3, 5, 10].
- Per-question wins compare the rank of the first relevant chunk; two misses or equal ranks are ties.
- The verified set is intentionally small and source-balanced enough for iteration, not statistical proof of production superiority.
- Questions were generated from and verified against exact source chunks, so retained source vocabulary can favor lexical retrieval.
- Use the machine-readable comparison artifact for downstream hybrid-retrieval experiments.
