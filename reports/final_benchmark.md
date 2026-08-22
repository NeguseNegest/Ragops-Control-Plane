# Final Benchmark and Ablation Run

## Central result

| Pipeline | Recall@5 | MRR@5 | Faithfulness (1-5) | Answer relevance (1-5) | Refusal correctness | p50 retrieval | p95 retrieval | Estimated generation cost/query | MLflow run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dense | 50.7% | 40.6% | 5.00 | 4.30 | N/A | 119.2 ms | 226.0 ms | $0.000067 | [15d6f764](http://127.0.0.1:5001/#/experiments/1/runs/15d6f764d1ba474ea4556f34948f2444) |
| BM25 | 77.7% | 57.4% | 5.00 | 4.50 | N/A | 67.9 ms | 93.0 ms | $0.000083 | [cc073a85](http://127.0.0.1:5001/#/experiments/1/runs/cc073a8526ee4b809f3d40d891c4be00) |
| Hybrid (RRF) | 72.7% | 58.2% | 4.80 | 4.30 | N/A | 174.5 ms | 272.6 ms | $0.000080 | [a1e03481](http://127.0.0.1:5001/#/experiments/1/runs/a1e0348130a646b8b7b6cf0c02d43f2d) |
| Hybrid + reranker | 81.0% | 64.7% | 5.00 | 4.50 | N/A | 4014.8 ms | 7669.2 ms | $0.000084 | [aff82d09](http://127.0.0.1:5001/#/experiments/1/runs/aff82d0914f8403e9479812a90096e09) |
| Routed | 66.0% | 53.1% | 5.00 | 4.60 | 83.3% | 3109.8 ms | 7821.4 ms | $0.000076 | [45b98920](http://127.0.0.1:5001/#/experiments/1/runs/45b98920cd68421c8d7094fae7128604) |

MRR is intentionally reported as MRR@5 so every pipeline is compared at the same final depth. Fixed retrieval pipelines show refusal correctness as N/A because they do not implement the router's explicit no-answer policy.

## Ablation counts

- BM25 beats dense: 25 questions.
- Reranking helps its own RRF-25 candidate order: 14 questions.
- Reranking hurts its own RRF-25 candidate order: 8 questions.
- Routing reduces projected generation cost versus always-reranked: 11 questions.
- Routing harms top-5 reciprocal-rank quality versus always-reranked: 10 questions.

## Retrieval wins by query type

All retrieval-labeled questions are supported, so the golden query-type view has one cohort. The second view uses the predeclared deterministic wording cohorts from the BM25 evaluation; unsupported behavior is evaluated separately as refusal correctness.

### Golden Query Type

| Cohort | Questions | Dense wins | BM25 wins | Hybrid wins | Reranked wins | Routed wins | Ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| supported | 50 | 1 | 5 | 1 | 2 | 0 | 41 |

### Retrieval Wording Cohort

| Cohort | Questions | Dense wins | BM25 wins | Hybrid wins | Reranked wins | Routed wins | Ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| behavioral/procedural | 13 | 1 | 1 | 0 | 0 | 0 | 11 |
| conceptual/descriptive | 22 | 0 | 2 | 0 | 1 | 0 | 19 |
| exact-reference | 15 | 0 | 2 | 1 | 1 | 0 | 11 |

## Cases where BM25 beats dense

| ID | Question | Wording cohort |
|---|---|---|
| sqa-8111f18ad4679ac4 | Which nested field key is used in the HTTP, Python, TypeScript, Rust, and Java examples to specify the sightseeing condition? | exact-reference |
| sqa-220fbcfa80439038 | What Python object does UploadFile expose that can be passed to libraries expecting a file-like object? | conceptual/descriptive |
| sqa-7f4b9c901bcac8d2 | What file format does MLflow use for saving models from a variety of tools? | exact-reference |
| sqa-43e609692540e39f | What is the exact MLflow command to serve a model located at runs:/<RUN_ID>/model? | exact-reference |
| sqa-fe6afefa50c92384 | What does an MLmodel file declare about each model to enable serving it as a Python function? | conceptual/descriptive |
| sqa-ebcd24ac3da514b9 | What HTTP method and path are used to observe the cluster state after synchronization? | exact-reference |
| sqa-50e37593177ea153 | What warning is given about handling a plain password in this section? | conceptual/descriptive |
| sqa-1333a2c4d6953cf4 | What happens to the request_preview size and where is the full request stored? | exact-reference |
| sqa-c550d812bb5272e0 | Which distance measures are given as examples for calculating similarity between embeddings? | conceptual/descriptive |
| sqa-ade53f89f8629636 | Which standards can be used to establish a secure authentication setup in FastAPI according to the tutorial recap? | conceptual/descriptive |
| sqa-9bbefdf430f9c31f | What mechanism is used to implement a fine-grained permission system in FastAPI? | conceptual/descriptive |
| sqa-ceaead1685abb224 | Where must pydantic model type hints be defined to use pydantic objects as inputs for an MLflow PythonModel? | conceptual/descriptive |
| sqa-673de786b69493ab | What initial methods can be used to retrieve a subset of documents before applying a reranker? | conceptual/descriptive |
| sqa-4cb482f1f218af58 | In what order does FastAPI evaluate incoming requests when serving static frontend files alongside API routes? | behavioral/procedural |
| sqa-0962a7db4ec6c78a | In which section of the MLflow Run details page can a user find and select the model folder to register? | conceptual/descriptive |
| sqa-a8ee8614cd6dc87b | What options are available in the Model dropdown menu on the model registration form? | conceptual/descriptive |
| sqa-7622236114712036 | What occurs when a user chooses the "Create New Model" option in the Model dropdown menu? | behavioral/procedural |
| sqa-6055940f3e30c975 | Which parameter in a Qdrant query allows retrieving a vector using a point ID located in a different collection? | exact-reference |
| sqa-d20c9187fa7725e6 | What two primary properties are required inside the lookup_from configuration when referencing a point in another collection? | exact-reference |
| sqa-8397e93659723d06 | How can extra cookies be forbidden in a FastAPI cookie parameter model? | exact-reference |
| sqa-a6c95a648b8cccd8 | What is the main difference between the legacy permission model and the role-based permission model in MLflow? | behavioral/procedural |
| gqa-005 | How should a callable dependency be passed to FastAPI's Depends, and who invokes it? | behavioral/procedural |
| gqa-009 | Can I use a wildcard CORS origin in FastAPI when requests include cookies or Authorization headers? | conceptual/descriptive |
| gqa-014 | What is an MLflow model alias, and how can a champion alias change which model version production loads? | conceptual/descriptive |
| gqa-018 | What dimensionality and distance-metric constraints apply to vectors in one Qdrant collection? | conceptual/descriptive |

## Cases where reranking helps

| ID | Question | Wording cohort |
|---|---|---|
| sqa-8111f18ad4679ac4 | Which nested field key is used in the HTTP, Python, TypeScript, Rust, and Java examples to specify the sightseeing condition? | exact-reference |
| sqa-7f4b9c901bcac8d2 | What file format does MLflow use for saving models from a variety of tools? | exact-reference |
| sqa-43e609692540e39f | What is the exact MLflow command to serve a model located at runs:/<RUN_ID>/model? | exact-reference |
| sqa-fe6afefa50c92384 | What does an MLmodel file declare about each model to enable serving it as a Python function? | conceptual/descriptive |
| sqa-50e37593177ea153 | What warning is given about handling a plain password in this section? | conceptual/descriptive |
| sqa-c550d812bb5272e0 | Which distance measures are given as examples for calculating similarity between embeddings? | conceptual/descriptive |
| sqa-5c5e1373d68176f8 | How does MLflow determine the input and output schema when logging a PythonModel that includes type hints? | behavioral/procedural |
| sqa-ceaead1685abb224 | Where must pydantic model type hints be defined to use pydantic objects as inputs for an MLflow PythonModel? | conceptual/descriptive |
| sqa-6055940f3e30c975 | Which parameter in a Qdrant query allows retrieving a vector using a point ID located in a different collection? | exact-reference |
| sqa-8397e93659723d06 | How can extra cookies be forbidden in a FastAPI cookie parameter model? | exact-reference |
| sqa-a6c95a648b8cccd8 | What is the main difference between the legacy permission model and the role-based permission model in MLflow? | behavioral/procedural |
| gqa-007 | When should I use FastAPI BackgroundTasks, and when should I consider a tool such as Celery instead? | behavioral/procedural |
| gqa-014 | What is an MLflow model alias, and how can a champion alias change which model version production loads? | conceptual/descriptive |
| gqa-018 | What dimensionality and distance-metric constraints apply to vectors in one Qdrant collection? | conceptual/descriptive |

## Cases where reranking hurts

| ID | Question | Wording cohort |
|---|---|---|
| sqa-1333a2c4d6953cf4 | What happens to the request_preview size and where is the full request stored? | exact-reference |
| sqa-9bbefdf430f9c31f | What mechanism is used to implement a fine-grained permission system in FastAPI? | conceptual/descriptive |
| sqa-7d97ec3c52aa5853 | What primary trade-off do rerankers make when evaluating document relevance? | behavioral/procedural |
| sqa-0962a7db4ec6c78a | In which section of the MLflow Run details page can a user find and select the model folder to register? | conceptual/descriptive |
| sqa-d20c9187fa7725e6 | What two primary properties are required inside the lookup_from configuration when referencing a point in another collection? | exact-reference |
| gqa-013 | How can I enable MLflow autologging only for PyTorch, or disable it for scikit-learn while keeping generic autologging enabled? | behavioral/procedural |
| gqa-016 | How do I create and log a pandas training dataset with MLflow while retaining its source and training context? | behavioral/procedural |
| gqa-025 | For Qdrant multitenancy, when is one payload-partitioned collection preferred over multiple collections? | conceptual/descriptive |

## Cases where routing reduces cost

| ID | Question | Wording cohort |
|---|---|---|
| sqa-8111f18ad4679ac4 | Which nested field key is used in the HTTP, Python, TypeScript, Rust, and Java examples to specify the sightseeing condition? | exact-reference |
| sqa-806e7d79beff73b8 | Which MLflow flavor can log scikit-learn models as artifacts for serving? | conceptual/descriptive |
| sqa-ebcd24ac3da514b9 | What HTTP method and path are used to observe the cluster state after synchronization? | exact-reference |
| sqa-50e37593177ea153 | What warning is given about handling a plain password in this section? | conceptual/descriptive |
| sqa-dc321a1b57e8d142 | Which library should be installed to use EmailStr according to the document? | exact-reference |
| sqa-1333a2c4d6953cf4 | What happens to the request_preview size and where is the full request stored? | exact-reference |
| sqa-c00b9f531d2e5586 | Which file contains the information required to restore the model environment using virtualenv, and what three things does it specify? | conceptual/descriptive |
| sqa-580f65fefad3cf3a | What is the primary function of a reranker in document search? | conceptual/descriptive |
| sqa-a8ee8614cd6dc87b | What options are available in the Model dropdown menu on the model registration form? | conceptual/descriptive |
| sqa-d20c9187fa7725e6 | What two primary properties are required inside the lookup_from configuration when referencing a point in another collection? | exact-reference |
| gqa-005 | How should a callable dependency be passed to FastAPI's Depends, and who invokes it? | behavioral/procedural |

## Cases where routing harms quality

| ID | Question | Wording cohort |
|---|---|---|
| sqa-8111f18ad4679ac4 | Which nested field key is used in the HTTP, Python, TypeScript, Rust, and Java examples to specify the sightseeing condition? | exact-reference |
| sqa-220fbcfa80439038 | What Python object does UploadFile expose that can be passed to libraries expecting a file-like object? | conceptual/descriptive |
| sqa-ebcd24ac3da514b9 | What HTTP method and path are used to observe the cluster state after synchronization? | exact-reference |
| sqa-50e37593177ea153 | What warning is given about handling a plain password in this section? | conceptual/descriptive |
| sqa-9bbefdf430f9c31f | What mechanism is used to implement a fine-grained permission system in FastAPI? | conceptual/descriptive |
| sqa-ceaead1685abb224 | Where must pydantic model type hints be defined to use pydantic objects as inputs for an MLflow PythonModel? | conceptual/descriptive |
| sqa-a8ee8614cd6dc87b | What options are available in the Model dropdown menu on the model registration form? | conceptual/descriptive |
| sqa-6055940f3e30c975 | Which parameter in a Qdrant query allows retrieving a vector using a point ID located in a different collection? | exact-reference |
| sqa-d20c9187fa7725e6 | What two primary properties are required inside the lookup_from configuration when referencing a point in another collection? | exact-reference |
| gqa-005 | How should a callable dependency be passed to FastAPI's Depends, and who invokes it? | behavioral/procedural |

## Measurement contract and limitations

- Retrieval Quality: 50 paired reviewed supported questions; all rankings truncated to top 5; MRR is MRR@5.
- Answer Quality: cross-provider LLM judge on the same 10 explicit supported questions per pipeline; scores are 1-5 rubric means.
- Refusal Correctness: routed policy only, across all 30 reviewed unsupported/adversarial questions; fixed pipelines are not applicable.
- Latency: retrieval-only wall-clock measurements including cold start; routed values use documented serial artifact replay.
- Cost: paired heuristic token projection over all supported retrieval questions using exact prompts and the same verified reference answer.
- Percentiles: linear interpolation over ordered per-question measurements.
- Answer-quality judge scores are estimates on a fixed 10-question supported sample, not human ground truth.
- Unsupported queries have refusal labels but no relevant-chunk labels, so retrieval Recall/MRR apply only to supported questions.
- Routed latency is a serial composition of measured artifacts; STANDARD uses dense top-10 latency as both probe and full-retrieval proxy.
- Cost excludes local embedding, sparse retrieval, reranking compute, judge calls, caching, and infrastructure; it estimates generation token charges only.
- The routed policy remains draft and its false-refusal behavior is visible in the supported retrieval metrics.
