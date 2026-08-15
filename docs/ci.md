# API Continuous Integration

## Day 34 Goal

The Day 34 gate makes the API request path reproducible in a fresh GitHub Actions runner. It covers `GET /health`, `POST /retrieve`, `POST /query`, and invalid request handling using real production composition while replacing only resources that would otherwise require a network, Docker, or a large model download.

## Test Topology

```text
FastAPI TestClient
        |
        +-> create_app
               |
               +-> PipelineRuntime / DenseRetriever
               |        |
               |        +-> deterministic query vector
               |        +-> in-memory Qdrant
               |
               +-> template generation
               +-> temporary SQLite TraceStore
```

`configs/ci_small.yaml` defines the collection name, vector size, checked-in JSONL path, and three named queries with exact vectors and expected top chunks. `tests/fixtures/ci_small_corpus.jsonl` contains four small records representing FastAPI, Qdrant, MLflow, and general RAG operations. The fixture loader strictly checks unknown fields, UUID chunk IDs, 64-character hashes, finite and correctly sized vectors, unique query/chunk identities, and valid expected-chunk references before the API starts.

The production dense retriever accepts an optional query-embedding callable at its runtime boundary. Normal callers still resolve the existing MiniLM embedder. The CI application injects an exact map from the three fixture questions to three-dimensional vectors; this isolates request orchestration from model availability without mocking Qdrant ranking or bypassing the retriever factory.

Each test gets a fresh in-memory Qdrant client and a temporary schema-v3 SQLite database. A lightweight client adapter makes request-level `close()` calls harmless while retaining the same backing in-memory collection for the life of that test. The actual Qdrant client is closed at fixture teardown.

## Coverage

The end-to-end fixture asserts HTTP payloads and their durable effects:

- health response and trace absence;
- dense retrieval rank/order, component timing shape, and persisted evidence;
- query route/config/version, trace ID header parity, citations, generation-use flags, debug metadata, and template zero-cost semantics;
- FastAPI 422 behavior for missing fields, invalid `top_k`, unknown configs, and incorrect types; and
- traced handler-level HTTP 400 behavior for a whitespace query.

The focused target also runs the API-evaluator, existing API, pipeline-runtime, Day 37 routing-probe, generation/cost/provider, trace-context/store, and dense-retrieval regression tests. This preserves failure-path, live-report validation logic, probe-feature validation, and selectable-pipeline coverage that does not need to be duplicated in the small-corpus file.

## GitHub Actions Job

`.github/workflows/ci.yml` runs on pushes, pull requests, and manual dispatch. The job:

1. checks out the repository and configures Python 3.12;
2. installs `requirements-ci.txt` rather than the full ML/application dependency set;
3. lints source, tests, scripts, and dashboard code with Ruff; and
4. runs `make test-api-ci PYTHON=python`.

`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` ensure an accidental Hugging Face dependency fails immediately. `RAGOPS_LLM_PROVIDER=template` prevents an external generation provider from being selected. No Qdrant/MLflow container, API key, corpus download, model download, or persistent database is needed.

Run the same target locally with:

```bash
make test-api-ci PYTHON=.venv/bin/python
```

## Boundary

This is an API reliability gate, not the Day 52 evaluation gate or the broader Day 53 multi-job CI design. It proves that the checked-in endpoint contract and core error behavior run in CI. It does not execute the full documentation corpus, measure retrieval/generation quality thresholds, contact MLflow, load the BM25 artifact, score with the cross-encoder, exercise an external LLM, or validate Docker deployment. Those broader integrations remain explicit later milestones.
