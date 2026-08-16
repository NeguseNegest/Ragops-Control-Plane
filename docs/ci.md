# API Continuous Integration

## Day 34 Goal

The API gate makes the request path reproducible in a fresh GitHub Actions runner. Its original Day 34 contract covers `GET /health`, `POST /retrieve`, `POST /query`, and invalid request handling; Day 38 adds `POST /route`, Day 39 adds deterministic refusal/refusal evaluation, and Day 40 adds model-cost/token-estimator/schema-v4 persistence coverage. It uses real production composition while replacing only resources that would otherwise require a network, Docker, or a large model download.

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

Each test gets a fresh in-memory Qdrant client and a temporary schema-v4 SQLite database. A lightweight client adapter makes request-level `close()` calls harmless while retaining the same backing in-memory collection for the life of that test. The actual Qdrant client is closed at fixture teardown.

## Coverage

The end-to-end fixture asserts HTTP payloads and their durable effects:

- health response and trace absence;
- dense retrieval rank/order, component timing shape, and persisted evidence;
- routing through the real dense probe, exact decision/reason/execution intent, minimal evidence, timings, deterministic NO_ANSWER refusal shape, and trace absence because no final query is executed;
- query route/config/version, trace ID header parity, citations, generation-use flags, debug metadata, and exact response/trace template-cost parity;
- FastAPI 422 behavior for missing fields, invalid `top_k`, unknown configs, and incorrect types; and
- traced handler-level HTTP 400 behavior for a whitespace query.

The focused target also runs the API-evaluator, existing API, pipeline-runtime, Day 36 router-config/policy validation, Day 37 routing-probe, Day 38 route/reason boundary tests, Day 39 no-answer dataset/calibration/prompt/evaluator tests, Day 40 price-table/heuristic/arithmetic/schema-migration/cost-parity tests, Day 41 miniature paired-comparison tests, Day 42 tuning/distribution tests, generation/provider, trace-context/store, and dense-retrieval regressions. This preserves failure-path, live-report validation logic, threshold/registry/calibration guards, exact inequality/precedence behavior, probe-feature validation, deterministic refusal invariants, false-refusal measurement, honest cost provenance, and selectable-pipeline coverage that does not need to be duplicated in the small-corpus file.

Day 41 adds `make test-router-evaluation`, which uses temporary miniature dense/reranked/refusal/chunk/cost artifacts to cover strict config loading, paired provenance, route-decision drift, quality/latency/cost aggregation, missing prompt evidence, deterministic recomputation, and atomic output protection. Those hermetic tests also run in `test-api-ci`. `make validate-router-evaluation` additionally checks the full local 13,481-chunk artifact and canonical report. The full-artifact validator is intentionally outside the lightweight GitHub API job because `data/processed/chunks.jsonl` is a generated, ignored corpus artifact; CI does not pretend that file is present in a fresh checkout.

Day 42 adds `make test-router-stabilization`. Its temporary fixture checks the deterministic SHA256 split, constrained candidate selection, latency/threshold tie-breakers, target-policy drift rejection, route transitions, report completeness, and overwrite protection. The target also rechecks current `0.03` boundary behavior and no-answer replay provenance. `make validate-router-tuning` remains a local full-artifact gate for the same ignored-corpus reason as the Day 41 validator.

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
