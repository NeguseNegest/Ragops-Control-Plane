# Week 5 Integration Review

## Outcome

Day 35 verified the service as one connected system rather than as isolated modules. FastAPI ran both from the production Compose image and from an isolated host process, dense retrieval queried the existing Docker Qdrant corpus, the 45-question label set was evaluated through `POST /query`, every returned trace was cross-checked in SQLite, and all four evidence-backed retrieval runs were reverified in live MLflow.

The integration acceptance passed after fixing two container defects: Linux installed a CUDA-enabled PyTorch dependency set for a CPU service, and an installed package resolved its config root under `site-packages` instead of `/app`.

## Full Local Service

| Component | Acceptance state | Evidence |
| --- | --- | --- |
| Qdrant | Running | Existing `rag_chunks` collection served real 384-dimensional dense searches on ports 6333–6334. |
| MLflow | Healthy | `ragops-retrieval` contains the four exact FINISHED evidence runs required by `configs/mlflow.yaml`. |
| FastAPI Compose service | Healthy | The rebuilt API passed its internal `/health` probe and was exposed on the isolated host port 8002. |
| Generation | Operational | The deterministic `template` provider returned cited responses with `zero_cost`; no external API key or paid call was used. |
| SQLite tracing | Operational | The Compose volume stored three container smoke traces, and a temporary host store preserved and verified all 45 evaluation traces and 450 child chunk rows. |

The existing development-server port was not commandeered. `RAGOPS_API_PORT` now allows the Compose API to expose a different host port while retaining container port 8000.

## Container Fixes

The Dockerfile now installs `torch==2.2.2+cpu` from the official PyTorch CPU wheel index before resolving application dependencies. The final image reports:

- PyTorch `2.2.2+cpu`;
- `torch.cuda.is_available() == False`;
- no importable `nvidia` package namespace; and
- no MLflow or Streamlit installation in the API image.

MLflow and dashboard packages moved to explicit `tracking` and `dashboard` extras while the development extra still installs both. `requirements-api.txt` gives Docker a source-independent dependency layer, and the project wheel is installed afterward with `--no-deps`. This reduced the measured image size from 597,711,103 bytes to 431,899,461 bytes, a reduction of 165,811,642 bytes (27.7%). It also means a source-only change rebuilds the small wheel layer instead of reinstalling every dependency.

`PipelineRuntime` now resolves its root from an explicit argument, `RAGOPS_PROJECT_ROOT`, or the process working directory. Compose supplies `/app`, which fixed the failed attempt to read `/usr/local/lib/python3.12/configs/dense_baseline.yaml`. The rebuilt container passed `pip check` before startup.

Compose sets `HF_HOME=/app/data/model_cache` and mounts `ragops_model_cache` there. A fresh environment still needs to acquire the configured models once, but later container recreation retains those files instead of tying the cache to one writable container layer. The seeded MiniLM cache occupied 88 MB. After an API restart, the same query succeeded using the retained cache; its cold process load took 8,226.05 ms instead of the fresh-volume request's 24,316.21 ms.

## Container Query

A real container request used `dense_baseline@1.0.0`, `top_k=3`, and the question “What does FastAPI read from the request body when you declare a Python type?” It returned HTTP 200 with three real indexed chunks, citations, template generation, and trace `cb2450bc-5282-4bb1-9fbb-eb4eab08c443`.

The cold request measured 27,926.85 ms total, including 27,682.85 ms for first-use embedding-model initialization, 17.20 ms for Qdrant search/result normalization, and 0.08 ms for template generation. The container SQLite volume contained the matching successful trace and three ordered child rows, all marked used for generation. The later cache-seeding and post-restart checks added traces `ba28c671-a650-4c18-98ca-187b05ae8e89` and `de96c03d-87b9-4c4a-bdd8-f309a5472fd2`, each with one used chunk. Final Compose counts were three traces, five retrieved-chunk rows, and zero feedback rows.

## Evaluation Through the API

`scripts/evaluate_api.py` sent all 45 verified retrieval questions through the live HTTP `/query` route with `dense_baseline`, `top_k=10`, and debug enabled. It validated every response schema, route/config/version, rank sequence, finite score, citation reference, generation-use reference, debug depth, and `X-Trace-ID` header.

| Metric | API evaluation | Recorded offline dense baseline |
| --- | ---: | ---: |
| MRR | 0.3359 | 0.3359 |
| Recall / Hit Rate@1 | 0.2667 | 0.2667 |
| Recall / Hit Rate@3 | 0.3111 | 0.3111 |
| Recall / Hit Rate@5 | 0.4444 | 0.4444 |
| Recall / Hit Rate@10 | 0.6000 | 0.6000 |
| nDCG@5 | 0.3473 | 0.3473 |
| Exact complete ranking matches | 45/45 | Reference |

All 45 complete top-10 chunk-ID rankings matched `reports/evaluations/dense_baseline.json` exactly, so every aggregate retrieval metric also matched. This demonstrates that the production HTTP composition has not drifted from the evaluated dense algorithm.

## API Latency

| Measurement | All 45 requests | Warm 44 requests |
| --- | ---: | ---: |
| Service average | 636.47 ms | 134.75 ms |
| Client round-trip average | 645.87 ms | 144.00 ms |
| Service p95 | 223.17 ms | 215.77 ms |
| Client round-trip p95 | 235.01 ms | 223.28 ms |
| Embedding average | 603.31 ms | 103.00 ms |
| Dense search average | 6.62 ms | 6.43 ms |
| Template generation average | 0.116 ms | 0.115 ms |

The first host request took 22,712.45 ms because model initialization is intentionally included in `embedding_ms` and total latency. Excluding that cold start, the API added approximately 9.26 ms beyond its service-reported time on average. BM25, fusion, and reranker timings were correctly null because this controlled parity run selected the dense route.

These values are integration measurements from one local machine, not a production capacity benchmark. The template provider exercises prompt/citation/generation composition but does not measure external-LLM quality or latency.

## Trace Verification

The evaluator used an isolated schema-v3 SQLite database. Counts moved from zero rows to:

- 45 terminal `traces` rows;
- 450 ordered `retrieved_chunks` rows; and
- zero feedback rows.

For every question it required the response and database to agree on UUID, endpoint, raw query, requested depth, pipeline name/version, success state, answer, retrieved count, total latency, all six component latency fields, chunk order, and generation-use flags. All 45 passed. The project-local trace database was not modified by this evaluation.

## MLflow Verification

The live verifier recomputed each configured evidence digest and checked parameters, metrics, provenance tags, effective configuration, and every required artifact before accepting these FINISHED runs:

| Pipeline | Run ID |
| --- | --- |
| Dense | `59c71a5926124ca59ef91c38ef9b69ae` |
| BM25 | `4b94ac6c0bee4cfa8ba83f106aa7de37` |
| RRF hybrid | `731c0e3766e64ed9a8d5fb239e98633a` |
| Hybrid plus cross-encoder | `a61923a5159d4042a36f2971fed00006` |

No new MLflow run was created: Day 35 verified the existing immutable evaluation evidence instead of duplicating it.

## Artifacts and Boundary

- `reports/evaluations/dense_baseline_api.json` contains all HTTP results, metrics, component timings, trace IDs, exact reference comparison, trace counts, and MLflow verification identities.
- `reports/evaluations/dense_baseline_api.csv` contains one retrieval-metric row per question.
- This report summarizes the controlled review and the integration fixes.

The 45-question parity run intentionally uses dense retrieval because it has a directly comparable offline report and is the default production API selection. Day 33 separately proved real dense, hybrid, and reranked request execution. Day 35 does not promote a pipeline, change registry aliases, evaluate generation quality, assign value to retrieval compute, or implement the Day 52 quality gate.
