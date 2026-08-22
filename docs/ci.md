# Continuous integration

GitHub Actions runs five independent jobs on pushes, pull requests, and manual dispatch.

| Job | Command | Current scope |
| --- | --- | --- |
| `lint` | `python -m ruff check src tests scripts dashboard` | Python lint |
| `unit` | `make test-unit-ci PYTHON=python` | Hermetic unit and artifact-contract suite |
| `api` | `make test-api-ci PYTHON=python` | API/control-plane suite |
| `evaluation-smoke` | `make test-evaluation-smoke EVAL_GATE_PYTHON=python` | Gate contract suite |
| `evaluation-gate` | `make eval-gate EVAL_GATE_PYTHON=python` | Live compact gate |

Jobs share `requirements-ci.txt`, Python 3.12, pip caching, read-only repository permissions, and a ten-minute timeout. They have no dependency edges, so one failure does not skip the others.

## Hermetic test path

CI uses:

- a checked-in four-document corpus;
- deterministic three-dimensional query vectors;
- in-memory Qdrant;
- the real dense retriever and router;
- template generation; and
- temporary SQLite.

No Docker service, API key, external LLM, model download, full corpus, or persistent database is required. Offline environment variables make accidental model/provider access fail immediately.

## Evaluation gate

The five-case gate has three supported and two unsupported queries. It checks:

1. Recall@2
2. Recall regression
3. MRR
4. Answer presence
5. Citation coverage
6. Citation precision
7. Refusal correctness
8. p95 latency
9. Error count

All thresholds live in [`configs/eval_gate.yaml`](../configs/eval_gate.yaml). The command exits non-zero on any failure.

## Run locally

```bash
make lint
make test-unit-ci PYTHON=.venv/bin/python
make test-api-ci PYTHON=.venv/bin/python
make test-evaluation-smoke
make eval-gate
```

## Not covered in pull requests

- full-corpus ingestion or retrieval;
- the generated BM25 index;
- cross-encoder execution;
- external generation/judging;
- live MLflow verification;
- Streamlit runtime tests; and
- Docker deployment.

Run those paths separately; a fresh CI checkout does not contain their artifacts or credentials.
