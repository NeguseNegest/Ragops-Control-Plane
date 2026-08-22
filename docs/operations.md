# Operations

## Requirements

- Python 3.11 or 3.12
- Git
- Docker with Compose v2
- about 8 GB of free disk space

## Clean setup

```bash
cp .env.example .env
make setup
make fetch-sources
docker compose up -d --build
make ingest
make build-index
make evaluate
make test
make dashboard
```

`make fetch-sources` checks out commit-pinned Qdrant, FastAPI, and MLflow files. Existing source directories are never replaced unless the fetch script receives `--force`.

`make ingest` downloads MiniLM once, then embeds the corpus on the CPU; expect several minutes. `make build-index` writes vectors to Qdrant and builds `data/processed/bm25_index.json.gz`. `make evaluate` runs the final 50-question dense evaluation, writes local results under `data/processed/local_evaluation`, and logs the run to MLflow.

Open:

| Service | Default URL |
| --- | --- |
| Dashboard | `http://127.0.0.1:8501` |
| API docs | `http://127.0.0.1:8000/docs` |
| MLflow | `http://127.0.0.1:5000` |
| Qdrant | `http://127.0.0.1:6333/dashboard` |

## Host API instead of Docker API

```bash
docker compose stop api
make serve
```

Run `make dashboard` in another terminal. Both commands load `.env`. Docker Compose also loads `.env`.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `API_HOST`, `API_PORT` | Host-run `make serve` |
| `DASHBOARD_PORT` | Host-run Streamlit port |
| `RAGOPS_API_PORT` | Published Compose API port |
| `RAGOPS_API_URL` | API used by Streamlit |
| `QDRANT_URL` | Qdrant used by host commands |
| `QDRANT_HTTP_PORT`, `QDRANT_GRPC_PORT` | Published Compose Qdrant ports |
| `MLFLOW_TRACKING_URI`, `MLFLOW_PORT` | Host tracking URL and published port |
| `RAGOPS_TRACE_DB_PATH` | Host SQLite trace path |
| `RAGOPS_LLM_PROVIDER` | `template`, `openai`, or `gemini` |

If a published Qdrant or MLflow port changes, update its matching host URL in `.env` too. Compose services use internal ports and need no change.

## Real generation

The default provider is the offline template. For real answers, set one provider and key, then recreate the API:

```text
RAGOPS_LLM_PROVIDER=gemini
GEMINI_API_KEY=...
```

```bash
docker compose up -d --build --force-recreate api
```

OpenAI is selected with `RAGOPS_LLM_PROVIDER=openai` and `OPENAI_API_KEY`.

## Evidence checks

```bash
make validate-sources
make validate-final-evaluation-dataset
make validate-final-benchmark
make validate-failure-analysis
make validate-pipeline-registry
make validate-trace-store
make eval-gate
```

Import and verify the checked-in retrieval runs in local MLflow:

```bash
make log-retrieval-runs
make verify-retrieval-runs
```

## CI-equivalent checks

```bash
make lint
make test-unit-ci PYTHON=.venv/bin/python
make test-api-ci PYTHON=.venv/bin/python
make test-evaluation-smoke
make eval-gate
```

## Common failures

| Error | Fix |
| --- | --- |
| Raw corpus missing | `make fetch-sources` |
| Qdrant collection missing | Start Qdrant, then `make build-index` |
| BM25 artifact missing | `make build-bm25-index` |
| Placeholder answer | Configure OpenAI or Gemini and recreate the API |
| Dashboard shows no traces | Run `/retrieve` or `/query`; `/route` is not traced |
| Port already used | Change the published port and its matching URL in `.env` |
