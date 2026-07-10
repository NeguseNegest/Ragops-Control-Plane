# RAGOps Control Plane

This project will:

1. Ingest documentation from FastAPI, MLflow, and Qdrant.
2. Build a RAG pipeline that can search those docs and answer with citations.
3. Add evaluation, tracing, caching, and deployment checks around the pipeline.

That is the whole idea: build the RAG system, then build the tooling around it
so different versions can be compared instead of guessed at.

## Current State

So far the project has:

- a basic FastAPI app with a `/health` endpoint
- Docker Compose services for the API, Qdrant, and MLflow
- raw docs downloaded into `data/raw`
- a source manifest for the docs
- document schemas
- loaders and cleaners for docs/code files
- a dry-run ingestion command
- tests for the current ingestion behavior

Run the current ingestion check:

```bash
python scripts/ingest.py --dry-run
```

## Tools

- Python
- FastAPI
- Pydantic
- Docker Compose
- Qdrant
- MLflow
- sentence-transformers
- rank-bm25
- pytest
- ruff
- Streamlit later for the dashboard

## Quickstart

Set up the project:

```bash
make setup
make lint
make test
```

Start Qdrant and MLflow:

```bash
docker compose up qdrant mlflow
```

Run the API:

```bash
make docker-up
```

Check the API:

```bash
curl http://localhost:8000/health
```

## Layout

```text
.
├── configs/
├── data/
├── dashboard/
├── docs/
├── scripts/
├── src/ragops/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```
