# RAGOps Control Plane

## Evaluation-Gated, Cost-Aware RAG Platform

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-2.8%2B-E92063?logo=pydantic&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-499848)
![Qdrant](https://img.shields.io/badge/Qdrant-vector_db-DC244C)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-traces-003B57?logo=sqlite&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-embeddings-FCC624)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.41%2B-FFD21E)
![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243?logo=numpy&logoColor=white)
![rank-bm25](https://img.shields.io/badge/rank--bm25-sparse_retrieval-6B7280)
![pytest](https://img.shields.io/badge/pytest-tests-0A9EDC?logo=pytest&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-linting-D7FF64)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI-2088FF?logo=githubactions&logoColor=white)

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

- Python for the core application and pipeline code
- FastAPI, Pydantic, and Uvicorn for the API layer
- Docker Compose for local services
- Qdrant for dense vector indexing and search
- MLflow for experiment tracking
- SQLite for trace storage
- sentence-transformers, PyTorch, Transformers, and NumPy for embeddings and model utilities
- rank-bm25 for sparse retrieval
- Streamlit for the dashboard
- pytest and Ruff for testing and linting
- GitHub Actions for CI later in the project

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
