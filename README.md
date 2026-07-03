# RAGOps Control Plane

RAGOps Control Plane is a Python project for building a production-style control
plane around Retrieval-Augmented Generation systems. The repository is intended
to contain the application code, evaluation tooling, local infrastructure, and
documentation needed to develop and compare RAG pipeline versions.

The planned system includes document ingestion, deterministic chunking, dense
retrieval with Qdrant, sparse BM25 retrieval, hybrid retrieval, reranking,
citation-grounded generation, offline evaluation, MLflow experiment tracking,
FastAPI serving, trace logging, query routing, semantic caching, canary
simulation, evaluation gates, failure mining, and a dashboard.

## Current Stage

The project is currently in the initial repository setup stage. It has a small
FastAPI package, a health endpoint, basic tests, Python project metadata, Docker
configuration, and Makefile commands. The core RAG, evaluation, tracing, routing,
cache, canary, and dashboard features have not been implemented yet.

## Quickstart

Requirements:

- Python 3.11, 3.12, or 3.13
- Docker Desktop or a compatible Docker engine

```bash
make setup
make lint
make test
```

Run the local API container:

```bash
make docker-up
```

Check the health endpoint:

```bash
curl http://localhost:8000/health
```

## Project Layout

```text
.
├── src/ragops/                 # Application package
├── tests/                      # Pytest suite
├── Dockerfile                  # Local API image
├── docker-compose.yml          # Local service launcher
├── Makefile                    # Setup, lint, test, Docker, and cleanup targets
├── pyproject.toml              # Python metadata and dependencies
└── README.md                   # Project overview
```
