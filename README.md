# RAGOps Control Plane

## Evaluation-Gated, Cost-Aware RAG Platform

RAGOps Control Plane is a production-style RAG project built around technical documentation from FastAPI, MLflow, and Qdrant. The current system ingests and chunks documents, stores dense embeddings in Qdrant, retrieves relevant context, builds citations, and exposes the full path through FastAPI.

The long-term goal is to compare and safely promote RAG pipeline versions using measured quality, latency, and cost instead of guesswork.

## Current State

The implemented pipeline supports:

- local Markdown, MDX, RST, text, HTML, and selected Python source loading
- deterministic heading-aware and overlapping chunking with stable IDs and hashes
- batched `sentence-transformers/all-MiniLM-L6-v2` embeddings
- Qdrant indexing and dense retrieval
- ranked retrieved chunks with source metadata
- deduplicated numbered citations
- citation-required generation prompts
- `GET /health`, `POST /retrieve`, and `POST /query`
- a basic Streamlit playground for queries, citations, evidence, and latency
- unit tests that do not require live Qdrant or an external model

`POST /query` currently uses a deterministic local template client. Retrieval and citation construction are real, but answer synthesis is still a placeholder for a future local or API-backed LLM.

## Quickstart

Create the environment and run the checks:

```bash
make setup
make lint
make test
```

Start the local services:

```bash
docker compose up -d qdrant mlflow
```

With the documentation corpus available in `data/raw`, inspect ingestion without writing embeddings:

```bash
.venv/bin/python scripts/ingest.py --dry-run
```

Generate embeddings and build the Qdrant index:

```bash
.venv/bin/python scripts/ingest.py
.venv/bin/python scripts/build_index.py --recreate
```

Run the API:

```bash
PYTHONPATH=src .venv/bin/uvicorn ragops.app:app --reload
```

In another terminal, run the dashboard:

```bash
PYTHONPATH=src .venv/bin/streamlit run dashboard/app.py
```

Open `http://localhost:8501` for the playground. The API documentation remains available at `http://127.0.0.1:8000/docs`.

You can also send a query directly:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I create a FastAPI app?","top_k":5}'
```

## Main Components

```text
data/raw -> cleaning -> chunking -> embeddings -> Qdrant
                                                   |
query -> dense retrieval -> citations -> generation -> FastAPI response
```

- `src/ragops/ingestion`: loading, cleaning, chunking, and embeddings
- `src/ragops/indexing`: Qdrant collection creation and indexing
- `src/ragops/retrieval`: dense retrieval and result normalization
- `src/ragops/generation`: citations, prompts, and generation client
- `src/ragops/app.py`: FastAPI endpoints and end-to-end request flow
- `dashboard/app.py`: Streamlit query playground
- `scripts`: ingestion and index-building commands
- `tests`: unit, API, and dashboard client tests

## Next Step

Days 1–12 are complete, and the Day 13 Streamlit playground is implemented. The next planned step is Week 2 stabilization: verify the browser workflow, fix integration issues, and keep the quickstart and architecture documentation aligned with the working system. Evaluation, hybrid retrieval, reranking, routing, caching, tracing, canary gates, and monitoring come later in the plan.
