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

Prerequisites: Python 3.12, Docker, and Docker Compose.

Create the environment and run the checks:

```bash
make setup
make lint
make test
```

Start Qdrant and MLflow:

```bash
make services-up
```

### Prepare the corpus and index

Raw documentation and generated embeddings are intentionally excluded from Git. A fresh checkout must place the source snapshots recorded in `data/manifests/source_manifest.json` at these locations:

```text
data/raw/fastapi/docs
data/raw/fastapi/docs_src
data/raw/mlflow/docs
data/raw/qdrant/qdrant_llms_full.txt
```

The manifest records the upstream URLs, selected paths, and exact FastAPI and MLflow commits. Once those files are present, inspect ingestion without writing anything:

```bash
make ingest-dry-run
```

Generate embeddings and build the Qdrant index:

```bash
make ingest
make index
```

`make ingest` writes `data/processed/chunks.jsonl` and may download the embedding model on its first run. Use `make index-recreate` only when you intentionally want to delete and rebuild the existing `rag_chunks` collection.

### Run the application

Run the API:

```bash
make serve
```

In another terminal, run the dashboard:

```bash
make dashboard
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
- `docs/architecture.md`: current data flow, request flow, configuration, and limitations

## Next Step

Days 1–14 are complete at their current acceptance level. The next planned step is Day 15: design the first golden QA evaluation dataset. Hybrid retrieval, reranking, routing, caching, tracing, canary gates, and monitoring come later in the plan.
