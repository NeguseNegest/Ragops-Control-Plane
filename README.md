# RAGOps Control Plane

## Evaluation-Gated, Cost-Aware RAG Platform

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-schemas-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector_search-DC244C?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-embeddings-FFD21E)](https://www.sbert.net/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![pytest](https://img.shields.io/badge/pytest-tested-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)
[![Ruff](https://img.shields.io/badge/Ruff-linted-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

RAGOps Control Plane is an evaluation-gated, cost-aware platform for developing and operating Retrieval-Augmented Generation systems over technical documentation.

## Project Objective

The project evaluates and compares versioned RAG pipelines across retrieval quality, generation quality, latency, and estimated cost. Candidate pipelines are promoted or rejected through explicit evaluation and canary gates.

Target capabilities:

- deterministic ingestion, chunking, embedding, and index versioning
- dense, BM25, hybrid, and cross-encoder-reranked retrieval
- citation-grounded generation with unsupported-query refusal
- golden, adversarial, and failure-mined evaluation datasets
- retrieval, generation, latency, and cost metrics tracked in MLflow
- FastAPI serving with SQLite traces and component-level timings
- rule-based query routing and corpus-aware semantic caching
- production-versus-candidate canary simulation and automated promotion gates
- failure mining, operational monitoring, Streamlit analytics, and CI evaluation checks

The primary outputs are reproducible pipeline comparisons and promotion decisions supported by measurable quality, latency, and cost constraints.

## Current Implementation

Development is complete through Day 14 of the project plan. The current baseline includes:

- loaders for Markdown, MDX, RST, text, HTML, and selected Python files
- deterministic fixed, overlapping, and heading-aware chunking with UUID5 identifiers and SHA256 hashes
- batched `sentence-transformers/all-MiniLM-L6-v2` embeddings
- Qdrant indexing and cosine-similarity dense retrieval
- ranked chunks, provenance metadata, and deduplicated citations
- selectable template, OpenAI, and Gemini generation clients
- `GET /health`, `POST /retrieve`, and `POST /query`
- Streamlit query interface with answers, citations, evidence, scores, and latency
- local and Docker Qdrant configuration through `QDRANT_URL`
- request validation, API error translation, and dashboard error handling
- 65 passing tests and a verified Streamlit–FastAPI–Qdrant integration path

Current limitations:

- `POST /query` defaults to the deterministic template client until `RAGOPS_LLM_PROVIDER` is set to `openai` or `gemini` and the corresponding API key is configured.
- Only dense retrieval is implemented.
- Evaluation, MLflow tracking, tracing, routing, caching, canary gates, failure mining, monitoring, and CI evaluation gates are not implemented.
- Raw corpora and generated embeddings are local artifacts and are not committed.

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

### Choose an answer-generation provider

The default `template` provider is deterministic, offline, and does not require an API key. To use OpenAI instead, export:

```bash
export RAGOPS_LLM_PROVIDER=openai
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL=gpt-5-nano
```

To use Gemini, export:

```bash
export RAGOPS_LLM_PROVIDER=gemini
export GEMINI_API_KEY="your-api-key"
export GEMINI_MODEL=gemini-3.6-flash
```

Keep real API keys out of the repository. The ignored `.env` file can hold local values, but `make serve` requires those values to be exported into the shell environment first. Docker Compose reads `.env` for variable substitution and passes the selected provider configuration to the API container.

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

### Generate and review synthetic QA candidates

Day 16 uses both configured LLM providers to create 100 source-grounded candidates without placing unreviewed output in the golden set:

```bash
make generate-synthetic-qa
```

The command reads API keys from the ignored `.env` file, allocates candidates evenly between OpenAI and Gemini, and writes `data/eval/synthetic_qa_candidates.jsonl`. Each row records its provider, model, source chunk ID, and `pending` review status.

Review candidates against their exact source chunks and merge approved examples interactively:

```bash
make review-synthetic-qa
```

The review command accepts `a` to approve, `r` to reject, `s` to leave a candidate pending, and `q` to save and quit. It stops at 40 approvals by default and only adds approved, non-duplicate examples to `data/eval/golden_qa.jsonl`.

The checked-in Day 16 run contains 100 reviewed candidates: 45 approved and 55 rejected. The approved set includes 25 OpenAI and 20 Gemini examples, and expands the golden dataset from 35 to 80 rows.

To intentionally regenerate the candidate file, pass `--overwrite` directly:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_synthetic_qa.py --overwrite
```

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

## Next Milestone

Day 17: create retrieval labels for at least 40 questions, including relevant chunk IDs and a helper workflow for inspecting chunks during labeling.
