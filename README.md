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

Development is complete through Day 19 of the project plan. The current baseline includes:

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
- reviewed golden QA and retrieval-label datasets
- deterministic retrieval metrics and a real dense-baseline evaluation CLI
- 157 passing tests and a verified Streamlit–FastAPI–Qdrant integration path

Current limitations:

- `POST /query` defaults to the deterministic template client until `RAGOPS_LLM_PROVIDER` is set to `openai` or `gemini` and the corresponding API key is configured.
- Only dense retrieval is implemented.
- Generation evaluation, MLflow tracking, tracing, routing, caching, canary gates, failure mining, monitoring, and CI evaluation gates are not implemented.
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

### Build and inspect retrieval labels

Day 17 stores retrieval-specific relevance judgments in `data/eval/retrieval_labels.jsonl`. Every label links a supported golden question to one or more verified chunk IDs and records how the decision was reviewed.

The checked-in dataset contains 45 labels bootstrapped from Day 16 candidates whose source chunks were already audited. The bootstrap validates the question, expected source, and exact chunk against both JSONL datasets before accepting it:

```bash
make bootstrap-retrieval-labels
make validate-retrieval-labels
```

Use the offline, resumable inspector to add more labels manually:

```bash
make label-retrieval
```

The helper ranks chunks only from the question's expected source, shows their text and IDs, and accepts one or more display numbers or exact chunk IDs. It saves after every decision and does not require Qdrant or an LLM API.

### Compute deterministic retrieval metrics

Day 18 provides pure metric functions in `src/ragops/evaluation/retrieval_metrics.py`:

- Recall@k: unique relevant chunks retrieved within the cutoff, divided by all labeled relevant chunks.
- MRR: the mean reciprocal rank of the first relevant chunk.
- Hit Rate@k: the fraction of questions with at least one relevant chunk within the cutoff.
- nDCG@k: binary normalized discounted gain; repeated retrieved IDs cannot earn duplicate gain.

The aggregate evaluator accepts a mapping of question IDs to ranked chunk IDs plus the Day 17 labels. Missing rankings and invalid cutoffs fail explicitly. Run its focused tests with:

```bash
make test-retrieval-metrics
```

Day 18 intentionally does not run retrieval itself or write result files; that orchestration belongs to the Day 19 evaluation CLI.

### Run the dense retrieval evaluation

Day 19 defines the baseline in `configs/dense_baseline.yaml` and implements the evaluator in `scripts/evaluate.py`. Validate configuration and all 45 input labels without connecting to Qdrant:

```bash
make validate-dense-evaluation
```

With an indexed `rag_chunks` collection already available, run the real dense retriever over every label:

```bash
make evaluate-dense
```

The evaluator creates one Qdrant client, checks the configured collection, retrieves the top 10 chunks for every question, computes the Day 18 metrics, and closes the client even if retrieval fails. It writes these artifacts atomically only after the complete run succeeds:

```text
reports/evaluations/dense_baseline.json
reports/evaluations/dense_baseline.csv
```

The JSON file contains configuration, aggregate metrics, latency statistics, and per-question results. The CSV contains one row per question with ranked IDs, scores, latency, per-question metrics, and aggregate Recall@k, MRR, Hit Rate@k, and nDCG@k. This evaluation does not call OpenAI or Gemini.

The Day 19 acceptance run evaluated all 45 labels against the real 13,481-chunk Qdrant index. It produced MRR `0.3359`, Recall/Hit Rate at k of `0.2667`, `0.3111`, `0.4444`, and `0.6000` for k = 1, 3, 5, and 10, and nDCG at k of `0.2667`, `0.2918`, `0.3473`, and `0.3964`. Average retrieval latency was `679.9 ms` including the first-query model cold start; the remaining 44 queries averaged `149.6 ms`.

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

Proceed to Day 20: the LLM-as-judge rubric for faithfulness, answer relevance, and refusal correctness.
