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

RAGOps Control Plane is a work-in-progress platform for developing and evaluating Retrieval-Augmented Generation systems over technical documentation. The repository currently implements dense and BM25 retrieval foundations, dense and LLM-as-judge evaluation workflows, and the first measured benchmark report through Day 22; cost controls and promotion gates remain roadmap features.

## Project Objective

The intended system will compare versioned RAG pipelines across retrieval quality, generation quality, latency, and estimated cost, then promote or reject candidates through explicit evaluation and canary gates.

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

Implementation is complete through Day 22 of the project plan, including the first persisted sparse index. The current baseline includes:

- loaders for Markdown, MDX, RST, text, HTML, and selected Python files
- deterministic fixed, overlapping, and heading-aware chunking with UUID5 identifiers and SHA256 hashes
- batched `sentence-transformers/all-MiniLM-L6-v2` embeddings
- Qdrant indexing and cosine-similarity dense retrieval
- technical-text tokenization, a portable gzip-compressed BM25 index, and ranked sparse retrieval
- ranked chunks, provenance metadata, and deduplicated citations
- selectable offline-template, OpenAI Responses API, and Gemini Interactions API generation clients
- `GET /health`, `POST /retrieve`, and `POST /query`
- Streamlit query interface with answers, citations, evidence, scores, and latency
- local and Docker Qdrant configuration through `QDRANT_URL`
- request validation, API error translation, and dashboard error handling
- an 80-row golden QA set, 100 reviewed synthetic candidates, and 45 verified retrieval labels
- deterministic retrieval metrics and a real dense-baseline evaluation CLI
- strict faithfulness and answer-relevance rubrics, query-type-aware refusal judging, and a manual spot-check workflow
- cross-provider OpenAI generation and Gemini judging for a deterministic 10-question Day 20 sample

Current limitations:

- Dense retrieval remains the only retriever connected to the API and evaluation runner. BM25 is implemented as a standalone offline retriever but is not evaluated until Day 23.
- The default offline template client returns a fixed placeholder answer; OpenAI and Gemini generation are implemented but only one provider is selected per API process.
- Grounding and refusal are prompt instructions in the online path; the offline judge measures them but does not enforce or repair runtime answers.
- Generation evaluation is currently a 10-question LLM-as-judge acceptance sample, not a statistically robust benchmark. MLflow tracking, cost accounting, tracing, routing, caching, reranking, canary gates, failure mining, monitoring, and CI evaluation gates are not implemented.
- Raw corpora and generated embeddings are local artifacts and are not committed.

## First Measured Baseline

The recorded dense baseline evaluated 45 verified questions against 13,481 indexed chunks. The full [Week 3 benchmark report](reports/week3_dense_baseline.md) documents the setup, corpus-level results, latency caveats, and concrete failure examples.

| Cutoff | Recall@k | Hit Rate@k | nDCG@k |
| ---: | ---: | ---: | ---: |
| 1 | 0.2667 | 0.2667 | 0.2667 |
| 3 | 0.3111 | 0.3111 | 0.2918 |
| 5 | 0.4444 | 0.4444 | 0.3473 |
| 10 | 0.6000 | 0.6000 | 0.3964 |

MRR was `0.3359`. Mean latency was `679.9 ms` including a `24,009.5 ms` first-query embedding-model cold start; the remaining 44 queries averaged `149.6 ms`. Because every question currently has one labeled relevant chunk, Recall@k and Hit Rate@k are equal in this run.

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

`POST /query` uses one generation client selected when the API process starts. The default `template` provider is deterministic and offline, but it returns a fixed, cited placeholder rather than synthesizing an answer from the retrieved text.

To use OpenAI instead, export:

```bash
export RAGOPS_LLM_PROVIDER=openai
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL=gpt-5-nano
```

To use Gemini instead, export:

```bash
export RAGOPS_LLM_PROVIDER=gemini
export GEMINI_API_KEY="your-api-key"
export GEMINI_MODEL=gemini-3.6-flash
```

Both API keys may be present, but `RAGOPS_LLM_PROVIDER` selects exactly one runtime provider: `template`, `openai`, or `gemini`. Restart the API after changing it. Keep real API keys out of the repository. The ignored `.env` file can hold local values, but `make serve` does not load that file automatically; export the values into the shell first. Docker Compose reads `.env` for variable substitution and passes the selected provider configuration to the API container.

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

### Build the BM25 index

Day 22 builds sparse retrieval from the same processed chunks without loading or persisting their embedding vectors. Validate the strict configuration, build the ignored local artifact, and verify that its source hash and scoring parameters still match:

```bash
make validate-bm25-config
make build-bm25-index
make validate-bm25-index
```

The resulting `data/processed/bm25_index.json.gz` contains schema-versioned chunk payloads, technical tokens, BM25 parameters, and the SHA256 of `chunks.jsonl`. The build refuses to replace an existing index. To intentionally rebuild it, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_bm25_index.py --overwrite
```

The recorded local Day 22 build indexed 13,476 searchable chunks, skipped five chunks containing no searchable tokens, and produced a 5.8 MB compressed artifact. A real `retrieve_bm25` sanity query ranked the labeled Qdrant dot-product evidence first. Run your own standalone sparse retrieval check while building or validating with `--query` and optional `--top-k`. `retrieve_bm25` is not yet connected to `POST /retrieve` or `POST /query`; the Day 23 milestone evaluates it against the same labels as the dense baseline.

### Generate and review synthetic QA candidates

Day 16 explicitly instantiates both configured LLM providers to create 100 source-grounded candidates without placing unreviewed output in the golden set. This multi-provider batch workflow is independent of the single `RAGOPS_LLM_PROVIDER` used by `POST /query`:

```bash
make generate-synthetic-qa
```

The command reads API keys from the ignored `.env` file, allocates candidates evenly between OpenAI and Gemini, and writes `data/eval/synthetic_qa_candidates.jsonl`. Each row records its provider, model, source chunk ID, and `pending` review status.

Review candidates against their exact source chunks and merge approved examples interactively:

```bash
make review-synthetic-qa
```

The review command accepts `a` to approve, `r` to reject, `s` to leave a candidate pending, and `q` to save and quit. It stops at 40 approvals by default and only adds approved, non-duplicate examples to `data/eval/golden_qa.jsonl`.

The checked-in Day 16 run contains 100 reviewed candidates: 45 approved and 55 rejected. The approved set includes 25 OpenAI and 20 Gemini examples, and expands the golden dataset from 35 to 80 rows. The resulting golden set contains 70 supported, 5 ambiguous, and 5 unsupported questions; those categories are dataset annotations and are not yet used by the runtime to route or refuse requests.

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

The recorded Day 19 acceptance run evaluated all 45 labels against a real 13,481-chunk Qdrant index. It produced MRR `0.3359`, Recall/Hit Rate at k of `0.2667`, `0.3111`, `0.4444`, and `0.6000` for k = 1, 3, 5, and 10, and nDCG at k of `0.2667`, `0.2918`, `0.3473`, and `0.3964`. Average retrieval latency was `679.9 ms` including a roughly 24-second first-query embedding-model cold start; the remaining 44 queries averaged `149.6 ms`. Latency values are measurements from that local run, not service-level guarantees.

### Generate and judge the Day 20 acceptance sample

Day 20 is configured in `configs/generation_judge.yaml`. It deterministically selects six supported, two ambiguous, and two unsupported golden questions. By default, `gpt-5-nano` generates answers from five retrieved chunks and `gemini-3.6-flash` independently judges them. The providers must differ unless the config explicitly disables that guard.

Validate the config and sample allocation without Qdrant or paid API calls:

```bash
make validate-generation-judge
make test-llm-judge
```

With Qdrant indexed and both keys present in the ignored `.env`, run the real acceptance evaluation:

```bash
make judge-answers
```

The command refuses to overwrite an existing run. Use `PYTHONPATH=src .venv/bin/python scripts/judge_answers.py --overwrite` only when intentionally replacing the artifacts. A successful run writes:

```text
reports/evaluations/day20_generation_judge_judgments.jsonl
reports/evaluations/day20_generation_judge_summary.json
```

Each JSONL record retains the question type, expected answer and behavior, exact retrieved chunk text and scores, generated answer, citations, generator and judge models, rubric scores and rationales, component timings, and manual-review status. The judge treats all supplied text as untrusted data, uses retrieved chunks—not the reference answer—as faithfulness evidence, requires strict JSON, and rejects semantically inconsistent refusal verdicts.

Manually compare every automatic judgment with the documented rubric and evidence:

```bash
make review-judgments
make validate-day20
```

The reviewer records `agree` or `disagree`; disagreement requires notes. `make validate-day20` passes only after all 10 configured spot-checks are complete. See `docs/evaluation.md` for the full rubric and review rules.

The recorded Day 20 run used OpenAI `gpt-5-nano` for generation and Gemini `gemini-3.6-flash` for judging. Its 10 answers received mean faithfulness `4.5/5` and mean answer relevance `3.4/5`; refusal verdicts were 2 correct, 4 incorrect, and 4 not applicable. A separate Codex evidence audit reviewed all 10 records, agreed with 8 judgments, and documented 2 relevance-score disagreements. This reviewer identity is stored as `codex-manual-audit` and must not be interpreted as human sign-off.

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
data/raw -> cleaning -> chunking -> chunks.jsonl -> embeddings -> Qdrant
                                          |
                                          +-> BM25 tokens -> persisted BM25 index

query -> dense retrieval -> citations -> generation -> FastAPI response
```

- `src/ragops/ingestion`: loading, cleaning, chunking, and embeddings
- `src/ragops/indexing`: Qdrant collection creation and indexing
- `src/ragops/retrieval`: dense retrieval, BM25 indexing/retrieval, and shared result normalization
- `src/ragops/generation`: citations, grounded prompts, provider selection, and template/OpenAI/Gemini clients
- `src/ragops/evaluation`: synthetic QA handling, retrieval labels, retrieval metrics, dense evaluation, and LLM-as-judge orchestration
- `src/ragops/app.py`: FastAPI endpoints and end-to-end request flow
- `dashboard/app.py`: Streamlit query playground
- `scripts`: ingestion, indexing, dataset-review, labeling, and evaluation commands; later-milestone script files are still empty placeholders
- `tests`: unit, API, dashboard, dataset, metric, evaluation-runner, and LLM-judge tests; later-milestone test files are still empty placeholders
- `docs/architecture.md`: current data flow, request flow, configuration, and limitations

## Next Milestone

Proceed to Day 23: evaluate the BM25 baseline on the 45 verified labels and compare it with the Week 3 dense baseline.
