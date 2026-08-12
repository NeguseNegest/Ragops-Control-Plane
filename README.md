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

RAGOps Control Plane is a work-in-progress platform for developing and evaluating Retrieval-Augmented Generation systems over technical documentation. The repository currently implements dense, BM25, RRF hybrid, and cross-encoder-reranked retrieval, strict four-way retrieval evaluation, LLM-as-judge evaluation, and measured benchmark reports through Day 27.

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

Implementation is complete through Day 27 of the project plan, including a measured hybrid-plus-cross-encoder benchmark. The current baseline includes:

- loaders for Markdown, MDX, RST, text, HTML, and selected Python files
- deterministic fixed, overlapping, and heading-aware chunking with UUID5 identifiers and SHA256 hashes
- batched `sentence-transformers/all-MiniLM-L6-v2` embeddings
- Qdrant indexing and cosine-similarity dense retrieval
- technical-text tokenization, a portable gzip-compressed BM25 index, and ranked sparse retrieval
- deterministic Reciprocal Rank Fusion over dense top-20 and BM25 top-20 candidates into a deduplicated hybrid top 10
- configurable cross-encoder reranking of a 25-chunk RRF candidate pool down to five results, with preserved fusion provenance and component latency
- ranked chunks, provenance metadata, and deduplicated citations
- selectable offline-template, OpenAI Responses API, and Gemini Interactions API generation clients
- `GET /health`, `POST /retrieve`, and `POST /query`
- Streamlit query interface with answers, citations, evidence, scores, and latency
- local and Docker Qdrant configuration through `QDRANT_URL`
- request validation, API error translation, and dashboard error handling
- an 80-row golden QA set, 100 reviewed synthetic candidates, and 45 verified retrieval labels
- deterministic retrieval metrics and a real dense-baseline evaluation CLI
- a provenance-checked BM25 evaluation CLI, paired per-question comparison, wording-cohort analysis, and reproducible JSON/CSV/Markdown reports
- a live hybrid evaluator with dense/BM25/fusion component timings, strict corpus and label parity checks, three-way paired outcomes, relevance-group analysis, and failure reporting
- a live cross-encoder evaluator with a common-depth four-way comparison, controlled pre-rerank ablation, cold/warm component latency, and explicit reranking regressions
- strict faithfulness and answer-relevance rubrics, query-type-aware refusal judging, and a manual spot-check workflow
- cross-provider OpenAI generation and Gemini judging for a deterministic 10-question Day 20 sample

Current limitations:

- Dense retrieval remains the only retriever connected to the online API. BM25, RRF hybrid, and hybrid-plus-reranker retrieval are available through offline CLIs but are not exposed through `POST /retrieve` and `POST /query`.
- Unweighted RRF improves substantially over dense retrieval on the current labels but does not beat BM25; it is an evaluated candidate, not the selected retrieval baseline.
- The cross-encoder is the strongest measured top-five pipeline on the current labels, but its warmed reranker stage averages about 4.27 seconds per query and is not suitable for the online path without latency optimization or selective routing.
- The default offline template client returns a fixed placeholder answer; OpenAI and Gemini generation are implemented but only one provider is selected per API process.
- Grounding and refusal are prompt instructions in the online path; the offline judge measures them but does not enforce or repair runtime answers.
- Generation evaluation is currently a 10-question LLM-as-judge acceptance sample, not a statistically robust benchmark. MLflow tracking, cost accounting, tracing, routing, caching, canary gates, failure mining, monitoring, and CI evaluation gates are not implemented.
- Raw corpora and generated embeddings are local artifacts and are not committed.

## Dense vs BM25 vs Hybrid vs Reranker Benchmark

Day 27 compares dense, BM25, Day 25 RRF, and hybrid plus cross-encoder retrieval over the same 45 verified questions. Because the reranker returns five results, every headline ranking is truncated to five and MRR is reported as MRR@5. The complete [four-way reranker report](reports/week4_reranker_comparison.md) includes the controlled RRF-25 ablation, latency tradeoff, and every gain and regression. The [three-way hybrid report](reports/week4_hybrid_comparison.md), [dense-vs-BM25 report](reports/week4_bm25_comparison.md), and [dense benchmark](reports/week3_dense_baseline.md) retain the earlier full-depth analyses.

| Metric | Dense | BM25 | RRF hybrid | Hybrid + reranker |
| --- | ---: | ---: | ---: | ---: |
| MRR@5 | 0.3163 | 0.6152 | 0.5641 | **0.6889** |
| Recall / Hit Rate@1 | 0.2667 | 0.4667 | 0.4667 | **0.5778** |
| Recall / Hit Rate@3 | 0.3111 | **0.7556** | 0.6444 | **0.7556** |
| Recall / Hit Rate@5 | 0.4444 | **0.8444** | 0.7556 | **0.8444** |
| nDCG@5 | 0.3473 | 0.6727 | 0.6112 | **0.7282** |

Reranking improves MRR@5 by 7.4 percentage points over BM25 and 12.5 points over the Day 25 RRF ranking. Against its own RRF-25 candidate order, it wins 16 question ranks, loses five, ties 24, recovers six top-five misses, and loses one prior hit; MRR@5 rises from `0.5644` to `0.6889`. It also matches BM25 Hit@5 while placing more labels first. The tradeoff is material: warmed end-to-end latency averages `4,476.4 ms`, including `4,274.9 ms` in the reranker, versus an in-run retrieval-plus-fusion estimate of about `333.7 ms` across the full run. Equal weighting over the 20 unique labeled chunks preserves the measured improvement, but source-derived single-chunk labels and an unpinned model revision limit generalization.

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
export OPENAI_API_KEY="api-key"
export OPENAI_MODEL=gpt-5-nano
```

To use Gemini instead, export:

```bash
export RAGOPS_LLM_PROVIDER=gemini
export GEMINI_API_KEY="api-key"
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

The recorded local Day 22 build indexed 13,476 searchable chunks, skipped five chunks containing no searchable tokens, and produced a 5.8 MB compressed artifact. A real `retrieve_bm25` sanity query ranked the labeled Qdrant dot-product evidence first. Run your own standalone sparse retrieval check while building or validating with `--query` and optional `--top-k`. `retrieve_bm25` is not yet connected to `POST /retrieve` or `POST /query`; Day 23 evaluates it offline against the same labels as the dense baseline.

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

### Evaluate BM25 and compare it with dense retrieval

Day 23 uses `configs/bm25_baseline.yaml` and the persisted Day 22 index. Preflight verifies the label count, dense-report coverage, index parameters, tokenizer, and source-chunk SHA256 without running any query:

```bash
make validate-bm25-evaluation
make test-bm25-evaluation
```

Run all 45 local sparse queries and regenerate the checked-in comparison artifacts:

```bash
make evaluate-bm25
```

The evaluator loads the index once, applies the same metric functions and cutoffs as the dense run, compares the first relevant rank for every question, and writes:

```text
reports/evaluations/bm25_baseline.json
reports/evaluations/bm25_baseline.csv
reports/evaluations/bm25_vs_dense.json
reports/week4_bm25_comparison.md
```

The BM25 JSON includes index provenance and complete rankings; the CSV is one row per question. The paired JSON stores aggregate deltas, wins, misses recovered by each retriever, deterministic wording cohorts, and per-question ranks. The Markdown report is rendered from that comparison JSON. The recorded local run produced MRR `0.6189`, Hit Rate@5 `0.8444`, and 27 BM25 wins versus 6 dense wins. Its average `87.9 ms` in-memory scoring time is not directly comparable with the dense `679.9 ms`, which includes embedding plus Qdrant search and a 24-second cold start.

### Run Reciprocal Rank Fusion hybrid retrieval

Day 24 is configured in `configs/hybrid.yaml`. It retrieves 20 dense candidates from Qdrant and 20 sparse candidates from the persisted BM25 index, then computes each unique chunk's score as the sum of `1 / (60 + rank)` across the rankings and returns the best 10.

Validate configuration, candidate depths, and BM25 index provenance without connecting to Qdrant:

```bash
make validate-hybrid
make test-hybrid
```

With the `rag_chunks` collection running, execute the real query CLI:

```bash
make retrieve-hybrid
make retrieve-hybrid HYBRID_QUERY="What is the exact MLflow serving command?"
```

The equivalent direct command supports readable or JSON output:

```bash
PYTHONPATH=src .venv/bin/python scripts/retrieve_hybrid.py \
  --config configs/hybrid.yaml \
  --query "What operation quantifies vector similarity?" \
  --json
```

Each fused result uses its RRF score and final rank while retaining the source chunk payload. Reserved `_fusion` metadata records the original dense and BM25 ranks, original scores, and individual RRF contributions. Duplicate IDs within one ranking, non-contiguous ranks, non-finite scores, and conflicting payloads for the same chunk ID fail explicitly. RRF avoids comparing incomparable cosine and BM25 score scales directly.

### Evaluate dense, BM25, and hybrid retrieval

Day 25 extends `configs/hybrid.yaml` with the verified labels, fixed baseline reports, and artifact paths. Preflight checks label coverage, metric cutoffs, component configurations, current BM25 SHA, and baseline provenance without querying Qdrant:

```bash
make validate-hybrid-evaluation
make test-hybrid-evaluation
```

With `rag_chunks` running, reproduce the live 45-question benchmark:

```bash
make evaluate-hybrid
```

Outputs:

```text
reports/evaluations/hybrid_rrf.json
reports/evaluations/hybrid_rrf.csv
reports/evaluations/hybrid_vs_baselines.json
reports/week4_hybrid_comparison.md
```

The live run verified 13,481 Qdrant points against 13,481 source records and the BM25 source SHA256. Hybrid MRR was `0.5765`, Hit Rate@5 was `0.7556`, and Hit Rate@10 was `0.8444`. Average end-to-end latency was `837.4 ms`, dominated by a `29,892.1 ms` first-query embedding warm-up; the remaining 44 queries averaged `177.1 ms`. Within the same hybrid run, dense retrieval averaged `772.5 ms`, BM25 `64.7 ms`, and fusion `0.2 ms`; component averages include the dense cold start.

### Run hybrid retrieval with cross-encoder reranking

Day 26 is configured in `configs/hybrid_rerank.yaml`. Dense top 25 and BM25 top 25 are fused into 25 unique candidates with RRF, then `cross-encoder/ms-marco-MiniLM-L-6-v2` jointly scores each query/chunk pair and returns the best five.

Validate the candidate depths and current BM25 index without connecting to Qdrant or loading the cross-encoder:

```bash
make validate-hybrid-rerank
make test-reranker
```

With `rag_chunks` running, execute the full pipeline:

```bash
make retrieve-hybrid-rerank
make retrieve-hybrid-rerank RERANK_QUERY="What is the exact MLflow serving command?"
```

The equivalent CLI supports structured output:

```bash
PYTHONPATH=src .venv/bin/python scripts/retrieve_hybrid_rerank.py \
  --config configs/hybrid_rerank.yaml \
  --query "What operation quantifies vector similarity?" \
  --json
```

The first execution may download and load the configured Hugging Face model. Readable and JSON output separate model-load time from dense, BM25, fusion, cross-encoder, and total pipeline latency. Final scores are raw cross-encoder relevance logits; `_reranker` metadata retains the RRF candidate rank and score, while `_fusion` retains dense and BM25 provenance. Day 26 establishes functional behavior; Day 27 below measures the fixed candidate.

The real Day 26 acceptance query used the live local collection and current 13,476-document BM25 index. The verified vector-operation chunk moved from RRF candidate rank 9 to final rank 2. This cold process measured `53,219.4 ms` to download/load the cross-encoder, `7,042.6 ms` for reranking, and `13,330.6 ms` for retrieval plus reranking; those one-query cold timings demonstrate observability, not representative steady-state performance.

### Evaluate hybrid plus cross-encoder reranking

Day 27 extends `configs/hybrid_rerank.yaml` with the fixed label set, dense/BM25/RRF reports, common `[1, 3, 5]` cutoffs, and output paths. Preflight validates all labels, component settings, report coverage, and the current BM25 source hash without connecting to Qdrant or loading the model:

```bash
make validate-reranker-evaluation
make test-reranker-evaluation
```

With `rag_chunks` running, reproduce the live benchmark:

```bash
make evaluate-reranker
```

Outputs:

```text
reports/evaluations/hybrid_rrf_cross_encoder.json
reports/evaluations/hybrid_rrf_cross_encoder.csv
reports/evaluations/reranker_vs_baselines.json
reports/week4_reranker_comparison.md
```

The evaluator loads one cross-encoder for all 45 questions, retains each full RRF-25 candidate order and final top five, validates `_fusion` and `_reranker` provenance, and records model-load plus dense/BM25/fusion/reranker latency. The comparison recomputes all four official pipelines at top five and adds an in-run RRF-25-before-reranking ablation. See the benchmark table above and the full report for the measured result and validity limits.

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

query -> dense top 20 --+
                         +-> RRF -> hybrid top 10 -> CLI
query -> BM25 top 20 ---+

query -> dense top 25 --+
                         +-> RRF top 25 -> cross-encoder -> top 5 -> CLI
query -> BM25 top 25 ---+
```

- `src/ragops/ingestion`: loading, cleaning, chunking, and embeddings
- `src/ragops/indexing`: Qdrant collection creation and indexing
- `src/ragops/retrieval`: dense retrieval, BM25 indexing/retrieval, deterministic RRF hybrid fusion, and shared result normalization
- `src/ragops/reranking`: validated cross-encoder model wrapper, candidate reranking, and the hybrid-plus-reranker pipeline
- `src/ragops/generation`: citations, grounded prompts, provider selection, and template/OpenAI/Gemini clients
- `src/ragops/evaluation`: synthetic QA handling, retrieval labels and metrics, dense/BM25/RRF/reranker evaluation and comparison, and LLM-as-judge orchestration
- `src/ragops/app.py`: FastAPI endpoints and end-to-end request flow
- `dashboard/app.py`: Streamlit query playground
- `scripts`: ingestion, indexing, dense/BM25/hybrid/reranked retrieval, dataset-review, labeling, and evaluation commands; later-milestone script files remain empty placeholders
- `tests`: unit, API, dashboard, dataset, retrieval/fusion/reranking, metric, evaluation-runner, and LLM-judge tests; later-milestone test files remain empty placeholders
- `docs/architecture.md`: current data flow, request flow, configuration, and limitations

## Next Milestone

Proceed to Day 28: refactor retrievers behind a common interface, consolidate RRF/reranking tests and configuration, and keep the complete suite green.
