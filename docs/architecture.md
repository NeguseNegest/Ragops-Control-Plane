# Architecture

## Current Scope

RAGOps Control Plane currently provides a dense-retrieval RAG path plus offline BM25, RRF hybrid, and cross-encoder-reranked retrievers over local FastAPI, MLflow, and Qdrant documentation. It has four implemented workflows:

- An offline workflow that cleans and chunks documentation, then builds both a dense Qdrant index and a portable BM25 index.
- An online workflow that retrieves chunks, builds citations, calls the selected template, OpenAI, or Gemini generation client, and exposes the result through FastAPI and Streamlit.
- An offline evaluation workflow that generates and reviews QA data, validates retrieval relevance labels, compares dense, persisted BM25, and live RRF hybrid rankings, and applies cross-provider LLM judging to generated answers.
- Offline hybrid and reranked CLIs that retrieve independent dense and BM25 candidate pools, fuse ranks without normalizing incompatible raw scores, and optionally apply a cross-encoder.
- A deterministic pipeline-registry workflow that binds versioned configs to validated evaluation evidence and guarded baseline/candidate/production aliases.

Dense, BM25, RRF hybrid, and cross-encoder retrieval evaluation, the Day 20 LLM-as-judge acceptance workflow, the Day 21 benchmark report, the Day 28 common-interface refactor, Day 29 MLflow retrieval tracking, and the Day 30 pipeline registry are implemented. Routing, caching, tracing, canary gates, failure mining, monitoring, and generation cost accounting remain planned.

## System Diagram

```mermaid
flowchart LR
    subgraph Offline[Offline ingestion and indexing]
        Raw["data/raw\nFastAPI, MLflow, Qdrant"] --> Load[Load and clean]
        Load --> Chunk[Deterministic chunking]
        Chunk --> Embed[MiniLM embeddings]
        Embed --> JSONL["data/processed/chunks.jsonl"]
        JSONL --> Index[Qdrant index builder]
        Index --> Qdrant[(Qdrant rag_chunks)]
        JSONL --> BM25Build[Technical tokenizer and BM25 builder]
        BM25Build --> BM25Index[(bm25_index.json.gz)]
    end

    subgraph Online[Online query path]
        User[Browser user] --> Streamlit[Streamlit :8501]
        Streamlit -->|"POST /query"| API[FastAPI :8000]
        API --> QueryEmbed[Embed query]
        QueryEmbed -->|Cosine search| Qdrant
        Qdrant --> Retrieved[Ranked chunks]
        Retrieved --> Citations[Citations and prompt]
        Citations --> Generator["Configured generator\ntemplate / OpenAI / Gemini"]
        Generator --> API
        API -->|JSON response| Streamlit
    end

    subgraph Hybrid[Offline hybrid CLI]
        HybridConfig[hybrid.yaml] --> HybridCLI[Hybrid query CLI]
        HybridCLI -->|"dense top 20"| Qdrant
        HybridCLI -->|"BM25 top 20"| BM25Index
        Qdrant --> RRF[Reciprocal Rank Fusion]
        BM25Index --> RRF
        RRF --> HybridTop10[Deduplicated top 10]
    end

    subgraph Evaluation[Offline retrieval evaluation]
        Golden[golden_qa.jsonl] --> Labels[retrieval_labels.jsonl]
        DenseConfig[dense_baseline.yaml] --> DenseRunner[Dense runner]
        BM25Config[bm25_baseline.yaml] --> BM25Runner[BM25 runner]
        HybridConfig --> HybridRunner[Hybrid evaluator]
        Labels --> DenseRunner
        Labels --> BM25Runner
        Labels --> HybridRunner
        DenseRunner -->|"45 dense queries"| Qdrant
        BM25Index --> BM25Runner
        Qdrant --> HybridRunner
        BM25Index --> HybridRunner
        DenseRunner --> Metrics["Recall / MRR / Hit Rate / nDCG"]
        BM25Runner --> Metrics
        HybridRunner --> Metrics
        Metrics --> Reports["JSON + CSV reports"]
        Reports --> Paired["Three-way ranks + cohorts + failures"]
        Paired --> Comparison["Comparison JSON + Markdown"]
        Reports --> MLflow[(MLflow retrieval experiment)]
        Comparison --> MLflow
    end

    subgraph Registry[Pipeline registry]
        VersionedConfigs["Versioned retrieval YAMLs"] --> RegistryBuilder[Registry builder and validator]
        Comparison --> RegistryBuilder
        MLflowCatalog["configs/mlflow.yaml"] --> RegistryBuilder
        RegistryBuilder --> RegistryJSON["reports/pipeline_registry.json"]
        RegistryJSON --> Aliases["baseline / candidate / production"]
    end

    subgraph GenerationEvaluation[Offline generation evaluation]
        JudgeConfig[generation_judge.yaml] --> Sample["10-question stratified sample"]
        Golden --> Sample
        Sample -->|"retrieve top 5"| Qdrant
        Qdrant --> EvalGenerator[OpenAI generator]
        EvalGenerator --> EvalJudge[Gemini judge]
        EvalJudge --> JudgeReports["Judgments JSONL + summary JSON"]
        JudgeReports --> SpotCheck[Manual spot-check]
    end
```

## Component Responsibilities

| Component | Location | Responsibility |
| --- | --- | --- |
| Loaders and cleaners | `src/ragops/ingestion` | Read supported local files, normalize text, and retain source provenance. |
| Chunker | `src/ragops/ingestion/chunking.py` | Create deterministic fixed, overlapping, or heading-aware chunks with UUID5 IDs and SHA256 hashes. |
| Embedder | `src/ragops/ingestion/embeddings.py` | Generate batched `sentence-transformers/all-MiniLM-L6-v2` vectors and cache the model in-process. |
| Indexer | `src/ragops/indexing/qdrant.py` | Create the `rag_chunks` collection and upsert embedded chunk records with payload metadata. |
| Retriever contract and factory | `src/ragops/retrieval/base.py`, `factory.py` | Provide the shared `retrieve(query, top_k, timings)` interface, validate the configured interface version, and construct any retrieval pipeline from its validated config and runtime resources. |
| Dense retriever | `src/ragops/retrieval/dense.py` | Embed a query, search Qdrant, and normalize ranked results into `RetrievedChunk` objects. |
| BM25 retriever | `src/ragops/retrieval/bm25.py`, `scripts/build_bm25_index.py` | Tokenize technical text, persist a versioned sparse index, validate source provenance, and return ranked `RetrievedChunk` objects without Qdrant. |
| Hybrid retriever | `src/ragops/retrieval/hybrid.py`, `scripts/retrieve_hybrid.py` | Retrieve independently ranked dense and BM25 candidate pools, validate their identities and ranks, fuse them with deterministic RRF, and expose readable or JSON CLI results. |
| Reranked retriever | `src/ragops/reranking/cross_encoder.py`, `scripts/retrieve_hybrid_rerank.py` | Compose a cross-encoder over the configured RRF candidate retriever while retaining candidate order, provenance, and component timings. |
| Citations and generation | `src/ragops/generation` | Deduplicate sources, assign citation IDs, build a context-only prompt, select one process-wide provider, and call the template, OpenAI, or Gemini client. |
| Evaluation datasets | `src/ragops/evaluation/synthetic_qa.py`, `retrieval_labels.py` | Generate, validate, review, and merge synthetic QA candidates; build and cross-validate retrieval relevance labels. |
| Retrieval metrics | `src/ragops/evaluation/retrieval_metrics.py` | Compute per-question and macro-average Recall@k, reciprocal rank/MRR, Hit Rate@k, and binary nDCG@k. |
| Evaluation runners | `src/ragops/evaluation/runner.py`, `bm25_runner.py`, `hybrid_runner.py`, `reranker_runner.py` | Build config-driven dense, BM25, hybrid, or reranked pipelines; write complete JSON/CSV runs; and produce paired metrics, latency, win/loss, cohort, relevance-group, and failure comparisons. |
| Experiment tracker | `src/ragops/tracking/mlflow.py`, `scripts/log_retrieval_runs.py` | Validate retrieval evidence, flatten configs and metrics, log or import idempotent MLflow runs, upload CSV/JSON/YAML/Markdown artifacts, and verify the four-run acceptance state. |
| Pipeline registry | `src/ragops/pipeline_registry.py`, `scripts/build_pipeline_registry.py` | Validate semantic versions and lifecycle status, bind configs to common-depth evidence and MLflow identity, compute checksums, enforce alias policy, and atomically generate the registry snapshot. |
| LLM judge | `src/ragops/evaluation/llm_judge.py`, `scripts/judge_answers.py` | Select a deterministic query-type mix, retrieve and generate answers, apply strict faithfulness/relevance/refusal rubrics, and persist evidence-rich judgments. |
| Judgment reviewer | `scripts/review_judgments.py` | Display each question, answer, evidence, and automatic rationale; atomically record reviewer agreement or disagreement. |
| API | `src/ragops/app.py` | Expose health, retrieval, and query endpoints; translate errors; and close Qdrant clients. |
| Dashboard | `dashboard/app.py` | Call `POST /query` over HTTP and display the answer, citations, chunks, scores, and latency. |

## Offline Data Flow

1. `scripts/ingest.py` walks `data/raw` in stable path order.
2. Supported documentation and selected FastAPI example files are cleaned into `Document` records.
3. Documents are split into deterministic `DocumentChunk` records. The default strategy is heading-aware chunking with 250 whitespace tokens and 50 tokens of overlap.
4. Chunk text is embedded with `sentence-transformers/all-MiniLM-L6-v2`.
5. Embedded records are written as JSONL to `data/processed/chunks.jsonl`. Each record contains the chunk text, IDs, hash, metadata, and vector.
6. `scripts/build_index.py` reads the JSONL file, creates the Qdrant `rag_chunks` collection when needed, and upserts records in batches.
7. Independently, `scripts/build_bm25_index.py` drops embeddings, adds normalized prose and exact technical tokens, and atomically writes `data/processed/bm25_index.json.gz` with the input SHA256 and BM25 parameters.

## Common Retrieval Interface

Day 28 moves runtime composition behind `Retriever.retrieve(query, top_k=None, timings=None)`. `DenseRetriever` owns a Qdrant client and embedding settings, `BM25Retriever` owns a loaded sparse index, `HybridRetriever` composes both candidate retrievers before RRF, and `CrossEncoderRerankedRetriever` composes over the hybrid candidate pool. The config factory selects one of these four pipelines from the validated model; it does not infer candidate depths or model names from call-site defaults.

The four checked-in retrieval configs explicitly declare `retriever_interface: common_v1`, a semantic `version`, and a lifecycle `status`. Their existing component sections remain authoritative for collection, index, model, RRF, candidate-depth, final-depth, and evaluation settings. An unknown interface or malformed semantic version is rejected during Pydantic validation. Legacy retrieval functions delegate to or adapt the same objects so CLI, API, evaluator, and test injection call sites remain compatible during the refactor.

## Pipeline Registry and Promotion Boundary

Day 30 generates `reports/pipeline_registry.json` from the versioned retrieval YAMLs, the Day 29 artifact catalog, and the Day 27 common top-five comparison. Registry generation repeats the source-evidence validation, computes each config SHA256, records the evidence digest and comparable quality/latency summary, and rejects a checked-in artifact that has drifted from any source.

Aliases are validated pointers to exact `name@version` identities. `baseline` points to approved BM25, `candidate` points to the evaluated cross-encoder pipeline, and `production` points to the approved dense config used by the current API algorithm. The negative unweighted-RRF result remains registered as rejected without an alias. Draft, rejected, retired, missing, and stale entries cannot be selected; baseline and production require approved status.

This is a control-plane boundary, not runtime deployment. Moving the `production` alias records a reviewed promotion decision but does not make FastAPI load a different retriever. Deployment wiring, evaluation gates, and canary automation remain later milestones. Detailed version, promotion, and rollback rules are in `docs/pipeline_registry.md`.

Day 24 combines the existing indexes at query time:

```text
                         query
                           |
              +------------+------------+
              |                         |
       dense top 20                 BM25 top 20
              |                         |
              +------------+------------+
                           |
             score(chunk) = sum(1 / (60 + rank))
                           |
                  deduplicated top 10
```

`configs/hybrid.yaml` pins both candidate depths, their index/model settings, the RRF constant, and final depth. Fusion uses list positions rather than raw cosine or BM25 scores, which are not on a shared scale. A chunk returned by both retrievers receives both reciprocal-rank contributions. The final `RetrievedChunk.score` is the fused score; `_fusion` metadata retains each original rank, score, and contribution. Equal scores are resolved by number of contributing rankings, best source rank, then chunk ID. Input rankings must be contiguous, unique, finite, and payload-consistent for matching IDs.

## Evaluation Dataset Flow

```text
processed chunks -> balanced source sampling -> OpenAI + Gemini
                                              |
                                              v
                              synthetic_qa_candidates.jsonl
                                              |
                                     manual source review
                                              |
                                              v
                                      golden_qa.jsonl
```

Synthetic generation uses exact chunk text as context and records provider, model, source path, source chunk ID, and review state. It explicitly creates OpenAI and Gemini clients and therefore does not depend on the runtime `RAGOPS_LLM_PROVIDER` selection. Candidate rows remain separate from the golden set until they are explicitly approved. The merge step rejects duplicate IDs and normalized questions.

The versioned datasets currently contain:

| Dataset | Current contents |
| --- | --- |
| `golden_qa.jsonl` | 80 questions: 70 supported, 5 ambiguous, and 5 unsupported; 35 manual and 45 approved synthetic rows. |
| `synthetic_qa_candidates.jsonl` | 100 reviewed candidates: 50 OpenAI and 50 Gemini; 45 approved and 55 rejected. |
| `retrieval_labels.jsonl` | 45 verified labels, each linked to one audited source chunk from an approved synthetic candidate. |

Retrieval labels follow a separate offline path:

```text
golden_qa.jsonl + processed chunks -> source-scoped chunk inspector
                                             |
                                      verified selection
                                             |
                                             v
                                  retrieval_labels.jsonl
```

The label validator requires supported golden questions, matching question text and expected sources, unique existing chunk IDs, and source-path agreement. Labeling is resumable and does not depend on a running vector database.

Day 18 metrics consume ranked chunk IDs and the verified label set without performing retrieval or external I/O:

```text
ranked chunk IDs + retrieval_labels.jsonl
                    |
                    v
       Recall@k / MRR / Hit Rate@k / nDCG@k
```

Metrics use binary relevance and macro-average questions equally. Duplicate retrieved IDs cannot increase Recall or nDCG, while missing question rankings and invalid cutoffs are rejected instead of being silently scored.

The Day 19 evaluation runner connects the dense retriever to those metrics:

```text
dense_baseline.yaml + retrieval_labels.jsonl
                      |
                 one Qdrant client
                      |
                retrieve every query
                      |
            validated complete rankings
                      |
                      v
        dense_baseline.json + dense_baseline.csv
```

Configuration paths are resolved from the project root. The runner verifies that `top_k` covers every requested metric cutoff, checks the collection before evaluating, preserves retrieval order, rejects duplicate or malformed results, and writes artifacts atomically only after every labeled question succeeds. JSON contains the complete run record; CSV provides stable per-question rows with aggregate metrics repeated for convenient analysis.

Day 23 runs the persisted BM25 index through the same labels and metric functions, then performs a strict paired comparison:

```text
bm25_baseline.yaml + retrieval_labels.jsonl + bm25_index.json.gz
                              |
              validate tokenizer, parameters, and source SHA256
                              |
                     retrieve every query
                              |
                    BM25 JSON + CSV
                              |
           pair by question ID with dense_baseline.json
                              |
                              v
              bm25_vs_dense.json + Markdown report
```

The comparison refuses different question IDs, question text, relevant chunks, sources, or metric cutoffs. It compares the first relevant rank for every question, counts wins and top-10 misses recovered, and groups results with a deterministic question-wording classifier. That classifier is an analysis aid rather than a relevance annotation: exact references are detected first, behavioral/procedural wording second, and all other questions are conceptual/descriptive. The recorded result is BM25 MRR `0.6189` versus dense MRR `0.3359`, with 27 BM25 wins, 6 dense wins, and 12 ties. The synthetic questions' lexical overlap with source chunks is a likely advantage for BM25 and limits generalization.

Day 25 runs the Day 24 candidate over the same paired label set:

```text
hybrid.yaml + labels + dense baseline + BM25 baseline
                    |
        verify labels, cutoffs, component configs,
        BM25 SHA, and live dense/BM25 record parity
                    |
     45 × (dense top 20 + BM25 top 20 -> RRF top 10)
                    |
   hybrid JSON/CSV + three-way JSON/Markdown benchmark
```

The hybrid report stores full fused rankings, RRF source provenance, live Qdrant point count, BM25 source hash, total latency, and separate dense/BM25/fusion latency. The comparison requires identical question IDs, wording, sources, labels, cutoffs, embedding settings, BM25 settings, and source hash where available. It reports per-question and per-relevance-group outcomes so repeated questions for one labeled chunk do not silently masquerade as independent evidence.

The measured RRF candidate reaches MRR `0.5765`, between dense `0.3359` and BM25 `0.6189`. It wins 26 paired questions versus dense and loses one, but wins only 10 versus BM25 while losing 14. Hybrid Hit Rate@10 is `0.8444`, below BM25's `0.8667`; it loses one BM25 top-10 hit. Unweighted consensus fusion therefore does not replace BM25 on this benchmark. Average hybrid latency is `837.4 ms` including a `29,892.1 ms` first-query cold start and `177.1 ms` after the first query; fusion itself averages `0.2 ms`.

Day 26 adds a separate, offline reranked candidate:

```text
hybrid_rerank.yaml + query
          |
 dense top 25 + BM25 top 25
          |
      RRF top 25
          |
 cross-encoder(query, chunk) for every candidate
          |
   score sort -> top 5 + stage timings
```

`cross-encoder/ms-marco-MiniLM-L-6-v2` receives paired query and chunk text rather than independently encoded vectors. The wrapper validates that the model returns one finite scalar per candidate. Sorting uses descending cross-encoder score, then the original RRF rank and chunk ID for deterministic ties. Every output remains a `RetrievedChunk`: its final `score` is the raw cross-encoder relevance logit, `_reranker` records model name plus the prior RRF rank and score, and `_fusion` continues to record dense/BM25 ranks, source scores, and contributions.

The CLI validates BM25 provenance before retrieval, closes Qdrant on success or failure, and reports model-load, dense, BM25, fusion, reranker, and total pipeline latency separately. `--validate-only` deliberately avoids Qdrant and model loading. This is functional Day 26 acceptance; Day 27 evaluates the same candidate on all verified labels.

Day 27 adds the evaluation path:

```text
dense report + BM25 report + Day 25 RRF report + labels
                              |
         validate identities, component settings, and BM25 SHA
                              |
  45 × (dense 25 + BM25 25 -> RRF 25 -> cross-encoder 5)
                              |
 retain RRF-25 candidates + reranked results + stage timings
                              |
 common top-5 four-way comparison + controlled reranker ablation
                              |
       JSON/CSV run + JSON/Markdown benchmark
```

The common depth is important: baseline rankings are truncated to five and the primary metric is MRR@5, avoiding a comparison between a five-result reranker and baseline MRR over ten results. The controlled ablation compares the exact RRF-25 order from the live run before and after cross-encoding, isolating reranking from Day 25's different candidate depths.

Measured MRR@5 is dense `0.3163`, BM25 `0.6152`, Day 25 RRF `0.5641`, and reranked `0.6889`. The cross-encoder improves its own pre-rerank MRR@5 from `0.5644` to `0.6889`, with 16 paired wins, five losses, and 24 ties. It recovers six pre-rerank top-five misses but loses one prior hit. Warmed end-to-end latency averages `4,476.4 ms`; the reranker alone averages `4,274.9 ms`, so the quality gain comes with a large serving-cost penalty.

The Day 20 generation evaluation is a separate pipeline:

```text
golden_qa.jsonl + generation_judge.yaml
                 |
     deterministic 6/2/2 query-type sample
                 |
       dense top-5 retrieval from Qdrant
                 |
      OpenAI answer generation
                 |
      Gemini rubric judgment
                 |
   judgments JSONL + summary JSON
                 |
       manual agree/disagree audit
```

The default cross-provider roles reduce direct self-evaluation: OpenAI `gpt-5-nano` generates and Gemini `gemini-3.6-flash` judges. Every record retains the full retrieved evidence and model provenance. Faithfulness is grounded only in retrieved evidence; the golden reference answer is used for relevance. The parser requires 1–5 scores and enforces refusal semantics: supported answers use `not_applicable`, ambiguous questions require clarification, and unsupported questions require refusal. This judge is an evaluation signal rather than ground truth, so the configured 10 spot-checks remain part of acceptance.

The recorded Day 20 acceptance run completed all 10 questions with mean faithfulness 4.5/5 and mean answer relevance 3.4/5. The stored `codex-manual-audit` reviewed every automatic judgment, agreeing with eight and documenting two relevance-score disagreements; it is an implementation audit, not human sign-off. Because the Docker-backed Qdrant endpoint was unresponsive, this run populated a temporary local Qdrant store from the same 13,481 processed chunks and used the production dense retriever against it. Its cold-start retrieval timings should not be treated as service latency.

Raw documents and processed embedding JSONL are intentionally ignored by Git. Their source URLs, selected paths, snapshot commits, and destination paths are recorded in `data/manifests/source_manifest.json`. Reviewed evaluation JSONL is versioned with the project.

## Online Request Flow

1. Streamlit sends `query` and `top_k` to `POST /query`.
2. FastAPI validates `top_k` as an integer from 1 through 20.
3. The API resolves Qdrant from `QDRANT_URL`, defaulting to `http://localhost:6333` for a host-run API.
4. The dense retriever embeds the stripped query with the same model used during indexing.
5. Qdrant performs cosine-similarity search and returns payloads without vectors.
6. Results are normalized with 1-based ranks, scores, metadata, and the best available source path or URL.
7. Citations are deduplicated by document and section and assigned IDs such as `[1]`.
8. The generation layer builds a context-only prompt and sends it to the client selected at API startup by `RAGOPS_LLM_PROVIDER`. The template client is the default; OpenAI and Gemini are implemented alternatives.
9. FastAPI returns the answer, structured citations, formatted citations, retrieved chunks, used chunk IDs, and total latency.
10. Streamlit renders the response. It does not connect to Qdrant or import the retrieval pipeline directly.

## Runtime Services and Configuration

| Service | Default address | Notes |
| --- | --- | --- |
| Qdrant HTTP | `http://127.0.0.1:6333` | Docker Compose exposes the Qdrant service on the host. |
| Qdrant gRPC | `127.0.0.1:6334` | Exposed but not used by the current Python path. |
| MLflow | `http://127.0.0.1:5000` | Stores retrieval evaluation runs and version/status tags; it is not used by the online request path. |
| FastAPI | `http://127.0.0.1:8000` | Provides `/health`, `/retrieve`, `/query`, and `/docs`. |
| Streamlit | `http://localhost:8501` | Calls FastAPI using `RAGOPS_API_URL`. |

When FastAPI runs on the host, leave `QDRANT_URL` unset or set it to `http://127.0.0.1:6333`. Docker Compose overrides it with `http://qdrant:6333` for the API container. Host-run evaluation commands use `http://127.0.0.1:5000` for MLflow; another Compose service would use `http://mlflow:5000`. Streamlit defaults to `http://127.0.0.1:8000`; override `RAGOPS_API_URL` when the API is elsewhere.

Generation configuration is resolved once when `create_app()` initializes its client:

| Variable | Default | Behavior |
| --- | --- | --- |
| `RAGOPS_LLM_PROVIDER` | `template` | Selects exactly one of `template`, `openai`, or `gemini` for `POST /query`. |
| `OPENAI_API_KEY` | none | Required only when the selected runtime provider is `openai`; also used by synthetic generation when OpenAI is requested. |
| `OPENAI_MODEL` | `gpt-5-nano` | Model passed to the OpenAI Responses API client. |
| `GEMINI_API_KEY` | none | Required only when the selected runtime provider is `gemini`; also used by synthetic generation when Gemini is requested. |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Model passed to the Gemini Interactions API client. |

Both provider credentials may be configured simultaneously, but the online API uses only the selected provider until it is restarted. The synthetic QA and Day 20 judge CLIs are different: they load `.env` themselves and can assign OpenAI and Gemini separate roles in the same batch. Host-run `make serve` and `make dashboard` do not load `.env` automatically. Docker Compose does read `.env` and forwards generation settings to the API container.

## Error Boundaries

- Pydantic request failures, including `top_k` outside 1–20, return HTTP 422.
- Query validation failures detected by the retrieval or generation layer return HTTP 400.
- Unexpected retrieval failures return HTTP 503 from `/retrieve`.
- Unexpected retrieval or generation failures return HTTP 503 from `/query`.
- Streamlit converts connection failures and API error details into readable page messages.
- Qdrant clients are closed after both successful and failed retrieval calls.

## Current Limitations

- Dense retrieval remains the only online retriever. BM25, RRF hybrid, and hybrid-plus-reranker retrieval are available offline, but none is exposed through the API.
- The `production` registry alias documents the selected online version but does not dynamically configure or deploy the API; deployment integration is intentionally not claimed by Day 30.
- The cross-encoder has the highest measured MRR@5 on the current labels, but its warmed stage averages about 4.27 seconds per query. It remains an offline candidate until latency is reduced or routing limits its use.
- The offline `template` provider returns a fixed placeholder response. OpenAI and Gemini clients are implemented, but the API selects only one provider at process startup and has no application-level model routing, fallback, retry policy, or provider comparison in the online path.
- The prompt asks the model to stay grounded and say “I do not know,” but the runtime does not classify unsupported queries, verify answer claims, check citation use, or enforce refusal behavior. The offline judge measures these qualities after the fact. Structured citations describe all retrieved context, not necessarily only evidence referenced by the answer.
- Day 20 generation scores come from one judge model over 10 questions. All 10 have a separate Codex evidence audit, but they do not have independent human sign-off and are not a substitute for larger samples, multiple judges, calibrated human labels, or statistical uncertainty estimates.
- Generation token usage and cost are not captured, and judge results are not logged to MLflow.
- The corpus and generated embeddings are local artifacts and are not distributed in Git.
- Ingestion and index building load the full current record set into memory.
- Source references are usually corpus-relative paths rather than public documentation URLs.
- `GET /health` reports process status and version; it does not probe Qdrant or an external generation provider.
- MLflow tracking currently covers retrieval evaluation only. Generation judgments, cost, online request traces, and promotion decisions are not logged yet.
- Tracing, SQLite persistence, routing, semantic caching, canary gates, failure mining, monitoring, and CI evaluation gates are not implemented.

## Planned Placeholders

The repository contains empty files reserved for later project-plan milestones. They are not active implementations:

- configurations: `routed.yaml`, `cached_routed.yaml`, and `ci_small.yaml`
- scripts: `eval_gate.py`, `mine_failures.py`, `run_canary.py`, `seed_demo_data.py`, and `simulate_traffic.py`
- tests: `test_cache.py`, `test_eval_gate.py`, and `test_router.py`
- topic documents: `canary_gates.md`, `failure_mining.md`, `limitations.md`, `monitoring.md`, `routing.md`, and `semantic_cache.md`
