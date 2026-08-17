# Limitations and Deferred Scope

This repository is a portfolio-scale RAG engineering system, not a claim of production readiness. The limitations below define the boundary between measured implementation, remaining work in the condensed 52-day plan, and optional Future Work.

## Evaluation evidence

- The current golden set contains 80 questions, retrieval evaluation uses 45 verified labels, refusal evaluation uses 12 manually authored unsupported questions, and generation judging covers a deterministic 10-question sample. Days 46–48 must strengthen and review these sets before the final benchmark.
- Most questions are derived from the documentation corpus and usually have one labeled relevant chunk. This can favor lexical overlap and does not represent unbiased production traffic or complete relevance judgments.
- Hit Rate, Recall, MRR, and nDCG measure retrieval evidence, not whether a generated answer is fully correct. The LLM judge is model-dependent and requires manual spot checks.
- Day 41/42 router latency is deterministic replay of previously measured artifacts. It is useful for paired comparison but is not a simultaneous live load test.
- The Day 44 gate is a five-case deterministic regression smoke test over in-memory Qdrant. Its perfect thresholds protect a known fixture, not generalization; it selects the executable dense pipeline rather than the registry's cross-encoder candidate and leaves the full comparative benchmark to Day 47.
- The template gate can measure answer presence and answer-referenced citation coverage/precision, but not semantic faithfulness. Faithfulness is explicitly reported as unavailable rather than inferred from citation structure.

## Routing and refusal

- `rule_router@0.2.0` is deterministic and explainable but remains `draft`. It refused all 12 reviewed unsupported questions while falsely refusing 9 of 45 supported questions.
- Only two supported evaluation questions select FAST, so that route has insufficient evidence for a broad quality claim.
- `/route` executes the probe and enforces NO_ANSWER refusal. `/query` still requires explicit pipeline selection and does not automatically dispatch FAST, STANDARD, or CAREFUL.
- Raw dense scores and score gaps are corpus/model/index-specific signals, not calibrated probabilities or general semantic-scope guarantees.

## Latency and cost

- The cross-encoder has the strongest recorded top-five ranking quality but adds material latency; warmed reranker latency is approximately 4.27 seconds per query in the current artifact.
- Recorded latency includes cold-start and cross-process effects in several historical reports. Results are suitable for disclosed comparisons, not service-level objectives.
- Provider usage is preferred when available, but fallback token counts use a UTF-8-byte heuristic. Checked rates can change and exclude cache discounts, tools, service tiers, media, credits, taxes, and negotiated pricing.
- Cost evidence is not an invoice and is not yet aggregated into production budgets or reconciled against provider billing.

## Serving, storage, and operations

- The local template generator returns a deterministic placeholder; OpenAI and Gemini clients require external credentials and network availability.
- The API selects one generation provider per process and does not implement provider fallback, retry orchestration, or application-level model routing.
- SQLite is appropriate for local query traces but does not provide distributed writes, retention automation, encryption, redaction, or full-text trace search.
- The existing feedback table and repository methods are tested schema work. No feedback HTTP API is required or claimed.
- `GET /health` reports process status; it does not prove that Qdrant or an external generation provider is healthy.
- The raw corpus, embeddings, and BM25 index are local generated artifacts rather than committed repository data. Clean-environment reproduction therefore depends on documented ingestion and index-building steps.

## Explicitly deferred Future Work

The condensed plan does not require partial implementations of the following:

- semantic caching and cache invalidation
- canary, shadow, or traffic-simulation systems
- automated failure mining from live feedback
- a feedback collection API
- OpenTelemetry, Prometheus, and Grafana integration
- PostgreSQL trace storage
- Kubernetes or Terraform deployment
- multi-user authentication or a complex frontend
- learned query routing, embedding training, or LLM fine-tuning

These extensions may be valuable for a deployed product, but empty configs, packages, scripts, tests, workflows, and documentation pages are intentionally not retained as evidence that they exist.

## Remaining required work

The revised plan still requires Day 45 CI integration, final reviewed evaluation data and benchmark, manual failure analysis and regression cases, the compact engineering dashboard, clean-environment hardening, final documentation, and portfolio packaging.
