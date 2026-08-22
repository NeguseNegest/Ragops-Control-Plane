# Limitations and Deferred Scope

This repository is a portfolio-scale RAG engineering system, not a claim of production readiness. The limitations below separate measured implementation from reproducibility/packaging work and optional Future Work.

## Evaluation evidence

- The Day 46 final snapshots contain 100 reviewed golden questions, 50 retrieval-labeled questions, and 30 adversarial/unsupported prompts. Day 47 completed five-way retrieval over all 50 labels, routed refusal over all 30 adversarial prompts, and 10 answer-quality judgments per pipeline. The 50 semantic judgments are cross-provider LLM estimates on a fixed supported sample, not human adjudication or statistically broad production evidence.
- The final set is manually understandable and source-audited, but it is curated rather than sampled from real production traffic. Its 72 supported questions are documentation-heavy, and the intentional overlap between unsupported golden and adversarial prompts is useful for paired refusal evaluation but is not independent evidence.
- Most questions are derived from the documentation corpus and usually have one labeled relevant chunk. This can favor lexical overlap and does not represent unbiased production traffic or complete relevance judgments.
- Hit Rate, Recall, MRR, and nDCG measure retrieval evidence, not whether a generated answer is fully correct. The LLM judge is model-dependent and requires manual spot checks.
- Day 41/42 router latency is deterministic replay of previously measured artifacts. It is useful for paired comparison but is not a simultaneous live load test.
- Day 47 routed latency is also measured-artifact composition. STANDARD uses dense top-10 latency as both the top-two probe proxy and full retrieval; CAREFUL adds that proxy to the reranked run. The report labels this scope and does not treat it as load-test evidence.
- The Day 44 gate is a five-case deterministic regression smoke test over in-memory Qdrant. Its perfect thresholds protect a known fixture, not generalization; it selects the executable dense pipeline rather than the registry's cross-encoder candidate and does not enforce the completed Day 47 comparative benchmark in pull requests.
- The template gate can measure answer presence and answer-referenced citation coverage/precision, but not semantic faithfulness. Faithfulness is explicitly reported as unavailable rather than inferred from citation structure.
- Day 45 CI intentionally excludes the ignored full corpus/index, live MLflow, external generation providers, cross-encoder model execution, Streamlit runtime tests, and Docker deployment. Those exclusions keep pull requests hermetic; passing CI is not evidence that the separately executed Day 47 live/provider benchmark can be reproduced without its data, services, credentials, and provider quota.
- Day 48 verifies that 15 reviewed diagnoses remain consistent with the frozen Day 47 evidence and promotes 14 to a regression-case dataset. It does not claim that the proposed fixes are implemented, that the diagnoses are controlled causal proof, or that future model/provider judgments will be identical.

## Routing and refusal

- `rule_router@0.2.0` is deterministic and explainable but remains `draft`. The earlier calibration artifact refused all 12 reviewed unsupported questions while falsely refusing 9 of 45 supported questions. The broader final benchmark correctly refused 25/30 adversarial questions and falsely refused 7/50 supported retrieval questions.
- Only two supported evaluation questions select FAST, so that route has insufficient evidence for a broad quality claim.
- `/route` executes the probe and enforces NO_ANSWER refusal. `/query` still requires explicit pipeline selection and does not automatically dispatch FAST, STANDARD, or CAREFUL. The Day 49 Query Playground composes the two endpoints for dashboard requests, but this client-side orchestration does not protect direct `/query` callers or reuse the FAST probe.
- Raw dense scores and score gaps are corpus/model/index-specific signals, not calibrated probabilities or general semantic-scope guarantees.

## Latency and cost

- The cross-encoder has the strongest recorded top-five ranking quality but adds material latency; the final benchmark records 7.67 seconds p95 retrieval latency including cold starts, while an earlier warmed run averaged approximately 4.27 seconds in the reranker stage alone.
- Recorded latency includes cold-start and cross-process effects in several historical reports. Results are suitable for disclosed comparisons, not service-level objectives.
- Provider usage is preferred when available, but fallback token counts use a UTF-8-byte heuristic. Checked rates can change and exclude cache discounts, tools, service tiers, media, credits, taxes, and negotiated pricing.
- Day 47 cost/query is a controlled token projection using the same reference-answer basis for every pipeline. It excludes judge calls, local embedding/sparse/reranking compute, infrastructure, caching, credits, and taxes; it is not an invoice or a production budget.

## Serving, storage, and operations

- The local template generator returns a deterministic placeholder; OpenAI and Gemini clients require external credentials and network availability.
- The API selects one generation provider per process and does not implement provider fallback, retry orchestration, or application-level model routing.
- SQLite is appropriate for local query traces but does not provide distributed writes, retention automation, encryption, redaction, or full-text trace search.
- The Engineering tab reads the configured host-visible SQLite file. When FastAPI runs in Docker's named trace volume and Streamlit runs on the host, container-only traces do not appear at the default host path; the frozen Day 47/48 benchmark and failure evidence is unaffected.
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

The final architecture and reviewer README are complete. The remaining planned work is clean-environment hardening followed by demo/portfolio packaging; neither is claimed complete here.
