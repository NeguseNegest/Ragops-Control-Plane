# Limitations

This is a local RAG control plane, not a production service.

## Quality evidence

- The final retrieval set has 50 supported labels; semantic judging uses 10 answers per pipeline.
- Questions come from curated documentation, not real traffic.
- Relevance labels are incomplete and can favor lexical overlap.
- LLM judge scores are directional, not human ground truth.
- The five-case CI gate protects a fixture, not generalization.

## Routing

- The router is `draft`: 7/50 supported questions were falsely refused and 5/30 adversarial questions were answered.
- FAST has only two supported final-benchmark examples.
- Streamlit orchestrates `/route` then `/query`; direct `/query` callers bypass routing.
- FAST probe reuse is configured but not implemented by the dashboard flow.
- Dense scores are corpus-specific, not calibrated probabilities.

## Performance and cost

- Reranked retrieval has 7.67 s p95 latency in the final benchmark.
- Recorded latency includes cold starts and is not a load test.
- Cost covers generation tokens only and is not an invoice.
- Provider fallback, retry orchestration, and budget enforcement are absent.

## Operations

- SQLite and process-local caches do not support horizontal scaling.
- There is no auth, rate limiting, redaction, retention automation, or distributed tracing.
- `/health` does not check Qdrant or the generation provider.
- A host dashboard cannot see traces stored only in Compose's named volume.
- Raw documentation and generated dense/BM25 indexes are not committed.

## Deferred

- backend-enforced route dispatch and probe reuse;
- larger human-reviewed routing and answer-quality sets;
- reranker latency work;
- semantic caching;
- canary/shadow traffic;
- automated failure mining;
- hosted trace storage and telemetry; and
- deployment automation.
