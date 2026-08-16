# Production Query API

## Request

Day 33 upgrades `POST /query` to execute one of three validated retrieval configs:

| `config` value | Route | Registry status | Configured candidate path |
| --- | --- | --- | --- |
| `dense_baseline` | `dense` | `approved` | Dense top 10 by default |
| `hybrid_rrf` | `hybrid` | `rejected` | Dense 20 + BM25 20 → RRF 10 |
| `hybrid_rrf_cross_encoder` | `reranked` | `evaluated` | Dense 25 + BM25 25 → RRF 25 → cross-encoder 5 |

The request-level `top_k` overrides the configured final output depth and is restricted to 1–20. Component candidate depths remain fixed by YAML. The rejected hybrid is intentionally executable for controlled comparison; making it selectable does not promote it or change the Day 30 `production` alias.

```json
{
  "query": "How does reciprocal rank fusion combine rankings?",
  "top_k": 5,
  "config": "hybrid_rrf_cross_encoder",
  "debug": true
}
```

Omitting `config` selects `dense_baseline`. Omitting `debug` keeps the stable `debug: null` response field.

## Success Response

```json
{
  "trace_id": "ed8ad2e8-1ea8-4d58-92c7-556aa24cc160",
  "route": "reranked",
  "config": "hybrid_rrf_cross_encoder",
  "config_version": "1.0.0",
  "query": "How does reciprocal rank fusion combine rankings?",
  "answer": "RRF adds reciprocal contributions derived from each source rank. [1]",
  "citations": [
    {
      "citation_id": "[1]",
      "document_id": "doc-1",
      "title": "RRF documentation",
      "url": "docs/rrf.md",
      "metadata": {},
      "chunk_ids": ["chunk-1"]
    }
  ],
  "citation_text": "[1] RRF documentation - docs/rrf.md",
  "chunks": [
    {
      "chunk_id": "chunk-1",
      "document_id": "doc-1",
      "text": "Reciprocal rank fusion combines ranked lists using reciprocal rank contributions.",
      "score": 0.91,
      "rank": 1,
      "metadata": {},
      "source_url": "docs/rrf.md"
    }
  ],
  "used_chunk_ids": ["chunk-1"],
  "latency_ms": 4512.3,
  "component_latencies": {
    "embedding_ms": 7.2,
    "dense_ms": 4.1,
    "bm25_ms": 88.0,
    "fusion_ms": 0.2,
    "reranker_ms": 4274.9,
    "generation_ms": 18.4
  },
  "cost": {
    "amount_usd": 0.00002409,
    "currency": "USD",
    "status": "estimated",
    "provider": "openai",
    "model": "gpt-5-nano",
    "input_tokens": 1800,
    "output_tokens": 60,
    "total_tokens": 1860,
    "token_source": "provider_reported",
    "token_estimator": null,
    "pricing_source": "model_cost_table",
    "price_table_id": "generation_model_costs@1.0.0",
    "input_usd_per_million_tokens": 0.05,
    "output_usd_per_million_tokens": 0.4
  },
  "debug": {
    "pipeline_id": "hybrid_rrf_cross_encoder@1.0.0",
    "pipeline_status": "evaluated",
    "retriever_interface": "common_v1",
    "requested_top_k": 5,
    "returned_chunks": 5,
    "configured_depths": {
      "dense": 25,
      "bm25": 25,
      "fusion": 25,
      "reranker_candidates": 25,
      "reranker_output": 5
    },
    "generation_provider": "openai",
    "generation_model": "gpt-5-nano",
    "resource_cache_hits": {
      "bm25_index": true,
      "reranker_model": true
    }
  }
}
```

The example values illustrate the schema, not a recorded benchmark response. A successful body `trace_id` equals the persisted SQLite primary key and the `X-Trace-ID` response header. Debug output contains no API keys, prompts, absolute artifact paths, or internal service addresses.

## Resource Lifecycle

The runtime validates all three YAML configs when the application starts. It opens a fresh Qdrant client for each request and closes it on success or failure. Hybrid/reranked paths lazily load and provenance-check the BM25 index once per process. The reranked path also lazily loads one cross-encoder per unique model configuration. Cache state is protected during initialization; later requests reuse those local resources.

Docker images include the checked-in configs. Compose mounts `./data/processed` read-only so the ignored local BM25 artifact is available without baking corpus state into the image. A fresh checkout must build that index before hybrid/reranked serving.

## Cost Semantics

`cost` describes answer-generation inference only; local retrieval/embedding/reranking compute is not assigned a dollar price.

- The deterministic local template returns `amount_usd: 0.0`, zero billable tokens, `status: zero_cost`, and not-applicable token/pricing sources.
- OpenAI Responses and Gemini Interactions usage is authoritative when supplied by the SDK and is labeled `provider_reported`.
- Without provider usage, Day 40 estimates the exact prompt and answer with the versioned UTF-8-byte heuristic and labels them `heuristic_estimate`; it never presents those counts as billed usage.
- Exact default-model rates come from `generation_model_costs@1.0.0`. A complete environment rate pair overrides the table. Unknown models never inherit another model's price.
- Missing rate/token evidence returns `status: unavailable` and `amount_usd: null`, while retaining whatever non-price provenance is known.
- The complete response cost record is persisted on its successful SQLite query trace and checked for exact parity by the live API evaluator.

Estimates are not provider invoices. Model prices change, and the standard table excludes cached input, tools, grounding, batch/flex/priority tiers, storage, media, credits, taxes, and negotiated rates. See [`cost_estimation.md`](cost_estimation.md) for precedence, formula, schema-v4 persistence, commands, and limitations.

## Error Contract

| Condition | HTTP | Public detail | Trace behavior |
| --- | ---: | --- | --- |
| Invalid body, config, or `top_k` | 422 | FastAPI validation detail | Handler never entered; no trace |
| Empty/invalid query detected after entry | 400 | Stable validation message | Error trace + `X-Trace-ID` |
| Qdrant/index/reranker/retriever initialization failure | 503 | `Selected query pipeline is unavailable.` | Error trace + `X-Trace-ID` |
| Retrieval execution failure | 503 | `Unable to retrieve chunks with the selected pipeline.` | Partial timings + `X-Trace-ID` |
| Generation failure | 503 | `Unable to generate answer.` | Retrieved chunks, partial timings + `X-Trace-ID` |
| Trace persistence failure | 503 | `Unable to persist query trace.` | No success is claimed |

Internal exception text is retained in the local trace for diagnosis but is not reflected to clients. Run the focused contract suite with `make test-query-endpoint`.

## Offline Integration Test

Day 34 adds a hermetic acceptance path for this contract. `tests/test_api_integration.py` creates the application through `create_app`, initializes a real in-memory Qdrant collection from `tests/fixtures/ci_small_corpus.jsonl`, injects deterministic query vectors defined by `configs/ci_small.yaml`, and writes traces to a per-test temporary SQLite database. It therefore crosses the HTTP, validation, dense-retrieval, citation, template-generation, latency, cost, and persistence boundaries without relying on Docker, a network service, an API key, or a downloaded model.

The integration cases verify:

- `/health` returns the package version without creating a trace;
- `/retrieve` returns the expected ranked fixture chunk and persists matching unused evidence;
- `/query` returns matching body/header/storage trace IDs, config provenance, citations, used chunks, timings, debug output, and an identical response/trace zero-cost template record;
- malformed bodies, invalid depths, unknown configs, and wrong field types return HTTP 422 before creating a trace; and
- an accepted whitespace-only query returns the documented traced HTTP 400 response.

Run the GitHub Actions-equivalent target locally with `make test-api-ci PYTHON=.venv/bin/python`. The complete CI design and its intentional exclusions are documented in [`ci.md`](ci.md).

## Live Evaluation Through HTTP

Day 35 adds `scripts/evaluate_api.py` for full-corpus service verification. Unlike the Day 34 in-memory fixture, it calls a running `/query` endpoint backed by the real Qdrant corpus. For every label it validates config/route/version, ranks and scores, citations and used chunks, debug metadata, cost shape, and response/header trace parity before computing retrieval metrics.

When the evaluator can read the API's SQLite path, it verifies the corresponding trace, generation-cost parity, and ordered evidence immediately after each response. It also requires exact top-10 ranking parity with the offline dense report and verifies all four configured MLflow runs unless those checks are explicitly skipped. The default Make target is `make evaluate-api`; `API_URL` and `API_TRACE_DB_PATH` must identify the running service and its database. The historical Day 35 report predates schema v4; cost parity applies to new Day 40 runs.

## Day 36–39 Routing and Refusal API

Day 36 defines the policy, Day 37 emits probe features, Day 38 selects a route/reason, and Day 39 returns an enforced refusal when that route is `NO_ANSWER`. Submit a query to `POST /route`:

```json
{
  "query": "What is FastAPI?"
}
```

The response contains the normalized query, the complete `decision`, the exact `features` used, minimal probe evidence, and probe timing:

```json
{
  "query": "What is FastAPI?",
  "decision": {
    "router_id": "rule_router@0.1.0",
    "router_status": "draft",
    "feature_schema_version": 1,
    "route": "STANDARD",
    "reason_code": "standard_fallback",
    "reason": "The query matches neither the earlier NO_ANSWER/CAREFUL rules nor every FAST condition.",
    "matched_reason_codes": ["standard_fallback"],
    "pipeline_config": "dense_baseline",
    "maximum_top_k": 10,
    "reuse_probe": false,
    "generate_answer": true,
    "response_mode": null
  },
  "features": {
    "schema_version": 1,
    "query_length": {"character_count": 16, "token_count": 3},
    "lexical_complexity": {
      "unique_token_count": 3,
      "unique_token_ratio": 1.0,
      "average_token_length": 4.333333333333333,
      "maximum_token_length": 7,
      "long_token_count": 0,
      "long_token_ratio": 0.0,
      "clause_marker_count": 0,
      "complexity_marker_count": 0
    },
    "retrieval_confidence": {
      "requested_top_k": 2,
      "result_count": 2,
      "top_score": 0.66855043,
      "score_gap": 0.01691073
    }
  },
  "probe_chunks": [
    {"chunk_id": "45fca43c-6f25-56b7-a4ce-cf43ce7718a7", "score": 0.66855043, "rank": 1},
    {"chunk_id": "18b16f8c-b010-5d65-911b-5529df8a4b5d", "score": 0.6516397, "rank": 2}
  ],
  "probe_timings": {"total_ms": 17619.19936002232, "embedding_ms": 17337.966794962995, "dense_ms": 25.63913504127413},
  "refusal": null
}
```

The values above come from the recorded Day 38 live CLI smoke query. They demonstrate the contract; cold process/model initialization dominates this one observation and is not a service-level latency claim.

For a query whose top score is below the strict Day 39 threshold, the same response has `decision.route: NO_ANSWER` and:

```json
{
  "refusal": {
    "answer": "I do not know based on the available FastAPI, MLflow, and Qdrant documentation.",
    "prompt_version": "no_answer_v1",
    "prompt_sha256": "<64 lowercase hexadecimal characters>",
    "generated_by": "deterministic_policy"
  }
}
```

The refusal is policy-generated rather than model-generated: no template/OpenAI/Gemini call occurs, probe evidence is not cited, and no unsupported factual content is introduced. FAST, STANDARD, and CAREFUL continue to return `refusal: null`.

`POST /route` does not execute the selected final retrieval/generation pipeline or expose document text. It also does not create a Day 31 query trace because the current trace schema records completed `/retrieve` and `/query` attempts. Invalid queries return HTTP 400; unavailable probe resources or failed probe execution return HTTP 503; malformed request bodies return HTTP 422 before entering the handler.

`POST /query` remains explicitly config-selected (or defaults to `dense_baseline`), and its lower-case `route` still describes the pipeline actually executed. Day 39 therefore enforces refusal on the routing surface without silently enabling general automatic dispatch.

Use `make validate-router-config`, `make validate-no-answer`, and `make evaluate-no-answer` for policy/evaluation evidence. See [`routing.md`](routing.md) for precedence and [`no_answer.md`](no_answer.md) for threshold derivation, metrics, and limitations.
