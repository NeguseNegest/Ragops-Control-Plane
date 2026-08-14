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
    "amount_usd": 0.00042,
    "currency": "USD",
    "status": "estimated",
    "input_tokens": 1800,
    "output_tokens": 60,
    "total_tokens": 1860
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
    "generation_model": "configured-model",
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

`cost` describes answer-generation cost only; local retrieval compute is not assigned a dollar price.

- The deterministic local template returns `amount_usd: 0.0` and `status: zero_cost`.
- OpenAI Responses and Gemini Interactions token usage is preserved when supplied by the SDK.
- When both per-million-token rate variables are configured, the API returns `status: estimated` and calculates input plus output token cost.
- When provider usage or either price is unavailable, the response returns `status: unavailable` and `amount_usd: null` rather than falsely reporting zero.

The price inputs are operator-controlled because model prices change. Estimates are not provider invoices, and Day 33 does not persist or aggregate them.

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
