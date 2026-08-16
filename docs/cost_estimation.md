# Generation Cost Estimation

## Day 40 Scope

Day 40 makes the existing response-level cost field reproducible and durable. Every successful `POST /query` response now contains a complete generation-cost record, and the identical record is stored on its SQLite trace. Retrieval, routing probes, and failed queries do not invent model cost.

The implementation estimates generation inference only. Qdrant, embedding, BM25, cross-encoder, host compute, network, storage, grounding, tools, cached input, and provider free-tier/volume adjustments are outside this cost.

## Token Counts

Token-count precedence is explicit:

1. If OpenAI or Gemini returns complete SDK usage metadata, those input/output/total values are retained with `token_source=provider_reported`.
2. If usage metadata is absent, the exact grounded generation prompt and returned answer are estimated separately with `utf8_bytes_div4_ceiling_v1`: `ceil(len(text encoded as UTF-8) / 4)`. The two estimates are added for total tokens and labeled `token_source=heuristic_estimate`.
3. If neither complete provider usage nor both input/output texts are available, counts and amount remain unavailable.
4. The deterministic local template performs no model inference, so it records zero billable tokens, zero cost, and `not_applicable` sources.

The heuristic is deliberately simple, deterministic, dependency-free, and auditable. It is not a provider tokenizer and must not be presented as billed usage.

## Model Cost Table

`configs/model_costs.yaml` defines `generation_model_costs@1.0.0` with standard synchronous text-token rates reviewed on 2026-08-16:

| Provider | Exact model | Input / 1M tokens | Output / 1M tokens | Official source |
| --- | --- | ---: | ---: | --- |
| OpenAI | `gpt-5-nano` | $0.05 | $0.40 | [OpenAI model page](https://developers.openai.com/api/docs/models/gpt-5-nano) |
| Gemini | `gemini-3.6-flash` | $1.50 | $7.50 | [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) |

Lookups require an exact provider/model pair; an unknown or overridden model never borrows a nearby model's rate. The YAML validator rejects unknown keys, duplicate identities, invalid versions/status, non-HTTPS sources, and negative/non-finite rates.

The existing paired environment variables remain an operator override:

```text
RAGOPS_LLM_INPUT_USD_PER_MILLION_TOKENS
RAGOPS_LLM_OUTPUT_USD_PER_MILLION_TOKENS
```

Both must be supplied together. An override takes precedence over the table and is labeled `pricing_source=environment_override`; it does not claim the table identity. `RAGOPS_MODEL_COST_CONFIG` may point to a different strict table.

## Calculation and Response Contract

For external models:

```text
amount_usd = (
    input_tokens  * input_usd_per_million_tokens
  + output_tokens * output_usd_per_million_tokens
) / 1,000,000
```

The response records the amount, currency, provider/model, three token counts, token source/estimator, pricing source/table identity, and both rates. Example with provider-reported usage:

```json
{
  "cost": {
    "amount_usd": 0.00013,
    "currency": "USD",
    "status": "estimated",
    "provider": "openai",
    "model": "gpt-5-nano",
    "input_tokens": 1000,
    "output_tokens": 200,
    "total_tokens": 1200,
    "token_source": "provider_reported",
    "token_estimator": null,
    "pricing_source": "model_cost_table",
    "price_table_id": "generation_model_costs@1.0.0",
    "input_usd_per_million_tokens": 0.05,
    "output_usd_per_million_tokens": 0.4
  }
}
```

The three states are:

- `zero_cost`: deterministic template generation; amount and billable tokens are exactly zero.
- `estimated`: complete token counts and rates produced a reproducible amount.
- `unavailable`: no matching rate and/or no usable token evidence; `amount_usd` is null, never zero.

Strict validation recomputes estimated amounts and rejects inconsistent totals, partial token/rate tuples, mismatched provenance, non-finite values, or a table rate without a table identity.

## Trace Persistence

SQLite schema version 4 adds nullable generation identity and cost columns to `traces`. New successful `/query` traces store exactly the response cost object. The live API evaluator compares every response field to its corresponding trace, just as it already does for pipeline identity, answer, timings, and chunks.

Version 3 databases migrate in place by adding nullable columns. Existing query/retrieval/error records remain readable with no cost object, because historical requests did not record enough evidence to reconstruct a trustworthy estimate. New `/retrieve` traces reject generation cost, and failed `/query` traces leave cost empty when no completed result exists.

## Commands

Validate the model identities, rates, sources, and estimator without contacting a provider:

```bash
make validate-model-costs
```

Run the focused cost/trace/API gate:

```bash
make test-cost
```

Initialize or migrate the configured trace database and then validate it:

```bash
make init-trace-store
make validate-trace-store
```

## Interpretation Limits

- Table rates are a reviewed snapshot, not a promise of future provider pricing. Review the official sources and version the table when rates change.
- The heuristic can differ materially from provider tokenization, especially for non-English text, code, Unicode, and hidden reasoning. Provider-reported usage always takes precedence.
- Standard text rates exclude caching, batch/flex/priority service classes, tools, search grounding, storage, media, taxes, credits, and negotiated pricing.
- A response estimate is not an invoice. Reconcile production spend against provider billing exports.
- Day 40 persists per-request cost but does not aggregate production cost by route, pipeline, user, or time window. Day 41 reuses the same estimator/table for a controlled offline router projection over exact selected-context prompts and verified reference-answer lengths; that comparison is not observed provider usage or an invoice.
