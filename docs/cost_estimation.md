# Generation cost

Every successful `/query` returns a generation-cost record and stores the same record on its SQLite trace.

This covers generation tokens only. Retrieval, reranking, infrastructure, caching, tools, taxes, and judge calls are excluded.

## Token precedence

1. Use complete OpenAI/Gemini SDK usage: `provider_reported`.
2. Otherwise estimate the exact prompt and answer as `ceil(UTF-8 bytes / 4)`: `heuristic_estimate`.
3. If token evidence is incomplete, cost is unavailable.
4. Template generation records zero tokens and zero cost.

The heuristic is deterministic, not provider billing data.

## Pricing precedence

1. A complete input/output environment-rate pair.
2. Exact provider/model match in [`configs/model_costs.yaml`](../configs/model_costs.yaml).
3. Unavailable; another model's rate is never reused.

The checked-in `generation_model_costs@1.0.0` snapshot contains:

| Provider/model | Input / 1M | Output / 1M |
| --- | ---: | ---: |
| OpenAI `gpt-5-nano` | $0.05 | $0.40 |
| Gemini `gemini-3.6-flash` | $1.50 | $7.50 |

Environment overrides:

```text
RAGOPS_LLM_INPUT_USD_PER_MILLION_TOKENS
RAGOPS_LLM_OUTPUT_USD_PER_MILLION_TOKENS
```

Both are required together.

## Formula and states

```text
amount_usd = (
  input_tokens  * input_rate_per_million
  + output_tokens * output_rate_per_million
) / 1,000,000
```

| Status | Meaning |
| --- | --- |
| `zero_cost` | Local template; amount and tokens are zero |
| `estimated` | Complete tokens and rates produced an amount |
| `unavailable` | Missing tokens or rate; amount is null |

The record also keeps provider/model, token source/estimator, pricing source/table, and both rates. Validation recomputes the amount and rejects partial or inconsistent provenance.

## Commands

```bash
make validate-model-costs
make test-cost
make init-trace-store
make validate-trace-store
```

## Limits

- Prices change; version the table when updating them.
- The byte heuristic can be inaccurate for code, Unicode, and non-English text.
- Estimates exclude caching tiers, tools, media, credits, taxes, and negotiated rates.
- Provider-reported tokens plus a price table still produce an estimate, not an invoice.
