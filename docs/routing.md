# Routing

`rule_router@0.2.0` is deterministic, explainable, and still `draft`.

## Routes

| Route | Use | Config | Max chunks |
| --- | --- | --- | ---: |
| FAST | High-confidence, simple query | `dense_baseline` | 2 |
| STANDARD | Normal supported query | `dense_baseline` | 10 |
| CAREFUL | Ambiguous or complex query | `hybrid_rrf_cross_encoder` | 5 |
| NO_ANSWER | Weak or missing evidence | No generation | 0 |

## Decision order

Rules run in this order:

1. **NO_ANSWER:** no results or top dense score `< 0.531`.
2. **CAREFUL:** any configured ambiguity/complexity rule matches. The main boundary is top-two score gap `< 0.03`.
3. **FAST:** every high-confidence and low-complexity rule matches.
4. **STANDARD:** fallback.

The exact policy is in [`configs/routed.yaml`](../configs/routed.yaml). Key thresholds:

| Signal | CAREFUL | FAST |
| --- | --- | --- |
| Top score | `< 0.56` | `>= 0.72` |
| Score gap | missing or `< 0.03` | `>= 0.05` |
| Token count | `> 20` | `<= 12` |
| Complexity markers | `>= 1` | `0` |
| Clause markers | `>= 3` | `<= 1` |
| Long-token ratio | `>= 0.40` | `<= 0.30` |

The response includes the primary reason, all matched reason codes, query features, probe scores, selected config, depth cap, and whether probe reuse is intended.

## Current request flow

```text
Streamlit -> POST /route
          -> NO_ANSWER: render refusal and stop
          -> FAST/STANDARD/CAREFUL: POST /query with explicit config
```

This is client orchestration. `/query` can be called directly and does not enforce the router. FAST is configured for probe reuse, but the dashboard currently performs dense retrieval again.

## Final evidence

On 50 supported retrieval questions:

| Route | Count |
| --- | ---: |
| FAST | 2 |
| STANDARD | 12 |
| CAREFUL | 29 |
| NO_ANSWER | 7 |

On 30 adversarial questions, 25 were correctly refused. The 7/50 supported false refusals and 5/30 adversarial misses keep the router in `draft` status.

Scores are specific to this corpus, embedding model, and index. They are not probabilities.

## Commands

```bash
make validate-router-config
make route-query ROUTER_QUERY="What is FastAPI?"
make test-routing-probe
make validate-router-tuning
make test-router-stabilization
```

See [no-answer behavior](no_answer.md) for refusal calibration and [evaluation](evaluation.md) for the routed benchmark.
