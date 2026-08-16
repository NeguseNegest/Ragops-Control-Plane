# Week 6 Router Comparison

Evaluation: `router_comparison@0.1.0` (evaluated)  
Router: `rule_router@0.1.0` (draft)  
Mode: `artifact_replay`

## Result

The routed policy improves the combined support/refusal proxy over always FAST and reduces latency/cost versus always CAREFUL, but loses substantial supported retrieval quality and retains a 20% supported false-refusal rate.

| Strategy | Supported Hit@5 | Supported MRR | Unsupported refusal | Policy accuracy | Combined proxy | Avg latency | p95 latency | Projected cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always FAST | 28.89% | 0.2778 | 0.00% | 78.95% | 22.81% | 679.9 ms | 611.5 ms | $0.00145935 |
| Always CAREFUL | 84.44% | 0.6889 | 0.00% | 78.95% | 66.67% | 4681.6 ms | 7807.0 ms | $0.00355245 |
| Routed | 55.56% | 0.4469 | 100.00% | 84.21% | 64.91% | 2514.3 ms | 6319.9 ms | $0.00320625 |

The combined proxy counts a supported question only when a relevant chunk is present at the strategy's final depth, and counts an unsupported question only when it is refused. It is not answer-quality scoring.

## Routed supported distribution

- CAREFUL: 16
- FAST: 2
- NO_ANSWER: 9
- STANDARD: 18

## Methodology

- Paired replay of fixed dense top-2, fixed reranked top-5, and the route-selected ranking on verified supported labels.
- Serial composition of recorded dense and reranked retrieval latency; the dense top-10 measurement proxies the top-2 probe and cold starts remain included.
- Day 40 UTF-8 heuristic over the exact selected-context prompt and verified reference answer, priced as the configured model; no provider call was made.
- Policy/refusal quality uses the reviewed Day 39 unsupported set; operational latency, retrieval quality, and projected generation cost are not fabricated for unsupported fixed baselines.

## Interpretation

- Versus Always FAST: Hit@5 +26.67%; combined proxy +42.11%; average latency +269.81%; projected total cost +119.70%.
- Versus Always CAREFUL: Hit@5 -28.89%; combined proxy -1.75%; average latency -46.29%; projected total cost -9.75%.

Decision: **keep_router_draft**. Day 42 must tune thresholds and inspect route distribution before automatic routed execution is considered stable.

## Limitations

- This is a deterministic replay of prior measured artifacts, not a simultaneous live benchmark.
- The dense top-10 timing is a conservative proxy for the top-2 probe; STANDARD latency assumes two serial dense calls and CAREFUL assumes a serial dense probe plus reranked retrieval.
- Generation cost uses verified reference-answer length, not an observed provider response, and excludes non-token charges listed by the Day 40 cost table.
- Retrieval Hit@5 is an evidence-availability proxy, not answer correctness, faithfulness, or relevance.
- Only 12 manually reviewed unsupported questions are included, and 9 of 45 supported questions are refused by the current draft threshold.
