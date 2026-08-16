# Week 6 Router Stabilization

Evaluation: `router_stabilization@0.1.0` (evaluated)
Baseline: `rule_router@0.1.0`
Target: `rule_router@0.2.0` (draft)

## Selection

Selected `thresholds.careful.score_gap_below = 0.030`. The 30-question tuning Hit@5 is 60.00%; the 15-question validation Hit@5 is 73.33%. Every configured refusal, validation, latency, and cost constraint passes.

| Candidate gap | Eligible | Tuning Hit@5 | Validation Hit@5 | Full Hit@5 | Avg latency | Total projected cost |
|---:|:---:|---:|---:|---:|---:|---:|
| 0.010 | yes | 50.00% | 66.67% | 55.56% | 2514.3 ms | $0.00320625 |
| 0.015 | yes | 53.33% | 66.67% | 57.78% | 2635.8 ms | $0.00319020 |
| 0.020 | yes | 56.67% | 66.67% | 60.00% | 2980.6 ms | $0.00320680 |
| 0.025 | yes | 56.67% | 66.67% | 60.00% | 3149.1 ms | $0.00318205 |
| 0.030 | yes | 60.00% | 73.33% | 64.44% | 3345.6 ms | $0.00312740 |
| 0.035 | yes | 60.00% | 73.33% | 64.44% | 3345.6 ms | $0.00312740 |
| 0.040 | no | 60.00% | 73.33% | 64.44% | 3531.0 ms | $0.00306045 |
| 0.045 | no | 63.33% | 73.33% | 66.67% | 3603.2 ms | $0.00300775 |

## Before and after

| Metric | v0.1.0 | v0.2.0 |
|---|---:|---:|
| Supported Hit@5 | 55.56% | 64.44% |
| Supported MRR | 0.4469 | 0.5267 |
| Combined proxy | 64.91% | 71.93% |
| Average replay latency | 2514.3 ms | 3345.6 ms |
| Projected cost | $0.00320625 | $0.00312740 |
| Supported false refusal | 20.00% | 20.00% |

## Target route distribution

| Scope | FAST | STANDARD | CAREFUL | NO_ANSWER |
|---|---:|---:|---:|---:|
| All | 2 | 11 | 23 | 21 |
| Supported | 2 | 11 | 23 | 9 |
| Unsupported | 0 | 0 | 0 | 12 |

7 supported questions move from STANDARD to CAREFUL; no other route transition occurs. Target status remains **keep_draft**: The selected CAREFUL threshold improves held-out supported retrieval without weakening refusal, but the evidence is small/in-sample and supported false refusal remains 20%.

## Limitations

- The 30/15 supported split is deterministic but small; the same 45-question artifact family was already inspected during Day 41.
- Only the CAREFUL score-gap threshold was tuned. NO_ANSWER remains locked to Day 39 calibration and FAST has only two observed supported examples.
- Hit@5 measures evidence availability rather than answer correctness, and latency/cost retain the Day 41 replay/projection limitations.
- The route distribution reflects 45 supported and 12 authored unsupported questions, not production traffic.
