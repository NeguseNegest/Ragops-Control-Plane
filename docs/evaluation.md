# Evaluation

The central result is [`final_benchmark@1.0.0`](../reports/final_benchmark.md).

## Data

| Artifact | Count | Use |
| --- | ---: | --- |
| `final_golden_qa.jsonl` | 100 | Reviewed question set |
| `final_retrieval_labels.jsonl` | 50 | Supported retrieval evaluation |
| `final_adversarial_qa.jsonl` | 30 | Refusal evaluation |
| `regression_cases.jsonl` | 14 | Stable failures to protect |

These files live under `data/eval`. The final snapshots are separate from historical datasets. Every retrieval label points to reviewed chunk IDs in the processed corpus.

## Metrics

| Metric | Scope |
| --- | --- |
| Recall@5, MRR@5 | All 50 supported labels |
| Faithfulness, relevance | Same 10 supported questions per pipeline; 1–5 LLM-judge scores |
| Refusal correctness | Routed policy on all 30 adversarial questions |
| p50/p95 latency | Retrieval only; routed values use serial artifact composition |
| Cost/query | Controlled prompt/reference-answer token projection |

OpenAI `gpt-5-nano` generated 50 benchmark answers. Gemini `gemini-3.6-flash` judged them.

## Final result

| Pipeline | Recall@5 | MRR@5 | Faith. | Relevance | p95 | Cost/query | Refusal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 50.7% | 40.6% | 5.0 | 4.3 | 226.0 ms | $0.00006713 | N/A |
| BM25 | 77.7% | 57.4% | 5.0 | 4.5 | 93.0 ms | $0.00008321 | N/A |
| Hybrid RRF | 72.7% | 58.2% | 4.8 | 4.3 | 272.6 ms | $0.00007965 | N/A |
| Hybrid + reranker | **81.0%** | **64.7%** | 5.0 | 4.5 | 7,669.2 ms | $0.00008394 | N/A |
| Routed | 66.0% | 53.1% | 5.0 | 4.6 | 7,821.4 ms | $0.00007603 | 83.3% |

BM25 is the strongest low-latency control. The cross-encoder wins retrieval quality but is slow. The router saves cost versus always reranking, but falsely refuses 7/50 supported questions and misses 5/30 adversarial ones.

## Reproduce

Validate existing artifacts:

```bash
make validate-final-evaluation-dataset
make validate-final-benchmark
make validate-failure-analysis
make verify-final-benchmark FINAL_MLFLOW_URI=http://127.0.0.1:5001
```

Rebuild the benchmark in phases:

```bash
make evaluate-final-retrieval
make evaluate-final-routed
make judge-final-answers
make aggregate-final-benchmark
```

`judge-final-answers` sends questions, retrieved excerpts, and generated answers to OpenAI and Gemini and incurs provider usage. MLflow verification needs the tracking server that owns the recorded run IDs.

## Failure review

[`failure_analysis.md`](../reports/failures/failure_analysis.md) contains 15 verified failures across nine categories. Fourteen deterministic cases were promoted to regression data. One host-dependent cold-start latency outlier remains analysis-only.

```bash
make analyze-failures
make test-failure-analysis
```

## Pull-request gate

`make eval-gate` runs a separate five-case offline fixture and enforces nine deterministic thresholds. It is fast regression protection, not the final benchmark.

## Limits

- The corpus is curated technical documentation, not production traffic.
- Relevance labels are incomplete and often identify one chunk.
- The semantic sample is small and model-judged, not human ground truth.
- Cold starts are included; latency is not a load test or SLO.
- Cost is a token projection, not billed spend.
