# No-Answer and Refusal Behavior

## Scope

Day 39 turns the Day 38 `NO_ANSWER` decision into an actual refusal and measures the safety/coverage tradeoff. It does not automatically dispatch FAST, STANDARD, or CAREFUL through `POST /query`; that endpoint remains explicitly config-selected.

The no-answer path is deliberately deterministic. When `POST /route` selects `NO_ANSWER`, the API returns:

```text
I do not know based on the available FastAPI, MLflow, and Qdrant documentation.
```

It does not call the configured template/OpenAI/Gemini client, make factual claims, attach citations, or mark probe chunks as generation evidence. The response includes `prompt_version=no_answer_v1`, a SHA256 of the internal prompt contract, and `generated_by=deterministic_policy`. The prompt treats the question as untrusted data, forbids answering/inference/citations, and requires the exact refusal.

## Threshold Calibration

The original provisional threshold was `top_score < 0.25`. A live check showed that it refused only one of the 12 reviewed unsupported examples, so it did not meet Day 39's purpose.

`configs/no_answer.yaml` now defines a reproducible safety-first method:

1. Use the five pre-existing golden unsupported questions only for calibration.
2. Find their maximum dense top score: `0.5302763`.
3. Add the configured `0.0005` margin.
4. Round upward to three decimal places.
5. Require `configs/routed.yaml` to contain the resulting strict threshold, `top_score < 0.531`.

A score exactly equal to `0.531` is not refused. Because router schema v1 requires ordered score bands, the CAREFUL top-score threshold moves from `0.50` to `0.56`; all other CAREFUL and FAST rules remain unchanged.

This threshold is tied to the current corpus, MiniLM embedding model, Qdrant index, and query distribution. Cosine score is not a probability.

## Evaluation Set

`data/eval/no_answer_queries.jsonl` contains 12 reviewed refusal examples:

| Split | Count | Source | Purpose |
| --- | ---: | --- | --- |
| Calibration | 5 | Existing `gqa-031` through `gqa-035` golden unsupported questions | Select threshold only |
| Evaluation | 7 | New Day 39 held-out questions | Measure generalization after threshold selection |

The held-out set includes near-domain hard negatives involving Flask, Weaviate, Cassandra, React/Redux, Terraform/AWS, and Kafka, plus a high-stakes acetaminophen question. The calibration rows are cross-checked byte-for-byte against their unsupported golden records. Every row has a fixed split, category, refusal expectation, provenance, and reviewer identity.

False-refusal measurement replays the immutable top-two scores for all 45 supported questions in `reports/evaluations/dense_baseline.json`. This avoids rerunning supported retrieval merely to measure the new threshold and ensures comparisons use the same evidence that informed the router design.

## Recorded Results

The live evaluation against the local 13,481-chunk Qdrant corpus produced `reports/evaluations/no_answer.json` and `.csv`. The artifacts retain the evaluation/router/probe identities and, for every refused row, the exact answer, prompt SHA256, and deterministic generator identity:

| Metric | Result |
| --- | ---: |
| Calibration unsupported refused | 5 / 5 |
| Held-out unsupported refused | 7 / 7 |
| Overall unsupported refusal accuracy | 12 / 12 = 100% |
| Supported questions answered | 36 / 45 = 80% |
| Supported false refusals | 9 / 45 = 20% |
| Refusal precision | 12 / 21 = 57.14% |
| Balanced accuracy | 90% |
| Overall accuracy | 48 / 57 = 84.21% |

All configured acceptance checks pass: unsupported and held-out refusal accuracy are 100%, supported answer rate is at least 80%, and refusal precision is at least 55%.

The result meets the Day 39 acceptance criterion on the checked-in examples, but the 20% supported false-refusal rate is material. The router stays `draft`; this is a conservative hallucination-reduction tradeoff, not a production-ready operating point.

## API Contract

`POST /route` still returns the Day 38 decision, exact features, minimal probe evidence, and timings. Day 39 adds a nullable `refusal` field.

- FAST/STANDARD/CAREFUL: `refusal` is null.
- NO_ANSWER: `refusal` contains the exact deterministic answer, prompt version/hash, and policy generator identity.

The endpoint remains untraced because it does not execute a final `/query` pipeline. Probe chunks are diagnostic only and are never exposed as document text. `POST /query` remains unchanged and does not yet consume route decisions.

## Commands

Validate config, split counts, golden provenance, supported report identity, prompt identity, and router threshold without querying Qdrant:

```bash
make validate-no-answer
```

Run the 12 live unsupported probes, replay the supported report, require every acceptance check, and atomically replace JSON/CSV artifacts:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 make evaluate-no-answer
```

Run the focused no-answer, router, and API tests:

```bash
make test-no-answer
```

## Interpretation Limits

- Twelve unsupported examples are too few for a population claim or confidence interval.
- Five examples select the threshold; only seven are held out.
- The examples are manually authored and do not cover adversarial paraphrases, multilingual questions, prompt injection, or every out-of-domain technology.
- Raw top score cannot distinguish all evidence adequacy failures. A stronger future policy should evaluate semantic scope and evidence entailment rather than continually raising a cosine threshold.
- The 45 supported rows are source-derived questions, not an unbiased production traffic sample.
- Automatic routed execution, routing trace persistence, route-level latency/cost comparison, and threshold hardening remain Days 41–42 work.
