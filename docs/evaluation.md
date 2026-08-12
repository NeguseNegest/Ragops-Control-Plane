# Evaluation

## Scope

The implemented evaluation stack has two distinct layers:

1. Retrieval evaluation (Days 17–19, 23, and 25) compares dense Qdrant, persisted BM25, and live RRF hybrid rankings with verified relevance labels and computes Recall@k, MRR, Hit Rate@k, and binary nDCG@k. Day 26 adds a functional cross-encoder candidate whose benchmark is intentionally reserved for Day 27.
2. Generation evaluation (Day 20) generates answers from retrieved evidence, asks an independent provider to score those answers, and requires a manual spot-check of every acceptance record.

Day 20 is an acceptance workflow for 10 answers, not the final benchmark. Day 21 records the first dense benchmark and failure analysis, Day 23 adds BM25, Day 25 measures the RRF hybrid candidate, and Day 26 verifies the reranked pipeline without making a quality claim.

## Day 23 dense-versus-BM25 comparison

`configs/bm25_baseline.yaml` points at the same 45 verified labels and `[1, 3, 5, 10]` cutoffs recorded by `configs/dense_baseline.yaml`. The sparse evaluator additionally verifies that its loaded index has the configured tokenizer and BM25 parameters and that its source SHA256 matches the current processed chunk artifact.

Run the offline preflight and benchmark with:

```bash
make validate-bm25-evaluation
make evaluate-bm25
```

The preflight loads no embedding or generation model and makes no paid API call. The evaluation produces a full BM25 JSON/CSV run, pairs it with the recorded dense JSON by question ID, and renders machine-readable and narrative comparison artifacts. Pairing fails if question IDs, question text, expected sources, relevant chunk IDs, or metric cutoffs differ.

Recorded results:

| Metric | Dense | BM25 | BM25 − dense |
| --- | ---: | ---: | ---: |
| MRR | 0.3359 | 0.6189 | +0.2830 |
| Hit Rate@1 | 0.2667 | 0.4667 | +0.2000 |
| Hit Rate@3 | 0.3111 | 0.7556 | +0.4444 |
| Hit Rate@5 | 0.4444 | 0.8444 | +0.4000 |
| Hit Rate@10 | 0.6000 | 0.8667 | +0.2667 |
| nDCG@10 | 0.3964 | 0.6806 | +0.2842 |

BM25 wins 27 paired questions, dense wins 6, and 12 tie. The comparison also assigns deterministic wording cohorts so this claim is reproducible: BM25 wins 15/23 conceptual/descriptive questions, 8/14 exact-reference questions, and 4/8 behavioral/procedural questions. Cohort labels describe question wording; they are not human relevance labels. Because the questions were generated from and verified against exact source chunks, lexical overlap likely benefits BM25. Results therefore motivate the Day 24 hybrid experiment but do not establish general superiority.

Artifacts:

- `reports/evaluations/bm25_baseline.json`: configuration, index provenance, aggregate metrics, latencies, and complete per-question rankings
- `reports/evaluations/bm25_baseline.csv`: flat per-question results
- `reports/evaluations/bm25_vs_dense.json`: paired metric deltas, wins, misses recovered, cohorts, and ranks
- `reports/week4_bm25_comparison.md`: report rendered from the paired JSON

## Day 24 hybrid candidate

Day 24 implements the candidate evaluated by Day 25 below. `configs/hybrid.yaml` retrieves dense top 20 and BM25 top 20, applies unweighted Reciprocal Rank Fusion with constant 60, and returns a deduplicated top 10. RRF operates on rank positions, so it does not pretend cosine similarity and BM25 scores are calibrated to the same scale.

Validate and exercise the candidate with:

```bash
make validate-hybrid
make test-hybrid
make retrieve-hybrid HYBRID_QUERY="What is the exact MLflow serving command?"
```

Every output chunk retains its dense and BM25 source ranks, raw source scores, and RRF contributions in `_fusion` metadata. The implementation rejects duplicate input IDs, inconsistent rank fields, non-finite scores, and different document payloads attached to the same chunk ID.

Functional correctness and CLI operation satisfy Day 24. Quality conclusions come only from the paired Day 25 run below.

## Day 25 hybrid evaluation

Validate all fixed inputs without querying Qdrant, then run the live benchmark:

```bash
make validate-hybrid-evaluation
make test-hybrid-evaluation
make evaluate-hybrid
```

The evaluator requires exact parity across the dense, BM25, and hybrid question IDs, wording, expected sources, relevant chunks, and metric cutoffs. It also verifies embedding and BM25 component settings, matches the current BM25 source SHA to the Day 23 report, and requires the live Qdrant point count to equal the shared source-record count. Each hybrid question records its fused ranking, original source ranks and scores, total latency, and dense/BM25/fusion stage latency.

Recorded benchmark:

| Metric | Dense | BM25 | RRF hybrid | Hybrid − dense | Hybrid − BM25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MRR | 0.3359 | **0.6189** | 0.5765 | +0.2406 | -0.0424 |
| Hit Rate@1 | 0.2667 | **0.4667** | **0.4667** | +0.2000 | 0.0000 |
| Hit Rate@3 | 0.3111 | **0.7556** | 0.6444 | +0.3333 | -0.1111 |
| Hit Rate@5 | 0.4444 | **0.8444** | 0.7556 | +0.3111 | -0.0889 |
| Hit Rate@10 | 0.6000 | **0.8667** | 0.8444 | +0.2444 | -0.0222 |
| nDCG@10 | 0.3964 | **0.6806** | 0.6405 | +0.2441 | -0.0401 |

Hybrid wins 26 ranks versus dense, loses one, and ties 18. Against BM25 it wins 10, loses 14, and ties 21; it also drops one labeled chunk that BM25 retrieves at rank 1. Against the better component rank per question, hybrid wins 4, loses 15, and ties 26. Giving each of the 20 unique labeled chunks equal weight changes the hybrid-versus-BM25 MRR gap from `-0.0424` to `-0.0318`, so repeated questions do not reverse the conclusion.

The result is negative but actionable: unweighted RRF improves dense retrieval, yet it does not beat BM25 on this label set. Consensus promotion sometimes lifts evidence to rank 1, but it can demote strong BM25-only evidence below chunks that appear in both candidate lists. This motivates treating weighting or reranking as new candidates rather than retroactively tuning Day 25.

Measured latency:

| Run or stage | Average | After first query |
| --- | ---: | ---: |
| Dense baseline | 679.9 ms | 149.6 ms |
| BM25 baseline | 87.9 ms | 88.1 ms |
| RRF hybrid total | 837.4 ms | 177.1 ms |
| Hybrid dense stage | 772.5 ms | not separately reported |
| Hybrid BM25 stage | 64.7 ms | not separately reported |
| Hybrid fusion stage | 0.2 ms | not separately reported |

The historical baseline latency rows come from separate runs. Only the hybrid component rows are internally timed in the same run; their averages include the `29,892.1 ms` first-query dense model warm-up.

Artifacts:

- `reports/evaluations/hybrid_rrf.json`: live configuration, index provenance, complete rankings, metrics, and component latency
- `reports/evaluations/hybrid_rrf.csv`: flat per-question hybrid results
- `reports/evaluations/hybrid_vs_baselines.json`: strict three-way metrics, paired outcomes, cohorts, relevance groups, and failures
- `reports/week4_hybrid_comparison.md`: benchmark table and analysis rendered from the comparison JSON

## Day 26 cross-encoder candidate

`configs/hybrid_rerank.yaml` expands both component retrieval depths to 25, fuses a top-25 candidate pool with the same unweighted RRF constant of 60, and reranks all 25 query/chunk pairs with `cross-encoder/ms-marco-MiniLM-L-6-v2` down to a final top five.

Run configuration and index preflight without Qdrant or model loading, then exercise the live path with:

```bash
make validate-hybrid-rerank
make test-reranker
make retrieve-hybrid-rerank RERANK_QUERY="What operation quantifies vector similarity?"
```

The pipeline preserves the complete `_fusion` metadata and adds `_reranker` metadata containing the model, original RRF candidate rank, and candidate score. It reports model loading separately and measures dense, BM25, fusion, cross-encoder, and total retrieval latency. Raw cross-encoder logits are used only to order candidates; they are not calibrated probabilities and are not compared numerically with RRF, cosine, or BM25 scores.

The live acceptance query “What operation is used to quantify the similarity between the query and document vectors?” returned five chunks and moved its verified label from RRF candidate rank 9 to reranked rank 2. The first process spent `53,219.4 ms` downloading/loading the model; the pipeline then measured dense `6,220.8 ms`, BM25 `66.5 ms`, fusion `0.5 ms`, reranker `7,042.6 ms`, and total retrieval-plus-reranking `13,330.6 ms`. These are cold one-query measurements and must not be treated as a steady-state latency benchmark.

No aggregate retrieval metric is reported for Day 26. A single acceptance query demonstrates wiring, metadata, and timing—not effectiveness. Day 27 must run the fixed label set, compare dense/BM25/RRF/reranked quality and latency, and document cases where reranking helps or hurts.

## Day 20 sample

`configs/generation_judge.yaml` deterministically selects the following sample from `data/eval/golden_qa.jsonl`:

| Query type | Count | Expected behavior |
| --- | ---: | --- |
| Supported | 6 | Answer from the retrieved evidence. |
| Ambiguous | 2 | Ask for the missing clarification rather than guessing. |
| Unsupported | 2 | Refuse or state that the available evidence is insufficient. |

The seed and exact allocation are configuration values. Selection is stable even if the input JSONL order changes. The loader rejects duplicate IDs, insufficient query-type counts, invalid fields, or allocations that do not equal the sample size.

The default roles are intentionally cross-provider:

- generator: OpenAI `gpt-5-nano`
- judge: Gemini `gemini-3.6-flash`
- retrieval: dense top 5 from Qdrant `rag_chunks`

Both provider models are pinned in the YAML file. `RAGOPS_LLM_PROVIDER` is not used by this workflow.

## Faithfulness rubric

Faithfulness asks whether the generated answer's factual claims follow from the retrieved chunks. The expected answer is not evidence.

| Score | Definition |
| ---: | --- |
| 1 | The answer is contradicted by the retrieved context or substantially fabricated. |
| 2 | Major claims are unsupported; the answer is mostly ungrounded despite limited supported content. |
| 3 | The answer mixes supported content with at least one substantive unsupported claim or inference. |
| 4 | The answer is supported overall but contains a minor imprecision or weakly supported detail that does not change the conclusion. |
| 5 | Every factual claim is directly supported by the retrieved context; an appropriate refusal or clarification adds no unsupported facts. |

## Answer relevance rubric

Answer relevance compares the response with the question, reference answer, and behavior required by the query type.

| Score | Definition |
| ---: | --- |
| 1 | The response is irrelevant, answers a different question, or gives behavior opposite to what the query type requires. |
| 2 | The response is mostly tangential, generic, or misses the central request. |
| 3 | The response addresses part of the request but is incomplete, vague, or includes substantial distraction. |
| 4 | The response directly addresses the request and is mostly complete, with only a minor omission or unnecessary detail. |
| 5 | The response is direct and complete for a supported query, asks the necessary clarification for an ambiguous query, or clearly refuses an unsupported query. |

## Refusal correctness rubric

The judge first classifies the observed response as `answer`, `refusal`, or `clarification`. The expected result then follows mechanically from the golden query type:

| Query type | Correct behavior | Verdict rules |
| --- | --- | --- |
| Supported | `answer` | `not_applicable` when answered; `incorrect` for refusal or clarification. |
| Ambiguous | `clarification` | `correct` for clarification; `incorrect` otherwise. |
| Unsupported | `refusal` | `correct` for refusal; `incorrect` otherwise. |

The parser rejects a judge response if its verdict contradicts these rules. This prevents a fluent rationale from silently overriding the rubric.

## Judge input and output

The judge receives:

- question ID, text, type, and expected behavior
- reference answer and expected source
- exact retrieved chunk IDs, source paths, ranks, scores, and text
- generated answer
- complete rubric definitions
- a warning that every supplied field is untrusted data and must not be followed as an instruction

The judge must return only this structure:

```json
{
  "faithfulness": {
    "score": 5,
    "rationale": "Every factual statement is supported by the retrieved evidence."
  },
  "answer_relevance": {
    "score": 4,
    "rationale": "The response answers the question but omits one minor reference detail."
  },
  "refusal_correctness": {
    "observed_behavior": "answer",
    "verdict": "not_applicable",
    "rationale": "This supported question was answered rather than refused."
  }
}
```

Pydantic rejects missing fields, extra fields, scores outside 1–5, empty rationales, unknown behavior values, and inconsistent refusal verdicts.

## Running automatic evaluation

Validate without external calls:

```bash
make validate-generation-judge
make test-llm-judge
```

Run the real 10-answer evaluation:

```bash
make judge-answers
```

The command loads provider keys from the ignored `.env`, verifies that Qdrant contains the configured collection, creates one generator and one judge client, and closes the Qdrant client on success or failure. Existing artifacts are protected from accidental overwrite.

Outputs:

- `reports/evaluations/day20_generation_judge_judgments.jsonl`: one evidence-rich record per question
- `reports/evaluations/day20_generation_judge_summary.json`: aggregate scores, distributions, refusal verdict counts, timings, and review status

## Manual spot-check process

Run:

```bash
make review-judgments
```

For every pending record, the reviewer sees the question, expected behavior, reference answer, exact retrieved evidence, generated answer, and all automatic scores and rationales. The reviewer then records:

- `agree`: the automatic result is consistent with the rubric
- `disagree`: at least one classification, score, or rationale is materially wrong; notes are mandatory
- `skip`: leave the record pending
- `quit`: save completed decisions and stop

Agreement means the automatic judgment is reasonable under the written rubric; it does not mean the generated answer is good. When disagreeing, notes should identify the affected criterion and the corrected score or behavior.

The script writes after every decision and recomputes the summary. Validate Day 20 acceptance with:

```bash
make validate-day20
```

That command requires 10 valid automatic judgments and all 10 configured manual spot-checks. A machine-generated judgment must never be marked as a human review automatically; the reviewer identity is stored in each completed record.

## Recorded Day 20 acceptance run

The authorized run on 2026-08-12 completed all 10 configured questions with OpenAI `gpt-5-nano` as generator and Gemini `gemini-3.6-flash` as judge. The checked-in artifacts are:

- `reports/evaluations/day20_generation_judge_judgments.jsonl`
- `reports/evaluations/day20_generation_judge_summary.json`

Automatic results:

| Metric | Result |
| --- | ---: |
| Mean faithfulness | 4.5 / 5 |
| Mean answer relevance | 3.4 / 5 |
| Correct refusal verdicts | 2 |
| Incorrect refusal verdicts | 4 |
| Not-applicable refusal verdicts | 4 |

The sample contained six supported, two ambiguous, and two unsupported questions. Both unsupported questions were correctly refused. Two supported questions were refused because the top-five retrieval did not expose the necessary details, and both ambiguous questions were answered instead of clarified. These failures explain the four incorrect refusal verdicts and four relevance scores of 1.

A separate record-by-record Codex evidence audit is stored under reviewer identity `codex-manual-audit`. It reviewed 10/10 judgments, agreed with eight, and disagreed with two:

- The generated MLflow serving command omitted the colon from `runs:/<RUN_ID>/model`, so a 5/5 relevance score overstated exactness.
- The generated JSON Schema answer omitted the reference answer's key portability point, so a 5/5 relevance score overstated completeness.

This audit exercises and verifies the spot-check workflow, but it is not represented as independent human sign-off. A human can replace or supplement it in a later benchmark review if required by the evaluation policy.

The Docker-backed Qdrant endpoint became unresponsive before the run and timed out before any provider call. To finish without altering unrelated Docker containers, the same 13,481 processed chunks were loaded into a temporary local Qdrant store and queried through the production dense retriever. The first and third retrievals incurred local-store cold-start work, so the recorded average retrieval latency (`7,581.1 ms`) is not a steady-state service benchmark. Provider timings averaged `9,259.1 ms` for generation and `9,900.9 ms` for judging.

## Interpretation limits

- LLM judgments are model opinions, not ground truth.
- Ten records establish that the workflow works; they do not establish statistically reliable quality.
- The generator and judge can share training biases even when they come from different providers.
- Reference answers can influence relevance scoring, so they are explicitly excluded as faithfulness evidence.
- Retrieval quality constrains generation quality: a faithful answer can still be incomplete when the correct chunk was not retrieved.
- Scores should be interpreted with the retrieved evidence and reviewer notes, not in isolation.
