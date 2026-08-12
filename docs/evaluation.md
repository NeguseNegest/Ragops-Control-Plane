# Evaluation

## Scope

The implemented evaluation stack has two distinct layers:

1. Retrieval evaluation (Days 17–19) compares ranked Qdrant chunk IDs with verified relevance labels and computes Recall@k, MRR, Hit Rate@k, and binary nDCG@k.
2. Generation evaluation (Day 20) generates answers from retrieved evidence, asks an independent provider to score those answers, and requires a manual spot-check of every acceptance record.

Day 20 is an acceptance workflow for 10 answers, not the final benchmark. The larger report and failure analysis belong to Day 21.

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
