# Week 3 Dense Retrieval Baseline

## Executive summary

The first measured RAGOps baseline evaluated dense retrieval over 45 verified questions and a real 13,481-chunk Qdrant index. The pipeline achieved MRR `0.3359`, Recall@5 `0.4444`, and Recall@10 `0.6000`. Twelve questions placed the labeled chunk first, while 18 did not retrieve it anywhere in the top 10.

This establishes a reproducible reference point, but it is not yet strong enough to treat dense retrieval as a promotion candidate. At the API's default depth of five chunks, the labeled evidence is available for only 20 of 45 questions. The failures support the planned next comparisons with BM25, hybrid fusion, and reranking.

This Day 21 report analyzes the existing real Day 19 artifacts rather than rerunning Qdrant or either paid generation provider:

- [Dense baseline JSON](evaluations/dense_baseline.json)
- [Dense baseline CSV](evaluations/dense_baseline.csv)
- [Retrieval labels](../data/eval/retrieval_labels.jsonl)
- [Dense baseline configuration](../configs/dense_baseline.yaml)

## Evaluation setup

| Setting | Value |
| --- | --- |
| Retriever | Dense cosine-similarity search |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | Qdrant collection `rag_chunks` |
| Indexed chunks | 13,481 |
| Evaluation questions | 45 supported questions |
| Relevant judgments | 45 verified chunk IDs, one per question |
| Retrieval depth | Top 10 |
| Metric cutoffs | 1, 3, 5, and 10 |
| Corpus sources | FastAPI, MLflow, and Qdrant documentation |

Every question in this evaluation has exactly one labeled relevant chunk. Recall@k and Hit Rate@k therefore have identical values in this run, although they are different metrics and will diverge once questions have multiple relevant chunks.

## Aggregate results

| Cutoff | Recall@k | Hit Rate@k | nDCG@k |
| ---: | ---: | ---: | ---: |
| 1 | 0.2667 | 0.2667 | 0.2667 |
| 3 | 0.3111 | 0.3111 | 0.2918 |
| 5 | 0.4444 | 0.4444 | 0.3473 |
| 10 | 0.6000 | 0.6000 | 0.3964 |

| Summary metric | Result |
| --- | ---: |
| MRR | 0.3359 |
| Questions with the relevant chunk at rank 1 | 12 / 45 |
| Questions with the relevant chunk at ranks 2–3 | 2 / 45 |
| Questions with the relevant chunk at ranks 4–5 | 6 / 45 |
| Questions with the relevant chunk at ranks 6–10 | 7 / 45 |
| Questions missing the relevant chunk in the top 10 | 18 / 45 |

Increasing retrieval depth from one to five recovers eight additional questions, and increasing it from five to ten recovers another seven. That gain comes with more generation context and does not solve the 18 complete top-10 misses.

## Results by corpus source

| Source | Questions | MRR | Hit@1 | Hit@5 | Hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FastAPI | 17 | 0.3775 | 0.2941 | 0.5294 | 0.6471 |
| MLflow | 14 | 0.3284 | 0.2857 | 0.3571 | 0.5714 |
| Qdrant | 14 | 0.2929 | 0.2143 | 0.4286 | 0.5714 |

FastAPI performs best on this sample, while Qdrant has the lowest MRR. These are descriptive slices, not statistically reliable source rankings: each source has only 14–17 questions, and the Qdrant corpus is one large aggregated text file rather than a collection of individual pages.

## Latency

| Measurement | Latency |
| --- | ---: |
| Mean across all 45 queries | 679.9 ms |
| Median | 117.8 ms |
| First-query cold start | 24,009.5 ms |
| Mean across the remaining 44 queries | 149.6 ms |

The first query loaded the embedding model and dominates the overall mean. The 149.6 ms warm-query average is a more useful local reference, but neither value is a service-level guarantee. The run did not measure concurrency, network variability, sustained throughput, or tail latency in a deployed environment.

## Failure analysis

### 1. Generic request-body language displaced the exact FastAPI result

- Question `sqa-22853a3ab950cd44`: “What does FastAPI read from the request body when you declare a Python type?”
- Labeled chunk: `cb4cd8b6-6c0d-535e-ac26-f6b00cfe2335` from `fastapi/docs/tutorial/body.md`.
- Outcome: labeled chunk absent from the top 10.
- Top result: `a777bdaf-3ec5-54df-988e-0ae68239000b` from `fastapi/docs/advanced/strict-content-type.md`, score `0.7482`.

The top result discusses parsing a request body as JSON when `Content-Type` is absent, while the labeled chunk explicitly says that a Python type declaration makes FastAPI read the request body as JSON, convert types, and validate the data. Dense similarity captured the broad request-body concept but did not preserve the question's declaration-specific intent.

### 2. Nearby MLflow serving prose outranked the exact command

- Question `sqa-43e609692540e39f`: “What is the exact MLflow command to serve a model located at `runs:/<RUN_ID>/model`?”
- Labeled chunk: `c62d01b9-14fc-5f93-aa8e-540cd36c9b37` from `mlflow/docs/docs/quickstart_drilldown/index.mdx`.
- Outcome: labeled chunk absent from the top 10.
- Top result: `f1ed4936-e9ee-5997-8b19-7bf8b8311aca` from the model-registry workflow, score `0.8190`.

The labeled evidence contains the exact `mlflow models serve` command. The top result has a highly similar “Serving an MLflow Model” heading but ends immediately after the opening shell block and does not contain the requested command. This is a strong candidate for sparse or hybrid retrieval because exact command tokens and punctuation carry more value than general semantic proximity.

### 3. General vector-similarity content displaced the exact operation

- Question `sqa-5c5e20dbf84c3f0b`: “What operation is used to quantify the similarity between the query and document vectors?”
- Labeled chunk: `ee5a3f6a-d8d9-560f-bd84-d0649f2d8a4d` from `qdrant/qdrant_llms_full.txt`.
- Outcome: labeled chunk absent from the top 10.
- Top result: `58cc0e54-4542-5d3e-a152-85b9ee37794c`, score `0.7576`.

The labeled chunk gives the dot-product formula. The top result defines vector similarity in general terms but does not identify the operation. The query is short and semantically broad, so multiple conceptual chunks look relevant to the embedding model. Hybrid lexical evidence and reranking should be tested against this failure class.

### 4. Some evidence is recoverable only at a deeper cutoff

- Question `sqa-7f4b9c901bcac8d2`: “What file format does MLflow use for saving models from a variety of tools?”
- Labeled chunk: `c62d01b9-14fc-5f93-aa8e-540cd36c9b37`.
- Outcome: rank 9; missed at the API's default top five but recovered at top 10.

This example helps explain the 15.6 percentage-point gain from Hit@5 to Hit@10. Simply increasing `top_k` can recover some evidence, but it also expands the generator's context and cannot address complete top-10 misses.

## Downstream generation evidence

The separate Day 20 acceptance sample provides limited supporting evidence that retrieval quality affects answer behavior. Across 10 generated answers, the judge reported mean faithfulness `4.5/5` and mean answer relevance `3.4/5`. Both unsupported questions were correctly refused, but two supported questions were also refused because their top-five context did not expose the required detail; both ambiguous questions were answered instead of clarified.

These generation results are supplementary rather than a second full benchmark. The sample contains only 10 questions, uses one generator and one judge, and its manual audit was performed by Codex rather than an independent human reviewer. See the [Day 20 summary](evaluations/day20_generation_judge_summary.json) and [evaluation documentation](../docs/evaluation.md).

## Limitations

- The 45 retrieval labels cover only supported questions and were bootstrapped from approved synthetic examples.
- Each question has one labeled relevant chunk, so the evaluation can penalize other genuinely useful chunks that were not labeled.
- The three corpus sources are not represented by equal document structures; Qdrant is ingested as one aggregated text file.
- There is no BM25, hybrid, or reranked comparison yet, so this report measures one pipeline rather than a winner among alternatives.
- Latency comes from one local sequential run and includes a large embedding-model cold start.
- The source snapshots, model package versions, and local machine state can affect reproducibility beyond the configuration captured in the report artifact.
- The Day 20 generation results are a small model-judged acceptance sample, not a statistically robust generation benchmark.

## Decision and next experiments

Day 21's acceptance criterion is satisfied: the project now has a documented, measured baseline with reproducible artifacts and concrete failure examples. No promotion threshold exists yet, so this report does not issue a production promotion decision.

The next experiments should use this dense run as the fixed reference:

1. Evaluate BM25 on the same 45 labels, emphasizing commands, identifiers, endpoints, and other exact-token questions.
2. Combine sparse and dense rankings with Reciprocal Rank Fusion and compare the complete metric table.
3. Test a cross-encoder reranker on the hybrid candidate pool.
4. Expand the relevance judgments to include all valid evidence chunks for each question.
5. Separate model-loading cold start from steady-state retrieval in future latency reports.
