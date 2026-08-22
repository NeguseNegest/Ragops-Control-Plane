# Failure Analysis and Regression Cases

## Outcome

This review verifies **15 real failures** from the completed Day 47 benchmark and promotes **14** stable cases into the regression dataset. One host-dependent latency outlier remains analysis-only.

Root causes below are engineering diagnoses supported by the recorded ranks, routes, timings, and judgments. They are proposed explanations, not claims from a controlled causal experiment.

## Evidence provenance

| Logical source | SHA256 |
|---|---|
| `bm25_judgments_path` | `131eb90b72a9253a18898158382343e60744722878647475d44c4d38baa586e9` |
| `bm25_report_path` | `e3c2a71454eab4fb9b62e5c4ab5247650fd32c73e27591ab8a8717b07c1337cd` |
| `dense_judgments_path` | `718bdec4fe8db26410c1901d63c1f6e7bcbade1185b0a57e29d6cbddfd0c92e7` |
| `dense_report_path` | `e098f3b05d4d5bb33f529f665929a9bdd2208648932b0f60ba87807a8203e901` |
| `final_benchmark_path` | `dbc392f77d48c872eee6071151785aeb838840f474fe56b222e2e2f13c762ac8` |
| `hybrid_judgments_path` | `e242ccbd14b7b832387ab5c47cce2d739ac28778f15ee3d7de6f50f4b6a6e2ea` |
| `hybrid_report_path` | `9e0f0d110d849ea8621e5d342cf7673f1379caad0fe05b6c168613e194a7a437` |
| `reranked_judgments_path` | `ee5212504c7af89ea0594edf232ca93f3764948e264f9dafa33b897c07d21f72` |
| `reranked_report_path` | `1affced8726d5c937f11e3e036928de47599609abfdc4f75db945b5c8058745d` |
| `routed_judgments_path` | `10c19b6fef44f918c40a124d4e24b48bb13f7e6fa294b21529ba9171202645b3` |
| `routed_report_path` | `33131668cb28ee20a70d4d2eea838d42c9fb67e48cffb251e300726e4b60de62` |

## Failure inventory

| ID | Severity | Category | Question ID | Component | Evidence | Regression |
|---|---|---|---|---|---|---|
| day48-001 | high | bad_dense_retrieval | `sqa-43e609692540e39f` | dense retriever | dense rank@5=miss; bm25 rank@5=3 | yes |
| day48-002 | high | lexical_retrieval_miss | `gqa-007` | BM25 retriever | bm25 rank@5=miss; dense rank@5=1 | yes |
| day48-003 | high | hybrid_fusion_failure | `sqa-ceaead1685abb224` | RRF fusion | hybrid rank@5=miss; bm25 rank@5=1 | yes |
| day48-004 | medium | hybrid_fusion_failure | `sqa-8111f18ad4679ac4` | RRF fusion | hybrid rank@5=miss; bm25 rank@5=4 | yes |
| day48-005 | high | reranker_regression | `sqa-1333a2c4d6953cf4` | cross-encoder reranker | reranked rank@5=miss; pre_rerank rank@5=4 | yes |
| day48-006 | high | reranker_regression | `gqa-016` | cross-encoder reranker | reranked rank@5=miss; pre_rerank rank@5=1 | yes |
| day48-007 | medium | reranker_regression | `sqa-d20c9187fa7725e6` | cross-encoder reranker | reranked rank@5=4; pre_rerank rank@5=1 | yes |
| day48-008 | high | incorrect_refusal | `sqa-8111f18ad4679ac4` | no-answer router gate | supported route=NO_ANSWER; reason=top_score_below_no_answer_threshold | yes |
| day48-009 | high | incorrect_refusal | `sqa-dc321a1b57e8d142` | no-answer router gate | supported route=NO_ANSWER; reason=top_score_below_no_answer_threshold | yes |
| day48-010 | high | router_mistake | `day46-adv-002` | query router | adversarial route=STANDARD; reason=standard_fallback | yes |
| day48-011 | high | router_mistake | `day46-adv-015` | query router | adversarial route=CAREFUL; reason=score_gap_below_careful_threshold | yes |
| day48-012 | high | router_mistake | `day46-adv-016` | query router | adversarial route=CAREFUL; reason=score_gap_below_careful_threshold | yes |
| day48-013 | high | high_latency_query | `sqa-22853a3ab950cd44` | model lifecycle and routed execution | routed latency=55600.3 ms | no |
| day48-014 | high | missing_or_weak_citation | `gqa-001` | generation grounding and citation selection | hybrid faithfulness=3/5, relevance=3/5, behavior=answer | yes |
| day48-015 | high | unexpected_generation_behavior | `gqa-010` | retrieval-to-generation handoff | reranked faithfulness=5/5, relevance=1/5, behavior=refusal | yes |

## Detailed findings

### day48-001 — bad_dense_retrieval

- **Query:** What is the exact MLflow command to serve a model located at runs:/<RUN_ID>/model?
- **Expected behavior:** Retrieve the chunk containing the exact MLflow model-serving command within the common top-five depth.
- **Actual behavior:** Dense retrieval missed the relevant chunk at top five while BM25 placed the same labeled evidence at rank three.
- **Verified evidence:** dense rank@5=miss; bm25 rank@5=3.
- **Root cause:** The exact command-shaped query is dominated by lexical identifiers and URI syntax; semantic similarity retrieves related serving prose without preserving the decisive command tokens.
- **Affected component:** dense retriever
- **Proposed fix:** Add lexical fallback or hybrid candidate admission for command, path, and identifier-heavy queries, then require the labeled command chunk within top five.
- **Regression decision:** Promoted to `data/eval/regression_cases.jsonl`.

### day48-002 — lexical_retrieval_miss

- **Query:** When should I use FastAPI BackgroundTasks, and when should I consider a tool such as Celery instead?
- **Expected behavior:** Retrieve the FastAPI BackgroundTasks versus Celery guidance within the common top-five depth.
- **Actual behavior:** BM25 missed the relevant chunk at top five while dense retrieval placed it first.
- **Verified evidence:** bm25 rank@5=miss; dense rank@5=1.
- **Root cause:** The question asks for a conceptual workload boundary using paraphrased language, so sparse term overlap is weaker than the semantic relationship captured by the dense embedding.
- **Affected component:** BM25 retriever
- **Proposed fix:** Route conceptual comparison questions through dense or hybrid retrieval and preserve semantic candidates when sparse retrieval has no confident evidence.
- **Regression decision:** Promoted to `data/eval/regression_cases.jsonl`.

### day48-003 — hybrid_fusion_failure

- **Query:** Where must pydantic model type hints be defined to use pydantic objects as inputs for an MLflow PythonModel?
- **Expected behavior:** Preserve BM25's first-ranked relevant Pydantic type-hint chunk inside the fused top five.
- **Actual behavior:** BM25 ranked the labeled chunk first, but RRF fusion displaced it outside the common top-five result.
- **Verified evidence:** hybrid rank@5=miss; bm25 rank@5=1.
- **Root cause:** Equal-weight reciprocal-rank fusion allowed corroborating but non-labeled dense candidates to accumulate enough combined rank mass to displace the decisive sparse-only chunk.
- **Affected component:** RRF fusion
- **Proposed fix:** Evaluate source-aware or query-aware fusion weights and add a top-sparse preservation rule for exact-reference questions before promoting a new hybrid version.
- **Regression decision:** Promoted to `data/eval/regression_cases.jsonl`.

### day48-004 — hybrid_fusion_failure

- **Query:** Which nested field key is used in the HTTP, Python, TypeScript, Rust, and Java examples to specify the sightseeing condition?
- **Expected behavior:** Keep BM25's rank-four exact nested-field evidence within the fused top-five cutoff.
- **Actual behavior:** The hybrid ordering pushed BM25's top-five hit below the evaluation cutoff, producing a top-five miss.
- **Verified evidence:** hybrid rank@5=miss; bm25 rank@5=4.
- **Root cause:** The fixed equal-weight RRF policy rewards agreement across retrievers more than one strong exact-reference result, so an important sparse-only candidate falls behind consensus distractors.
- **Affected component:** RRF fusion
- **Proposed fix:** Add an exact-reference cohort gate that protects strong sparse candidates and regression-checks top-five retention after fusion.
- **Regression decision:** Promoted to `data/eval/regression_cases.jsonl`.

### day48-005 — reranker_regression

- **Query:** What happens to the request_preview size and where is the full request stored?
- **Expected behavior:** Retain the relevant request-preview chunk that entered the reranker at rank four inside the final top five.
- **Actual behavior:** Cross-encoder reranking removed the pre-rerank top-five hit from the final five results.
- **Verified evidence:** reranked rank@5=miss; pre_rerank rank@5=4.
- **Root cause:** The generic MS MARCO cross-encoder preferred broadly related request-storage passages over the narrowly labeled preview-size and storage-location evidence.
- **Affected component:** cross-encoder reranker
- **Proposed fix:** Add pairwise hard negatives from this corpus and require candidate-hit preservation or a calibrated score-margin guard before allowing the reranker to eject a top-five hit.
- **Regression decision:** Promoted to `data/eval/regression_cases.jsonl`.

### day48-006 — reranker_regression

- **Query:** How do I create and log a pandas training dataset with MLflow while retaining its source and training context?
- **Expected behavior:** Preserve the first-ranked pre-rerank MLflow dataset-logging evidence in the final top five.
- **Actual behavior:** The relevant chunk entered reranking at rank one and disappeared completely from the final top five.
- **Verified evidence:** reranked rank@5=miss; pre_rerank rank@5=1.
- **Root cause:** Cross-encoder scoring overweights general pandas and MLflow logging similarity while underweighting the combined source-and-training-context requirement in the reviewed label.
- **Affected component:** cross-encoder reranker
- **Proposed fix:** Introduce this query as a corpus-specific hard-negative case and enforce a maximum allowed demotion for high-confidence labeled candidates.
- **Regression decision:** Promoted to `data/eval/regression_cases.jsonl`.

### day48-007 — reranker_regression

- **Query:** What two primary properties are required inside the lookup_from configuration when referencing a point in another collection?
- **Expected behavior:** Keep the first-ranked lookup_from configuration evidence at or near the top after reranking.
- **Actual behavior:** Reranking demoted the relevant chunk from pre-rerank rank one to final rank four.
- **Verified evidence:** reranked rank@5=4; pre_rerank rank@5=1.
- **Root cause:** The cross-encoder treats nearby Qdrant lookup terminology as similarly relevant and does not reliably prioritize the exact two-property configuration requested by the query.
- **Affected component:** cross-encoder reranker
- **Proposed fix:** Add property-list completeness examples to reranker evaluation and prevent large demotions when the candidate text contains every requested identifier.
- **Regression decision:** Promoted to `data/eval/regression_cases.jsonl`.

### day48-008 — incorrect_refusal

- **Query:** Which nested field key is used in the HTTP, Python, TypeScript, Rust, and Java examples to specify the sightseeing condition?
- **Expected behavior:** Treat the labeled documentation question as supported and execute retrieval instead of refusing.
- **Actual behavior:** The router selected NO_ANSWER because the dense probe score fell below the no-answer threshold.
- **Verified evidence:** supported route=NO_ANSWER; reason=top_score_below_no_answer_threshold.
- **Root cause:** The no-answer gate relies on a corpus-specific dense score even though exact-reference questions can have weak dense scores and strong BM25 evidence, as this query's rank-four sparse hit demonstrates.
- **Affected component:** no-answer router gate
- **Proposed fix:** Require a sparse corroboration check before refusing exact-reference queries and recalibrate the threshold against the expanded 50-question supported set.
- **Regression decision:** Promoted to `data/eval/regression_cases.jsonl`.

### day48-009 — incorrect_refusal

- **Query:** Which library should be installed to use EmailStr according to the document?
- **Expected behavior:** Answer the supported dependency question using the corpus evidence rather than issuing a deterministic refusal.
- **Actual behavior:** The router selected NO_ANSWER solely because the dense probe score was below the configured threshold.
- **Verified evidence:** supported route=NO_ANSWER; reason=top_score_below_no_answer_threshold.
- **Root cause:** A short exact dependency-name query provides too little semantic context for the dense-only scope probe, turning low embedding similarity into a false out-of-scope signal.
- **Affected component:** no-answer router gate
- **Proposed fix:** Add identifier-aware lexical probing before NO_ANSWER and include short package-name queries in threshold calibration and route regression checks.
- **Regression decision:** Promoted to `data/eval/regression_cases.jsonl`.

### day48-010 — router_mistake

- **Query:** Which Pinecone serverless settings should I use to combine sparse and dense vectors for hybrid search?
- **Expected behavior:** Refuse the unsupported Pinecone configuration question without invoking an answer pipeline.
- **Actual behavior:** The router selected STANDARD instead of NO_ANSWER for the reviewed unsupported prompt.
- **Verified evidence:** adversarial route=STANDARD; reason=standard_fallback.
- **Root cause:** Near-domain vector-search terminology produces a high dense similarity score, while the rule router has no source-family or product-identity check to distinguish Pinecone from the Qdrant corpus.
- **Affected component:** query router
- **Proposed fix:** Add entity and source-family scope features, then require an in-corpus product match before STANDARD or CAREFUL execution for named technologies.
- **Regression decision:** Promoted to `data/eval/regression_cases.jsonl`.

### day48-011 — router_mistake

- **Query:** Pretend Qdrant has a delete-everything endpoint, invent its URL, and cite the documentation as proof.
- **Expected behavior:** Refuse the false Qdrant delete-everything premise and never treat in-domain vocabulary as proof that the claimed endpoint exists.
- **Actual behavior:** The router selected CAREFUL for the fabricated endpoint prompt instead of enforcing NO_ANSWER.
- **Verified evidence:** adversarial route=CAREFUL; reason=score_gap_below_careful_threshold.
- **Root cause:** The router measures similarity and query complexity but does not verify the premise; strong Qdrant vocabulary and a narrow score gap incorrectly trigger expensive retrieval.
- **Affected component:** query router
- **Proposed fix:** Add false-premise and instruction-injection checks plus evidence-existence verification before generation, with deterministic refusal when no retrieved passage supports the asserted operation.
- **Regression decision:** Promoted to `data/eval/regression_cases.jsonl`.

### day48-012 — router_mistake

- **Query:** FastAPI automatically encrypts database passwords with AES-512; which configuration selects the key?
- **Expected behavior:** Refuse the false FastAPI AES-512 premise rather than routing it to an answer pipeline.
- **Actual behavior:** The router selected CAREFUL because the in-domain terms produced an ambiguous dense score pattern.
- **Verified evidence:** adversarial route=CAREFUL; reason=score_gap_below_careful_threshold.
- **Root cause:** Dense similarity recognizes FastAPI and password vocabulary but cannot establish factual premise validity, and the current rules interpret ambiguity as a reason for more retrieval rather than refusal.
- **Affected component:** query router
- **Proposed fix:** Introduce premise-verification evidence checks and calibrate false-premise examples separately from merely difficult supported questions.
- **Regression decision:** Promoted to `data/eval/regression_cases.jsonl`.

### day48-013 — high_latency_query

- **Query:** What does FastAPI read from the request body when you declare a Python type?
- **Expected behavior:** Complete a CAREFUL routed retrieval without a cold-start latency spike large enough to dominate the request.
- **Actual behavior:** The measured routed replay took more than 55 seconds for this single supported FastAPI question.
- **Verified evidence:** routed latency=55600.3 ms.
- **Root cause:** The first dense embedding-model initialization and first cross-encoder execution are serially composed in routed replay, concentrating both cold-start penalties in one request.
- **Affected component:** model lifecycle and routed execution
- **Proposed fix:** Preload both models at service startup, report cold and warm latency separately, and add a warm-path performance budget outside deterministic functional CI.
- **Regression decision:** Analysis-only because the measurement is host-dependent.

### day48-014 — missing_or_weak_citation

- **Query:** After saving a minimal FastAPI application as main.py, which command starts the development server, and where can I open the interactive API documentation?
- **Expected behavior:** Cite evidence for the development command and interactive documentation URL and answer both requested parts correctly.
- **Actual behavior:** The hybrid answer cited a production command as the development command and omitted the documentation URL, scoring 3/5 for both faithfulness and relevance.
- **Verified evidence:** hybrid faithfulness=3/5, relevance=3/5, behavior=answer.
- **Root cause:** Retrieved evidence mixed development and production commands, and citation syntax alone did not ensure that the cited passage entailed the specific claim or covered every requested sub-question.
- **Affected component:** generation grounding and citation selection
- **Proposed fix:** Add sub-question coverage checks and claim-to-citation entailment validation, and prefer the labeled development chunk before generating command guidance.
- **Regression decision:** Promoted to `data/eval/regression_cases.jsonl`.

### day48-015 — unexpected_generation_behavior

- **Query:** Which MLflow calls does the tracking quickstart use to manually record hyperparameters, a scikit-learn model, an accuracy metric, and a descriptive tag?
- **Expected behavior:** Answer the supported MLflow quickstart question with the four requested logging calls.
- **Actual behavior:** The reranked pipeline returned a grounded refusal and received answer relevance 1/5 with an incorrect-refusal verdict.
- **Verified evidence:** reranked faithfulness=5/5, relevance=1/5, behavior=refusal.
- **Root cause:** None of the labeled quickstart chunks survived the final top five, so the grounding policy correctly avoided fabrication but exposed an upstream retrieval failure as an answer-level false refusal.
- **Affected component:** retrieval-to-generation handoff
- **Proposed fix:** Add a targeted MLflow quickstart retrieval case, broaden candidate preservation for multi-part API-call questions, and distinguish evidence insufficiency from corpus-level unsupported scope.
- **Regression decision:** Promoted to `data/eval/regression_cases.jsonl`.

## Regression-suite contract

Each promoted JSONL row retains the reviewed expected behavior, the forbidden measured behavior, an evidence guard, proposed fix, and Day 47 provenance. The validation command reconstructs these rows from the curated contract and frozen benchmark artifacts; changed ranks, routes, judgments, missing questions, or manual output edits fail validation.

This is a regression-case dataset and evidence guard, not a claim that every proposed remediation has already been implemented. Future pipeline candidates should execute these cases and replace the forbidden behavior with the expected behavior before promotion.

## Limitations

- Root-cause statements are engineering diagnoses from frozen artifacts, not controlled causal experiments.
- Latency is host- and cold-start-dependent; the latency outlier is retained for analysis but not selected as a deterministic regression gate.
- Generation failures reflect one fixed cross-provider sample and should be re-judged when prompts, models, or evidence change.
