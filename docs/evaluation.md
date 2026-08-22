# Evaluation

## Scope

The implemented evaluation stack has six distinct layers:

1. Retrieval evaluation (Days 17–19, 23, 25, 27, the Day 28 refactor, Day 29 tracking, and the Day 30 registry) compares dense Qdrant, persisted BM25, live RRF hybrid, and cross-encoder-reranked rankings with verified relevance labels; computes Recall@k, depth-bounded MRR, Hit Rate@k, and binary nDCG@k; records the runs in MLflow; and binds those evidence bundles to versioned pipeline identities.
2. Generation evaluation (Day 20) generates answers from retrieved evidence, asks an independent provider to score those answers, and requires a manual spot-check of every acceptance record.
3. Refusal evaluation (Day 39) calibrates the NO_ANSWER score threshold from one unsupported split, measures it on a held-out unsupported split, replays supported-query scores to expose false refusals, and requires explicit accuracy/precision checks before writing its report.
4. Router evaluation (Day 41) replays the same supported questions through always-FAST, always-CAREFUL, and routed strategies, combines that evidence with reviewed unsupported refusal outcomes, and compares quality, serially composed retrieval latency, and a controlled Day 40 generation-cost projection.
5. Router stabilization (Day 42) evaluates a predeclared CAREFUL-gap grid on a deterministic tuning/validation split, enforces refusal/validation/latency/cost constraints, and publishes a complete route distribution and transition audit.
6. The automated evaluation gate (Day 44) executes a SHA-pinned five-case regression suite through the real dense retriever, rule router, and template citation path; calculates deterministic quality, refusal, latency, and error metrics; and returns a shell-enforceable pass/fail decision.

Day 20 is an acceptance workflow for 10 answers, not the final benchmark. Day 21 records the first dense benchmark and failure analysis, Day 23 adds BM25, Day 25 measures the RRF hybrid candidate, Day 26 verifies the reranked pipeline, Day 27 measures its quality and latency tradeoff, Day 28 moves every retrieval candidate behind the same config-driven interface, Day 29 tracks all four retrieval runs, Day 30 registers their versions and promotion roles without changing their measurements, Day 35 proves the dense results survive the complete online HTTP and tracing composition, Day 39 adds measured no-answer behavior without claiming that a small authored sample is production calibration, Day 41 makes the router's actual tradeoff explicit, Day 42 hardens and explains that policy without promoting it, and Day 44 adds fast local regression enforcement without replacing the larger Day 47 benchmark.

## Day 44 automated evaluation gate

`configs/eval_gate.yaml` is the complete gate contract. It pins `dense_baseline@1.0.0`, `rule_router@0.2.0`, the existing four-record CI corpus, and `tests/fixtures/eval_gate_cases.jsonl` by SHA256. The separate gate dataset contains three supported questions with exact relevant chunk IDs and two unsupported questions with required refusal behavior. Strict models reject unknown fields, stale hashes, duplicate identities/questions/relevance labels, invalid behavior/query-type combinations, missing corpus references, non-finite or zero vectors, dimension drift, pipeline identity/status drift, collection mismatch, and incoherent thresholds before execution.

The current compact candidate is the executable dense pipeline. That means “candidate” here is the pipeline selected for this regression run, not the pipeline-registry `candidate` alias. The cross-encoder alias requires the full BM25 artifact and model stack and remains part of the final comparative benchmark rather than being replaced by a fake CI reranker. The gate instantiates the production `DenseRetriever` through the common factory, seeds real in-memory Qdrant, injects only the checked deterministic query vectors, applies the actual router, and uses the production template citation builder. It requires no Docker service, external provider, API key, downloaded model, mutable trace database, or ignored full-corpus artifact.

Five cases produce these metrics and thresholds:

| Check | Threshold | Interpretation |
| --- | ---: | --- |
| Recall@2 | `>= 1.0` | Every supported case retains its labeled evidence within the compact cutoff. |
| Recall regression | `<= 0.0` from the `1.0` fixture baseline | The checked deterministic baseline cannot silently degrade. |
| MRR | `>= 1.0` | Every supported relevant chunk remains rank one. |
| Answer presence | `>= 1.0` | Every supported case reaches generation and returns a non-empty template answer. |
| Citation coverage | `>= 1.0` | Every labeled relevant chunk is referenced by a citation ID actually present in the answer. |
| Citation precision | `>= 1.0` | Every answer-referenced citation chunk is relevant under the compact labels. |
| Refusal correctness | `>= 1.0` | Supported cases answer and unsupported cases select `NO_ANSWER`. |
| Whole-case p95 latency | `<= 100 ms` | The in-memory smoke path remains bounded with wide cold-runner headroom. |
| Runtime errors | `<= 0` | Any per-case exception fails the gate while remaining visible in its report. |

The p95 calculation uses nearest rank across all five whole-case measurements. Ten separate cold-process local runs observed p95 values from `1.074` to `1.272 ms`; the config records the maximum `1.272 ms` observation and the reason for the wider `100 ms` ceiling. This number measures the compact deterministic path, not full-corpus or cross-encoder latency.

Generation metrics are limited to deterministic evidence available offline. The template path can prove that an answer exists and that its cited IDs map to labeled evidence; it cannot establish semantic faithfulness. The report therefore emits `faithfulness: null` with `not_available_without_external_judge`. Day 20/47 judge evidence remains the appropriate source for semantic generation quality.

Run the gate and its focused tests with:

```bash
make eval-gate
make test-eval-gate
```

The CLI prints all nine comparisons and returns `0` only when every check passes, `1` for a measured threshold failure, and `2` when configuration or execution setup prevents a valid run. The acceptance regression deliberately cycles the supported deterministic embeddings through the real dense retriever. Recall, recall-regression, MRR, and citation checks fail, and the report maps that outcome to exit status `1`. A separate test injects `300 ms` whole-case measurements and proves the latency check fails without masking perfect quality; another proves a per-case retrieval exception increments the error count and fails the gate. Day 45 invokes the focused smoke suite and this executable gate as independent GitHub Actions jobs.

## Day 39 refusal evaluation

`configs/no_answer.yaml` defines the dataset identities, calibration rule, exact refusal text/version, acceptance thresholds, router path, and output artifacts. Five pre-existing unsupported golden questions form the calibration split; seven newly reviewed near-domain/high-stakes questions form the held-out evaluation split. Calibration takes the maximum unsupported top score, adds `0.0005`, and rounds upward to three decimal places, yielding the strict router rule `top_score < 0.531`.

The live evaluator sends all 12 unsupported questions through the real configured dense probe and router. It separately replays the top-two scores from all 45 supported rows in the immutable dense report, so false-refusal measurement uses fixed, reviewable evidence. It then checks unsupported refusal accuracy, held-out refusal accuracy, supported answer rate, and refusal precision before atomically replacing `reports/evaluations/no_answer.json` and `.csv`.

The recorded run refused 12/12 unsupported questions and 7/7 held-out questions. It answered 36/45 supported questions, producing 9 false refusals; refusal precision was 57.14%, balanced accuracy 90%, and overall accuracy 84.21%. These checks meet the configured Day 39 acceptance boundary, but the 20% supported false-refusal rate is material and keeps the policy in `draft` status. Full methodology, commands, and interpretation limits are in [`no_answer.md`](no_answer.md).

## Day 40 cost evidence

Day 40 does not add a quality benchmark. It makes each successful query's generation-cost evidence explicit and durable so later router comparisons can aggregate comparable records. The response and trace retain exact provider/model, token counts/source/estimator, rate source/table identity, rates, currency, status, and amount. Provider usage takes precedence over heuristic counts, environment rates take precedence over the checked-in exact-model table, and unavailable evidence never becomes zero.

The live API evaluator now requires response/trace cost parity in addition to its earlier answer, timing, ranking, and chunk checks. The historical Day 35 artifact predates schema v4 and is not retroactively rewritten; new evaluations exercise the strengthened contract. See [`cost_estimation.md`](cost_estimation.md) for the formula and limits.

## Day 41 router comparison

`configs/router_evaluation.yaml` pins every input, comparison definition, latency composition rule, cost model, expected dataset count, and output path for `router_comparison@0.2.0`. The evaluator validates exact question/source/relevance parity across the 45 verified labels, dense top-10 report, reranked top-5 report, golden reference answers, and Day 39 refusal report. It recomputes every router decision from the current `rule_router@0.2.0` features and rejects stale route/reason evidence. It also requires all selected chunk IDs in the processed artifact and an exact provider/model match in the Day 40 table.

The three strategies are paired per supported question:

- always FAST uses the dense ranking at depth two;
- always CAREFUL uses the hybrid-plus-cross-encoder ranking at depth five; and
- routed uses dense top two for FAST, dense top ten for STANDARD, reranked top five for CAREFUL, and no ranking/provider generation for NO_ANSWER.

Supported quality reports MRR, Recall@5, Hit@5, and binary nDCG@5. Policy quality uses all 57 questions: fixed strategies always answer, while routed uses the recorded/recomputed refusal. The explicitly named combined proxy counts a supported question only when a relevant chunk is available in the final ranking and an unsupported question only when it is correctly refused. It is not answer correctness or an LLM-judge score.

Latency is `measured_artifact_serial_replay`: always FAST uses the recorded dense latency and always CAREFUL uses the recorded reranked latency. Every routed question pays the dense probe proxy; FAST reuses it, NO_ANSWER stops, STANDARD adds a second dense measurement, and CAREFUL adds the reranked measurement. The dense report measured top ten rather than top two, so it is a conservative probe proxy. Cold starts remain included and the result is not described as a simultaneous live benchmark.

Cost uses `generation_model_costs@1.0.0` and the Day 40 `utf8_bytes_div4_ceiling_v1` estimator over the exact prompt built from each selected chunk plus that question's verified reference-answer text. This controls answer length across strategies without a paid provider call. Routed NO_ANSWER rows record deterministic zero provider cost. The projection is comparable within this report but is not observed usage or an invoice.

| Strategy | Supported Hit@5 | MRR | Unsupported refusal | Combined proxy | Avg replay latency | Projected total cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always FAST | 28.89% | 0.2778 | 0% | 22.81% | 679.9 ms | $0.00145935 |
| Always CAREFUL | 84.44% | 0.6889 | 0% | 66.67% | 4681.6 ms | $0.00355245 |
| Routed v0.2.0 | 64.44% | 0.5267 | 100% | 71.93% | 3345.6 ms | $0.00312740 |

Routed improves Hit@5 by 35.56 percentage points and the combined proxy by 49.12 points over always FAST, at 392.10% higher average replay latency and 114.30% higher projected cost. Against always CAREFUL, routed cuts average replay latency by 28.54% and projected cost by 11.96%, but loses 20 Hit@5 points. Its supported distribution is 2 FAST, 11 STANDARD, 23 CAREFUL, and 9 NO_ANSWER. Because those nine are false refusals and broader evidence is still missing, the recorded decision is `keep_router_draft`.

Run `make validate-router-evaluation` to recompute and compare the canonical JSON to all current sources, `make evaluate-router` to atomically regenerate JSON/CSV/Markdown, and `make test-router-evaluation` for focused schema/provenance/decision/cost/output coverage. The canonical human-readable result is [`../reports/week6_router_comparison.md`](../reports/week6_router_comparison.md).

## Day 42 router stabilization

`configs/router_tuning.yaml` makes the tuning procedure reviewable before execution. `configs/routed_v0.1.0.yaml` is the immutable baseline; `configs/routed.yaml` is the v0.2.0 target. The only permitted policy difference is the CAREFUL score-gap threshold plus version/status identity. The evaluator orders supported IDs by SHA256, assigns 30 to tuning and 15 to validation, and tests gaps from `0.010` through `0.045`.

An eligible candidate must avoid validation Hit@5 regression, retain 100% unsupported refusal accuracy, keep full supported average replay latency at or below 75% of always CAREFUL, and keep projected total cost at or below always CAREFUL. Selection maximizes tuning Hit@5, then minimizes tuning latency and threshold. The selected `0.030` candidate reaches 60% tuning Hit@5 and 73.33% validation Hit@5; all constraints pass. `0.040` and `0.045` are rejected by the latency ceiling even though `0.045` has higher full-set Hit@5.

Relative to v0.1.0, v0.2.0 moves seven supported questions from STANDARD to CAREFUL and no questions across any other route boundary. Supported Hit@5 rises by 8.89 percentage points, MRR rises from `0.4469` to `0.5267`, and the combined proxy rises by 7.02 points. Average replay latency rises 33.07%; projected cost falls 2.46% because the affected CAREFUL path builds prompts from five reranked chunks rather than STANDARD's ten dense chunks. The unchanged NO_ANSWER rule still refuses 12/12 unsupported and 9/45 supported questions.

`make replay-no-answer` updates refusal decisions from persisted probe evidence under the new router identity without claiming a new live Qdrant measurement. `make tune-router` writes the canonical JSON, 57-row CSV, and Markdown artifacts; `make validate-router-tuning` recomputes and rejects stale evidence; `make test-router-stabilization` covers split/selection/tie-breaking, threshold boundaries, refusal replay, policy drift, and atomic outputs. See [`../reports/week6_router_stabilization.md`](../reports/week6_router_stabilization.md). The deterministic policy is stable and explainable for Day 42, but remains `draft` because this small, previously inspected artifact family is not production validation.

## Day 35 API-path evaluation

`scripts/evaluate_api.py` is a live integration evaluator, not a second in-process dense runner. It probes `/health`, submits every verified label to `POST /query` with debug enabled, validates the full Day 33 response contract, computes retrieval metrics from returned chunks, and writes `dense_baseline_api.json` plus `dense_baseline_api.csv`.

By default it also enforces three external invariants:

1. all 45 complete top-10 chunk rankings and aggregate metrics exactly match `reports/evaluations/dense_baseline.json`;
2. every response UUID exists in the configured SQLite store with matching pipeline identity, answer, total/component timings, Day 40 generation cost, ordered chunks, and generation-use flags; and
3. the four current evidence digests in `configs/mlflow.yaml` each resolve to a complete FINISHED MLflow run with exact parameters, metrics, tags, and artifacts.

Run it while a host API is using the same trace database:

```bash
make evaluate-api
```

The recorded acceptance run matched 45/45 rankings, reproduced MRR 0.3359 and Recall@5 0.4444, verified 45 traces with 450 child chunk rows, and verified all four MLflow runs. Warm service latency averaged 134.75 ms over the final 44 requests; the first request's 22.7-second model initialization remains honestly included in the full-run average. See `reports/week5_integration_review.md` for the exact service, latency, container, trace, and run-ID evidence.

This controlled check uses dense retrieval because it has a directly comparable full-depth offline report and is the default production selection. It does not score template-answer quality or repeat the slower hybrid/reranked benchmark through HTTP.

## Day 28 retrieval construction

The dense, BM25, hybrid, and reranked evaluation entry points now construct `Retriever` objects through the validated pipeline config. Each object exposes `retrieve(query, top_k=None, timings=None)`; the reranked object additionally exposes `retrieve_with_candidates` so the controlled pre-rerank ablation retains the exact candidate order. All checked-in retrieval configs pin `retriever_interface: common_v1`, and the focused regression suite covers interface construction, exact RRF behavior, reranking order/provenance, resource validation, and legacy function compatibility:

```bash
make test-retrieval-interface
```

Day 28 does not rerun or rewrite the Day 27 benchmark. Its acceptance check is behavioral equivalence under the common interface plus the complete test suite; the reported quality and latency values remain tied to the recorded Day 27 live run.

## Day 29 MLflow tracking

`configs/mlflow.yaml` declares the tracking URI, `ragops-retrieval` experiment, and exact config/JSON/CSV/comparison/Markdown artifact set for dense, BM25, RRF hybrid, and reranked runs. `MLFLOW_TRACKING_URI` is the only connection override. Run names always come from the validated pipeline config and must match the evaluation JSON and every CSV row.

Validate without contacting MLflow:

```bash
make test-mlflow
make validate-mlflow
```

Import the recorded benchmark artifacts and verify the acceptance state:

```bash
make services-up
make log-retrieval-runs
make verify-retrieval-runs
```

The import path validates the current algorithm sections against each report, requires finite aggregate metrics, requires complete and identically ordered JSON/CSV question IDs, and rejects missing artifacts. It logs flattened configuration parameters; rank metrics; total, warmed, and component latency where available; pipeline YAML; effective configuration; evaluation JSON/CSV; comparison JSON; and Markdown. Each imported run is tagged `ragops_run_source=validated_artifact_import`. A content digest prevents duplicate imports of identical evidence.

The four live evaluation CLIs use the same logger after they successfully write artifacts and tag those runs `ragops_run_source=live_evaluation`. `--validate-only` returns before tracker construction, and `--skip-mlflow` is an explicit opt-out. If artifact upload fails after run creation, the MLflow run is marked `FAILED`; the already-written local evaluation artifacts are retained. The Compose server proxies client artifact uploads into its persistent `mlflow_data` volume, so host-run evaluators never need direct access to the container's `/mlflow` path.

## Day 30 pipeline registry

All four retrieval YAMLs now explicitly declare `version` and `status`. The generated `reports/pipeline_registry.json` binds each `name@version` to its config checksum, validated Day 29 evidence digest, MLflow experiment/run identity, and metrics from `reports/evaluations/reranker_vs_baselines.json`.

Registry metrics deliberately come from the Day 27 common top-five comparison. Using raw MRR from each historical report would compare dense/BM25/RRF depth 10 against reranked depth 5, so the registry records MRR@5 and the shared Recall/Hit Rate/nDCG cutoffs instead. Historical average latency is retained as context and remains subject to the cold-start and cross-process limitations already documented in the Day 27 report.

Current decisions are:

- `dense_baseline@1.0.0`: approved, `production`, because dense is the online API implementation.
- `bm25_baseline@1.0.0`: approved, `baseline`, because it is the practical measured comparison control.
- `hybrid_rrf@1.0.0`: rejected, no alias, because it trails BM25.
- `hybrid_rrf_cross_encoder@1.0.0`: evaluated, `candidate`, because it leads MRR@5 but has material latency.

Run `make build-pipeline-registry` after an intentional version/status/alias change and `make validate-pipeline-registry` in normal validation. The validator rejects stale source hashes or evidence, missing explicit metadata, invalid semantic versions, dangling aliases, and aliases to ineligible lifecycle states. Promotion and rollback policy is documented in `docs/pipeline_registry.md`; alias movement does not itself deploy a pipeline.

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

No aggregate retrieval metric is assigned to Day 26. Its single acceptance query demonstrates wiring, metadata, and timing—not effectiveness. The complete measurement follows below.

## Day 27 reranker evaluation

Preflight and execution:

```bash
make validate-reranker-evaluation
make test-reranker-evaluation
make evaluate-reranker
```

The evaluator loads one cross-encoder for the full run, retains all 25 RRF candidates plus the final five results for every question, validates both fusion and reranker provenance, and records model-load, dense, BM25, fusion, reranker, and total latency. The four-way comparison requires exact question/source/label parity, compatible component settings, a shared BM25 source SHA, and live dense/BM25 record-count parity.

All headline rankings are truncated to five and MRR is explicitly MRR@5. This gives every official pipeline the same output depth. A second controlled ablation compares the first five positions of the exact RRF-25 candidate ranking from the Day 27 run against its cross-encoded top five, avoiding the candidate-depth confound in the historical Day 25 RRF run.

Recorded benchmark:

| Metric | Dense | BM25 | RRF hybrid | Hybrid + reranker |
| --- | ---: | ---: | ---: | ---: |
| MRR@5 | 0.3163 | 0.6152 | 0.5641 | **0.6889** |
| Hit Rate@1 | 0.2667 | 0.4667 | 0.4667 | **0.5778** |
| Hit Rate@3 | 0.3111 | **0.7556** | 0.6444 | **0.7556** |
| Hit Rate@5 | 0.4444 | **0.8444** | 0.7556 | **0.8444** |
| nDCG@5 | 0.3473 | 0.6727 | 0.6112 | **0.7282** |

The controlled RRF-25 MRR@5 is `0.5644`; cross-encoding raises it by `0.1244`. Reranking wins 16 paired questions, loses five, and ties 24 against that exact order, recovering six top-five misses while losing one prior hit. Against BM25 it wins 14, loses 10, and ties 21. Five explicit controlled regressions and all seven final top-five failures are listed in `reports/week4_reranker_comparison.md`.

Latency tradeoff:

| Day 27 stage | Average | After first query |
| --- | ---: | ---: |
| End to end | 4,681.6 ms | 4,476.4 ms |
| Dense | 261.2 ms | 128.6 ms |
| BM25 | 72.2 ms | 72.4 ms |
| RRF fusion | 0.3 ms | 0.3 ms |
| Cross-encoder | 4,347.6 ms | 4,274.9 ms |

The model loaded once in `28,693.4 ms` before the question loop and that cost is excluded from per-query totals. The internally measured retrieval-plus-fusion stages average `333.7 ms`; the reranker therefore dominates the latency increase. Historical baseline timings come from different processes and are contextual, not controlled latency comparisons.

Artifacts:

- `reports/evaluations/hybrid_rrf_cross_encoder.json`: configuration, model/index provenance, full candidate and final rankings, aggregate metrics, and component latency
- `reports/evaluations/hybrid_rrf_cross_encoder.csv`: flat final top-five question results
- `reports/evaluations/reranker_vs_baselines.json`: common-depth metrics, controlled ablation, paired outcomes, cohorts, relevance groups, and failures
- `reports/week4_reranker_comparison.md`: generated four-way benchmark, regressions, latency analysis, validity limits, and decision

The conclusion is measured rather than universal. The 45 questions map to only 20 labeled chunks, have one relevance judgment each, and were generated from source text that retains lexical overlap. Equal weighting across relevance groups preserves the reranker gain, but unjudged useful chunks and an unpinned Hugging Face model revision remain reproducibility limitations.

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

## Day 46 final evaluation dataset

Day 46 creates new final snapshots rather than rewriting `golden_qa.jsonl`, `retrieval_labels.jsonl`, or `no_answer_queries.jsonl`. Those earlier inputs are retained because Days 20, 39, 41, and 42 already produced reports and hashes from them. The Day 47 benchmark must use the `final_*` artifacts below; older reports remain historically reproducible.

| Artifact | Count | Composition |
| --- | ---: | --- |
| `data/eval/final_golden_qa.jsonl` | 100 | 72 supported, 5 ambiguous, 23 unsupported; 30 easy, 35 medium, 35 hard. |
| `data/eval/final_retrieval_labels.jsonl` | 50 | 35 retained verified-synthetic labels and 15 newly manual labels; 61 relevant chunk judgments. |
| `data/eval/final_adversarial_qa.jsonl` | 30 | 18 near-domain, 7 high-stakes, 2 instruction-injection, 1 false-premise, and 2 general out-of-scope prompts. |

### Construction and review

The source contract is `configs/final_evaluation_dataset.yaml`. It records the reviewer and date, exact count bounds, ten exclusions with individual rationales, 15 manual retrieval decisions, and all historical/addition/chunk/output paths. `data/eval/day46_additions.jsonl` is the human-readable addition manifest.

The process was:

1. Review all 80 historical golden questions and their provenance.
2. Exclude ten approved synthetic rows that were context-free, semantically duplicated stronger manual questions, or too trivial for comparative evaluation. No historical file is mutated.
3. Retain the other 70 historical rows and attach uniform final-review metadata while preserving their original origin/provider/source metadata.
4. Add 12 hard supported questions: four each for FastAPI, MLflow, and Qdrant. Every answer was checked against one or more exact processed chunk IDs recorded in its metadata.
5. Add 18 unsupported/adversarial prompts and include each in both the final golden set and the final adversarial set. This intentional overlap allows generation/refusal behavior to be compared on the same reviewed prompts.
6. Retain the 35 synthetic retrieval labels whose questions survived curation. Add 15 independent manual relevance decisions—five per source family—against exact source-scoped chunks.
7. Carry forward all 12 Day 39 unsupported cases into the 30-case final adversarial set, preserving their calibration/evaluation role and original reviewer provenance.

The final-set builder rejects duplicate IDs or normalized questions, unknown exclusions, output/source path collisions, unsupported questions with a source, supported sources absent from the corpus, unknown or cross-source chunk IDs, duplicate relevance decisions, insufficient query-type/difficulty/source/category/manual-label coverage, and any count outside the reviewed bounds. Every final golden row must carry `final_review_status`, `final_reviewed_by`, and `final_reviewed_on` metadata.

### Reproduction and audit

Build or validate with:

```bash
make finalize-evaluation-dataset
make validate-final-evaluation-dataset
make test-final-dataset
```

Building reconstructs all three snapshots from immutable historical inputs plus the explicit additions/config and atomically writes `reports/evaluations/final_dataset_review.json`. Validation independently reconstructs the expected rows and report and rejects any stale or hand-edited output. The report contains distributions, every exclusion decision, source SHA256s—including the processed corpus—and final artifact SHA256s.

Full construction/validation needs the ignored generated `data/processed/chunks.jsonl` to verify source evidence. `tests/test_final_dataset.py` is hermetic: miniature local chunks test the complete curation contract, while a structural test protects the committed final artifacts and hashes without requiring the full corpus in CI.

The set is deliberately capped at 100/50/30 rather than expanded indefinitely. It is large enough to compare five pipeline variants across supported retrieval and refusal behavior, but every row, exclusion, added answer, manual relevance judgment, and adversarial category remains inspectable in a small number of text files. It is still a curated documentation benchmark, not an unbiased sample of production traffic.

## Day 47 final benchmark contract and execution status

`configs/final_benchmark.yaml` is the single contract for the central five-way comparison. It pins the final Day 46 datasets; five distinct pipeline config/report/judgment bundles; a ten-ID supported answer-quality sample; OpenAI generation and cross-provider Gemini judging; a common retrieval depth of five; routed depths; percentile method; the exact reference-answer cost basis; MLflow experiment; and final JSON/CSV/Markdown paths. Strict validation rejects sample IDs outside the retrieval-labeled subset, dataset count drift, duplicate adversarial identities, missing pricing, mismatched report run names, partial rankings/scores, and any question/source/relevance-label mismatch.

The metric scopes are deliberately different where the available labels differ:

- Recall@5 and MRR@5 use all 50 paired supported retrieval labels. MRR is truncated to five for every pipeline.
- Faithfulness and answer relevance are 1–5 rubric means over the same explicit ten supported questions for every pipeline.
- Refusal correctness applies to the routed policy over all 30 reviewed unsupported/adversarial prompts. Fixed retrieval pipelines have no explicit refusal policy, so their value is N/A rather than zero.
- p50/p95 are linear-interpolated retrieval-only wall-clock measurements and include cold starts. Routed timings are documented serial artifact replay.
- Estimated generation cost/query uses all 50 supported questions, each exact retrieved prompt, the same verified reference answer, the checked model-price table, and the deterministic token estimator. This isolates context-size effects instead of letting different generated prose determine the ablation.

The local phase completed against Qdrant and the 13,481-chunk corpus. Common-depth results are dense `0.5067` Recall@5 / `0.4057` MRR@5, BM25 `0.7767` / `0.5737`, hybrid `0.7267` / `0.5820`, reranked `0.8100` / `0.6473`, and routed `0.6600` / `0.5307`. The controlled reranker ablation records 14 gains, 8 regressions, and 28 ties against its own RRF-25 top-five order. The routed policy selects 2 FAST, 12 STANDARD, 29 CAREFUL, and 7 NO_ANSWER supported cases; on the 30 adversarial cases it correctly refuses 25 and misses five.

The approved external phase completed after Gemini billing became available. OpenAI generated 50 real answers and Gemini produced 50 cross-provider judgments: ten ordered rows for each of dense, BM25, hybrid, reranked, and routed. Mean faithfulness/relevance scores are respectively dense `5.00/4.30`, BM25 `5.00/4.50`, hybrid `4.80/4.30`, reranked `5.00/4.50`, and routed `5.00/4.60`. These are model-based estimates on a fixed ten-question supported sample, not human ground truth. The retained judgments include low-relevance and refusal cases; no missing row was filled with a citation proxy or hand-authored score.

The first attempt exposed a Gemini quota boundary, so the runner now validates and reuses complete pipeline artifacts, atomically checkpoints every completed question, resumes only an ordered valid prefix, and retries only explicit 429/rate-limit/temporary-availability errors with bounded delays. It never retries permanent schema or rubric failures. That checkpoint path was exercised by the completed run: dense was reused, BM25 resumed after its second row, and the other pipelines ran to ten rows.

The final table is available as [Markdown](../reports/final_benchmark.md), [JSON](../reports/evaluations/final_benchmark.json), and [CSV](../reports/evaluations/final_benchmark.csv). Its five exact `ragops-final-benchmark` MLflow run IDs are dense `15d6f764d1ba474ea4556f34948f2444`, BM25 `cc073a8526ee4b809f3d40d891c4be00`, hybrid `a1e0348130a646b8b7b6cf0c02d43f2d`, reranked `aff82d0914f8403e9479812a90096e09`, and routed `45b98920cd68421c8d7094fae7128604`. Because the Docker tracking endpoint on port 5000 was unresponsive during publication, the CLI used its explicit `--mlflow-uri http://127.0.0.1:5001` override with a repository-local tracking server. A strict verification pass confirmed the experiment, finished status, source digest, tags, metrics, direct links, and required source artifacts for all five runs.

Commands are phase-separated so this boundary remains observable:

```bash
make validate-final-benchmark
make evaluate-final-retrieval
make evaluate-final-routed
make judge-final-answers       # external data sharing and provider usage
make aggregate-final-benchmark
make verify-final-benchmark
make test-final-benchmark
```

The aggregator refuses incomplete or out-of-order judgment sets. The completed run records the full central table, wording-cohort and supported-query win counts, every BM25-over-dense case, controlled reranker gains/regressions, routed cost reductions, routed quality regressions, exact file paths, source digests, and direct MLflow run URLs. The verifier independently reopens those URLs by run ID and checks that the logged evidence still matches the final source bundle.

## Day 48 failure analysis and regression dataset

`configs/failure_analysis.yaml` is the reviewed contract for converting measured Day 47 failures into engineering diagnoses and future regression inputs. It selects 15 cases across nine categories: one bad dense retrieval, one lexical BM25 miss, two hybrid-fusion failures, three reranker regressions, two incorrect supported refusals, three unsupported router mistakes, one high-latency query, one weak claim-to-citation alignment, and one unexpected answer-level refusal.

Each case records the query, expected behavior, actual behavior, diagnosed root cause, affected component, proposed fix, severity, regression decision, and a machine-checkable evidence assertion. Rank cases recompute relevant rank at the common top-five depth; route cases verify dataset side, exact route, reason code, and incorrect adversarial decision; judgment cases verify pipeline, behavior, faithfulness, relevance, and refusal verdict; the latency case verifies the recorded routed request remains above its declared analysis threshold.

Fourteen cases are written to `data/eval/regression_cases.jsonl`. Each row carries its source failure ID, expected behavior, forbidden measured behavior, evidence guard, proposed fix, provenance, and verified-review status. The `55,600.3 ms` first routed CAREFUL case is not promoted because its serial dense-plus-reranker cold start is host-dependent; it belongs in a separate warm-path performance budget, not a deterministic functional gate.

The generated [failure report](../reports/failures/failure_analysis.md) distinguishes measured facts from engineering inference. Root-cause descriptions are diagnoses based on ranks, routes, timings, retrieved context, and judge rationales; they are not presented as controlled causal experiments. Likewise, regression promotion records what future candidates must improve and does not falsely claim that the proposed fixes are already implemented.

```bash
make analyze-failures             # rebuild both reviewed outputs
make validate-failure-analysis    # reconstruct and reject drift/manual edits
make test-failure-analysis        # six focused contract/evidence tests
```

The validator reads only frozen local artifacts and makes no Qdrant, Docker, MLflow, model, or provider call. It rejects Day 47 benchmark identity/status drift, missing or duplicate question evidence, changed ranks/routes/reason codes/judgments, a latency case falling below its recorded failure threshold, insufficient category/regression coverage, stale Markdown, and modified JSONL.
