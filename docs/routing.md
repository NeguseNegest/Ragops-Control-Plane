# Query Routing

## Current Status

Day 36 defines the versioned routing policy, Day 37 implements the initial probe, Day 38 implements deterministic selection, and Day 39 calibrates `NO_ANSWER`, adds a deterministic refusal, and measures refusal correctness.

`POST /route` exposes decisions without executing FAST/STANDARD/CAREFUL. For `NO_ANSWER`, it now returns the exact policy refusal without calling a generation provider. `POST /query` still uses explicit config selection. The Day 37 diagnostic probe continues to print `route: null`; use `route-query` or `/route` when a decision is wanted.

```text
query
  |
  +-> deterministic lexical and length features
  |
  +-> configured dense probe (currently top 2)
          |
          +-> result count
          +-> top score
          +-> top-1 minus top-2 gap
          +-> reusable ranked chunks
                     |
                     v
          InitialRetrievalFeatures v1
                     |
                     v
          Day 36 policy in routed.yaml
                     |
                     v
          deterministic RuleBasedRouter
                     |
                     v
          route + reason + execution intent
```

## Route Definitions

The checked-in policy is `rule_router@0.1.0` with lifecycle status `draft`. It defines four uppercase decision values:

| Route | Intended use | Pipeline intent | Output ceiling | Probe reuse | Current lifecycle guard |
| --- | --- | --- | ---: | --- | --- |
| `FAST` | Simple query with strong, separated dense evidence | `dense_baseline` | 2 | Yes | `approved` only |
| `STANDARD` | Normal query in the middle confidence/complexity band | `dense_baseline` | 10 | No | `approved` only |
| `CAREFUL` | Complex, ambiguous, or low-confidence query | `hybrid_rrf_cross_encoder` | 5 | No | `evaluated` or `approved` |
| `NO_ANSWER` | No usable evidence or score below the conservative refusal-candidate floor | No retrieval pipeline and no ordinary generation | 0 | No | Refusal mode |

`FAST` and `STANDARD` deliberately use the approved dense pipeline. The reranked pipeline has the best measured top-five quality and is the intended `CAREFUL` path, but it is only `evaluated` and has a costly warmed reranker stage. The whole routing policy therefore remains `draft`; defining CAREFUL intent is not the same as promoting or deploying it.

`NO_ANSWER` now produces a deterministic corpus-scoped refusal. It does not use retrieved chunks as citations and does not call an LLM.

## Decision Precedence and Thresholds

`configs/routed.yaml` fixes the only allowed schema-v1 decision order:

1. `NO_ANSWER`
2. `CAREFUL`
3. `FAST`
4. `STANDARD` fallback

This order matters. A short query with a low score remains CAREFUL or NO_ANSWER; simplicity cannot override missing corpus evidence.

### NO_ANSWER

The selector chooses `NO_ANSWER` when either condition is true:

- the dense probe returns zero results; or
- `top_score < 0.531`.

The inequality is strict. A score of exactly `0.531` continues to later rules.

Day 39 selected `0.531` from five calibration unsupported examples: maximum score `0.5302763`, plus margin `0.0005`, rounded upward to three decimals. All seven newly authored held-out unsupported questions also fell below this value. The price is explicit: 9 of 45 supported evaluation questions now fall below the threshold and are falsely refused. See [`no_answer.md`](no_answer.md) for the complete method and report.

### CAREFUL

After NO_ANSWER, `CAREFUL` matches when **any** condition is true:

- only one result was returned, so the score gap is unavailable;
- `top_score < 0.56`;
- `score_gap < 0.01`;
- `token_count > 20`;
- `complexity_marker_count >= 1`;
- `clause_marker_count >= 3`; or
- `long_token_ratio >= 0.40`.

The OR rule treats either weak retrieval evidence or linguistic complexity as enough reason to use the higher-quality path. Missing gap data is explicitly CAREFUL rather than silently ordinary.

### FAST

After NO_ANSWER and CAREFUL, `FAST` matches only when **every** condition is true:

- `top_score >= 0.72`;
- `score_gap >= 0.05`;
- `token_count <= 12`;
- `complexity_marker_count <= 0`;
- `clause_marker_count <= 1`; and
- `long_token_ratio <= 0.30`.

The AND rule makes FAST intentionally narrow. `maximum_top_k=2` is no greater than the configured probe depth, and FAST uses the same dense pipeline, so the existing probe chunks can be reused without another retrieval.

### STANDARD

`STANDARD` is the fallback when no earlier rule matches. The schema validators require gaps between the FAST and CAREFUL thresholds—for example, 13–20 tokens occupy a normal band unless another CAREFUL condition matches. This prevents overlapping configuration from making precedence do hidden policy work.

## Deterministic Decision Contract

`RuleBasedRouter.select()` accepts only `InitialRetrievalFeatures` schema version 1 (or data that strictly validates as that schema). It has no network access, model call, clock input, mutable counters, or randomness. Re-evaluating identical features under identical config produces an equal frozen `RouterDecision`.

Every decision contains:

- `router_id` and `router_status`, currently `rule_router@0.1.0` and `draft`;
- `feature_schema_version` and uppercase `route`;
- `reason_code`, stable human-readable `reason`, and ordered `matched_reason_codes`;
- `pipeline_config`, `maximum_top_k`, `reuse_probe`, and `generate_answer`; and
- `response_mode: refusal` only for `NO_ANSWER`.

The first matching route wins. Within CAREFUL, every matching condition is retained in a stable order, and the first becomes the primary `reason_code`. This gives operators the complete explanation without allowing the explanation order to change the selected route.

| Reason code | Route | Meaning |
| --- | --- | --- |
| `empty_probe` | `NO_ANSWER` | Dense retrieval returned no evidence |
| `top_score_below_no_answer_threshold` | `NO_ANSWER` | Rank-one score is below the refusal-candidate floor |
| `missing_score_gap` | `CAREFUL` | Only one dense result exists |
| `top_score_below_careful_threshold` | `CAREFUL` | Rank-one score is in the low-confidence band |
| `score_gap_below_careful_threshold` | `CAREFUL` | Top two results are insufficiently separated |
| `token_count_above_careful_threshold` | `CAREFUL` | Query is over the configured token threshold |
| `complexity_marker_count_at_least_careful_threshold` | `CAREFUL` | Query contains a configured complexity marker |
| `clause_marker_count_at_least_careful_threshold` | `CAREFUL` | Query contains at least the configured clause count |
| `long_token_ratio_at_least_careful_threshold` | `CAREFUL` | Long-token ratio reaches the configured floor |
| `fast_conditions_satisfied` | `FAST` | Every FAST confidence and simplicity rule matches |
| `standard_fallback` | `STANDARD` | No earlier route matches and at least one FAST rule fails |

`RouterDecision` rejects mismatched reason text, duplicate/misordered reason codes, a retrieval pipeline attached to `NO_ANSWER`, a refusal response attached to a retrieval route, or a retrieval route without positive output depth and generation intent. These checks prevent downstream code from receiving a route label whose execution fields say something else.

## Feature Contract

`InitialRetrievalFeatures` is a frozen Pydantic object with `extra="forbid"` and `schema_version: 1`.

| Group | Field | Meaning | Decision-active in v0.1.0 |
| --- | --- | --- | --- |
| `query_length` | `character_count` | Stripped query length including internal whitespace and punctuation | No; retained for analysis |
| `query_length` | `token_count` | Normalized Unicode word/number token count | Yes |
| `lexical_complexity` | `unique_token_count` | Distinct case-folded tokens | No; retained for analysis |
| `lexical_complexity` | `unique_token_ratio` | Distinct tokens divided by total tokens | No; retained for analysis |
| `lexical_complexity` | `average_token_length` | Mean token length | No; retained for analysis |
| `lexical_complexity` | `maximum_token_length` | Longest token length | No; retained for analysis |
| `lexical_complexity` | `long_token_count` | Tokens with at least eight characters | No; ratio is used |
| `lexical_complexity` | `long_token_ratio` | Long tokens divided by total tokens | Yes |
| `lexical_complexity` | `clause_marker_count` | Checked-in terms such as `and`, `because`, `if`, and `whereas` | Yes |
| `lexical_complexity` | `complexity_marker_count` | Checked-in terms such as `compare`, `explain`, `trade-off`, and `why` | Yes |
| `retrieval_confidence` | `requested_top_k` | Probe depth loaded from router config | Contract/provenance |
| `retrieval_confidence` | `result_count` | Dense results returned | Yes for empty/missing-gap behavior |
| `retrieval_confidence` | `top_score` | Raw rank-one dense score | Yes |
| `retrieval_confidence` | `score_gap` | Rank-one score minus rank-two score | Yes |

Unused schema-v1 fields remain available for analysis without affecting decisions. Turning one into a threshold requires an explicit config/version change.

Scores remain raw cosine-search outputs, not probabilities. Negative finite top scores are valid. The score gap is always computed from ranks one and two even when a configured probe depth greater than two returns additional evidence.

## Initial Probe Configuration and Validation

Day 37 no longer owns a hard-coded depth. `PipelineRuntime` loads `configs/routed.yaml` during initialization, validates it, selects `probe.pipeline_config`, and passes `probe.top_k` into `run_initial_retrieval_probe`. Schema version 1 allows a depth from two through five; the current policy uses exactly two because that is the minimum that yields a gap.

The probe remains cheap relative to the other routes:

- one query embedding;
- one dense Qdrant search;
- no BM25 index load;
- no RRF;
- no cross-encoder; and
- no generation call.

It validates that returned evidence does not exceed configured depth, ranks are one-based and contiguous, IDs are unique, and scores are finite and descending. Zero results have no score or gap. One result has a top score but no gap. Two or more results require both values.

`InitialProbeResult` retains the normalized chunks for possible FAST reuse. Diagnostic `total_ms`, `embedding_ms`, and `dense_ms` values are kept outside the router feature object so latency cannot accidentally affect decisions.

## Calibration Evidence and Limitations

The draft thresholds reference `reports/evaluations/dense_baseline.json` and the validated 45-question set. `make validate-router-config` recomputes the recorded confidence range and checks route pipeline references/statuses against `reports/pipeline_registry.json`:

| Statistic | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Dense top score | 0.3018593 | 0.6597541 | 0.8521882 |
| Top-two score gap | 0.0003204 | 0.0299468 | 0.1451546 |

Applying the Day 39 conditions to those 45 supported questions yields:

| Draft route | Questions | Dense Hit@1 | Dense Hit@5 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `NO_ANSWER` | 9 | — | — | False refusals under the safety-first Day 39 threshold |
| `CAREFUL` | 16 | — | — | Complexity/ambiguity or the narrow `0.531`–`0.56` score band |
| `FAST` | 2 | 2 | 2 | Narrow 2/2 observation, far too small to claim general precision |
| `STANDARD` | 18 | — | — | Middle band retained on the approved production dense pipeline |

These counts were initially descriptive rather than a routed quality benchmark. The dedicated Day 39 evaluation adds 12 unsupported examples and records 12/12 refusals, 9/45 supported false refusals, and 57.14% refusal precision. Day 41 subsequently turns the same fixed evidence into a paired routed-versus-fixed tradeoff report described below. Dense score/gap distributions still overlap, so no universal semantic-confidence claim is made.

The thresholds remain `draft` because the unsupported set is small and the supported false-refusal rate is material. Later work must add broader adversarial coverage, route-level latency/quality measurement, and threshold hardening.

## Configuration Safety

`ragops.routing.config` rejects:

- missing, empty, malformed, or non-mapping YAML;
- unknown keys;
- unsupported schema or feature-schema versions;
- invalid semantic versions or router names;
- a decision order other than NO_ANSWER → CAREFUL → FAST → STANDARD;
- overlapping score, gap, length, marker, or long-token bands;
- FAST probe reuse with a different pipeline or output depth above the probe depth;
- STANDARD/CAREFUL probe reuse in schema version 1;
- duplicate or unsafe lifecycle-status lists;
- route references missing from the pipeline registry;
- a current pipeline status not allowed by its route; and
- FAST/STANDARD references that diverge from the registry production alias or a CAREFUL reference that diverges from the candidate alias;
- missing/stale calibration report identity, question count, or score shape.

Validate all checked-in relationships with:

```bash
make validate-router-config
```

## Run and Test Routing

With Qdrant running and the embedding model available locally:

```bash
make probe-query ROUTER_QUERY="What is FastAPI?"
```

The adjusted Day 37 live verification loaded probe depth two from `routed.yaml` and returned the same two corpus chunks as the original run: top score `0.66855043`, second score `0.65163970`, and gap `0.01691073`. A fresh host process measured about 26.92 seconds in model initialization/embedding and 18.28 ms in dense search. Cold initialization varies and is not a routing threshold or steady-state benchmark.

The probe command deliberately returns `route: null` and `route_reason: null` because it is the Day 37 feature diagnostic. Run the Day 38 decision command with:

```bash
make route-query ROUTER_QUERY="What is FastAPI?"
```

The route report omits document text and returns the policy identity, route, primary and matching reasons, execution intent, exact features, probe chunk IDs/scores, and probe timings.

The Day 38 live CLI smoke check used the same local Qdrant corpus and offline model cache as Day 37. `What is FastAPI?` returned the same two chunks/scores and selected `STANDARD` with `standard_fallback`: its `0.66855043` top score remains safely above the Day 39-adjusted CAREFUL floor of `0.56`, its `0.01691073` gap is above CAREFUL's `0.01` floor, and its simple three-token query has no complexity match, but the top score is below FAST's `0.72` requirement. The cold process measured `17619.20 ms` total, including `17337.97 ms` embedding/model initialization and `25.64 ms` dense search; this is smoke evidence, not steady-state performance.

The equivalent HTTP diagnostic is:

```bash
curl -X POST http://127.0.0.1:8000/route \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is FastAPI?"}'
```

`POST /route` runs the real configured dense probe and returns `decision`, `features`, minimal `probe_chunks` (`chunk_id`, `score`, `rank`), `probe_timings`, and nullable `refusal`. For NO_ANSWER, `refusal` contains the exact Day 39 response, `no_answer_v1`, its prompt SHA256, and `generated_by=deterministic_policy`. Other routes return null. It does not execute a final pipeline, return document text, call an LLM for refusal, or create a Day 31 query trace.

Invalid queries return HTTP 400. Probe resource failures, retrieval failures, and unexpected routing failures return stable HTTP 503 details without leaking internal exception text. Request bodies rejected before the handler return FastAPI HTTP 422.

Run focused validation with:

```bash
make test-routing-probe
make test-no-answer
make validate-no-answer
```

## Day 41 Router Evaluation

`configs/router_evaluation.yaml` defines a deterministic `artifact_replay` comparison among always FAST, always CAREFUL, and the current router. It cross-checks the dense and reranked question records against all 45 labels, recomputes every supported and unsupported decision against `rule_router@0.1.0`, reconstructs exact generation prompts from selected processed chunks, and rejects stale routing, dataset, chunk, or pricing evidence.

The recorded supported route mix is 2 FAST, 18 STANDARD, 16 CAREFUL, and 9 NO_ANSWER. Always FAST reaches 28.89% Hit@5; always CAREFUL reaches 84.44%; routed reaches 55.56%. Routed correctly refuses all 12 reviewed unsupported questions, but the nine supported NO_ANSWER decisions remain false refusals. The combined supported-evidence/unsupported-refusal proxy is 22.81% for always FAST, 66.67% for always CAREFUL, and 64.91% for routed.

Measured-artifact serial replay estimates average retrieval latency at 679.9 ms for always FAST, 4681.6 ms for always CAREFUL, and 2514.3 ms for routed. Controlled `gpt-5-nano` prompt/reference-answer projections total `$0.00145935`, `$0.00355245`, and `$0.00320625`, respectively. These values include cold artifacts, use dense top-10 latency as a conservative top-2 probe proxy, and are neither a simultaneous live benchmark nor a provider invoice.

The result is `keep_router_draft`: routed makes a useful latency/quality compromise against the two extremes, but supported retrieval remains 28.89 Hit@5 points below always CAREFUL and false refusal is still 20%. Run:

```bash
make validate-router-evaluation
make evaluate-router
make test-router-evaluation
```

The canonical explanation and limitations are in [`../reports/week6_router_comparison.md`](../reports/week6_router_comparison.md).

## Remaining Work

Day 42 must tune thresholds, expand router tests, and produce the dedicated distribution report before the policy can move beyond draft. Later routing execution work must connect non-refusal decisions to final retrieval/generation, cap requested output depth, reuse FAST evidence, and persist routing provenance in traces. `/query` remains explicitly selected; `/route` enforces NO_ANSWER refusals but is not yet a general routed query endpoint.
