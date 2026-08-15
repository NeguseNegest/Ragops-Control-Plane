# Query Routing

## Day 37 Status

Day 37 implements the initial retrieval probe that will supply the rule-based router with corpus-aware confidence and deterministic query features. It does **not** select `FAST`, `STANDARD`, `CAREFUL`, or `NO_ANSWER`; route thresholds and reasons belong to the router-policy milestones.

The boundary is deliberate:

```text
clean query
    |
    +-> lexical/query-length extraction (no model call)
    |
    +-> dense top 2 (one embedding + one Qdrant search)
             |
             +-> top score
             +-> top-1 minus top-2 score gap
             +-> reusable top-two chunks
                         |
                         +-> InitialRetrievalFeatures schema v1
                                      |
                                      +-> future deterministic router
```

`PipelineRuntime.initial_probe()` always uses `dense_baseline`. It requests exactly two results—the minimum depth that can produce both a top score and a score gap—and does not load the BM25 index, run RRF, load the cross-encoder, or call a generation provider. Its request-scoped Qdrant client is closed on both success and failure through the existing runtime lifecycle.

## Structured Feature Contract

`InitialRetrievalFeatures` is a frozen Pydantic object with `extra="forbid"` and `schema_version: 1`. The future router receives three nested groups:

| Group | Field | Meaning |
| --- | --- | --- |
| `query_length` | `character_count` | Length of the stripped query, including internal spaces and punctuation. |
| `query_length` | `token_count` | Number of normalized Unicode word/number tokens. |
| `lexical_complexity` | `unique_token_count` | Number of distinct case-folded tokens. |
| `lexical_complexity` | `unique_token_ratio` | Distinct tokens divided by all tokens. |
| `lexical_complexity` | `average_token_length` | Mean token length in characters. |
| `lexical_complexity` | `maximum_token_length` | Longest token length. |
| `lexical_complexity` | `long_token_count` | Tokens containing at least eight characters. |
| `lexical_complexity` | `long_token_ratio` | Long tokens divided by all tokens. |
| `lexical_complexity` | `clause_marker_count` | Count of a checked-in closed set such as `and`, `because`, `if`, and `whereas`. |
| `lexical_complexity` | `complexity_marker_count` | Count of a checked-in closed set such as `compare`, `explain`, `trade-off`, and `why`. |
| `retrieval_confidence` | `requested_top_k` | Fixed at two for feature-schema stability. |
| `retrieval_confidence` | `result_count` | Number of dense results returned, from zero to two. |
| `retrieval_confidence` | `top_score` | Raw score of rank one, or `null` when the corpus returned no result. |
| `retrieval_confidence` | `score_gap` | Rank-one score minus rank-two score, or `null` with fewer than two results. |

Scores remain raw dense cosine-search outputs. Day 37 does not convert them to probabilities or label a score as “high confidence.” Threshold calibration must use observed corpus scores rather than inventing a universal interpretation.

`InitialProbeResult` also retains the two normalized chunks. This allows a future `FAST` path to reuse the probe evidence instead of immediately repeating the same dense search. Probe timings (`total_ms`, `embedding_ms`, and `dense_ms`) are diagnostic fields kept outside the router feature object, so latency cannot accidentally influence route selection without an explicit schema change.

## Validation and Edge Cases

The probe validates its evidence before handing it to routing logic:

- the query is stripped and must contain at least one Unicode word or number token;
- no more than two results may be returned;
- ranks must be exactly one-based and contiguous;
- chunk IDs must be unique;
- scores must be finite and ordered from highest to lowest;
- zero results produce `top_score: null` and `score_gap: null`;
- one result produces a top score but no fabricated gap; and
- two results require both a top score and a non-negative gap.

Negative top scores are valid because a low-similarity corpus match can still be the highest result. They are preserved for later `NO_ANSWER` threshold design.

## Run the Probe

With Qdrant running and the embedding model available locally:

```bash
make probe-query ROUTER_QUERY="What is FastAPI?"
```

The command prints the feature object, chunk IDs/scores, and timings. It deliberately prints `route: null` and `route_reason: null`, preventing a Day 37 diagnostic from being mistaken for an implemented router decision.

The recorded Day 37 smoke query returned two real corpus chunks with `top_score=0.66855043` and `score_gap=0.01691073`. The new host process spent about 39.63 seconds initializing the cached embedding model, while the Qdrant dense-search stage took about 19.59 ms. These are smoke-test observations, not routing thresholds or a latency benchmark.

Run focused tests with:

```bash
make test-routing-probe
```

## Remaining Routing Work

Day 37 supplies features only. The following remain unimplemented:

- the `FAST`, `STANDARD`, `CAREFUL`, and `NO_ANSWER` decision policy;
- calibrated score, gap, length, and complexity thresholds;
- the non-empty `configs/routed.yaml` contract;
- automatic routing in `POST /query` and response/trace route reasons; and
- unsupported-query refusal evaluation.

Because Day 36 was skipped in the execution sequence, its documented policy/config acceptance criterion still needs to be completed before the Day 38 rule-based router can be implemented soundly.
