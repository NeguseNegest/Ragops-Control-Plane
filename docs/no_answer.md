# No-answer behavior

When `/route` selects `NO_ANSWER`, FastAPI returns exactly:

```text
I do not know based on the available FastAPI, MLflow, and Qdrant documentation.
```

The response is deterministic (`no_answer_v1`). It makes no LLM call, creates no citations, and does not treat probe chunks as answer evidence.

## Threshold

The current rule is:

```text
no results OR dense top score < 0.531
```

`0.531` came from five calibration queries: maximum unsupported score `0.5302763`, plus `0.0005`, rounded upward to three decimals. Equality does not refuse.

This threshold only applies to the current MiniLM model, corpus, and Qdrant index. A cosine score is not a confidence probability.

## Evidence

The original refusal study used 5 calibration and 7 held-out unsupported queries:

| Metric | Result |
| --- | ---: |
| Unsupported refused | 12/12 |
| Supported answered | 36/45 |
| Supported false refusals | 9/45 |
| Refusal precision | 57.14% |

The broader final benchmark refused 25/30 adversarial queries and falsely refused 7/50 supported retrieval questions. The policy remains `draft`.

## API boundary

- `/route` returns the refusal plus prompt version/hash and generator identity.
- `/route` is not traced.
- `/query` does not enforce this policy and can be called directly.

## Commands

```bash
make validate-no-answer
make evaluate-no-answer
make replay-no-answer
make test-no-answer
```

`evaluate-no-answer` runs live Qdrant probes. `replay-no-answer` recomputes decisions from stored scores after a router-only change.

## Limits

- The calibration sample is small and manually authored.
- Raw score thresholds do not prove evidence sufficiency.
- The set is documentation-heavy, monolingual, and not production traffic.

See [routing](routing.md) for the full decision policy.
