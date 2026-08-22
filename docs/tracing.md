# Request tracing

SQLite schema v4 stores accepted `/retrieve` and `/query` attempts.

| Endpoint | Stored | Trace ID returned |
| --- | --- | --- |
| `/health` | No | No |
| `/route` | No | No |
| `/retrieve` | Yes | No |
| `/query` | Yes | Body and `X-Trace-ID` |

FastAPI validation failures (HTTP 422) happen before the handler and are not traced.

## Stored data

`traces` contains:

- UUID, UTC timestamps, endpoint, query, and requested depth;
- pipeline name/version and success/error status;
- answer and retrieved-chunk count;
- total and component latency;
- provider/model, token counts, rates, cost, and provenance; and
- error type/message.

`retrieved_chunks` stores ordered evidence, scores, source, metadata, and `used_for_generation`. Trace and chunk rows commit in one transaction.

`feedback` stores optional rating/comment records against existing traces. There is no feedback API.

## Timings

The stable component fields are:

```text
embedding_ms
dense_ms
bm25_ms
fusion_ms
reranker_ms
generation_ms
```

Only executed stages are populated. Failed stages retain elapsed time. Components do not have to sum to total latency because initialization, orchestration, serialization, and trace persistence have separate boundaries.

## Failure behavior

- Retrieval errors store any completed timings.
- Generation errors also store retrieved evidence.
- Trace-write failure returns HTTP 503; the API does not claim success.
- Invalid ranks, duplicate chunks, count drift, non-finite values, or invalid JSON roll back the transaction.

## Configuration

| Variable | Default |
| --- | --- |
| `RAGOPS_TRACE_DB_PATH` | `data/traces/ragops_traces.sqlite3` |
| `RAGOPS_PIPELINE_NAME` | `dense_baseline` for `/retrieve` |
| `RAGOPS_PIPELINE_VERSION` | `1.0.0` for `/retrieve` |

`/query` records its selected config identity instead of those defaults.

```bash
make init-trace-store
make validate-trace-store
make test-tracing
```

Compose stores traces in the `ragops_trace_data` volume. A host-run dashboard cannot read that database unless it is made host-visible.

## Operational limits

Traces contain raw queries, answers, retrieved text, metadata, and errors. Restrict access and define retention before deployment. SQLite provides no redaction, encryption, retention jobs, distributed writes, or full-text trace search.

See [cost estimation](cost_estimation.md) for cost fields and caveats.
