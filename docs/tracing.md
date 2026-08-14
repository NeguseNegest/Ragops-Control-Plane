# SQLite Request Tracing

## Day 31–33 Contract

The Day 31 trace store makes accepted online retrieval and generation attempts durable. FastAPI creates a UUID and UTC start timestamp when a valid request enters `POST /retrieve` or `POST /query`, then records exactly one terminal `success` or `error` trace before returning.

Day 32 adds a request-scoped monotonic timing context. Successful API responses and their SQLite traces contain the same optional embedding, dense, BM25, fusion, reranker, and generation latencies. Day 33 returns the stored trace ID and selected route/config in every successful `/query` body and repeats the ID in `X-Trace-ID`. Accepted `/query` errors expose their attempted stored trace through the same header.

## Schema

The database uses SQLite schema version 3 and contains three application tables:

### `traces`

| Field | Meaning |
| --- | --- |
| `trace_id` | Canonical UUID primary key. |
| `created_at`, `completed_at` | Timezone-aware UTC ISO-8601 timestamps. |
| `endpoint` | `retrieve` or `query`. |
| `query` | Exact request query, including whitespace for a failed empty-query attempt. |
| `requested_top_k` | Requested positive retrieval depth. |
| `pipeline_name`, `pipeline_version` | Explicit deployed pipeline identity. |
| `status` | `success` or `error`. |
| `retrieved_chunk_count` | Must equal the related chunk-row count at write time. |
| `answer` | Generated answer for `/query`; always null for `/retrieve`. |
| `total_latency_ms` | Finite, non-negative whole-request latency. |
| `embedding_ms` | Query embedding latency when a dense stage ran. |
| `dense_ms` | Qdrant search and result-normalization latency, excluding embedding. |
| `bm25_ms` | Sparse search latency when a BM25 stage ran. |
| `fusion_ms` | Reciprocal Rank Fusion latency when a hybrid stage ran. |
| `reranker_ms` | Cross-encoder model-scoring latency when that stage ran. |
| `generation_ms` | Prompt/citation construction and generation-client latency for `/query`. |
| `error_type`, `error_message` | Both required for errors and both null for successes. |

### `retrieved_chunks`

Each row belongs to one trace through a cascading foreign key. The composite primary key is `(trace_id, rank)`, ranks must be contiguous and one-based when written, and a chunk ID may appear only once per trace. Rows preserve document/chunk IDs, full text, finite score, optional source URL, deterministic JSON metadata, and a `used_for_generation` flag.

### `feedback`

Feedback belongs to an existing trace through a cascading foreign key. A record has its own UUID and UTC timestamp and requires a rating (`-1` or `1`), a non-empty comment, or both. Metadata is deterministic JSON. Day 31 provides `TraceStore.record_feedback()` and read methods; collection through an HTTP endpoint is intentionally deferred.

## Atomicity and Failure Behavior

`TraceStore.record_trace()` validates the models, rank sequence, per-trace chunk uniqueness, count parity, finite numbers, and JSON metadata before committing. The trace and all retrieved chunks use one SQLite transaction. If any child row fails, the parent insert rolls back.

FastAPI stores:

- a successful retrieval with its ranked chunks;
- a successful query with its answer, ranked chunks, and generation-use flags;
- a validation/retrieval error with zero chunks when retrieval never completed;
- a generation error with any chunks already retrieved.

Component timings completed before an error are retained. A failing embedding, dense-search, BM25, reranker, or generation stage also records its elapsed time through a `finally` path. Stages that never started remain null.

If trace persistence fails, the endpoint returns HTTP 503 with `Unable to persist query trace.` This fail-closed choice makes the acceptance statement meaningful: a successful endpoint response implies its trace was written. FastAPI request-model failures such as a missing field or `top_k` outside 1–20 return 422 before the endpoint handler starts and are outside the Day 31 trace boundary.

## Storage and Concurrency

The store opens short-lived connections, enables foreign keys on every connection, configures a finite busy timeout, and switches initialized databases to WAL journal mode. This supports the API's threaded synchronous handlers without sharing SQLite connection objects. Reads are deterministic; `list_traces()` returns newest-first results and caps its limit at 1,000.

The schema is validated at application startup. Missing tables, unexpected columns, invalid foreign keys, an unsupported version, or a database path that is a directory fail startup. Version 1 databases migrate through version 2, and version 2 databases add the six nullable timing columns in version 3. Migration tests preserve trace, chunk, and feedback relationships; newer unknown versions are rejected rather than guessed at.

## Component Timing Semantics

`TraceContext` starts once per accepted endpoint request and uses `time.perf_counter()` so duration measurement is not affected by wall-clock changes. Its context manager accumulates repeated measurements of a named stage and records elapsed time even when the body raises. The retriever timing dictionary and context snapshot are validated as finite, non-negative milliseconds.

The response shape is stable across pipeline types:

```json
{
  "component_latencies": {
    "embedding_ms": 7.2,
    "dense_ms": 4.1,
    "bm25_ms": null,
    "fusion_ms": null,
    "reranker_ms": null,
    "generation_ms": 6.5
  }
}
```

`/retrieve` remains dense and normally populates embedding and dense. `/query` can select dense, hybrid, or reranked execution, so its applicable BM25, fusion, reranker, and generation values are populated. `latency_ms` measures overall handler work through response preparation but is captured before SQLite persistence. Component values are diagnostic stages and should not be assumed to sum exactly to the total because resource initialization, orchestration, and serialization overhead remains outside them. In particular, first-time cross-encoder loading is included in total latency but not reranker scoring latency; debug cache flags make that cold load visible.

## Configuration and Commands

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAGOPS_TRACE_DB_PATH` | `data/traces/ragops_traces.sqlite3` | Local database path. Relative paths resolve from the project root. |
| `RAGOPS_PIPELINE_NAME` | `dense_baseline` | Pipeline name stored for dense-only `/retrieve`; `/query` stores the selected config name. |
| `RAGOPS_PIPELINE_VERSION` | `1.0.0` | Version stored for `/retrieve`; `/query` stores the selected config version. |

```bash
make init-trace-store
make validate-trace-store
make test-tracing
```

`make init-trace-store` creates, migrates, and validates the configured database. `make validate-trace-store` is read-only with respect to schema/data and requires the database to exist. Both commands print table counts. Set `TRACE_DB_PATH=/some/path.sqlite3` on the Make invocation to inspect a different database.

Docker Compose sets the database path to `/app/data/traces/ragops_traces.sqlite3` and mounts the named `ragops_trace_data` volume. Local database, WAL, and shared-memory files under `data/traces` are ignored by Git.

## Privacy and Operational Limits

Traces intentionally contain raw queries, generated answers, complete retrieved text, metadata, and error messages. Treat the database as potentially sensitive application data, restrict filesystem/volume access, define retention before production use, and avoid placing secrets or personal data in queries.

Days 31–33 do not provide retention jobs, redaction, encryption, feedback endpoints, trace search APIs, dashboard trace views, distributed tracing, persisted token/cost accounting, or PostgreSQL replication. Those are separate operational milestones rather than properties of this local SQLite baseline.
