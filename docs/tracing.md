# SQLite Request Tracing

## Day 31 Contract

The Day 31 trace store makes accepted online retrieval and generation attempts durable. FastAPI creates a UUID and UTC start timestamp when a valid request enters `POST /retrieve` or `POST /query`, then records exactly one terminal `success` or `error` trace before returning.

This milestone records whole-request latency only. Day 32 owns embedding, dense, BM25, reranker, and generation timings. Day 33 owns returning the trace ID and selected route in the API response.

## Schema

The database uses SQLite schema version 2 and contains three application tables:

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

If trace persistence fails, the endpoint returns HTTP 503 with `Unable to persist query trace.` This fail-closed choice makes the acceptance statement meaningful: a successful endpoint response implies its trace was written. FastAPI request-model failures such as a missing field or `top_k` outside 1–20 return 422 before the endpoint handler starts and are outside the Day 31 trace boundary.

## Storage and Concurrency

The store opens short-lived connections, enables foreign keys on every connection, configures a finite busy timeout, and switches initialized databases to WAL journal mode. This supports the API's threaded synchronous handlers without sharing SQLite connection objects. Reads are deterministic; `list_traces()` returns newest-first results and caps its limit at 1,000.

The schema is validated at application startup. Missing tables, unexpected columns, invalid foreign keys, an unsupported version, or a database path that is a directory fail startup. Version 1 databases are migrated transactionally to version 2 without deleting trace rows; newer unknown versions are rejected rather than guessed at.

## Configuration and Commands

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAGOPS_TRACE_DB_PATH` | `data/traces/ragops_traces.sqlite3` | Local database path. Relative paths resolve from the project root. |
| `RAGOPS_PIPELINE_NAME` | `dense_baseline` | Pipeline name stored on each trace. |
| `RAGOPS_PIPELINE_VERSION` | `1.0.0` | Semantic pipeline version stored on each trace. |

```bash
make init-trace-store
make validate-trace-store
make test-tracing
```

`make init-trace-store` creates, migrates, and validates the configured database. `make validate-trace-store` is read-only with respect to schema/data and requires the database to exist. Both commands print table counts. Set `TRACE_DB_PATH=/some/path.sqlite3` on the Make invocation to inspect a different database.

Docker Compose sets the database path to `/app/data/traces/ragops_traces.sqlite3` and mounts the named `ragops_trace_data` volume. Local database, WAL, and shared-memory files under `data/traces` are ignored by Git.

## Privacy and Operational Limits

Traces intentionally contain raw queries, generated answers, complete retrieved text, metadata, and error messages. Treat the database as potentially sensitive application data, restrict filesystem/volume access, define retention before production use, and avoid placing secrets or personal data in queries.

Day 31 does not provide retention jobs, redaction, encryption, feedback endpoints, trace search APIs, dashboard views, distributed tracing, per-component timings, token/cost accounting, or PostgreSQL replication. Those are separate operational milestones rather than properties of this local SQLite baseline.
