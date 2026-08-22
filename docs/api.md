# API

FastAPI exposes four endpoints:

| Endpoint | Purpose | Trace |
| --- | --- | --- |
| `GET /health` | Process status and package version | No |
| `POST /retrieve` | Dense retrieval | Yes, but no trace ID in the response |
| `POST /route` | Route decision or deterministic refusal | No |
| `POST /query` | Retrieve and generate a cited answer | Yes; ID in body and `X-Trace-ID` |

`/health` does not check Qdrant or an LLM provider.

## Query

```json
{
  "query": "How does reciprocal rank fusion work?",
  "top_k": 5,
  "config": "hybrid_rrf_cross_encoder",
  "debug": true
}
```

`top_k` must be 1–20. `config` defaults to `dense_baseline`.

| Config | Pipeline | Lifecycle |
| --- | --- | --- |
| `dense_baseline` | Dense | `approved` |
| `hybrid_rrf` | Dense 20 + BM25 20 -> RRF 10 | `rejected` |
| `hybrid_rrf_cross_encoder` | Dense 25 + BM25 25 -> RRF 25 -> rerank 5 | `evaluated` |

The response contains:

- trace, config, version, and low-level pipeline route;
- answer, citations, chunks, and used chunk IDs;
- total and component latency;
- generation cost and token/rate provenance; and
- optional config depths, provider identity, and cache hits under `debug`.

Selecting a rejected config makes it executable for comparison. It does not promote it.

## Routing

```json
{"query": "What is FastAPI?"}
```

`POST /route` returns the decision, reason codes, probe features, top-two chunk IDs/scores, and probe timing. `NO_ANSWER` also returns:

```json
{
  "answer": "I do not know based on the available FastAPI, MLflow, and Qdrant documentation.",
  "prompt_version": "no_answer_v1",
  "prompt_sha256": "<sha256>",
  "generated_by": "deterministic_policy"
}
```

No LLM is called for that branch. FAST, STANDARD, and CAREFUL are not executed by `/route`; the dashboard submits a second explicit `/query` request.

## Generation

Set one provider per API process:

```text
RAGOPS_LLM_PROVIDER=template | openai | gemini
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

`template` is offline and returns a placeholder. OpenAI and Gemini return real answers. Cost covers generation tokens only; see [cost estimation](cost_estimation.md).

## Errors

The trace behavior below applies to `/retrieve` and `/query`. `/route` is never traced.

| Case | HTTP | Trace behavior |
| --- | ---: | --- |
| Invalid body, config, or depth | 422 | Handler not entered; no trace |
| Accepted but invalid query | 400 | Stored error trace |
| Missing Qdrant/index/model resource | 503 | Stored error trace |
| Retrieval or generation failure | 503 | Stored partial evidence/timings |
| Trace write failure | 503 | No success returned |

Internal errors stay in SQLite; clients receive stable messages. `/query` errors include `X-Trace-ID` after the handler starts.

## Resource lifecycle

- Qdrant clients are request-scoped and closed on success or failure.
- BM25 and cross-encoder resources load lazily and are cached per process.
- Compose mounts `data/processed` read-only; build the local BM25 artifact first.

## Commands

```bash
make serve
make test-query-endpoint
make test-api-ci PYTHON=.venv/bin/python
make evaluate-api
```

`make evaluate-api` needs a running full-corpus API and a host-visible trace database.
