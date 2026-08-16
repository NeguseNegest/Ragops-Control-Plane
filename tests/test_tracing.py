import sqlite3
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import ragops.app as app_module
from ragops.generation.client import GenerationResult
from ragops.generation.cost import GenerationCost
from ragops.retrieval.dense import RetrievedChunk
from ragops.tracing import store as store_module
from ragops.tracing.context import COMPONENT_TIMING_FIELDS
from ragops.tracing.store import (
    COST_TRACE_FIELDS,
    TRACE_SCHEMA_VERSION,
    FeedbackRecord,
    RetrievedChunkTrace,
    TraceRecord,
    TraceStore,
    configured_pipeline_identity,
    configured_trace_db_path,
)


def trace_cost_fields(cost):
    return {
        "generation_provider": cost.provider,
        "generation_model": cost.model,
        "cost_amount_usd": cost.amount_usd,
        "cost_currency": cost.currency,
        "cost_status": cost.status,
        "cost_input_tokens": cost.input_tokens,
        "cost_output_tokens": cost.output_tokens,
        "cost_total_tokens": cost.total_tokens,
        "cost_token_source": cost.token_source,
        "cost_token_estimator": cost.token_estimator,
        "cost_pricing_source": cost.pricing_source,
        "cost_price_table_id": cost.price_table_id,
        "cost_input_usd_per_million_tokens": cost.input_usd_per_million_tokens,
        "cost_output_usd_per_million_tokens": cost.output_usd_per_million_tokens,
    }


def schema_without_cost_columns():
    endpoint_check = """    CHECK (
        endpoint = 'query'
        OR (
            answer IS NULL
            AND generation_provider IS NULL
            AND generation_model IS NULL
            AND cost_amount_usd IS NULL
            AND cost_currency IS NULL
            AND cost_status IS NULL
            AND cost_input_tokens IS NULL
            AND cost_output_tokens IS NULL
            AND cost_total_tokens IS NULL
            AND cost_token_source IS NULL
            AND cost_token_estimator IS NULL
            AND cost_pricing_source IS NULL
            AND cost_price_table_id IS NULL
            AND cost_input_usd_per_million_tokens IS NULL
            AND cost_output_usd_per_million_tokens IS NULL
        )
    )"""
    schema = store_module._TRACE_SCHEMA_SQL.replace(endpoint_check, "    CHECK (endpoint = 'query' OR answer IS NULL)")
    return "\n".join(
        line
        for line in schema.splitlines()
        if not any(line.strip().startswith(f"{field} ") for field in COST_TRACE_FIELDS)
    )


def make_trace(**updates):
    created_at = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    payload = {
        "trace_id": str(uuid4()),
        "created_at": created_at,
        "completed_at": created_at + timedelta(milliseconds=25),
        "endpoint": "query",
        "query": "What is FastAPI?",
        "requested_top_k": 2,
        "pipeline_name": "dense_baseline",
        "pipeline_version": "1.0.0",
        "status": "success",
        "retrieved_chunk_count": 2,
        "answer": "FastAPI is a Python web framework.",
        "total_latency_ms": 25.0,
        "error_type": None,
        "error_message": None,
    }
    payload.update(updates)
    return TraceRecord.model_validate(payload)


def make_chunks():
    return [
        RetrievedChunkTrace(
            rank=1,
            chunk_id="chunk-1",
            document_id="doc-1",
            text="FastAPI is a Python web framework.",
            score=0.91,
            source_url="fastapi/tutorial.md",
            metadata={"title": "FastAPI", "section": 1},
            used_for_generation=True,
        ),
        RetrievedChunkTrace(
            rank=2,
            chunk_id="chunk-2",
            document_id="doc-2",
            text="Qdrant is a vector database.",
            score=0.72,
            metadata={"title": "Qdrant"},
        ),
    ]


def make_store(tmp_path):
    return TraceStore(tmp_path / "traces.sqlite3").initialize()


def test_initialize_creates_versioned_schema_and_required_tables(tmp_path):
    store = make_store(tmp_path)

    assert store.validate_schema() is True
    assert store.counts() == {"feedback": 0, "retrieved_chunks": 0, "traces": 0}
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == TRACE_SCHEMA_VERSION
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
    assert {"traces", "retrieved_chunks", "feedback"}.issubset(tables)


def test_trace_and_ranked_chunks_round_trip(tmp_path):
    store = make_store(tmp_path)
    cost = GenerationCost(
        amount_usd=0.000004,
        status="estimated",
        provider="openai",
        model="gpt-5-nano",
        input_tokens=40,
        output_tokens=5,
        total_tokens=45,
        token_source="provider_reported",
        pricing_source="model_cost_table",
        price_table_id="generation_model_costs@1.0.0",
        input_usd_per_million_tokens=0.05,
        output_usd_per_million_tokens=0.4,
    )
    trace = make_trace(embedding_ms=4.0, dense_ms=3.0, generation_ms=10.0, **trace_cost_fields(cost))

    assert store.record_trace(trace, make_chunks()) == trace.trace_id

    loaded = store.get_trace(trace.trace_id)
    chunks = store.list_retrieved_chunks(trace.trace_id)
    assert loaded == trace
    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1", "chunk-2"]
    assert chunks[0].metadata == {"section": 1, "title": "FastAPI"}
    assert chunks[0].used_for_generation is True
    assert chunks[1].used_for_generation is False
    assert loaded.component_latencies().recorded() == {
        "embedding_ms": 4.0,
        "dense_ms": 3.0,
        "generation_ms": 10.0,
    }
    assert loaded.generation_cost() == cost
    assert store.counts() == {"feedback": 0, "retrieved_chunks": 2, "traces": 1}


def test_failed_trace_preserves_whitespace_query_and_error(tmp_path):
    store = make_store(tmp_path)
    trace = make_trace(
        endpoint="retrieve",
        query="   ",
        status="error",
        retrieved_chunk_count=0,
        answer=None,
        error_type="ValueError",
        error_message="query must not be empty.",
    )

    store.record_trace(trace)

    assert store.get_trace(trace.trace_id) == trace


def test_trace_rejects_invalid_status_fields_and_naive_timestamps():
    with pytest.raises(ValidationError, match="Error traces must contain"):
        make_trace(status="error")
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_trace(created_at=datetime(2026, 8, 14, 12, 0))
    with pytest.raises(ValidationError, match="greater than or equal"):
        make_trace(embedding_ms=-1.0)
    with pytest.raises(ValidationError, match="finite"):
        make_trace(generation_ms=float("inf"))
    with pytest.raises(ValidationError, match="generation latency"):
        make_trace(endpoint="retrieve", answer=None, generation_ms=1.0)
    zero_cost = GenerationCost(
        amount_usd=0.0,
        status="zero_cost",
        provider="template",
        model="local-template-v1",
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        token_source="not_applicable",
        pricing_source="not_applicable",
    )
    with pytest.raises(ValidationError, match="generation cost"):
        make_trace(endpoint="retrieve", answer=None, **trace_cost_fields(zero_cost))
    with pytest.raises(ValidationError):
        make_trace(generation_provider="openai", cost_status="estimated")


def test_trace_chunks_require_contiguous_ranks_unique_ids_and_matching_count(tmp_path):
    store = make_store(tmp_path)
    chunks = make_chunks()
    chunks[1] = chunks[1].model_copy(update={"rank": 3})

    with pytest.raises(ValueError, match="contiguous"):
        store.record_trace(make_trace(), chunks)
    duplicate_chunks = make_chunks()
    duplicate_chunks[1] = duplicate_chunks[1].model_copy(update={"chunk_id": "chunk-1"})
    with pytest.raises(ValueError, match="unique chunk IDs"):
        store.record_trace(make_trace(), duplicate_chunks)
    with pytest.raises(ValueError, match="must match"):
        store.record_trace(make_trace(retrieved_chunk_count=1), make_chunks())

    assert store.counts()["traces"] == 0


def test_non_json_metadata_rolls_back_trace_and_chunks(tmp_path):
    store = make_store(tmp_path)
    trace = make_trace()
    chunks = make_chunks()
    chunks[0] = chunks[0].model_copy(update={"metadata": {"invalid": object()}})

    with pytest.raises(ValueError, match="JSON-serializable"):
        store.record_trace(trace, chunks)

    assert store.get_trace(trace.trace_id) is None
    assert store.counts() == {"feedback": 0, "retrieved_chunks": 0, "traces": 0}


def test_duplicate_trace_id_does_not_overwrite_existing_trace(tmp_path):
    store = make_store(tmp_path)
    trace = make_trace()
    store.record_trace(trace, make_chunks())

    with pytest.raises(sqlite3.IntegrityError):
        store.record_trace(trace, make_chunks())

    assert store.get_trace(trace.trace_id) == trace
    assert store.counts()["retrieved_chunks"] == 2


def test_feedback_round_trip_and_foreign_key_enforcement(tmp_path):
    store = make_store(tmp_path)
    trace = make_trace()
    store.record_trace(trace, make_chunks())
    feedback = FeedbackRecord(
        feedback_id=str(uuid4()),
        trace_id=trace.trace_id,
        created_at=datetime(2026, 8, 14, 12, 1, tzinfo=UTC),
        rating=1,
        comment="Useful answer",
        metadata={"source": "pytest"},
    )

    assert store.record_feedback(feedback) == feedback.feedback_id
    assert store.list_feedback(trace.trace_id) == [feedback]

    missing = feedback.model_copy(update={"feedback_id": str(uuid4()), "trace_id": str(uuid4())})
    with pytest.raises(ValueError, match="does not exist"):
        store.record_feedback(missing)


def test_feedback_requires_rating_or_comment():
    with pytest.raises(ValidationError, match="requires a rating or comment"):
        FeedbackRecord(
            feedback_id=str(uuid4()),
            trace_id=str(uuid4()),
            created_at=datetime.now(UTC),
        )


def test_schema_migrates_version_one_without_deleting_related_records(tmp_path):
    path = tmp_path / "version-one.sqlite3"
    trace = make_trace()
    feedback = FeedbackRecord(
        feedback_id=str(uuid4()),
        trace_id=trace.trace_id,
        created_at=datetime(2026, 8, 14, 12, 1, tzinfo=UTC),
        rating=1,
        comment="Keep this feedback",
    )
    legacy_schema = store_module._TRACE_SCHEMA_SQL.replace(
        "query TEXT NOT NULL,",
        "query TEXT NOT NULL CHECK (length(trim(query)) > 0),",
    )
    with sqlite3.connect(path) as connection:
        connection.executescript(legacy_schema)
        connection.execute("PRAGMA user_version = 1")
    legacy_store = TraceStore(path)
    legacy_store.record_trace(trace, make_chunks())
    legacy_store.record_feedback(feedback)

    store = legacy_store.initialize()

    assert store.get_trace(trace.trace_id) == trace
    assert store.list_retrieved_chunks(trace.trace_id) == make_chunks()
    assert store.list_feedback(trace.trace_id) == [feedback]
    assert store.counts() == {"feedback": 1, "retrieved_chunks": 2, "traces": 1}
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == TRACE_SCHEMA_VERSION


def test_schema_migrates_version_two_and_adds_empty_component_timings(tmp_path):
    path = tmp_path / "version-two.sqlite3"
    trace = make_trace(retrieved_chunk_count=0)
    legacy_schema = "\n".join(
        line
        for line in schema_without_cost_columns().splitlines()
        if not any(line.strip().startswith(f"{field} REAL") for field in COMPONENT_TIMING_FIELDS)
    )
    with sqlite3.connect(path) as connection:
        connection.executescript(legacy_schema)
        connection.execute("PRAGMA user_version = 2")
        connection.execute(
            """
            INSERT INTO traces (
                trace_id, created_at, completed_at, endpoint, query, requested_top_k,
                pipeline_name, pipeline_version, status, retrieved_chunk_count,
                answer, total_latency_ms, error_type, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.trace_id,
                trace.created_at.isoformat(),
                trace.completed_at.isoformat(),
                trace.endpoint,
                trace.query,
                trace.requested_top_k,
                trace.pipeline_name,
                trace.pipeline_version,
                trace.status,
                0,
                trace.answer,
                trace.total_latency_ms,
                None,
                None,
            ),
        )

    store = TraceStore(path).initialize()
    migrated = store.get_trace(trace.trace_id)

    assert migrated == trace
    assert migrated.component_latencies().recorded() == {}
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(traces)").fetchall()}
        assert set(COMPONENT_TIMING_FIELDS).issubset(columns)
        assert connection.execute("PRAGMA user_version").fetchone()[0] == TRACE_SCHEMA_VERSION


def test_schema_migrates_version_three_and_preserves_traces_with_empty_cost(tmp_path):
    path = tmp_path / "version-three.sqlite3"
    trace = make_trace(retrieved_chunk_count=0, embedding_ms=2.0, dense_ms=1.0, generation_ms=3.0)
    with sqlite3.connect(path) as connection:
        connection.executescript(schema_without_cost_columns())
        connection.execute("PRAGMA user_version = 3")
        connection.execute(
            """
            INSERT INTO traces (
                trace_id, created_at, completed_at, endpoint, query, requested_top_k,
                pipeline_name, pipeline_version, status, retrieved_chunk_count,
                answer, total_latency_ms, embedding_ms, dense_ms, bm25_ms,
                fusion_ms, reranker_ms, generation_ms, error_type, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.trace_id,
                trace.created_at.isoformat(),
                trace.completed_at.isoformat(),
                trace.endpoint,
                trace.query,
                trace.requested_top_k,
                trace.pipeline_name,
                trace.pipeline_version,
                trace.status,
                0,
                trace.answer,
                trace.total_latency_ms,
                trace.embedding_ms,
                trace.dense_ms,
                None,
                None,
                None,
                trace.generation_ms,
                None,
                None,
            ),
        )

    store = TraceStore(path).initialize()
    migrated = store.get_trace(trace.trace_id)

    assert migrated == trace
    assert migrated.generation_cost() is None
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(traces)").fetchall()}
        assert set(COST_TRACE_FIELDS).issubset(columns)
        assert connection.execute("PRAGMA user_version").fetchone()[0] == TRACE_SCHEMA_VERSION


def test_newer_schema_version_is_rejected(tmp_path):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {TRACE_SCHEMA_VERSION + 1}")

    with pytest.raises(ValueError, match="newer than supported"):
        TraceStore(path).initialize()


def test_configured_path_and_pipeline_identity_use_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGOPS_TRACE_DB_PATH", "custom/traces.sqlite3")
    monkeypatch.setenv("RAGOPS_PIPELINE_NAME", "bm25_baseline")
    monkeypatch.setenv("RAGOPS_PIPELINE_VERSION", "2.1.0")

    assert configured_trace_db_path(project_root=tmp_path) == tmp_path / "custom/traces.sqlite3"
    identity = configured_pipeline_identity()
    assert identity.name == "bm25_baseline"
    assert identity.version == "2.1.0"


def test_real_api_store_records_every_successful_and_failed_query(monkeypatch, tmp_path):
    store = make_store(tmp_path)
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="FastAPI is a Python web framework.",
        score=0.91,
        rank=1,
        metadata={"title": "FastAPI"},
        source_url="fastapi/tutorial.md",
    )
    class Definition:
        name = "dense_baseline"
        route = "dense"
        config = type(
            "Config",
            (),
            {"version": "1.0.0", "status": "approved", "retriever_interface": "common_v1"},
        )()

        @property
        def identity(self):
            return configured_pipeline_identity(name=self.name, version=self.config.version)

    class Execution:
        chunks = [chunk]

        @staticmethod
        def cache_status():
            return {}

    class Runtime:
        @staticmethod
        def select(name):
            assert name == "dense_baseline"
            return Definition()

        @staticmethod
        def retrieve(definition, query, top_k, timings):
            timings.update({"embedding_ms": 2.0, "dense_ms": 1.0})
            return Execution()

    monkeypatch.setattr(
        app_module,
        "generate_answer",
        lambda query, chunks, client: GenerationResult(
            answer="FastAPI is a Python web framework. [1]",
            citations=[],
            citation_text="",
            used_chunk_ids=["chunk-1"],
        ),
    )
    app = app_module.create_app(
        generation_client=object(),
        trace_store=store,
        pipeline_name="dense_baseline",
        pipeline_version="1.0.0",
        pipeline_runtime=Runtime(),
    )
    client = TestClient(app)

    successful = client.post("/query", json={"query": "What is FastAPI?", "top_k": 1})
    app.state.retrieve_chunks = lambda query, top_k, timings: (_ for _ in ()).throw(RuntimeError("offline"))
    failed = client.post("/retrieve", json={"query": "What is Qdrant?", "top_k": 1})

    assert successful.status_code == 200
    assert failed.status_code == 503
    traces = store.list_traces()
    assert len(traces) == 2
    assert {(trace.endpoint, trace.status) for trace in traces} == {("query", "success"), ("retrieve", "error")}
    query_trace = next(trace for trace in traces if trace.endpoint == "query")
    assert query_trace.pipeline_name == "dense_baseline"
    assert query_trace.pipeline_version == "1.0.0"
    assert query_trace.embedding_ms == 2.0
    assert query_trace.dense_ms == 1.0
    assert query_trace.generation_ms is not None
    assert store.list_retrieved_chunks(query_trace.trace_id)[0].used_for_generation is True
    assert store.counts() == {"feedback": 0, "retrieved_chunks": 1, "traces": 2}


def test_list_traces_enforces_bounded_limit(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(ValueError, match="1 through 1000"):
        store.list_traces(limit=0)
