import pytest
from fastapi.testclient import TestClient

import ragops.app as app_module
from ragops import __version__
from ragops.api.pipelines import PipelineExecutionError, PipelineResourceError
from ragops.generation.client import GenerationResult
from ragops.retrieval.dense import RetrievedChunk
from ragops.tracing.store import PipelineIdentity


class RecordingTraceStore:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def record_trace(self, trace, chunks):
        if self.fail:
            raise OSError("trace database unavailable")
        self.calls.append((trace, chunks))
        return trace.trace_id


class FakeDefinition:
    def __init__(self, name, route):
        self.name = name
        self.route = route
        self.config = type(
            "Config",
            (),
            {
                "version": "1.0.0",
                "status": "approved" if route == "dense" else "evaluated",
                "retriever_interface": "common_v1",
            },
        )()

    @property
    def identity(self):
        return PipelineIdentity(name=self.name, version=self.config.version)

    def candidate_depths(self):
        return {
            "dense": 10 if self.route == "dense" else 25,
            **({"bm25": 25, "fusion": 25} if self.route != "dense" else {}),
            **({"reranker_candidates": 25, "reranker_output": 5} if self.route == "reranked" else {}),
        }


class FakeExecution:
    def __init__(self, definition, chunks, cache_hits=None):
        self.definition = definition
        self.chunks = chunks
        self._cache_hits = cache_hits or {}

    def cache_status(self):
        return self._cache_hits


class RecordingPipelineRuntime:
    routes = {
        "dense_baseline": "dense",
        "hybrid_rrf": "hybrid",
        "hybrid_rrf_cross_encoder": "reranked",
    }

    def __init__(self, chunks=None, error=None):
        self.chunks = list(chunks if chunks is not None else [make_chunk()])
        self.error = error
        self.calls = []

    def select(self, name):
        return FakeDefinition(name, self.routes[name])

    def retrieve(self, definition, query, top_k, timings):
        self.calls.append((definition.name, query, top_k))
        if definition.route == "dense":
            timings.update({"embedding_ms": 1.5, "dense_ms": 0.5})
        else:
            timings.update({"embedding_ms": 1.5, "dense_ms": 0.5, "bm25_ms": 0.25, "fusion_ms": 0.1})
        if definition.route == "reranked":
            timings["reranker_ms"] = 2.5
        if self.error is not None:
            raise self.error
        cache_hits = {}
        if definition.route != "dense":
            cache_hits["bm25_index"] = True
        if definition.route == "reranked":
            cache_hits["reranker_model"] = True
        return FakeExecution(definition, self.chunks, cache_hits)


def make_chunk():
    return RetrievedChunk(chunk_id="chunk-1", document_id="doc-1", text="FastAPI is a Python web framework.", score=0.91, rank=1, metadata={"title": "FastAPI Docs", "relative_path": "fastapi/tutorial.md"}, source_url="fastapi/tutorial.md")


def make_client(trace_store=None, pipeline_runtime=None, generation_client=None):
    return TestClient(
        app_module.create_app(
            trace_store=trace_store or RecordingTraceStore(),
            pipeline_runtime=pipeline_runtime or RecordingPipelineRuntime(),
            generation_client=generation_client,
        )
    )


def test_package_version_is_declared():
    assert __version__


def test_health_route_is_registered():
    app = app_module.create_app()

    paths = {route.path for route in app.routes}

    assert "/health" in paths


def test_health_returns_status_and_version():
    client = make_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_docs_are_available():
    client = make_client()

    response = client.get("/docs")

    assert response.status_code == 200
    assert "Swagger UI" in response.text


def test_create_app_uses_injected_generation_client():
    generation_client = object()
    trace_store = RecordingTraceStore()
    pipeline_runtime = RecordingPipelineRuntime()

    app = app_module.create_app(generation_client=generation_client, trace_store=trace_store, pipeline_runtime=pipeline_runtime)

    assert app.state.generation_client is generation_client
    assert app.state.trace_store is trace_store
    assert app.state.pipeline_runtime is pipeline_runtime
    assert app.state.pipeline_identity.name == "dense_baseline"
    assert app.state.pipeline_identity.version == "1.0.0"


def test_get_qdrant_url_uses_local_default(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)

    assert app_module.get_qdrant_url() == app_module.DEFAULT_QDRANT_URL


def test_get_qdrant_url_uses_configured_value(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "  http://qdrant:6333/  ")

    assert app_module.get_qdrant_url() == "http://qdrant:6333"


def test_retrieve_chunks_uses_configured_qdrant_and_closes_client(monkeypatch):
    calls = {}

    class FakeClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    client = FakeClient()

    def fake_create_qdrant_client(qdrant_url):
        calls["qdrant_url"] = qdrant_url
        return client

    timings = {}

    def fake_retrieve_dense(query, client, top_k, timings):
        calls["retrieve"] = (query, client, top_k, timings)
        return [make_chunk()]

    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setattr(app_module, "create_qdrant_client", fake_create_qdrant_client)
    monkeypatch.setattr(app_module, "retrieve_dense", fake_retrieve_dense)

    chunks = app_module.retrieve_chunks("What is FastAPI?", 1, timings=timings)

    assert calls["qdrant_url"] == "http://qdrant:6333"
    assert calls["retrieve"] == ("What is FastAPI?", client, 1, timings)
    assert chunks[0].chunk_id == "chunk-1"
    assert client.closed


def test_retrieve_chunks_closes_client_after_failure(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    client = FakeClient()

    def fake_retrieve_dense(**kwargs):
        raise RuntimeError("search failed")

    monkeypatch.setattr(app_module, "create_qdrant_client", lambda qdrant_url: client)
    monkeypatch.setattr(app_module, "retrieve_dense", fake_retrieve_dense)

    with pytest.raises(RuntimeError, match="search failed"):
        app_module.retrieve_chunks("What is FastAPI?", 1)

    assert client.closed


def test_retrieve_returns_chunks(monkeypatch):
    calls = {}
    trace_store = RecordingTraceStore()

    def fake_retrieve_chunks(query, top_k, timings):
        calls["query"] = query
        calls["top_k"] = top_k
        timings.update({"embedding_ms": 1.25, "dense_ms": 0.75})
        return [make_chunk()]

    monkeypatch.setattr(app_module, "retrieve_chunks", fake_retrieve_chunks)
    client = make_client(trace_store)

    response = client.post("/retrieve", json={"query": "What is FastAPI?", "top_k": 1})
    body = response.json()

    assert response.status_code == 200
    assert calls == {"query": "What is FastAPI?", "top_k": 1}
    assert body["query"] == "What is FastAPI?"
    assert body["top_k"] == 1
    assert body["latency_ms"] >= 0
    assert body["component_latencies"] == {
        "embedding_ms": 1.25,
        "dense_ms": 0.75,
        "bm25_ms": None,
        "fusion_ms": None,
        "reranker_ms": None,
        "generation_ms": None,
    }
    assert body["chunks"][0]["chunk_id"] == "chunk-1"
    assert body["chunks"][0]["source_url"] == "fastapi/tutorial.md"
    trace, stored_chunks = trace_store.calls[0]
    assert trace.endpoint == "retrieve"
    assert trace.status == "success"
    assert trace.retrieved_chunk_count == 1
    assert trace.embedding_ms == 1.25
    assert trace.dense_ms == 0.75
    assert trace.component_latencies().recorded() == {"embedding_ms": 1.25, "dense_ms": 0.75}
    assert stored_chunks[0].chunk_id == "chunk-1"
    assert stored_chunks[0].used_for_generation is False


@pytest.mark.parametrize(("endpoint", "top_k"), [("/retrieve", 0), ("/retrieve", 21), ("/query", 0), ("/query", 21)])
def test_top_k_outside_supported_range_returns_422(endpoint, top_k):
    client = make_client()

    response = client.post(endpoint, json={"query": "What is FastAPI?", "top_k": top_k})

    assert response.status_code == 422


def test_query_returns_production_response_and_selected_trace_identity(monkeypatch):
    chunk = make_chunk()
    calls = {}
    generation_client = object()

    def fake_generate_answer(query, chunks, client):
        calls["generate"] = (query, chunks, client)
        citations = [{"citation_id": "[1]", "document_id": "doc-1", "title": "FastAPI Docs", "url": "fastapi/tutorial.md", "metadata": {"title": "FastAPI Docs"}, "chunk_ids": ["chunk-1"]}]
        return GenerationResult(answer="FastAPI is a Python web framework. [1]", citations=citations, citation_text="[1] FastAPI Docs - fastapi/tutorial.md", used_chunk_ids=["chunk-1"])

    monkeypatch.setattr(app_module, "generate_answer", fake_generate_answer)
    trace_store = RecordingTraceStore()
    pipeline_runtime = RecordingPipelineRuntime([chunk])
    client = make_client(trace_store, pipeline_runtime=pipeline_runtime, generation_client=generation_client)

    response = client.post("/query", json={"query": "What is FastAPI?", "top_k": 1})
    body = response.json()

    assert response.status_code == 200
    assert pipeline_runtime.calls == [("dense_baseline", "What is FastAPI?", 1)]
    assert calls["generate"] == ("What is FastAPI?", [chunk], generation_client)
    assert body["trace_id"] == response.headers["x-trace-id"]
    assert body["route"] == "dense"
    assert body["config"] == "dense_baseline"
    assert body["config_version"] == "1.0.0"
    assert body["answer"] == "FastAPI is a Python web framework. [1]"
    assert body["citations"][0]["citation_id"] == "[1]"
    assert body["citation_text"] == "[1] FastAPI Docs - fastapi/tutorial.md"
    assert body["chunks"][0]["chunk_id"] == "chunk-1"
    assert body["used_chunk_ids"] == ["chunk-1"]
    assert body["latency_ms"] >= 0
    assert body["component_latencies"]["embedding_ms"] == 1.5
    assert body["component_latencies"]["dense_ms"] == 0.5
    assert body["component_latencies"]["generation_ms"] >= 0
    assert body["component_latencies"]["bm25_ms"] is None
    assert body["cost"] == {
        "amount_usd": None,
        "currency": "USD",
        "status": "unavailable",
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    assert body["debug"] is None
    trace, stored_chunks = trace_store.calls[0]
    assert trace.endpoint == "query"
    assert trace.status == "success"
    assert trace.answer == "FastAPI is a Python web framework. [1]"
    assert trace.embedding_ms == 1.5
    assert trace.dense_ms == 0.5
    assert trace.generation_ms == body["component_latencies"]["generation_ms"]
    assert trace.pipeline_name == "dense_baseline"
    assert trace.pipeline_version == "1.0.0"
    assert stored_chunks[0].used_for_generation is True


@pytest.mark.parametrize(
    ("config", "route", "expected_timings"),
    [
        ("dense_baseline", "dense", {"embedding_ms", "dense_ms", "generation_ms"}),
        ("hybrid_rrf", "hybrid", {"embedding_ms", "dense_ms", "bm25_ms", "fusion_ms", "generation_ms"}),
        (
            "hybrid_rrf_cross_encoder",
            "reranked",
            {"embedding_ms", "dense_ms", "bm25_ms", "fusion_ms", "reranker_ms", "generation_ms"},
        ),
    ],
)
def test_query_runs_dense_hybrid_and_reranked_configs(config, route, expected_timings):
    trace_store = RecordingTraceStore()
    pipeline_runtime = RecordingPipelineRuntime()
    client = make_client(trace_store, pipeline_runtime=pipeline_runtime)

    response = client.post(
        "/query",
        json={"query": "What is FastAPI?", "top_k": 1, "config": config, "debug": True},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["config"] == config
    assert body["route"] == route
    assert body["cost"]["amount_usd"] == 0.0
    assert body["cost"]["status"] == "zero_cost"
    assert {name for name, value in body["component_latencies"].items() if value is not None} == expected_timings
    assert body["debug"]["pipeline_id"] == f"{config}@1.0.0"
    assert body["debug"]["requested_top_k"] == 1
    assert body["debug"]["returned_chunks"] == 1
    assert body["debug"]["generation_provider"] == "template"
    assert trace_store.calls[0][0].pipeline_name == config


def test_unknown_query_config_is_rejected_before_endpoint_and_not_traced():
    trace_store = RecordingTraceStore()
    client = make_client(trace_store)

    response = client.post(
        "/query",
        json={"query": "What is FastAPI?", "config": "unknown"},
    )

    assert response.status_code == 422
    assert trace_store.calls == []


def test_empty_query_returns_400(monkeypatch):
    def fake_retrieve_chunks(query, top_k, timings):
        raise ValueError("query must not be empty.")

    monkeypatch.setattr(app_module, "retrieve_chunks", fake_retrieve_chunks)
    trace_store = RecordingTraceStore()
    client = make_client(trace_store)

    response = client.post("/retrieve", json={"query": "   ", "top_k": 1})

    assert response.status_code == 400
    assert response.json()["detail"] == "query must not be empty."
    trace, chunks = trace_store.calls[0]
    assert trace.query == "   "
    assert trace.status == "error"
    assert trace.error_type == "ValueError"
    assert trace.component_latencies().recorded() == {}
    assert chunks == []


def test_retrieve_failure_returns_503(monkeypatch):
    def fake_retrieve_chunks(query, top_k, timings):
        timings["embedding_ms"] = 0.5
        raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr(app_module, "retrieve_chunks", fake_retrieve_chunks)
    trace_store = RecordingTraceStore()
    client = make_client(trace_store)

    response = client.post("/retrieve", json={"query": "What is FastAPI?", "top_k": 1})

    assert response.status_code == 503
    assert response.json()["detail"] == "Unable to retrieve chunks."
    assert trace_store.calls[0][0].status == "error"
    assert trace_store.calls[0][0].error_message == "qdrant unavailable"
    assert trace_store.calls[0][0].embedding_ms == 0.5


def test_query_generation_failure_returns_503(monkeypatch):
    def fake_generate_answer(query, chunks, client):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(app_module, "generate_answer", fake_generate_answer)
    trace_store = RecordingTraceStore()
    pipeline_runtime = RecordingPipelineRuntime()
    client = make_client(trace_store, pipeline_runtime=pipeline_runtime)

    response = client.post("/query", json={"query": "What is FastAPI?", "top_k": 1})

    assert response.status_code == 503
    assert response.json()["detail"] == "Unable to generate answer."
    trace, stored_chunks = trace_store.calls[0]
    assert trace.status == "error"
    assert trace.retrieved_chunk_count == 1
    assert trace.embedding_ms == 1.5
    assert trace.dense_ms == 0.5
    assert trace.generation_ms is not None
    assert stored_chunks[0].chunk_id == "chunk-1"
    assert response.headers["x-trace-id"] == trace.trace_id


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (PipelineResourceError("index unavailable"), "Selected query pipeline is unavailable."),
        (PipelineExecutionError("qdrant unavailable"), "Unable to retrieve chunks with the selected pipeline."),
    ],
)
def test_query_pipeline_failures_are_stage_aware_and_traced(error, detail):
    trace_store = RecordingTraceStore()
    runtime = RecordingPipelineRuntime(error=error)
    client = make_client(trace_store, pipeline_runtime=runtime)

    response = client.post(
        "/query",
        json={"query": "What is RRF?", "top_k": 1, "config": "hybrid_rrf"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == detail
    trace, chunks = trace_store.calls[0]
    assert response.headers["x-trace-id"] == trace.trace_id
    assert trace.pipeline_name == "hybrid_rrf"
    assert trace.status == "error"
    assert trace.error_message == str(error)
    assert chunks == []


def test_trace_persistence_failure_returns_503(monkeypatch):
    monkeypatch.setattr(app_module, "retrieve_chunks", lambda query, top_k, timings: [make_chunk()])
    client = make_client(RecordingTraceStore(fail=True))

    response = client.post("/retrieve", json={"query": "What is FastAPI?", "top_k": 1})

    assert response.status_code == 503
    assert response.json()["detail"] == "Unable to persist query trace."
