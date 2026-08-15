from pathlib import Path

import pytest

from ragops.api.pipelines import (
    DEFAULT_PIPELINE_CONFIG_PATHS,
    PipelineExecutionError,
    PipelineResourceError,
    PipelineRuntime,
    configured_project_root,
)
from ragops.retrieval.dense import RetrievedChunk


class FakeQdrantClient:
    def __init__(self, url):
        self.url = url
        self.closed = False

    def close(self):
        self.closed = True


class FakeRetriever:
    def __init__(self, config_name, calls, error=None):
        self.config_name = config_name
        self.calls = calls
        self.error = error

    def retrieve(self, query, top_k, timings):
        self.calls.append((self.config_name, query, top_k, timings))
        if self.error:
            raise self.error
        return [{"config": self.config_name, "rank": 1}]


def make_runtime(monkeypatch, retriever_error=None, index_error=None, reranker_error=None):
    clients = []
    loaded_indexes = []
    validated_indexes = []
    loaded_rerankers = []
    built_retrievers = []
    retrieval_calls = []

    def client_factory(url):
        client = FakeQdrantClient(url)
        clients.append(client)
        return client

    def index_loader(path):
        loaded_indexes.append(path)
        if index_error:
            raise index_error
        return object()

    def index_validator(index, config):
        validated_indexes.append((index, config.name))

    def reranker_factory(config):
        loaded_rerankers.append(config.model)
        if reranker_error:
            raise reranker_error
        return object()

    def retriever_factory(config, client, index, reranker):
        built_retrievers.append((config.name, client, index, reranker))
        return FakeRetriever(config.name, retrieval_calls, error=retriever_error)

    monkeypatch.setenv("QDRANT_URL", "http://runtime-qdrant:6333/")
    runtime = PipelineRuntime(
        project_root=Path.cwd(),
        client_factory=client_factory,
        index_loader=index_loader,
        index_validator=index_validator,
        reranker_factory=reranker_factory,
        retriever_factory=retriever_factory,
    )
    evidence = {
        "clients": clients,
        "loaded_indexes": loaded_indexes,
        "validated_indexes": validated_indexes,
        "loaded_rerankers": loaded_rerankers,
        "built_retrievers": built_retrievers,
        "retrieval_calls": retrieval_calls,
    }
    return runtime, evidence


def test_runtime_loads_exact_validated_query_catalog(monkeypatch):
    runtime, _ = make_runtime(monkeypatch)

    assert runtime.available_configs == tuple(DEFAULT_PIPELINE_CONFIG_PATHS)
    assert runtime.select("dense_baseline").route == "dense"
    assert runtime.select("hybrid_rrf").route == "hybrid"
    reranked = runtime.select("hybrid_rrf_cross_encoder")
    assert reranked.route == "reranked"
    assert reranked.identity.name == "hybrid_rrf_cross_encoder"
    assert reranked.identity.version == "1.0.0"
    assert reranked.candidate_depths() == {
        "dense": 25,
        "bm25": 25,
        "fusion": 25,
        "reranker_candidates": 25,
        "reranker_output": 5,
    }

    with pytest.raises(ValueError, match="Unsupported query config"):
        runtime.select("unknown")

    with pytest.raises(ValueError, match="must configure"):
        PipelineRuntime(project_root=Path.cwd(), config_paths={})


def test_runtime_project_root_uses_deployment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGOPS_PROJECT_ROOT", str(tmp_path))

    assert configured_project_root() == tmp_path.resolve()
    assert configured_project_root(Path.cwd()) == Path.cwd().resolve()


def test_dense_execution_only_builds_request_scoped_qdrant(monkeypatch):
    runtime, evidence = make_runtime(monkeypatch)
    definition = runtime.select("dense_baseline")
    timings = {}

    execution = runtime.retrieve(definition, "What is FastAPI?", 3, timings=timings)

    assert execution.chunks == [{"config": "dense_baseline", "rank": 1}]
    assert execution.cache_status() == {}
    assert evidence["clients"][0].url == "http://runtime-qdrant:6333"
    assert evidence["clients"][0].closed
    assert evidence["loaded_indexes"] == []
    assert evidence["loaded_rerankers"] == []
    assert evidence["built_retrievers"][0][2:] == (None, None)


def test_initial_probe_uses_configured_dense_depth_only_and_closes_its_client(monkeypatch):
    runtime, evidence = make_runtime(monkeypatch)
    probe_chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="document-1",
            text="First result",
            score=0.9,
            rank=1,
            metadata={},
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            document_id="document-1",
            text="Second result",
            score=0.75,
            rank=2,
            metadata={},
        ),
        RetrievedChunk(
            chunk_id="chunk-3",
            document_id="document-1",
            text="Third result",
            score=0.7,
            rank=3,
            metadata={},
        ),
    ]
    evidence["built_retrievers"].clear()

    class ProbeRetriever(FakeRetriever):
        def retrieve(self, query, top_k, timings):
            self.calls.append((self.config_name, query, top_k, timings))
            timings.update({"embedding_ms": 2.0, "dense_ms": 1.0})
            return probe_chunks

    def retriever_factory(config, client, index, reranker):
        evidence["built_retrievers"].append((config.name, client, index, reranker))
        return ProbeRetriever(config.name, evidence["retrieval_calls"])

    runtime.retriever_factory = retriever_factory
    runtime.router_config = runtime.router_config.model_copy(
        update={"probe": runtime.router_config.probe.model_copy(update={"top_k": 3})}
    )
    clock_values = iter([1.0, 1.01])

    result = runtime.initial_probe("How does retrieval work?", clock=lambda: next(clock_values))

    assert result.features.retrieval_confidence.top_score == 0.9
    assert result.features.retrieval_confidence.score_gap == pytest.approx(0.15)
    assert result.features.retrieval_confidence.requested_top_k == 3
    assert result.features.retrieval_confidence.result_count == 3
    assert evidence["retrieval_calls"][0][0:3] == ("dense_baseline", "How does retrieval work?", 3)
    assert evidence["built_retrievers"][0][2:] == (None, None)
    assert evidence["loaded_indexes"] == []
    assert evidence["loaded_rerankers"] == []
    assert evidence["clients"][0].closed


def test_runtime_route_query_runs_one_probe_and_returns_the_deterministic_decision(monkeypatch):
    runtime, evidence = make_runtime(monkeypatch)
    probe_chunks = [
        RetrievedChunk(chunk_id="chunk-1", document_id="document-1", text="First result", score=0.9, rank=1, metadata={}),
        RetrievedChunk(chunk_id="chunk-2", document_id="document-1", text="Second result", score=0.8, rank=2, metadata={}),
    ]

    class ProbeRetriever(FakeRetriever):
        def retrieve(self, query, top_k, timings):
            self.calls.append((self.config_name, query, top_k, timings))
            timings.update({"embedding_ms": 2.0, "dense_ms": 1.0})
            return probe_chunks

    runtime.retriever_factory = lambda config, client, index, reranker: ProbeRetriever(config.name, evidence["retrieval_calls"])

    result = runtime.route_query("FastAPI basics", clock=lambda: 1.0)

    assert result.probe.chunks == tuple(probe_chunks)
    assert result.decision.route == "FAST"
    assert result.decision.reason_code == "fast_conditions_satisfied"
    assert result.decision.pipeline_config == "dense_baseline"
    assert result.decision.reuse_probe
    assert [(name, query, top_k) for name, query, top_k, _ in evidence["retrieval_calls"]] == [
        ("dense_baseline", "FastAPI basics", 2)
    ]
    assert evidence["loaded_indexes"] == []
    assert evidence["loaded_rerankers"] == []
    assert evidence["clients"][0].closed


def test_hybrid_reuses_validated_bm25_index_but_not_qdrant_client(monkeypatch):
    runtime, evidence = make_runtime(monkeypatch)
    definition = runtime.select("hybrid_rrf")

    cold = runtime.retrieve(definition, "What is RRF?", 5)
    warm = runtime.retrieve(definition, "How does fusion work?", 5)

    assert cold.cache_status() == {"bm25_index": False}
    assert warm.cache_status() == {"bm25_index": True}
    assert len(evidence["loaded_indexes"]) == 1
    assert len(evidence["validated_indexes"]) == 1
    assert len(evidence["clients"]) == 2
    assert all(client.closed for client in evidence["clients"])
    assert evidence["built_retrievers"][0][2] is evidence["built_retrievers"][1][2]


def test_reranked_execution_reuses_index_and_cross_encoder(monkeypatch):
    runtime, evidence = make_runtime(monkeypatch)
    definition = runtime.select("hybrid_rrf_cross_encoder")

    cold = runtime.retrieve(definition, "What is reranking?", 5)
    warm = runtime.retrieve(definition, "Why rerank candidates?", 5)

    assert cold.cache_status() == {"bm25_index": False, "reranker_model": False}
    assert warm.cache_status() == {"bm25_index": True, "reranker_model": True}
    assert len(evidence["loaded_indexes"]) == 1
    assert len(evidence["validated_indexes"]) == 1
    assert evidence["loaded_rerankers"] == ["cross-encoder/ms-marco-MiniLM-L-6-v2"]
    first_resources = evidence["built_retrievers"][0][2:]
    second_resources = evidence["built_retrievers"][1][2:]
    assert first_resources == second_resources


def test_resource_failure_is_wrapped_and_closes_qdrant(monkeypatch):
    runtime, evidence = make_runtime(monkeypatch, index_error=OSError("index missing"))

    with pytest.raises(PipelineResourceError, match="Unable to load the BM25 index"):
        runtime.retrieve(runtime.select("hybrid_rrf"), "What is RRF?", 5)

    assert evidence["clients"][0].closed


def test_retrieval_failure_is_wrapped_and_closes_qdrant(monkeypatch):
    runtime, evidence = make_runtime(monkeypatch, retriever_error=TimeoutError("search timed out"))

    with pytest.raises(PipelineExecutionError, match="Retrieval failed for dense_baseline"):
        runtime.retrieve(runtime.select("dense_baseline"), "What is FastAPI?", 5)

    assert evidence["clients"][0].closed


def test_empty_query_is_a_client_error_before_resources_are_created(monkeypatch):
    runtime, evidence = make_runtime(monkeypatch)

    with pytest.raises(ValueError, match="query must not be empty"):
        runtime.retrieve(runtime.select("dense_baseline"), "   ", 5)

    assert evidence["clients"] == []
