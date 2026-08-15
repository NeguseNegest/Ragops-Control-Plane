import hashlib
import json
import math
from pathlib import Path
from uuid import UUID

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from qdrant_client import QdrantClient

from ragops import __version__
from ragops.app import create_app
from ragops.indexing.qdrant import create_collection, upsert_records
from ragops.tracing.store import TraceStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CI_CONFIG_PATH = PROJECT_ROOT / "configs/ci_small.yaml"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CISmallQuery(StrictModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    query: str = Field(min_length=1)
    vector: list[float] = Field(min_length=1)
    expected_chunk_id: str

    @field_validator("query")
    @classmethod
    def clean_query(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("CI query must not be empty.")
        return value


class CISmallConfig(StrictModel):
    schema_version: int = Field(ge=1)
    collection_name: str = Field(min_length=1)
    vector_size: int = Field(gt=0)
    records_path: Path
    queries: list[CISmallQuery] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_queries(self):
        names = [query.name for query in self.queries]
        texts = [query.query for query in self.queries]
        if len(names) != len(set(names)) or len(texts) != len(set(texts)):
            raise ValueError("CI query names and texts must be unique.")
        if any(len(query.vector) != self.vector_size for query in self.queries):
            raise ValueError("Every CI query vector must match vector_size.")
        if any(not all(math.isfinite(value) for value in query.vector) for query in self.queries):
            raise ValueError("CI query vectors must be finite.")
        return self


class CISmallRecord(StrictModel):
    chunk_id: str
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    chunk_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding: list[float] = Field(min_length=1)
    metadata: dict

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value):
        return str(UUID(value))


class SharedClient:
    """Delegate to one in-memory Qdrant instance while ignoring request closes."""

    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    def close(self):
        return None


def load_ci_fixture():
    payload = yaml.safe_load(CI_CONFIG_PATH.read_text(encoding="utf-8"))
    config = CISmallConfig.model_validate(payload)
    records_path = (PROJECT_ROOT / config.records_path).resolve()
    records = []
    with records_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                record = CISmallRecord.model_validate(json.loads(line))
            except Exception as error:
                raise ValueError(f"Invalid CI corpus record on line {line_number}: {error}") from error
            if len(record.embedding) != config.vector_size or not all(math.isfinite(value) for value in record.embedding):
                raise ValueError(f"CI corpus vector on line {line_number} must contain {config.vector_size} finite values.")
            if hashlib.sha256(record.text.encode()).hexdigest() != record.chunk_hash:
                raise ValueError(f"CI corpus hash on line {line_number} does not match its text.")
            records.append(record)
    chunk_ids = [record.chunk_id for record in records]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("CI corpus chunk IDs must be unique.")
    if not {query.expected_chunk_id for query in config.queries}.issubset(chunk_ids):
        raise ValueError("Every CI query must reference a corpus chunk.")
    return config, records


@pytest.fixture
def api_harness(tmp_path):
    config, records = load_ci_fixture()
    qdrant = QdrantClient(location=":memory:")
    create_collection(qdrant, config.collection_name, config.vector_size)
    upsert_records(qdrant, config.collection_name, [record.model_dump() for record in records])
    shared_client = SharedClient(qdrant)
    vectors = {query.query: query.vector for query in config.queries}

    def client_factory(url):
        return shared_client

    def query_embedder(query, embedding_model):
        try:
            return vectors[query]
        except KeyError as error:
            raise ValueError(f"CI fixture has no deterministic vector for query: {query}") from error

    trace_store = TraceStore(tmp_path / "api-ci.sqlite3").initialize()
    app = create_app(
        trace_store=trace_store,
        retrieval_client_factory=client_factory,
        query_embedder=query_embedder,
    )
    with TestClient(app) as client:
        yield client, trace_store, config
    qdrant.close()


def configured_query(config, name):
    return next(query for query in config.queries if query.name == name)


def test_ci_health_is_available_without_creating_a_trace(api_harness):
    client, trace_store, _ = api_harness

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
    assert trace_store.counts() == {"traces": 0, "retrieved_chunks": 0, "feedback": 0}


def test_ci_retrieve_uses_real_dense_path_and_persists_ranked_evidence(api_harness):
    client, trace_store, config = api_harness
    query = configured_query(config, "fastapi_validation")

    response = client.post("/retrieve", json={"query": query.query, "top_k": 2})
    body = response.json()

    assert response.status_code == 200
    assert body["chunks"][0]["chunk_id"] == query.expected_chunk_id
    assert [chunk["rank"] for chunk in body["chunks"]] == [1, 2]
    assert body["component_latencies"]["embedding_ms"] is not None
    assert body["component_latencies"]["dense_ms"] is not None
    assert body["component_latencies"]["generation_ms"] is None
    traces = trace_store.list_traces()
    assert len(traces) == 1
    assert traces[0].endpoint == "retrieve"
    assert traces[0].status == "success"
    stored_chunks = trace_store.list_retrieved_chunks(traces[0].trace_id)
    assert [chunk.chunk_id for chunk in stored_chunks] == [chunk["chunk_id"] for chunk in body["chunks"]]
    assert not any(chunk.used_for_generation for chunk in stored_chunks)


def test_ci_route_uses_real_probe_and_returns_fast_decision_without_executing_or_tracing_it(api_harness):
    client, trace_store, config = api_harness
    query = configured_query(config, "fastapi_validation")

    response = client.post("/route", json={"query": query.query})
    body = response.json()

    assert response.status_code == 200
    assert body["query"] == query.query
    assert body["decision"]["router_id"] == "rule_router@0.1.0"
    assert body["decision"]["router_status"] == "draft"
    assert body["decision"]["route"] == "FAST"
    assert body["decision"]["reason_code"] == "fast_conditions_satisfied"
    assert body["decision"]["pipeline_config"] == "dense_baseline"
    assert body["decision"]["maximum_top_k"] == 2
    assert body["decision"]["reuse_probe"] is True
    assert body["probe_chunks"][0]["chunk_id"] == query.expected_chunk_id
    assert [chunk["rank"] for chunk in body["probe_chunks"]] == [1, 2]
    assert body["features"]["retrieval_confidence"]["top_score"] == pytest.approx(1.0)
    assert body["probe_timings"]["embedding_ms"] is not None
    assert body["probe_timings"]["dense_ms"] is not None
    assert trace_store.counts() == {"traces": 0, "retrieved_chunks": 0, "feedback": 0}


def test_ci_query_returns_complete_contract_and_matching_trace(api_harness):
    client, trace_store, config = api_harness
    query = configured_query(config, "qdrant_storage")

    response = client.post(
        "/query",
        json={"query": query.query, "top_k": 2, "config": "dense_baseline", "debug": True},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["trace_id"] == response.headers["x-trace-id"]
    assert body["route"] == "dense"
    assert body["config"] == "dense_baseline"
    assert body["chunks"][0]["chunk_id"] == query.expected_chunk_id
    assert body["citations"][0]["citation_id"] == "[1]"
    assert query.expected_chunk_id in body["used_chunk_ids"]
    assert body["cost"]["status"] == "zero_cost"
    assert body["cost"]["amount_usd"] == 0.0
    assert body["debug"]["pipeline_id"] == "dense_baseline@1.0.0"
    assert body["debug"]["returned_chunks"] == 2
    trace = trace_store.get_trace(body["trace_id"])
    assert trace.status == "success"
    assert trace.pipeline_name == "dense_baseline"
    assert trace.answer == body["answer"]
    stored_chunks = trace_store.list_retrieved_chunks(trace.trace_id)
    assert [chunk.chunk_id for chunk in stored_chunks] == body["used_chunk_ids"]
    assert all(chunk.used_for_generation for chunk in stored_chunks)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": "How does FastAPI validate request data?", "top_k": 0},
        {"query": "How does FastAPI validate request data?", "top_k": 21},
        {"query": "How does FastAPI validate request data?", "config": "unknown"},
        {"query": ["not", "a", "string"]},
    ],
)
def test_ci_malformed_query_requests_return_422_without_traces(api_harness, payload):
    client, trace_store, _ = api_harness

    response = client.post("/query", json=payload)

    assert response.status_code == 422
    assert trace_store.counts() == {"traces": 0, "retrieved_chunks": 0, "feedback": 0}


def test_ci_whitespace_query_returns_traced_400(api_harness):
    client, trace_store, _ = api_harness

    response = client.post("/query", json={"query": "   ", "top_k": 1})

    assert response.status_code == 400
    trace = trace_store.list_traces()[0]
    assert response.headers["x-trace-id"] == trace.trace_id
    assert response.json()["detail"] == "query must not be empty."
    assert trace.status == "error"
    assert trace.error_type == "ValueError"
    assert trace.retrieved_chunk_count == 0
