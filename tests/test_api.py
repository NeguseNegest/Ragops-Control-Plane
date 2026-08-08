from fastapi.testclient import TestClient

import ragops.app as app_module
from ragops import __version__
from ragops.generation.client import GenerationResult
from ragops.retrieval.dense import RetrievedChunk


def make_chunk():
    return RetrievedChunk(chunk_id="chunk-1", document_id="doc-1", text="FastAPI is a Python web framework.", score=0.91, rank=1, metadata={"title": "FastAPI Docs", "relative_path": "fastapi/tutorial.md"}, source_url="fastapi/tutorial.md")


def make_client():
    return TestClient(app_module.create_app())


def test_package_version_is_declared():
    assert __version__


def test_health_route_is_registered():
    app = app_module.create_app()

    paths = {route.path for route in app.routes}

    assert "/health" in paths


def test_docs_are_available():
    client = make_client()

    response = client.get("/docs")

    assert response.status_code == 200
    assert "Swagger UI" in response.text


def test_retrieve_returns_chunks(monkeypatch):
    calls = {}

    def fake_retrieve_chunks(query, top_k):
        calls["query"] = query
        calls["top_k"] = top_k
        return [make_chunk()]

    monkeypatch.setattr(app_module, "retrieve_chunks", fake_retrieve_chunks)
    client = make_client()

    response = client.post("/retrieve", json={"query": "What is FastAPI?", "top_k": 1})
    body = response.json()

    assert response.status_code == 200
    assert calls == {"query": "What is FastAPI?", "top_k": 1}
    assert body["query"] == "What is FastAPI?"
    assert body["top_k"] == 1
    assert body["latency_ms"] >= 0
    assert body["chunks"][0]["chunk_id"] == "chunk-1"
    assert body["chunks"][0]["source_url"] == "fastapi/tutorial.md"


def test_query_returns_answer_citations_chunks_and_latency(monkeypatch):
    chunk = make_chunk()
    calls = {}

    def fake_retrieve_chunks(query, top_k):
        calls["retrieve"] = (query, top_k)
        return [chunk]

    def fake_generate_answer(query, chunks):
        calls["generate"] = (query, chunks)
        citations = [{"citation_id": "[1]", "document_id": "doc-1", "title": "FastAPI Docs", "url": "fastapi/tutorial.md", "metadata": {"title": "FastAPI Docs"}, "chunk_ids": ["chunk-1"]}]
        return GenerationResult(answer="FastAPI is a Python web framework. [1]", citations=citations, citation_text="[1] FastAPI Docs - fastapi/tutorial.md", used_chunk_ids=["chunk-1"])

    monkeypatch.setattr(app_module, "retrieve_chunks", fake_retrieve_chunks)
    monkeypatch.setattr(app_module, "generate_answer", fake_generate_answer)
    client = make_client()

    response = client.post("/query", json={"query": "What is FastAPI?", "top_k": 1})
    body = response.json()

    assert response.status_code == 200
    assert calls["retrieve"] == ("What is FastAPI?", 1)
    assert calls["generate"] == ("What is FastAPI?", [chunk])
    assert body["answer"] == "FastAPI is a Python web framework. [1]"
    assert body["citations"][0]["citation_id"] == "[1]"
    assert body["citation_text"] == "[1] FastAPI Docs - fastapi/tutorial.md"
    assert body["chunks"][0]["chunk_id"] == "chunk-1"
    assert body["used_chunk_ids"] == ["chunk-1"]
    assert body["latency_ms"] >= 0


def test_empty_query_returns_400(monkeypatch):
    def fake_retrieve_chunks(query, top_k):
        raise ValueError("query must not be empty.")

    monkeypatch.setattr(app_module, "retrieve_chunks", fake_retrieve_chunks)
    client = make_client()

    response = client.post("/retrieve", json={"query": "   ", "top_k": 1})

    assert response.status_code == 400
    assert response.json()["detail"] == "query must not be empty."


def test_retrieve_failure_returns_503(monkeypatch):
    def fake_retrieve_chunks(query, top_k):
        raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr(app_module, "retrieve_chunks", fake_retrieve_chunks)
    client = make_client()

    response = client.post("/retrieve", json={"query": "What is FastAPI?", "top_k": 1})

    assert response.status_code == 503
    assert response.json()["detail"] == "Unable to retrieve chunks."


def test_query_generation_failure_returns_503(monkeypatch):
    def fake_retrieve_chunks(query, top_k):
        return [make_chunk()]

    def fake_generate_answer(query, chunks):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(app_module, "retrieve_chunks", fake_retrieve_chunks)
    monkeypatch.setattr(app_module, "generate_answer", fake_generate_answer)
    client = make_client()

    response = client.post("/query", json={"query": "What is FastAPI?", "top_k": 1})

    assert response.status_code == 503
    assert response.json()["detail"] == "Unable to generate answer."
