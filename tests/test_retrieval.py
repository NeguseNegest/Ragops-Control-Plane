from types import SimpleNamespace

import pytest

from ragops.retrieval import dense
from ragops.retrieval.dense import RetrievedChunk, build_retrieved_chunk, build_retrieved_chunks, retrieve_dense, source_url_from_metadata, validate_query


def make_result(chunk_id="chunk-1", score=0.9, metadata=None):
    if metadata is None:
        metadata = {"relative_path": "docs/example.md", "source_name": "test"}

    payload = {"chunk_id": chunk_id, "document_id": "doc-1", "text": f"Text for {chunk_id}", "metadata": metadata}
    return SimpleNamespace(id=chunk_id, score=score, payload=payload)


def test_validate_query_strips_text():
    assert validate_query("  What is FastAPI?  ") == "What is FastAPI?"


def test_validate_query_rejects_empty_text():
    with pytest.raises(ValueError, match="empty"):
        validate_query("   ")


def test_source_url_from_metadata_uses_best_available_source():
    assert source_url_from_metadata({"source_url": "https://example.com/docs", "relative_path": "docs/example.md"}) == "https://example.com/docs"
    assert source_url_from_metadata({"relative_path": "docs/example.md"}) == "docs/example.md"
    assert source_url_from_metadata({}) is None


def test_build_retrieved_chunk_keeps_payload_score_and_rank():
    result = make_result("chunk-1", score=0.82)

    chunk = build_retrieved_chunk(result, rank=1)

    assert isinstance(chunk, RetrievedChunk)
    assert chunk.chunk_id == "chunk-1"
    assert chunk.document_id == "doc-1"
    assert chunk.text == "Text for chunk-1"
    assert chunk.score == 0.82
    assert chunk.rank == 1
    assert chunk.metadata == {"relative_path": "docs/example.md", "source_name": "test"}
    assert chunk.source_url == "docs/example.md"


def test_build_retrieved_chunks_adds_ranks_in_result_order():
    results = [
        make_result("chunk-1", score=0.9),
        make_result("chunk-2", score=0.7),
    ]

    chunks = build_retrieved_chunks(results)

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1", "chunk-2"]
    assert [chunk.rank for chunk in chunks] == [1, 2]


def test_retrieve_dense_embeds_searches_and_returns_chunks(monkeypatch):
    calls = {}

    def fake_embed_query(query, embedding_model):
        calls["embed"] = (query, embedding_model)
        return [0.1, 0.2, 0.3]

    def fake_search_index(client, collection_name, query_vector, top_k):
        calls["search"] = (client, collection_name, query_vector, top_k)
        return [make_result("chunk-1", score=0.95), make_result("chunk-2", score=0.85)]

    monkeypatch.setattr(dense, "embed_query", fake_embed_query)
    monkeypatch.setattr(dense, "search_index", fake_search_index)

    chunks = retrieve_dense("  install qdrant  ", client="fake-client", top_k=2, collection_name="test_chunks", embedding_model="fake-model")

    assert calls["embed"] == ("install qdrant", "fake-model")
    assert calls["search"] == ("fake-client", "test_chunks", [0.1, 0.2, 0.3], 2)
    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1", "chunk-2"]
