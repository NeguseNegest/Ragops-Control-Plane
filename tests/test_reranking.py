import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

import scripts.retrieve_hybrid_rerank as rerank_cli
from ragops.reranking.cross_encoder import (
    RERANKER_METADATA_KEY,
    CrossEncoderReranker,
    HybridRerankConfig,
    build_cross_encoder_reranker,
    load_hybrid_rerank_config,
    rerank_chunks,
    retrieve_hybrid_reranked,
    retrieve_hybrid_reranked_config,
)
from ragops.retrieval.dense import RetrievedChunk
from ragops.retrieval.hybrid import FUSION_METADATA_KEY


def make_chunk(chunk_id, rank, score=1.0, text=None, metadata=None):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        text=text or f"Evidence for {chunk_id}",
        score=score,
        rank=rank,
        metadata=metadata or {"relative_path": f"docs/{chunk_id}.md"},
        source_url=f"docs/{chunk_id}.md",
    )


def make_config(tmp_path):
    return HybridRerankConfig.model_validate(
        {
            "name": "hybrid_rerank_test",
            "input": {"chunks_path": tmp_path / "chunks.jsonl"},
            "dense": {"type": "dense", "collection_name": "test_chunks", "embedding_model": "test-model", "top_k": 4},
            "bm25": {"type": "bm25", "index_path": tmp_path / "bm25.json.gz", "tokenizer": "technical_v1", "top_k": 4},
            "fusion": {"type": "rrf", "rank_constant": 60, "top_k": 4},
            "reranker": {"type": "cross_encoder", "model": "test-reranker", "candidate_top_k": 4, "top_k": 2, "batch_size": 2, "max_length": 128},
        }
    )


class FakeReranker:
    model_name = "test-reranker"

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, query, chunks):
        self.calls.append((query, [chunk.chunk_id for chunk in chunks]))
        return self.scores[: len(chunks)]


class IncrementingClock:
    def __init__(self, step=0.001):
        self.value = -step
        self.step = step

    def __call__(self):
        self.value += self.step
        return self.value


def test_cross_encoder_wrapper_builds_pairs_and_validates_scores():
    calls = {}

    class FakeModel:
        def predict(self, pairs, **kwargs):
            calls["predict"] = {"pairs": pairs, **kwargs}
            return [0.25, -1.5]

    def factory(model_name, **kwargs):
        calls["factory"] = {"model_name": model_name, **kwargs}
        return FakeModel()

    reranker = CrossEncoderReranker(model_name="local/model", batch_size=8, max_length=256, device="cpu", model_factory=factory)
    chunks = [make_chunk("a", 1), make_chunk("b", 2)]

    assert reranker.score("  technical query  ", chunks) == [0.25, -1.5]
    assert calls["factory"] == {"model_name": "local/model", "max_length": 256, "device": "cpu"}
    assert calls["predict"] == {
        "pairs": [("technical query", "Evidence for a"), ("technical query", "Evidence for b")],
        "batch_size": 8,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }


@pytest.mark.parametrize("scores", [[0.1], [float("nan"), 0.2], [[0.1], [0.2]]])
def test_cross_encoder_wrapper_rejects_invalid_model_outputs(scores):
    class FakeModel:
        def predict(self, pairs, **kwargs):
            return scores

    reranker = CrossEncoderReranker(model_factory=lambda *args, **kwargs: FakeModel())

    with pytest.raises(ValueError, match="scores|finite scalar"):
        reranker.score("query", [make_chunk("a", 1), make_chunk("b", 2)])


def test_build_cross_encoder_reranker_uses_configured_model_settings(tmp_path):
    config = make_config(tmp_path).reranker
    calls = {}

    def factory(model_name, **kwargs):
        calls.update({"model_name": model_name, **kwargs})
        return SimpleNamespace()

    reranker = build_cross_encoder_reranker(config, model_factory=factory)

    assert reranker.model_name == "test-reranker"
    assert reranker.batch_size == 2
    assert calls == {"model_name": "test-reranker", "max_length": 128}


def test_rerank_chunks_orders_scores_preserves_provenance_and_tracks_latency():
    fusion_metadata = {
        "method": "rrf",
        "rank_constant": 60.0,
        "sources": {"dense": {"rank": 1, "score": 0.9, "rrf_contribution": 1 / 61}},
    }
    candidates = [
        make_chunk("a", 1, score=0.04, metadata={FUSION_METADATA_KEY: fusion_metadata}),
        make_chunk("b", 2, score=0.03),
        make_chunk("c", 3, score=0.02),
    ]
    reranker = FakeReranker([0.1, 0.9, 0.4])
    timings = {}

    results = rerank_chunks("query", candidates, reranker, candidate_top_k=3, top_k=2, clock=IncrementingClock(step=0.125), timings=timings)

    assert [result.chunk_id for result in results] == ["b", "c"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.score for result in results] == [0.9, 0.4]
    assert results[0].metadata[RERANKER_METADATA_KEY] == {
        "method": "cross_encoder",
        "model": "test-reranker",
        "candidate_rank": 2,
        "candidate_score": 0.03,
    }
    assert candidates[1].metadata == {"relative_path": "docs/b.md"}
    assert candidates[0].metadata[FUSION_METADATA_KEY] == fusion_metadata
    assert timings == {"reranker_ms": 125.0}
    assert reranker.calls == [("query", ["a", "b", "c"])]


def test_rerank_chunks_uses_original_rank_then_chunk_id_for_stable_ties():
    candidates = [make_chunk("z", 1), make_chunk("a", 2), make_chunk("b", 3)]

    results = rerank_chunks("query", candidates, FakeReranker([0.5, 0.5, 0.4]), candidate_top_k=3, top_k=3)

    assert [result.chunk_id for result in results] == ["z", "a", "b"]


def test_rerank_chunks_limits_input_candidate_depth_before_scoring():
    candidates = [make_chunk(str(position), position) for position in range(1, 6)]
    reranker = FakeReranker([0.1, 0.2, 0.3])

    results = rerank_chunks("query", candidates, reranker, candidate_top_k=3, top_k=2)

    assert reranker.calls == [("query", ["1", "2", "3"])]
    assert [result.chunk_id for result in results] == ["3", "2"]


def test_rerank_chunks_handles_empty_candidates_without_calling_model():
    reranker = FakeReranker([])
    timings = {}

    assert rerank_chunks("query", [], reranker, timings=timings) == []
    assert reranker.calls == []
    assert timings == {"reranker_ms": 0.0}


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        ([make_chunk("a", 2)], "expected 1"),
        ([make_chunk("a", 1), make_chunk("a", 2)], "duplicate"),
        ([make_chunk("a", 1, text=" ")], "empty text"),
        ([make_chunk("a", 1, score=float("inf"))], "non-finite"),
    ],
)
def test_rerank_chunks_rejects_invalid_candidates(chunks, message):
    with pytest.raises(ValueError, match=message):
        rerank_chunks("query", chunks, FakeReranker([0.1, 0.2]))


def test_rerank_chunks_identifies_model_failure():
    class FailedReranker:
        def score(self, query, chunks):
            raise OSError("model unavailable")

    with pytest.raises(RuntimeError, match="Cross-encoder reranking failed.*model unavailable"):
        rerank_chunks("query", [make_chunk("a", 1)], FailedReranker())


def test_retrieve_hybrid_reranked_runs_all_stages_and_records_timings():
    calls = {}

    def fake_dense(**kwargs):
        calls["dense"] = kwargs
        return [make_chunk("dense", 1), make_chunk("shared", 2)]

    def fake_bm25(**kwargs):
        calls["bm25"] = kwargs
        return [make_chunk("bm25", 1), make_chunk("shared", 2)]

    reranker = FakeReranker([0.1, 0.9, 0.3])
    timings = {}
    results = retrieve_hybrid_reranked(
        "query",
        client="qdrant",
        index="index",
        reranker=reranker,
        dense_top_k=4,
        bm25_top_k=4,
        candidate_top_k=3,
        top_k=2,
        collection_name="chunks",
        embedding_model="embedding",
        dense_retriever=fake_dense,
        bm25_retriever=fake_bm25,
        clock=IncrementingClock(),
        timings=timings,
    )

    assert [result.chunk_id for result in results] == ["bm25", "dense"]
    assert calls["dense"]["top_k"] == 4
    assert calls["bm25"] == {"query": "query", "index": "index", "top_k": 4}
    assert reranker.calls == [("query", ["shared", "bm25", "dense"])]
    assert set(timings) == {"dense_ms", "bm25_ms", "fusion_ms", "reranker_ms", "total_ms"}
    assert all(value >= 0 for value in timings.values())


def test_retrieve_hybrid_reranked_config_forwards_settings(tmp_path):
    config = make_config(tmp_path)

    def fake_dense(**kwargs):
        return [make_chunk("dense", 1)]

    def fake_bm25(**kwargs):
        return [make_chunk("bm25", 1)]

    results = retrieve_hybrid_reranked_config(
        "query",
        config,
        client="client",
        index="index",
        reranker=FakeReranker([0.2, 0.8]),
        dense_retriever=fake_dense,
        bm25_retriever=fake_bm25,
    )

    assert [result.chunk_id for result in results] == ["dense", "bm25"]
    assert len(results) == config.reranker.top_k


def test_load_hybrid_rerank_config_resolves_paths(tmp_path):
    config_path = tmp_path / "rerank.yaml"
    raw_config = make_config(tmp_path).model_dump(mode="json")
    raw_config["input"]["chunks_path"] = "data/chunks.jsonl"
    raw_config["bm25"]["index_path"] = "data/bm25.json.gz"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    config = load_hybrid_rerank_config(config_path, project_root=tmp_path)

    assert config.input.chunks_path == tmp_path / "data/chunks.jsonl"
    assert config.bm25.index_path == tmp_path / "data/bm25.json.gz"
    assert config.bm25_validation_config().retriever == config.bm25


@pytest.mark.parametrize(
    "change",
    [
        {"name": "Invalid Name"},
        {"dense": {"top_k": 3}},
        {"bm25": {"top_k": 3}},
        {"fusion": {"top_k": 3}},
        {"reranker": {"candidate_top_k": 3}},
        {"reranker": {"top_k": 5}},
        {"reranker": {"model": " "}},
        {"reranker": {"type": "other"}},
    ],
)
def test_hybrid_rerank_config_rejects_invalid_settings(tmp_path, change):
    data = make_config(tmp_path).model_dump(mode="python")
    for section, values in change.items():
        if isinstance(values, dict):
            data[section].update(values)
        else:
            data[section] = values

    with pytest.raises(ValidationError):
        HybridRerankConfig.model_validate(data)


def test_checked_in_config_implements_day26_candidate_and_result_depths():
    project_root = Path(__file__).resolve().parents[1]
    config = load_hybrid_rerank_config(project_root / "configs/hybrid_rerank.yaml", project_root=project_root)

    assert config.name == "hybrid_rrf_cross_encoder"
    assert config.dense.top_k == 25
    assert config.bm25.top_k == 25
    assert config.fusion.top_k == 25
    assert config.reranker.type == "cross_encoder"
    assert config.reranker.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.reranker.candidate_top_k == 25
    assert config.reranker.top_k == 5


def test_reranker_cli_validate_only_avoids_qdrant_and_model(monkeypatch, capsys, tmp_path):
    config = make_config(tmp_path)
    monkeypatch.setattr(rerank_cli, "parse_args", lambda: SimpleNamespace(config=tmp_path / "config.yaml", query=None, qdrant_url=None, json=False, validate_only=True))
    monkeypatch.setattr(rerank_cli, "load_hybrid_rerank_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(rerank_cli, "load_bm25_index", lambda path: "index")
    monkeypatch.setattr(rerank_cli, "validate_bm25_index", lambda index, validation_config: SimpleNamespace(document_count=12))
    monkeypatch.setattr(rerank_cli, "create_qdrant_client", lambda url: pytest.fail("Qdrant must not be created during validation"))
    monkeypatch.setattr(rerank_cli, "build_cross_encoder_reranker", lambda config: pytest.fail("Model must not load during validation"))

    rerank_cli.main()

    output = capsys.readouterr().out
    assert "RRF 4 -> cross-encoder top 2" in output
    assert "12 documents" in output


def test_reranker_cli_emits_json_timings_and_closes_qdrant(monkeypatch, capsys, tmp_path):
    config = make_config(tmp_path)
    client = SimpleNamespace(collection_exists=lambda collection_name: True, closed=False)
    client.close = lambda: setattr(client, "closed", True)
    result = make_chunk("reranked", 1, score=2.4, metadata={RERANKER_METADATA_KEY: {"candidate_rank": 3}})
    monkeypatch.setattr(rerank_cli, "parse_args", lambda: SimpleNamespace(config=tmp_path / "config.yaml", query="query", qdrant_url=None, json=True, validate_only=False))
    monkeypatch.setattr(rerank_cli, "load_hybrid_rerank_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(rerank_cli, "load_bm25_index", lambda path: "index")
    monkeypatch.setattr(rerank_cli, "validate_bm25_index", lambda index, validation_config: SimpleNamespace(document_count=12))
    monkeypatch.setattr(rerank_cli, "create_qdrant_client", lambda url: client)
    monkeypatch.setattr(rerank_cli, "build_cross_encoder_reranker", lambda config: "reranker")

    def fake_retrieve(query, config, client, index, reranker, timings):
        timings.update({"dense_ms": 1.0, "bm25_ms": 2.0, "fusion_ms": 0.1, "reranker_ms": 3.0, "total_ms": 6.1})
        return [result]

    monkeypatch.setattr(rerank_cli, "retrieve_hybrid_reranked_config", fake_retrieve)

    rerank_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert client.closed is True
    assert output["query"] == "query"
    assert output["timings_ms"]["reranker_ms"] == 3.0
    assert output["timings_ms"]["model_load_ms"] >= 0
    assert output["results"][0]["chunk_id"] == "reranked"
