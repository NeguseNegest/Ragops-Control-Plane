from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

import scripts.retrieve_hybrid as hybrid_cli
from ragops.retrieval.dense import RetrievedChunk
from ragops.retrieval.hybrid import (
    FUSION_METADATA_KEY,
    HybridConfig,
    load_hybrid_config,
    reciprocal_rank_fusion,
    retrieve_hybrid,
    retrieve_hybrid_config,
)


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
    return HybridConfig.model_validate(
        {
            "name": "hybrid_test",
            "input": {"chunks_path": tmp_path / "chunks.jsonl"},
            "dense": {
                "type": "dense",
                "collection_name": "test_chunks",
                "embedding_model": "test-model",
                "top_k": 3,
            },
            "bm25": {
                "type": "bm25",
                "index_path": tmp_path / "bm25.json.gz",
                "tokenizer": "technical_v1",
                "top_k": 3,
            },
            "fusion": {"type": "rrf", "rank_constant": 60, "top_k": 2},
        }
    )


def test_reciprocal_rank_fusion_scores_deduplicates_and_preserves_provenance():
    dense = [make_chunk("a", 1, 0.9), make_chunk("b", 2, 0.8), make_chunk("c", 3, 0.7)]
    bm25 = [make_chunk("b", 1, 12.0), make_chunk("a", 2, 11.0), make_chunk("d", 3, 10.0)]

    results = reciprocal_rank_fusion({"dense": dense, "bm25": bm25}, top_k=3, rank_constant=60)

    assert [result.chunk_id for result in results] == ["a", "b", "c"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert results[1].score == pytest.approx(1 / 61 + 1 / 62)
    assert results[2].score == pytest.approx(1 / 63)
    assert results[0].metadata[FUSION_METADATA_KEY] == {
        "method": "rrf",
        "rank_constant": 60.0,
        "sources": {
            "dense": {"rank": 1, "score": 0.9, "rrf_contribution": pytest.approx(1 / 61)},
            "bm25": {"rank": 2, "score": 11.0, "rrf_contribution": pytest.approx(1 / 62)},
        },
    }
    assert FUSION_METADATA_KEY not in dense[0].metadata


def test_rrf_prefers_consensus_then_uses_stable_tie_breaks():
    dense = [make_chunk("dense-only", 1), make_chunk("shared", 2)]
    bm25 = [make_chunk("bm25-only", 1), make_chunk("shared", 2)]

    results = reciprocal_rank_fusion({"dense": dense, "bm25": bm25}, top_k=3, rank_constant=60)

    assert [result.chunk_id for result in results] == ["shared", "bm25-only", "dense-only"]


def test_rrf_allows_one_empty_ranking_and_all_empty_results():
    dense = [make_chunk("a", 1), make_chunk("b", 2)]

    assert [result.chunk_id for result in reciprocal_rank_fusion({"dense": dense, "bm25": []})] == ["a", "b"]
    assert reciprocal_rank_fusion({"dense": [], "bm25": []}) == []


@pytest.mark.parametrize(
    ("rankings", "message"),
    [
        ({}, "non-empty mapping"),
        ({"dense": [make_chunk("a", 2)]}, "expected 1"),
        ({"dense": [make_chunk("a", 1), make_chunk("a", 2)]}, "duplicate"),
        ({"dense": [make_chunk("a", 1, score=float("nan"))]}, "non-finite"),
    ],
)
def test_rrf_rejects_invalid_rankings(rankings, message):
    with pytest.raises(ValueError, match=message):
        reciprocal_rank_fusion(rankings)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_rrf_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion({"dense": []}, top_k=top_k)


@pytest.mark.parametrize("rank_constant", [0, -1, float("inf"), float("nan"), True, "bad"])
def test_rrf_rejects_invalid_rank_constant(rank_constant):
    with pytest.raises(ValueError, match="rank_constant"):
        reciprocal_rank_fusion({"dense": []}, rank_constant=rank_constant)


def test_rrf_rejects_conflicting_payloads_for_one_chunk_id():
    dense = [make_chunk("same", 1, text="dense text")]
    bm25 = [make_chunk("same", 1, text="different BM25 text")]

    with pytest.raises(ValueError, match="conflicting text"):
        reciprocal_rank_fusion({"dense": dense, "bm25": bm25})


def test_retrieve_hybrid_uses_configured_candidate_depths_and_fuses():
    calls = {}

    def fake_dense(**kwargs):
        calls["dense"] = kwargs
        return [make_chunk("dense", 1), make_chunk("shared", 2)]

    def fake_bm25(**kwargs):
        calls["bm25"] = kwargs
        return [make_chunk("bm25", 1), make_chunk("shared", 2)]

    results = retrieve_hybrid(
        "  technical query  ",
        client="qdrant",
        index="sparse-index",
        dense_top_k=20,
        bm25_top_k=20,
        top_k=2,
        rank_constant=40,
        collection_name="chunks",
        embedding_model="embedding-model",
        dense_retriever=fake_dense,
        bm25_retriever=fake_bm25,
    )

    assert [result.chunk_id for result in results] == ["shared", "bm25"]
    assert calls["dense"] == {
        "query": "technical query",
        "client": "qdrant",
        "top_k": 20,
        "collection_name": "chunks",
        "embedding_model": "embedding-model",
    }
    assert calls["bm25"] == {"query": "technical query", "index": "sparse-index", "top_k": 20}


def test_retrieve_hybrid_config_forwards_validated_settings(tmp_path):
    config = make_config(tmp_path)

    def fake_dense(**kwargs):
        return [make_chunk("dense", 1)]

    def fake_bm25(**kwargs):
        return [make_chunk("bm25", 1)]

    results = retrieve_hybrid_config("query", config, client="client", index="index", dense_retriever=fake_dense, bm25_retriever=fake_bm25)

    assert len(results) == 2
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(1 / 61)


def test_retrieve_hybrid_identifies_failed_candidate_retriever():
    def failed_dense(**kwargs):
        raise OSError("dense unavailable")

    with pytest.raises(RuntimeError, match="Dense candidate retrieval failed.*dense unavailable"):
        retrieve_hybrid("query", client=None, index=None, top_k=1, dense_retriever=failed_dense)

    def successful_dense(**kwargs):
        return []

    def failed_bm25(**kwargs):
        raise OSError("index unavailable")

    with pytest.raises(RuntimeError, match="BM25 candidate retrieval failed.*index unavailable"):
        retrieve_hybrid("query", client=None, index=None, top_k=1, dense_retriever=successful_dense, bm25_retriever=failed_bm25)


def test_load_hybrid_config_resolves_paths_and_builds_bm25_validation_config(tmp_path):
    config_path = tmp_path / "hybrid.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "hybrid_test",
                "input": {"chunks_path": "data/chunks.jsonl"},
                "dense": {"type": "dense", "top_k": 20},
                "bm25": {"type": "bm25", "index_path": "data/bm25.json.gz", "top_k": 20},
                "fusion": {"type": "rrf", "rank_constant": 60, "top_k": 10},
            }
        ),
        encoding="utf-8",
    )

    config = load_hybrid_config(config_path, project_root=tmp_path)
    validation_config = config.bm25_validation_config()

    assert config.input.chunks_path == tmp_path / "data/chunks.jsonl"
    assert config.bm25.index_path == tmp_path / "data/bm25.json.gz"
    assert validation_config.input == config.input
    assert validation_config.retriever == config.bm25


@pytest.mark.parametrize(
    "change",
    [
        {"name": "Invalid Name"},
        {"dense": {"top_k": 1}},
        {"bm25": {"top_k": 1}},
        {"fusion": {"rank_constant": 0}},
        {"fusion": {"rank_constant": float("inf")}},
        {"fusion": {"type": "other"}},
    ],
)
def test_hybrid_config_rejects_invalid_settings(tmp_path, change):
    data = make_config(tmp_path).model_dump(mode="python")
    for section, values in change.items():
        if isinstance(values, dict):
            data[section].update(values)
        else:
            data[section] = values

    with pytest.raises(ValidationError):
        HybridConfig.model_validate(data)


def test_checked_in_hybrid_config_implements_day24_depths():
    project_root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(project_root / "configs/hybrid.yaml", project_root=project_root)

    assert config.name == "hybrid_rrf"
    assert config.dense.top_k == 20
    assert config.bm25.top_k == 20
    assert config.fusion.type == "rrf"
    assert config.fusion.rank_constant == 60
    assert config.fusion.top_k == 10
    assert config.evaluation.labels_path == project_root / "data/eval/retrieval_labels.jsonl"
    assert config.evaluation.dense_baseline_path == project_root / "reports/evaluations/dense_baseline.json"
    assert config.evaluation.bm25_baseline_path == project_root / "reports/evaluations/bm25_baseline.json"
    assert config.output.comparison_path == project_root / "reports/evaluations/hybrid_vs_baselines.json"
    assert config.output.report_path == project_root / "reports/week4_hybrid_comparison.md"


def test_hybrid_cli_runs_query_and_closes_qdrant(monkeypatch, capsys, tmp_path):
    config = make_config(tmp_path)
    client = SimpleNamespace(collection_exists=lambda **kwargs: True, close=lambda: setattr(client, "closed", True), closed=False)
    fused = reciprocal_rank_fusion(
        {"dense": [make_chunk("shared", 1, 0.9)], "bm25": [make_chunk("shared", 1, 9.0)]},
        top_k=1,
    )

    monkeypatch.setattr(hybrid_cli, "load_hybrid_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(hybrid_cli, "load_bm25_index", lambda path: "index")
    monkeypatch.setattr(hybrid_cli, "validate_bm25_index", lambda index, validation_config: SimpleNamespace(document_count=2))
    monkeypatch.setattr(hybrid_cli, "create_qdrant_client", lambda url: client)
    monkeypatch.setattr(hybrid_cli, "retrieve_hybrid_config", lambda query, config, client, index: fused)
    monkeypatch.setattr("sys.argv", ["retrieve_hybrid.py", "--query", "test query"])

    hybrid_cli.main()

    output = capsys.readouterr().out
    assert "Hybrid results for 'test query'" in output
    assert "dense_rank=1 bm25_rank=1" in output
    assert client.closed is True
