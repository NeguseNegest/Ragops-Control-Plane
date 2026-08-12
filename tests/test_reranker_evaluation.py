import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragops.evaluation.reranker_runner import (
    compare_reranker_reports,
    evaluate_reranker_config,
    render_reranker_comparison_markdown,
    run_reranker_evaluation,
    write_reranker_comparison_artifacts,
)
from ragops.evaluation.retrieval_labels import RetrievalLabel
from ragops.evaluation.retrieval_metrics import evaluate_retrieval_metrics
from ragops.evaluation.runner import build_question_metrics
from ragops.reranking.cross_encoder import RERANKER_METADATA_KEY, HybridRerankConfig, rerank_chunks
from ragops.retrieval.bm25 import BM25Index, build_bm25_index
from ragops.retrieval.dense import RetrievedChunk
from ragops.retrieval.hybrid import reciprocal_rank_fusion


class FakeClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class FakeClient:
    def __init__(self, points_count=2, exists=True):
        self.points_count = points_count
        self.exists = exists
        self.closed = False

    def collection_exists(self, collection_name):
        return self.exists

    def get_collection(self, collection_name):
        return SimpleNamespace(points_count=self.points_count)

    def close(self):
        self.closed = True


class FakeReranker:
    model_name = "test-reranker"

    def __init__(self, scores):
        self.scores = scores

    def score(self, query, chunks):
        return self.scores[: len(chunks)]


def make_label(question_id, question, relevant_chunk_id):
    return RetrievalLabel(
        question_id=question_id,
        question=question,
        relevant_chunk_ids=[relevant_chunk_id],
        expected_source="docs/example.md",
        metadata={"label_method": "manual", "review_status": "verified", "reviewed_by": "pytest"},
    )


def make_record(chunk_id, text):
    return {
        "chunk_id": chunk_id,
        "document_id": f"document-{chunk_id}",
        "text": text,
        "token_count": len(text.split()),
        "chunk_hash": f"hash-{chunk_id}",
        "metadata": {"relative_path": "docs/example.md"},
    }


def make_index():
    return BM25Index(
        build_bm25_index(
            [make_record("a", "alpha evidence"), make_record("b", "beta evidence")],
            source_path="chunks.jsonl",
            source_sha256="a" * 64,
        )
    )


def make_config(tmp_path):
    return HybridRerankConfig.model_validate(
        {
            "name": "reranker_test",
            "input": {"chunks_path": tmp_path / "chunks.jsonl"},
            "dense": {"type": "dense", "collection_name": "chunks", "embedding_model": "model", "top_k": 3},
            "bm25": {"type": "bm25", "index_path": tmp_path / "bm25.json.gz", "tokenizer": "technical_v1", "top_k": 3, "k1": 1.5, "b": 0.75, "epsilon": 0.25},
            "fusion": {"type": "rrf", "rank_constant": 60, "top_k": 3},
            "reranker": {"type": "cross_encoder", "model": "test-reranker", "candidate_top_k": 3, "top_k": 2, "batch_size": 2, "max_length": 128},
            "evaluation": {
                "labels_path": tmp_path / "labels.jsonl",
                "k_values": [1, 2],
                "minimum_labels": 1,
                "dense_baseline_path": tmp_path / "dense.json",
                "bm25_baseline_path": tmp_path / "bm25.json",
                "hybrid_baseline_path": tmp_path / "hybrid.json",
            },
            "output": {
                "directory": tmp_path / "reports",
                "comparison_path": tmp_path / "comparison.json",
                "report_path": tmp_path / "comparison.md",
            },
        }
    )


def make_chunk(chunk_id, rank, score=1.0):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        text=f"Evidence for {chunk_id}",
        score=score,
        rank=rank,
        metadata={"relative_path": "docs/example.md"},
        source_url="docs/example.md",
    )


def fused_candidates(chunk_ids):
    dense = [make_chunk(chunk_id, rank, score=1 / rank) for rank, chunk_id in enumerate(chunk_ids, start=1)]
    return reciprocal_rank_fusion({"dense": dense, "bm25": []}, top_k=len(chunk_ids))


def test_run_reranker_evaluation_records_candidate_ablation_provenance_and_latency(tmp_path):
    config = make_config(tmp_path)
    labels = [make_label("q1", "What is alpha?", "a"), make_label("q2", "What is beta?", "b")]

    def fake_retriever(query, config, client, index, reranker, clock, timings):
        timings.update({"dense_ms": 5.0, "bm25_ms": 3.0, "fusion_ms": 1.0, "reranker_ms": 4.0})
        if "alpha" in query:
            candidates = fused_candidates(["b", "a", "x"])
            scores = [0.1, 0.9, 0.2]
        else:
            candidates = fused_candidates(["b", "a", "x"])
            scores = [0.8, 0.2, 0.1]
        return candidates, rerank_chunks(query, candidates, FakeReranker(scores), candidate_top_k=3, top_k=2, clock=lambda: 0.0)

    report = run_reranker_evaluation(
        config,
        labels,
        client="client",
        index=make_index(),
        reranker="model",
        retriever=fake_retriever,
        clock=FakeClock([0.0, 0.02, 1.0, 1.03]),
        dense_index={"collection_name": "chunks", "points_count": 2},
        model_load_ms=12.0,
    )

    assert report["metrics"]["mrr"] == 1.0
    assert report["pre_rerank_metrics"]["mrr"] == pytest.approx(0.75)
    assert report["latency_ms"]["average"] == pytest.approx(25.0)
    assert report["latency_after_first_ms"]["average"] == pytest.approx(30.0)
    assert report["component_latency_ms"]["reranker"]["average"] == 4.0
    assert report["model"]["load_latency_ms"] == 12.0
    assert report["questions"][0]["candidate_chunk_ids"] == ["b", "a", "x"]
    assert report["questions"][0]["retrieved_chunk_ids"] == ["a", "x"]
    assert report["questions"][0]["retrieved_reranker_provenance"][0]["candidate_rank"] == 2


def test_run_reranker_evaluation_rejects_missing_timing_and_bad_provenance(tmp_path):
    config = make_config(tmp_path)
    label = make_label("q1", "What is alpha?", "a")

    def missing_timing(**kwargs):
        candidates = fused_candidates(["a"])
        return candidates, rerank_chunks("query", candidates, FakeReranker([1.0]), candidate_top_k=1, top_k=1, clock=lambda: 0.0)

    with pytest.raises(ValueError, match="component timings"):
        run_reranker_evaluation(config, [label], "client", make_index(), "model", retriever=missing_timing, clock=FakeClock([0.0, 0.1]))

    def bad_provenance(query, config, client, index, reranker, clock, timings):
        timings.update({"dense_ms": 1.0, "bm25_ms": 1.0, "fusion_ms": 1.0, "reranker_ms": 1.0})
        candidates = fused_candidates(["a"])
        result = make_chunk("a", 1)
        result.metadata[RERANKER_METADATA_KEY] = {"method": "cross_encoder", "model": "wrong", "candidate_rank": 1, "candidate_score": candidates[0].score}
        return candidates, [result]

    with pytest.raises(ValueError, match="model provenance"):
        run_reranker_evaluation(config, [label], "client", make_index(), "model", retriever=bad_provenance, clock=FakeClock([0.0, 0.1]))


def test_evaluate_reranker_config_loads_model_once_and_closes_client(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config.input.chunks_path.write_text("test", encoding="utf-8")
    index = make_index()
    client = FakeClient(points_count=2)
    labels = [make_label("q1", "What is alpha?", "a")]
    calls = []
    monkeypatch.setattr("ragops.evaluation.reranker_runner.validate_bm25_index", lambda loaded_index, validation_config: loaded_index.payload)

    def model_factory(reranker_config):
        calls.append(reranker_config.model)
        return "model"

    def fake_retriever(query, config, client, index, reranker, clock, timings):
        timings.update({"dense_ms": 1.0, "bm25_ms": 1.0, "fusion_ms": 1.0, "reranker_ms": 1.0})
        candidates = fused_candidates(["a"])
        return candidates, rerank_chunks(query, candidates, FakeReranker([1.0]), candidate_top_k=1, top_k=1, clock=lambda: 0.0)

    report = evaluate_reranker_config(
        config,
        labels,
        client_factory=lambda url: client,
        index_loader=lambda path: index,
        reranker_factory=model_factory,
        retriever=fake_retriever,
        clock=FakeClock([0.0, 0.01, 1.0, 1.02]),
    )

    assert calls == ["test-reranker"]
    assert report["model"]["load_latency_ms"] == 10.0
    assert client.closed is True


def test_evaluate_reranker_config_rejects_index_count_drift_and_closes_client(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config.input.chunks_path.write_text("test", encoding="utf-8")
    index = make_index()
    client = FakeClient(points_count=3)
    monkeypatch.setattr("ragops.evaluation.reranker_runner.validate_bm25_index", lambda loaded_index, validation_config: loaded_index.payload)

    with pytest.raises(RuntimeError, match="contains 3 points"):
        evaluate_reranker_config(config, [make_label("q1", "What is alpha?", "a")], client_factory=lambda url: client, index_loader=lambda path: index)

    assert client.closed is True


def build_baseline_report(run_name, retriever_name, labels, rankings):
    k_values = [1, 2]
    questions = [
        {
            "question_id": label.question_id,
            "question": label.question,
            "expected_source": label.expected_source,
            "relevant_chunk_ids": label.relevant_chunk_ids,
            "retrieved_chunk_ids": rankings[label.question_id],
            "retrieved_scores": [1.0] * len(rankings[label.question_id]),
            "latency_ms": 10.0,
            **build_question_metrics(rankings[label.question_id], label.relevant_chunk_ids, k_values),
        }
        for label in labels
    ]
    if retriever_name == "dense":
        configuration = {"retriever": {"type": "dense", "collection_name": "chunks", "embedding_model": "model"}}
    elif retriever_name == "bm25":
        configuration = {"retriever": {"type": "bm25", "tokenizer": "technical_v1", "k1": 1.5, "b": 0.75, "epsilon": 0.25}}
    else:
        configuration = {
            "dense": {"type": "dense", "collection_name": "chunks", "embedding_model": "model"},
            "bm25": {"type": "bm25", "tokenizer": "technical_v1", "k1": 1.5, "b": 0.75, "epsilon": 0.25},
            "fusion": {"type": "rrf", "rank_constant": 60, "top_k": 2},
        }
    report = {
        "schema_version": 1,
        "run_name": run_name,
        "configuration": configuration,
        "metrics": evaluate_retrieval_metrics(rankings, labels, k_values),
        "latency_ms": {"total": 30.0, "average": 10.0, "minimum": 10.0, "maximum": 10.0},
        "questions": questions,
    }
    if retriever_name == "bm25":
        report["index"] = {"source_sha256": "a" * 64}
    if retriever_name == "hybrid":
        report["dense_index"] = {"collection_name": "chunks", "points_count": 3}
        report["bm25_index"] = {"source_sha256": "a" * 64, "source_record_count": 3, "document_count": 3, "skipped_document_count": 0}
    return report


def build_reranked_report(labels, candidates, rankings):
    k_values = [1, 2]
    questions = []
    for label in labels:
        candidate_ids = candidates[label.question_id]
        retrieved_ids = rankings[label.question_id]
        questions.append(
            {
                "question_id": label.question_id,
                "question": label.question,
                "expected_source": label.expected_source,
                "relevant_chunk_ids": label.relevant_chunk_ids,
                "candidate_chunk_ids": candidate_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved_scores": [1.0] * len(retrieved_ids),
                "latency_ms": 20.0,
                "component_latency_ms": {"dense_ms": 5.0, "bm25_ms": 4.0, "fusion_ms": 1.0, "reranker_ms": 10.0},
                **build_question_metrics(retrieved_ids, label.relevant_chunk_ids, k_values),
            }
        )
    pre_rankings = {question_id: ranking[:2] for question_id, ranking in candidates.items()}
    return {
        "schema_version": 1,
        "run_name": "reranked",
        "configuration": {
            "dense": {"type": "dense", "collection_name": "chunks", "embedding_model": "model", "top_k": 3},
            "bm25": {"type": "bm25", "tokenizer": "technical_v1", "k1": 1.5, "b": 0.75, "epsilon": 0.25, "top_k": 3},
            "fusion": {"type": "rrf", "rank_constant": 60, "top_k": 3},
            "reranker": {"type": "cross_encoder", "model": "test-reranker", "candidate_top_k": 3, "top_k": 2, "batch_size": 2, "max_length": 128},
        },
        "model": {"load_latency_ms": 15.0},
        "dense_index": {"collection_name": "chunks", "points_count": 3},
        "bm25_index": {"source_sha256": "a" * 64, "source_record_count": 3, "document_count": 3, "skipped_document_count": 0},
        "metrics": evaluate_retrieval_metrics(rankings, labels, k_values),
        "pre_rerank_metrics": evaluate_retrieval_metrics(pre_rankings, labels, k_values),
        "latency_ms": {"total": 60.0, "average": 20.0, "minimum": 20.0, "maximum": 20.0},
        "latency_after_first_ms": {"question_count": 2, "total": 40.0, "average": 20.0, "minimum": 20.0, "maximum": 20.0},
        "component_latency_ms": {name: {"total": total, "average": total / 3, "minimum": total / 3, "maximum": total / 3} for name, total in (("dense", 15.0), ("bm25", 12.0), ("fusion", 3.0), ("reranker", 30.0))},
        "component_latency_after_first_ms": {name: {"total": total, "average": total / 2, "minimum": total / 2, "maximum": total / 2} for name, total in (("dense", 10.0), ("bm25", 8.0), ("fusion", 2.0), ("reranker", 20.0))},
        "questions": questions,
    }


def comparison_fixture():
    labels = [
        make_label("q1", "What is the exact alpha parameter?", "a"),
        make_label("q2", "What happens when beta fails?", "b"),
        make_label("q3", "Describe gamma behavior.", "c"),
    ]
    dense = build_baseline_report("dense", "dense", labels, {"q1": ["a"], "q2": ["x"], "q3": ["x", "c"]})
    bm25 = build_baseline_report("bm25", "bm25", labels, {"q1": ["x", "a"], "q2": ["b"], "q3": ["x"]})
    hybrid = build_baseline_report("hybrid", "hybrid", labels, {"q1": ["a"], "q2": ["x", "b"], "q3": ["c"]})
    candidates = {"q1": ["x", "a", "z"], "q2": ["b", "x", "z"], "q3": ["x", "c", "z"]}
    reranked = build_reranked_report(labels, candidates, {"q1": ["a", "x"], "q2": ["x", "b"], "q3": ["c", "x"]})
    return dense, bm25, hybrid, reranked


def test_compare_reranker_reports_builds_common_depth_metrics_and_ablation():
    comparison = compare_reranker_reports(*comparison_fixture())

    assert comparison["question_count"] == 3
    assert comparison["comparison_depth"] == 2
    assert comparison["metrics"]["mrr_at_depth"]["dense"] == pytest.approx(0.5)
    assert comparison["metrics"]["mrr_at_depth"]["reranked"] == pytest.approx(5 / 6)
    assert comparison["metrics"]["mrr_at_depth"]["pre_rerank"] == pytest.approx(2 / 3)
    assert comparison["paired"]["pre_rerank"] == {
        "reranked_wins": 2,
        "pre_rerank_wins": 1,
        "ties": 0,
        "reranked_recovers_baseline_miss": 0,
        "reranked_loses_baseline_hit": 0,
    }
    assert comparison["decision"]["best_retriever"] == "hybrid"
    assert comparison["decision"]["reranking_ablation_improves"] is True


def test_compare_reranker_reports_rejects_hash_question_and_stored_metric_drift():
    reports = comparison_fixture()
    reports[3]["bm25_index"]["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="SHA256"):
        compare_reranker_reports(*reports)

    reports = comparison_fixture()
    reports[3]["questions"][0]["question"] = "different"
    with pytest.raises(ValueError, match="question differs"):
        compare_reranker_reports(*reports)

    reports = comparison_fixture()
    reports[3]["metrics"]["mrr"] = 0.0
    with pytest.raises(ValueError, match="stored MRR"):
        compare_reranker_reports(*reports)


def test_write_reranker_comparison_artifacts_emits_ablation_and_failure_sections(tmp_path):
    comparison = compare_reranker_reports(*comparison_fixture())

    json_path, markdown_path = write_reranker_comparison_artifacts(comparison, tmp_path / "comparison.json", tmp_path / "report.md")

    assert json.loads(json_path.read_text(encoding="utf-8"))["comparison_depth"] == 2
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown == render_reranker_comparison_markdown(comparison)
    assert "## Four-way benchmark table" in markdown
    assert "## Controlled reranker ablation" in markdown
    assert "## Failure cases where reranking hurts" in markdown
    assert "## Quality and latency tradeoff" in markdown


def test_checked_in_day27_artifacts_are_consistent():
    project_root = Path(__file__).resolve().parents[1]
    reranked = json.loads((project_root / "reports/evaluations/hybrid_rrf_cross_encoder.json").read_text(encoding="utf-8"))
    comparison = json.loads((project_root / "reports/evaluations/reranker_vs_baselines.json").read_text(encoding="utf-8"))
    markdown = (project_root / "reports/week4_reranker_comparison.md").read_text(encoding="utf-8")

    assert reranked["metrics"]["question_count"] == comparison["question_count"] == 45
    assert reranked["dense_index"]["points_count"] == reranked["bm25_index"]["source_record_count"] == 13_481
    assert reranked["bm25_index"]["document_count"] == 13_476
    assert reranked["metrics"]["mrr"] == comparison["metrics"]["mrr_at_depth"]["reranked"]
    assert reranked["pre_rerank_metrics"]["mrr"] == comparison["metrics"]["mrr_at_depth"]["pre_rerank"]
    assert comparison["decision"]["best_retriever"] == "reranked"
    assert sum(comparison["paired"]["pre_rerank"][key] for key in ("reranked_wins", "pre_rerank_wins", "ties")) == 45
    assert comparison["relevance_groups"]["group_count"] == 20
    assert all(len(question["candidate_chunk_ids"]) == 25 and len(question["retrieved_chunk_ids"]) == 5 for question in reranked["questions"])
    assert markdown == render_reranker_comparison_markdown(comparison)
