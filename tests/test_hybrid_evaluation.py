import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragops.evaluation.hybrid_runner import (
    compare_hybrid_reports,
    evaluate_hybrid_config,
    render_hybrid_comparison_markdown,
    run_hybrid_evaluation,
    write_hybrid_comparison_artifacts,
)
from ragops.evaluation.retrieval_labels import RetrievalLabel
from ragops.evaluation.retrieval_metrics import evaluate_retrieval_metrics
from ragops.evaluation.runner import build_question_metrics
from ragops.retrieval.bm25 import BM25Index, build_bm25_index
from ragops.retrieval.dense import RetrievedChunk
from ragops.retrieval.hybrid import HybridConfig, reciprocal_rank_fusion


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
    return HybridConfig.model_validate(
        {
            "name": "hybrid_test",
            "input": {"chunks_path": tmp_path / "chunks.jsonl"},
            "dense": {"type": "dense", "collection_name": "test_chunks", "embedding_model": "test-model", "top_k": 3},
            "bm25": {"type": "bm25", "index_path": tmp_path / "index.json.gz", "top_k": 3},
            "fusion": {"type": "rrf", "rank_constant": 60, "top_k": 2},
            "evaluation": {
                "labels_path": tmp_path / "labels.jsonl",
                "k_values": [1, 2],
                "minimum_labels": 1,
                "dense_baseline_path": tmp_path / "dense.json",
                "bm25_baseline_path": tmp_path / "bm25.json",
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


def test_run_hybrid_evaluation_computes_metrics_timings_and_provenance(tmp_path):
    config = make_config(tmp_path)
    labels = [make_label("q1", "What is alpha?", "a"), make_label("q2", "What is beta?", "b")]

    def fake_retriever(query, config, client, index, clock, timings):
        timings.update({"dense_ms": 10.0, "bm25_ms": 4.0, "fusion_ms": 1.0})
        if "alpha" in query:
            return reciprocal_rank_fusion(
                {"dense": [make_chunk("a", 1)], "bm25": [make_chunk("a", 1)]},
                top_k=2,
            )
        return reciprocal_rank_fusion(
            {"dense": [make_chunk("a", 1)], "bm25": [make_chunk("b", 1)]},
            top_k=2,
        )

    report = run_hybrid_evaluation(
        config,
        labels,
        client="client",
        index=make_index(),
        retriever=fake_retriever,
        clock=FakeClock([0.0, 0.02, 1.0, 1.03]),
        dense_index={"collection_name": "test_chunks", "points_count": 2},
    )

    assert report["metrics"]["mrr"] == pytest.approx(0.75)
    assert report["metrics"]["hit_rate_at_k"] == {"1": 0.5, "2": 1.0}
    assert report["latency_ms"]["average"] == pytest.approx(25.0)
    assert report["component_latency_ms"]["dense"]["average"] == 10.0
    assert report["component_latency_ms"]["fusion"]["total"] == 2.0
    assert report["dense_index"]["points_count"] == 2
    assert report["bm25_index"]["source_sha256"] == "a" * 64
    assert report["questions"][0]["retrieved_fusion_sources"][0]["dense"]["rank"] == 1


def test_evaluate_hybrid_config_rejects_index_count_drift_and_closes_client(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config.input.chunks_path.write_text("test", encoding="utf-8")
    index = make_index()
    client = FakeClient(points_count=3)
    monkeypatch.setattr("ragops.evaluation.hybrid_runner.validate_bm25_index", lambda loaded_index, config: loaded_index.payload)

    with pytest.raises(RuntimeError, match="contains 3 points"):
        evaluate_hybrid_config(
            config,
            [make_label("q1", "What is alpha?", "a")],
            client_factory=lambda url: client,
            index_loader=lambda path: index,
        )

    assert client.closed is True


def build_report(run_name, retriever_name, labels, rankings):
    questions = []
    for label in labels:
        retrieved = rankings[label.question_id]
        question = {
            "question_id": label.question_id,
            "question": label.question,
            "expected_source": label.expected_source,
            "relevant_chunk_ids": label.relevant_chunk_ids,
            "retrieved_chunk_ids": retrieved,
            "retrieved_scores": [1.0] * len(retrieved),
            "latency_ms": 10.0,
            **build_question_metrics(retrieved, label.relevant_chunk_ids, [1, 2]),
        }
        if retriever_name == "hybrid":
            question["retrieved_fusion_sources"] = [{"dense": {"rank": position}} for position, _ in enumerate(retrieved, start=1)]
            question["component_latency_ms"] = {"dense_ms": 5.0, "bm25_ms": 4.0, "fusion_ms": 1.0}
        questions.append(question)

    if retriever_name == "hybrid":
        configuration = {
            "dense": {"type": "dense", "collection_name": "chunks", "embedding_model": "model"},
            "bm25": {"type": "bm25", "tokenizer": "technical_v1", "k1": 1.5, "b": 0.75, "epsilon": 0.25},
            "fusion": {"type": "rrf"},
        }
    elif retriever_name == "dense":
        configuration = {"retriever": {"type": "dense", "collection_name": "chunks", "embedding_model": "model"}}
    else:
        configuration = {
            "retriever": {"type": "bm25", "tokenizer": "technical_v1", "k1": 1.5, "b": 0.75, "epsilon": 0.25}
        }
    report = {
        "schema_version": 1,
        "run_name": run_name,
        "configuration": configuration,
        "metrics": evaluate_retrieval_metrics(rankings, labels, [1, 2]),
        "latency_ms": {"total": 30.0, "average": 10.0, "minimum": 10.0, "maximum": 10.0},
        "questions": questions,
    }
    if retriever_name == "bm25":
        report["index"] = {"source_sha256": "a" * 64}
    if retriever_name == "hybrid":
        report.update(
            {
                "dense_index": {"collection_name": "chunks", "points_count": 3},
                "bm25_index": {
                    "source_sha256": "a" * 64,
                    "source_record_count": 3,
                    "document_count": 3,
                    "skipped_document_count": 0,
                },
                "component_latency_ms": {
                    name: {"total": 3.0, "average": 1.0, "minimum": 1.0, "maximum": 1.0}
                    for name in ("dense", "bm25", "fusion")
                },
            }
        )
    return report


def comparison_fixture():
    labels = [
        make_label("q1", "What is the exact alpha parameter?", "a"),
        make_label("q2", "What happens when beta fails?", "b"),
        make_label("q3", "Describe gamma behavior.", "c"),
    ]
    dense = build_report("dense", "dense", labels, {"q1": ["a"], "q2": ["x"], "q3": ["x", "c"]})
    bm25 = build_report("bm25", "bm25", labels, {"q1": ["x", "a"], "q2": ["b"], "q3": ["x"]})
    hybrid = build_report("hybrid", "hybrid", labels, {"q1": ["a"], "q2": ["x", "b"], "q3": ["c"]})
    return dense, bm25, hybrid


def test_compare_hybrid_reports_builds_three_way_metrics_and_paired_outcomes():
    dense, bm25, hybrid = comparison_fixture()

    comparison = compare_hybrid_reports(dense, bm25, hybrid)

    assert comparison["question_count"] == 3
    assert comparison["metrics"]["mrr"]["dense"] == pytest.approx(0.5)
    assert comparison["metrics"]["mrr"]["bm25"] == pytest.approx(0.5)
    assert comparison["metrics"]["mrr"]["hybrid"] == pytest.approx(5 / 6)
    assert comparison["decision"] == {
        "primary_metric": "mrr",
        "best_retriever": "hybrid",
        "hybrid_improves_over_dense": True,
        "hybrid_improves_over_bm25": True,
    }
    assert comparison["paired"]["hybrid_vs_bm25"]["hybrid_wins"] == 2
    assert comparison["paired"]["hybrid_vs_bm25"]["bm25_wins"] == 1
    assert comparison["paired"]["hybrid_vs_best_component"] == {"hybrid_wins": 1, "best_component_wins": 1, "ties": 1}
    assert comparison["query_types"]["exact-reference"]["hybrid_vs_bm25_wins"] == 1


def test_compare_hybrid_reports_rejects_source_hash_and_question_drift():
    dense, bm25, hybrid = comparison_fixture()
    hybrid["bm25_index"]["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="SHA256"):
        compare_hybrid_reports(dense, bm25, hybrid)

    dense, bm25, hybrid = comparison_fixture()
    hybrid["questions"][0]["question"] = "different"
    with pytest.raises(ValueError, match="question differs"):
        compare_hybrid_reports(dense, bm25, hybrid)


def test_write_hybrid_comparison_artifacts_emits_benchmark_and_analysis(tmp_path):
    comparison = compare_hybrid_reports(*comparison_fixture())

    json_path, markdown_path = write_hybrid_comparison_artifacts(comparison, tmp_path / "comparison.json", tmp_path / "report.md")

    assert json.loads(json_path.read_text(encoding="utf-8"))["decision"]["best_retriever"] == "hybrid"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown == render_hybrid_comparison_markdown(comparison)
    assert "## Benchmark table" in markdown
    assert "## Hybrid regressions versus BM25" in markdown
    assert "## Hybrid top-10 failures" in markdown


def test_checked_in_day25_artifacts_are_consistent():
    project_root = Path(__file__).resolve().parents[1]
    hybrid = json.loads((project_root / "reports/evaluations/hybrid_rrf.json").read_text(encoding="utf-8"))
    comparison = json.loads((project_root / "reports/evaluations/hybrid_vs_baselines.json").read_text(encoding="utf-8"))
    markdown = (project_root / "reports/week4_hybrid_comparison.md").read_text(encoding="utf-8")

    assert hybrid["metrics"]["question_count"] == comparison["question_count"] == 45
    assert hybrid["dense_index"]["points_count"] == hybrid["bm25_index"]["source_record_count"] == 13_481
    assert comparison["metrics"]["mrr"]["hybrid"] == hybrid["metrics"]["mrr"]
    assert comparison["decision"]["best_retriever"] == "bm25"
    assert sum(comparison["paired"]["hybrid_vs_bm25"][key] for key in ("hybrid_wins", "bm25_wins", "ties")) == 45
    assert comparison["relevance_groups"]["group_count"] == 20
    assert markdown == render_hybrid_comparison_markdown(comparison)
