import json
from pathlib import Path

import pytest

from ragops.evaluation.bm25_runner import (
    classify_query_type,
    compare_evaluation_reports,
    render_comparison_markdown,
    run_bm25_evaluation,
    write_comparison_artifacts,
)
from ragops.evaluation.retrieval_labels import RetrievalLabel
from ragops.evaluation.retrieval_metrics import evaluate_retrieval_metrics
from ragops.evaluation.runner import build_question_metrics
from ragops.retrieval.bm25 import BM25Config, BM25Index, build_bm25_index


class FakeClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def make_label(question_id, question, relevant_chunk_id, expected_source="docs/example.md"):
    return RetrievalLabel(
        question_id=question_id,
        question=question,
        relevant_chunk_ids=[relevant_chunk_id],
        expected_source=expected_source,
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


def make_config(tmp_path):
    return BM25Config.model_validate(
        {
            "name": "bm25_test",
            "input": {"chunks_path": tmp_path / "chunks.jsonl"},
            "retriever": {"type": "bm25", "index_path": tmp_path / "index.json.gz", "top_k": 2},
            "evaluation": {
                "labels_path": tmp_path / "labels.jsonl",
                "k_values": [1, 2],
                "minimum_labels": 1,
                "dense_baseline_path": tmp_path / "dense.json",
            },
            "output": {
                "directory": tmp_path / "evaluations",
                "comparison_path": tmp_path / "comparison.json",
                "report_path": tmp_path / "comparison.md",
            },
        }
    )


def make_index():
    payload = build_bm25_index(
        [make_record("relevant-a", "alpha evidence"), make_record("relevant-b", "beta evidence")],
        source_path="chunks.jsonl",
        source_sha256="a" * 64,
    )
    return BM25Index(payload)


def test_run_bm25_evaluation_reuses_dense_metrics_and_records_provenance(tmp_path):
    labels = [
        make_label("q-1", "What is alpha?", "relevant-a"),
        make_label("q-2", "What is beta?", "relevant-b"),
    ]

    def fake_retriever(query, index, top_k):
        assert isinstance(index, BM25Index)
        assert top_k == 2
        if "alpha" in query:
            return [{"chunk_id": "noise", "score": 2.0}, {"chunk_id": "relevant-a", "score": 1.0}]
        return [{"chunk_id": "relevant-b", "score": 3.0}]

    report = run_bm25_evaluation(
        make_config(tmp_path),
        labels,
        make_index(),
        retriever=fake_retriever,
        clock=FakeClock([0.0, 0.01, 1.0, 1.02]),
    )

    assert report["metrics"]["question_count"] == 2
    assert report["metrics"]["mrr"] == pytest.approx(0.75)
    assert report["metrics"]["hit_rate_at_k"] == {"1": 0.5, "2": 1.0}
    assert report["latency_ms"]["average"] == pytest.approx(15.0)
    assert report["index"]["document_count"] == 2
    assert report["index"]["source_sha256"] == "a" * 64


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the exact MLflow command to use?", "exact-reference"),
        ("How does FastAPI validate this input?", "behavioral/procedural"),
        ("Name two advantages of sparse retrieval.", "conceptual/descriptive"),
    ],
)
def test_classify_query_type_uses_reproducible_wording_cohorts(question, expected):
    assert classify_query_type(question) == expected


def build_report(run_name, labels, rankings):
    questions = []
    for label in labels:
        retrieved = rankings[label.question_id]
        questions.append(
            {
                "question_id": label.question_id,
                "question": label.question,
                "expected_source": label.expected_source,
                "relevant_chunk_ids": label.relevant_chunk_ids,
                "retrieved_chunk_ids": retrieved,
                "retrieved_scores": [1.0] * len(retrieved),
                "latency_ms": 1.0,
                **build_question_metrics(retrieved, label.relevant_chunk_ids, [1, 2]),
            }
        )
    return {
        "schema_version": 1,
        "run_name": run_name,
        "configuration": {"retriever": {"type": run_name}},
        "metrics": evaluate_retrieval_metrics(rankings, labels, [1, 2]),
        "latency_ms": {"total": 3.0, "average": 1.0, "minimum": 1.0, "maximum": 1.0},
        "questions": questions,
    }


def comparison_fixture():
    labels = [
        make_label("q-exact", "What is the exact API endpoint?", "a"),
        make_label("q-behavior", "What happens when validation fails?", "b"),
        make_label("q-concept", "Name two advantages of UploadFile.", "c"),
    ]
    dense = build_report("dense", labels, {"q-exact": ["x", "a"], "q-behavior": ["b"], "q-concept": []})
    bm25 = build_report("bm25", labels, {"q-exact": ["a"], "q-behavior": [], "q-concept": ["c"]})
    return dense, bm25


def test_compare_evaluation_reports_builds_paired_wins_and_cohorts():
    dense, bm25 = comparison_fixture()

    comparison = compare_evaluation_reports(dense, bm25, dense_report_path="dense.json", bm25_report_path="bm25.json")

    assert comparison["question_count"] == 3
    assert comparison["wins"] == {
        "bm25": 2,
        "dense": 1,
        "tie": 0,
        "bm25_recovered_dense_miss": 1,
        "dense_recovered_bm25_miss": 1,
    }
    assert comparison["metrics"]["mrr"]["bm25"] == pytest.approx(2 / 3)
    assert comparison["metrics"]["mrr"]["dense"] == pytest.approx(0.5)
    assert comparison["query_types"]["exact-reference"]["bm25_wins"] == 1
    assert comparison["query_types"]["behavioral/procedural"]["dense_wins"] == 1


def test_compare_evaluation_reports_rejects_unpaired_questions():
    dense, bm25 = comparison_fixture()
    bm25["questions"][-1]["question_id"] = "q-other"

    with pytest.raises(ValueError, match="question IDs differ"):
        compare_evaluation_reports(dense, bm25)


def test_write_comparison_artifacts_emits_json_and_markdown(tmp_path):
    dense, bm25 = comparison_fixture()
    comparison = compare_evaluation_reports(dense, bm25)

    json_path, markdown_path = write_comparison_artifacts(comparison, tmp_path / "comparison.json", tmp_path / "comparison.md")

    assert json.loads(json_path.read_text(encoding="utf-8"))["wins"]["bm25"] == 2
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown == render_comparison_markdown(comparison)
    assert "# Dense vs BM25 Retrieval Baseline" in markdown
    assert "Questions where BM25 wins" in markdown


def test_checked_in_day23_artifacts_are_consistent():
    project_root = Path(__file__).resolve().parents[1]
    bm25_report = json.loads((project_root / "reports/evaluations/bm25_baseline.json").read_text(encoding="utf-8"))
    comparison = json.loads((project_root / "reports/evaluations/bm25_vs_dense.json").read_text(encoding="utf-8"))
    markdown = (project_root / "reports/week4_bm25_comparison.md").read_text(encoding="utf-8")

    assert bm25_report["metrics"]["question_count"] == 45
    assert comparison["question_count"] == 45
    assert sum(comparison["wins"][name] for name in ("bm25", "dense", "tie")) == 45
    assert comparison["metrics"]["mrr"]["bm25"] == bm25_report["metrics"]["mrr"]
    assert markdown == render_comparison_markdown(comparison)
