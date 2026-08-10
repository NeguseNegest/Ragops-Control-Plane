import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ragops.evaluation.retrieval_labels import (
    RetrievalLabel,
    bootstrap_labels_from_approved_candidates,
    build_retrieval_label,
    load_retrieval_labels,
    merge_retrieval_labels,
    rank_candidate_chunks,
    resolve_chunk_selection,
    validate_retrieval_labels,
)
from ragops.evaluation.synthetic_qa import SyntheticQACandidate, read_jsonl, write_jsonl


def make_golden(question_id="gqa-001", question="How does the documented feature work?", expected_source="fastapi/docs/example.md", query_type="supported"):
    return {
        "id": question_id,
        "question": question,
        "expected_answer": "The documented feature works according to the source.",
        "expected_source": expected_source,
        "query_type": query_type,
        "difficulty": "easy",
    }


def make_chunk(chunk_id="chunk-1", source="fastapi/docs/example.md", text="The documented feature validates input data.", chunk_index=0):
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "text": text,
        "metadata": {
            "relative_path": source,
            "source_name": source.split("/", 1)[0],
            "heading": "Feature",
            "chunk_index": chunk_index,
        },
    }


def make_label(golden=None, chunk_ids=None):
    return build_retrieval_label(
        golden or make_golden(),
        chunk_ids or ["chunk-1"],
        reviewer="test-reviewer",
    )


def make_candidate(status="approved"):
    return SyntheticQACandidate(
        id="gqa-001",
        question="How does the documented feature work?",
        expected_answer="The documented feature works according to the source.",
        expected_source="fastapi/docs/example.md",
        difficulty="easy",
        metadata={
            "provider": "openai",
            "model": "test-model",
            "source_chunk_id": "chunk-1",
            "review_status": status,
            "reviewed_by": "test-reviewer",
        },
    )


def test_retrieval_label_rejects_empty_or_duplicate_chunk_ids():
    base_record = {
        "question_id": "gqa-001",
        "question": "How does the documented feature work?",
        "expected_source": "fastapi/docs/example.md",
        "metadata": {"label_method": "manual", "reviewed_by": "reviewer"},
    }

    with pytest.raises(ValidationError):
        RetrievalLabel(**base_record, relevant_chunk_ids=[])

    with pytest.raises(ValidationError, match="duplicates"):
        RetrievalLabel(**base_record, relevant_chunk_ids=["chunk-1", "chunk-1"])


def test_validate_retrieval_labels_accepts_consistent_records():
    label = make_label()

    summary = validate_retrieval_labels([label], [make_golden()], [make_chunk()], minimum_count=1)

    assert summary == {"label_count": 1, "question_count": 1, "relevant_chunk_count": 1}


@pytest.mark.parametrize(
    ("label", "golden", "chunks", "message"),
    [
        (make_label(make_golden(question_id="missing")), [make_golden()], [make_chunk()], "unknown golden question"),
        (make_label(make_golden(question="A different supported question?")), [make_golden()], [make_chunk()], "question text does not match"),
        (make_label(make_golden(expected_source="mlflow/docs/example.mdx")), [make_golden()], [make_chunk()], "expected source does not match"),
        (make_label(chunk_ids=["missing-chunk"]), [make_golden()], [make_chunk()], "unknown chunk ID"),
        (make_label(), [make_golden()], [make_chunk(source="fastapi/docs/other.md")], "does not belong to expected source"),
    ],
)
def test_validate_retrieval_labels_rejects_inconsistent_references(label, golden, chunks, message):
    with pytest.raises(ValueError, match=message):
        validate_retrieval_labels([label], golden, chunks)


def test_validate_retrieval_labels_rejects_unsupported_and_duplicate_questions():
    unsupported = make_golden(query_type="unsupported")

    with pytest.raises(ValueError, match="Only supported"):
        make_label(unsupported)

    with pytest.raises(ValueError, match="must be supported"):
        validate_retrieval_labels([make_label()], [unsupported], [make_chunk()])

    with pytest.raises(ValueError, match="Duplicate retrieval label"):
        validate_retrieval_labels([make_label(), make_label()], [make_golden()], [make_chunk()])


def test_validate_retrieval_labels_enforces_minimum_count():
    with pytest.raises(ValueError, match="at least 1"):
        validate_retrieval_labels([], [make_golden()], [make_chunk()], minimum_count=1)


def test_bootstrap_uses_only_approved_candidates_and_verified_source_chunk():
    approved = make_candidate("approved")
    rejected = make_candidate("rejected").model_copy(update={"id": "gqa-002", "question": "Why is the second feature rejected?"})
    labels = bootstrap_labels_from_approved_candidates(
        [make_golden()],
        [approved, rejected],
        [make_chunk()],
        reviewer="source-reviewer",
    )

    assert len(labels) == 1
    assert labels[0].question_id == "gqa-001"
    assert labels[0].relevant_chunk_ids == ["chunk-1"]
    assert labels[0].metadata.label_method == "verified_synthetic_source"
    assert labels[0].metadata.reviewed_by == "source-reviewer"


def test_bootstrap_rejects_missing_or_wrong_source_chunk():
    with pytest.raises(ValueError, match="unknown source chunk"):
        bootstrap_labels_from_approved_candidates([make_golden()], [make_candidate()], [make_chunk("other")])

    with pytest.raises(ValueError, match="does not match expected source"):
        bootstrap_labels_from_approved_candidates([make_golden()], [make_candidate()], [make_chunk(source="fastapi/docs/other.md")])


def test_merge_retrieval_labels_is_idempotent_and_rejects_conflicts():
    label = make_label()
    merged, added = merge_retrieval_labels([], [label])
    merged_again, added_again = merge_retrieval_labels(merged, [label])

    assert merged == [label]
    assert added == [label]
    assert merged_again == [label]
    assert added_again == []

    conflicting = label.model_copy(deep=True)
    conflicting.relevant_chunk_ids = ["chunk-2"]
    with pytest.raises(ValueError, match="Conflicting retrieval label"):
        merge_retrieval_labels(merged, [conflicting])


def test_rank_candidate_chunks_stays_within_source_and_prefers_known_chunk():
    chunks = [
        make_chunk("chunk-1", text="Unrelated setup details.", chunk_index=0),
        make_chunk("chunk-2", text="The feature validates input data and returns errors.", chunk_index=1),
        make_chunk("chunk-3", source="mlflow/docs/example.mdx", text="The feature validates input data.", chunk_index=0),
    ]

    ranked = rank_candidate_chunks(
        "How does the feature validate input data?",
        "fastapi/docs/example.md",
        chunks,
        preferred_chunk_ids=["chunk-1"],
        limit=2,
    )

    assert [chunk["chunk_id"] for chunk, _ in ranked] == ["chunk-1", "chunk-2"]
    assert all(chunk["metadata"]["relative_path"] == "fastapi/docs/example.md" for chunk, _ in ranked)


def test_resolve_chunk_selection_accepts_numbers_and_ids():
    displayed = [(make_chunk("chunk-1"), 2), (make_chunk("chunk-2"), 1)]

    assert resolve_chunk_selection("1, chunk-2", displayed) == ["chunk-1", "chunk-2"]

    with pytest.raises(ValueError, match="outside"):
        resolve_chunk_selection("3", displayed)
    with pytest.raises(ValueError, match="more than once"):
        resolve_chunk_selection("1,chunk-1", displayed)


def test_retrieval_label_jsonl_round_trip(tmp_path):
    path = tmp_path / "retrieval_labels.jsonl"
    write_jsonl([make_label()], path)

    labels = load_retrieval_labels(path)

    assert labels == [make_label()]
    assert json.loads(path.read_text(encoding="utf-8"))["question_id"] == "gqa-001"


def test_day_17_dataset_meets_acceptance_criteria():
    project_root = Path(__file__).resolve().parents[1]
    labels_path = project_root / "data/eval/retrieval_labels.jsonl"

    if not labels_path.exists():
        pytest.skip("Day 17 label dataset has not been generated yet.")

    labels = load_retrieval_labels(labels_path)
    golden_records = read_jsonl(project_root / "data/eval/golden_qa.jsonl")
    golden_by_id = {record["id"]: record for record in golden_records}

    assert len(labels) >= 40
    assert len({label.question_id for label in labels}) == len(labels)
    assert all(label.question_id in golden_by_id for label in labels)
    assert all(golden_by_id[label.question_id]["query_type"] == "supported" for label in labels)
    assert all(golden_by_id[label.question_id]["question"] == label.question for label in labels)
    assert all(label.metadata.review_status == "verified" for label in labels)
