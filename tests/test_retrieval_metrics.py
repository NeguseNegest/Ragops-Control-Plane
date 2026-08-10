import math
from pathlib import Path

import pytest

from ragops.evaluation.retrieval_labels import RetrievalLabel, load_retrieval_labels
from ragops.evaluation.retrieval_metrics import (
    evaluate_retrieval_metrics,
    hit_at_k,
    hit_rate_at_k,
    mean_ndcg_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    normalize_k_values,
    recall_at_k,
    reciprocal_rank,
    relevance_from_labels,
)


def make_label(question_id, relevant_chunk_ids):
    return RetrievalLabel(
        question_id=question_id,
        question=f"How does documented feature {question_id} work?",
        relevant_chunk_ids=relevant_chunk_ids,
        expected_source="fastapi/docs/example.md",
        metadata={"label_method": "manual", "reviewed_by": "test-reviewer"},
    )


def test_recall_at_k_counts_unique_relevant_chunks():
    retrieved = ["irrelevant", "relevant-a", "relevant-a", "relevant-b"]
    relevant = ["relevant-a", "relevant-b", "relevant-c"]

    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, relevant, 10) == pytest.approx(2 / 3)


def test_reciprocal_rank_uses_first_relevant_position_and_optional_cutoff():
    retrieved = ["irrelevant", "relevant-b", "relevant-a"]
    relevant = ["relevant-a", "relevant-b"]

    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert reciprocal_rank(retrieved, relevant, k=1) == 0.0
    assert reciprocal_rank(retrieved, relevant, k=2) == 0.5
    assert reciprocal_rank(["irrelevant"], relevant) == 0.0


def test_hit_at_k_is_binary():
    retrieved = ["irrelevant", "relevant"]
    relevant = ["relevant"]

    assert hit_at_k(retrieved, relevant, 1) == 0.0
    assert hit_at_k(retrieved, relevant, 2) == 1.0
    assert hit_at_k(["relevant", "relevant"], relevant, 2) == 1.0


def test_ndcg_at_k_uses_binary_discounted_gain_without_duplicate_gain():
    retrieved = ["relevant-a", "irrelevant", "relevant-b"]
    relevant = ["relevant-a", "relevant-b"]
    ideal_dcg = 1.0 + 1.0 / math.log2(3)
    expected_dcg = 1.0 + 1.0 / math.log2(4)

    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(expected_dcg / ideal_dcg)
    assert ndcg_at_k(["relevant-a", "relevant-a", "relevant-b"], relevant, 3) == pytest.approx(expected_dcg / ideal_dcg)
    assert ndcg_at_k(["irrelevant"], relevant, 3) == 0.0


def test_empty_ranking_returns_zero_for_every_metric():
    relevant = ["relevant-a", "relevant-b"]

    assert recall_at_k([], relevant, 5) == 0.0
    assert reciprocal_rank([], relevant) == 0.0
    assert hit_at_k([], relevant, 5) == 0.0
    assert ndcg_at_k([], relevant, 5) == 0.0


def test_perfect_order_can_have_full_ndcg_before_full_recall():
    relevant = ["relevant-a", "relevant-b"]

    assert ndcg_at_k(["relevant-a"], relevant, 1) == 1.0
    assert recall_at_k(["relevant-a"], relevant, 1) == 0.5


@pytest.mark.parametrize("metric", [recall_at_k, hit_at_k, ndcg_at_k])
@pytest.mark.parametrize("invalid_k", [0, -1, 1.5, True, "5"])
def test_at_k_metrics_reject_invalid_cutoffs(metric, invalid_k):
    with pytest.raises(ValueError, match="positive integer"):
        metric(["chunk-1"], ["chunk-1"], invalid_k)


def test_reciprocal_rank_rejects_invalid_optional_cutoff():
    with pytest.raises(ValueError, match="positive integer"):
        reciprocal_rank(["chunk-1"], ["chunk-1"], k=0)


@pytest.mark.parametrize(
    ("retrieved", "relevant", "message"),
    [
        ("chunk-1", ["chunk-1"], "not a string"),
        (["chunk-1"], "chunk-1", "not a string"),
        ([""], ["chunk-1"], "non-empty"),
        (["chunk-1"], [], "must not be empty"),
        (["chunk-1"], ["chunk-1", "chunk-1"], "duplicates"),
    ],
)
def test_metrics_reject_invalid_chunk_id_inputs(retrieved, relevant, message):
    with pytest.raises(ValueError, match=message):
        recall_at_k(retrieved, relevant, 1)


def test_macro_metrics_average_each_question_equally():
    rankings = {
        "q1": ["a", "x"],
        "q2": ["x", "c"],
    }
    relevance = {
        "q1": ["a"],
        "q2": ["c", "d"],
    }

    assert mean_recall_at_k(rankings, relevance, 1) == 0.5
    assert mean_recall_at_k(rankings, relevance, 2) == 0.75
    assert mean_reciprocal_rank(rankings, relevance) == 0.75
    assert hit_rate_at_k(rankings, relevance, 1) == 0.5
    assert hit_rate_at_k(rankings, relevance, 2) == 1.0

    q2_ndcg_at_2 = (1.0 / math.log2(3)) / (1.0 + 1.0 / math.log2(3))
    assert mean_ndcg_at_k(rankings, relevance, 2) == pytest.approx((1.0 + q2_ndcg_at_2) / 2)


def test_aggregate_metrics_require_every_labeled_question_ranking():
    relevance = {"q1": ["a"], "q2": ["b"]}

    with pytest.raises(ValueError, match="Missing retrieved ranking.*q2"):
        mean_recall_at_k({"q1": ["a"]}, relevance, 1)

    with pytest.raises(ValueError, match="non-empty mapping"):
        mean_recall_at_k({}, {}, 1)


def test_relevance_from_labels_rejects_empty_and_duplicate_labels():
    with pytest.raises(ValueError, match="At least one"):
        relevance_from_labels([])

    label = make_label("q1", ["a"])
    with pytest.raises(ValueError, match="Duplicate retrieval label"):
        relevance_from_labels([label, label])


def test_normalize_k_values_preserves_order_and_rejects_bad_sequences():
    assert normalize_k_values((5, 1, 10)) == [5, 1, 10]

    with pytest.raises(ValueError, match="At least one"):
        normalize_k_values([])
    with pytest.raises(ValueError, match="duplicates"):
        normalize_k_values([1, 1])
    with pytest.raises(ValueError, match="sequence"):
        normalize_k_values("1,5")


def test_evaluate_retrieval_metrics_returns_json_ready_summary():
    labels = [make_label("q1", ["a"]), make_label("q2", ["c", "d"])]
    rankings = {"q1": ["a", "x"], "q2": ["x", "c"]}

    metrics = evaluate_retrieval_metrics(rankings, labels, k_values=(1, 2))

    assert metrics["question_count"] == 2
    assert metrics["k_values"] == [1, 2]
    assert metrics["mrr"] == 0.75
    assert metrics["recall_at_k"] == {"1": 0.5, "2": 0.75}
    assert metrics["hit_rate_at_k"] == {"1": 0.5, "2": 1.0}
    assert metrics["ndcg_at_k"]["1"] == 0.5
    assert 0.0 < metrics["ndcg_at_k"]["2"] < 1.0


def test_day_17_labels_produce_perfect_scores_for_perfect_rankings():
    project_root = Path(__file__).resolve().parents[1]
    labels = load_retrieval_labels(project_root / "data/eval/retrieval_labels.jsonl")
    rankings = {label.question_id: list(label.relevant_chunk_ids) for label in labels}

    metrics = evaluate_retrieval_metrics(rankings, labels, k_values=(1, 5))

    assert metrics["question_count"] >= 40
    assert metrics["mrr"] == 1.0
    assert metrics["recall_at_k"] == {"1": 1.0, "5": 1.0}
    assert metrics["hit_rate_at_k"] == {"1": 1.0, "5": 1.0}
    assert metrics["ndcg_at_k"] == {"1": 1.0, "5": 1.0}
