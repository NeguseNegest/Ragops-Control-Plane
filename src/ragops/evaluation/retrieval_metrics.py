import math
from collections.abc import Mapping

from ragops.evaluation.retrieval_labels import RetrievalLabel


def validate_k(k):
    """Return a valid positive retrieval cutoff."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    return k


def clean_chunk_ids(chunk_ids, name):
    """Return a validated list of chunk ID strings."""
    if isinstance(chunk_ids, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of chunk IDs, not a string.")

    try:
        values = list(chunk_ids)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of chunk IDs.") from error

    cleaned_ids = []
    for chunk_id in values:
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"{name} must contain only non-empty string chunk IDs.")
        cleaned_ids.append(chunk_id.strip())

    return cleaned_ids


def relevant_chunk_set(relevant_chunk_ids):
    """Return a non-empty, duplicate-free relevance set."""
    relevant_ids = clean_chunk_ids(relevant_chunk_ids, "relevant_chunk_ids")
    if not relevant_ids:
        raise ValueError("relevant_chunk_ids must not be empty.")
    if len(relevant_ids) != len(set(relevant_ids)):
        raise ValueError("relevant_chunk_ids must not contain duplicates.")
    return set(relevant_ids)


def recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, k):
    """Return the fraction of relevant chunks retrieved in the first k ranks."""
    k = validate_k(k)
    retrieved_ids = clean_chunk_ids(retrieved_chunk_ids, "retrieved_chunk_ids")
    relevant_ids = relevant_chunk_set(relevant_chunk_ids)
    retrieved_relevant_ids = set(retrieved_ids[:k]) & relevant_ids
    return len(retrieved_relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids, k=None):
    """Return the reciprocal rank of the first relevant chunk."""
    retrieved_ids = clean_chunk_ids(retrieved_chunk_ids, "retrieved_chunk_ids")
    relevant_ids = relevant_chunk_set(relevant_chunk_ids)

    if k is not None:
        retrieved_ids = retrieved_ids[: validate_k(k)]

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def hit_at_k(retrieved_chunk_ids, relevant_chunk_ids, k):
    """Return 1.0 when at least one relevant chunk appears in the first k ranks."""
    k = validate_k(k)
    retrieved_ids = clean_chunk_ids(retrieved_chunk_ids, "retrieved_chunk_ids")
    relevant_ids = relevant_chunk_set(relevant_chunk_ids)
    return float(any(chunk_id in relevant_ids for chunk_id in retrieved_ids[:k]))


def ndcg_at_k(retrieved_chunk_ids, relevant_chunk_ids, k):
    """Return binary normalized discounted cumulative gain at k."""
    k = validate_k(k)
    retrieved_ids = clean_chunk_ids(retrieved_chunk_ids, "retrieved_chunk_ids")
    relevant_ids = relevant_chunk_set(relevant_chunk_ids)
    seen_relevant_ids = set()
    discounted_gain = 0.0

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids and chunk_id not in seen_relevant_ids:
            discounted_gain += 1.0 / math.log2(rank + 1)
            seen_relevant_ids.add(chunk_id)

    ideal_result_count = min(len(relevant_ids), k)
    ideal_discounted_gain = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_result_count + 1))
    return discounted_gain / ideal_discounted_gain


def normalize_metric_inputs(rankings, relevance_by_question):
    """Validate and materialize aggregate metric inputs."""
    if not isinstance(rankings, Mapping):
        raise ValueError("rankings must map question IDs to ranked chunk IDs.")
    if not isinstance(relevance_by_question, Mapping) or not relevance_by_question:
        raise ValueError("relevance_by_question must be a non-empty mapping.")

    normalized_relevance = {}
    normalized_rankings = {}

    for question_id, relevant_ids in relevance_by_question.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("Relevance question IDs must be non-empty strings.")
        if question_id not in rankings:
            raise ValueError(f"Missing retrieved ranking for question: {question_id}")

        normalized_relevance[question_id] = list(relevant_chunk_set(relevant_ids))
        normalized_rankings[question_id] = clean_chunk_ids(rankings[question_id], f"ranking for {question_id}")

    return normalized_rankings, normalized_relevance


def mean_recall_at_k(rankings, relevance_by_question, k):
    """Return macro-average Recall@k across labeled questions."""
    k = validate_k(k)
    rankings, relevance_by_question = normalize_metric_inputs(rankings, relevance_by_question)
    scores = [recall_at_k(rankings[question_id], relevant_ids, k) for question_id, relevant_ids in relevance_by_question.items()]
    return sum(scores) / len(scores)


def mean_reciprocal_rank(rankings, relevance_by_question, k=None):
    """Return mean reciprocal rank across labeled questions."""
    if k is not None:
        k = validate_k(k)
    rankings, relevance_by_question = normalize_metric_inputs(rankings, relevance_by_question)
    scores = [reciprocal_rank(rankings[question_id], relevant_ids, k=k) for question_id, relevant_ids in relevance_by_question.items()]
    return sum(scores) / len(scores)


def hit_rate_at_k(rankings, relevance_by_question, k):
    """Return the fraction of questions with a relevant result in the first k ranks."""
    k = validate_k(k)
    rankings, relevance_by_question = normalize_metric_inputs(rankings, relevance_by_question)
    scores = [hit_at_k(rankings[question_id], relevant_ids, k) for question_id, relevant_ids in relevance_by_question.items()]
    return sum(scores) / len(scores)


def mean_ndcg_at_k(rankings, relevance_by_question, k):
    """Return macro-average binary nDCG@k across labeled questions."""
    k = validate_k(k)
    rankings, relevance_by_question = normalize_metric_inputs(rankings, relevance_by_question)
    scores = [ndcg_at_k(rankings[question_id], relevant_ids, k) for question_id, relevant_ids in relevance_by_question.items()]
    return sum(scores) / len(scores)


def relevance_from_labels(labels):
    """Convert retrieval label rows into a question-to-relevant-chunks mapping."""
    relevance_by_question = {}

    for raw_label in labels:
        label = raw_label if isinstance(raw_label, RetrievalLabel) else RetrievalLabel.model_validate(raw_label)
        if label.question_id in relevance_by_question:
            raise ValueError(f"Duplicate retrieval label question_id: {label.question_id}")
        relevance_by_question[label.question_id] = list(label.relevant_chunk_ids)

    if not relevance_by_question:
        raise ValueError("At least one retrieval label is required.")

    return relevance_by_question


def normalize_k_values(k_values):
    """Return unique positive cutoffs in caller-specified order."""
    if isinstance(k_values, (str, bytes)):
        raise ValueError("k_values must be a sequence of positive integers.")

    try:
        normalized_values = [validate_k(k) for k in k_values]
    except TypeError as error:
        raise ValueError("k_values must be a sequence of positive integers.") from error

    if not normalized_values:
        raise ValueError("At least one k value is required.")
    if len(normalized_values) != len(set(normalized_values)):
        raise ValueError("k_values must not contain duplicates.")

    return normalized_values


def evaluate_retrieval_metrics(rankings, labels, k_values=(1, 3, 5, 10)):
    """Compute aggregate retrieval metrics for validated label rows."""
    k_values = normalize_k_values(k_values)
    relevance_by_question = relevance_from_labels(labels)
    rankings, relevance_by_question = normalize_metric_inputs(rankings, relevance_by_question)

    return {
        "question_count": len(relevance_by_question),
        "k_values": k_values,
        "mrr": mean_reciprocal_rank(rankings, relevance_by_question),
        "recall_at_k": {str(k): mean_recall_at_k(rankings, relevance_by_question, k) for k in k_values},
        "hit_rate_at_k": {str(k): hit_rate_at_k(rankings, relevance_by_question, k) for k in k_values},
        "ndcg_at_k": {str(k): mean_ndcg_at_k(rankings, relevance_by_question, k) for k in k_values},
    }
