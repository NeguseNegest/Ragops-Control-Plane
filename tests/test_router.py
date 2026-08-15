import math

import pytest
from pydantic import ValidationError

from ragops.retrieval.dense import RetrievedChunk
from ragops.routing.probe import (
    FEATURE_SCHEMA_VERSION,
    INITIAL_PROBE_TOP_K,
    InitialRetrievalFeatures,
    RetrievalConfidenceFeatures,
    build_initial_retrieval_features,
    extract_lexical_complexity,
    run_initial_retrieval_probe,
    tokenize_query,
)


def make_chunk(chunk_id, score, rank):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        text=f"Text for {chunk_id}",
        score=score,
        rank=rank,
        metadata={"relative_path": "docs/example.md"},
        source_url="docs/example.md",
    )


def test_lexical_features_are_normalized_deterministic_and_model_free():
    query = "Compare caching and retrieval because latency matters"

    assert tokenize_query(f"  {query}  ") == (
        "compare",
        "caching",
        "and",
        "retrieval",
        "because",
        "latency",
        "matters",
    )
    features = extract_lexical_complexity(query)

    assert features.unique_token_count == 7
    assert features.unique_token_ratio == 1.0
    assert features.average_token_length == pytest.approx(47 / 7)
    assert features.maximum_token_length == 9
    assert features.long_token_count == 1
    assert features.long_token_ratio == pytest.approx(1 / 7)
    assert features.clause_marker_count == 2
    assert features.complexity_marker_count == 1


def test_feature_contract_contains_query_length_top_score_and_score_gap():
    query = "  Why compare dense retrieval and BM25?  "
    chunks = [make_chunk("chunk-1", 0.91, 1), make_chunk("chunk-2", 0.78, 2)]

    features = build_initial_retrieval_features(query, chunks)

    assert isinstance(features, InitialRetrievalFeatures)
    assert features.schema_version == FEATURE_SCHEMA_VERSION
    assert features.query_length.character_count == len(query.strip())
    assert features.query_length.token_count == 6
    assert features.lexical_complexity.complexity_marker_count == 2
    assert features.lexical_complexity.clause_marker_count == 1
    assert features.retrieval_confidence.requested_top_k == INITIAL_PROBE_TOP_K
    assert features.retrieval_confidence.result_count == 2
    assert features.retrieval_confidence.top_score == 0.91
    assert features.retrieval_confidence.score_gap == pytest.approx(0.13)


@pytest.mark.parametrize(
    ("chunks", "expected_top_score", "expected_gap"),
    [
        ([], None, None),
        ([make_chunk("chunk-1", -0.2, 1)], -0.2, None),
    ],
)
def test_feature_contract_represents_sparse_probe_results_without_fabricated_confidence(chunks, expected_top_score, expected_gap):
    confidence = build_initial_retrieval_features("supported query", chunks).retrieval_confidence

    assert confidence.result_count == len(chunks)
    assert confidence.top_score == expected_top_score
    assert confidence.score_gap == expected_gap


def test_probe_runs_dense_top_two_once_and_keeps_evidence_for_route_reuse():
    calls = []
    clock_values = iter([10.0, 10.025])
    chunks = [make_chunk("chunk-1", 0.88, 1), make_chunk("chunk-2", 0.8, 2)]

    def retrieve(*, query, top_k, timings):
        calls.append((query, top_k, timings))
        timings.update({"embedding_ms": 12.0, "dense_ms": 3.0})
        return chunks

    result = run_initial_retrieval_probe(
        "  What is dense retrieval?  ",
        retrieve,
        clock=lambda: next(clock_values),
    )

    assert len(calls) == 1
    assert calls[0][0] == "What is dense retrieval?"
    assert calls[0][1] == 2
    assert result.query == "What is dense retrieval?"
    assert result.chunks == tuple(chunks)
    assert result.features.retrieval_confidence.top_score == 0.88
    assert result.features.retrieval_confidence.score_gap == pytest.approx(0.08)
    assert result.timings.total_ms == pytest.approx(25.0)
    assert result.timings.embedding_ms == 12.0
    assert result.timings.dense_ms == 3.0


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        (
            [make_chunk("chunk-1", 0.9, 1), make_chunk("chunk-2", 0.8, 2), make_chunk("chunk-3", 0.7, 3)],
            "at most 2",
        ),
        ([make_chunk("chunk-1", 0.9, 2)], "contiguous and one-based"),
        ([make_chunk("chunk-1", 0.9, 1), make_chunk("chunk-1", 0.8, 2)], "duplicate chunk IDs"),
        ([make_chunk("chunk-1", 0.7, 1), make_chunk("chunk-2", 0.8, 2)], "descending score"),
        ([make_chunk("chunk-1", math.inf, 1)], "non-finite score"),
    ],
)
def test_probe_rejects_malformed_or_unordered_dense_evidence(chunks, message):
    with pytest.raises(ValueError, match=message):
        build_initial_retrieval_features("valid query", chunks)


def test_probe_validates_query_and_dependencies_before_retrieval():
    calls = []

    with pytest.raises(ValueError, match="empty"):
        run_initial_retrieval_probe("   ", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="retrieve must be callable"):
        run_initial_retrieval_probe("query", None)
    with pytest.raises(ValueError, match="clock must be callable"):
        run_initial_retrieval_probe("query", lambda **kwargs: [], clock=None)

    assert calls == []


def test_structured_features_reject_unknown_fields_and_inconsistent_confidence():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RetrievalConfidenceFeatures(result_count=0, extra_feature=True)
    with pytest.raises(ValidationError, match="empty probe"):
        RetrievalConfidenceFeatures(result_count=0, top_score=0.9)
    with pytest.raises(ValidationError, match="one-result probe"):
        RetrievalConfidenceFeatures(result_count=1, top_score=0.9, score_gap=0.1)
