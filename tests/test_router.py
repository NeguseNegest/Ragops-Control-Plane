import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ragops.pipeline_registry import load_pipeline_registry
from ragops.retrieval.dense import RetrievedChunk
from ragops.routing.config import (
    EXPECTED_DECISION_ORDER,
    RouterConfig,
    load_router_config,
    validate_router_calibration,
    validate_router_registry_references,
)
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
from ragops.routing.router import ROUTE_REASONS, RouterDecision, RuleBasedRouter
from scripts.probe_query import probe_report
from scripts.route_query import route_report


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


def make_features(
    *,
    top_score=0.8,
    score_gap=0.1,
    result_count=2,
    token_count=5,
    complexity_marker_count=0,
    clause_marker_count=0,
    long_token_ratio=0.0,
):
    return InitialRetrievalFeatures.model_validate(
        {
            "query_length": {"character_count": max(token_count, 1), "token_count": token_count},
            "lexical_complexity": {
                "unique_token_count": token_count,
                "unique_token_ratio": 1.0,
                "average_token_length": 5.0,
                "maximum_token_length": 8,
                "long_token_count": round(token_count * long_token_ratio),
                "long_token_ratio": long_token_ratio,
                "clause_marker_count": clause_marker_count,
                "complexity_marker_count": complexity_marker_count,
            },
            "retrieval_confidence": {
                "requested_top_k": 2,
                "result_count": result_count,
                "top_score": top_score,
                "score_gap": score_gap,
            },
        }
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

    router_config = load_router_config(Path("configs/routed.yaml"), project_root=Path.cwd())
    report = probe_report(result, router_config)
    assert report["router_policy"] == {
        "router_id": "rule_router@0.2.0",
        "status": "draft",
        "probe": {"pipeline_config": "dense_baseline", "top_k": 2},
    }
    assert report["route"] is None
    assert report["route_reason"] is None


def test_probe_depth_is_configurable_within_the_cheap_schema_v1_range():
    calls = []
    chunks = [make_chunk("chunk-1", 0.9, 1), make_chunk("chunk-2", 0.8, 2), make_chunk("chunk-3", 0.7, 3)]

    def retrieve(*, query, top_k, timings):
        calls.append((query, top_k))
        return chunks

    result = run_initial_retrieval_probe("configured probe", retrieve, top_k=3, clock=lambda: 1.0)

    assert calls == [("configured probe", 3)]
    assert result.features.retrieval_confidence.requested_top_k == 3
    assert result.features.retrieval_confidence.result_count == 3
    assert result.features.retrieval_confidence.score_gap == pytest.approx(0.1)
    assert result.chunks == tuple(chunks)


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
    with pytest.raises(ValueError, match="integer from 2 to 5"):
        run_initial_retrieval_probe("query", lambda **kwargs: [], top_k=1)

    assert calls == []


def test_structured_features_reject_unknown_fields_and_inconsistent_confidence():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RetrievalConfidenceFeatures(result_count=0, extra_feature=True)
    with pytest.raises(ValidationError, match="empty probe"):
        RetrievalConfidenceFeatures(result_count=0, top_score=0.9)
    with pytest.raises(ValidationError, match="one-result probe"):
        RetrievalConfidenceFeatures(result_count=1, top_score=0.9, score_gap=0.1)
    with pytest.raises(ValidationError, match="cannot exceed requested_top_k"):
        RetrievalConfidenceFeatures(requested_top_k=2, result_count=3, top_score=0.9, score_gap=0.1)


def test_checked_in_router_config_defines_all_routes_features_and_threshold_bands():
    config = load_router_config(Path("configs/routed.yaml"), project_root=Path.cwd())

    assert config.name == "rule_router"
    assert config.version == "0.2.0"
    assert config.status == "draft"
    assert config.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert config.decision_order == EXPECTED_DECISION_ORDER
    assert config.probe.pipeline_config == "dense_baseline"
    assert config.probe.top_k == 2
    assert config.thresholds.no_answer.top_score_below == 0.531
    assert config.thresholds.careful.top_score_below == 0.56
    assert config.thresholds.careful.on_missing_score_gap
    assert config.thresholds.careful.score_gap_below == 0.03
    assert config.thresholds.fast.top_score_at_least == 0.72
    assert config.routes.fast.reuse_probe
    assert config.routes.fast.maximum_top_k == config.probe.top_k
    assert config.routes.standard.pipeline_config == "dense_baseline"
    assert config.routes.careful.pipeline_config == "hybrid_rrf_cross_encoder"
    assert not config.routes.no_answer.generate_answer
    assert config.routes.no_answer.response_mode == "refusal"
    assert config.calibration.question_count == 45
    assert config.calibration.unsupported_source == Path("data/eval/no_answer_queries.jsonl")
    assert config.calibration.unsupported_question_count == 12


def test_router_config_matches_registry_lifecycle_and_calibration_evidence():
    config = load_router_config(Path("configs/routed.yaml"), project_root=Path.cwd())
    registry = load_pipeline_registry(Path("reports/pipeline_registry.json"))

    assert validate_router_registry_references(config, registry) == {
        "FAST": "dense_baseline@1.0.0",
        "STANDARD": "dense_baseline@1.0.0",
        "CAREFUL": "hybrid_rrf_cross_encoder@1.0.0",
    }
    calibration = validate_router_calibration(config, project_root=Path.cwd())
    assert calibration["question_count"] == 45
    assert calibration["unsupported_question_count"] == 12
    assert calibration["unsupported_source"].endswith("data/eval/no_answer_queries.jsonl")
    assert calibration["top_score"] == {
        "minimum": 0.3018593,
        "median": 0.6597541,
        "maximum": 0.8521882,
    }
    assert calibration["score_gap"]["minimum"] == pytest.approx(0.0003204)
    assert calibration["score_gap"]["median"] == pytest.approx(0.0299468)
    assert calibration["score_gap"]["maximum"] == pytest.approx(0.1451546)


def test_router_config_rejects_overlapping_thresholds_and_unsafe_route_shapes():
    config = load_router_config(Path("configs/routed.yaml"), project_root=Path.cwd())

    payload = config.model_dump(mode="python")
    payload["thresholds"]["careful"]["top_score_below"] = 0.8
    with pytest.raises(ValidationError, match="Score thresholds must increase"):
        RouterConfig.model_validate(payload)

    payload = config.model_dump(mode="python")
    payload["decision_order"] = ["FAST", "CAREFUL", "NO_ANSWER", "STANDARD"]
    with pytest.raises(ValidationError, match="decision_order"):
        RouterConfig.model_validate(payload)

    payload = config.model_dump(mode="python")
    payload["routes"]["fast"]["maximum_top_k"] = 3
    with pytest.raises(ValidationError, match="cannot exceed probe.top_k"):
        RouterConfig.model_validate(payload)


def test_router_registry_guard_rejects_a_pipeline_status_not_allowed_by_route():
    config = load_router_config(Path("configs/routed.yaml"), project_root=Path.cwd())
    registry = load_pipeline_registry(Path("reports/pipeline_registry.json"))
    payload = config.model_dump(mode="python")
    payload["routes"]["careful"]["allowed_pipeline_statuses"] = ["approved"]

    with pytest.raises(ValueError, match="does not allow pipeline status 'evaluated'"):
        validate_router_registry_references(RouterConfig.model_validate(payload), registry)

    changed_aliases = registry.aliases.model_copy(update={"candidate": "dense_baseline@1.0.0"})
    changed_registry = registry.model_copy(update={"aliases": changed_aliases})
    with pytest.raises(ValueError, match="CAREFUL must reference the registry candidate alias"):
        validate_router_registry_references(config, changed_registry)


def test_router_config_loader_rejects_empty_unknown_and_invalid_yaml(tmp_path):
    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_router_config(empty_path)

    config = load_router_config(Path("configs/routed.yaml"), project_root=Path.cwd())
    unknown_path = tmp_path / "unknown.yaml"
    payload = config.model_dump(mode="json")
    payload["unknown"] = True
    unknown_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_router_config(unknown_path)

    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("routes: [", encoding="utf-8")
    with pytest.raises(ValueError, match="contains invalid YAML"):
        load_router_config(invalid_path)


def test_router_selects_fast_with_stable_reason_and_probe_reuse_intent():
    router = RuleBasedRouter(load_router_config(Path("configs/routed.yaml"), project_root=Path.cwd()))

    decision = router.select(make_features(top_score=0.72, score_gap=0.05, token_count=12, long_token_ratio=0.3))

    assert decision.route == "FAST"
    assert decision.reason_code == "fast_conditions_satisfied"
    assert decision.reason == ROUTE_REASONS[decision.reason_code]
    assert decision.matched_reason_codes == ("fast_conditions_satisfied",)
    assert decision.pipeline_config == "dense_baseline"
    assert decision.maximum_top_k == 2
    assert decision.reuse_probe
    assert decision.generate_answer
    assert decision.response_mode is None


def test_router_selects_standard_at_non_careful_boundaries_when_fast_is_not_fully_satisfied():
    router = RuleBasedRouter(load_router_config(Path("configs/routed.yaml"), project_root=Path.cwd()))

    decision = router.select(make_features(top_score=0.56, score_gap=0.03, token_count=13, long_token_ratio=0.3))

    assert decision.route == "STANDARD"
    assert decision.reason_code == "standard_fallback"
    assert decision.pipeline_config == "dense_baseline"
    assert decision.maximum_top_k == 10
    assert not decision.reuse_probe


@pytest.mark.parametrize(
    ("feature_overrides", "reason_code"),
    [
        ({"result_count": 1, "top_score": 0.6, "score_gap": None}, "missing_score_gap"),
        ({"top_score": 0.559}, "top_score_below_careful_threshold"),
        ({"score_gap": 0.029}, "score_gap_below_careful_threshold"),
        ({"token_count": 21}, "token_count_above_careful_threshold"),
        ({"complexity_marker_count": 1}, "complexity_marker_count_at_least_careful_threshold"),
        ({"clause_marker_count": 3}, "clause_marker_count_at_least_careful_threshold"),
        ({"long_token_ratio": 0.4}, "long_token_ratio_at_least_careful_threshold"),
    ],
)
def test_each_careful_condition_is_active_with_documented_inequality(feature_overrides, reason_code):
    router = RuleBasedRouter(load_router_config(Path("configs/routed.yaml"), project_root=Path.cwd()))

    decision = router.select(make_features(**feature_overrides))

    assert decision.route == "CAREFUL"
    assert decision.reason_code == reason_code
    assert decision.matched_reason_codes == (reason_code,)
    assert decision.pipeline_config == "hybrid_rrf_cross_encoder"
    assert decision.maximum_top_k == 5
    assert not decision.reuse_probe
    assert decision.generate_answer


def test_careful_returns_all_matches_in_stable_priority_order():
    router = RuleBasedRouter(load_router_config(Path("configs/routed.yaml"), project_root=Path.cwd()))
    features = make_features(
        top_score=0.55,
        score_gap=0.005,
        token_count=21,
        complexity_marker_count=2,
        clause_marker_count=3,
        long_token_ratio=0.5,
    )

    first = router.select(features)
    second = router.select(features.model_dump(mode="python"))

    assert first == second
    assert first.matched_reason_codes == (
        "top_score_below_careful_threshold",
        "score_gap_below_careful_threshold",
        "token_count_above_careful_threshold",
        "complexity_marker_count_at_least_careful_threshold",
        "clause_marker_count_at_least_careful_threshold",
        "long_token_ratio_at_least_careful_threshold",
    )
    assert first.reason_code == first.matched_reason_codes[0]


@pytest.mark.parametrize(
    ("features", "reason_code"),
    [
        (make_features(result_count=0, top_score=None, score_gap=None), "empty_probe"),
        (make_features(top_score=0.53), "top_score_below_no_answer_threshold"),
    ],
)
def test_no_answer_precedes_other_routes_and_has_no_execution_pipeline(features, reason_code):
    router = RuleBasedRouter(load_router_config(Path("configs/routed.yaml"), project_root=Path.cwd()))

    decision = router.select(features)

    assert decision.route == "NO_ANSWER"
    assert decision.reason_code == reason_code
    assert decision.pipeline_config is None
    assert decision.maximum_top_k == 0
    assert not decision.reuse_probe
    assert not decision.generate_answer
    assert decision.response_mode == "refusal"


def test_no_answer_score_floor_is_strict_and_does_not_capture_equal_score():
    router = RuleBasedRouter(load_router_config(Path("configs/routed.yaml"), project_root=Path.cwd()))

    decision = router.select(make_features(top_score=0.531, score_gap=0.02))

    assert decision.route == "CAREFUL"
    assert decision.reason_code == "top_score_below_careful_threshold"


def test_no_answer_precedence_over_careful_complexity():
    router = RuleBasedRouter(load_router_config(Path("configs/routed.yaml"), project_root=Path.cwd()))

    decision = router.select(make_features(top_score=0.53, complexity_marker_count=1))

    assert decision.route == "NO_ANSWER"
    assert decision.matched_reason_codes == ("top_score_below_no_answer_threshold",)


def test_router_decision_rejects_reason_or_execution_shape_drift():
    router = RuleBasedRouter(load_router_config(Path("configs/routed.yaml"), project_root=Path.cwd()))
    payload = router.select(make_features()).model_dump(mode="python")
    payload["reason"] = "Changed prose"

    with pytest.raises(ValidationError, match="stable text"):
        RouterDecision.model_validate(payload)

    payload = router.select(make_features()).model_dump(mode="python")
    payload["generate_answer"] = False
    with pytest.raises(ValidationError, match="Retrieval routes"):
        RouterDecision.model_validate(payload)

    payload = router.select(make_features()).model_dump(mode="python")
    payload.update(
        {
            "reason_code": "standard_fallback",
            "reason": ROUTE_REASONS["standard_fallback"],
            "matched_reason_codes": ["standard_fallback"],
        }
    )
    with pytest.raises(ValidationError, match="selected route"):
        RouterDecision.model_validate(payload)


def test_route_report_returns_route_reason_features_and_execution_intent_without_document_text():
    config = load_router_config(Path("configs/routed.yaml"), project_root=Path.cwd())
    chunks = [make_chunk("chunk-1", 0.9, 1), make_chunk("chunk-2", 0.8, 2)]
    probe = run_initial_retrieval_probe("FastAPI basics", lambda **kwargs: chunks, clock=lambda: 1.0)

    report = route_report(RuleBasedRouter(config).select_probe(probe))

    assert report["route"] == "FAST"
    assert report["reason_code"] == "fast_conditions_satisfied"
    assert report["execution_intent"]["reuse_probe"] is True
    assert report["probe"]["chunk_ids"] == ["chunk-1", "chunk-2"]
    assert "Text for" not in str(report)
