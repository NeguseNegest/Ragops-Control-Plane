import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ragops.evaluation.failure_analysis import (
    FailureAnalysisConfig,
    build_failure_analysis,
    load_failure_analysis_config,
    validate_failure_analysis_outputs,
    write_failure_analysis_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_failure_analysis_config(PROJECT_ROOT / "configs/failure_analysis.yaml", project_root=PROJECT_ROOT)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_checked_in_failure_analysis_has_real_evidence_and_regressions():
    config = _config()
    analysis = validate_failure_analysis_outputs(config)

    assert analysis["status"] == "complete"
    assert analysis["summary"]["failure_count"] == 15
    assert analysis["summary"]["regression_case_count"] == 14
    assert analysis["summary"]["category_counts"] == {
        "bad_dense_retrieval": 1,
        "high_latency_query": 1,
        "hybrid_fusion_failure": 2,
        "incorrect_refusal": 2,
        "lexical_retrieval_miss": 1,
        "missing_or_weak_citation": 1,
        "reranker_regression": 3,
        "router_mistake": 3,
        "unexpected_generation_behavior": 1,
    }
    cases = {case["id"]: case for case in analysis["cases"]}
    assert cases["day48-006"]["evidence"] == {
        "kind": "retrieval_rank",
        "pipeline": "reranked",
        "rank_at_5": None,
        "baseline_pipeline": "pre_rerank",
        "baseline_rank_at_5": 1,
    }
    assert cases["day48-013"]["regression_test"] is False
    assert cases["day48-014"]["evidence"]["faithfulness"] == 3
    assert len(analysis["source_artifacts"]) == 11


def test_retrieval_rank_drift_is_rejected(tmp_path):
    config = _config()
    dense = _json(config.sources.dense_report_path)
    row = next(row for row in dense["questions"] if row["question_id"] == "sqa-43e609692540e39f")
    row["retrieved_chunk_ids"][0] = row["relevant_chunk_ids"][0]
    changed_path = tmp_path / "dense.json"
    _write_json(changed_path, dense)
    sources = config.sources.model_copy(update={"dense_report_path": changed_path})

    with pytest.raises(ValueError, match="Retrieval evidence drift for day48-001"):
        build_failure_analysis(config.model_copy(update={"sources": sources}))


def test_route_decision_drift_is_rejected(tmp_path):
    config = _config()
    routed = _json(config.sources.routed_report_path)
    row = next(row for row in routed["policy_questions"] if row["question_id"] == "day46-adv-002")
    row["route"] = "NO_ANSWER"
    changed_path = tmp_path / "routed.json"
    _write_json(changed_path, routed)
    sources = config.sources.model_copy(update={"routed_report_path": changed_path})

    with pytest.raises(ValueError, match="Route evidence drift for day48-010"):
        build_failure_analysis(config.model_copy(update={"sources": sources}))


def test_judgment_drift_is_rejected(tmp_path):
    config = _config()
    rows = [json.loads(line) for line in config.sources.hybrid_judgments_path.read_text(encoding="utf-8").splitlines()]
    row = next(row for row in rows if row["question_id"] == "gqa-001")
    row["automatic_judgment"]["faithfulness"]["score"] = 5
    changed_path = tmp_path / "hybrid_judgments.jsonl"
    changed_path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    sources = config.sources.model_copy(update={"hybrid_judgments_path": changed_path})

    with pytest.raises(ValueError, match="Judgment evidence drift for day48-014"):
        build_failure_analysis(config.model_copy(update={"sources": sources}))


def test_output_validation_rejects_manual_edit(tmp_path):
    config = _config()
    outputs = config.outputs.model_copy(
        update={
            "report_path": tmp_path / "failure_analysis.md",
            "regression_cases_path": tmp_path / "regression_cases.jsonl",
        }
    )
    config = config.model_copy(update={"outputs": outputs})
    analysis = build_failure_analysis(config)
    write_failure_analysis_outputs(config, analysis=analysis)
    validate_failure_analysis_outputs(config, analysis=analysis)

    outputs.report_path.write_text("manually changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale or manually edited"):
        validate_failure_analysis_outputs(config, analysis=analysis)


def test_config_rejects_duplicate_targets_and_too_few_cases():
    payload = _config().model_dump(mode="json")
    payload["cases"][1]["question_id"] = payload["cases"][0]["question_id"]
    payload["cases"][1]["category"] = payload["cases"][0]["category"]
    with pytest.raises(ValidationError, match="Question/category failure targets must be unique"):
        FailureAnalysisConfig.model_validate(payload)

    payload = _config().model_dump(mode="json")
    payload["cases"] = payload["cases"][:9]
    with pytest.raises(ValidationError, match="outside the accepted 10-20 range"):
        FailureAnalysisConfig.model_validate(payload)
