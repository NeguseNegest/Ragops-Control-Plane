import csv
import json

import pytest
import yaml
from pydantic import ValidationError
from test_router_evaluation import PROJECT_ROOT, _evaluation_fixture

from ragops.evaluation.router_tuning import (
    load_router_tuning_config,
    run_router_tuning,
    write_router_tuning_artifacts,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _replace_question(path, old_question, new_question):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row.get("question") == old_question:
            row["question"] = new_question
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _tuning_fixture(tmp_path):
    evaluation_path, _ = _evaluation_fixture(tmp_path)
    old_question = "Explain why retrieval differs between systems."
    new_question = "What retrieval mode applies?"
    for relative_path in ("data/eval/labels.jsonl", "data/eval/golden.jsonl"):
        _replace_question(tmp_path / relative_path, old_question, new_question)
    for relative_path in (
        "reports/evaluations/dense.json",
        "reports/evaluations/careful.json",
        "reports/evaluations/no_answer.json",
    ):
        path = tmp_path / relative_path
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["questions"]:
            if row["question"] == old_question:
                row["question"] = new_question
                if relative_path.endswith("no_answer.json"):
                    row["reason_code"] = "score_gap_below_careful_threshold"
        _write_json(path, payload)

    archived = yaml.safe_load((PROJECT_ROOT / "configs/routed_v0.1.0.yaml").read_text(encoding="utf-8"))
    (tmp_path / "configs/routed_v0.1.0.yaml").write_text(
        yaml.safe_dump(archived, sort_keys=False), encoding="utf-8"
    )
    evaluation = yaml.safe_load(evaluation_path.read_text(encoding="utf-8"))
    evaluation["version"] = "0.2.0"
    evaluation_path.write_text(yaml.safe_dump(evaluation, sort_keys=False), encoding="utf-8")
    tuning = {
        "schema_version": 1,
        "name": "router_stabilization",
        "version": "0.1.0",
        "status": "evaluated",
        "baseline_router_config": "configs/routed_v0.1.0.yaml",
        "target_router_config": "configs/routed.yaml",
        "router_evaluation_config": "configs/router_evaluation.yaml",
        "split": {
            "method": "sha256_question_id_order",
            "tuning_questions": 1,
            "validation_questions": 1,
        },
        "tuning": {
            "parameter": "thresholds.careful.score_gap_below",
            "candidates": [0.01, 0.03, 0.04],
            "expected_selected_value": 0.03,
        },
        "constraints": {
            "minimum_validation_hit_rate_delta": 0,
            "minimum_unsupported_refusal_accuracy": 1,
            "maximum_full_average_latency_fraction_of_always_careful": 1,
            "maximum_full_total_cost_fraction_of_always_careful": 1,
        },
        "selection": {
            "objective": "maximize_tuning_hit_rate",
            "tie_breakers": ["minimize_tuning_average_latency", "minimize_threshold"],
        },
        "output": {
            "json_path": "reports/evaluations/router_distribution.json",
            "csv_path": "reports/evaluations/router_distribution.csv",
            "markdown_path": "reports/router_stabilization.md",
        },
    }
    path = tmp_path / "configs/router_tuning.yaml"
    path.write_text(yaml.safe_dump(tuning, sort_keys=False), encoding="utf-8")
    return path, tuning


def test_router_tuning_selects_quality_gain_with_deterministic_tie_breaker(tmp_path):
    config_path, _ = _tuning_fixture(tmp_path)
    config = load_router_tuning_config(config_path, project_root=tmp_path)

    report = run_router_tuning(config)

    assert report["baseline_router_id"] == "rule_router@0.1.0"
    assert report["target_router_id"] == "rule_router@0.2.0"
    assert report["selection"]["selected_value"] == 0.03
    assert all(report["selection"]["selected_candidate"]["constraint_checks"].values())
    assert report["split"] == {
        "method": "sha256_question_id_order",
        "tuning_question_ids": ["q2"],
        "validation_question_ids": ["q1"],
    }
    assert report["comparison"]["hit_rate_at_5_delta"] == 0.5
    assert report["distribution"]["all"]["route_counts"] == {"CAREFUL": 1, "FAST": 1, "NO_ANSWER": 2}
    assert report["transitions"]["counts"] == {
        "FAST->FAST": 1,
        "NO_ANSWER->NO_ANSWER": 2,
        "STANDARD->CAREFUL": 1,
    }
    assert report["stability"]["no_answer_threshold_unchanged"]
    assert report["stability"]["fast_thresholds_unchanged"]
    assert set(report["sources"]) == {
        "baseline_router",
        "careful_report",
        "chunks",
        "dense_report",
        "golden",
        "labels",
        "model_costs",
        "no_answer_config",
        "no_answer_report",
        "router_evaluation_config",
        "target_router",
    }
    assert all(len(source["sha256"]) == 64 for source in report["sources"].values())


def test_router_tuning_rejects_stale_expected_selection_and_target_policy_drift(tmp_path):
    config_path, payload = _tuning_fixture(tmp_path)
    payload["tuning"]["expected_selected_value"] = 0.01
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_router_tuning_config(config_path, project_root=tmp_path)
    with pytest.raises(ValueError, match="differs from configured expectation"):
        run_router_tuning(config)

    payload["tuning"]["expected_selected_value"] = 0.03
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    target_path = tmp_path / "configs/routed.yaml"
    target = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    target["thresholds"]["fast"]["top_score_at_least"] = 0.73
    target_path.write_text(yaml.safe_dump(target, sort_keys=False), encoding="utf-8")
    config = load_router_tuning_config(config_path, project_root=tmp_path)
    with pytest.raises(ValueError, match="changes beyond the selected"):
        run_router_tuning(config)


def test_router_tuning_config_is_strict_and_requires_ordered_candidates(tmp_path):
    config_path, payload = _tuning_fixture(tmp_path)
    payload["tuning"]["candidates"] = [0.03, 0.01]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValidationError, match="strictly increasing"):
        load_router_tuning_config(config_path, project_root=tmp_path)


def test_router_tuning_artifacts_are_atomic_complete_and_refuse_overwrite(tmp_path):
    config_path, _ = _tuning_fixture(tmp_path)
    config = load_router_tuning_config(config_path, project_root=tmp_path)
    report = run_router_tuning(config)

    paths = write_router_tuning_artifacts(report, config)

    assert all(path.is_file() for path in paths)
    assert json.loads(paths[0].read_text(encoding="utf-8")) == report
    with paths[1].open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == 4
    assert sum(row["route_changed"] == "True" for row in rows) == 1
    assert "Selected `thresholds.careful.score_gap_below = 0.030`" in paths[2].read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_router_tuning_artifacts(report, config)
