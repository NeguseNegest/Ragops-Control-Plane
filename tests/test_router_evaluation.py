import csv
import json
from pathlib import Path

import pytest
import yaml

from ragops.evaluation.router_comparison import (
    load_router_evaluation_config,
    run_router_evaluation,
    validate_router_evaluation_inputs,
    write_router_evaluation_artifacts,
)
from ragops.generation.no_answer import NO_ANSWER_RESPONSE

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _retrieval_row(question_id, question, relevant, retrieved, scores, latency):
    return {
        "question_id": question_id,
        "question": question,
        "expected_source": f"docs/{question_id}.md",
        "relevant_chunk_ids": [relevant],
        "retrieved_chunk_ids": retrieved,
        "retrieved_scores": scores,
        "latency_ms": latency,
    }


def _evaluation_fixture(tmp_path):
    root = tmp_path
    (root / "configs").mkdir()
    (root / "reports/evaluations").mkdir(parents=True)
    (root / "data/eval").mkdir(parents=True)
    (root / "data/processed").mkdir(parents=True)
    router_payload = yaml.safe_load((PROJECT_ROOT / "configs/routed.yaml").read_text(encoding="utf-8"))
    (root / "configs/routed.yaml").write_text(yaml.safe_dump(router_payload, sort_keys=False), encoding="utf-8")

    supported = [
        ("q1", "What is FastAPI?", "d1", "A Python API framework."),
        ("q2", "Explain why retrieval differs between systems.", "c2", "Retrieval systems use different ranking signals."),
    ]
    unsupported = [
        ("u1", "How do I configure an unrelated framework migration?", "calibration", 0.40, 0.02),
        ("u2", "What dosage should I take for an unrelated medicine?", "evaluation", 0.45, 0.01),
    ]
    labels = [
        {
            "question_id": question_id,
            "question": question,
            "relevant_chunk_ids": [relevant],
            "expected_source": f"docs/{question_id}.md",
            "metadata": {"label_method": "manual", "review_status": "verified", "reviewed_by": "test"},
        }
        for question_id, question, relevant, _ in supported
    ]
    golden = [
        {
            "id": question_id,
            "question": question,
            "expected_answer": answer,
            "expected_source": f"docs/{question_id}.md",
            "query_type": "supported",
            "difficulty": "easy",
        }
        for question_id, question, _, answer in supported
    ]
    golden.append(
        {
            "id": "u1",
            "question": unsupported[0][1],
            "expected_answer": NO_ANSWER_RESPONSE,
            "expected_source": None,
            "query_type": "unsupported",
            "difficulty": "easy",
        }
    )
    _write_jsonl(root / "data/eval/labels.jsonl", labels)
    _write_jsonl(root / "data/eval/golden.jsonl", golden)
    unsupported_rows = [
        {
            "id": question_id,
            "question": question,
            "split": split,
            "category": "near_domain_technology" if question_id == "u1" else "high_stakes_out_of_scope",
            "expected_behavior": "refusal",
            "provenance": "test",
            "review_status": "verified",
            "reviewed_by": "test",
        }
        for question_id, question, split, _, _ in unsupported
    ]
    _write_jsonl(root / "data/eval/unsupported.jsonl", unsupported_rows)

    dense_rows = [
        _retrieval_row("q1", supported[0][1], "d1", [f"d{i}" for i in range(1, 11)], [0.80, 0.70, *[0.69 - i / 100 for i in range(8)]], 10),
        _retrieval_row("q2", supported[1][1], "c2", [f"d{i}" for i in range(11, 21)], [0.60, 0.58, *[0.57 - i / 100 for i in range(8)]], 20),
    ]
    careful_rows = [
        _retrieval_row("q1", supported[0][1], "d1", ["c1", "d1", "c3", "c4", "c5"], [5, 4, 3, 2, 1], 100),
        _retrieval_row("q2", supported[1][1], "c2", ["c2", "c6", "c7", "c8", "c9"], [5, 4, 3, 2, 1], 200),
    ]
    dense_report = {
        "schema_version": 1,
        "run_name": "dense_baseline",
        "configuration": {"retriever": {"top_k": 10}},
        "metrics": {},
        "latency_ms": {},
        "questions": dense_rows,
    }
    careful_report = {
        "schema_version": 1,
        "run_name": "hybrid_rrf_cross_encoder",
        "configuration": {"reranker": {"top_k": 5}},
        "metrics": {},
        "latency_ms": {},
        "questions": careful_rows,
    }
    _write_json(root / "reports/evaluations/dense.json", dense_report)
    _write_json(root / "reports/evaluations/careful.json", careful_report)

    no_answer_questions = [
        {
            "question_id": "u1",
            "question": unsupported[0][1],
            "query_type": "unsupported",
            "top_score": unsupported[0][3],
            "score_gap": unsupported[0][4],
            "route": "NO_ANSWER",
            "reason_code": "top_score_below_no_answer_threshold",
            "refused": True,
        },
        {
            "question_id": "u2",
            "question": unsupported[1][1],
            "query_type": "unsupported",
            "top_score": unsupported[1][3],
            "score_gap": unsupported[1][4],
            "route": "NO_ANSWER",
            "reason_code": "top_score_below_no_answer_threshold",
            "refused": True,
        },
        {
            "question_id": "q1",
            "question": supported[0][1],
            "query_type": "supported",
            "top_score": 0.80,
            "score_gap": 0.10,
            "route": "FAST",
            "reason_code": "fast_conditions_satisfied",
            "refused": False,
        },
        {
            "question_id": "q2",
            "question": supported[1][1],
            "query_type": "supported",
            "top_score": 0.60,
            "score_gap": 0.02,
            "route": "CAREFUL",
            "reason_code": "complexity_marker_count_at_least_careful_threshold",
            "refused": False,
        },
    ]
    no_answer_report = {
        "schema_version": 1,
        "run_name": "no_answer",
        "router_id": "rule_router@0.1.0",
        "threshold": {"configured": 0.531},
        "questions": no_answer_questions,
    }
    _write_json(root / "reports/evaluations/no_answer.json", no_answer_report)

    no_answer_config = {
        "schema_version": 1,
        "name": "no_answer",
        "version": "0.1.0",
        "status": "draft",
        "router_config": "configs/routed.yaml",
        "dataset": {
            "unsupported_path": "data/eval/unsupported.jsonl",
            "golden_path": "data/eval/golden.jsonl",
            "supported_report_path": "reports/evaluations/dense.json",
            "minimum_calibration_unsupported": 1,
            "minimum_evaluation_unsupported": 1,
            "expected_supported_questions": 2,
        },
        "threshold_selection": {"method": "calibration_max_plus_margin_ceiling", "score_margin": 0.0005, "decimal_places": 3},
        "acceptance": {
            "minimum_unsupported_refusal_accuracy": 1,
            "minimum_evaluation_refusal_accuracy": 1,
            "minimum_supported_answer_rate": 0,
            "minimum_refusal_precision": 0,
        },
        "refusal": {"prompt_version": "no_answer_v1", "answer": NO_ANSWER_RESPONSE},
        "output": {"json_path": "reports/evaluations/no_answer.json", "csv_path": "reports/evaluations/no_answer.csv"},
    }
    (root / "configs/no_answer.yaml").write_text(yaml.safe_dump(no_answer_config, sort_keys=False), encoding="utf-8")

    chunk_ids = {chunk_id for row in dense_rows + careful_rows for chunk_id in row["retrieved_chunk_ids"]}
    chunks = [
        {
            "chunk_id": chunk_id,
            "document_id": f"doc-{chunk_id}",
            "text": f"Documentation evidence for {chunk_id}.",
            "token_count": 5,
            "chunk_hash": f"hash-{chunk_id}",
            "metadata": {"relative_path": f"docs/{chunk_id}.md"},
        }
        for chunk_id in sorted(chunk_ids)
    ]
    _write_jsonl(root / "data/processed/chunks.jsonl", chunks)
    costs = {
        "schema_version": 1,
        "name": "generation_model_costs",
        "version": "1.0.0",
        "status": "approved",
        "currency": "USD",
        "token_estimator": "utf8_bytes_div4_ceiling_v1",
        "models": [
            {
                "provider": "openai",
                "model": "gpt-5-nano",
                "input_usd_per_million_tokens": 0.05,
                "output_usd_per_million_tokens": 0.4,
                "source_url": "https://example.com/pricing",
                "source_checked_at": "2026-08-16",
                "notes": "Test rate.",
            }
        ],
    }
    (root / "configs/model_costs.yaml").write_text(yaml.safe_dump(costs, sort_keys=False), encoding="utf-8")
    evaluation = {
        "schema_version": 1,
        "name": "router_comparison",
        "version": "0.1.0",
        "status": "evaluated",
        "mode": "artifact_replay",
        "router_config": "configs/routed.yaml",
        "inputs": {
            "labels_path": "data/eval/labels.jsonl",
            "golden_path": "data/eval/golden.jsonl",
            "chunks_path": "data/processed/chunks.jsonl",
            "dense_report_path": "reports/evaluations/dense.json",
            "careful_report_path": "reports/evaluations/careful.json",
            "no_answer_config_path": "configs/no_answer.yaml",
            "no_answer_report_path": "reports/evaluations/no_answer.json",
            "expected_supported_questions": 2,
            "expected_unsupported_questions": 2,
        },
        "quality": {"retrieval_cutoff": 5, "combined_proxy_definition": "supported_hit_or_correct_unsupported_refusal"},
        "latency": {"method": "measured_artifact_serial_replay", "include_cold_start": True, "probe_proxy": "dense_top_10_measurement"},
        "cost_projection": {
            "provider": "openai",
            "model": "gpt-5-nano",
            "model_cost_config": "configs/model_costs.yaml",
            "token_basis": "exact_prompt_plus_verified_reference_answer",
        },
        "output": {
            "json_path": "reports/evaluations/router_comparison.json",
            "csv_path": "reports/evaluations/router_comparison.csv",
            "markdown_path": "reports/router_comparison.md",
        },
    }
    config_path = root / "configs/router_evaluation.yaml"
    config_path.write_text(yaml.safe_dump(evaluation, sort_keys=False), encoding="utf-8")
    return config_path, evaluation


def test_router_evaluation_config_is_strict_and_resolves_paths(tmp_path):
    config_path, payload = _evaluation_fixture(tmp_path)
    config = load_router_evaluation_config(config_path, project_root=tmp_path)
    assert config.inputs.labels_path == (tmp_path / "data/eval/labels.jsonl").resolve()
    assert config.output.markdown_path == (tmp_path / "reports/router_comparison.md").resolve()

    payload["unexpected"] = True
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        load_router_evaluation_config(config_path, project_root=tmp_path)


def test_router_evaluation_validates_all_paired_inputs(tmp_path):
    config_path, _ = _evaluation_fixture(tmp_path)
    config = load_router_evaluation_config(config_path, project_root=tmp_path)
    inputs = validate_router_evaluation_inputs(config)
    assert len(inputs["labels"]) == 2
    assert len(inputs["unsupported_examples"]) == 2
    assert inputs["router_config"].routes.fast.maximum_top_k == 2
    assert inputs["cost_table"].identity == "generation_model_costs@1.0.0"


def test_router_evaluation_rejects_decision_drift(tmp_path):
    config_path, _ = _evaluation_fixture(tmp_path)
    report_path = tmp_path / "reports/evaluations/no_answer.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["questions"][2]["route"] = "STANDARD"
    _write_json(report_path, report)
    config = load_router_evaluation_config(config_path, project_root=tmp_path)
    with pytest.raises(ValueError, match="decision drifted"):
        validate_router_evaluation_inputs(config)


def test_router_evaluation_compares_quality_latency_cost_and_refusal(tmp_path):
    config_path, _ = _evaluation_fixture(tmp_path)
    config = load_router_evaluation_config(config_path, project_root=tmp_path)
    report = run_router_evaluation(config)

    assert report["scope"] == {
        "supported_operational_questions": 2,
        "unsupported_policy_questions": 2,
        "policy_questions": 4,
        "operational_metrics_exclude_unsupported": True,
    }
    assert report["strategies"]["always_fast"]["retrieval_quality_supported"]["hit_rate_at_5"] == 0.5
    assert report["strategies"]["always_careful"]["retrieval_quality_supported"]["hit_rate_at_5"] == 1.0
    assert report["strategies"]["routed"]["retrieval_quality_supported"]["hit_rate_at_5"] == 1.0
    assert report["strategies"]["routed"]["route_counts_supported"] == {"CAREFUL": 1, "FAST": 1}
    assert report["strategies"]["routed"]["latency_ms_supported"]["average"] == 115
    assert report["strategies"]["always_careful"]["latency_ms_supported"]["average"] == 150
    assert report["strategies"]["routed"]["refusal_quality"]["unsupported_refusal_accuracy"] == 1
    assert report["strategies"]["always_fast"]["refusal_quality"]["unsupported_refusal_accuracy"] == 0
    assert report["strategies"]["routed"]["generation_cost_projection_supported"]["amount_usd"]["total"] > 0
    assert all(
        row["strategies"]["routed"]["cost"]["token_source"] == "heuristic_estimate" for row in report["questions"]
    )
    assert run_router_evaluation(config) == report


def test_router_evaluation_rejects_missing_prompt_chunk(tmp_path):
    config_path, _ = _evaluation_fixture(tmp_path)
    chunks_path = tmp_path / "data/processed/chunks.jsonl"
    rows = chunks_path.read_text(encoding="utf-8").splitlines()
    chunks_path.write_text("\n".join(rows[1:]) + "\n", encoding="utf-8")
    config = load_router_evaluation_config(config_path, project_root=tmp_path)
    with pytest.raises(ValueError, match="missing .* evaluated chunks"):
        validate_router_evaluation_inputs(config)


def test_routed_supported_refusal_has_zero_provider_cost(tmp_path):
    config_path, _ = _evaluation_fixture(tmp_path)
    dense_path = tmp_path / "reports/evaluations/dense.json"
    dense = json.loads(dense_path.read_text(encoding="utf-8"))
    dense["questions"][1]["retrieved_scores"][:2] = [0.50, 0.48]
    _write_json(dense_path, dense)
    no_answer_path = tmp_path / "reports/evaluations/no_answer.json"
    no_answer = json.loads(no_answer_path.read_text(encoding="utf-8"))
    supported = next(row for row in no_answer["questions"] if row["question_id"] == "q2")
    supported.update(
        {
            "top_score": 0.50,
            "score_gap": 0.02,
            "route": "NO_ANSWER",
            "reason_code": "top_score_below_no_answer_threshold",
            "refused": True,
        }
    )
    _write_json(no_answer_path, no_answer)

    config = load_router_evaluation_config(config_path, project_root=tmp_path)
    report = run_router_evaluation(config)
    result = next(row for row in report["questions"] if row["question_id"] == "q2")["strategies"]["routed"]
    assert result["retrieved_chunk_ids"] == []
    assert result["cost"]["status"] == "zero_cost"
    assert result["cost"]["amount_usd"] == 0
    assert report["strategies"]["routed"]["generation_cost_projection_supported"]["status_counts"] == {
        "estimated": 1,
        "zero_cost": 1,
    }


def test_router_evaluation_writes_atomic_artifacts_and_refuses_overwrite(tmp_path):
    config_path, _ = _evaluation_fixture(tmp_path)
    config = load_router_evaluation_config(config_path, project_root=tmp_path)
    report = run_router_evaluation(config)
    paths = write_router_evaluation_artifacts(report, config)
    assert all(path.is_file() for path in paths)
    assert json.loads(paths[0].read_text(encoding="utf-8")) == report
    with paths[1].open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == 6
    assert {row["strategy"] for row in rows} == {"always_fast", "always_careful", "routed"}
    assert "keep_router_draft" in paths[2].read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_router_evaluation_artifacts(report, config)
