import hashlib
from pathlib import Path

import pytest
import yaml

from ragops.evaluation.gate import (
    GateObservation,
    build_gate_report,
    execute_evaluation_gate,
    gate_exit_code,
    load_evaluation_gate,
    referenced_citation_chunk_ids,
    render_gate_summary,
    run_gate_cases,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE_CONFIG_PATH = PROJECT_ROOT / "configs/eval_gate.yaml"


class StepClock:
    def __init__(self, step_seconds):
        self.value = 0.0
        self.step_seconds = step_seconds

    def __call__(self):
        current = self.value
        self.value += self.step_seconds
        return current


def successful_observation(case):
    if case.expected_behavior == "refusal":
        return GateObservation(observed_behavior="refusal", route="NO_ANSWER")
    relevant = case.relevant_chunk_ids[0]
    return GateObservation(
        observed_behavior="answer",
        route="FAST",
        retrieved_chunk_ids=(relevant, "non-relevant"),
        cited_chunk_ids=(relevant,),
        answer="A grounded answer. [1]",
    )


def check_by_id(report, check_id):
    return next(check for check in report["checks"] if check["id"] == check_id)


def test_load_gate_cross_validates_hash_pinned_inputs():
    loaded = load_evaluation_gate(GATE_CONFIG_PATH, project_root=PROJECT_ROOT)

    assert loaded.config.name == "compact_evaluation_gate"
    assert loaded.candidate_config.name == "dense_baseline"
    assert loaded.candidate_config.version == "1.0.0"
    assert loaded.router_config.version == "0.2.0"
    assert len(loaded.records) == 4
    assert [case.query_type for case in loaded.cases].count("supported") == 3
    assert [case.query_type for case in loaded.cases].count("unsupported") == 2


def test_load_gate_rejects_candidate_checksum_drift(tmp_path):
    payload = yaml.safe_load(GATE_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["candidate"]["config_sha256"] = "0" * 64
    config_path = tmp_path / "eval_gate.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Candidate config checksum mismatch"):
        load_evaluation_gate(config_path, project_root=PROJECT_ROOT)


def test_load_gate_rejects_case_vector_dimension_drift(tmp_path):
    case_path = tmp_path / "cases.jsonl"
    source = (PROJECT_ROOT / "tests/fixtures/eval_gate_cases.jsonl").read_text(encoding="utf-8")
    case_path.write_text(source.replace('"vector":[1.0,0.0,0.0]', '"vector":[1.0,0.0]', 1), encoding="utf-8")
    payload = yaml.safe_load(GATE_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["dataset"]["cases_path"] = str(case_path)
    payload["dataset"]["cases_sha256"] = hashlib.sha256(case_path.read_bytes()).hexdigest()
    config_path = tmp_path / "eval_gate.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="vector must match corpus.vector_size"):
        load_evaluation_gate(config_path, project_root=PROJECT_ROOT)


def test_real_compact_candidate_passes_every_gate_check():
    loaded = load_evaluation_gate(GATE_CONFIG_PATH, project_root=PROJECT_ROOT)

    report = execute_evaluation_gate(loaded)

    assert report["status"] == "pass"
    assert report["candidate_pipeline_id"] == "dense_baseline@1.0.0"
    assert report["case_count"] == 5
    assert report["metrics"]["retrieval"]["recall_at_k"] == 1.0
    assert report["metrics"]["retrieval"]["mrr"] == 1.0
    assert report["metrics"]["generation"]["citation_coverage"] == 1.0
    assert report["metrics"]["generation"]["citation_precision"] == 1.0
    assert report["metrics"]["generation"]["faithfulness"] is None
    assert report["metrics"]["refusal"]["correctness"] == 1.0
    assert report["metrics"]["errors"]["count"] == 0
    assert all(check["passed"] for check in report["checks"])
    assert [case["route"] for case in report["cases"][-2:]] == ["NO_ANSWER", "NO_ANSWER"]


def test_deliberately_degraded_dense_candidate_fails_gate():
    loaded = load_evaluation_gate(GATE_CONFIG_PATH, project_root=PROJECT_ROOT)
    supported = [case for case in loaded.cases if case.query_type == "supported"]
    degraded_vectors = {case.query: case.vector for case in loaded.cases}
    for index, case in enumerate(supported):
        degraded_vectors[case.query] = supported[(index + 1) % len(supported)].vector

    report = execute_evaluation_gate(loaded, query_vectors=degraded_vectors)

    assert report["status"] == "fail"
    assert report["metrics"]["retrieval"]["recall_at_k"] < 1.0
    assert check_by_id(report, "retrieval.minimum_recall_at_k")["passed"] is False
    assert check_by_id(report, "retrieval.maximum_recall_regression")["passed"] is False
    assert check_by_id(report, "generation.minimum_citation_coverage")["passed"] is False
    assert gate_exit_code(report) == 1


def test_slow_candidate_fails_p95_latency_without_masking_quality():
    loaded = load_evaluation_gate(GATE_CONFIG_PATH, project_root=PROJECT_ROOT)
    results = run_gate_cases(loaded.cases, successful_observation, clock=StepClock(0.3))

    report = build_gate_report(loaded, results)

    assert report["metrics"]["retrieval"]["recall_at_k"] == 1.0
    assert report["metrics"]["latency_ms"]["p95"] == pytest.approx(300.0)
    assert check_by_id(report, "latency.maximum_p95_ms")["passed"] is False
    assert report["status"] == "fail"


def test_runtime_error_is_counted_and_fails_gate():
    loaded = load_evaluation_gate(GATE_CONFIG_PATH, project_root=PROJECT_ROOT)
    failed_case = loaded.cases[0].case_id

    def sometimes_fails(case):
        if case.case_id == failed_case:
            raise RuntimeError("deliberate retrieval failure")
        return successful_observation(case)

    results = run_gate_cases(loaded.cases, sometimes_fails, clock=StepClock(0.001))
    report = build_gate_report(loaded, results)

    assert report["metrics"]["errors"]["count"] == 1
    assert report["cases"][0]["observed_behavior"] == "error"
    assert report["cases"][0]["error"] == "RuntimeError: deliberate retrieval failure"
    assert check_by_id(report, "errors.maximum_count")["passed"] is False
    assert report["status"] == "fail"


def test_summary_and_exit_code_make_failure_human_readable():
    loaded = load_evaluation_gate(GATE_CONFIG_PATH, project_root=PROJECT_ROOT)
    results = run_gate_cases(loaded.cases, successful_observation, clock=StepClock(0.001))
    report = build_gate_report(loaded, results)

    summary = render_gate_summary(report)

    assert "Evaluation gate: compact_evaluation_gate@1.0.0" in summary
    assert "[PASS] retrieval.minimum_recall_at_k" in summary
    assert "Overall: PASS (9/9 checks passed)" in summary
    assert "faithfulness is not scored" in summary
    assert gate_exit_code(report) == 0


def test_referenced_citations_ignore_uncited_context_chunks():
    citations = [
        {"citation_id": "[1]", "chunk_ids": ["one"]},
        {"citation_id": "[2]", "chunk_ids": ["two", "three"]},
    ]

    assert referenced_citation_chunk_ids("Use [2], repeat [2], and ignore source one.", citations) == ("two", "three")
