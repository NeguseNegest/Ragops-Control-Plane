import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ragops.evaluation.no_answer import (
    NoAnswerExample,
    calibrated_threshold,
    load_no_answer_config,
    load_supported_report,
    replay_no_answer_evaluation,
    run_no_answer_evaluation,
    validate_no_answer_inputs,
    write_no_answer_artifacts,
)
from ragops.generation.no_answer import NO_ANSWER_PROMPT_VERSION, NO_ANSWER_RESPONSE, NoAnswerResult, build_no_answer_prompt, generate_no_answer
from ragops.retrieval.dense import RetrievedChunk
from ragops.routing.config import load_router_config
from ragops.routing.probe import run_initial_retrieval_probe
from ragops.routing.router import RuleBasedRouter

UNSUPPORTED_SCORES = {
    "How do I create and apply a migration with the Django ORM?": (0.4572904, 0.00018515),
    "How should I configure namespaces and metadata filtering in Pinecone?": (0.48581672, 0.01481832),
    "How do I define and launch a hyperparameter sweep in Weights & Biases?": (0.5302763, 0.02283365),
    "Which Redis eviction policy should I choose for a write-heavy cache?": (0.44578928, 0.01210892),
    "What are the recommended adult dosages and side effects of ibuprofen?": (0.37081277, 0.14504233),
    "How do I configure blueprints and application factories in Flask?": (0.50113547, 0.01575321),
    "How do I define schema and multi-tenancy in Weaviate?": (0.5156742, 0.03993097),
    "How should I set partition keys and consistency in Apache Cassandra?": (0.45859376, 0.025141),
    "How do I configure Redux Toolkit state persistence in React?": (0.3132227, 0.0160005),
    "What Terraform resources create a private AWS VPC with NAT gateways?": (0.41723347, 0.01037383),
    "How do I configure Kafka consumer offsets and retention?": (0.49200958, 0.01437771),
    "What acetaminophen dose is safe for a child with fever?": (0.2304779, 0.0200688),
}


def routed_probe(query, top_score, score_gap):
    chunks = [
        RetrievedChunk(chunk_id="chunk-1", document_id="doc-1", text="Unrelated evidence one.", score=top_score, rank=1, metadata={}),
        RetrievedChunk(chunk_id="chunk-2", document_id="doc-2", text="Unrelated evidence two.", score=top_score - score_gap, rank=2, metadata={}),
    ]
    probe = run_initial_retrieval_probe(query, lambda **kwargs: chunks, clock=lambda: 1.0)
    router = RuleBasedRouter(load_router_config("configs/routed.yaml", project_root=Path.cwd()))
    return router.select_probe(probe)


class FakeRuntime:
    def route_query(self, query):
        top_score, score_gap = UNSUPPORTED_SCORES[query]
        return routed_probe(query, top_score, score_gap)


def test_checked_in_no_answer_inputs_are_reviewed_split_and_cross_validated():
    config = load_no_answer_config("configs/no_answer.yaml", project_root=Path.cwd())

    inputs = validate_no_answer_inputs(config)

    assert config.name == "no_answer"
    assert config.status == "draft"
    assert inputs["counts"] == {"calibration": 5, "evaluation": 7}
    assert len(inputs["examples"]) == 12
    assert len(inputs["supported_report"]["questions"]) == 45
    assert inputs["router_config"].thresholds.no_answer.top_score_below == 0.531
    assert all(example.review_status == "verified" for example in inputs["examples"])


def test_no_answer_examples_reject_unknown_fields_and_non_refusal_behavior():
    payload = {
        "id": "unsupported-1",
        "question": "How does an unsupported product work?",
        "split": "evaluation",
        "category": "near_domain_technology",
        "expected_behavior": "answer",
        "provenance": "test",
        "review_status": "verified",
        "reviewed_by": "reviewer",
        "extra": True,
    }

    with pytest.raises(ValidationError):
        NoAnswerExample.model_validate(payload)


def test_threshold_is_calibration_max_plus_margin_rounded_up():
    config = load_no_answer_config("configs/no_answer.yaml", project_root=Path.cwd())
    scores = [0.4572904, 0.48581672, 0.5302763, 0.44578928, 0.37081277]

    assert calibrated_threshold(scores, config.threshold_selection) == 0.531

    with pytest.raises(ValueError, match="finite unsupported scores"):
        calibrated_threshold([], config.threshold_selection)


def test_supported_report_rejects_a_non_mapping_json_root(tmp_path):
    report_path = tmp_path / "supported.json"
    report_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dense_baseline question report"):
        load_supported_report(report_path, expected_questions=45)


def test_no_answer_prompt_and_result_are_exact_deterministic_and_citation_free():
    result = routed_probe("How do I configure Pinecone?", 0.48, 0.02)
    assert result.decision.route == "NO_ANSWER"

    prompt = build_no_answer_prompt(result.probe.query, result.decision)
    first = generate_no_answer(result.probe.query, result.decision)
    second = generate_no_answer(result.probe.query, result.decision.model_dump(mode="python"))

    assert "Do not answer the question" in prompt
    assert NO_ANSWER_RESPONSE in prompt
    assert first == second
    assert first.answer == NO_ANSWER_RESPONSE
    assert first.prompt_version == NO_ANSWER_PROMPT_VERSION
    assert first.generated_by == "deterministic_policy"
    assert first.citations == ()
    assert first.used_chunk_ids == ()


def test_no_answer_generation_rejects_non_refusal_route_and_output_drift():
    result = routed_probe("What is FastAPI?", 0.8, 0.1)
    assert result.decision.route == "FAST"

    with pytest.raises(ValueError, match="NO_ANSWER refusal decision"):
        generate_no_answer(result.probe.query, result.decision)

    with pytest.raises(ValidationError, match="exact policy refusal"):
        NoAnswerResult(
            answer="A fabricated answer.",
            prompt_version="no_answer_v1",
            prompt_sha256="a" * 64,
            generated_by="deterministic_policy",
        )


def test_no_answer_evaluation_measures_refusal_and_false_refusal_tradeoff():
    config = load_no_answer_config("configs/no_answer.yaml", project_root=Path.cwd())

    report = run_no_answer_evaluation(config, FakeRuntime())

    assert report["threshold"] == {
        "method": "calibration_max_plus_margin_ceiling",
        "maximum_calibration_unsupported_score": 0.5302763,
        "score_margin": 0.0005,
        "decimal_places": 3,
        "recommended": 0.531,
        "configured": 0.531,
    }
    assert report["counts"] == {
        "unsupported": 12,
        "calibration_unsupported": 5,
        "evaluation_unsupported": 7,
        "supported": 45,
    }
    assert report["evaluation_id"] == "no_answer@0.1.0"
    assert report["probe"] == {"pipeline_config": "dense_baseline", "top_k": 2, "feature_schema_version": 1}
    assert report["refusal_policy"] == {
        "answer": NO_ANSWER_RESPONSE,
        "prompt_version": "no_answer_v1",
        "generated_by": "deterministic_policy",
    }
    assert report["metrics"]["confusion_matrix"] == {
        "true_refusal": 12,
        "missed_unsupported": 0,
        "false_refusal": 9,
        "supported_answer": 36,
    }
    assert report["metrics"]["unsupported_refusal_accuracy"] == 1.0
    assert report["metrics"]["evaluation_refusal_accuracy"] == 1.0
    assert report["metrics"]["supported_answer_rate"] == 0.8
    assert report["metrics"]["refusal_precision"] == pytest.approx(12 / 21)
    assert report["metrics"]["balanced_accuracy"] == 0.9
    assert report["acceptance"]["passed"]
    unsupported = [row for row in report["questions"] if row["query_type"] == "unsupported"]
    assert all(row["refusal_answer"] == NO_ANSWER_RESPONSE and row["correct"] for row in unsupported)
    assert all(len(row["refusal_prompt_sha256"]) == 64 for row in unsupported)
    assert all(row["refusal_generated_by"] == "deterministic_policy" for row in unsupported)


def test_no_answer_replay_recomputes_current_router_without_live_retrieval():
    config = load_no_answer_config("configs/no_answer.yaml", project_root=Path.cwd())
    source = json.loads(Path("reports/evaluations/no_answer.json").read_text(encoding="utf-8"))

    report = replay_no_answer_evaluation(config, source)

    assert report == source
    assert report["router_id"] == "rule_router@0.2.0"
    assert report["metrics"]["unsupported_refusal_accuracy"] == 1.0
    assert report["metrics"]["supported_false_refusal_rate"] == 0.2


def test_no_answer_replay_rejects_missing_or_changed_provenance():
    config = load_no_answer_config("configs/no_answer.yaml", project_root=Path.cwd())
    source = json.loads(Path("reports/evaluations/no_answer.json").read_text(encoding="utf-8"))
    source["questions"][0]["question"] = "Changed question provenance"

    with pytest.raises(ValueError, match="provenance differs"):
        replay_no_answer_evaluation(config, source)

    source = json.loads(Path("reports/evaluations/no_answer.json").read_text(encoding="utf-8"))
    supported = next(row for row in source["questions"] if row["query_type"] == "supported")
    supported["top_score"] += 0.01
    with pytest.raises(ValueError, match="scores differ from current supported evidence"):
        replay_no_answer_evaluation(config, source)


def test_no_answer_artifacts_are_atomic_complete_and_refuse_overwrite(tmp_path):
    config = load_no_answer_config("configs/no_answer.yaml", project_root=Path.cwd())
    output = config.output.model_copy(
        update={"json_path": tmp_path / "no_answer.json", "csv_path": tmp_path / "no_answer.csv"}
    )
    config = config.model_copy(update={"output": output})
    report = run_no_answer_evaluation(config, FakeRuntime())

    paths = write_no_answer_artifacts(report, config)

    assert paths == (tmp_path / "no_answer.json", tmp_path / "no_answer.csv")
    assert json.loads(paths[0].read_text())["metrics"] == report["metrics"]
    with paths[1].open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == 57
    assert sum(row["refused"] == "True" for row in rows) == 21
    assert b"\r\n" not in paths[1].read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_no_answer_artifacts(report, config)
