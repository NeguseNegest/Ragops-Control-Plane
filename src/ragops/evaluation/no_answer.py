import csv
import json
import math
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragops.evaluation.llm_judge import load_golden_questions
from ragops.evaluation.synthetic_qa import read_jsonl
from ragops.generation.no_answer import NO_ANSWER_PROMPT_VERSION, NO_ANSWER_RESPONSE, generate_no_answer
from ragops.pipeline_registry import PipelineStatus, PipelineVersion
from ragops.retrieval.dense import RetrievedChunk
from ragops.routing.config import load_router_config
from ragops.routing.probe import build_initial_retrieval_features
from ragops.routing.router import RuleBasedRouter

NO_ANSWER_EVALUATION_SCHEMA_VERSION = 1


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoAnswerExample(StrictModel):
    """One reviewed unsupported query with a fixed calibration/evaluation role."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    question: str = Field(min_length=10, max_length=500)
    split: Literal["calibration", "evaluation"]
    category: Literal["near_domain_technology", "high_stakes_out_of_scope"]
    expected_behavior: Literal["refusal"] = "refusal"
    provenance: str = Field(min_length=1)
    review_status: Literal["verified"] = "verified"
    reviewed_by: str = Field(min_length=1)

    @field_validator("question", "provenance", "reviewed_by")
    @classmethod
    def clean_text(cls, value):
        return value.strip()


class NoAnswerDatasetConfig(StrictModel):
    unsupported_path: Path
    golden_path: Path
    supported_report_path: Path
    minimum_calibration_unsupported: int = Field(gt=0)
    minimum_evaluation_unsupported: int = Field(gt=0)
    expected_supported_questions: int = Field(gt=0)


class ThresholdSelectionConfig(StrictModel):
    method: Literal["calibration_max_plus_margin_ceiling"]
    score_margin: float = Field(gt=0, lt=1)
    decimal_places: int = Field(ge=1, le=6)

    @field_validator("score_margin")
    @classmethod
    def require_finite_margin(cls, value):
        if not math.isfinite(value):
            raise ValueError("Threshold score margin must be finite.")
        return value


class NoAnswerAcceptanceConfig(StrictModel):
    minimum_unsupported_refusal_accuracy: float = Field(ge=0, le=1)
    minimum_evaluation_refusal_accuracy: float = Field(ge=0, le=1)
    minimum_supported_answer_rate: float = Field(ge=0, le=1)
    minimum_refusal_precision: float = Field(ge=0, le=1)


class RefusalConfig(StrictModel):
    prompt_version: Literal["no_answer_v1"] = NO_ANSWER_PROMPT_VERSION
    answer: str = Field(min_length=1)

    @field_validator("answer")
    @classmethod
    def require_exact_policy_answer(cls, value):
        if value != NO_ANSWER_RESPONSE:
            raise ValueError("Configured refusal answer must match the deterministic no-answer response.")
        return value


class NoAnswerOutputConfig(StrictModel):
    json_path: Path
    csv_path: Path


class NoAnswerEvaluationConfig(StrictModel):
    schema_version: Literal[1] = NO_ANSWER_EVALUATION_SCHEMA_VERSION
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: PipelineVersion
    status: PipelineStatus
    router_config: Path
    dataset: NoAnswerDatasetConfig
    threshold_selection: ThresholdSelectionConfig
    acceptance: NoAnswerAcceptanceConfig
    refusal: RefusalConfig
    output: NoAnswerOutputConfig

    @model_validator(mode="after")
    def require_distinct_paths(self):
        paths = (
            self.dataset.unsupported_path,
            self.dataset.golden_path,
            self.dataset.supported_report_path,
            self.output.json_path,
            self.output.csv_path,
        )
        if len(paths) != len(set(paths)):
            raise ValueError("No-answer input and output paths must be distinct.")
        return self


def _resolve_path(path, project_root):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(project_root) / path).resolve()


def load_no_answer_config(config_path, project_root=None):
    """Load the strict Day 39 config and resolve every project-relative path."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"No-answer config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid no-answer YAML in {config_path}: {error}") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"No-answer config must contain a YAML mapping: {config_path}")
    config = NoAnswerEvaluationConfig.model_validate(payload)
    root = Path(project_root or Path.cwd()).resolve()
    dataset = config.dataset.model_copy(
        update={
            "unsupported_path": _resolve_path(config.dataset.unsupported_path, root),
            "golden_path": _resolve_path(config.dataset.golden_path, root),
            "supported_report_path": _resolve_path(config.dataset.supported_report_path, root),
        }
    )
    output = config.output.model_copy(
        update={
            "json_path": _resolve_path(config.output.json_path, root),
            "csv_path": _resolve_path(config.output.csv_path, root),
        }
    )
    return config.model_copy(update={"router_config": _resolve_path(config.router_config, root), "dataset": dataset, "output": output})


def load_no_answer_examples(path):
    """Load reviewed unsupported rows and reject duplicate identities/questions."""
    examples = [NoAnswerExample.model_validate(row) for row in read_jsonl(path)]
    ids = [example.id for example in examples]
    questions = [re.sub(r"\s+", " ", example.question.casefold()).strip() for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("No-answer examples must have unique IDs.")
    if len(questions) != len(set(questions)):
        raise ValueError("No-answer examples must have unique questions.")
    return examples


def load_supported_report(path, expected_questions):
    """Load the immutable dense evidence used to measure false refusals."""
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid supported report JSON: {path}: {error}") from error
    if not isinstance(report, dict):
        raise ValueError("Supported report must be the dense_baseline question report.")
    questions = report.get("questions")
    if report.get("run_name") != "dense_baseline" or not isinstance(questions, list):
        raise ValueError("Supported report must be the dense_baseline question report.")
    if len(questions) != expected_questions:
        raise ValueError(f"Supported report contains {len(questions)} questions; expected {expected_questions}.")
    return report


def validate_no_answer_inputs(config):
    """Cross-check reviewed unsupported rows, golden provenance, and supported evidence."""
    examples = load_no_answer_examples(config.dataset.unsupported_path)
    counts = {split: sum(example.split == split for example in examples) for split in ("calibration", "evaluation")}
    if counts["calibration"] < config.dataset.minimum_calibration_unsupported:
        raise ValueError("No-answer dataset has too few calibration examples.")
    if counts["evaluation"] < config.dataset.minimum_evaluation_unsupported:
        raise ValueError("No-answer dataset has too few held-out evaluation examples.")

    golden = {question.id: question for question in load_golden_questions(config.dataset.golden_path)}
    for example in examples:
        if example.split != "calibration":
            continue
        golden_question = golden.get(example.id)
        if golden_question is None or golden_question.query_type != "unsupported":
            raise ValueError(f"Calibration example {example.id} must reference an unsupported golden question.")
        if golden_question.question != example.question:
            raise ValueError(f"Calibration question text differs from golden question {example.id}.")

    supported_report = load_supported_report(config.dataset.supported_report_path, config.dataset.expected_supported_questions)
    router_config = load_router_config(config.router_config, project_root=config.router_config.parent.parent)
    return {"examples": examples, "supported_report": supported_report, "router_config": router_config, "counts": counts}


def calibrated_threshold(calibration_scores, selection):
    """Return max calibration score plus margin, conservatively rounded upward."""
    scores = [float(score) for score in calibration_scores]
    if not scores or any(not math.isfinite(score) for score in scores):
        raise ValueError("Threshold calibration requires finite unsupported scores.")
    factor = 10**selection.decimal_places
    unrounded = max(scores) + selection.score_margin
    return math.ceil(unrounded * factor) / factor


def _supported_features(question):
    scores = question.get("retrieved_scores") if isinstance(question, dict) else None
    chunk_ids = question.get("retrieved_chunk_ids") if isinstance(question, dict) else None
    if not isinstance(scores, list) or len(scores) < 2 or not isinstance(chunk_ids, list) or len(chunk_ids) < 2:
        raise ValueError("Every supported report row requires at least two retrieved scores and chunk IDs.")
    chunks = [
        RetrievedChunk(
            chunk_id=str(chunk_ids[index]),
            document_id="supported-evaluation",
            text="Supported evaluation evidence placeholder.",
            score=float(scores[index]),
            rank=index + 1,
            metadata={},
        )
        for index in range(2)
    ]
    return build_initial_retrieval_features(question.get("question"), chunks, requested_top_k=2)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def refusal_metrics(unsupported_rows, supported_rows):
    true_positive = sum(row["correct"] for row in unsupported_rows)
    false_negative = len(unsupported_rows) - true_positive
    false_positive = sum(row["refused"] for row in supported_rows)
    true_negative = len(supported_rows) - false_positive
    evaluation_rows = [row for row in unsupported_rows if row["split"] == "evaluation"]
    calibration_rows = [row for row in unsupported_rows if row["split"] == "calibration"]
    return {
        "confusion_matrix": {
            "true_refusal": true_positive,
            "missed_unsupported": false_negative,
            "false_refusal": false_positive,
            "supported_answer": true_negative,
        },
        "unsupported_refusal_accuracy": _ratio(true_positive, len(unsupported_rows)),
        "calibration_refusal_accuracy": _ratio(sum(row["refused"] for row in calibration_rows), len(calibration_rows)),
        "evaluation_refusal_accuracy": _ratio(sum(row["refused"] for row in evaluation_rows), len(evaluation_rows)),
        "supported_answer_rate": _ratio(true_negative, len(supported_rows)),
        "supported_false_refusal_rate": _ratio(false_positive, len(supported_rows)),
        "refusal_precision": _ratio(true_positive, true_positive + false_positive),
        "balanced_accuracy": (
            _ratio(true_positive, len(unsupported_rows)) + _ratio(true_negative, len(supported_rows))
        )
        / 2,
        "overall_accuracy": _ratio(true_positive + true_negative, len(unsupported_rows) + len(supported_rows)),
    }


def acceptance_results(metrics, acceptance):
    return {
        "unsupported_refusal_accuracy": metrics["unsupported_refusal_accuracy"] >= acceptance.minimum_unsupported_refusal_accuracy,
        "evaluation_refusal_accuracy": metrics["evaluation_refusal_accuracy"] >= acceptance.minimum_evaluation_refusal_accuracy,
        "supported_answer_rate": metrics["supported_answer_rate"] >= acceptance.minimum_supported_answer_rate,
        "refusal_precision": metrics["refusal_precision"] >= acceptance.minimum_refusal_precision,
    }


def run_no_answer_evaluation(config, runtime, progress=None):
    """Run unsupported probes, replay supported evidence, and measure refusal correctness."""
    inputs = validate_no_answer_inputs(config)
    examples = inputs["examples"]
    router_config = inputs["router_config"]
    unsupported_rows = []
    for index, example in enumerate(examples, start=1):
        result = runtime.route_query(example.question)
        confidence = result.probe.features.retrieval_confidence
        refused = result.decision.route == "NO_ANSWER"
        refusal = generate_no_answer(example.question, result.decision) if refused else None
        unsupported_rows.append(
            {
                "question_id": example.id,
                "question": example.question,
                "query_type": "unsupported",
                "split": example.split,
                "category": example.category,
                "top_score": confidence.top_score,
                "score_gap": confidence.score_gap,
                "route": result.decision.route,
                "reason_code": result.decision.reason_code,
                "refused": refused,
                "refusal_answer": refusal.answer if refusal else None,
                "refusal_prompt_sha256": refusal.prompt_sha256 if refusal else None,
                "refusal_generated_by": refusal.generated_by if refusal else None,
                "correct": refusal is not None and refusal.answer == config.refusal.answer,
            }
        )
        if progress:
            progress({"index": index, "total": len(examples), "question_id": example.id, "route": result.decision.route})

    calibration_scores = [row["top_score"] for row in unsupported_rows if row["split"] == "calibration"]
    recommended_threshold = calibrated_threshold(calibration_scores, config.threshold_selection)
    configured_threshold = router_config.thresholds.no_answer.top_score_below
    if not math.isclose(recommended_threshold, configured_threshold, rel_tol=0, abs_tol=1e-12):
        raise ValueError(
            f"Configured NO_ANSWER threshold {configured_threshold} does not match calibrated threshold {recommended_threshold}."
        )

    router = RuleBasedRouter(router_config)
    supported_rows = []
    for question in inputs["supported_report"]["questions"]:
        features = _supported_features(question)
        decision = router.select(features)
        refused = decision.route == "NO_ANSWER"
        refusal = generate_no_answer(question["question"], decision) if refused else None
        supported_rows.append(
            {
                "question_id": question["question_id"],
                "question": question["question"],
                "query_type": "supported",
                "split": "supported_evidence",
                "category": "supported",
                "top_score": features.retrieval_confidence.top_score,
                "score_gap": features.retrieval_confidence.score_gap,
                "route": decision.route,
                "reason_code": decision.reason_code,
                "refused": refused,
                "refusal_answer": refusal.answer if refusal else None,
                "refusal_prompt_sha256": refusal.prompt_sha256 if refusal else None,
                "refusal_generated_by": refusal.generated_by if refusal else None,
                "correct": not refused,
            }
        )

    metrics = refusal_metrics(unsupported_rows, supported_rows)
    acceptance = acceptance_results(metrics, config.acceptance)
    return {
        "schema_version": NO_ANSWER_EVALUATION_SCHEMA_VERSION,
        "run_name": config.name,
        "evaluation_id": f"{config.name}@{config.version}",
        "evaluation_status": config.status,
        "router_id": f"{router_config.name}@{router_config.version}",
        "router_status": router_config.status,
        "probe": {
            "pipeline_config": router_config.probe.pipeline_config,
            "top_k": router_config.probe.top_k,
            "feature_schema_version": router_config.feature_schema_version,
        },
        "refusal_policy": {
            "answer": config.refusal.answer,
            "prompt_version": config.refusal.prompt_version,
            "generated_by": "deterministic_policy",
        },
        "threshold": {
            "method": config.threshold_selection.method,
            "maximum_calibration_unsupported_score": max(calibration_scores),
            "score_margin": config.threshold_selection.score_margin,
            "decimal_places": config.threshold_selection.decimal_places,
            "recommended": recommended_threshold,
            "configured": configured_threshold,
        },
        "counts": {
            "unsupported": len(unsupported_rows),
            "calibration_unsupported": inputs["counts"]["calibration"],
            "evaluation_unsupported": inputs["counts"]["evaluation"],
            "supported": len(supported_rows),
        },
        "metrics": metrics,
        "acceptance": {**acceptance, "passed": all(acceptance.values())},
        "limitations": [
            "The unsupported set is small and manually authored; refusal accuracy is not a population estimate.",
            "The threshold is corpus/model/index specific and raw cosine scores are not probabilities.",
            "The 20 percent supported false-refusal ceiling is a safety-first draft tradeoff, not a production target.",
            "Only score-threshold refusal is measured; adversarial paraphrases and semantic scope classification remain future work.",
        ],
        "questions": unsupported_rows + supported_rows,
    }


def write_no_answer_artifacts(report, config, overwrite=False):
    """Write deterministic JSON/CSV evidence atomically."""
    paths = (config.output.json_path, config.output.csv_path)
    if not overwrite:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing no-answer artifacts: {', '.join(existing)}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    json_temporary = config.output.json_path.with_name(f".{config.output.json_path.name}.tmp")
    json_temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json_temporary.replace(config.output.json_path)

    fieldnames = [
        "question_id",
        "question",
        "query_type",
        "split",
        "category",
        "top_score",
        "score_gap",
        "route",
        "reason_code",
        "refused",
        "refusal_answer",
        "refusal_prompt_sha256",
        "refusal_generated_by",
        "correct",
    ]
    csv_temporary = config.output.csv_path.with_name(f".{config.output.csv_path.name}.tmp")
    with csv_temporary.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for question in report["questions"]:
            writer.writerow({field: question[field] for field in fieldnames})
    csv_temporary.replace(config.output.csv_path)
    return paths
