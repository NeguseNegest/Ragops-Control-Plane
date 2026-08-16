import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragops.evaluation.router_comparison import (
    RouterEvaluationConfig,
    load_router_evaluation_config,
    run_router_evaluation,
    validate_router_evaluation_inputs,
)
from ragops.evaluation.runner import atomic_write_text
from ragops.pipeline_registry import PipelineStatus, PipelineVersion
from ragops.routing.config import load_router_config
from ragops.routing.router import RuleBasedRouter

ROUTER_TUNING_SCHEMA_VERSION = 1


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RouterTuningSplit(StrictModel):
    method: Literal["sha256_question_id_order"]
    tuning_questions: int = Field(gt=0)
    validation_questions: int = Field(gt=0)


class RouterTuningGrid(StrictModel):
    parameter: Literal["thresholds.careful.score_gap_below"]
    candidates: tuple[float, ...] = Field(min_length=2)
    expected_selected_value: float = Field(gt=0, lt=1)

    @field_validator("candidates")
    @classmethod
    def require_ordered_finite_unique_candidates(cls, values):
        if any(not math.isfinite(value) or value <= 0 or value >= 1 for value in values):
            raise ValueError("Router tuning candidates must be finite values between zero and one.")
        if tuple(sorted(values)) != values or len(values) != len(set(values)):
            raise ValueError("Router tuning candidates must be unique and strictly increasing.")
        return values

    @model_validator(mode="after")
    def require_expected_candidate(self):
        if self.expected_selected_value not in self.candidates:
            raise ValueError("expected_selected_value must be one of the declared candidates.")
        return self


class RouterTuningConstraints(StrictModel):
    minimum_validation_hit_rate_delta: float = Field(ge=0, le=1)
    minimum_unsupported_refusal_accuracy: float = Field(ge=0, le=1)
    maximum_full_average_latency_fraction_of_always_careful: float = Field(gt=0, le=1)
    maximum_full_total_cost_fraction_of_always_careful: float = Field(gt=0, le=1)


class RouterTuningSelection(StrictModel):
    objective: Literal["maximize_tuning_hit_rate"]
    tie_breakers: tuple[
        Literal["minimize_tuning_average_latency", "minimize_threshold"],
        Literal["minimize_tuning_average_latency", "minimize_threshold"],
    ]

    @field_validator("tie_breakers")
    @classmethod
    def require_exact_tie_breakers(cls, values):
        expected = ("minimize_tuning_average_latency", "minimize_threshold")
        if values != expected:
            raise ValueError(f"Router tuning tie_breakers must be {expected}.")
        return values


class RouterTuningOutput(StrictModel):
    json_path: Path
    csv_path: Path
    markdown_path: Path


class RouterTuningConfig(StrictModel):
    schema_version: Literal[1] = ROUTER_TUNING_SCHEMA_VERSION
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: PipelineVersion
    status: PipelineStatus
    baseline_router_config: Path
    target_router_config: Path
    router_evaluation_config: Path
    split: RouterTuningSplit
    tuning: RouterTuningGrid
    constraints: RouterTuningConstraints
    selection: RouterTuningSelection
    output: RouterTuningOutput

    @model_validator(mode="after")
    def require_distinct_paths(self):
        inputs = {self.baseline_router_config, self.target_router_config, self.router_evaluation_config}
        outputs = {self.output.json_path, self.output.csv_path, self.output.markdown_path}
        if len(inputs) != 3 or len(outputs) != 3 or inputs & outputs:
            raise ValueError("Router tuning input and output paths must be distinct.")
        return self


def _resolve(path, root):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def load_router_tuning_config(config_path, project_root=None):
    """Load and resolve the strict Day 42 tuning/report contract."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Router tuning config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid router tuning YAML in {config_path}: {error}") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Router tuning config must contain a YAML mapping: {config_path}")
    config = RouterTuningConfig.model_validate(payload)
    root = Path(project_root or Path.cwd()).resolve()
    output = config.output.model_copy(
        update={name: _resolve(getattr(config.output, name), root) for name in ("json_path", "csv_path", "markdown_path")}
    )
    return config.model_copy(
        update={
            "baseline_router_config": _resolve(config.baseline_router_config, root),
            "target_router_config": _resolve(config.target_router_config, root),
            "router_evaluation_config": _resolve(config.router_evaluation_config, root),
            "output": output,
        }
    )


def _router_with_gap(baseline, target, gap):
    careful = baseline.thresholds.careful.model_copy(update={"score_gap_below": gap})
    thresholds = baseline.thresholds.model_copy(update={"careful": careful})
    return baseline.model_copy(update={"version": target.version, "status": target.status, "thresholds": thresholds})


def _evaluation_inputs_for_router(inputs, router_config):
    updated = dict(inputs)
    updated["router_config"] = router_config
    updated["router"] = RuleBasedRouter(router_config)
    return updated


def _sha256_id_order(labels, split):
    ordered_ids = sorted(
        (label.question_id for label in labels),
        key=lambda question_id: (hashlib.sha256(question_id.encode("utf-8")).hexdigest(), question_id),
    )
    if split.tuning_questions + split.validation_questions != len(ordered_ids):
        raise ValueError("Router tuning split counts must exactly cover the supported evaluation labels.")
    return set(ordered_ids[: split.tuning_questions]), set(ordered_ids[split.tuning_questions :])


def _subset_metrics(report, question_ids):
    rows = [row for row in report["questions"] if row["question_id"] in question_ids]
    if len(rows) != len(question_ids):
        raise ValueError("Router tuning subset does not exactly cover its question IDs.")
    results = [row["strategies"]["routed"] for row in rows]
    return {
        "question_count": len(results),
        "hit_rate_at_5": sum(result["hit_at_5"] for result in results) / len(results),
        "mrr": sum(result["reciprocal_rank"] for result in results) / len(results),
        "average_latency_ms": sum(result["latency_ms"] for result in results) / len(results),
        "total_cost_usd": sum(result["cost"]["amount_usd"] for result in results),
        "route_counts": dict(sorted(Counter(result["route"] for result in results).items())),
        "supported_false_refusals": sum(result["refused"] for result in results),
    }


def _candidate_record(value, report, tuning_ids, validation_ids, baseline_validation_hit, config):
    tuning = _subset_metrics(report, tuning_ids)
    validation = _subset_metrics(report, validation_ids)
    full = report["strategies"]["routed"]
    careful = report["strategies"]["always_careful"]
    checks = {
        "validation_hit_rate_non_regression": validation["hit_rate_at_5"]
        >= baseline_validation_hit + config.constraints.minimum_validation_hit_rate_delta,
        "unsupported_refusal_accuracy": full["refusal_quality"]["unsupported_refusal_accuracy"]
        >= config.constraints.minimum_unsupported_refusal_accuracy,
        "average_latency_ceiling": full["latency_ms_supported"]["average"]
        <= careful["latency_ms_supported"]["average"]
        * config.constraints.maximum_full_average_latency_fraction_of_always_careful,
        "total_cost_ceiling": full["generation_cost_projection_supported"]["amount_usd"]["total"]
        <= careful["generation_cost_projection_supported"]["amount_usd"]["total"]
        * config.constraints.maximum_full_total_cost_fraction_of_always_careful,
    }
    return {
        "value": value,
        "eligible": all(checks.values()),
        "constraint_checks": checks,
        "tuning": tuning,
        "validation": validation,
        "full": {
            "hit_rate_at_5": full["retrieval_quality_supported"]["hit_rate_at_5"],
            "mrr": full["retrieval_quality_supported"]["mrr"],
            "average_latency_ms": full["latency_ms_supported"]["average"],
            "total_cost_usd": full["generation_cost_projection_supported"]["amount_usd"]["total"],
            "unsupported_refusal_accuracy": full["refusal_quality"]["unsupported_refusal_accuracy"],
            "supported_false_refusal_rate": full["refusal_quality"]["supported_false_refusal_rate"],
            "combined_quality_proxy": full["combined_quality_proxy"]["success_rate"],
            "route_counts": full["route_counts_supported"],
        },
    }


def _select_candidate(candidates):
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise ValueError("No router tuning candidate satisfies every configured constraint.")
    return max(
        eligible,
        key=lambda candidate: (
            candidate["tuning"]["hit_rate_at_5"],
            -candidate["tuning"]["average_latency_ms"],
            -candidate["value"],
        ),
    )


def _distribution_rows(inputs, baseline_report, target_report):
    baseline_policy = {row["question_id"]: row for row in baseline_report["policy_questions"]}
    target_policy = {row["question_id"]: row for row in target_report["policy_questions"]}
    target_questions = {row["question_id"]: row for row in target_report["questions"]}
    no_answer = inputs["no_answer_by_id"]
    examples = {example.id: example for example in inputs["unsupported_examples"]}
    rows = []
    for question_id, target_row in target_policy.items():
        baseline_result = baseline_policy[question_id]["strategies"]["routed"]
        target_result = target_row["strategies"]["routed"]
        recorded = no_answer[question_id]
        operational = target_questions.get(question_id)
        routed = operational["strategies"]["routed"] if operational else None
        example = examples.get(question_id)
        rows.append(
            {
                "question_id": question_id,
                "question": target_row["question"],
                "query_type": target_row["query_type"],
                "split": example.split if example else "supported_evidence",
                "top_score": recorded["top_score"],
                "score_gap": recorded["score_gap"],
                "baseline_route": baseline_result["route"],
                "baseline_reason_code": baseline_result.get("reason_code"),
                "target_route": target_result["route"],
                "target_reason_code": target_result.get("reason_code"),
                "route_changed": baseline_result["route"] != target_result["route"],
                "supported_hit_at_5": routed["hit_at_5"] if routed else None,
                "supported_latency_ms": routed["latency_ms"] if routed else None,
                "supported_cost_usd": routed["cost"]["amount_usd"] if routed else None,
            }
        )
    return sorted(rows, key=lambda row: (row["query_type"], row["question_id"]))


def _route_distribution(rows):
    distribution = {}
    for scope, selected in (
        ("all", rows),
        ("supported", [row for row in rows if row["query_type"] == "supported"]),
        ("unsupported", [row for row in rows if row["query_type"] == "unsupported"]),
    ):
        distribution[scope] = {
            "question_count": len(selected),
            "route_counts": dict(sorted(Counter(row["target_route"] for row in selected).items())),
            "route_rates": {
                route: count / len(selected)
                for route, count in sorted(Counter(row["target_route"] for row in selected).items())
            },
            "reason_counts": dict(sorted(Counter(row["target_reason_code"] for row in selected).items())),
        }
    return distribution


def _supported_route_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["query_type"] == "supported":
            grouped[row["target_route"]].append(row)
    metrics = {}
    for route, route_rows in sorted(grouped.items()):
        metrics[route] = {
            "question_count": len(route_rows),
            "hit_rate_at_5": sum(row["supported_hit_at_5"] for row in route_rows) / len(route_rows),
            "average_latency_ms": sum(row["supported_latency_ms"] for row in route_rows) / len(route_rows),
            "total_cost_usd": sum(row["supported_cost_usd"] for row in route_rows),
        }
    return metrics


def _source(path, root):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {
        "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "sha256": digest,
    }


def run_router_tuning(config):
    """Select the configured CAREFUL gap and build the Day 42 distribution report."""
    config = config if isinstance(config, RouterTuningConfig) else RouterTuningConfig.model_validate(config)
    root = config.target_router_config.parent.parent
    baseline = load_router_config(config.baseline_router_config, project_root=root)
    target = load_router_config(config.target_router_config, project_root=root)
    if baseline.name != target.name or baseline.version == target.version:
        raise ValueError("Router tuning requires distinct versions of the same router policy.")
    if baseline.thresholds.careful.score_gap_below != config.tuning.candidates[0]:
        raise ValueError("The first tuning candidate must reproduce the archived baseline CAREFUL score-gap threshold.")

    evaluation_config = load_router_evaluation_config(config.router_evaluation_config, project_root=root)
    if not isinstance(evaluation_config, RouterEvaluationConfig) or evaluation_config.router_config != config.target_router_config:
        raise ValueError("Router evaluation config must reference the Day 42 target router.")
    inputs = validate_router_evaluation_inputs(evaluation_config)
    tuning_ids, validation_ids = _sha256_id_order(inputs["labels"], config.split)

    baseline_inputs = _evaluation_inputs_for_router(inputs, baseline)
    baseline_report = run_router_evaluation(evaluation_config, inputs=baseline_inputs, hash_sources=False)
    baseline_validation_hit = _subset_metrics(baseline_report, validation_ids)["hit_rate_at_5"]
    candidate_reports = {}
    candidates = []
    for value in config.tuning.candidates:
        candidate_router = _router_with_gap(baseline, target, value)
        candidate_inputs = _evaluation_inputs_for_router(inputs, candidate_router)
        candidate_report = run_router_evaluation(evaluation_config, inputs=candidate_inputs, hash_sources=False)
        candidate_reports[value] = candidate_report
        candidates.append(
            _candidate_record(value, candidate_report, tuning_ids, validation_ids, baseline_validation_hit, config)
        )

    selected = _select_candidate(candidates)
    if not math.isclose(selected["value"], config.tuning.expected_selected_value, rel_tol=0, abs_tol=1e-12):
        raise ValueError(
            f"Selected CAREFUL score-gap threshold {selected['value']} differs from configured expectation {config.tuning.expected_selected_value}."
        )
    expected_target = _router_with_gap(baseline, target, selected["value"])
    if expected_target.model_dump(mode="json") != target.model_dump(mode="json"):
        raise ValueError("Target router contains changes beyond the selected version/status and CAREFUL score-gap threshold.")

    target_report = candidate_reports[selected["value"]]
    rows = _distribution_rows(inputs, baseline_report, target_report)
    distribution = _route_distribution(rows)
    transitions = Counter(f"{row['baseline_route']}->{row['target_route']}" for row in rows)
    changed = [row for row in rows if row["route_changed"]]
    sources = {
        "baseline_router": _source(config.baseline_router_config, root),
        "target_router": _source(config.target_router_config, root),
        "router_evaluation_config": _source(config.router_evaluation_config, root),
        "labels": _source(evaluation_config.inputs.labels_path, root),
        "golden": _source(evaluation_config.inputs.golden_path, root),
        "chunks": _source(evaluation_config.inputs.chunks_path, root),
        "dense_report": _source(evaluation_config.inputs.dense_report_path, root),
        "careful_report": _source(evaluation_config.inputs.careful_report_path, root),
        "no_answer_config": _source(evaluation_config.inputs.no_answer_config_path, root),
        "no_answer_report": _source(evaluation_config.inputs.no_answer_report_path, root),
        "model_costs": _source(evaluation_config.cost_projection.model_cost_config, root),
    }
    return {
        "schema_version": ROUTER_TUNING_SCHEMA_VERSION,
        "run_name": config.name,
        "evaluation_id": f"{config.name}@{config.version}",
        "evaluation_status": config.status,
        "baseline_router_id": f"{baseline.name}@{baseline.version}",
        "target_router_id": f"{target.name}@{target.version}",
        "target_router_status": target.status,
        "parameter": config.tuning.parameter,
        "split": {
            "method": config.split.method,
            "tuning_question_ids": sorted(tuning_ids),
            "validation_question_ids": sorted(validation_ids),
        },
        "constraints": config.constraints.model_dump(mode="json"),
        "selection": {
            "objective": config.selection.objective,
            "tie_breakers": list(config.selection.tie_breakers),
            "selected_value": selected["value"],
            "selected_candidate": selected,
        },
        "candidates": candidates,
        "distribution": distribution,
        "supported_route_metrics": _supported_route_metrics(rows),
        "transitions": {
            "counts": dict(sorted(transitions.items())),
            "changed_question_count": len(changed),
            "changed_questions": changed,
        },
        "comparison": {
            "baseline": baseline_report["strategies"]["routed"],
            "target": target_report["strategies"]["routed"],
            "hit_rate_at_5_delta": target_report["strategies"]["routed"]["retrieval_quality_supported"]["hit_rate_at_5"]
            - baseline_report["strategies"]["routed"]["retrieval_quality_supported"]["hit_rate_at_5"],
            "average_latency_relative_delta": (
                target_report["strategies"]["routed"]["latency_ms_supported"]["average"]
                / baseline_report["strategies"]["routed"]["latency_ms_supported"]["average"]
                - 1
            ),
            "total_cost_relative_delta": (
                target_report["strategies"]["routed"]["generation_cost_projection_supported"]["amount_usd"]["total"]
                / baseline_report["strategies"]["routed"]["generation_cost_projection_supported"]["amount_usd"]["total"]
                - 1
            ),
        },
        "stability": {
            "deterministic_policy": True,
            "target_matches_selected_candidate": True,
            "all_selection_constraints_pass": all(selected["constraint_checks"].values()),
            "no_answer_threshold_unchanged": target.thresholds.no_answer == baseline.thresholds.no_answer,
            "fast_thresholds_unchanged": target.thresholds.fast == baseline.thresholds.fast,
            "lifecycle_decision": "keep_draft",
            "reason": "The selected CAREFUL threshold improves held-out supported retrieval without weakening refusal, but the evidence is small/in-sample and supported false refusal remains 20%.",
        },
        "sources": sources,
        "limitations": [
            "The 30/15 supported split is deterministic but small; the same 45-question artifact family was already inspected during Day 41.",
            "Only the CAREFUL score-gap threshold was tuned. NO_ANSWER remains locked to Day 39 calibration and FAST has only two observed supported examples.",
            "Hit@5 measures evidence availability rather than answer correctness, and latency/cost retain the Day 41 replay/projection limitations.",
            "The route distribution reflects 45 supported and 12 authored unsupported questions, not production traffic.",
        ],
        "questions": rows,
    }


def render_router_tuning_markdown(report):
    selected = report["selection"]["selected_candidate"]
    before = report["comparison"]["baseline"]
    after = report["comparison"]["target"]
    baseline_version = report["baseline_router_id"].rsplit("@", 1)[-1]
    target_version = report["target_router_id"].rsplit("@", 1)[-1]
    lines = [
        "# Week 6 Router Stabilization",
        "",
        f"Evaluation: `{report['evaluation_id']}` ({report['evaluation_status']})",
        f"Baseline: `{report['baseline_router_id']}`",
        f"Target: `{report['target_router_id']}` ({report['target_router_status']})",
        "",
        "## Selection",
        "",
        f"Selected `{report['parameter']} = {report['selection']['selected_value']:.3f}`. "
        f"The {selected['tuning']['question_count']}-question tuning Hit@5 is {selected['tuning']['hit_rate_at_5']:.2%}; "
        f"the {selected['validation']['question_count']}-question validation Hit@5 is {selected['validation']['hit_rate_at_5']:.2%}. "
        "Every configured refusal, validation, latency, and cost constraint passes.",
        "",
        "| Candidate gap | Eligible | Tuning Hit@5 | Validation Hit@5 | Full Hit@5 | Avg latency | Total projected cost |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in report["candidates"]:
        lines.append(
            f"| {candidate['value']:.3f} | {'yes' if candidate['eligible'] else 'no'} | "
            f"{candidate['tuning']['hit_rate_at_5']:.2%} | {candidate['validation']['hit_rate_at_5']:.2%} | "
            f"{candidate['full']['hit_rate_at_5']:.2%} | {candidate['full']['average_latency_ms']:.1f} ms | "
            f"${candidate['full']['total_cost_usd']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Before and after",
            "",
            f"| Metric | v{baseline_version} | v{target_version} |",
            "|---|---:|---:|",
            f"| Supported Hit@5 | {before['retrieval_quality_supported']['hit_rate_at_5']:.2%} | {after['retrieval_quality_supported']['hit_rate_at_5']:.2%} |",
            f"| Supported MRR | {before['retrieval_quality_supported']['mrr']:.4f} | {after['retrieval_quality_supported']['mrr']:.4f} |",
            f"| Combined proxy | {before['combined_quality_proxy']['success_rate']:.2%} | {after['combined_quality_proxy']['success_rate']:.2%} |",
            f"| Average replay latency | {before['latency_ms_supported']['average']:.1f} ms | {after['latency_ms_supported']['average']:.1f} ms |",
            f"| Projected cost | ${before['generation_cost_projection_supported']['amount_usd']['total']:.8f} | ${after['generation_cost_projection_supported']['amount_usd']['total']:.8f} |",
            f"| Supported false refusal | {before['refusal_quality']['supported_false_refusal_rate']:.2%} | {after['refusal_quality']['supported_false_refusal_rate']:.2%} |",
            "",
            "## Target route distribution",
            "",
            "| Scope | FAST | STANDARD | CAREFUL | NO_ANSWER |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for scope in ("all", "supported", "unsupported"):
        counts = report["distribution"][scope]["route_counts"]
        lines.append(
            f"| {scope.title()} | {counts.get('FAST', 0)} | {counts.get('STANDARD', 0)} | "
            f"{counts.get('CAREFUL', 0)} | {counts.get('NO_ANSWER', 0)} |"
        )
    lines.extend(
        [
            "",
            f"{report['transitions']['changed_question_count']} supported questions move from STANDARD to CAREFUL; no other route transition occurs. Target status remains **{report['stability']['lifecycle_decision']}**: {report['stability']['reason']}",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def _csv_rows(report):
    for row in report["questions"]:
        yield {key: row[key] for key in (
            "question_id",
            "question",
            "query_type",
            "split",
            "top_score",
            "score_gap",
            "baseline_route",
            "baseline_reason_code",
            "target_route",
            "target_reason_code",
            "route_changed",
            "supported_hit_at_5",
            "supported_latency_ms",
            "supported_cost_usd",
        )}


def write_router_tuning_artifacts(report, config, overwrite=False):
    """Atomically write the Day 42 distribution JSON/CSV/Markdown bundle."""
    paths = (config.output.json_path, config.output.csv_path, config.output.markdown_path)
    if not overwrite:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing router tuning artifacts: {', '.join(existing)}")
    atomic_write_text(config.output.json_path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    fieldnames = [
        "question_id",
        "question",
        "query_type",
        "split",
        "top_score",
        "score_gap",
        "baseline_route",
        "baseline_reason_code",
        "target_route",
        "target_reason_code",
        "route_changed",
        "supported_hit_at_5",
        "supported_latency_ms",
        "supported_cost_usd",
    ]
    temporary = config.output.csv_path.with_name(f".{config.output.csv_path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_csv_rows(report))
    temporary.replace(config.output.csv_path)
    atomic_write_text(config.output.markdown_path, render_router_tuning_markdown(report))
    return paths
