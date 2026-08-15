import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragops.pipeline_registry import PipelineRegistry, PipelineStatus, PipelineVersion
from ragops.routing.probe import FEATURE_SCHEMA_VERSION

ROUTER_CONFIG_SCHEMA_VERSION = 1
DEFAULT_ROUTER_CONFIG_PATH = Path("configs/routed.yaml")

RouterRoute = Literal["FAST", "STANDARD", "CAREFUL", "NO_ANSWER"]
RetrievalPipelineName = Literal["dense_baseline", "hybrid_rrf", "hybrid_rrf_cross_encoder"]
RouteEligibleStatus = Literal["evaluated", "approved"]
EXPECTED_DECISION_ORDER = ("NO_ANSWER", "CAREFUL", "FAST", "STANDARD")


class StrictRouterModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InitialProbeConfig(StrictRouterModel):
    """Cheap corpus-confidence probe shared by the design and Day 37 runtime."""

    pipeline_config: Literal["dense_baseline"] = "dense_baseline"
    top_k: int = Field(default=2, ge=2, le=5)


class NoAnswerThresholds(StrictRouterModel):
    """Conservative refusal-candidate thresholds evaluated first."""

    on_empty_results: Literal[True] = True
    top_score_below: float = Field(ge=-1, le=1)

    @field_validator("top_score_below")
    @classmethod
    def require_finite_score(cls, value):
        if not math.isfinite(value):
            raise ValueError("NO_ANSWER score threshold must be finite.")
        return value


class CarefulThresholds(StrictRouterModel):
    """Any matching ambiguity or complexity condition makes a query CAREFUL."""

    top_score_below: float = Field(ge=-1, le=1)
    on_missing_score_gap: Literal[True] = True
    score_gap_below: float = Field(ge=0, le=2)
    token_count_above: int = Field(gt=0)
    complexity_marker_count_at_least: int = Field(ge=1)
    clause_marker_count_at_least: int = Field(ge=1)
    long_token_ratio_at_least: float = Field(ge=0, le=1)

    @field_validator("top_score_below", "score_gap_below", "long_token_ratio_at_least")
    @classmethod
    def require_finite_values(cls, value):
        if not math.isfinite(value):
            raise ValueError("CAREFUL thresholds must be finite.")
        return value


class FastThresholds(StrictRouterModel):
    """Every high-confidence/simple-query condition must match for FAST."""

    top_score_at_least: float = Field(ge=-1, le=1)
    score_gap_at_least: float = Field(ge=0, le=2)
    token_count_at_most: int = Field(gt=0)
    complexity_marker_count_at_most: int = Field(ge=0)
    clause_marker_count_at_most: int = Field(ge=0)
    long_token_ratio_at_most: float = Field(ge=0, le=1)

    @field_validator("top_score_at_least", "score_gap_at_least", "long_token_ratio_at_most")
    @classmethod
    def require_finite_values(cls, value):
        if not math.isfinite(value):
            raise ValueError("FAST thresholds must be finite.")
        return value


class RouterThresholds(StrictRouterModel):
    no_answer: NoAnswerThresholds
    careful: CarefulThresholds
    fast: FastThresholds

    @model_validator(mode="after")
    def require_ordered_non_overlapping_bands(self):
        if not self.no_answer.top_score_below < self.careful.top_score_below < self.fast.top_score_at_least:
            raise ValueError("Score thresholds must increase from NO_ANSWER to CAREFUL to FAST.")
        if not self.careful.score_gap_below < self.fast.score_gap_at_least:
            raise ValueError("CAREFUL score-gap threshold must be below the FAST score-gap threshold.")
        if not self.fast.token_count_at_most < self.careful.token_count_above:
            raise ValueError("FAST and CAREFUL token thresholds must leave a STANDARD band.")
        if not self.fast.complexity_marker_count_at_most < self.careful.complexity_marker_count_at_least:
            raise ValueError("FAST and CAREFUL complexity-marker thresholds must not overlap.")
        if not self.fast.clause_marker_count_at_most < self.careful.clause_marker_count_at_least:
            raise ValueError("FAST and CAREFUL clause-marker thresholds must not overlap.")
        if not self.fast.long_token_ratio_at_most < self.careful.long_token_ratio_at_least:
            raise ValueError("FAST and CAREFUL long-token thresholds must leave a STANDARD band.")
        return self


class RetrievalRouteConfig(StrictRouterModel):
    """Execution intent for a route that retrieves and generates an answer."""

    pipeline_config: RetrievalPipelineName
    allowed_pipeline_statuses: tuple[RouteEligibleStatus, ...] = Field(min_length=1)
    maximum_top_k: int = Field(gt=0, le=20)
    reuse_probe: bool = False
    generate_answer: Literal[True] = True

    @field_validator("allowed_pipeline_statuses")
    @classmethod
    def require_unique_pipeline_statuses(cls, values):
        if len(values) != len(set(values)):
            raise ValueError("Route pipeline statuses must be unique.")
        return values


class NoAnswerRouteConfig(StrictRouterModel):
    """Execution intent for an unsupported-query decision."""

    pipeline_config: None = None
    maximum_top_k: Literal[0] = 0
    reuse_probe: Literal[False] = False
    generate_answer: Literal[False] = False
    response_mode: Literal["refusal"] = "refusal"


class RouterRoutes(StrictRouterModel):
    fast: RetrievalRouteConfig
    standard: RetrievalRouteConfig
    careful: RetrievalRouteConfig
    no_answer: NoAnswerRouteConfig


class RouterCalibrationConfig(StrictRouterModel):
    """Provenance and known coverage limits for the initial draft thresholds."""

    source_report: Path
    source_run_name: Literal["dense_baseline"] = "dense_baseline"
    question_count: int = Field(gt=0)
    unsupported_question_count: int = Field(ge=0)

    @field_validator("source_report", mode="before")
    @classmethod
    def require_non_empty_path(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("Router calibration report path must not be empty.")
        return value

    @model_validator(mode="after")
    def require_valid_coverage_counts(self):
        if self.unsupported_question_count > self.question_count:
            raise ValueError("Unsupported calibration count cannot exceed the total question count.")
        return self


class RouterConfig(StrictRouterModel):
    """Versioned Day 36 router policy and Day 37 probe contract."""

    schema_version: Literal[1] = ROUTER_CONFIG_SCHEMA_VERSION
    name: str = Field(min_length=1)
    version: PipelineVersion
    status: PipelineStatus
    feature_schema_version: Literal[1] = FEATURE_SCHEMA_VERSION
    decision_order: tuple[RouterRoute, ...]
    probe: InitialProbeConfig
    thresholds: RouterThresholds
    routes: RouterRoutes
    calibration: RouterCalibrationConfig

    @field_validator("name")
    @classmethod
    def require_stable_name(cls, value):
        value = value.strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
            raise ValueError("Router name must use lowercase letters, numbers, underscores, or hyphens.")
        return value

    @model_validator(mode="after")
    def require_safe_policy_shape(self):
        if self.decision_order != EXPECTED_DECISION_ORDER:
            raise ValueError(f"Router decision_order must be {EXPECTED_DECISION_ORDER}.")
        if self.routes.fast.reuse_probe:
            if self.routes.fast.pipeline_config != self.probe.pipeline_config:
                raise ValueError("FAST can reuse probe chunks only when it uses the probe pipeline.")
            if self.routes.fast.maximum_top_k > self.probe.top_k:
                raise ValueError("FAST maximum_top_k cannot exceed probe.top_k when reuse_probe is enabled.")
        for route_name in ("standard", "careful"):
            if getattr(self.routes, route_name).reuse_probe:
                raise ValueError(f"{route_name.upper()} cannot reuse the dense probe in router schema version 1.")
        return self


def resolve_router_path(path, project_root=None):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (Path(project_root or Path.cwd()).resolve() / path).resolve()


def load_router_config(config_path=DEFAULT_ROUTER_CONFIG_PATH, project_root=None):
    """Load and strictly validate the versioned router YAML."""
    config_path = resolve_router_path(config_path, project_root)
    if not config_path.is_file():
        raise FileNotFoundError(f"Router config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Router config contains invalid YAML: {config_path}: {error}") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Router config must contain a YAML mapping: {config_path}")
    return RouterConfig.model_validate(payload)


def validate_router_registry_references(config, registry):
    """Require every retrieval route to reference an eligible registered pipeline."""
    config = config if isinstance(config, RouterConfig) else RouterConfig.model_validate(config)
    registry = registry if isinstance(registry, PipelineRegistry) else PipelineRegistry.model_validate(registry)
    entries = {entry.name: entry for entry in registry.pipelines}
    verified = {}
    for route_name in ("fast", "standard", "careful"):
        route = getattr(config.routes, route_name)
        try:
            entry = entries[route.pipeline_config]
        except KeyError as error:
            raise ValueError(f"Router route {route_name.upper()} references an unregistered pipeline {route.pipeline_config!r}.") from error
        if entry.status not in route.allowed_pipeline_statuses:
            raise ValueError(
                f"Router route {route_name.upper()} does not allow pipeline status {entry.status!r} for {entry.pipeline_id}."
            )
        verified[route_name.upper()] = entry.pipeline_id
    for route_name in ("FAST", "STANDARD"):
        if verified[route_name] != registry.aliases.production:
            raise ValueError(f"Router route {route_name} must reference the registry production alias {registry.aliases.production}.")
    if verified["CAREFUL"] != registry.aliases.candidate:
        raise ValueError(f"Router route CAREFUL must reference the registry candidate alias {registry.aliases.candidate}.")
    return verified


def validate_router_calibration(config, project_root=None):
    """Validate threshold provenance and summarize the recorded dense confidence range."""
    config = config if isinstance(config, RouterConfig) else RouterConfig.model_validate(config)
    report_path = resolve_router_path(config.calibration.source_report, project_root)
    if not report_path.is_file():
        raise FileNotFoundError(f"Router calibration report does not exist: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Router calibration report contains invalid JSON: {report_path}: {error}") from error
    if not isinstance(report, dict) or report.get("run_name") != config.calibration.source_run_name:
        raise ValueError("Router calibration report run_name does not match the configured source_run_name.")
    questions = report.get("questions")
    if not isinstance(questions, list) or len(questions) != config.calibration.question_count:
        actual_count = len(questions) if isinstance(questions, list) else None
        raise ValueError(
            f"Router calibration report question count is {actual_count}; expected {config.calibration.question_count}."
        )

    top_scores = []
    score_gaps = []
    for question in questions:
        scores = question.get("retrieved_scores") if isinstance(question, dict) else None
        if not isinstance(scores, list) or len(scores) < config.probe.top_k:
            raise ValueError("Every router calibration question must contain at least probe.top_k retrieved scores.")
        try:
            top_score = float(scores[0])
            second_score = float(scores[1])
        except (TypeError, ValueError) as error:
            raise ValueError("Router calibration scores must be numeric.") from error
        if not math.isfinite(top_score) or not math.isfinite(second_score) or top_score < second_score:
            raise ValueError("Router calibration scores must be finite and descending.")
        top_scores.append(top_score)
        score_gaps.append(top_score - second_score)

    return {
        "report_path": str(report_path),
        "question_count": len(questions),
        "unsupported_question_count": config.calibration.unsupported_question_count,
        "top_score": {"minimum": min(top_scores), "median": median(top_scores), "maximum": max(top_scores)},
        "score_gap": {"minimum": min(score_gaps), "median": median(score_gaps), "maximum": max(score_gaps)},
    }
