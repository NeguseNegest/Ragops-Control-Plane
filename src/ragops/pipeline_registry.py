import hashlib
import json
import math
import os
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

PIPELINE_REGISTRY_SCHEMA_VERSION = 1
DEFAULT_PIPELINE_REGISTRY_CONFIG_PATH = Path("configs/pipeline_registry.yaml")
DEFAULT_PIPELINE_REGISTRY_PATH = Path("reports/pipeline_registry.json")
PIPELINE_TYPES = ("dense", "bm25", "hybrid", "reranked")

PipelineVersion = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$",
    ),
]
PipelineStatus = Literal["draft", "evaluated", "approved", "rejected", "retired"]
PipelineType = Literal["dense", "bm25", "hybrid", "reranked"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PipelineVersionMetadata(StrictModel):
    """Version and lifecycle state embedded in every registered pipeline config."""

    version: PipelineVersion
    status: PipelineStatus


class PipelineAliases(StrictModel):
    """Human-facing pointers to immutable pipeline IDs."""

    baseline: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*@.+$")
    candidate: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*@.+$")
    production: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*@.+$")

    @field_validator("baseline", "candidate", "production")
    @classmethod
    def clean_alias_target(cls, value):
        return value.strip()


class PipelineRegistryConfig(StrictModel):
    """Sources and aliases used to build the checked-in registry artifact."""

    schema_version: Literal[1] = PIPELINE_REGISTRY_SCHEMA_VERSION
    registry_name: str = Field(min_length=1)
    catalog_path: Path = Path("configs/mlflow.yaml")
    comparison_path: Path = Path("reports/evaluations/reranker_vs_baselines.json")
    output_path: Path = DEFAULT_PIPELINE_REGISTRY_PATH
    aliases: PipelineAliases

    @field_validator("registry_name")
    @classmethod
    def clean_registry_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Pipeline registry name must not be empty.")
        return value

    @field_validator("catalog_path", "comparison_path", "output_path", mode="before")
    @classmethod
    def validate_path(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("Pipeline registry paths must not be empty.")
        return value


class RegistryConfigArtifact(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RegistryEvaluationMetrics(StrictModel):
    comparison_depth: Literal[5] = 5
    mrr_at_5: float = Field(ge=0, le=1)
    recall_at_1: float = Field(ge=0, le=1)
    recall_at_3: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    hit_rate_at_1: float = Field(ge=0, le=1)
    hit_rate_at_3: float = Field(ge=0, le=1)
    hit_rate_at_5: float = Field(ge=0, le=1)
    ndcg_at_1: float = Field(ge=0, le=1)
    ndcg_at_3: float = Field(ge=0, le=1)
    ndcg_at_5: float = Field(ge=0, le=1)
    average_latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_finite_metrics(self):
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Registry metric {field_name} must be finite.")
        return self


class RegistryEvaluationEvidence(StrictModel):
    run_name: str = Field(min_length=1)
    report_path: str = Field(min_length=1)
    comparison_path: str = Field(min_length=1)
    benchmark_path: str = Field(min_length=1)
    question_count: int = Field(gt=0)
    metric_scope: str = Field(min_length=1)
    metrics: RegistryEvaluationMetrics
    artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class RegistryTrackingReference(StrictModel):
    experiment_name: str = Field(min_length=1)
    run_name: str = Field(min_length=1)


class PipelineRegistryEntry(StrictModel):
    pipeline_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*@.+$")
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: PipelineVersion
    status: PipelineStatus
    pipeline_type: PipelineType
    retriever_interface: Literal["common_v1"]
    config: RegistryConfigArtifact
    evaluation: RegistryEvaluationEvidence
    tracking: RegistryTrackingReference

    @model_validator(mode="after")
    def validate_identity(self):
        expected_id = f"{self.name}@{self.version}"
        if self.pipeline_id != expected_id:
            raise ValueError(f"Pipeline ID must equal name@version ({expected_id}).")
        if self.evaluation.run_name != self.name or self.tracking.run_name != self.name:
            raise ValueError("Registry run names must match the pipeline name.")
        return self


class PipelineRegistry(StrictModel):
    """Deterministic registry of immutable retrieval pipeline versions."""

    schema_version: Literal[1] = PIPELINE_REGISTRY_SCHEMA_VERSION
    registry_name: str = Field(min_length=1)
    aliases: PipelineAliases
    pipelines: list[PipelineRegistryEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_registry(self):
        ids = [pipeline.pipeline_id for pipeline in self.pipelines]
        if len(ids) != len(set(ids)):
            raise ValueError("Pipeline registry IDs must be unique.")
        names = [pipeline.name for pipeline in self.pipelines]
        if len(names) != len(set(names)):
            raise ValueError("Pipeline registry names must be unique within one registry snapshot.")
        config_paths = [pipeline.config.path for pipeline in self.pipelines]
        if len(config_paths) != len(set(config_paths)):
            raise ValueError("Pipeline registry config paths must be unique.")
        pipeline_types = {pipeline.pipeline_type for pipeline in self.pipelines}
        if pipeline_types != set(PIPELINE_TYPES):
            missing = sorted(set(PIPELINE_TYPES) - pipeline_types)
            unexpected = sorted(pipeline_types - set(PIPELINE_TYPES))
            raise ValueError(f"Registry must contain dense, bm25, hybrid, and reranked pipelines; missing={missing}, unexpected={unexpected}.")

        entries = {pipeline.pipeline_id: pipeline for pipeline in self.pipelines}
        alias_targets = self.aliases.model_dump()
        for alias, target in alias_targets.items():
            if target not in entries:
                raise ValueError(f"Pipeline alias {alias!r} points to unknown pipeline ID {target!r}.")
            if entries[target].status in ("rejected", "retired", "draft"):
                raise ValueError(f"Pipeline alias {alias!r} cannot point to status {entries[target].status!r}.")
        for alias in ("baseline", "production"):
            target = alias_targets[alias]
            if entries[target].status != "approved":
                raise ValueError(f"Pipeline alias {alias!r} must point to an approved version.")
        candidate = entries[alias_targets["candidate"]]
        if candidate.status not in ("evaluated", "approved"):
            raise ValueError("Pipeline alias 'candidate' must point to an evaluated or approved version.")
        return self


def resolve_project_path(path, project_root):
    path = Path(path)
    return path if path.is_absolute() else (Path(project_root) / path).resolve()


def _relative_project_path(path, project_root):
    path = Path(path).resolve()
    project_root = Path(project_root).resolve()
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError as error:
        raise ValueError(f"Registry artifacts must be inside the project root: {path}") from error


def _load_yaml_mapping(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{label} contains invalid YAML: {path}") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{label} must contain a YAML mapping: {path}")
    return payload


def _load_json_mapping(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} contains invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def load_pipeline_registry_config(config_path=DEFAULT_PIPELINE_REGISTRY_CONFIG_PATH, project_root=None):
    """Load registry-build configuration and resolve its source/output paths."""
    config_path = Path(config_path)
    payload = _load_yaml_mapping(config_path, "Pipeline registry config")
    config = PipelineRegistryConfig.model_validate(payload)
    project_root = Path(project_root or Path.cwd()).resolve()
    return config.model_copy(
        update={
            "catalog_path": resolve_project_path(config.catalog_path, project_root),
            "comparison_path": resolve_project_path(config.comparison_path, project_root),
            "output_path": resolve_project_path(config.output_path, project_root),
        }
    )


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        while block := input_file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _finite_metric(mapping, key, label):
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Registry comparison is missing finite metric {label}.")
    return float(value)


def _comparison_metrics(comparison, comparison_key, latency_key):
    depth = comparison.get("comparison_depth")
    if depth != 5:
        raise ValueError("Pipeline registry requires the common Day 27 comparison depth of 5.")
    metrics = comparison.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Pipeline registry comparison metrics must be a mapping.")
    latency = comparison.get("latency_ms", {}).get(latency_key)
    if not isinstance(latency, dict):
        raise ValueError(f"Pipeline registry comparison is missing latency for {latency_key}.")
    return RegistryEvaluationMetrics(
        comparison_depth=depth,
        mrr_at_5=_finite_metric(metrics.get("mrr_at_depth", {}), comparison_key, f"mrr_at_depth.{comparison_key}"),
        recall_at_1=_finite_metric(metrics.get("recall_at_k", {}).get("1", {}), comparison_key, f"recall_at_k.1.{comparison_key}"),
        recall_at_3=_finite_metric(metrics.get("recall_at_k", {}).get("3", {}), comparison_key, f"recall_at_k.3.{comparison_key}"),
        recall_at_5=_finite_metric(metrics.get("recall_at_k", {}).get("5", {}), comparison_key, f"recall_at_k.5.{comparison_key}"),
        hit_rate_at_1=_finite_metric(metrics.get("hit_rate_at_k", {}).get("1", {}), comparison_key, f"hit_rate_at_k.1.{comparison_key}"),
        hit_rate_at_3=_finite_metric(metrics.get("hit_rate_at_k", {}).get("3", {}), comparison_key, f"hit_rate_at_k.3.{comparison_key}"),
        hit_rate_at_5=_finite_metric(metrics.get("hit_rate_at_k", {}).get("5", {}), comparison_key, f"hit_rate_at_k.5.{comparison_key}"),
        ndcg_at_1=_finite_metric(metrics.get("ndcg_at_k", {}).get("1", {}), comparison_key, f"ndcg_at_k.1.{comparison_key}"),
        ndcg_at_3=_finite_metric(metrics.get("ndcg_at_k", {}).get("3", {}), comparison_key, f"ndcg_at_k.3.{comparison_key}"),
        ndcg_at_5=_finite_metric(metrics.get("ndcg_at_k", {}).get("5", {}), comparison_key, f"ndcg_at_k.5.{comparison_key}"),
        average_latency_ms=_finite_metric(latency, "average", f"latency_ms.{latency_key}.average"),
    )


def build_pipeline_registry(config, project_root):
    """Build one validated registry snapshot from configs and evaluation evidence."""
    from ragops.tracking.mlflow import load_mlflow_config, prepare_configured_runs

    project_root = Path(project_root).resolve()
    tracking_config = load_mlflow_config(config.catalog_path, project_root=project_root)
    prepared_runs = prepare_configured_runs(tracking_config, project_root)
    configured_runs = {run.pipeline: run for run in tracking_config.runs}
    comparison = _load_json_mapping(config.comparison_path, "Common-depth pipeline comparison")
    if comparison.get("decision", {}).get("primary_metric") != "mrr_at_5":
        raise ValueError("Pipeline registry requires mrr_at_5 as the common comparison primary metric.")
    metric_scope = comparison.get("metric_scope")
    if not isinstance(metric_scope, str) or not metric_scope.strip():
        raise ValueError("Pipeline registry comparison must describe its metric scope.")

    comparison_keys = {
        "dense": ("dense", "dense_baseline"),
        "bm25": ("bm25", "bm25_baseline"),
        "hybrid": ("hybrid", "hybrid_baseline"),
        "reranked": ("reranked", "reranked"),
    }
    entries = []
    for prepared in prepared_runs:
        pipeline_type = prepared["pipeline"]
        run_config = configured_runs[pipeline_type]
        raw_pipeline_config = _load_yaml_mapping(run_config.config_path, f"{pipeline_type} pipeline config")
        if "version" not in raw_pipeline_config or "status" not in raw_pipeline_config:
            raise ValueError(f"Registered pipeline config must explicitly declare version and status: {run_config.config_path}")
        metadata = PipelineVersionMetadata.model_validate(
            {"version": raw_pipeline_config["version"], "status": raw_pipeline_config["status"]}
        )
        name = raw_pipeline_config.get("name")
        if name != prepared["run_name"]:
            raise ValueError(f"Pipeline config name does not match validated evaluation run: {run_config.config_path}")
        retriever_interface = raw_pipeline_config.get("retriever_interface")
        if retriever_interface != "common_v1":
            raise ValueError(f"Registered pipeline must use retriever_interface common_v1: {run_config.config_path}")
        comparison_key, latency_key = comparison_keys[pipeline_type]
        evidence_metrics = _comparison_metrics(comparison, comparison_key, latency_key)
        entries.append(
            PipelineRegistryEntry(
                pipeline_id=f"{name}@{metadata.version}",
                name=name,
                version=metadata.version,
                status=metadata.status,
                pipeline_type=pipeline_type,
                retriever_interface=retriever_interface,
                config=RegistryConfigArtifact(
                    path=_relative_project_path(run_config.config_path, project_root),
                    sha256=_sha256_file(run_config.config_path),
                ),
                evaluation=RegistryEvaluationEvidence(
                    run_name=prepared["run_name"],
                    report_path=_relative_project_path(run_config.report_path, project_root),
                    comparison_path=_relative_project_path(config.comparison_path, project_root),
                    benchmark_path=_relative_project_path(run_config.benchmark_path, project_root),
                    question_count=int(prepared["metrics"]["question_count"]),
                    metric_scope=metric_scope.strip(),
                    metrics=evidence_metrics,
                    artifact_digest=prepared["artifact_digest"],
                ),
                tracking=RegistryTrackingReference(
                    experiment_name=tracking_config.experiment_name,
                    run_name=prepared["run_name"],
                ),
            )
        )
    return PipelineRegistry(
        registry_name=config.registry_name,
        aliases=config.aliases,
        pipelines=entries,
    )


def registry_json(registry):
    """Render a stable, human-readable registry JSON document."""
    return json.dumps(registry.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_pipeline_registry(registry, output_path, overwrite=False):
    """Atomically write a registry snapshot, protecting existing files by default."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Pipeline registry already exists: {output_path}. Pass overwrite=True to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(registry_json(registry), encoding="utf-8")
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def load_pipeline_registry(path=DEFAULT_PIPELINE_REGISTRY_PATH):
    """Load and validate one generated registry JSON artifact."""
    path = Path(path)
    return PipelineRegistry.model_validate(_load_json_mapping(path, "Pipeline registry artifact"))


def validate_registry_matches_sources(config, project_root):
    """Require the checked-in registry to exactly match its current source evidence."""
    expected = build_pipeline_registry(config, project_root)
    actual = load_pipeline_registry(config.output_path)
    if actual.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise ValueError(
            f"Pipeline registry is stale: {config.output_path}. Rebuild it after reviewing version, status, and alias changes."
        )
    return actual
