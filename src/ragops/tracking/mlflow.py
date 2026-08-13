import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MLFLOW_EXPERIMENT = "ragops-retrieval"
DEFAULT_MLFLOW_CONFIG_PATH = Path("configs/mlflow.yaml")
REQUIRED_RETRIEVAL_PIPELINES = ("dense", "bm25", "hybrid", "reranked")
METRIC_SUMMARY_FIELDS = ("average", "minimum", "maximum", "total")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrackedRetrievalRunConfig(StrictModel):
    """Artifacts and pipeline identity for one retrieval run."""

    pipeline: Literal["dense", "bm25", "hybrid", "reranked"]
    config_path: Path
    report_path: Path
    csv_path: Path
    comparison_path: Path | None = None
    benchmark_path: Path

    @field_validator("config_path", "report_path", "csv_path", "comparison_path", "benchmark_path", mode="before")
    @classmethod
    def validate_path(cls, value):
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            raise ValueError("MLflow artifact paths must not be empty.")
        return value


class MLflowTrackingConfig(StrictModel):
    """Central MLflow connection and retrieval-run import configuration."""

    schema_version: Literal[1] = 1
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI
    experiment_name: str = Field(default=DEFAULT_MLFLOW_EXPERIMENT, min_length=1)
    runs: list[TrackedRetrievalRunConfig] = Field(min_length=1)

    @field_validator("tracking_uri", "experiment_name")
    @classmethod
    def clean_required_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("MLflow text settings must not be empty.")
        return value.rstrip("/") if "://" in value else value

    @model_validator(mode="after")
    def validate_unique_pipelines(self):
        pipelines = [run.pipeline for run in self.runs]
        if len(pipelines) != len(set(pipelines)):
            raise ValueError("MLflow retrieval pipelines must be unique.")
        return self


def resolve_project_path(path, project_root):
    path = Path(path)
    return path if path.is_absolute() else (Path(project_root) / path).resolve()


def load_mlflow_config(config_path=DEFAULT_MLFLOW_CONFIG_PATH, project_root=None):
    """Load strict MLflow YAML and resolve every tracked artifact path."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"MLflow config does not exist: {config_path}")
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
    if not isinstance(raw_config, dict) or not raw_config:
        raise ValueError(f"MLflow config must contain a YAML mapping: {config_path}")

    config = MLflowTrackingConfig.model_validate(raw_config)
    project_root = Path(project_root or Path.cwd()).resolve()
    resolved_runs = []
    for run in config.runs:
        updates = {
            "config_path": resolve_project_path(run.config_path, project_root),
            "report_path": resolve_project_path(run.report_path, project_root),
            "csv_path": resolve_project_path(run.csv_path, project_root),
            "benchmark_path": resolve_project_path(run.benchmark_path, project_root),
        }
        if run.comparison_path is not None:
            updates["comparison_path"] = resolve_project_path(run.comparison_path, project_root)
        resolved_runs.append(run.model_copy(update=updates))
    return config.model_copy(update={"runs": resolved_runs})


def configured_tracking_uri(config):
    """Return the environment override or configured MLflow tracking URI."""
    environment_uri = os.getenv("MLFLOW_TRACKING_URI")
    if environment_uri is not None and environment_uri.strip():
        return environment_uri.strip().rstrip("/")
    return config.tracking_uri


def require_complete_retrieval_suite(config):
    """Require exactly the four retrieval pipelines needed for Day 29."""
    pipelines = {run.pipeline for run in config.runs}
    required = set(REQUIRED_RETRIEVAL_PIPELINES)
    if pipelines != required:
        missing = sorted(required - pipelines)
        unexpected = sorted(pipelines - required)
        raise ValueError(f"MLflow config must define dense, bm25, hybrid, and reranked runs; missing={missing}, unexpected={unexpected}.")
    return config


def _load_json(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _load_pipeline_config(run, project_root):
    if run.pipeline == "dense":
        from ragops.evaluation.runner import load_evaluation_config

        return load_evaluation_config(run.config_path, project_root=project_root)
    if run.pipeline == "bm25":
        from ragops.retrieval.bm25 import load_bm25_config

        return load_bm25_config(run.config_path, project_root=project_root)
    if run.pipeline == "hybrid":
        from ragops.retrieval.hybrid import load_hybrid_config

        return load_hybrid_config(run.config_path, project_root=project_root)
    from ragops.reranking.cross_encoder import load_hybrid_rerank_config

    return load_hybrid_rerank_config(run.config_path, project_root=project_root)


def pipeline_type_from_configuration(configuration):
    """Identify one supported pipeline from its effective configuration."""
    if not isinstance(configuration, Mapping):
        raise ValueError("Evaluation configuration must be a mapping.")
    if isinstance(configuration.get("reranker"), Mapping) and configuration["reranker"].get("type") == "cross_encoder":
        return "reranked"
    if isinstance(configuration.get("fusion"), Mapping) and configuration["fusion"].get("type") == "rrf":
        return "hybrid"
    retriever = configuration.get("retriever")
    retriever_type = retriever.get("type") if isinstance(retriever, Mapping) else None
    if retriever_type in ("dense", "bm25"):
        return retriever_type
    raise ValueError("Evaluation configuration does not describe a supported retrieval pipeline.")


def _pipeline_sections(pipeline):
    if pipeline in ("dense", "bm25"):
        return ("retriever",)
    if pipeline == "hybrid":
        return ("dense", "bm25", "fusion")
    return ("dense", "bm25", "fusion", "reranker")


def _validate_pipeline_settings(pipeline, configured, recorded):
    for section in _pipeline_sections(pipeline):
        if configured.get(section) != recorded.get(section):
            raise ValueError(f"Recorded {pipeline} report does not match the current {section} configuration.")


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"MLflow metric {name} must be a finite number.")
    return float(value)


def _rank_metrics(payload, prefix=""):
    if not isinstance(payload, Mapping):
        raise ValueError(f"{prefix or 'retrieval'} metrics must be a mapping.")
    metrics = {}
    for name in ("mrr", "question_count", "depth"):
        if name in payload:
            metrics[f"{prefix}{name}"] = _finite_number(payload[name], f"{prefix}{name}")
    for source_name, target_name in (
        ("recall_at_k", "recall_at"),
        ("hit_rate_at_k", "hit_rate_at"),
        ("ndcg_at_k", "ndcg_at"),
    ):
        values = payload.get(source_name)
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise ValueError(f"MLflow metric group {source_name} must be a mapping.")
        for k, value in values.items():
            key = f"{prefix}{target_name}_{k}"
            metrics[key] = _finite_number(value, key)
    return metrics


def _summary_metrics(summary, prefix):
    if not isinstance(summary, Mapping):
        raise ValueError(f"MLflow latency summary {prefix} must be a mapping.")
    metrics = {}
    for statistic in METRIC_SUMMARY_FIELDS:
        if statistic in summary:
            key = f"{prefix}_{statistic}_ms"
            metrics[key] = _finite_number(summary[statistic], key)
    return metrics


def extract_mlflow_metrics(report):
    """Flatten retrieval quality and latency summaries into MLflow metrics."""
    metrics = _rank_metrics(report.get("metrics"))
    if "mrr" not in metrics or "question_count" not in metrics:
        raise ValueError("Evaluation report must contain finite MRR and question_count metrics.")
    if report.get("pre_rerank_metrics") is not None:
        metrics.update(_rank_metrics(report["pre_rerank_metrics"], prefix="pre_rerank_"))
    if report.get("latency_ms") is not None:
        metrics.update(_summary_metrics(report["latency_ms"], "latency"))
    if report.get("latency_after_first_ms") is not None:
        metrics.update(_summary_metrics(report["latency_after_first_ms"], "latency_after_first"))
    for field_name, suffix in (("component_latency_ms", ""), ("component_latency_after_first_ms", "_after_first")):
        components = report.get(field_name)
        if components is None:
            continue
        if not isinstance(components, Mapping):
            raise ValueError(f"{field_name} must be a mapping.")
        for component, summary in components.items():
            metrics.update(_summary_metrics(summary, f"{component}_latency{suffix}"))
    model = report.get("model")
    if isinstance(model, Mapping) and model.get("load_latency_ms") is not None:
        metrics["model_load_ms"] = _finite_number(model["load_latency_ms"], "model_load_ms")
    return dict(sorted(metrics.items()))


def _parameter_value(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def flatten_mlflow_params(configuration, prefix=""):
    """Flatten nested configuration into deterministic MLflow parameters."""
    if not isinstance(configuration, Mapping):
        raise ValueError("MLflow parameter configuration must be a mapping.")
    flattened = {}
    for key in sorted(configuration):
        value = configuration[key]
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_mlflow_params(value, prefix=name))
        else:
            rendered = _parameter_value(value)
            if len(name) > 250:
                raise ValueError(f"MLflow parameter name exceeds 250 characters: {name}")
            if len(rendered) > 6000:
                raise ValueError(f"MLflow parameter value exceeds 6000 characters: {name}")
            flattened[name] = rendered
    return flattened


def _validate_report(report, pipeline, expected_run_name=None):
    run_name = report.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("Evaluation report must contain a non-empty run_name.")
    if expected_run_name is not None and run_name != expected_run_name:
        raise ValueError(f"Evaluation report run_name {run_name!r} does not match config name {expected_run_name!r}.")
    if report.get("schema_version") != 1:
        raise ValueError("Evaluation report schema_version must be 1.")
    configuration = report.get("configuration")
    if pipeline_type_from_configuration(configuration) != pipeline:
        raise ValueError(f"Evaluation report does not describe the configured {pipeline} pipeline.")
    if configuration.get("name") not in (None, run_name):
        raise ValueError("Evaluation report configuration name does not match run_name.")

    metrics = extract_mlflow_metrics(report)
    questions = report.get("questions")
    if not isinstance(questions, Sequence) or isinstance(questions, (str, bytes)) or not questions:
        raise ValueError("Evaluation report must contain question results.")
    question_ids = [question.get("question_id") if isinstance(question, Mapping) else None for question in questions]
    if any(not isinstance(question_id, str) or not question_id.strip() for question_id in question_ids):
        raise ValueError("Evaluation report contains an invalid question ID.")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Evaluation report contains duplicate question IDs.")
    if int(metrics["question_count"]) != len(question_ids):
        raise ValueError("Evaluation metric question_count does not match question results.")
    return question_ids, metrics


def _validate_csv(csv_path, run_name, report_question_ids):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Evaluation CSV does not exist: {csv_path}")
    with csv_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"Evaluation CSV contains no rows: {csv_path}")
    if any(row.get("run_name") != run_name for row in rows):
        raise ValueError(f"Evaluation CSV run names do not match {run_name!r}.")
    csv_question_ids = [row.get("question_id") for row in rows]
    if len(csv_question_ids) != len(set(csv_question_ids)):
        raise ValueError("Evaluation CSV contains duplicate question IDs.")
    if csv_question_ids != report_question_ids:
        raise ValueError("Evaluation CSV question order does not match the JSON report.")


def _validate_artifact(path, label, suffixes):
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if path.suffix.lower() not in suffixes:
        raise ValueError(f"{label} must use one of {sorted(suffixes)}: {path}")


def _artifact_digest(artifacts, run_source):
    digest = hashlib.sha256()
    digest.update(run_source.encode("utf-8"))
    for path, artifact_directory in sorted(artifacts, key=lambda item: (item[1], str(item[0]))):
        digest.update(artifact_directory.encode("utf-8"))
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as input_file:
            while block := input_file.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def prepare_retrieval_run(
    pipeline,
    config_path,
    report_path,
    csv_path,
    comparison_path=None,
    benchmark_path=None,
    run_source="live_evaluation",
    report=None,
    parameter_configuration=None,
):
    """Validate one run and prepare deterministic MLflow payloads."""
    config_path = Path(config_path)
    report_path = Path(report_path)
    csv_path = Path(csv_path)
    comparison_path = Path(comparison_path) if comparison_path is not None else None
    benchmark_path = Path(benchmark_path) if benchmark_path is not None else None
    _validate_artifact(config_path, "Pipeline config", {".yaml", ".yml"})
    _validate_artifact(report_path, "Evaluation JSON", {".json"})
    _validate_artifact(csv_path, "Evaluation CSV", {".csv"})
    if comparison_path is not None:
        _validate_artifact(comparison_path, "Comparison JSON", {".json"})
    if benchmark_path is not None:
        _validate_artifact(benchmark_path, "Benchmark report", {".md"})

    report = report or _load_json(report_path, "Evaluation report")
    question_ids, metrics = _validate_report(report, pipeline)
    _validate_csv(csv_path, report["run_name"], question_ids)
    parameter_configuration = parameter_configuration or report["configuration"]
    if parameter_configuration.get("retriever_interface") is None:
        parameter_configuration = dict(parameter_configuration)
        parameter_configuration["retriever_interface"] = "common_v1"
    params = flatten_mlflow_params(parameter_configuration)

    artifacts = [(config_path, "config"), (report_path, "evaluation"), (csv_path, "evaluation")]
    if comparison_path is not None:
        artifacts.append((comparison_path, "comparison"))
    if benchmark_path is not None:
        artifacts.append((benchmark_path, "comparison"))
    artifact_digest = _artifact_digest(artifacts, run_source)
    tags = {
        "ragops_pipeline_type": pipeline,
        "ragops_run_source": run_source,
        "ragops_retriever_interface": params["retriever_interface"],
        "ragops_report_schema_version": str(report["schema_version"]),
        "ragops_artifact_validation": "passed",
        "ragops_artifact_digest": artifact_digest,
    }
    return {
        "run_name": report["run_name"],
        "pipeline": pipeline,
        "report": report,
        "params": params,
        "metrics": metrics,
        "tags": tags,
        "artifacts": artifacts,
        "artifact_digest": artifact_digest,
    }


def prepare_configured_run(run, project_root):
    """Strictly validate a configured historical run before importing it."""
    project_root = Path(project_root).resolve()
    pipeline_config = _load_pipeline_config(run, project_root)
    report = _load_json(run.report_path, "Evaluation report")
    _validate_report(report, run.pipeline, expected_run_name=pipeline_config.name)
    recorded_configuration = report["configuration"]
    configured = pipeline_config.model_dump(mode="json")
    _validate_pipeline_settings(run.pipeline, configured, recorded_configuration)
    if configured["evaluation"]["k_values"] != report["metrics"]["k_values"]:
        raise ValueError(f"Recorded {run.pipeline} report does not use the configured metric cutoffs.")
    return prepare_retrieval_run(
        run.pipeline,
        run.config_path,
        run.report_path,
        run.csv_path,
        comparison_path=run.comparison_path,
        benchmark_path=run.benchmark_path,
        run_source="validated_artifact_import",
        report=report,
        parameter_configuration=configured,
    )


def prepare_configured_runs(config, project_root):
    """Prepare the complete configured suite in stable pipeline order."""
    require_complete_retrieval_suite(config)
    runs_by_pipeline = {run.pipeline: run for run in config.runs}
    return [prepare_configured_run(runs_by_pipeline[pipeline], project_root) for pipeline in REQUIRED_RETRIEVAL_PIPELINES]


def _mlflow_client(tracking_uri):
    from mlflow import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri)


def _get_or_create_experiment(client, experiment_name):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(experiment_name, tags={"ragops_project": "ragops-control-plane", "ragops_scope": "retrieval"})


def _existing_run(client, experiment_id, artifact_digest):
    matches = client.search_runs(
        [experiment_id],
        filter_string=f"tags.ragops_artifact_digest = '{artifact_digest}'",
        max_results=100,
        order_by=["attributes.start_time DESC"],
    )
    return next((run for run in matches if run.info.status == "FINISHED"), None)


def log_prepared_run(prepared, config, client=None, force=False):
    """Create one fully populated MLflow run, or reuse an identical finished run."""
    tracking_uri = configured_tracking_uri(config)
    try:
        client = client or _mlflow_client(tracking_uri)
        experiment_id = _get_or_create_experiment(client, config.experiment_name)
        if not force:
            existing = _existing_run(client, experiment_id, prepared["artifact_digest"])
            if existing is not None:
                return {
                    "created": False,
                    "run_id": existing.info.run_id,
                    "run_name": prepared["run_name"],
                    "experiment_id": experiment_id,
                    "tracking_uri": tracking_uri,
                }

        run = client.create_run(experiment_id, tags=prepared["tags"], run_name=prepared["run_name"])
        run_id = run.info.run_id
        try:
            for key, value in prepared["params"].items():
                client.log_param(run_id, key, value)
            for key, value in prepared["metrics"].items():
                client.log_metric(run_id, key, value)
            client.log_dict(run_id, prepared["report"]["configuration"], "config/effective_configuration.json")
            for path, artifact_directory in prepared["artifacts"]:
                client.log_artifact(run_id, str(path), artifact_path=artifact_directory)
            client.set_terminated(run_id, status="FINISHED")
        except Exception:
            try:
                client.set_terminated(run_id, status="FAILED")
            except Exception:
                pass
            raise
        return {
            "created": True,
            "run_id": run_id,
            "run_name": prepared["run_name"],
            "experiment_id": experiment_id,
            "tracking_uri": tracking_uri,
        }
    except Exception as error:
        raise RuntimeError(f"Unable to log retrieval run {prepared['run_name']!r} to MLflow at {tracking_uri}: {error}") from error


def log_prepared_runs(prepared_runs, config, client=None, force=False):
    """Log a stable sequence of prepared retrieval runs."""
    return [log_prepared_run(prepared, config, client=client, force=force) for prepared in prepared_runs]


def verify_prepared_runs(prepared_runs, config, client=None):
    """Verify that every prepared run is complete and visible in MLflow."""
    tracking_uri = configured_tracking_uri(config)
    try:
        client = client or _mlflow_client(tracking_uri)
        experiment = client.get_experiment_by_name(config.experiment_name)
        if experiment is None:
            raise ValueError(f"MLflow experiment does not exist: {config.experiment_name}")
        verified = []
        for prepared in prepared_runs:
            run = _existing_run(client, experiment.experiment_id, prepared["artifact_digest"])
            if run is None:
                raise ValueError(f"No finished MLflow run matches {prepared['run_name']!r} and its artifact digest.")
            if run.info.run_name != prepared["run_name"]:
                raise ValueError(f"MLflow run name mismatch for {prepared['pipeline']}.")

            missing_params = sorted(key for key, value in prepared["params"].items() if run.data.params.get(key) != value)
            if missing_params:
                raise ValueError(f"MLflow run {prepared['run_name']!r} has missing or mismatched parameters: {missing_params}.")

            mismatched_metrics = sorted(
                key
                for key, value in prepared["metrics"].items()
                if key not in run.data.metrics
                or not math.isclose(float(run.data.metrics[key]), value, rel_tol=1e-12, abs_tol=1e-12)
            )
            if mismatched_metrics:
                raise ValueError(f"MLflow run {prepared['run_name']!r} has missing or mismatched metrics: {mismatched_metrics}.")

            mismatched_tags = sorted(key for key, value in prepared["tags"].items() if run.data.tags.get(key) != value)
            if mismatched_tags:
                raise ValueError(f"MLflow run {prepared['run_name']!r} has missing or mismatched tags: {mismatched_tags}.")

            expected_artifacts = {"config": {"effective_configuration.json"}}
            for path, directory in prepared["artifacts"]:
                expected_artifacts.setdefault(directory, set()).add(path.name)
            for directory, expected_names in sorted(expected_artifacts.items()):
                artifact_names = {Path(artifact.path).name for artifact in client.list_artifacts(run.info.run_id, directory)}
                missing_artifacts = sorted(expected_names - artifact_names)
                if missing_artifacts:
                    raise ValueError(
                        f"MLflow run {prepared['run_name']!r} is missing tracked artifacts in {directory!r}: {missing_artifacts}."
                    )
            verified.append(run)
        return verified
    except Exception as error:
        raise RuntimeError(f"Unable to verify retrieval runs in MLflow at {tracking_uri}: {error}") from error
