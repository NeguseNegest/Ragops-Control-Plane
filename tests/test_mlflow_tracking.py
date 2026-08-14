import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

import scripts.log_retrieval_runs as tracking_cli
from ragops.tracking import mlflow as tracking
from ragops.tracking.mlflow import (
    MLflowTrackingConfig,
    configured_tracking_uri,
    extract_mlflow_metrics,
    flatten_mlflow_params,
    load_mlflow_config,
    log_prepared_run,
    prepare_configured_runs,
    prepare_retrieval_run,
    verify_prepared_runs,
)


class FakeRun:
    def __init__(self, run_id, run_name, tags):
        self.info = SimpleNamespace(run_id=run_id, run_name=run_name, status="RUNNING")
        self.data = SimpleNamespace(params={}, metrics={}, tags=dict(tags))


class FakeMLflowClient:
    def __init__(self, fail_artifact=False):
        self.experiments = {}
        self.runs = []
        self.artifacts = {}
        self.logged_dicts = {}
        self.fail_artifact = fail_artifact

    def get_experiment_by_name(self, name):
        return self.experiments.get(name)

    def create_experiment(self, name, tags=None):
        experiment_id = str(len(self.experiments) + 1)
        self.experiments[name] = SimpleNamespace(experiment_id=experiment_id, tags=tags or {})
        return experiment_id

    def search_runs(self, experiment_ids, filter_string="", **kwargs):
        digest = filter_string.split("'", 2)[1] if "'" in filter_string else None
        return [
            run
            for run in reversed(self.runs)
            if run.info.status == "FINISHED" and (digest is None or run.data.tags.get("ragops_artifact_digest") == digest)
        ]

    def create_run(self, experiment_id, tags=None, run_name=None):
        run = FakeRun(f"run-{len(self.runs) + 1}", run_name, tags or {})
        self.runs.append(run)
        return run

    def log_param(self, run_id, key, value):
        self._run(run_id).data.params[key] = value

    def log_metric(self, run_id, key, value):
        self._run(run_id).data.metrics[key] = value

    def log_dict(self, run_id, dictionary, artifact_file):
        self.logged_dicts[(run_id, artifact_file)] = dictionary
        self.artifacts.setdefault(run_id, []).append(SimpleNamespace(path=artifact_file))

    def log_artifact(self, run_id, local_path, artifact_path=None):
        if self.fail_artifact:
            raise OSError("artifact store unavailable")
        path = f"{artifact_path}/{Path(local_path).name}" if artifact_path else Path(local_path).name
        self.artifacts.setdefault(run_id, []).append(SimpleNamespace(path=path))

    def set_terminated(self, run_id, status=None):
        self._run(run_id).info.status = status

    def list_artifacts(self, run_id, path=None):
        prefix = f"{path}/" if path else ""
        return [artifact for artifact in self.artifacts.get(run_id, []) if artifact.path.startswith(prefix)]

    def _run(self, run_id):
        return next(run for run in self.runs if run.info.run_id == run_id)


def make_report(run_name="dense_test"):
    return {
        "schema_version": 1,
        "run_name": run_name,
        "configuration": {
            "name": run_name,
            "retriever_interface": "common_v1",
            "retriever": {"type": "dense", "top_k": 2, "collection_name": "chunks", "embedding_model": "model"},
            "evaluation": {"k_values": [1, 2]},
            "output": {"directory": "reports"},
        },
        "metrics": {
            "question_count": 2,
            "k_values": [1, 2],
            "mrr": 0.75,
            "recall_at_k": {"1": 0.5, "2": 1.0},
            "hit_rate_at_k": {"1": 0.5, "2": 1.0},
            "ndcg_at_k": {"1": 0.5, "2": 0.815},
        },
        "latency_ms": {"total": 30.0, "average": 15.0, "minimum": 10.0, "maximum": 20.0},
        "questions": [
            {"question_id": "q1"},
            {"question_id": "q2"},
        ],
    }


def write_run_artifacts(tmp_path, report=None):
    report = report or make_report()
    config_path = tmp_path / "dense.yaml"
    report_path = tmp_path / "dense.json"
    csv_path = tmp_path / "dense.csv"
    benchmark_path = tmp_path / "benchmark.md"
    config_path.write_text(yaml.safe_dump(report["configuration"]), encoding="utf-8")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    csv_path.write_text(
        "run_name,question_id\n" + "\n".join(f"{report['run_name']},{question_id}" for question_id in ("q1", "q2")) + "\n",
        encoding="utf-8",
    )
    benchmark_path.write_text("# Benchmark\n", encoding="utf-8")
    return config_path, report_path, csv_path, benchmark_path


def make_tracking_config(tmp_path):
    return MLflowTrackingConfig.model_validate(
        {
            "tracking_uri": "http://configured:5000/",
            "experiment_name": "retrieval-tests",
            "runs": [
                {
                    "pipeline": "dense",
                    "config_path": tmp_path / "dense.yaml",
                    "report_path": tmp_path / "dense.json",
                    "csv_path": tmp_path / "dense.csv",
                    "benchmark_path": tmp_path / "benchmark.md",
                }
            ],
        }
    )


def test_checked_in_mlflow_config_prepares_all_four_validated_runs():
    project_root = Path(__file__).resolve().parents[1]
    config = load_mlflow_config(project_root / "configs/mlflow.yaml", project_root=project_root)

    prepared = prepare_configured_runs(config, project_root)

    assert [run["pipeline"] for run in prepared] == ["dense", "bm25", "hybrid", "reranked"]
    assert [run["run_name"] for run in prepared] == ["dense_baseline", "bm25_baseline", "hybrid_rrf", "hybrid_rrf_cross_encoder"]
    assert all(any(path.suffix == ".csv" for path, _ in run["artifacts"]) for run in prepared)
    assert prepared[-1]["metrics"]["mrr"] == pytest.approx(0.6888888888888889)
    assert prepared[-1]["metrics"]["reranker_latency_after_first_average_ms"] == pytest.approx(4274.931078750259)
    assert prepared[-1]["tags"]["ragops_pipeline_id"] == "hybrid_rrf_cross_encoder@1.0.0"


def test_tracking_uri_uses_environment_override(monkeypatch, tmp_path):
    config = make_tracking_config(tmp_path)

    assert configured_tracking_uri(config) == "http://configured:5000"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", " http://environment:5000/ ")
    assert configured_tracking_uri(config) == "http://environment:5000"


def test_mlflow_config_rejects_duplicate_pipeline_entries(tmp_path):
    run = {
        "pipeline": "dense",
        "config_path": tmp_path / "dense.yaml",
        "report_path": tmp_path / "dense.json",
        "csv_path": tmp_path / "dense.csv",
        "benchmark_path": tmp_path / "benchmark.md",
    }

    with pytest.raises(ValidationError, match="unique"):
        MLflowTrackingConfig.model_validate({"runs": [run, run]})


def test_flatten_params_is_deterministic_and_preserves_lists_and_nulls():
    flattened = flatten_mlflow_params({"z": None, "dense": {"top_k": 25, "enabled": True}, "cutoffs": [1, 3, 5]})

    assert list(flattened) == ["cutoffs", "dense.enabled", "dense.top_k", "z"]
    assert flattened == {"cutoffs": "[1,3,5]", "dense.enabled": "true", "dense.top_k": "25", "z": "null"}


def test_extract_metrics_includes_quality_component_and_model_latency():
    report = make_report()
    report["pre_rerank_metrics"] = {"question_count": 2, "mrr": 0.5, "hit_rate_at_k": {"2": 0.5}}
    report["component_latency_ms"] = {"dense": {"average": 3.0, "minimum": 2.0, "maximum": 4.0, "total": 6.0}}
    report["model"] = {"load_latency_ms": 100.0}

    metrics = extract_mlflow_metrics(report)

    assert metrics["mrr"] == 0.75
    assert metrics["hit_rate_at_2"] == 1.0
    assert metrics["pre_rerank_mrr"] == 0.5
    assert metrics["dense_latency_average_ms"] == 3.0
    assert metrics["model_load_ms"] == 100.0


def test_extract_metrics_rejects_non_finite_values():
    report = make_report()
    report["metrics"]["mrr"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        extract_mlflow_metrics(report)


def test_prepare_run_validates_csv_parity_and_builds_digest(tmp_path):
    config_path, report_path, csv_path, benchmark_path = write_run_artifacts(tmp_path)

    prepared = prepare_retrieval_run("dense", config_path, report_path, csv_path, benchmark_path=benchmark_path, run_source="test")

    assert prepared["run_name"] == "dense_test"
    assert prepared["params"]["retriever_interface"] == "common_v1"
    assert len(prepared["artifact_digest"]) == 64
    assert [directory for _, directory in prepared["artifacts"]] == ["config", "evaluation", "evaluation", "comparison"]

    csv_path.write_text("run_name,question_id\ndense_test,q2\ndense_test,q1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="order"):
        prepare_retrieval_run("dense", config_path, report_path, csv_path, benchmark_path=benchmark_path)


def test_log_run_records_params_metrics_artifacts_and_reuses_digest(tmp_path):
    config_path, report_path, csv_path, benchmark_path = write_run_artifacts(tmp_path)
    prepared = prepare_retrieval_run("dense", config_path, report_path, csv_path, benchmark_path=benchmark_path, run_source="test")
    config = make_tracking_config(tmp_path)
    client = FakeMLflowClient()

    first = log_prepared_run(prepared, config, client=client)
    second = log_prepared_run(prepared, config, client=client)
    verified = verify_prepared_runs([prepared], config, client=client)

    assert first["created"] is True
    assert second == {**first, "created": False}
    assert len(client.runs) == 1
    assert client.runs[0].info.status == "FINISHED"
    assert client.runs[0].data.params["retriever.top_k"] == "2"
    assert client.runs[0].data.metrics["mrr"] == 0.75
    assert {artifact.path for artifact in client.artifacts[first["run_id"]]} == {
        "comparison/benchmark.md",
        "config/dense.yaml",
        "config/effective_configuration.json",
        "evaluation/dense.csv",
        "evaluation/dense.json",
    }
    assert len(verified) == 1

    client.artifacts[first["run_id"]] = [
        artifact for artifact in client.artifacts[first["run_id"]] if artifact.path != "comparison/benchmark.md"
    ]
    with pytest.raises(RuntimeError, match="missing tracked artifacts"):
        verify_prepared_runs([prepared], config, client=client)


def test_log_run_marks_failed_when_artifact_upload_fails(tmp_path):
    config_path, report_path, csv_path, benchmark_path = write_run_artifacts(tmp_path)
    prepared = prepare_retrieval_run("dense", config_path, report_path, csv_path, benchmark_path=benchmark_path)
    client = FakeMLflowClient(fail_artifact=True)

    with pytest.raises(RuntimeError, match="artifact store unavailable"):
        log_prepared_run(prepared, make_tracking_config(tmp_path), client=client)

    assert client.runs[0].info.status == "FAILED"


def test_validation_only_cli_never_constructs_an_mlflow_client(monkeypatch, capsys):
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr("sys.argv", ["log_retrieval_runs.py", "--config", str(project_root / "configs/mlflow.yaml"), "--validate-only"])
    monkeypatch.setattr(tracking, "_mlflow_client", lambda uri: pytest.fail("validation-only must not contact MLflow"))

    tracking_cli.main()

    assert "Validated four retrieval runs without contacting MLflow" in capsys.readouterr().out
