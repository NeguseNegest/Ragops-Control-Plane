import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.evaluation.api_runner import configured_api_url, load_reference_report, run_api_evaluation  # noqa: E402
from ragops.evaluation.runner import load_evaluation_config, load_evaluation_labels, write_evaluation_artifacts  # noqa: E402
from ragops.tracing.store import DEFAULT_TRACE_DB_PATH, TraceStore  # noqa: E402
from ragops.tracking.mlflow import (  # noqa: E402
    DEFAULT_MLFLOW_CONFIG_PATH,
    configured_tracking_uri,
    load_mlflow_config,
    prepare_configured_runs,
    verify_prepared_runs,
)

DEFAULT_REFERENCE_REPORT = Path("reports/evaluations/dense_baseline.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval through the live POST /query API and verify Week 5 integration boundaries.")
    parser.add_argument("--config", type=Path, default=Path("configs/dense_baseline.yaml"), help="Dense retrieval configuration exposed by POST /query.")
    parser.add_argument("--api-url", help="FastAPI base URL; defaults to RAGOPS_API_URL or the local service.")
    parser.add_argument("--trace-db-path", type=Path, default=DEFAULT_TRACE_DB_PATH, help="SQLite database written by the API.")
    parser.add_argument("--reference-report", type=Path, default=DEFAULT_REFERENCE_REPORT, help="Offline evaluation report requiring exact ranking parity.")
    parser.add_argument("--mlflow-config", type=Path, default=DEFAULT_MLFLOW_CONFIG_PATH, help="MLflow suite whose live runs must verify.")
    parser.add_argument("--output-dir", type=Path, help="Optional API evaluation artifact directory.")
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="Per-request HTTP timeout.")
    parser.add_argument("--skip-trace-verification", action="store_true", help="Do not compare returned trace IDs with a local SQLite store.")
    parser.add_argument("--skip-reference-check", action="store_true", help="Do not require exact parity with the offline dense report.")
    parser.add_argument("--skip-mlflow-verification", action="store_true", help="Do not verify the four retrieval evidence runs in live MLflow.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing API evaluation JSON/CSV artifacts.")
    return parser.parse_args()


def project_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def print_progress(event):
    print(
        f"[{event['index']}/{event['total']}] {event['question_id']} "
        f"(service={event['latency_ms']:.1f} ms, client={event['client_latency_ms']:.1f} ms, trace={event['trace_id']})",
        flush=True,
    )


def verify_mlflow(config_path):
    config = load_mlflow_config(config_path, project_root=PROJECT_ROOT)
    prepared_runs = prepare_configured_runs(config, PROJECT_ROOT)
    verified_runs = verify_prepared_runs(prepared_runs, config)
    return {
        "experiment_name": config.experiment_name,
        "tracking_uri": configured_tracking_uri(config),
        "verified_run_count": len(verified_runs),
        "runs": [
            {
                "pipeline": prepared["pipeline"],
                "run_name": run.info.run_name,
                "run_id": run.info.run_id,
                "status": run.info.status,
                "artifact_digest": prepared["artifact_digest"],
            }
            for prepared, run in zip(prepared_runs, verified_runs, strict=True)
        ],
    }


def main():
    args = parse_args()
    config_path = project_path(args.config)
    config = load_evaluation_config(config_path, project_root=PROJECT_ROOT)
    if args.output_dir is not None:
        output = config.output.model_copy(update={"directory": project_path(args.output_dir)})
        config = config.model_copy(update={"output": output})
    labels = load_evaluation_labels(config)

    run_name = f"{config.name}_api"
    output_directory = config.output.directory
    expected_outputs = [output_directory / f"{run_name}.json", output_directory / f"{run_name}.csv"]
    existing_outputs = [path for path in expected_outputs if path.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(f"API evaluation artifacts already exist; pass --overwrite to replace them: {existing_outputs}")

    trace_store = None
    trace_db_path = project_path(args.trace_db_path)
    if not args.skip_trace_verification:
        if not trace_db_path.exists():
            raise FileNotFoundError(f"Trace database does not exist; start the API with this path first: {trace_db_path}")
        trace_store = TraceStore(trace_db_path)
        trace_store.validate_schema()

    reference_report = None
    reference_report_path = project_path(args.reference_report)
    if not args.skip_reference_check:
        reference_report = load_reference_report(reference_report_path)

    report = run_api_evaluation(
        config,
        labels,
        api_url=configured_api_url(args.api_url),
        trace_store=trace_store,
        reference_report=reference_report,
        timeout_seconds=args.timeout_seconds,
        progress=print_progress,
    )
    if report["reference_comparison"] is not None:
        report["reference_comparison"]["reference_report_path"] = str(reference_report_path)
    report["trace_verification"]["database_path"] = str(trace_db_path) if trace_store is not None else None

    if args.skip_mlflow_verification:
        report["tracking_verification"] = None
    else:
        report["tracking_verification"] = verify_mlflow(project_path(args.mlflow_config))

    json_path, csv_path = write_evaluation_artifacts(report, output_directory=output_directory)
    metrics = report["metrics"]
    print(f"Evaluated {metrics['question_count']} questions through {report['api']['base_url']}/query.")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Recall@5: {metrics['recall_at_k']['5']:.4f}")
    if report["reference_comparison"] is not None:
        comparison = report["reference_comparison"]
        print(f"Reference parity: {comparison['exact_ranking_match_count']}/{comparison['question_count']} exact rankings.")
    if trace_store is not None:
        print(f"Verified SQLite traces: {report['trace_verification']['verified_trace_count']}")
    if report["tracking_verification"] is not None:
        print(f"Verified MLflow runs: {report['tracking_verification']['verified_run_count']}")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"API evaluation failed: {error}") from error
