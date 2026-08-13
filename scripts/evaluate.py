import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.evaluation.runner import evaluate_dense_config, load_evaluation_config, load_evaluation_labels, write_evaluation_artifacts  # noqa: E402
from ragops.tracking.mlflow import DEFAULT_MLFLOW_CONFIG_PATH, load_mlflow_config, log_prepared_run, prepare_retrieval_run  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a retrieval configuration against verified relevance labels.")
    parser.add_argument("--config", type=Path, default=Path("configs/dense_baseline.yaml"), help="Evaluation YAML configuration.")
    parser.add_argument("--qdrant-url", help="Optional Qdrant URL override.")
    parser.add_argument("--output-dir", type=Path, help="Optional artifact directory override.")
    parser.add_argument("--validate-only", action="store_true", help="Validate configuration and labels without connecting to Qdrant.")
    parser.add_argument("--mlflow-config", type=Path, default=DEFAULT_MLFLOW_CONFIG_PATH, help="MLflow tracking YAML.")
    parser.add_argument("--skip-mlflow", action="store_true", help="Write evaluation artifacts without logging an MLflow run.")
    return parser.parse_args()


def print_progress(event):
    print(f"[{event['index']}/{event['total']}] {event['question_id']} ({event['latency_ms']:.1f} ms)", flush=True)


def apply_overrides(config, qdrant_url=None, output_dir=None):
    if qdrant_url is not None:
        retriever = config.retriever.model_copy(update={"qdrant_url": qdrant_url.strip().rstrip("/") or None})
        config = config.model_copy(update={"retriever": retriever})
    if output_dir is not None:
        output = config.output.model_copy(update={"directory": output_dir.resolve()})
        config = config.model_copy(update={"output": output})
    return config


def main():
    args = parse_args()
    config = load_evaluation_config(args.config, project_root=PROJECT_ROOT)
    config = apply_overrides(config, qdrant_url=args.qdrant_url, output_dir=args.output_dir)
    labels = load_evaluation_labels(config)

    if args.validate_only:
        print(f"Valid evaluation config '{config.name}' with {len(labels)} labels.")
        return

    report = evaluate_dense_config(config, labels, progress=print_progress)
    json_path, csv_path = write_evaluation_artifacts(report)
    if not args.skip_mlflow:
        mlflow_config_path = args.mlflow_config if args.mlflow_config.is_absolute() else (PROJECT_ROOT / args.mlflow_config).resolve()
        tracking_config = load_mlflow_config(mlflow_config_path, project_root=PROJECT_ROOT)
        source_config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
        prepared = prepare_retrieval_run("dense", source_config_path, json_path, csv_path, report=report)
        tracking_result = log_prepared_run(prepared, tracking_config)
        action = "created" if tracking_result["created"] else "reused"
        print(f"MLflow run {action}: {tracking_result['run_id']}")
    metrics = report["metrics"]
    recall_at_5 = metrics["recall_at_k"].get("5")

    print(f"Evaluated {metrics['question_count']} questions.")
    print(f"MRR: {metrics['mrr']:.4f}")
    if recall_at_5 is not None:
        print(f"Recall@5: {recall_at_5:.4f}")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Evaluation failed: {error}") from error
