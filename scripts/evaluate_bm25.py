import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.evaluation.bm25_runner import (  # noqa: E402
    compare_evaluation_reports,
    evaluate_bm25_config,
    load_evaluation_report,
    require_evaluation_settings,
    write_comparison_artifacts,
)
from ragops.evaluation.runner import load_evaluation_labels, write_evaluation_artifacts  # noqa: E402
from ragops.retrieval.bm25 import load_bm25_config, load_bm25_index, validate_bm25_index  # noqa: E402
from ragops.tracking.mlflow import DEFAULT_MLFLOW_CONFIG_PATH, load_mlflow_config, log_prepared_run, prepare_retrieval_run  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate persisted BM25 retrieval and compare it with the dense baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/bm25_baseline.yaml"), help="BM25 YAML configuration.")
    parser.add_argument("--index", type=Path, help="Optional persisted BM25 index override.")
    parser.add_argument("--dense-report", type=Path, help="Optional dense evaluation JSON override.")
    parser.add_argument("--output-dir", type=Path, help="Optional BM25 JSON/CSV directory override.")
    parser.add_argument("--comparison-output", type=Path, help="Optional paired comparison JSON path.")
    parser.add_argument("--report-output", type=Path, help="Optional Markdown report path.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing BM25 evaluation and comparison artifacts.")
    parser.add_argument("--validate-only", action="store_true", help="Validate config, labels, index provenance, and dense report without evaluating.")
    parser.add_argument("--mlflow-config", type=Path, default=DEFAULT_MLFLOW_CONFIG_PATH, help="MLflow tracking YAML.")
    parser.add_argument("--skip-mlflow", action="store_true", help="Write evaluation artifacts without logging an MLflow run.")
    return parser.parse_args()


def resolve_override(path):
    if path is None:
        return None
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def apply_overrides(config, args):
    require_evaluation_settings(config)
    retriever_updates = {}
    if args.index is not None:
        retriever_updates["index_path"] = resolve_override(args.index)
    evaluation_updates = {}
    if args.dense_report is not None:
        evaluation_updates["dense_baseline_path"] = resolve_override(args.dense_report)
    output_updates = {}
    if args.output_dir is not None:
        output_updates["directory"] = resolve_override(args.output_dir)
    if args.comparison_output is not None:
        output_updates["comparison_path"] = resolve_override(args.comparison_output)
    if args.report_output is not None:
        output_updates["report_path"] = resolve_override(args.report_output)

    return config.model_copy(
        update={
            "retriever": config.retriever.model_copy(update=retriever_updates),
            "evaluation": config.evaluation.model_copy(update=evaluation_updates),
            "output": config.output.model_copy(update=output_updates),
        }
    )


def print_progress(event):
    print(f"[{event['index']}/{event['total']}] {event['question_id']} ({event['latency_ms']:.1f} ms)", flush=True)


def artifact_paths(config):
    return (
        config.output.directory / f"{config.name}.json",
        config.output.directory / f"{config.name}.csv",
        config.output.comparison_path,
        config.output.report_path,
    )


def refuse_existing_artifacts(config, overwrite):
    existing = [path for path in artifact_paths(config) if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing evaluation artifacts: {rendered}. Pass --overwrite to replace them.")


def validate_dense_questions(dense_report, labels, k_values):
    retriever_type = dense_report.get("configuration", {}).get("retriever", {}).get("type")
    if retriever_type != "dense":
        raise ValueError(f"Configured dense report must use a dense retriever, got {retriever_type!r}.")
    dense_questions = [question for question in dense_report.get("questions", []) if isinstance(question, dict)]
    dense_ids = [question.get("question_id") for question in dense_questions]
    labels_by_id = {label.question_id: label for label in labels}
    label_ids = list(labels_by_id)
    if len(dense_ids) != len(set(dense_ids)):
        raise ValueError("Dense evaluation report contains duplicate question IDs.")
    if set(dense_ids) != set(label_ids):
        raise ValueError("Dense evaluation report does not cover the configured verified label set.")
    if dense_report.get("metrics", {}).get("k_values") != k_values:
        raise ValueError("Dense evaluation report does not use the configured metric cutoffs.")
    for question in dense_questions:
        label = labels_by_id[question["question_id"]]
        if question.get("question") != label.question:
            raise ValueError(f"Dense evaluation question text differs for {label.question_id}.")
        if question.get("expected_source") != label.expected_source:
            raise ValueError(f"Dense evaluation source differs for {label.question_id}.")
        if question.get("relevant_chunk_ids") != label.relevant_chunk_ids:
            raise ValueError(f"Dense evaluation relevant chunks differ for {label.question_id}.")


def main():
    args = parse_args()
    config = apply_overrides(load_bm25_config(args.config, project_root=PROJECT_ROOT), args)
    labels = load_evaluation_labels(config)
    dense_report = load_evaluation_report(config.evaluation.dense_baseline_path)
    validate_dense_questions(dense_report, labels, config.evaluation.k_values)

    if args.validate_only:
        index = load_bm25_index(config.retriever.index_path)
        payload = validate_bm25_index(index, config)
        print(
            f"Valid BM25 evaluation config '{config.name}' with {len(labels)} labels, "
            f"{payload.document_count} indexed chunks, and dense run '{dense_report['run_name']}'."
        )
        return

    refuse_existing_artifacts(config, args.overwrite)
    bm25_report = evaluate_bm25_config(config, labels, progress=print_progress)
    expected_bm25_json_path = config.output.directory / f"{config.name}.json"
    comparison = compare_evaluation_reports(
        dense_report,
        bm25_report,
        dense_report_path=config.evaluation.dense_baseline_path,
        bm25_report_path=expected_bm25_json_path,
    )

    bm25_json_path, bm25_csv_path = write_evaluation_artifacts(bm25_report)
    comparison_path, report_path = write_comparison_artifacts(comparison, config.output.comparison_path, config.output.report_path)
    if not args.skip_mlflow:
        mlflow_config_path = args.mlflow_config if args.mlflow_config.is_absolute() else (PROJECT_ROOT / args.mlflow_config).resolve()
        tracking_config = load_mlflow_config(mlflow_config_path, project_root=PROJECT_ROOT)
        source_config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
        prepared = prepare_retrieval_run(
            "bm25",
            source_config_path,
            bm25_json_path,
            bm25_csv_path,
            comparison_path=comparison_path,
            benchmark_path=report_path,
            report=bm25_report,
        )
        tracking_result = log_prepared_run(prepared, tracking_config)
        action = "created" if tracking_result["created"] else "reused"
        print(f"MLflow run {action}: {tracking_result['run_id']}")
    metrics = bm25_report["metrics"]

    print(f"Evaluated {metrics['question_count']} BM25 questions.")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Hit Rate@5: {metrics['hit_rate_at_k']['5']:.4f}")
    print(f"Paired wins: BM25={comparison['wins']['bm25']}, dense={comparison['wins']['dense']}, ties={comparison['wins']['tie']}")
    print(f"BM25 JSON report: {bm25_json_path}")
    print(f"BM25 CSV report: {bm25_csv_path}")
    print(f"Comparison JSON: {comparison_path}")
    print(f"Comparison report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"BM25 evaluation failed: {error}") from error
