import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.evaluation.bm25_runner import load_evaluation_report  # noqa: E402
from ragops.evaluation.hybrid_runner import (  # noqa: E402
    compare_hybrid_reports,
    evaluate_hybrid_config,
    require_hybrid_evaluation_settings,
    write_hybrid_comparison_artifacts,
)
from ragops.evaluation.runner import load_evaluation_labels, write_evaluation_artifacts  # noqa: E402
from ragops.retrieval.bm25 import load_bm25_index, validate_bm25_index  # noqa: E402
from ragops.retrieval.hybrid import load_hybrid_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RRF hybrid retrieval and compare dense, BM25, and hybrid baselines.")
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid.yaml"), help="Hybrid YAML configuration.")
    parser.add_argument("--qdrant-url", help="Optional Qdrant URL override.")
    parser.add_argument("--index", type=Path, help="Optional persisted BM25 index override.")
    parser.add_argument("--dense-report", type=Path, help="Optional dense baseline JSON override.")
    parser.add_argument("--bm25-report", type=Path, help="Optional BM25 baseline JSON override.")
    parser.add_argument("--output-dir", type=Path, help="Optional hybrid JSON/CSV directory override.")
    parser.add_argument("--comparison-output", type=Path, help="Optional three-way comparison JSON path.")
    parser.add_argument("--report-output", type=Path, help="Optional Markdown benchmark path.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing hybrid evaluation artifacts.")
    parser.add_argument("--validate-only", action="store_true", help="Validate config, labels, baselines, and BM25 provenance without querying Qdrant.")
    return parser.parse_args()


def resolve_override(path):
    if path is None:
        return None
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def apply_overrides(config, args):
    require_hybrid_evaluation_settings(config)
    dense_updates = {}
    if args.qdrant_url is not None:
        dense_updates["qdrant_url"] = args.qdrant_url.strip().rstrip("/") or None
    bm25_updates = {}
    if args.index is not None:
        bm25_updates["index_path"] = resolve_override(args.index)
    evaluation_updates = {}
    if args.dense_report is not None:
        evaluation_updates["dense_baseline_path"] = resolve_override(args.dense_report)
    if args.bm25_report is not None:
        evaluation_updates["bm25_baseline_path"] = resolve_override(args.bm25_report)
    output_updates = {}
    if args.output_dir is not None:
        output_updates["directory"] = resolve_override(args.output_dir)
    if args.comparison_output is not None:
        output_updates["comparison_path"] = resolve_override(args.comparison_output)
    if args.report_output is not None:
        output_updates["report_path"] = resolve_override(args.report_output)

    return config.model_copy(
        update={
            "dense": config.dense.model_copy(update=dense_updates),
            "bm25": config.bm25.model_copy(update=bm25_updates),
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
        raise FileExistsError(f"Refusing to overwrite existing hybrid artifacts: {rendered}. Pass --overwrite to replace them.")


def validate_baseline_against_labels(report, retriever_name, labels, k_values):
    retriever_type = report.get("configuration", {}).get("retriever", {}).get("type")
    if retriever_type != retriever_name:
        raise ValueError(f"Configured {retriever_name} report has retriever type {retriever_type!r}.")
    questions = report.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"Configured {retriever_name} report has no question results.")
    by_id = {question.get("question_id"): question for question in questions if isinstance(question, dict)}
    if len(by_id) != len(questions):
        raise ValueError(f"Configured {retriever_name} report has invalid or duplicate question IDs.")
    labels_by_id = {label.question_id: label for label in labels}
    if set(by_id) != set(labels_by_id):
        raise ValueError(f"Configured {retriever_name} report does not cover the verified label set.")
    if report.get("metrics", {}).get("k_values") != k_values:
        raise ValueError(f"Configured {retriever_name} report does not use the requested metric cutoffs.")
    for question_id, question in by_id.items():
        label = labels_by_id[question_id]
        expected = (label.question, label.expected_source, label.relevant_chunk_ids)
        actual = (question.get("question"), question.get("expected_source"), question.get("relevant_chunk_ids"))
        if actual != expected:
            raise ValueError(f"Configured {retriever_name} report differs from label {question_id}.")


def main():
    args = parse_args()
    config = apply_overrides(load_hybrid_config(args.config, project_root=PROJECT_ROOT), args)
    labels = load_evaluation_labels(config)
    dense_report = load_evaluation_report(config.evaluation.dense_baseline_path)
    bm25_report = load_evaluation_report(config.evaluation.bm25_baseline_path)
    validate_baseline_against_labels(dense_report, "dense", labels, config.evaluation.k_values)
    validate_baseline_against_labels(bm25_report, "bm25", labels, config.evaluation.k_values)

    index = load_bm25_index(config.bm25.index_path)
    payload = validate_bm25_index(index, config.bm25_validation_config())
    bm25_baseline_hash = bm25_report.get("index", {}).get("source_sha256")
    if bm25_baseline_hash != payload.source_sha256:
        raise ValueError("BM25 baseline report does not match the current hybrid source artifact SHA256.")

    if args.validate_only:
        print(
            f"Valid hybrid evaluation config '{config.name}' with {len(labels)} labels, dense run '{dense_report['run_name']}', "
            f"BM25 run '{bm25_report['run_name']}', and {payload.document_count} sparse documents."
        )
        return

    refuse_existing_artifacts(config, args.overwrite)
    hybrid_report = evaluate_hybrid_config(config, labels, index_loader=lambda path: index, progress=print_progress)
    expected_hybrid_path = config.output.directory / f"{config.name}.json"
    comparison = compare_hybrid_reports(
        dense_report,
        bm25_report,
        hybrid_report,
        report_paths={
            "dense": str(config.evaluation.dense_baseline_path),
            "bm25": str(config.evaluation.bm25_baseline_path),
            "hybrid": str(expected_hybrid_path),
        },
    )

    hybrid_json_path, hybrid_csv_path = write_evaluation_artifacts(hybrid_report)
    comparison_path, report_path = write_hybrid_comparison_artifacts(comparison, config.output.comparison_path, config.output.report_path)
    metrics = hybrid_report["metrics"]
    decision = comparison["decision"]

    print(f"Evaluated {metrics['question_count']} RRF hybrid questions.")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Hit Rate@5: {metrics['hit_rate_at_k']['5']:.4f}")
    print(f"Average hybrid latency: {hybrid_report['latency_ms']['average']:.1f} ms")
    print(
        f"MRR decision: best={decision['best_retriever']}; improves over dense={decision['hybrid_improves_over_dense']}; "
        f"improves over BM25={decision['hybrid_improves_over_bm25']}"
    )
    print(f"Hybrid JSON report: {hybrid_json_path}")
    print(f"Hybrid CSV report: {hybrid_csv_path}")
    print(f"Comparison JSON: {comparison_path}")
    print(f"Benchmark report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Hybrid evaluation failed: {error}") from error
