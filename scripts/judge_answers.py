import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.evaluation.llm_judge import (  # noqa: E402
    evaluate_generation_config,
    judgment_artifact_paths,
    load_generation_judge_config,
    load_golden_questions,
    select_evaluation_sample,
    summarize_judgments,
    write_judgment_artifacts,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and automatically judge the Day 20 acceptance sample.")
    parser.add_argument("--config", type=Path, default=Path("configs/generation_judge.yaml"), help="Generation-judge YAML configuration.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional local KEY=VALUE file containing provider credentials.")
    parser.add_argument("--qdrant-url", help="Optional Qdrant URL override.")
    parser.add_argument("--output-dir", type=Path, help="Optional artifact directory override.")
    parser.add_argument("--validate-only", action="store_true", help="Validate configuration and sample allocation without Qdrant or provider calls.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing Day 20 judgment artifacts.")
    return parser.parse_args()


def load_env_file(path):
    """Load simple KEY=VALUE entries without replacing exported environment values."""
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


def apply_overrides(config, qdrant_url=None, output_dir=None):
    if qdrant_url is not None:
        retrieval = config.retrieval.model_copy(update={"qdrant_url": qdrant_url.strip().rstrip("/") or None})
        config = config.model_copy(update={"retrieval": retrieval})
    if output_dir is not None:
        output = config.output.model_copy(update={"directory": output_dir.resolve()})
        config = config.model_copy(update={"output": output})
    return config


def print_progress(event):
    timings = event["timings"]
    print(
        f"[{event['index']}/{event['total']}] {event['question_id']} ({event['query_type']}) "
        f"retrieval={timings.retrieval_ms:.1f} ms generation={timings.generation_ms:.1f} ms judge={timings.judge_ms:.1f} ms",
        flush=True,
    )


def main():
    args = parse_args()
    config = load_generation_judge_config(args.config, project_root=PROJECT_ROOT)
    config = apply_overrides(config, qdrant_url=args.qdrant_url, output_dir=args.output_dir)
    questions = load_golden_questions(config.dataset.golden_path)
    sample = select_evaluation_sample(questions, config)
    sample_description = ", ".join(f"{query_type}={count}" for query_type, count in config.dataset.query_type_counts.items())

    if args.validate_only:
        print(
            f"Valid generation-judge config '{config.name}' with {len(sample)} selected questions "
            f"({sample_description}); generator={config.generation.provider}/{config.generation.model}; "
            f"judge={config.judge.provider}/{config.judge.model}."
        )
        return

    judgments_path, summary_path = judgment_artifact_paths(config)
    existing_paths = [path for path in (judgments_path, summary_path) if path.exists()]
    if existing_paths and not args.overwrite:
        paths = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Refusing to overwrite existing artifacts: {paths}. Pass --overwrite to replace them.")

    load_env_file(args.env_file)
    print(f"Running {len(sample)} questions ({sample_description}).")
    print(f"Generator: {config.generation.provider}/{config.generation.model}")
    print(f"Judge: {config.judge.provider}/{config.judge.model}")
    records = evaluate_generation_config(config, sample, progress=print_progress)
    judgments_path, summary_path = write_judgment_artifacts(records, config)
    summary = summarize_judgments(records)
    automatic = summary["automatic_metrics"]
    print(f"Judged answers: {summary['question_count']}")
    print(f"Mean faithfulness: {automatic['mean_faithfulness']:.3f}")
    print(f"Mean answer relevance: {automatic['mean_answer_relevance']:.3f}")
    print(f"Judgments: {judgments_path}")
    print(f"Summary: {summary_path}")
    print("Manual spot-checks are pending; run scripts/review_judgments.py.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Generation judging failed: {error}") from error
