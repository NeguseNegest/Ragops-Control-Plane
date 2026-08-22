#!/usr/bin/env python3
"""Run, validate, aggregate, and track the Day 47 final benchmark."""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.api.pipelines import PipelineRuntime  # noqa: E402
from ragops.evaluation.final_benchmark import (  # noqa: E402
    PIPELINE_NAMES,
    _load_chunks,
    attach_final_artifacts_to_runs,
    build_final_benchmark_report,
    build_routed_report,
    load_benchmark_judgments,
    load_final_benchmark_config,
    log_final_benchmark_runs,
    run_answer_quality_pipeline,
    validate_benchmark_judgments,
    validate_final_benchmark_inputs,
    validate_retrieval_reports,
    verify_final_benchmark_runs,
    write_benchmark_judgment_checkpoint,
    write_benchmark_judgments,
    write_final_benchmark_artifacts,
    write_routed_report,
)
from ragops.generation.factory import create_generation_client  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final_benchmark.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional provider credential file.")
    parser.add_argument("--validate-only", action="store_true", help="Validate the contract and frozen datasets without requiring run artifacts.")
    parser.add_argument("--routed-only", action="store_true", help="Build only the routed retrieval/refusal artifact; make no provider calls.")
    parser.add_argument("--judgments-only", action="store_true", help="Run only external generation/judging against existing retrieval artifacts.")
    parser.add_argument("--aggregate-only", action="store_true", help="Aggregate existing retrieval, routed, and judgment artifacts without live calls.")
    parser.add_argument("--verify-only", action="store_true", help="Verify the complete final report and its five MLflow runs without writing.")
    parser.add_argument("--skip-mlflow", action="store_true", help="Do not publish the five final runs to MLflow.")
    parser.add_argument("--mlflow-uri", help="Override the final benchmark MLflow URI without editing the checked-in contract.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing routed, judgment, and final report artifacts.")
    parser.add_argument("--resume", action="store_true", help="Reuse complete judgments and resume valid per-question checkpoints after provider throttling.")
    return parser.parse_args()


def load_env_file(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


def routed_progress(event):
    print(f"[policy {event['index']}/{event['total']}] {event['question_id']}: {event['route']}", file=sys.stderr, flush=True)


def quality_progress(event):
    print(
        f"[{event['pipeline']} {event['index']}/{event['total']}] {event['question_id']}: "
        f"generation={event['generation_ms']:.1f} ms judge={event['judge_ms']:.1f} ms",
        file=sys.stderr,
        flush=True,
    )


class RetryingGenerationClient:
    """Retry only provider throttling/transient availability without hiding permanent errors."""

    def __init__(self, client, maximum_attempts=8):
        self.client = client
        self.provider = client.provider
        self.model = client.model
        self.maximum_attempts = maximum_attempts

    @staticmethod
    def _retryable(error):
        message = str(error).casefold()
        return any(marker in message for marker in ("429", "too_many_requests", "resource_exhausted", "rate limit", "temporarily unavailable", "503"))

    @staticmethod
    def _delay(error, attempt):
        match = re.search(r"retry in ([0-9.]+)s", str(error), flags=re.IGNORECASE)
        provider_delay = float(match.group(1)) + 1 if match else 0
        return min(30.0, max(provider_delay, 3.0 * (2 ** (attempt - 1))))

    def _call(self, method, prompt):
        for attempt in range(1, self.maximum_attempts + 1):
            try:
                return method(prompt)
            except Exception as error:
                if attempt == self.maximum_attempts or not self._retryable(error):
                    raise
                delay = self._delay(error, attempt)
                print(
                    f"[{self.provider}] transient provider throttle on attempt {attempt}/{self.maximum_attempts}; retrying in {delay:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError("Provider retry loop ended unexpectedly.")

    def generate(self, prompt):
        return self._call(self.client.generate, prompt)

    def generate_with_metadata(self, prompt):
        return self._call(self.client.generate_with_metadata, prompt)


def run_routed_artifact(config, overwrite):
    inputs = validate_final_benchmark_inputs(config, require_reports=True, require_judgments=False)
    labels, _, indexed = validate_retrieval_reports(config, reports=inputs["reports"])
    runtime = PipelineRuntime(project_root=PROJECT_ROOT, router_config_path=config.pipelines.routed.config_path)
    routed = build_routed_report(
        config,
        labels,
        indexed,
        inputs["adversarial"],
        runtime.route_query,
        progress=routed_progress,
    )
    write_routed_report(routed, config, overwrite=overwrite)
    return inputs, {**indexed, "routed": {row["question_id"]: row for row in routed["questions"]}}


def run_quality_artifacts(config, overwrite, resume=False):
    inputs = validate_final_benchmark_inputs(config, require_reports=True, require_judgments=False)
    _, _, indexed = validate_retrieval_reports(config, reports=inputs["reports"])
    routed = json.loads(config.pipelines.routed.retrieval_report_path.read_text(encoding="utf-8"))
    indexed["routed"] = {row["question_id"]: row for row in routed["questions"]}
    generator = RetryingGenerationClient(
        create_generation_client(
            config.answer_quality.generation.provider,
            model=config.answer_quality.generation.model,
        )
    )
    judge = RetryingGenerationClient(
        create_generation_client(
            config.answer_quality.judge.provider,
            model=config.answer_quality.judge.model,
        )
    )
    chunks = _load_chunks(config.datasets.chunks_path)
    for pipeline in PIPELINE_NAMES:
        path = getattr(config.pipelines, pipeline).judgments_path
        existing = load_benchmark_judgments(path) if resume and path.exists() else []
        if existing:
            try:
                validate_benchmark_judgments(existing, config, pipeline)
            except ValueError:
                pass
            else:
                print(f"[{pipeline}] complete judgment artifact reused.", file=sys.stderr, flush=True)
                continue
        records = run_answer_quality_pipeline(
            config,
            pipeline,
            indexed,
            inputs["golden_by_id"],
            chunks,
            generator,
            judge,
            progress=quality_progress,
            existing_records=existing,
            checkpoint=lambda current, selected=pipeline: write_benchmark_judgment_checkpoint(current, config, selected),
        )
        write_benchmark_judgments(records, config, pipeline, overwrite=overwrite or resume)


def main():
    args = parse_args()
    config = load_final_benchmark_config(args.config, project_root=PROJECT_ROOT)
    if args.mlflow_uri:
        mlflow_config = config.mlflow.model_copy(update={"tracking_uri": args.mlflow_uri.strip().rstrip("/")})
        config = config.model_copy(update={"mlflow": mlflow_config})
    phase_count = sum((args.validate_only, args.routed_only, args.judgments_only, args.aggregate_only, args.verify_only))
    if phase_count > 1:
        raise ValueError("Choose at most one execution phase flag.")
    if args.validate_only:
        inputs = validate_final_benchmark_inputs(config, require_reports=False, require_judgments=False)
        print(
            json.dumps(
                {
                    "benchmark_id": f"{config.name}@{config.version}",
                    "retrieval_questions": len(inputs["labels"]),
                    "adversarial_questions": len(inputs["adversarial"]),
                    "answer_quality_questions_per_pipeline": len(config.answer_quality.sample_question_ids),
                    "pipelines": list(PIPELINE_NAMES),
                    "generator": config.answer_quality.generation.model_dump(),
                    "judge": config.answer_quality.judge.model_dump(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.routed_only:
        run_routed_artifact(config, overwrite=args.overwrite)
        print(str(config.pipelines.routed.retrieval_report_path))
        return

    if args.verify_only:
        report = json.loads(config.output.json_path.read_text(encoding="utf-8"))
        verified = verify_final_benchmark_runs(config, report)
        print(json.dumps({"benchmark_id": report["benchmark_id"], "verified_mlflow_runs": verified}, indent=2, sort_keys=True))
        return

    if not args.aggregate_only:
        load_env_file(args.env_file)
        if not args.judgments_only:
            run_routed_artifact(config, overwrite=args.overwrite)
        run_quality_artifacts(config, overwrite=args.overwrite, resume=args.resume)
        if args.judgments_only:
            print("Answer-quality judgment artifacts completed.")
            return

    report = build_final_benchmark_report(config)
    client = None
    if not args.skip_mlflow:
        references, client = log_final_benchmark_runs(config, report)
        report["mlflow_runs"] = references
    write_final_benchmark_artifacts(report, config, overwrite=args.overwrite)
    if client is not None:
        attach_final_artifacts_to_runs(config, report, client)
    print(
        json.dumps(
            {
                "benchmark_id": report["benchmark_id"],
                "pipelines": len(report["pipelines"]),
                "mlflow_runs": {name: value["run_id"] for name, value in report["mlflow_runs"].items()},
                "json": str(config.output.json_path),
                "csv": str(config.output.csv_path),
                "markdown": str(config.output.markdown_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Final benchmark failed: {error}") from error
