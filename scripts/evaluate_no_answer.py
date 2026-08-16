#!/usr/bin/env python3
"""Validate or run the Day 39 no-answer/refusal evaluation."""

import argparse
import json
import sys
from pathlib import Path

from ragops.api.pipelines import PipelineRuntime
from ragops.evaluation.no_answer import (
    load_no_answer_config,
    replay_no_answer_evaluation,
    run_no_answer_evaluation,
    validate_no_answer_inputs,
    write_no_answer_artifacts,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/no_answer.yaml"))
    parser.add_argument("--validate-only", action="store_true", help="Validate config, datasets, report provenance, and router policy without querying Qdrant.")
    parser.add_argument("--replay-existing", action="store_true", help="Recompute decisions from the existing report's persisted probe scores without querying Qdrant.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing JSON/CSV evaluation artifacts.")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path.cwd().resolve()
    config = load_no_answer_config(args.config, project_root=project_root)
    inputs = validate_no_answer_inputs(config)
    if args.validate_only and args.replay_existing:
        raise ValueError("--validate-only and --replay-existing cannot be combined.")
    if args.validate_only:
        summary = {
            "config_id": f"{config.name}@{config.version}",
            "status": config.status,
            "router_id": f"{inputs['router_config'].name}@{inputs['router_config'].version}",
            "configured_threshold": inputs["router_config"].thresholds.no_answer.top_score_below,
            "calibration_unsupported": inputs["counts"]["calibration"],
            "evaluation_unsupported": inputs["counts"]["evaluation"],
            "supported": len(inputs["supported_report"]["questions"]),
            "prompt_version": config.refusal.prompt_version,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.replay_existing:
        if not config.output.json_path.is_file():
            raise FileNotFoundError(f"No existing no-answer report to replay: {config.output.json_path}")
        try:
            source_report = json.loads(config.output.json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Existing no-answer report is invalid JSON: {config.output.json_path}") from error
        report = replay_no_answer_evaluation(config, source_report)
        write_no_answer_artifacts(report, config, overwrite=args.overwrite)
        print(json.dumps({key: report[key] for key in ("run_name", "router_id", "threshold", "counts", "metrics", "acceptance")}, indent=2, sort_keys=True))
        return

    def progress(state):
        print(
            f"[{state['index']}/{state['total']}] {state['question_id']}: {state['route']}",
            file=sys.stderr,
            flush=True,
        )

    runtime = PipelineRuntime(project_root=project_root, router_config_path=config.router_config)
    report = run_no_answer_evaluation(config, runtime, progress=progress)
    write_no_answer_artifacts(report, config, overwrite=args.overwrite)
    print(json.dumps({key: report[key] for key in ("run_name", "router_id", "threshold", "counts", "metrics", "acceptance")}, indent=2, sort_keys=True))
    if not report["acceptance"]["passed"]:
        raise SystemExit("No-answer acceptance thresholds failed.")


if __name__ == "__main__":
    main()
