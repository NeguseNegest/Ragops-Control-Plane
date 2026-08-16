#!/usr/bin/env python3
"""Validate or generate the Day 42 router stabilization artifacts."""

import argparse
import json
from pathlib import Path

from ragops.evaluation.router_tuning import (
    load_router_tuning_config,
    run_router_tuning,
    write_router_tuning_artifacts,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/router_tuning.yaml"))
    parser.add_argument("--validate-only", action="store_true", help="Recompute selection/distribution and reject stale canonical JSON without writing.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing JSON, CSV, and Markdown artifacts.")
    return parser.parse_args()


def summary(report):
    return {
        "evaluation_id": report["evaluation_id"],
        "baseline_router_id": report["baseline_router_id"],
        "target_router_id": report["target_router_id"],
        "selected_value": report["selection"]["selected_value"],
        "selected_constraint_checks": report["selection"]["selected_candidate"]["constraint_checks"],
        "supported_distribution": report["distribution"]["supported"],
        "transitions": report["transitions"]["counts"],
        "stability": report["stability"],
    }


def main():
    args = parse_args()
    config = load_router_tuning_config(args.config, project_root=Path.cwd())
    report = run_router_tuning(config)
    if args.validate_only:
        if config.output.json_path.is_file():
            try:
                recorded = json.loads(config.output.json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Recorded router distribution is invalid JSON: {config.output.json_path}") from error
            if recorded != report:
                raise ValueError("Recorded router distribution is stale relative to current tuning evidence.")
        print(json.dumps(summary(report), indent=2, sort_keys=True))
        return
    paths = write_router_tuning_artifacts(report, config, overwrite=args.overwrite)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    print("Artifacts:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Router tuning failed: {error}") from error
