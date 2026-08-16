#!/usr/bin/env python3
"""Validate or generate the Day 41 routed-versus-fixed comparison."""

import argparse
import json
from pathlib import Path

from ragops.evaluation.router_comparison import (
    load_router_evaluation_config,
    run_router_evaluation,
    validate_router_evaluation_inputs,
    write_router_evaluation_artifacts,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/router_evaluation.yaml"))
    parser.add_argument("--validate-only", action="store_true", help="Validate all source artifacts and recompute the comparison without writing outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing JSON, CSV, and Markdown artifacts.")
    return parser.parse_args()


def summary(report):
    return {
        "evaluation_id": report["evaluation_id"],
        "evaluation_status": report["evaluation_status"],
        "mode": report["mode"],
        "router": report["router"],
        "scope": report["scope"],
        "strategies": {
            name: {
                "hit_rate_at_5": strategy["retrieval_quality_supported"]["hit_rate_at_5"],
                "unsupported_refusal_accuracy": strategy["refusal_quality"]["unsupported_refusal_accuracy"],
                "combined_quality_proxy": strategy["combined_quality_proxy"]["success_rate"],
                "average_latency_ms": strategy["latency_ms_supported"]["average"],
                "projected_cost_usd": strategy["generation_cost_projection_supported"]["amount_usd"]["total"],
            }
            for name, strategy in report["strategies"].items()
        },
        "decision": report["decision"],
    }


def main():
    args = parse_args()
    project_root = Path.cwd().resolve()
    config = load_router_evaluation_config(args.config, project_root=project_root)
    inputs = validate_router_evaluation_inputs(config)
    report = run_router_evaluation(config, inputs=inputs)
    if args.validate_only:
        if config.output.json_path.is_file():
            try:
                recorded = json.loads(config.output.json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Recorded router comparison is invalid JSON: {config.output.json_path}") from error
            if recorded != report:
                raise ValueError("Recorded router comparison is stale relative to its current source artifacts.")
        print(json.dumps(summary(report), indent=2, sort_keys=True))
        return

    paths = write_router_evaluation_artifacts(report, config, overwrite=args.overwrite)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    print("Artifacts:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Router evaluation failed: {error}") from error
