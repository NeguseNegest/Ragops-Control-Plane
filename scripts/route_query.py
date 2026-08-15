#!/usr/bin/env python3
"""Run the Day 38 initial probe and print its deterministic route decision."""

import argparse
import json
from pathlib import Path

from ragops.api.pipelines import PipelineRuntime


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Question to classify with the configured routing policy.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository/deployment root containing the checked-in pipeline and router configs.",
    )
    return parser.parse_args()


def route_report(result):
    """Return a JSON-safe decision report without exposing document text."""
    decision = result.decision.model_dump(mode="json")
    return {
        "query": result.probe.query,
        "router_policy": {
            "router_id": decision["router_id"],
            "status": decision["router_status"],
            "feature_schema_version": decision["feature_schema_version"],
        },
        "route": decision["route"],
        "reason_code": decision["reason_code"],
        "reason": decision["reason"],
        "matched_reason_codes": decision["matched_reason_codes"],
        "execution_intent": {
            "pipeline_config": decision["pipeline_config"],
            "maximum_top_k": decision["maximum_top_k"],
            "reuse_probe": decision["reuse_probe"],
            "generate_answer": decision["generate_answer"],
            "response_mode": decision["response_mode"],
        },
        "features": result.probe.features.model_dump(mode="json"),
        "probe": {
            "chunk_ids": [chunk.chunk_id for chunk in result.probe.chunks],
            "scores": [chunk.score for chunk in result.probe.chunks],
            "timings": result.probe.timings.model_dump(mode="json"),
        },
    }


def main():
    args = parse_args()
    runtime = PipelineRuntime(project_root=args.project_root)
    print(json.dumps(route_report(runtime.route_query(args.query)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
