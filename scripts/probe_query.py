#!/usr/bin/env python3
"""Run the Day 37 initial dense probe and print structured router features."""

import argparse
import json
from pathlib import Path

from ragops.api.pipelines import PipelineRuntime


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Question to probe before route selection.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository/deployment root containing the checked-in pipeline configs.",
    )
    return parser.parse_args()


def probe_report(result):
    """Return non-document probe evidence suitable for CLI inspection."""
    return {
        "query": result.query,
        "features": result.features.model_dump(mode="json"),
        "probe": {
            "chunk_ids": [chunk.chunk_id for chunk in result.chunks],
            "scores": [chunk.score for chunk in result.chunks],
            "timings": result.timings.model_dump(mode="json"),
        },
        "route": None,
        "route_reason": None,
    }


def main():
    args = parse_args()
    runtime = PipelineRuntime(project_root=args.project_root)
    result = runtime.initial_probe(args.query)
    print(json.dumps(probe_report(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
