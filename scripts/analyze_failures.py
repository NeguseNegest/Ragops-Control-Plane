#!/usr/bin/env python3
"""Build or validate the reviewed Day 48 failure analysis and regression cases."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.evaluation.failure_analysis import (  # noqa: E402
    build_failure_analysis,
    load_failure_analysis_config,
    validate_failure_analysis_outputs,
    write_failure_analysis_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/failure_analysis.yaml"))
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the Markdown analysis and JSONL regression cases; otherwise validate checked-in outputs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow --write to replace existing Day 48 artifacts.")
    args = parser.parse_args()
    if args.overwrite and not args.write:
        parser.error("--overwrite requires --write.")
    return args


def main():
    args = parse_args()
    try:
        config = load_failure_analysis_config(args.config, project_root=PROJECT_ROOT)
        analysis = build_failure_analysis(config)
        if args.write:
            write_failure_analysis_outputs(config, analysis=analysis, overwrite=args.overwrite)
            action = "written"
        else:
            validate_failure_analysis_outputs(config, analysis=analysis)
            action = "validated"
    except (FileNotFoundError, ValueError) as error:
        print(f"Day 48 failure analysis: FAIL: {error}", file=sys.stderr)
        return 1

    summary = analysis["summary"]
    print(f"Day 48 failure analysis {action}: PASS")
    print(
        f"Failures: {summary['failure_count']}; regression cases: {summary['regression_case_count']}; "
        f"analysis-only: {summary['non_regression_case_count']}"
    )
    print(f"Categories: {summary['category_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
