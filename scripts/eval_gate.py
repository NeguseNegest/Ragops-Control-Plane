#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from ragops.evaluation.gate import execute_evaluation_gate, gate_exit_code, load_evaluation_gate, render_gate_summary


def build_parser():
    parser = argparse.ArgumentParser(description="Run the compact Day 44 RAG evaluation gate.")
    parser.add_argument("--config", default="configs/eval_gate.yaml", help="Path to the strict evaluation-gate YAML config.")
    parser.add_argument("--project-root", default=".", help="Project root used to resolve configured paths.")
    parser.add_argument("--json", action="store_true", help="Print the complete JSON report after the human-readable summary.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    try:
        loaded_gate = load_evaluation_gate(config_path, project_root=project_root)
        report = execute_evaluation_gate(loaded_gate)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Evaluation gate could not run: {error}", file=sys.stderr)
        return 2

    print(render_gate_summary(report))
    if args.json:
        print()
        print(json.dumps(report, indent=2, sort_keys=True))
    return gate_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
