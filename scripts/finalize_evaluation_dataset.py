import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.evaluation.final_dataset import (  # noqa: E402
    build_final_dataset_snapshot,
    load_final_dataset_config,
    validate_final_dataset_outputs,
    write_final_dataset_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build or validate the reviewed Day 46 evaluation datasets.")
    parser.add_argument("--config", type=Path, default=Path("configs/final_evaluation_dataset.yaml"))
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the reviewed snapshots and audit report; otherwise validate checked-in outputs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow --write to replace existing final artifacts.")
    args = parser.parse_args()
    if args.overwrite and not args.write:
        parser.error("--overwrite requires --write.")
    return args


def print_summary(report, action):
    counts = report["counts"]
    golden = report["golden_distribution"]
    retrieval = report["retrieval_distribution"]
    adversarial = report["adversarial_distribution"]
    print(f"Final evaluation dataset {action}: PASS")
    print(
        f"Golden: {counts['golden']} "
        f"(query types={golden['query_type']}, difficulty={golden['difficulty']})"
    )
    print(
        f"Retrieval labels: {counts['retrieval_labels']} "
        f"(methods={retrieval['label_method']}, relevant chunks={retrieval['relevant_chunk_count']})"
    )
    print(
        f"Adversarial/unsupported: {counts['adversarial']} "
        f"(categories={adversarial['category']})"
    )


def main():
    args = parse_args()
    try:
        config = load_final_dataset_config(args.config, project_root=PROJECT_ROOT)
        snapshot = build_final_dataset_snapshot(config)
        if args.write:
            report = write_final_dataset_outputs(config, snapshot=snapshot, overwrite=args.overwrite)
            action = "written"
        else:
            report = validate_final_dataset_outputs(config, snapshot=snapshot)
            action = "validated"
    except (FileNotFoundError, ValueError) as error:
        print(f"Final evaluation dataset: FAIL: {error}", file=sys.stderr)
        return 1

    print_summary(report, action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
