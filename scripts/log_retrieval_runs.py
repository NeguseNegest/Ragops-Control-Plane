import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.tracking.mlflow import (  # noqa: E402
    DEFAULT_MLFLOW_CONFIG_PATH,
    configured_tracking_uri,
    load_mlflow_config,
    log_prepared_runs,
    prepare_configured_runs,
    verify_prepared_runs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate and log the four recorded retrieval benchmarks to MLflow.")
    parser.add_argument("--config", type=Path, default=DEFAULT_MLFLOW_CONFIG_PATH, help="Central MLflow tracking YAML.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="Validate configs and artifacts without contacting MLflow.")
    mode.add_argument("--verify-only", action="store_true", help="Verify matching finished runs already exist in MLflow.")
    parser.add_argument("--force", action="store_true", help="Create new runs even when identical finished runs already exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    config = load_mlflow_config(config_path, project_root=PROJECT_ROOT)
    prepared_runs = prepare_configured_runs(config, PROJECT_ROOT)

    if args.validate_only:
        for prepared in prepared_runs:
            print(
                f"Valid {prepared['pipeline']} run '{prepared['run_name']}': "
                f"{len(prepared['params'])} params, {len(prepared['metrics'])} metrics, {len(prepared['artifacts'])} artifacts."
            )
        print("Validated four retrieval runs without contacting MLflow.")
        return

    if args.verify_only:
        verified = verify_prepared_runs(prepared_runs, config)
        print(
            f"Verified {len(verified)} retrieval runs in experiment '{config.experiment_name}' "
            f"at {configured_tracking_uri(config)}."
        )
        return

    results = log_prepared_runs(prepared_runs, config, force=args.force)
    for result in results:
        action = "created" if result["created"] else "reused"
        print(f"{action.capitalize()} MLflow run '{result['run_name']}': {result['run_id']}")
    verified = verify_prepared_runs(prepared_runs, config)
    print(
        f"Day 29 acceptance verified: {len(verified)} retrieval runs in experiment '{config.experiment_name}' "
        f"at {configured_tracking_uri(config)}."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"MLflow retrieval tracking failed: {error}") from error
