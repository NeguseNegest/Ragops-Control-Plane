#!/usr/bin/env python3
"""Validate the Day 36 router policy, registry references, and calibration evidence."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.pipeline_registry import DEFAULT_PIPELINE_REGISTRY_PATH, load_pipeline_registry  # noqa: E402
from ragops.routing.config import (  # noqa: E402
    DEFAULT_ROUTER_CONFIG_PATH,
    load_router_config,
    resolve_router_path,
    validate_router_calibration,
    validate_router_registry_references,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ROUTER_CONFIG_PATH, help="Router policy YAML.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_PIPELINE_REGISTRY_PATH, help="Pipeline registry JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_router_config(args.config, project_root=PROJECT_ROOT)
    registry = load_pipeline_registry(resolve_router_path(args.registry, PROJECT_ROOT))
    routes = validate_router_registry_references(config, registry)
    calibration = validate_router_calibration(config, project_root=PROJECT_ROOT)
    report = {
        "router_id": f"{config.name}@{config.version}",
        "status": config.status,
        "feature_schema_version": config.feature_schema_version,
        "decision_order": list(config.decision_order),
        "probe": config.probe.model_dump(mode="json"),
        "registered_routes": routes,
        "calibration": calibration,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Router config validation failed: {error}") from error
