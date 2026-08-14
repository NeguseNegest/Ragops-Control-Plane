import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.pipeline_registry import (  # noqa: E402
    DEFAULT_PIPELINE_REGISTRY_CONFIG_PATH,
    build_pipeline_registry,
    load_pipeline_registry_config,
    validate_registry_matches_sources,
    write_pipeline_registry,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build or validate the versioned retrieval pipeline registry.")
    parser.add_argument("--config", type=Path, default=DEFAULT_PIPELINE_REGISTRY_CONFIG_PATH, help="Pipeline registry YAML.")
    parser.add_argument("--validate-only", action="store_true", help="Validate that the checked-in registry exactly matches current sources.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing pipeline registry artifact.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    config = load_pipeline_registry_config(config_path, project_root=PROJECT_ROOT)
    if args.validate_only:
        registry = validate_registry_matches_sources(config, PROJECT_ROOT)
        print(
            f"Valid pipeline registry '{registry.registry_name}': {len(registry.pipelines)} versions; "
            f"baseline={registry.aliases.baseline}, candidate={registry.aliases.candidate}, "
            f"production={registry.aliases.production}."
        )
        return

    registry = build_pipeline_registry(config, PROJECT_ROOT)
    output_path = write_pipeline_registry(registry, config.output_path, overwrite=args.overwrite)
    print(f"Wrote {len(registry.pipelines)} pipeline versions to {output_path}.")
    print(
        f"Aliases: baseline={registry.aliases.baseline}, candidate={registry.aliases.candidate}, "
        f"production={registry.aliases.production}."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Pipeline registry failed: {error}") from error
