import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.tracing.store import TRACE_SCHEMA_VERSION, TraceStore, configured_trace_db_path  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize or validate the local SQLite trace store.")
    parser.add_argument("--db-path", type=Path, help="Optional path override; defaults to RAGOPS_TRACE_DB_PATH or data/traces/ragops_traces.sqlite3.")
    parser.add_argument("--validate-only", action="store_true", help="Validate an existing database without creating or migrating it.")
    return parser.parse_args()


def main():
    args = parse_args()
    path = configured_trace_db_path(args.db_path, project_root=PROJECT_ROOT)
    store = TraceStore(path)
    if args.validate_only:
        store.validate_schema()
        action = "Validated"
    else:
        store.initialize()
        action = "Initialized"
    counts = store.counts()
    print(
        f"{action} SQLite trace store schema v{TRACE_SCHEMA_VERSION} at {store.path}: "
        f"traces={counts['traces']}, retrieved_chunks={counts['retrieved_chunks']}, feedback={counts['feedback']}."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Trace store failed: {error}") from error
