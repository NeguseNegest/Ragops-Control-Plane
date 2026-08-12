import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.retrieval.bm25 import (  # noqa: E402
    BM25Index,
    build_bm25_index_from_jsonl,
    load_bm25_config,
    load_bm25_index,
    retrieve_bm25,
    save_bm25_index,
    validate_bm25_index,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build and inspect the persisted BM25 chunk index.")
    parser.add_argument("--config", type=Path, default=Path("configs/bm25_baseline.yaml"), help="BM25 YAML configuration.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing persisted BM25 index.")
    parser.add_argument("--query", help="Optional sparse retrieval sanity query after building the index.")
    parser.add_argument("--top-k", type=int, help="Optional result count override for the sanity query.")
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--validate-only", action="store_true", help="Validate configuration and chunk input without building.")
    action_group.add_argument("--validate-index", action="store_true", help="Load the persisted index and verify it matches the chunk input and configuration.")
    args = parser.parse_args()
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be greater than zero.")
    return args


def validate_source(config):
    chunks_path = config.input.chunks_path
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunk input does not exist: {chunks_path}")


def print_results(results):
    for result in results:
        source = result.source_url or "unknown"
        preview = " ".join(result.text.split())
        if len(preview) > 180:
            preview = f"{preview[:177]}..."
        print(f"#{result.rank} score={result.score:.4f} chunk={result.chunk_id} source={source}")
        print(f"  {preview}")


def main():
    args = parse_args()
    config = load_bm25_config(args.config, project_root=PROJECT_ROOT)
    validate_source(config)

    if args.validate_only:
        print(f"Valid BM25 config '{config.name}' with input {config.input.chunks_path}.")
        return

    if args.validate_index:
        index = load_bm25_index(config.retriever.index_path)
        payload = validate_bm25_index(index, config)
        print(
            f"Valid BM25 index with {payload.document_count} searchable documents "
            f"({payload.skipped_document_count} skipped) at {config.retriever.index_path}."
        )
        if args.query:
            top_k = args.top_k or config.retriever.top_k
            print_results(retrieve_bm25(args.query, index, top_k=top_k))
        return

    payload = build_bm25_index_from_jsonl(config.input.chunks_path, parameters=config.retriever.parameters())
    output_path = save_bm25_index(payload, config.retriever.index_path, overwrite=args.overwrite)
    index = BM25Index(payload)
    print(
        f"Indexed {payload.document_count} of {payload.source_record_count} chunks into {output_path}; "
        f"skipped {payload.skipped_document_count} chunks without searchable tokens."
    )
    print(f"Source SHA256: {payload.source_sha256}")

    if args.query:
        top_k = args.top_k or config.retriever.top_k
        print_results(retrieve_bm25(args.query, index, top_k=top_k))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"BM25 index operation failed: {error}") from error
