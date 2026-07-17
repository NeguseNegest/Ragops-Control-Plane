import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.indexing.qdrant import DEFAULT_BATCH_SIZE, DEFAULT_COLLECTION_NAME, DEFAULT_QDRANT_URL, create_qdrant_client, embed_query, index_chunks, search_index  # noqa: E402


def parse_args():
    """Parse command-line options for building the Qdrant index."""
    parser = argparse.ArgumentParser(description="Build the Qdrant index from embedded chunks.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/chunks.jsonl"), help="Path to embedded chunks JSONL.")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL), help="Qdrant HTTP URL.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Qdrant collection name.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of points to upsert per batch.")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the collection before indexing.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of chunks to index for a quick test run.")
    parser.add_argument("--query", default=None, help="Optional query text for a search sanity check after indexing.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of search results to show for the sanity check.")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model for the optional query sanity check.")
    return parser.parse_args()


def validate_args(args):
    """Validate CLI arguments before connecting to Qdrant."""
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero.")

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero.")

    if args.top_k <= 0:
        raise SystemExit("--top-k must be greater than zero.")

    if not args.input.is_file():
        raise SystemExit(f"Input file does not exist: {args.input}")


def preview_text(text, max_chars=200):
    """Return a compact one-line preview of a chunk's text."""
    text = " ".join(str(text).split())

    if len(text) <= max_chars:
        return text

    return f"{text[: max_chars - 3]}..."


def print_search_results(results):
    """Print readable search results from the optional sanity check."""
    if not results:
        print("No search results found.")
        return

    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}
        metadata = payload.get("metadata") or {}
        score = getattr(result, "score", None)
        score_text = f"{score:.4f}" if score is not None else "n/a"
        source = metadata.get("relative_path") or metadata.get("source_path") or "unknown"

        print(f"\nResult {rank}")
        print(f"  Score:  {score_text}")
        print(f"  Chunk:  {payload.get('chunk_id', result.id)}")
        print(f"  Source: {source}")
        print(f"  Text:   {preview_text(payload.get('text', ''))}")


def main():
    """Run the Day 7 indexing CLI."""
    args = parse_args()

    if not args.input.is_absolute():
        args.input = PROJECT_ROOT / args.input

    validate_args(args)
    client = create_qdrant_client(args.qdrant_url)

    try:
        indexed_count = index_chunks(input_path=args.input, client=client, collection_name=args.collection, batch_size=args.batch_size, recreate=args.recreate, limit=args.limit)
        print(f"Indexed {indexed_count} chunk(s) into collection {args.collection!r}.")

        if args.query:
            query_vector = embed_query(args.query, args.embedding_model)
            results = search_index(client, args.collection, query_vector, top_k=args.top_k)
            print_search_results(results)
    finally:
        client.close()


if __name__ == "__main__":
    main()
