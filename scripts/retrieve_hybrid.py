import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.indexing.qdrant import create_qdrant_client  # noqa: E402
from ragops.retrieval.bm25 import load_bm25_index, validate_bm25_index  # noqa: E402
from ragops.retrieval.hybrid import FUSION_METADATA_KEY, configured_qdrant_url, load_hybrid_config, retrieve_hybrid_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieve dense and BM25 candidates and fuse them with Reciprocal Rank Fusion.")
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid.yaml"), help="Hybrid YAML configuration.")
    parser.add_argument("--query", help="Query to retrieve when not validating only.")
    parser.add_argument("--qdrant-url", help="Optional Qdrant URL override.")
    parser.add_argument("--json", action="store_true", help="Print fused results as JSON instead of readable text.")
    parser.add_argument("--validate-only", action="store_true", help="Validate config and BM25 index provenance without connecting to Qdrant.")
    return parser.parse_args()


def apply_overrides(config, qdrant_url=None):
    if qdrant_url is None:
        return config
    dense = config.dense.model_copy(update={"qdrant_url": qdrant_url.strip().rstrip("/") or None})
    return config.model_copy(update={"dense": dense})


def close_client(client):
    close = getattr(client, "close", None)
    if close:
        close()


def readable_result(result):
    fusion = result.metadata.get(FUSION_METADATA_KEY, {})
    sources = fusion.get("sources", {})
    dense_rank = sources.get("dense", {}).get("rank", "-")
    bm25_rank = sources.get("bm25", {}).get("rank", "-")
    source = result.source_url or "unknown"
    preview = " ".join(result.text.split())
    if len(preview) > 180:
        preview = f"{preview[:177]}..."
    return (
        f"#{result.rank} rrf={result.score:.6f} dense_rank={dense_rank} bm25_rank={bm25_rank} "
        f"chunk={result.chunk_id} source={source}\n  {preview}"
    )


def main():
    args = parse_args()
    if not args.validate_only and (not isinstance(args.query, str) or not args.query.strip()):
        raise ValueError("--query is required unless --validate-only is used.")

    config = apply_overrides(load_hybrid_config(args.config, project_root=PROJECT_ROOT), qdrant_url=args.qdrant_url)
    index = load_bm25_index(config.bm25.index_path)
    payload = validate_bm25_index(index, config.bm25_validation_config())

    if args.validate_only:
        print(
            f"Valid hybrid config '{config.name}': dense top {config.dense.top_k} + BM25 top {config.bm25.top_k} "
            f"-> RRF top {config.fusion.top_k} (k={config.fusion.rank_constant:g}); "
            f"BM25 index has {payload.document_count} documents."
        )
        return

    qdrant_url = configured_qdrant_url(config)
    client = create_qdrant_client(qdrant_url)
    try:
        if not client.collection_exists(collection_name=config.dense.collection_name):
            raise RuntimeError(f"Qdrant collection does not exist: {config.dense.collection_name}")
        results = retrieve_hybrid_config(args.query, config, client, index)
    finally:
        close_client(client)

    if args.json:
        print(json.dumps([result.model_dump(mode="json") for result in results], ensure_ascii=False, indent=2))
        return

    print(
        f"Hybrid results for {args.query.strip()!r}: dense top {config.dense.top_k} + BM25 top {config.bm25.top_k} "
        f"-> RRF top {config.fusion.top_k}"
    )
    for result in results:
        print(readable_result(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Hybrid retrieval failed: {error}") from error
