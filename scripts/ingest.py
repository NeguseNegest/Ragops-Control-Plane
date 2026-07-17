import argparse
import json
import sys
from pathlib import Path

from ragops.schemas import DocumentChunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.ingestion.chunking import ChunkConfig, chunk_documents  # noqa: E402
from ragops.ingestion.loaders import is_supported_path, iter_documents  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse raw documentation files into cleaned documents.",
    )
    parser.add_argument(
        "--raw-root",
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing raw source files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print parsed document summaries without writing output files.",
        default=False,
    )


    parser.add_argument(
        "--preview-count",
        type=int,
        default=5,
        help="Number of parsed documents to preview.",
    )
    parser.add_argument(
        "--preview-chunks",
        type=int,
        default=5,
        help="Number of generated chunks to preview.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Target chunk size in whitespace tokens.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Token overlap between neighboring chunks.",
    )
    parser.add_argument(
        "--chunk-strategy",
        default="heading",
        help="Chunking strategy: fixed_size, with_overlap, or heading.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="JSONL path for processed chunks with embeddings.",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    if args.preview_count < 0:
        parser.error("--preview-count must be zero or greater.")

    if args.preview_chunks < 0:
        parser.error("--preview-chunks must be zero or greater.")

    try:
        args.chunk_config = ChunkConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            strategy=args.chunk_strategy,
        )
    except ValueError as exc:
        parser.error(str(exc))

    return args


def summarize_files(raw_root, chunk_config):

    raw_root = Path(raw_root)
    files = sorted(path for path in raw_root.rglob("*") if path.is_file())

    supported_files = [
        path for path in files if is_supported_path(path, raw_root=raw_root)
    ]

    documents = list(iter_documents(raw_root))
    chunks = list(chunk_documents(documents, chunk_config))

    return {
        "raw_root": raw_root,
        "files_scanned": len(files),
        "supported_files": len(supported_files),
        "parsed_documents": len(documents),
        "unsupported_or_empty_files": len(files) - len(documents),
        "supported_but_empty_files": len(supported_files) - len(documents),
        "documents": documents,
        "chunks": chunks,
        "chunk_config": chunk_config,
    }


def print_summary(summary):
    print("Ingestion dry run")
    print("=================")
    print(f"Raw directory: {summary['raw_root']}")
    print(f"Files scanned: {summary['files_scanned']}")
    print(f"Supported files: {summary['supported_files']}")
    print(f"Parsed documents: {summary['parsed_documents']}")
    print(f"Generated chunks: {len(summary['chunks'])}")
    print(f"Skipped files: {summary['unsupported_or_empty_files']}")
    print(
        "Chunking: "
        f"{summary['chunk_config'].strategy}, "
        f"size={summary['chunk_config'].chunk_size}, "
        f"overlap={summary['chunk_config'].chunk_overlap}"
    )

    empty_count = summary["supported_but_empty_files"]
    if empty_count:
        print(f"Supported but empty files: {empty_count}")


def print_previews(documents, preview_count):
    if preview_count == 0:
        return

    print()
    print(f"Document previews ({min(preview_count, len(documents))})")
    print("=" * 19)

    for index, document in enumerate(documents[:preview_count], start=1):
        metadata = document.metadata
        preview = preview_text(document.text)

        print(f"{index}. {metadata['relative_path']}")
        print(f"   document_id: {document.document_id}")
        print(f"   source_name: {metadata['source_name']}")
        print(f"   content_type: {metadata['content_type']}")
        print(f"   extension: {metadata['extension']}")
        print(f"   preview: {preview}")


def print_chunk_previews(chunks, preview_count):
    if preview_count == 0:
        return

    print()
    print(f"Chunk previews ({min(preview_count, len(chunks))})")
    print("=" * 17)

    for index, chunk in enumerate(chunks[:preview_count], start=1):
        metadata = chunk.metadata
        preview = preview_text(chunk.text)
        heading = metadata.get("heading")

        print(
            f"{index}. {metadata['relative_path']} "
            f":: chunk {metadata['chunk_index']}"
        )
        print(f"   chunk_id: {chunk.chunk_id}")
        print(f"   document_id: {chunk.document_id}")
        print(f"   chunk_hash: {chunk.chunk_hash}")
        print(f"   token_count: {chunk.token_count}")

        if heading:
            print(f"   heading: {heading}")

        print(f"   preview: {preview}")


def write_embedded_chunks(chunks, output_path, embedding_model, embedding_batch_size, embedder=None):
    if embedder is None:
        from ragops.ingestion.embeddings import embed_texts

        embedder = embed_texts

    texts = [chunk.text for chunk in chunks]
    embeddings = embedder(texts, model_name=embedding_model, batch_size=embedding_batch_size, show_progress_bar=True)

    if len(embeddings) != len(chunks):
        raise ValueError("Embedding count must match chunk count")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            output_file.write(json.dumps(build_embedded_chunk_record(chunk, embedding, embedding_model), ensure_ascii=False))
            output_file.write("\n")


def build_embedded_chunk_record(chunk: DocumentChunk, embedding, embedding_model):
    metadata = dict(chunk.metadata)
    metadata["embedding_model"] = embedding_model
    metadata["embedding_dimension"] = len(embedding)

    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "chunk_hash": chunk.chunk_hash,
        "metadata": metadata,
        "embedding": embedding,
    }


def preview_text(text, max_chars=300):
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}"


def main():
    args = parse_args()
    summary = summarize_files(args.raw_root, args.chunk_config)

    if args.dry_run:
        print_summary(summary)
        print_previews(summary["documents"], args.preview_count)
        print_chunk_previews(summary["chunks"], args.preview_chunks)
        return

    write_embedded_chunks(
        summary["chunks"],
        args.output,
        args.embedding_model,
        args.embedding_batch_size,
    )
    print(f"Wrote {len(summary['chunks'])} embedded chunks to {args.output}")


if __name__ == "__main__":
    main()
