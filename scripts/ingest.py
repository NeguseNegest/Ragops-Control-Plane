import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=5,
        help="Number of parsed documents to preview.",
    )

    args = parser.parse_args()

    if not args.dry_run:
        parser.error("Day 4 ingestion currently supports --dry-run only.")

    if args.preview_count < 0:
        parser.error("--preview-count must be zero or greater.")

    return args


def summarize_files(raw_root):

    raw_root = Path(raw_root)
    files = sorted(path for path in raw_root.rglob("*") if path.is_file())

    supported_files = [
        path for path in files if is_supported_path(path, raw_root=raw_root)
    ]

    documents = list(iter_documents(raw_root))

    return {
        "raw_root": raw_root,
        "files_scanned": len(files),
        "supported_files": len(supported_files),
        "parsed_documents": len(documents),
        "unsupported_or_empty_files": len(files) - len(documents),
        "supported_but_empty_files": len(supported_files) - len(documents),
        "documents": documents,
    }


def print_summary(summary):
    print("Ingestion dry run")
    print("=================")
    print(f"Raw directory: {summary['raw_root']}")
    print(f"Files scanned: {summary['files_scanned']}")
    print(f"Supported files: {summary['supported_files']}")
    print(f"Parsed documents: {summary['parsed_documents']}")
    print(f"Skipped files: {summary['unsupported_or_empty_files']}")

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


def preview_text(text, max_chars=300):
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."


def main():
    args = parse_args()
    summary = summarize_files(args.raw_root)
    print_summary(summary)
    print_previews(summary["documents"], args.preview_count)


if __name__ == "__main__":
    main()
