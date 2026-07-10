import uuid
from pathlib import Path

from ragops.ingestion.cleaning import clean_text
from ragops.schemas import Document

SUPPORTED_EXTENSIONS = {".md", ".mdx", ".rst", ".txt", ".html", ".py"}
_TEXT_EXTENSIONS = SUPPORTED_EXTENSIONS - {".py"}
_FASTAPI_DOCS_SRC_PREFIX = Path("fastapi/docs_src")
_DOCUMENT_ID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def is_supported_path(path, raw_root="data/raw"):
    """Return whether a raw file should be loaded as a document."""
    path_obj = Path(path)
    extension = path_obj.suffix.lower()

    if extension in _TEXT_EXTENSIONS:
        return True

    if extension == ".py":
        relative_path = relative_to_raw_root(path_obj, raw_root)
        return is_fastapi_docs_source(relative_path)

    return False


def is_fastapi_docs_source(relative_path):
    """Return whether a relative path points to FastAPI docs example code."""
    relative_path = Path(relative_path)
    return len(relative_path.parts) >= 2 and relative_path.parts[:2] == (
        _FASTAPI_DOCS_SRC_PREFIX.parts
    )


def relative_to_raw_root(path, raw_root):
    """Return a stable path relative to the raw data root."""
    path_obj = Path(path)
    raw_root_obj = Path(raw_root)

    try:
        return path_obj.relative_to(raw_root_obj)
    except ValueError:
        return path_obj


def load_document(path, raw_root="data/raw"):
    """Load a supported raw file into a Document."""
    path_obj = Path(path)
    raw_root_obj = Path(raw_root)
    full_path = resolve_full_path(path_obj, raw_root_obj)
    relative_path = relative_to_raw_root(full_path, raw_root_obj)

    if not is_supported_path(relative_path, raw_root="."):
        return None

    raw_text = full_path.read_text(encoding="utf-8")
    extension = full_path.suffix.lower()
    cleaned_text = clean_text(raw_text, extension)

    if not cleaned_text:
        return None

    return Document(
        document_id=build_document_id(relative_path),
        source_path=full_path.as_posix(),
        text=cleaned_text,
        metadata=build_metadata(full_path, relative_path),
    )


def resolve_full_path(path, raw_root):
    """Resolve either a raw-root-relative path or a path already under raw_root."""
    if path.is_absolute():
        return path

    try:
        path.relative_to(raw_root)
    except ValueError:
        return raw_root / path

    return path


def build_document_id(relative_path):
    """Build a deterministic document ID from the raw-root-relative path."""
    return str(uuid.uuid5(_DOCUMENT_ID_NAMESPACE, Path(relative_path).as_posix()))


def build_metadata(full_path, relative_path):
    """Build provenance metadata for a loaded document."""
    extension = full_path.suffix.lower()

    metadata = {
        "source_path": full_path.as_posix(),
        "relative_path": relative_path.as_posix(),
        "extension": extension,
        "source_name": infer_source_name(relative_path),
    }

    if extension == ".py":
        metadata["content_type"] = "code_example"
        metadata["language"] = "python"
    else:
        metadata["content_type"] = "documentation"

    return metadata


def infer_source_name(path):
    """Infer corpus source from a raw-root-relative path."""
    path_obj = Path(path)
    if path_obj.parts:
        return path_obj.parts[0]
    return "unknown"


def iter_documents(raw_root="data/raw"):
    """Yield supported documents from a raw data directory in stable order."""
    raw_root_obj = Path(raw_root)

    for current_path in sorted(raw_root_obj.rglob("*")):
        if not current_path.is_file():
            continue

        document = load_document(current_path, raw_root_obj)
        if document is not None:
            yield document