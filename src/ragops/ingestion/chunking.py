import re
import uuid
from bisect import bisect_left, bisect_right
from hashlib import sha256

from ragops.schemas import Document, DocumentChunk

CHUNK_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "ragops-control-plane/chunks/v1")

DEFAULT_CHUNK_SIZE = 250
DEFAULT_CHUNK_OVERLAP = 50

_TOKEN_RE = re.compile(r"\S+")
_HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
_HEADING_ANCHOR_RE = re.compile(r"\s*\{\s*#?[^}]+\}\s*$")


class ChunkConfig:
    """Configuration for deterministic document chunking."""

    def __init__(self, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, strategy="heading"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = normalize_strategy(strategy)
        validate_chunk_settings(self.chunk_size, self.chunk_overlap)


def normalize_strategy(strategy):
    """Normalize supported strategy aliases into stable metadata values."""
    normalized = strategy.strip().lower().replace("-", "_")
    aliases = {"fixed": "fixed_size", "fixed_size": "fixed_size", "overlap": "with_overlap", "overlapping": "with_overlap", "with_overlap": "with_overlap", "heading": "heading", "headings": "heading", "heading_aware": "heading"}

    if normalized not in aliases:
        supported = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported chunking strategy: {strategy}. Use one of: {supported}")

    return aliases[normalized]


def validate_chunk_settings(chunk_size, chunk_overlap):
    """Validate token-window settings before chunking starts."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def chunk_document(document, chunk_config=None):
    """Chunk one document using the configured strategy."""
    if chunk_config is None:
        chunk_config = ChunkConfig()

    if chunk_config.strategy == "fixed_size":
        return split_fixed_size(document, chunk_config.chunk_size, chunk_config.chunk_overlap)

    if chunk_config.strategy == "with_overlap":
        return split_with_overlap(document, chunk_config.chunk_size, chunk_config.chunk_overlap)

    if chunk_config.strategy == "heading":
        return split_by_headings(document, chunk_config.chunk_size, chunk_config.chunk_overlap)

    raise ValueError(f"Unsupported chunking strategy: {chunk_config.strategy}")


def chunk_documents(documents, chunk_config=None):
    """Yield chunks for documents in the same stable order as the input."""
    if chunk_config is None:
        chunk_config = ChunkConfig()

    for document in documents:
        yield from chunk_document(document, chunk_config)


def split_fixed_size(document: Document, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=0):
    """Split a document into fixed-size token windows without overlap."""
    validate_chunk_settings(chunk_size, chunk_overlap)
    spans = token_spans(document.text)
    windows = build_token_windows(document.text, spans, chunk_size=chunk_size, chunk_overlap=0)
    return build_chunks(document, windows, strategy="fixed_size", chunk_size=chunk_size, chunk_overlap=0)


def split_with_overlap(document: Document, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Split a document into overlapping token windows."""
    validate_chunk_settings(chunk_size, chunk_overlap)
    spans = token_spans(document.text)
    windows = build_token_windows(document.text, spans, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return build_chunks(document, windows, strategy="with_overlap", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_by_headings(document: Document, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Split by Markdown headings, then token-window long sections."""
    validate_chunk_settings(chunk_size, chunk_overlap)
    spans = token_spans(document.text)

    if not spans:
        return []

    sections = heading_sections(document.text, spans)
    windows = []

    for section_index, section in enumerate(sections):
        section_windows = build_token_windows(document.text, spans, chunk_size=chunk_size, chunk_overlap=chunk_overlap, start_token=section["start_token"], end_token=section["end_token"])

        for window in section_windows:
            window["section_index"] = section_index
            window["heading"] = section["heading"]
            window["heading_level"] = section["heading_level"]
            windows.append(window)

    return build_chunks(document, windows, strategy="heading", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def token_spans(text):
    """Return start and end character offsets for each whitespace token."""
    return [(match.start(), match.end()) for match in _TOKEN_RE.finditer(text)]


def build_token_windows(text, spans, chunk_size, chunk_overlap, start_token=0, end_token=None):
    """Build token windows while preserving original text between token offsets."""
    if end_token is None:
        end_token = len(spans)

    if start_token >= end_token:
        return []

    windows = []
    step = chunk_size - chunk_overlap
    current_start = start_token

    while current_start < end_token:
        current_end = min(current_start + chunk_size, end_token)
        start_char = spans[current_start][0]
        end_char = spans[current_end - 1][1]
        chunk_text = text[start_char:end_char].strip()

        if chunk_text:
            windows.append({"text": chunk_text, "start_token": current_start, "end_token": current_end, "start_char": start_char, "end_char": end_char})

        if current_end >= end_token:
            break

        current_start += step

    return windows


def heading_sections(text, spans):
    """Return token ranges for best-effort Markdown heading sections."""
    heading_matches = list(_iter_heading_matches(text))
    span_starts = [span[0] for span in spans]
    span_ends = [span[1] for span in spans]

    if not heading_matches:
        return [{"start_token": 0, "end_token": len(spans), "heading": None, "heading_level": None}]

    sections = []

    if heading_matches[0]["start_char"] > 0:
        sections.append(build_section(spans, span_starts, span_ends, start_char=0, end_char=heading_matches[0]["start_char"], heading=None, heading_level=None))

    for index, heading in enumerate(heading_matches):
        next_start = heading_matches[index + 1]["start_char"] if index + 1 < len(heading_matches) else len(text)
        sections.append(build_section(spans, span_starts, span_ends, start_char=heading["start_char"], end_char=next_start, heading=heading["heading"], heading_level=heading["level"]))

    return [section for section in sections if section is not None]


def build_section(spans, span_starts, span_ends, start_char, end_char, heading, heading_level):
    """Map a character range onto the global token range."""
    start_token = bisect_right(span_ends, start_char)
    end_token = bisect_left(span_starts, end_char)

    if start_token >= end_token:
        return None

    return {"start_token": start_token, "end_token": end_token, "heading": heading, "heading_level": heading_level}


def build_chunks(document, windows, strategy, chunk_size, chunk_overlap):
    """Convert token windows into DocumentChunk records."""
    chunks = []

    for chunk_index, window in enumerate(windows):
        chunk_text = window["text"]
        chunk_hash = build_chunk_hash(chunk_text)
        metadata = dict(document.metadata)
        metadata.update({"chunk_index": chunk_index, "chunking_strategy": strategy, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "chunk_start_token": window["start_token"], "chunk_end_token": window["end_token"], "chunk_start_char": window["start_char"], "chunk_end_char": window["end_char"]})

        if "section_index" in window:
            metadata["section_index"] = window["section_index"]

        if window.get("heading") is not None:
            metadata["heading"] = window["heading"]
            metadata["heading_level"] = window["heading_level"]

        chunks.append(DocumentChunk(chunk_id=build_chunk_id(document.document_id, strategy, chunk_index, chunk_hash), document_id=document.document_id, text=chunk_text, token_count=window["end_token"] - window["start_token"], chunk_hash=chunk_hash, metadata=metadata))

    return chunks


def build_chunk_hash(text):
    """Build a SHA256 hash from exact chunk text."""
    return sha256(text.encode("utf-8")).hexdigest()


def build_chunk_id(document_id, strategy, chunk_index, chunk_hash):
    """Build a deterministic UUID5 chunk ID."""
    name = f"{document_id}:{strategy}:{chunk_index}:{chunk_hash}"
    return str(uuid.uuid5(CHUNK_NAMESPACE, name))


def _iter_heading_matches(text):
    current_char = 0

    for line in text.splitlines(keepends=True):
        line_without_newline = line.rstrip("\n")
        match = _HEADING_RE.match(line_without_newline.strip())

        if match:
            yield {"start_char": current_char, "level": len(match.group("marks")), "heading": clean_heading_title(match.group("title"))}

        current_char += len(line)


def clean_heading_title(title):
    """Normalize heading text stored in metadata."""
    title = _HEADING_ANCHOR_RE.sub("", title).strip()
    return title
