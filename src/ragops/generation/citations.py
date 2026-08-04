def citation_key(chunk):
    """Return the stable key used to deduplicate citations."""
    metadata = chunk.metadata or {}
    section_value = metadata.get("heading") or metadata.get("section_index")
    return (chunk.document_id, section_value)


def source_title_from_chunk(chunk):
    """Return a readable source title for one retrieved chunk."""
    metadata = chunk.metadata or {}

    for key in ("heading", "source_title", "title", "relative_path"):
        value = metadata.get(key)

        if value:
            return str(value)

    return str(chunk.document_id)


def source_url_from_chunk(chunk):
    """Return the best URL or path available for one retrieved chunk."""
    if getattr(chunk, "source_url", None):
        return str(chunk.source_url)

    metadata = chunk.metadata or {}

    for key in ("source_url", "url", "documentation_url", "relative_path", "source_path"):
        value = metadata.get(key)

        if value:
            return str(value)

    return None


def make_citation(citation_number, chunk):
    """Create the first citation dictionary for one source."""
    return {"citation_id": f"[{citation_number}]", "document_id": chunk.document_id, "title": source_title_from_chunk(chunk), "url": source_url_from_chunk(chunk), "metadata": chunk.metadata or {}, "chunk_ids": [chunk.chunk_id]}


def add_chunk_to_citation(citation, chunk):
    """Add another chunk ID to an existing citation if needed."""
    if chunk.chunk_id not in citation["chunk_ids"]:
        citation["chunk_ids"].append(chunk.chunk_id)

    return citation


def build_citations(chunks):
    """Build ordered citations and deduplicate chunks from the same source section."""
    citations = []
    citations_by_key = {}

    for chunk in chunks:
        key = citation_key(chunk)

        if key not in citations_by_key:
            citation = make_citation(len(citations) + 1, chunk)
            citations.append(citation)
            citations_by_key[key] = citation
        else:
            add_chunk_to_citation(citations_by_key[key], chunk)

    return citations


def format_citation(citation):
    """Format one citation dictionary into a readable text line."""
    text = f"{citation['citation_id']} {citation['title']}"

    if citation.get("url"):
        return f"{text} - {citation['url']}"

    return text


def format_citations(citations):
    """Format all citations into newline-separated citation lines."""
    if not citations:
        return ""

    return "\n".join(format_citation(citation) for citation in citations)
