from ragops.generation.citations import build_citations, citation_key, format_citation, format_citations, source_title_from_chunk, source_url_from_chunk
from ragops.retrieval.dense import RetrievedChunk


def make_chunk(chunk_id="chunk-1", document_id="doc-1", metadata=None, source_url=None, rank=1):
    if metadata is None:
        metadata = {"relative_path": "docs/example.md", "source_name": "test"}

    return RetrievedChunk(chunk_id=chunk_id, document_id=document_id, text=f"Text for {chunk_id}", score=0.9, rank=rank, metadata=metadata, source_url=source_url)


def test_citation_key_uses_document_and_heading():
    chunk = make_chunk(metadata={"heading": "Install", "relative_path": "docs/install.md"})

    assert citation_key(chunk) == ("doc-1", "Install")


def test_source_title_prefers_heading_then_relative_path():
    heading_chunk = make_chunk(metadata={"heading": "Quickstart", "relative_path": "docs/quickstart.md"})
    path_chunk = make_chunk(metadata={"relative_path": "docs/example.md"})

    assert source_title_from_chunk(heading_chunk) == "Quickstart"
    assert source_title_from_chunk(path_chunk) == "docs/example.md"


def test_source_url_prefers_chunk_source_url_then_metadata_path():
    chunk_with_url = make_chunk(source_url="https://example.com/docs")
    chunk_with_path = make_chunk(metadata={"relative_path": "docs/example.md"})

    assert source_url_from_chunk(chunk_with_url) == "https://example.com/docs"
    assert source_url_from_chunk(chunk_with_path) == "docs/example.md"


def test_build_citations_deduplicates_same_document_and_section():
    chunks = [
        make_chunk("chunk-1", metadata={"heading": "Install", "relative_path": "docs/install.md"}, rank=1),
        make_chunk("chunk-2", metadata={"heading": "Install", "relative_path": "docs/install.md"}, rank=2),
        make_chunk("chunk-3", metadata={"heading": "Deploy", "relative_path": "docs/deploy.md"}, rank=3),
    ]

    citations = build_citations(chunks)

    assert [citation["citation_id"] for citation in citations] == ["[1]", "[2]"]
    assert citations[0]["chunk_ids"] == ["chunk-1", "chunk-2"]
    assert citations[1]["chunk_ids"] == ["chunk-3"]


def test_build_citations_assigns_new_id_for_different_documents():
    chunks = [
        make_chunk("chunk-1", document_id="doc-1", metadata={"heading": "Install", "relative_path": "docs/a.md"}),
        make_chunk("chunk-2", document_id="doc-2", metadata={"heading": "Install", "relative_path": "docs/b.md"}),
    ]

    citations = build_citations(chunks)

    assert [citation["citation_id"] for citation in citations] == ["[1]", "[2]"]
    assert [citation["document_id"] for citation in citations] == ["doc-1", "doc-2"]


def test_format_citation_includes_id_title_and_url():
    citation = {"citation_id": "[1]", "title": "FastAPI Tutorial", "url": "fastapi/docs/tutorial.md"}

    assert format_citation(citation) == "[1] FastAPI Tutorial - fastapi/docs/tutorial.md"


def test_format_citations_joins_lines_and_handles_empty_list():
    citations = [
        {"citation_id": "[1]", "title": "FastAPI", "url": "fastapi/docs.md"},
        {"citation_id": "[2]", "title": "Qdrant", "url": "qdrant/docs.md"},
    ]

    assert format_citations(citations) == "[1] FastAPI - fastapi/docs.md\n[2] Qdrant - qdrant/docs.md"
    assert format_citations([]) == ""
