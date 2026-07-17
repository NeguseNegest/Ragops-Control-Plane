import pytest

from ragops.ingestion.chunking import (
    ChunkConfig,
    chunk_document,
    chunk_documents,
    split_by_headings,
    split_fixed_size,
    split_with_overlap,
)
from ragops.schemas import Document


def make_document(text, document_id="doc-1", metadata=None):
    if metadata is None:
        metadata = {
            "relative_path": "docs/example.md",
            "source_name": "test",
            "content_type": "documentation",
        }

    return Document(
        document_id=document_id,
        source_path="data/raw/docs/example.md",
        text=text,
        metadata=metadata,
    )


def test_fixed_size_chunks_do_not_overlap():
    document = make_document("one two three four five six seven")

    chunks = split_fixed_size(document, chunk_size=3, chunk_overlap=0)

    assert [chunk.text for chunk in chunks] == [
        "one two three",
        "four five six",
        "seven",
    ]
    assert [chunk.token_count for chunk in chunks] == [3, 3, 1]
    assert all(chunk.metadata["chunk_overlap"] == 0 for chunk in chunks)


def test_overlapping_chunks_reuse_trailing_tokens():
    document = make_document("one two three four five six seven")

    chunks = split_with_overlap(document, chunk_size=3, chunk_overlap=1)

    assert [chunk.text for chunk in chunks] == [
        "one two three",
        "three four five",
        "five six seven",
    ]
    assert chunks[1].metadata["chunk_start_token"] == 2
    assert chunks[1].metadata["chunk_end_token"] == 5


def test_chunk_ids_and_hashes_are_deterministic():
    document = make_document("alpha beta gamma delta epsilon")
    config = ChunkConfig(chunk_size=2, chunk_overlap=1, strategy="with_overlap")

    first_run = chunk_document(document, config)
    second_run = chunk_document(document, config)

    assert [chunk.chunk_id for chunk in first_run] == [
        chunk.chunk_id for chunk in second_run
    ]
    assert [chunk.chunk_hash for chunk in first_run] == [
        chunk.chunk_hash for chunk in second_run
    ]
    assert all(len(chunk.chunk_hash) == 64 for chunk in first_run)


def test_heading_chunks_keep_heading_metadata():
    document = make_document(
        "# Intro { #intro }\nalpha beta\n\n## Install\ngamma delta",
    )

    chunks = split_by_headings(document, chunk_size=10, chunk_overlap=2)

    assert [chunk.metadata.get("heading") for chunk in chunks] == [
        "Intro",
        "Install",
    ]
    assert [chunk.metadata.get("heading_level") for chunk in chunks] == [1, 2]
    assert chunks[0].text.startswith("# Intro")
    assert chunks[1].text.startswith("## Install")


def test_heading_strategy_overlaps_long_sections():
    document = make_document("# Intro\none two three four five six")
    config = ChunkConfig(chunk_size=4, chunk_overlap=2, strategy="heading")

    chunks = chunk_document(document, config)

    assert [chunk.text for chunk in chunks] == [
        "# Intro\none two",
        "one two three four",
        "three four five six",
    ]
    assert all(chunk.metadata["heading"] == "Intro" for chunk in chunks)


def test_code_spacing_is_preserved_inside_chunks():
    document = make_document(
        "def app():\n    return True\n\nprint(app())",
        metadata={
            "relative_path": "fastapi/docs_src/example.py",
            "source_name": "fastapi",
            "content_type": "code_example",
            "language": "python",
        },
    )

    chunks = split_with_overlap(document, chunk_size=3, chunk_overlap=1)

    assert chunks[0].text == "def app():\n    return"


def test_chunk_documents_yields_chunks_in_document_order():
    first = make_document("alpha beta gamma", document_id="doc-1")
    second = make_document("delta epsilon zeta", document_id="doc-2")
    config = ChunkConfig(chunk_size=2, chunk_overlap=0, strategy="fixed")

    chunks = list(chunk_documents([first, second], config))

    assert [chunk.document_id for chunk in chunks] == [
        "doc-1",
        "doc-1",
        "doc-2",
        "doc-2",
    ]


def test_invalid_chunk_config_rejects_bad_overlap():
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkConfig(chunk_size=3, chunk_overlap=3)
