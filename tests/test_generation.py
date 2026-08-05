import pytest

from ragops.generation.citations import build_citations
from ragops.generation.client import GenerationResult, LocalTemplateGenerationClient, build_context, build_generation_prompt, generate_answer, used_chunk_ids, validate_generation_inputs
from ragops.retrieval.dense import RetrievedChunk


def make_chunk(chunk_id="chunk-1", document_id="doc-1", text="FastAPI creates APIs.", metadata=None, rank=1):
    if metadata is None:
        metadata = {"heading": "FastAPI Tutorial", "relative_path": "fastapi/docs/tutorial.md"}

    return RetrievedChunk(chunk_id=chunk_id, document_id=document_id, text=text, score=0.9, rank=rank, metadata=metadata, source_url=metadata.get("relative_path"))


def test_validate_generation_inputs_strips_query():
    chunk = make_chunk()

    assert validate_generation_inputs("  What is FastAPI?  ", [chunk]) == "What is FastAPI?"


def test_validate_generation_inputs_rejects_empty_query_and_chunks():
    chunk = make_chunk()

    with pytest.raises(ValueError, match="empty"):
        validate_generation_inputs("   ", [chunk])

    with pytest.raises(ValueError, match="chunks"):
        validate_generation_inputs("What is FastAPI?", [])


def test_build_context_includes_citation_title_and_chunk_text():
    chunks = [make_chunk()]
    citations = build_citations(chunks)

    context = build_context(chunks, citations)

    assert "Source [1] FastAPI Tutorial" in context
    assert "FastAPI creates APIs." in context


def test_build_generation_prompt_includes_query_context_and_instructions():
    chunks = [make_chunk()]
    citations = build_citations(chunks)

    prompt = build_generation_prompt("What is FastAPI?", chunks, citations)

    assert "What is FastAPI?" in prompt
    assert "FastAPI creates APIs." in prompt
    assert "Use citation IDs like [1]" in prompt
    assert "I do not know" in prompt


def test_local_template_client_returns_cited_answer_when_context_exists():
    client = LocalTemplateGenerationClient()

    answer = client.generate("Context:\nSource [1] FastAPI Tutorial:\nFastAPI creates APIs.")

    assert "[1]" in answer


def test_generate_answer_uses_fake_client_and_returns_generation_result():
    class FakeClient:
        def __init__(self):
            self.prompt = None

        def generate(self, prompt):
            self.prompt = prompt
            return "FastAPI is used to build APIs. [1]"

    chunk = make_chunk()
    client = FakeClient()

    result = generate_answer("What is FastAPI?", [chunk], client=client)

    assert isinstance(result, GenerationResult)
    assert result.answer == "FastAPI is used to build APIs. [1]"
    assert result.citation_text == "[1] FastAPI Tutorial - fastapi/docs/tutorial.md"
    assert result.used_chunk_ids == ["chunk-1"]
    assert "What is FastAPI?" in client.prompt


def test_used_chunk_ids_preserves_retrieval_order():
    chunks = [
        make_chunk("chunk-1", rank=1),
        make_chunk("chunk-2", text="Qdrant stores vectors.", rank=2),
    ]

    assert used_chunk_ids(chunks) == ["chunk-1", "chunk-2"]
