from pydantic import BaseModel

from ragops.generation.citations import build_citations, format_citations


class GenerationResult(BaseModel):
    """Output from one grounded generation call."""

    answer: str
    citations: list
    citation_text: str
    used_chunk_ids: list


class GenerationClient:

    def generate(self, prompt):
        """Generate answer text from one completed prompt."""
        raise NotImplementedError("Generation clients must implement generate(prompt).")


class LocalTemplateGenerationClient(GenerationClient):

    def generate(self, prompt):
        """Return a simple cited answer without calling an external model."""
        if "Source [" not in prompt:
            return "I do not know."

        return "Based on the provided context, the answer is supported by the retrieved documentation. [1]"


def validate_generation_inputs(query, chunks):
    """Validate generation inputs and return the cleaned query."""
    if not isinstance(query, str):
        raise ValueError("query must be a string.")

    query = query.strip()

    if not query:
        raise ValueError("query must not be empty.")

    if not chunks:
        raise ValueError("chunks must not be empty.")

    return query


def citation_for_chunk(chunk, citations):
    """Return the citation dictionary that contains the chunk ID."""
    for citation in citations:
        if chunk.chunk_id in citation.get("chunk_ids", []):
            return citation

    return None


def build_context(chunks, citations):
    """Build plain-text context from retrieved chunks and citation IDs."""
    context_blocks = []

    for chunk in chunks:
        citation = citation_for_chunk(chunk, citations)
        citation_id = citation["citation_id"] if citation else ""
        source_title = citation["title"] if citation else ""
        header = f"{citation_id} {source_title}".strip()
        context_blocks.append(f"Source {header}:\n{chunk.text}")

    return "\n\n".join(context_blocks)


def build_generation_prompt(query, chunks, citations):
    """Build the prompt used by the generation client."""
    context = build_context(chunks, citations)
    return (
        "Answer the question using only the provided context.\n"
        "If the context does not contain the answer, say \"I do not know.\"\n"
        "Use citation IDs like [1] whenever you use information from the context.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def used_chunk_ids(chunks):
    """Return chunk IDs used as generation context."""
    return [chunk.chunk_id for chunk in chunks]


def generate_answer(query, chunks, client=None):
    """Generate one grounded answer from a query and retrieved chunks."""
    chunks = list(chunks or [])
    query = validate_generation_inputs(query, chunks)
    citations = build_citations(chunks)
    prompt = build_generation_prompt(query, chunks, citations)

    if client is None:
        client = LocalTemplateGenerationClient()

    answer = client.generate(prompt)
    return GenerationResult(answer=answer, citations=citations, citation_text=format_citations(citations), used_chunk_ids=used_chunk_ids(chunks))
