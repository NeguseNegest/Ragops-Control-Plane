import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ragops import __version__
from ragops.api.schemas import CitationResponse, QueryRequest, QueryResponse, RetrievedChunkResponse, RetrieveRequest, RetrieveResponse
from ragops.generation.client import generate_answer
from ragops.indexing.qdrant import DEFAULT_QDRANT_URL, create_qdrant_client
from ragops.retrieval.dense import retrieve_dense


class HealthResponse(BaseModel):
    status: str
    version: str


def chunk_to_response(chunk):
    """Convert one retrieved chunk into the API response shape."""
    return RetrievedChunkResponse(chunk_id=chunk.chunk_id, document_id=chunk.document_id, text=chunk.text, score=chunk.score, rank=chunk.rank, metadata=chunk.metadata or {}, source_url=chunk.source_url)


def citation_to_response(citation):
    """Convert one citation dictionary into the API response shape."""
    return CitationResponse(citation_id=citation.get("citation_id", ""), document_id=citation.get("document_id", ""), title=citation.get("title", ""), url=citation.get("url"), metadata=citation.get("metadata", {}), chunk_ids=citation.get("chunk_ids", []))


def close_qdrant_client(client):
    """Close the Qdrant client when the installed client exposes close()."""
    close = getattr(client, "close", None)

    if close:
        close()


def get_qdrant_url():
    """Return the configured Qdrant URL with a stable local default."""
    qdrant_url = os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL).strip()

    if not qdrant_url:
        qdrant_url = DEFAULT_QDRANT_URL

    return qdrant_url.rstrip("/")


def retrieve_chunks(query, top_k):
    """Retrieve chunks from Qdrant and close the client afterward."""
    client = create_qdrant_client(get_qdrant_url())

    try:
        return retrieve_dense(query=query, client=client, top_k=top_k)
    finally:
        close_qdrant_client(client)


def elapsed_ms(start_time):
    """Return elapsed milliseconds from a perf_counter start time."""
    return (time.perf_counter() - start_time) * 1000


def create_app():
    app = FastAPI(
        title="RAGOps Control Plane",
        version=__version__,
        summary="Evaluation-gated control plane for RAG systems.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok", version=__version__)

    @app.post("/retrieve", response_model=RetrieveResponse)
    def retrieve(request: RetrieveRequest):
        """Retrieve relevant context chunks for a given query."""
        start_time = time.perf_counter()

        try:
            chunks = retrieve_chunks(request.query, request.top_k)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail="Unable to retrieve chunks.") from error

        response_chunks = [chunk_to_response(chunk) for chunk in chunks]
        return RetrieveResponse(query=request.query, top_k=request.top_k, chunks=response_chunks, latency_ms=elapsed_ms(start_time))

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest):
        """Retrieve context and generate a cited answer."""
        start_time = time.perf_counter()

        try:
            chunks = retrieve_chunks(request.query, request.top_k)
            generation_result = generate_answer(query=request.query, chunks=chunks)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail="Unable to generate answer.") from error

        response_chunks = [chunk_to_response(chunk) for chunk in chunks]
        response_citations = [citation_to_response(citation) for citation in generation_result.citations]
        return QueryResponse(query=request.query, answer=generation_result.answer, citations=response_citations, citation_text=generation_result.citation_text, chunks=response_chunks, used_chunk_ids=generation_result.used_chunk_ids, latency_ms=elapsed_ms(start_time))

    return app


app = create_app()
