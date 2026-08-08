from typing import Any

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    """Request body for POST /retrieve."""
    query: str = Field(..., description="The user query to retrieve context for.")
    top_k: int = Field(5, ge=1, le=20, description="The number of chunks to retrieve.")


class RetrievedChunkResponse(BaseModel):
    """One retrieved chunk returned by the API."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_url: str | None = None


class RetrieveResponse(BaseModel):
    """Response body for POST /retrieve."""
    query: str
    top_k: int
    chunks: list[RetrievedChunkResponse]
    latency_ms: float


class QueryRequest(BaseModel):
    """Request body for POST /query."""
    query: str = Field(..., description="The user question to answer.")
    top_k: int = Field(5, ge=1, le=20, description="The number of retrieved chunks to use.")


class CitationResponse(BaseModel):
    """One citation returned with a generated answer."""
    citation_id: str
    document_id: str
    title: str
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_ids: list[str]


class QueryResponse(BaseModel):
    """Response body for POST /query."""
    query: str
    answer: str
    citations: list[CitationResponse]
    citation_text: str
    chunks: list[RetrievedChunkResponse]
    used_chunk_ids: list[str]
    latency_ms: float
