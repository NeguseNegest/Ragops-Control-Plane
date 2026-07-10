from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A parsed source document before chunking."""

    document_id: str = Field(
        ...,
        description="The unique identifier for the document.",
    )
    source_path: str = Field(
        ...,
        description="The raw source path for the document.",
    )
    text: str = Field(
        ...,
        description="The cleaned text of the document.",
    )
    metadata: dict[str, Any] = Field(
        ...,
        description="Document provenance and loader metadata.",
    )


class DocumentChunk(BaseModel):
    """A deterministic chunk derived from a source document."""

    chunk_id: str = Field(
        ...,
        description="The unique identifier for the document chunk.",
    )
    document_id: str = Field(
        ...,
        description="The source document identifier.",
    )
    text: str = Field(
        ...,
        description="The text of the document chunk.",
    )
    token_count: int = Field(
        ...,
        description="The number of tokens in the document chunk.",
    )
    chunk_hash: str = Field(
        ...,
        description="The hash of the document chunk.",
    )
    metadata: dict[str, Any] = Field(
        ...,
        description="Chunk provenance and processing metadata.",
    )
