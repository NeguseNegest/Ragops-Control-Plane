from pydantic import BaseModel

from ragops.indexing.qdrant import DEFAULT_COLLECTION_NAME, embed_query, search_index

DEFAULT_TOP_K = 5
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RetrievedChunk(BaseModel):
    """A normalized chunk returned by dense retrieval."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    metadata: dict
    source_url: str | None = None


def validate_query(query):
    """Return a cleaned query or raise ValueError when it cannot be retrieved."""
    if not isinstance(query, str):
        raise ValueError("query must be a string.")

    query = query.strip()

    if not query:
        raise ValueError("query must not be empty.")

    return query


def source_url_from_metadata(metadata):
    """Return the best source reference available in chunk metadata."""
    if not isinstance(metadata, dict):
        return None

    for key in ("source_url", "url", "documentation_url", "relative_path", "source_path"):
        value = metadata.get(key)

        if value:
            return str(value)

    return None


def build_retrieved_chunk(result, rank):
    """Convert one Qdrant search result into a RetrievedChunk."""
    payload = result.payload or {}
    metadata = payload.get("metadata") or {}
    score = getattr(result, "score", 0.0)

    if score is None:
        score = 0.0

    return RetrievedChunk(chunk_id=str(payload.get("chunk_id") or result.id), document_id=str(payload.get("document_id") or ""), text=str(payload.get("text") or ""), score=float(score), rank=rank, metadata=metadata, source_url=source_url_from_metadata(metadata))


def build_retrieved_chunks(results):
    """Convert Qdrant search results into ranked RetrievedChunk objects."""
    return [build_retrieved_chunk(result, rank) for rank, result in enumerate(results, start=1)]


def retrieve_dense(query, client, top_k=DEFAULT_TOP_K, collection_name=DEFAULT_COLLECTION_NAME, embedding_model=DEFAULT_EMBEDDING_MODEL):
    """Embed one query, search Qdrant, and return ranked dense results."""
 
    query = validate_query(query)
    query_vector = embed_query(query, embedding_model)
    results = search_index(client, collection_name, query_vector, top_k=top_k)

    return build_retrieved_chunks(results)
