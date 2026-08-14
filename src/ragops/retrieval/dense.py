import time

from pydantic import BaseModel

from ragops.indexing.qdrant import DEFAULT_COLLECTION_NAME, embed_query, search_index
from ragops.retrieval.base import Retriever, resolve_top_k, validate_timings

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


class DenseRetriever(Retriever):
    """Configured dense retriever implementing the common retrieval interface."""

    def __init__(
        self,
        client,
        collection_name=DEFAULT_COLLECTION_NAME,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        default_top_k=DEFAULT_TOP_K,
        clock=time.perf_counter,
        query_embedder=None,
    ):
        super().__init__(default_top_k)
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.clock = clock
        if query_embedder is None:
            query_embedder = embed_query
        if not callable(query_embedder):
            raise ValueError("query_embedder must be callable.")
        self.query_embedder = query_embedder

    def retrieve(self, query, top_k=None, timings=None):
        query = validate_query(query)
        top_k = resolve_top_k(top_k, self.default_top_k)
        validate_timings(timings)
        embedding_started_at = self.clock() if timings is not None else None
        try:
            query_vector = self.query_embedder(query, self.embedding_model)
        finally:
            if embedding_started_at is not None:
                timings["embedding_ms"] = max(0.0, (self.clock() - embedding_started_at) * 1000)

        dense_started_at = self.clock() if timings is not None else None
        try:
            results = search_index(self.client, self.collection_name, query_vector, top_k=top_k)
            return build_retrieved_chunks(results)
        finally:
            if dense_started_at is not None:
                timings["dense_ms"] = max(0.0, (self.clock() - dense_started_at) * 1000)


def retrieve_dense(
    query,
    client,
    top_k=DEFAULT_TOP_K,
    collection_name=DEFAULT_COLLECTION_NAME,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    timings=None,
    clock=time.perf_counter,
    query_embedder=None,
):
    """Embed one query, search Qdrant, and return ranked dense results."""
    retriever = DenseRetriever(
        client,
        collection_name=collection_name,
        embedding_model=embedding_model,
        default_top_k=top_k,
        clock=clock,
        query_embedder=query_embedder,
    )
    return retriever.retrieve(query, timings=timings)
