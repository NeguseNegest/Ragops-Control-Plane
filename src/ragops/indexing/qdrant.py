import json
from pathlib import Path

from qdrant_client import QdrantClient, models

DEFAULT_COLLECTION_NAME = "rag_chunks"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_BATCH_SIZE = 128


def load_chunk_records(input_path, limit=None):
    """Load embedded chunk records from the Day 6 JSONL file.

    The input is one JSONL file, usually `data/processed/chunks.jsonl`.
    Each line should be one embedded chunk record. This returns records in file
    order and stops early when `limit` is provided.
    """
    records = []

    with Path(input_path).open(encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

            if limit is not None and len(records) >= limit:
                break

    return records


def get_vector_size(records):
    """Find the embedding dimension used by the chunk records.

    This looks at the first record's `embedding` field and returns its length.
    For `sentence-transformers/all-MiniLM-L6-v2`, the expected value is 384.
    """
    if not records:
        raise ValueError("Cannot determine vector size: no records were loaded.")

    embedding = records[0].get("embedding")

    if not embedding:
        raise ValueError("Cannot determine vector size: first record has no embedding.")

    return len(embedding)


def create_qdrant_client(qdrant_url=DEFAULT_QDRANT_URL):
    """Create and return a Qdrant client connected to the local service."""
    return QdrantClient(url=qdrant_url)


def create_collection(client, collection_name, vector_size, recreate=False):
    """Create the Qdrant collection that will store chunk vectors.

    Uses cosine distance. If `recreate` is true, an existing collection is
    deleted and created again. If `recreate` is false and the collection exists,
    this leaves it alone.
    """
    exists = client.collection_exists(collection_name=collection_name)

    if exists and not recreate:
        return False

    if exists and recreate:
        client.delete_collection(collection_name=collection_name)

    client.create_collection(collection_name=collection_name, vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE))
    return True


def build_point(record):
    """Convert one embedded chunk record into a Qdrant point.

    The point ID is the chunk ID, the vector is the embedding, and the payload
    keeps the text plus metadata needed later for citations and debugging.
    """
    payload = {"chunk_id": record["chunk_id"], "document_id": record["document_id"], "text": record["text"], "token_count": record["token_count"], "chunk_hash": record["chunk_hash"], "metadata": record["metadata"]}
    return models.PointStruct(id=record["chunk_id"], vector=record["embedding"], payload=payload)


def batch_records(records, batch_size=DEFAULT_BATCH_SIZE):
    """Group records into fixed-size batches for efficient Qdrant upserts."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    batch = []

    for record in records:
        batch.append(record)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def upsert_records(client, collection_name, records, batch_size=DEFAULT_BATCH_SIZE):
    """Insert or update chunk points in Qdrant and return the indexed count."""
    indexed_count = 0

    for batch in batch_records(records, batch_size=batch_size):
        points = [build_point(record) for record in batch]
        client.upsert(collection_name=collection_name, points=points)
        indexed_count += len(points)

    return indexed_count
 

def index_chunks(input_path, client, collection_name=DEFAULT_COLLECTION_NAME, batch_size=DEFAULT_BATCH_SIZE, recreate=False, limit=None):
    """Run the full Day 7 indexing workflow.

    Loads records, infers vector size, creates the Qdrant collection, upserts
    points, and returns the number of indexed chunks.
    """
    records = load_chunk_records(input_path, limit=limit)
    vector_size = get_vector_size(records)
    create_collection(client, collection_name, vector_size, recreate=recreate)
    return upsert_records(client, collection_name, records, batch_size=batch_size)


def embed_query(query, embedding_model):
    """Embed a search query for the optional sanity check."""
    from ragops.ingestion.embeddings import embed_texts

    return embed_texts([query], model_name=embedding_model, show_progress_bar=False)[0]


def search_index(client, collection_name, query_vector, top_k=5):
    """Search the Qdrant collection with one query vector and return points."""
    response = client.query_points(collection_name=collection_name, query=query_vector, limit=top_k, with_payload=True, with_vectors=False)
    return response.points
