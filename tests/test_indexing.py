import json

from qdrant_client import models

from ragops.indexing.qdrant import batch_records, build_point, create_collection, get_vector_size, load_chunk_records, upsert_records


def make_record(chunk_id="chunk-1", embedding=None):
    if embedding is None:
        embedding = [0.1, 0.2, 0.3]

    return {"chunk_id": chunk_id, "document_id": "doc-1", "text": f"Text for {chunk_id}", "token_count": 3, "chunk_hash": f"hash-{chunk_id}", "embedding": embedding, "metadata": {"relative_path": "docs/example.md", "source_name": "test"}}


def test_load_chunk_records_reads_jsonl_in_order(tmp_path):
    records = [
        make_record("chunk-1", [0.1, 0.2]),
        make_record("chunk-2", [0.3, 0.4]),
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    loaded_records = load_chunk_records(chunks_path)

    assert [record["chunk_id"] for record in loaded_records] == ["chunk-1", "chunk-2"]
    assert loaded_records[0]["text"] == "Text for chunk-1"
    assert loaded_records[1]["embedding"] == [0.3, 0.4]


def test_get_vector_size_uses_embedding_length():
    records = [
        make_record("chunk-1", [0.1, 0.2, 0.3, 0.4]),
        make_record("chunk-2", [0.5, 0.6]),
    ]

    vector_size = get_vector_size(records)

    assert vector_size == 4


def test_batch_records_yields_final_partial_batch():
    records = [make_record(f"chunk-{index}") for index in range(5)]

    batches = list(batch_records(records, batch_size=2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert batches[0][0]["chunk_id"] == "chunk-0"
    assert batches[2][0]["chunk_id"] == "chunk-4"


def test_build_point_keeps_vector_and_payload():
    record = make_record("chunk-1", [0.1, 0.2, 0.3])

    point = build_point(record)

    assert point.id == "chunk-1"
    assert point.vector == [0.1, 0.2, 0.3]
    assert point.payload["chunk_id"] == "chunk-1"
    assert point.payload["document_id"] == "doc-1"
    assert point.payload["text"] == "Text for chunk-1"
    assert point.payload["chunk_hash"] == "hash-chunk-1"
    assert point.payload["metadata"] == {"relative_path": "docs/example.md", "source_name": "test"}


def test_create_collection_uses_cosine_distance():
    class FakeClient:
        def __init__(self):
            self.exists_checks = []
            self.created_collections = []

        def collection_exists(self, collection_name):
            self.exists_checks.append(collection_name)
            return False

        def create_collection(self, collection_name, vectors_config):
            self.created_collections.append((collection_name, vectors_config))

    client = FakeClient()

    was_created = create_collection(client, "test_chunks", 384)

    assert was_created is True
    assert client.exists_checks == ["test_chunks"]
    assert len(client.created_collections) == 1
    collection_name, vectors_config = client.created_collections[0]
    assert collection_name == "test_chunks"
    assert isinstance(vectors_config, models.VectorParams)
    assert vectors_config.size == 384
    assert vectors_config.distance == models.Distance.COSINE


def test_upsert_records_sends_points_in_batches():
    class FakeClient:
        def __init__(self):
            self.upsert_calls = []

        def upsert(self, collection_name, points):
            self.upsert_calls.append((collection_name, points))

    client = FakeClient()
    records = [make_record(f"chunk-{index}") for index in range(5)]

    indexed_count = upsert_records(client, "test_chunks", records, batch_size=2)

    assert indexed_count == 5
    assert [collection_name for collection_name, points in client.upsert_calls] == ["test_chunks", "test_chunks", "test_chunks"]
    assert [len(points) for collection_name, points in client.upsert_calls] == [2, 2, 1]
    assert client.upsert_calls[0][1][0].id == "chunk-0"
    assert client.upsert_calls[2][1][0].payload["text"] == "Text for chunk-4"
