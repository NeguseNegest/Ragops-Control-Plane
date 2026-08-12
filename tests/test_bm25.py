import gzip
import json

import pytest
import yaml
from pydantic import ValidationError

from ragops.retrieval.bm25 import (
    BM25Config,
    BM25Index,
    BM25Parameters,
    build_bm25_index,
    build_bm25_index_from_jsonl,
    load_bm25_config,
    load_bm25_index,
    retrieve_bm25,
    save_bm25_index,
    sha256_file,
    tokenize_bm25,
)


def make_record(chunk_id, text, source="docs/example.md"):
    return {
        "chunk_id": chunk_id,
        "document_id": f"document-{chunk_id}",
        "text": text,
        "token_count": len(text.split()),
        "chunk_hash": f"hash-{chunk_id}",
        "metadata": {"relative_path": source, "heading": f"Heading {chunk_id}"},
        "embedding": [0.1, 0.2, 0.3],
    }


def make_payload(records=None):
    records = records or [
        make_record("chunk-fastapi", "FastAPI validates a JSON request body."),
        make_record("chunk-mlflow", "Run mlflow models serve -m runs:/<RUN_ID>/model to serve the model.", "mlflow/serve.md"),
        make_record("chunk-qdrant", "Qdrant uses dot product to calculate vector similarity.", "qdrant/search.md"),
    ]
    return build_bm25_index(records, source_path="chunks.jsonl", source_sha256="a" * 64)


def test_tokenizer_normalizes_prose_and_preserves_technical_terms():
    tokens = tokenize_bm25("MLflow models serve -m runs:/<RUN_ID>/model with strict_content_type=False.")

    assert tokens[:7] == ["mlflow", "models", "serve", "m", "runs", "run_id", "model"]
    assert "runs:/<run_id>/model" in tokens
    assert "strict_content_type=false" in tokens


def test_tokenizer_rejects_non_string_and_can_return_no_tokens():
    with pytest.raises(ValueError, match="must be a string"):
        tokenize_bm25(None)
    assert tokenize_bm25("... !!!") == []


def test_tokenizer_preserves_repeated_technical_term_frequency():
    tokens = tokenize_bm25("runs:/<RUN_ID>/model then runs:/<RUN_ID>/model")

    assert tokens.count("runs:/<run_id>/model") == 2


def test_retrieve_bm25_ranks_exact_technical_match_and_preserves_payload():
    index = BM25Index(make_payload())

    results = retrieve_bm25("exact command runs:/<RUN_ID>/model", index, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-mlflow"
    assert results[0].rank == 1
    assert results[0].source_url == "mlflow/serve.md"
    assert results[0].metadata["heading"] == "Heading chunk-mlflow"
    assert results[0].score > results[1].score


def test_retrieve_bm25_breaks_score_ties_in_source_order():
    index = BM25Index(make_payload())

    results = retrieve_bm25("zzzxxyyqqq", index, top_k=3)

    assert [result.chunk_id for result in results] == ["chunk-fastapi", "chunk-mlflow", "chunk-qdrant"]


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_retrieve_bm25_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        retrieve_bm25("valid query", BM25Index(make_payload()), top_k=top_k)


def test_retrieve_bm25_rejects_empty_query_and_invalid_index():
    index = BM25Index(make_payload())
    with pytest.raises(ValueError, match="empty"):
        retrieve_bm25("   ", index)
    with pytest.raises(ValueError, match="searchable"):
        retrieve_bm25("...", index)
    with pytest.raises(ValueError, match="BM25Index"):
        retrieve_bm25("query", object())


def test_save_and_load_bm25_index_round_trip_without_embeddings(tmp_path):
    payload = make_payload()
    output_path = tmp_path / "bm25_index.json.gz"

    save_bm25_index(payload, output_path)
    loaded = load_bm25_index(output_path)
    results = retrieve_bm25("dot product similarity", loaded, top_k=1)

    assert loaded.payload.document_count == 3
    assert loaded.payload.source_sha256 == "a" * 64
    assert results[0].chunk_id == "chunk-qdrant"
    with gzip.open(output_path, "rt", encoding="utf-8") as input_file:
        persisted = json.load(input_file)
    assert "embedding" not in json.dumps(persisted)
    assert retrieve_bm25("dot product similarity", output_path, top_k=1)[0].chunk_id == "chunk-qdrant"

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_bm25_index(payload, output_path)


def test_load_bm25_index_rejects_corrupt_and_inconsistent_payloads(tmp_path):
    corrupt_path = tmp_path / "corrupt.json.gz"
    corrupt_path.write_bytes(b"not a gzip index")
    with pytest.raises(ValueError, match="invalid"):
        load_bm25_index(corrupt_path)

    payload = make_payload().model_dump(mode="json")
    payload["document_count"] = 99
    inconsistent_path = tmp_path / "inconsistent.json.gz"
    with gzip.open(inconsistent_path, "wt", encoding="utf-8") as output_file:
        json.dump(payload, output_file)
    with pytest.raises(ValidationError, match="document_count"):
        load_bm25_index(inconsistent_path)


def test_build_index_from_jsonl_hashes_source_and_drops_embedding(tmp_path):
    records = [make_record("chunk-1", "alpha beta"), make_record("chunk-2", "gamma delta")]
    input_path = tmp_path / "chunks.jsonl"
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    payload = build_bm25_index_from_jsonl(input_path, parameters=BM25Parameters(k1=1.2, b=0.6, epsilon=0.1))

    assert payload.document_count == 2
    assert payload.source_sha256 == sha256_file(input_path)
    assert payload.parameters.k1 == 1.2
    assert payload.documents[0].tokens == ["alpha", "beta"]
    assert not hasattr(payload.documents[0], "embedding")


def test_build_index_rejects_empty_records_and_duplicate_chunk_ids():
    with pytest.raises(ValueError, match="At least one"):
        build_bm25_index([], source_path="chunks.jsonl", source_sha256="a" * 64)

    records = [make_record("duplicate", "alpha"), make_record("duplicate", "beta")]
    with pytest.raises(ValidationError, match="unique chunk IDs"):
        build_bm25_index(records, source_path="chunks.jsonl", source_sha256="a" * 64)


def test_build_index_records_and_skips_chunks_without_searchable_tokens():
    records = [make_record("searchable", "alpha beta"), make_record("markup-only", "... !!!")]

    payload = build_bm25_index(records, source_path="chunks.jsonl", source_sha256="a" * 64)

    assert payload.source_record_count == 2
    assert payload.document_count == 1
    assert payload.skipped_document_count == 1
    assert [document.chunk_id for document in payload.documents] == ["searchable"]


def test_build_index_rejects_malformed_tokenless_record():
    record = make_record("markup-only", "... !!!")
    record["chunk_id"] = ""

    with pytest.raises(ValueError, match="chunk_id"):
        build_bm25_index([record], source_path="chunks.jsonl", source_sha256="a" * 64)


def test_load_bm25_config_resolves_paths_and_parameters(tmp_path):
    config_path = tmp_path / "bm25.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "bm25_test",
                "input": {"chunks_path": "data/chunks.jsonl"},
                "retriever": {
                    "type": "bm25",
                    "index_path": "data/bm25.json.gz",
                    "tokenizer": "technical_v1",
                    "top_k": 7,
                    "k1": 1.2,
                    "b": 0.6,
                    "epsilon": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_bm25_config(config_path, project_root=tmp_path)

    assert config.input.chunks_path == tmp_path / "data/chunks.jsonl"
    assert config.retriever.index_path == tmp_path / "data/bm25.json.gz"
    assert config.retriever.top_k == 7
    assert config.retriever.parameters() == BM25Parameters(k1=1.2, b=0.6, epsilon=0.1)


@pytest.mark.parametrize(
    "change",
    [
        {"name": "Invalid Name"},
        {"retriever": {"tokenizer": "unknown"}},
        {"retriever": {"b": 1.5}},
        {"retriever": {"top_k": 0}},
        {"retriever": {"index_path": "   "}},
        {"input": {"chunks_path": "   "}},
    ],
)
def test_bm25_config_rejects_invalid_settings(change):
    data = {
        "name": "bm25_test",
        "input": {"chunks_path": "chunks.jsonl"},
        "retriever": {"type": "bm25", "index_path": "index.json.gz"},
    }
    for section, values in change.items():
        if isinstance(values, dict):
            data[section].update(values)
        else:
            data[section] = values

    with pytest.raises(ValidationError):
        BM25Config.model_validate(data)


def test_checked_in_bm25_config_is_valid():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    config = load_bm25_config(project_root / "configs/bm25_baseline.yaml", project_root=project_root)

    assert config.name == "bm25_baseline"
    assert config.retriever.type == "bm25"
    assert config.retriever.tokenizer == "technical_v1"
    assert config.retriever.top_k == 10
