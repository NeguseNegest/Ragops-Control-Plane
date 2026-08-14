import gzip
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rank_bm25 import BM25Okapi

from ragops.pipeline_registry import PipelineStatus, PipelineVersion
from ragops.retrieval.base import COMMON_RETRIEVER_INTERFACE, Retriever, resolve_top_k, validate_timings
from ragops.retrieval.dense import DEFAULT_TOP_K, RetrievedChunk, source_url_from_metadata, validate_query

BM25_INDEX_SCHEMA_VERSION = 1
BM25_TOKENIZER = "technical_v1"
DEFAULT_BM25_INDEX_PATH = Path("data/processed/bm25_index.json.gz")
DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")

_BASE_TOKEN_RE = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
_TECHNICAL_PUNCTUATION_RE = re.compile(r"[._:/=<>{}\[\]-]")
_TECHNICAL_STRIP_CHARS = "`'\"(),;!?*#|"


class StrictModel(BaseModel):
    """Base model for persisted BM25 data and configuration."""

    model_config = ConfigDict(extra="forbid")


class BM25Parameters(StrictModel):
    """Tunable parameters passed to rank-bm25's BM25Okapi scorer."""

    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    epsilon: float = Field(default=0.25, ge=0)


class BM25InputConfig(StrictModel):
    """Chunk input used to build the persisted sparse index."""

    chunks_path: Path = DEFAULT_CHUNKS_PATH

    @field_validator("chunks_path", mode="before")
    @classmethod
    def validate_chunks_path(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("chunks_path must not be empty.")
        return value


class BM25RetrieverConfig(StrictModel):
    """Sparse retrieval and index settings."""

    type: Literal["bm25"] = "bm25"
    index_path: Path = DEFAULT_BM25_INDEX_PATH
    tokenizer: Literal["technical_v1"] = BM25_TOKENIZER
    top_k: int = Field(default=10, ge=1, le=100)
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    epsilon: float = Field(default=0.25, ge=0)

    @field_validator("index_path", mode="before")
    @classmethod
    def validate_index_path(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("index_path must not be empty.")
        return value

    def parameters(self):
        return BM25Parameters(k1=self.k1, b=self.b, epsilon=self.epsilon)


class BM25EvaluationDatasetConfig(StrictModel):
    """Verified labels, metric cutoffs, and dense run used for Day 23."""

    labels_path: Path = Path("data/eval/retrieval_labels.jsonl")
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10], min_length=1)
    minimum_labels: int = Field(default=40, gt=0)
    dense_baseline_path: Path = Path("reports/evaluations/dense_baseline.json")

    @field_validator("labels_path", "dense_baseline_path", mode="before")
    @classmethod
    def validate_evaluation_path(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("Evaluation paths must not be empty.")
        return value

    @field_validator("k_values")
    @classmethod
    def validate_metric_cutoffs(cls, values):
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("k_values must contain positive integers.")
        if len(values) != len(set(values)):
            raise ValueError("k_values must not contain duplicates.")
        return sorted(values)


class BM25EvaluationOutputConfig(StrictModel):
    """Machine-readable and narrative Day 23 artifact destinations."""

    directory: Path = Path("reports/evaluations")
    comparison_path: Path = Path("reports/evaluations/bm25_vs_dense.json")
    report_path: Path = Path("reports/week4_bm25_comparison.md")

    @field_validator("directory", "comparison_path", "report_path", mode="before")
    @classmethod
    def validate_output_path(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("Output paths must not be empty.")
        return value


class BM25Config(StrictModel):
    """Complete BM25 index configuration with optional Day 23 evaluation."""

    name: str = Field(min_length=1)
    version: PipelineVersion = "0.1.0"
    status: PipelineStatus = "draft"
    retriever_interface: Literal["common_v1"] = COMMON_RETRIEVER_INTERFACE
    input: BM25InputConfig
    retriever: BM25RetrieverConfig
    evaluation: BM25EvaluationDatasetConfig | None = None
    output: BM25EvaluationOutputConfig | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        value = value.strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
            raise ValueError("BM25 configuration name must use lowercase letters, numbers, underscores, or hyphens.")
        return value

    @model_validator(mode="after")
    def validate_evaluation_settings(self):
        if (self.evaluation is None) != (self.output is None):
            raise ValueError("BM25 evaluation and output settings must be configured together.")
        if self.evaluation is not None and self.retriever.top_k < max(self.evaluation.k_values):
            raise ValueError("retriever.top_k must be at least the largest evaluation cutoff.")
        return self


class BM25IndexDocument(StrictModel):
    """One searchable document stored in the portable BM25 index."""

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    token_count: int = Field(ge=1)
    chunk_hash: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tokens: list[str] = Field(min_length=1)

    @field_validator("chunk_id", "document_id", "text", "chunk_hash")
    @classmethod
    def clean_required_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("BM25 document text fields must not be empty.")
        return value

    @field_validator("tokens")
    @classmethod
    def validate_tokens(cls, tokens):
        if any(not isinstance(token, str) or not token.strip() for token in tokens):
            raise ValueError("BM25 document tokens must be non-empty strings.")
        return tokens


class BM25IndexPayload(StrictModel):
    """Versioned, JSON-serializable representation of a BM25 index."""

    schema_version: Literal[1] = BM25_INDEX_SCHEMA_VERSION
    tokenizer: Literal["technical_v1"] = BM25_TOKENIZER
    parameters: BM25Parameters
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_record_count: int = Field(gt=0)
    skipped_document_count: int = Field(ge=0)
    document_count: int = Field(gt=0)
    documents: list[BM25IndexDocument] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_documents(self):
        if self.document_count != len(self.documents):
            raise ValueError("document_count must match the number of indexed documents.")
        if self.source_record_count != self.document_count + self.skipped_document_count:
            raise ValueError("source_record_count must equal indexed plus skipped documents.")
        chunk_ids = [document.chunk_id for document in self.documents]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("BM25 index documents must have unique chunk IDs.")
        return self


def tokenize_bm25(text):
    """Tokenize prose while retaining exact technical strings as extra terms."""
    if not isinstance(text, str):
        raise ValueError("BM25 text must be a string.")

    normalized = text.casefold()
    tokens = _BASE_TOKEN_RE.findall(normalized)
    base_tokens = set(tokens)

    for raw_token in normalized.split():
        technical_token = raw_token.strip(f"{_TECHNICAL_STRIP_CHARS}.")
        if not technical_token or not _TECHNICAL_PUNCTUATION_RE.search(technical_token):
            continue
        if not any(character.isalnum() for character in technical_token):
            continue
        if technical_token not in base_tokens:
            tokens.append(technical_token)

    return tokens


def resolve_project_path(path, project_root):
    """Resolve one configured path relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else (Path(project_root) / path).resolve()


def load_bm25_config(config_path, project_root=None):
    """Load strict BM25 YAML configuration and resolve its paths."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"BM25 config does not exist: {config_path}")

    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error

    if not isinstance(raw_config, dict) or not raw_config:
        raise ValueError(f"BM25 config must contain a YAML mapping: {config_path}")

    config = BM25Config.model_validate(raw_config)
    project_root = Path(project_root or Path.cwd()).resolve()
    input_config = config.input.model_copy(update={"chunks_path": resolve_project_path(config.input.chunks_path, project_root)})
    retriever = config.retriever.model_copy(update={"index_path": resolve_project_path(config.retriever.index_path, project_root)})
    updates = {"input": input_config, "retriever": retriever}
    if config.evaluation is not None:
        updates["evaluation"] = config.evaluation.model_copy(
            update={
                "labels_path": resolve_project_path(config.evaluation.labels_path, project_root),
                "dense_baseline_path": resolve_project_path(config.evaluation.dense_baseline_path, project_root),
            }
        )
    if config.output is not None:
        updates["output"] = config.output.model_copy(
            update={
                "directory": resolve_project_path(config.output.directory, project_root),
                "comparison_path": resolve_project_path(config.output.comparison_path, project_root),
                "report_path": resolve_project_path(config.output.report_path, project_root),
            }
        )
    return config.model_copy(update=updates)


def sha256_file(path, block_size=1024 * 1024):
    """Return the SHA256 digest of one source artifact without loading it whole."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        while block := input_file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def iter_chunk_records(path):
    """Yield JSON objects from the embedded chunk artifact with useful errors."""
    path = Path(path)
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} on line {line_number}: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path} on line {line_number}.")
            yield record


def build_index_document(record):
    """Convert one embedded chunk record into a validated sparse document."""
    if not isinstance(record, dict):
        raise ValueError("BM25 source records must be JSON objects.")
    for field_name in ("chunk_id", "document_id", "text", "chunk_hash"):
        value = record.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"BM25 source field {field_name} must be a non-empty string.")
    token_count = record.get("token_count")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
        raise ValueError("BM25 source field token_count must be a positive integer.")
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("BM25 source field metadata must be an object.")

    text = record.get("text")
    tokens = tokenize_bm25(text)
    if not tokens:
        return None

    return BM25IndexDocument(
        chunk_id=record.get("chunk_id"),
        document_id=record.get("document_id"),
        text=text,
        token_count=token_count,
        chunk_hash=record.get("chunk_hash"),
        metadata=metadata,
        tokens=tokens,
    )


def build_bm25_index(records, source_path, source_sha256, parameters=None):
    """Build a portable BM25 payload from chunk records, excluding embeddings."""
    parameters = parameters or BM25Parameters()
    documents = []
    source_record_count = 0
    skipped_document_count = 0

    for index, record in enumerate(records, start=1):
        source_record_count += 1
        try:
            document = build_index_document(record)
        except Exception as error:
            raise ValueError(f"Invalid BM25 source record {index}: {error}") from error
        if document is None:
            skipped_document_count += 1
            continue
        documents.append(document)

    if not documents:
        raise ValueError("At least one chunk record is required to build a BM25 index.")

    return BM25IndexPayload(
        parameters=parameters,
        source_path=str(source_path),
        source_sha256=source_sha256,
        source_record_count=source_record_count,
        skipped_document_count=skipped_document_count,
        document_count=len(documents),
        documents=documents,
    )


def build_bm25_index_from_jsonl(input_path, parameters=None):
    """Build a BM25 payload from the processed chunk JSONL artifact."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Chunk input does not exist: {input_path}")
    return build_bm25_index(iter_chunk_records(input_path), source_path=input_path, source_sha256=sha256_file(input_path), parameters=parameters)


def save_bm25_index(index, output_path, overwrite=False):
    """Atomically save a portable gzip-compressed BM25 index."""
    payload = index.payload if isinstance(index, BM25Index) else index
    if not isinstance(payload, BM25IndexPayload):
        payload = BM25IndexPayload.model_validate(payload)

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing BM25 index: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8") as output_file:
            json.dump(payload.model_dump(mode="json"), output_file, ensure_ascii=False, separators=(",", ":"))
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return output_path


class BM25Index:
    """In-memory BM25 scorer backed by a validated portable index payload."""

    def __init__(self, payload):
        self.payload = payload if isinstance(payload, BM25IndexPayload) else BM25IndexPayload.model_validate(payload)
        tokenized_corpus = [document.tokens for document in self.payload.documents]
        parameters = self.payload.parameters
        self.scorer = BM25Okapi(tokenized_corpus, k1=parameters.k1, b=parameters.b, epsilon=parameters.epsilon)

    def search(self, query, top_k=DEFAULT_TOP_K):
        query = validate_query(query)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_tokens = tokenize_bm25(query)
        if not query_tokens:
            raise ValueError("query must contain at least one searchable BM25 token.")

        scores = self.scorer.get_scores(query_tokens)
        ranked_indices = sorted(range(len(self.payload.documents)), key=lambda index: (-float(scores[index]), index))[:top_k]
        results = []

        for rank, document_index in enumerate(ranked_indices, start=1):
            document = self.payload.documents[document_index]
            metadata = dict(document.metadata)
            results.append(
                RetrievedChunk(
                    chunk_id=document.chunk_id,
                    document_id=document.document_id,
                    text=document.text,
                    score=float(scores[document_index]),
                    rank=rank,
                    metadata=metadata,
                    source_url=source_url_from_metadata(metadata),
                )
            )

        return results


def load_bm25_index(path):
    """Load and validate a persisted BM25 index, then initialize its scorer."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"BM25 index does not exist: {path}")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, EOFError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"BM25 index is invalid: {path}") from error
    return BM25Index(payload)


def validate_bm25_index(index, config, verify_source_hash=True):
    """Verify persisted provenance and scoring settings against a BM25 config."""
    if not isinstance(index, BM25Index):
        raise ValueError("index must be a loaded BM25Index.")

    payload = index.payload
    if payload.tokenizer != config.retriever.tokenizer:
        raise ValueError("Persisted BM25 tokenizer does not match the configuration.")
    if payload.parameters != config.retriever.parameters():
        raise ValueError("Persisted BM25 parameters do not match the configuration.")
    if verify_source_hash:
        chunks_path = config.input.chunks_path
        if not chunks_path.is_file():
            raise FileNotFoundError(f"Chunk input does not exist: {chunks_path}")
        if payload.source_sha256 != sha256_file(chunks_path):
            raise ValueError("Persisted BM25 index does not match the current chunk input SHA256.")
    return payload


class BM25Retriever(Retriever):
    """Configured sparse retriever implementing the common retrieval interface."""

    def __init__(self, index, default_top_k=DEFAULT_TOP_K, clock=time.perf_counter):
        super().__init__(default_top_k)
        if isinstance(index, (str, Path)):
            index = load_bm25_index(index)
        if not isinstance(index, BM25Index):
            raise ValueError("index must be a BM25Index or a path to a persisted BM25 index.")
        self.index = index
        self.clock = clock

    def retrieve(self, query, top_k=None, timings=None):
        top_k = resolve_top_k(top_k, self.default_top_k)
        validate_timings(timings)
        started_at = self.clock() if timings is not None else None
        try:
            return self.index.search(query, top_k=top_k)
        finally:
            if started_at is not None:
                timings["bm25_ms"] = max(0.0, (self.clock() - started_at) * 1000)


def retrieve_bm25(query, index, top_k=DEFAULT_TOP_K):
    """Return ranked sparse chunks from a loaded index or persisted index path."""
    return BM25Retriever(index, default_top_k=top_k).retrieve(query)
