import math
import re
import time
from collections.abc import Sequence
from numbers import Real
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from ragops.indexing.qdrant import DEFAULT_COLLECTION_NAME
from ragops.retrieval.bm25 import BM25Config, BM25InputConfig, BM25RetrieverConfig
from ragops.retrieval.dense import DEFAULT_EMBEDDING_MODEL, RetrievedChunk, retrieve_dense, validate_query
from ragops.retrieval.hybrid import (
    DEFAULT_RRF_CONSTANT,
    HybridDenseConfig,
    ReciprocalRankFusionConfig,
    StrictModel,
    configured_qdrant_url,
    resolve_project_path,
    retrieve_hybrid,
)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_CANDIDATES = 25
DEFAULT_RERANK_TOP_K = 5
DEFAULT_RERANK_BATCH_SIZE = 16
DEFAULT_RERANK_MAX_LENGTH = 512
RERANKER_METADATA_KEY = "_reranker"


class CrossEncoderConfig(StrictModel):
    """Cross-encoder model and candidate-depth settings."""

    type: Literal["cross_encoder"] = "cross_encoder"
    model: str = DEFAULT_RERANKER_MODEL
    candidate_top_k: int = Field(default=DEFAULT_RERANK_CANDIDATES, ge=1, le=100)
    top_k: int = Field(default=DEFAULT_RERANK_TOP_K, ge=1, le=100)
    batch_size: int = Field(default=DEFAULT_RERANK_BATCH_SIZE, ge=1, le=256)
    max_length: int = Field(default=DEFAULT_RERANK_MAX_LENGTH, ge=8, le=8192)
    device: str | None = None

    @field_validator("model")
    @classmethod
    def clean_model_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Reranker model must not be empty.")
        return value

    @field_validator("device")
    @classmethod
    def clean_optional_device(cls, value):
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_depths(self):
        if self.top_k > self.candidate_top_k:
            raise ValueError("reranker.top_k must not exceed reranker.candidate_top_k.")
        return self


class HybridRerankConfig(StrictModel):
    """Dense plus BM25 fusion followed by cross-encoder reranking."""

    name: str = Field(min_length=1)
    input: BM25InputConfig
    dense: HybridDenseConfig
    bm25: BM25RetrieverConfig
    fusion: ReciprocalRankFusionConfig
    reranker: CrossEncoderConfig

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        value = value.strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
            raise ValueError("Hybrid reranker configuration name must use lowercase letters, numbers, underscores, or hyphens.")
        return value

    @model_validator(mode="after")
    def validate_retrieval_depths(self):
        if self.dense.top_k < self.fusion.top_k:
            raise ValueError("dense.top_k must be at least fusion.top_k.")
        if self.bm25.top_k < self.fusion.top_k:
            raise ValueError("bm25.top_k must be at least fusion.top_k.")
        if self.fusion.top_k != self.reranker.candidate_top_k:
            raise ValueError("fusion.top_k must equal reranker.candidate_top_k so every fused candidate is reranked.")
        return self

    def bm25_validation_config(self):
        """Build the index-only BM25 config expected by provenance validation."""
        return BM25Config(name=f"{self.name}_bm25", input=self.input, retriever=self.bm25)


def load_hybrid_rerank_config(config_path, project_root=None):
    """Load strict hybrid-reranker YAML and resolve local artifact paths."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Hybrid reranker config does not exist: {config_path}")

    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
    if not isinstance(raw_config, dict) or not raw_config:
        raise ValueError(f"Hybrid reranker config must contain a YAML mapping: {config_path}")

    config = HybridRerankConfig.model_validate(raw_config)
    project_root = Path(project_root or Path.cwd()).resolve()
    input_config = config.input.model_copy(update={"chunks_path": resolve_project_path(config.input.chunks_path, project_root)})
    bm25_config = config.bm25.model_copy(update={"index_path": resolve_project_path(config.bm25.index_path, project_root)})
    return config.model_copy(update={"input": input_config, "bm25": bm25_config})


class CrossEncoderReranker:
    """Thin sentence-transformers CrossEncoder wrapper with validated scores."""

    def __init__(self, model_name=DEFAULT_RERANKER_MODEL, batch_size=DEFAULT_RERANK_BATCH_SIZE, max_length=DEFAULT_RERANK_MAX_LENGTH, device=None, model_factory=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        if model_factory is None:
            from sentence_transformers import CrossEncoder

            model_factory = CrossEncoder
        model_kwargs = {"max_length": max_length}
        if device is not None:
            model_kwargs["device"] = device
        self.model = model_factory(model_name, **model_kwargs)

    def score(self, query, chunks):
        """Score every query/chunk pair without changing their input order."""
        query = validate_query(query)
        chunks = list(chunks)
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        raw_scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        if isinstance(raw_scores, Real):
            raw_scores = [raw_scores]
        elif not isinstance(raw_scores, list):
            raw_scores = list(raw_scores)
        if len(raw_scores) != len(chunks):
            raise ValueError(f"Cross-encoder returned {len(raw_scores)} scores for {len(chunks)} candidates.")

        scores = []
        for position, score in enumerate(raw_scores, start=1):
            if not isinstance(score, Real) or not math.isfinite(float(score)):
                raise ValueError(f"Cross-encoder score at position {position} must be a finite scalar.")
            scores.append(float(score))
        return scores


def build_cross_encoder_reranker(config, model_factory=None):
    """Construct a reranker from validated configuration."""
    return CrossEncoderReranker(
        model_name=config.model,
        batch_size=config.batch_size,
        max_length=config.max_length,
        device=config.device,
        model_factory=model_factory,
    )


def _validate_positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _coerce_candidates(chunks, candidate_top_k):
    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
        raise ValueError("chunks must be a ranked sequence.")
    candidates = []
    seen_chunk_ids = set()
    for position, raw_chunk in enumerate(chunks[:candidate_top_k], start=1):
        try:
            chunk = raw_chunk if isinstance(raw_chunk, RetrievedChunk) else RetrievedChunk.model_validate(raw_chunk)
        except Exception as error:
            raise ValueError(f"Reranker candidate at position {position} is invalid: {error}") from error
        if not chunk.chunk_id.strip():
            raise ValueError(f"Reranker candidate at position {position} has an empty chunk ID.")
        if chunk.chunk_id in seen_chunk_ids:
            raise ValueError(f"Reranker candidates contain duplicate chunk ID {chunk.chunk_id}.")
        if chunk.rank != position:
            raise ValueError(f"Reranker candidate {chunk.chunk_id} has rank {chunk.rank}; expected {position}.")
        if not chunk.text.strip():
            raise ValueError(f"Reranker candidate {chunk.chunk_id} has empty text.")
        if not math.isfinite(float(chunk.score)):
            raise ValueError(f"Reranker candidate {chunk.chunk_id} has a non-finite score.")
        seen_chunk_ids.add(chunk.chunk_id)
        candidates.append(chunk)
    return candidates


def rerank_chunks(query, chunks, reranker, candidate_top_k=DEFAULT_RERANK_CANDIDATES, top_k=DEFAULT_RERANK_TOP_K, clock=time.perf_counter, timings=None):
    """Rerank the leading candidates by cross-encoder relevance score."""
    query = validate_query(query)
    candidate_top_k = _validate_positive_integer(candidate_top_k, "candidate_top_k")
    top_k = _validate_positive_integer(top_k, "top_k")
    if top_k > candidate_top_k:
        raise ValueError("top_k must not exceed candidate_top_k.")
    if timings is not None and not isinstance(timings, dict):
        raise ValueError("timings must be a dictionary when provided.")
    candidates = _coerce_candidates(chunks, candidate_top_k)
    if not candidates:
        if timings is not None:
            timings["reranker_ms"] = 0.0
        return []

    started_at = clock()
    try:
        scores = reranker.score(query, candidates)
    except Exception as error:
        raise RuntimeError(f"Cross-encoder reranking failed: {error}") from error
    reranker_latency_ms = max(0.0, (clock() - started_at) * 1000)
    if len(scores) != len(candidates):
        raise ValueError(f"Reranker returned {len(scores)} scores for {len(candidates)} candidates.")

    scored = []
    model_name = getattr(reranker, "model_name", "unknown")
    for candidate, score in zip(candidates, scores, strict=True):
        if not isinstance(score, Real) or not math.isfinite(float(score)):
            raise ValueError(f"Reranker score for chunk {candidate.chunk_id} must be a finite scalar.")
        scored.append((candidate, float(score)))
    scored.sort(key=lambda item: (-item[1], item[0].rank, item[0].chunk_id))

    results = []
    for final_rank, (candidate, score) in enumerate(scored[:top_k], start=1):
        metadata = dict(candidate.metadata)
        metadata[RERANKER_METADATA_KEY] = {
            "method": "cross_encoder",
            "model": model_name,
            "candidate_rank": candidate.rank,
            "candidate_score": float(candidate.score),
        }
        results.append(
            RetrievedChunk(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                text=candidate.text,
                score=score,
                rank=final_rank,
                metadata=metadata,
                source_url=candidate.source_url,
            )
        )
    if timings is not None:
        timings["reranker_ms"] = reranker_latency_ms
    return results


def retrieve_hybrid_reranked(
    query,
    client,
    index,
    reranker,
    dense_top_k=DEFAULT_RERANK_CANDIDATES,
    bm25_top_k=DEFAULT_RERANK_CANDIDATES,
    candidate_top_k=DEFAULT_RERANK_CANDIDATES,
    top_k=DEFAULT_RERANK_TOP_K,
    rank_constant=DEFAULT_RRF_CONSTANT,
    collection_name=DEFAULT_COLLECTION_NAME,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    dense_retriever=retrieve_dense,
    bm25_retriever=None,
    clock=time.perf_counter,
    timings=None,
):
    """Retrieve and fuse a candidate pool, then cross-encode its top results."""
    if timings is not None and not isinstance(timings, dict):
        raise ValueError("timings must be a dictionary when provided.")
    candidate_top_k = _validate_positive_integer(candidate_top_k, "candidate_top_k")
    top_k = _validate_positive_integer(top_k, "top_k")
    if top_k > candidate_top_k:
        raise ValueError("top_k must not exceed candidate_top_k.")
    if bm25_retriever is None:
        from ragops.retrieval.bm25 import retrieve_bm25

        bm25_retriever = retrieve_bm25
    stage_timings = {}
    started_at = clock()
    candidates = retrieve_hybrid(
        query=query,
        client=client,
        index=index,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
        top_k=candidate_top_k,
        rank_constant=rank_constant,
        collection_name=collection_name,
        embedding_model=embedding_model,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        clock=clock,
        timings=stage_timings,
    )
    results = rerank_chunks(
        query=query,
        chunks=candidates,
        reranker=reranker,
        candidate_top_k=candidate_top_k,
        top_k=top_k,
        clock=clock,
        timings=stage_timings,
    )
    stage_timings["total_ms"] = max(0.0, (clock() - started_at) * 1000)
    if timings is not None:
        timings.update(stage_timings)
    return results


def retrieve_hybrid_reranked_config(query, config, client, index, reranker, dense_retriever=retrieve_dense, bm25_retriever=None, clock=time.perf_counter, timings=None):
    """Run the complete Day 26 pipeline from validated configuration."""
    return retrieve_hybrid_reranked(
        query=query,
        client=client,
        index=index,
        reranker=reranker,
        dense_top_k=config.dense.top_k,
        bm25_top_k=config.bm25.top_k,
        candidate_top_k=config.reranker.candidate_top_k,
        top_k=config.reranker.top_k,
        rank_constant=config.fusion.rank_constant,
        collection_name=config.dense.collection_name,
        embedding_model=config.dense.embedding_model,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        clock=clock,
        timings=timings,
    )


__all__ = [
    "CrossEncoderConfig",
    "CrossEncoderReranker",
    "HybridRerankConfig",
    "RERANKER_METADATA_KEY",
    "build_cross_encoder_reranker",
    "configured_qdrant_url",
    "load_hybrid_rerank_config",
    "rerank_chunks",
    "retrieve_hybrid_reranked",
    "retrieve_hybrid_reranked_config",
]
