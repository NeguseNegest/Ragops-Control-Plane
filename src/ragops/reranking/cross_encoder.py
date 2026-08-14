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
from ragops.pipeline_registry import PipelineStatus, PipelineVersion
from ragops.retrieval.base import COMMON_RETRIEVER_INTERFACE, FunctionRetriever, Retriever, resolve_top_k, validate_timings
from ragops.retrieval.bm25 import BM25Config, BM25InputConfig, BM25RetrieverConfig
from ragops.retrieval.dense import DEFAULT_EMBEDDING_MODEL, RetrievedChunk, retrieve_dense, validate_query
from ragops.retrieval.hybrid import (
    DEFAULT_RRF_CONSTANT,
    HybridDenseConfig,
    HybridRetriever,
    ReciprocalRankFusionConfig,
    StrictModel,
    build_hybrid_retriever,
    configured_qdrant_url,
    resolve_project_path,
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


class RerankerEvaluationDatasetConfig(StrictModel):
    """Verified labels and fixed reports used for the Day 27 comparison."""

    labels_path: Path = Path("data/eval/retrieval_labels.jsonl")
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5], min_length=1)
    minimum_labels: int = Field(default=40, gt=0)
    dense_baseline_path: Path = Path("reports/evaluations/dense_baseline.json")
    bm25_baseline_path: Path = Path("reports/evaluations/bm25_baseline.json")
    hybrid_baseline_path: Path = Path("reports/evaluations/hybrid_rrf.json")

    @field_validator("labels_path", "dense_baseline_path", "bm25_baseline_path", "hybrid_baseline_path", mode="before")
    @classmethod
    def validate_evaluation_path(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("Reranker evaluation paths must not be empty.")
        return value

    @field_validator("k_values")
    @classmethod
    def validate_metric_cutoffs(cls, values):
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("k_values must contain positive integers.")
        if len(values) != len(set(values)):
            raise ValueError("k_values must not contain duplicates.")
        return sorted(values)


class RerankerEvaluationOutputConfig(StrictModel):
    """Day 27 run and four-way comparison artifact destinations."""

    directory: Path = Path("reports/evaluations")
    comparison_path: Path = Path("reports/evaluations/reranker_vs_baselines.json")
    report_path: Path = Path("reports/week4_reranker_comparison.md")

    @field_validator("directory", "comparison_path", "report_path", mode="before")
    @classmethod
    def validate_output_path(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("Reranker output paths must not be empty.")
        return value


class HybridRerankConfig(StrictModel):
    """Dense plus BM25 fusion followed by cross-encoder reranking."""

    name: str = Field(min_length=1)
    version: PipelineVersion = "0.1.0"
    status: PipelineStatus = "draft"
    retriever_interface: Literal["common_v1"] = COMMON_RETRIEVER_INTERFACE
    input: BM25InputConfig
    dense: HybridDenseConfig
    bm25: BM25RetrieverConfig
    fusion: ReciprocalRankFusionConfig
    reranker: CrossEncoderConfig
    evaluation: RerankerEvaluationDatasetConfig | None = None
    output: RerankerEvaluationOutputConfig | None = None

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
        if (self.evaluation is None) != (self.output is None):
            raise ValueError("Reranker evaluation and output settings must be configured together.")
        if self.evaluation is not None and self.reranker.top_k < max(self.evaluation.k_values):
            raise ValueError("reranker.top_k must be at least the largest evaluation cutoff.")
        return self

    def bm25_validation_config(self):
        """Build the index-only BM25 config expected by provenance validation."""
        return BM25Config(
            name=f"{self.name}_bm25",
            version=self.version,
            status=self.status,
            input=self.input,
            retriever=self.bm25,
        )


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
    updates = {"input": input_config, "bm25": bm25_config}
    if config.evaluation is not None:
        updates["evaluation"] = config.evaluation.model_copy(
            update={
                "labels_path": resolve_project_path(config.evaluation.labels_path, project_root),
                "dense_baseline_path": resolve_project_path(config.evaluation.dense_baseline_path, project_root),
                "bm25_baseline_path": resolve_project_path(config.evaluation.bm25_baseline_path, project_root),
                "hybrid_baseline_path": resolve_project_path(config.evaluation.hybrid_baseline_path, project_root),
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


class CrossEncoderRerankedRetriever(Retriever):
    """Cross-encoder stage composed over a common-interface candidate retriever."""

    def __init__(self, candidate_retriever, reranker, candidate_top_k=DEFAULT_RERANK_CANDIDATES, default_top_k=DEFAULT_RERANK_TOP_K, clock=time.perf_counter):
        super().__init__(default_top_k)
        if not callable(getattr(candidate_retriever, "retrieve", None)):
            raise ValueError("candidate_retriever must implement retrieve(query, top_k, timings).")
        self.candidate_retriever = candidate_retriever
        self.reranker = reranker
        self.candidate_top_k = _validate_positive_integer(candidate_top_k, "candidate_top_k")
        self.clock = clock
        if self.default_top_k > self.candidate_top_k:
            raise ValueError("default_top_k must not exceed candidate_top_k.")

    def retrieve_with_candidates(self, query, top_k=None, timings=None):
        """Return both the candidate order and final reranked order."""
        query = validate_query(query)
        top_k = resolve_top_k(top_k, self.default_top_k)
        validate_timings(timings)
        if top_k > self.candidate_top_k:
            raise ValueError("top_k must not exceed candidate_top_k.")

        stage_timings = {}
        started_at = self.clock()
        candidates = self.candidate_retriever.retrieve(query, top_k=self.candidate_top_k, timings=stage_timings)
        results = rerank_chunks(
            query,
            candidates,
            self.reranker,
            candidate_top_k=self.candidate_top_k,
            top_k=top_k,
            clock=self.clock,
            timings=stage_timings,
        )
        stage_timings["total_ms"] = max(0.0, (self.clock() - started_at) * 1000)
        if timings is not None:
            timings.update(stage_timings)
        return candidates, results

    def retrieve(self, query, top_k=None, timings=None):
        return self.retrieve_with_candidates(query, top_k=top_k, timings=timings)[1]


def build_hybrid_reranked_retriever(config, client, index, reranker, dense_retriever=retrieve_dense, bm25_retriever=None, clock=time.perf_counter):
    """Build the complete common-interface reranked pipeline from config."""
    if bm25_retriever is None:
        from ragops.retrieval.bm25 import retrieve_bm25

        bm25_retriever = retrieve_bm25
    candidate_retriever = build_hybrid_retriever(
        config,
        client,
        index,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        clock=clock,
    )
    return CrossEncoderRerankedRetriever(
        candidate_retriever,
        reranker,
        candidate_top_k=config.reranker.candidate_top_k,
        default_top_k=config.reranker.top_k,
        clock=clock,
    )


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
    if bm25_retriever is None:
        from ragops.retrieval.bm25 import retrieve_bm25

        bm25_retriever = retrieve_bm25
    dense = FunctionRetriever(
        dense_retriever,
        default_top_k=dense_top_k,
        client=client,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    sparse = FunctionRetriever(bm25_retriever, bm25_top_k, index=index)
    candidate_retriever = HybridRetriever(
        dense,
        sparse,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
        default_top_k=candidate_top_k,
        rank_constant=rank_constant,
        clock=clock,
    )
    retriever = CrossEncoderRerankedRetriever(
        candidate_retriever,
        reranker,
        candidate_top_k=candidate_top_k,
        default_top_k=top_k,
        clock=clock,
    )
    return retriever.retrieve(query, timings=timings)


def retrieve_hybrid_reranked_config(query, config, client, index, reranker, dense_retriever=retrieve_dense, bm25_retriever=None, clock=time.perf_counter, timings=None):
    """Run the complete Day 26 pipeline from validated configuration."""
    retriever = build_hybrid_reranked_retriever(
        config,
        client,
        index,
        reranker,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        clock=clock,
    )
    return retriever.retrieve(query, timings=timings)


__all__ = [
    "CrossEncoderConfig",
    "CrossEncoderRerankedRetriever",
    "CrossEncoderReranker",
    "HybridRerankConfig",
    "RERANKER_METADATA_KEY",
    "RerankerEvaluationDatasetConfig",
    "RerankerEvaluationOutputConfig",
    "build_cross_encoder_reranker",
    "build_hybrid_reranked_retriever",
    "configured_qdrant_url",
    "load_hybrid_rerank_config",
    "rerank_chunks",
    "retrieve_hybrid_reranked",
    "retrieve_hybrid_reranked_config",
]
