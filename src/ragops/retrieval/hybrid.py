import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragops.indexing.qdrant import DEFAULT_COLLECTION_NAME, DEFAULT_QDRANT_URL
from ragops.pipeline_registry import PipelineStatus, PipelineVersion
from ragops.retrieval.base import COMMON_RETRIEVER_INTERFACE, FunctionRetriever, Retriever, resolve_top_k, validate_timings, validate_top_k
from ragops.retrieval.bm25 import BM25Config, BM25InputConfig, BM25Retriever, BM25RetrieverConfig, retrieve_bm25
from ragops.retrieval.dense import DEFAULT_EMBEDDING_MODEL, DenseRetriever, RetrievedChunk, retrieve_dense, validate_query

DEFAULT_DENSE_CANDIDATES = 20
DEFAULT_BM25_CANDIDATES = 20
DEFAULT_HYBRID_TOP_K = 10
DEFAULT_RRF_CONSTANT = 60.0
FUSION_METADATA_KEY = "_fusion"


class StrictModel(BaseModel):
    """Base model for strict hybrid retrieval configuration."""

    model_config = ConfigDict(extra="forbid")


class HybridDenseConfig(StrictModel):
    """Dense candidate retrieval settings for the hybrid path."""

    type: Literal["dense"] = "dense"
    collection_name: str = DEFAULT_COLLECTION_NAME
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = Field(default=DEFAULT_DENSE_CANDIDATES, ge=1, le=100)
    qdrant_url: str | None = None

    @field_validator("collection_name", "embedding_model")
    @classmethod
    def clean_required_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Dense hybrid text settings must not be empty.")
        return value

    @field_validator("qdrant_url")
    @classmethod
    def clean_optional_url(cls, value):
        if value is None:
            return None
        value = value.strip().rstrip("/")
        return value or None


class ReciprocalRankFusionConfig(StrictModel):
    """RRF scoring and final result depth."""

    type: Literal["rrf"] = "rrf"
    rank_constant: float = Field(default=DEFAULT_RRF_CONSTANT, gt=0)
    top_k: int = Field(default=DEFAULT_HYBRID_TOP_K, ge=1, le=100)

    @field_validator("rank_constant")
    @classmethod
    def validate_finite_rank_constant(cls, value):
        if not math.isfinite(value):
            raise ValueError("rank_constant must be finite.")
        return value


class HybridEvaluationDatasetConfig(StrictModel):
    """Verified labels and fixed component baselines used for Day 25."""

    labels_path: Path = Path("data/eval/retrieval_labels.jsonl")
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10], min_length=1)
    minimum_labels: int = Field(default=40, gt=0)
    dense_baseline_path: Path = Path("reports/evaluations/dense_baseline.json")
    bm25_baseline_path: Path = Path("reports/evaluations/bm25_baseline.json")

    @field_validator("labels_path", "dense_baseline_path", "bm25_baseline_path", mode="before")
    @classmethod
    def validate_evaluation_path(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("Hybrid evaluation paths must not be empty.")
        return value

    @field_validator("k_values")
    @classmethod
    def validate_metric_cutoffs(cls, values):
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("k_values must contain positive integers.")
        if len(values) != len(set(values)):
            raise ValueError("k_values must not contain duplicates.")
        return sorted(values)


class HybridEvaluationOutputConfig(StrictModel):
    """Hybrid run and three-way comparison artifact destinations."""

    directory: Path = Path("reports/evaluations")
    comparison_path: Path = Path("reports/evaluations/hybrid_vs_baselines.json")
    report_path: Path = Path("reports/week4_hybrid_comparison.md")

    @field_validator("directory", "comparison_path", "report_path", mode="before")
    @classmethod
    def validate_output_path(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("Hybrid output paths must not be empty.")
        return value


class HybridConfig(StrictModel):
    """Complete dense-plus-BM25 retrieval and optional evaluation config."""

    name: str = Field(min_length=1)
    version: PipelineVersion = "0.1.0"
    status: PipelineStatus = "draft"
    retriever_interface: Literal["common_v1"] = COMMON_RETRIEVER_INTERFACE
    input: BM25InputConfig
    dense: HybridDenseConfig
    bm25: BM25RetrieverConfig
    fusion: ReciprocalRankFusionConfig
    evaluation: HybridEvaluationDatasetConfig | None = None
    output: HybridEvaluationOutputConfig | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        value = value.strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
            raise ValueError("Hybrid configuration name must use lowercase letters, numbers, underscores, or hyphens.")
        return value

    @model_validator(mode="after")
    def validate_retrieval_depths(self):
        if self.dense.top_k < self.fusion.top_k:
            raise ValueError("dense.top_k must be at least fusion.top_k.")
        if self.bm25.top_k < self.fusion.top_k:
            raise ValueError("bm25.top_k must be at least fusion.top_k.")
        if (self.evaluation is None) != (self.output is None):
            raise ValueError("Hybrid evaluation and output settings must be configured together.")
        if self.evaluation is not None and self.fusion.top_k < max(self.evaluation.k_values):
            raise ValueError("fusion.top_k must be at least the largest evaluation cutoff.")
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


def resolve_project_path(path, project_root):
    """Resolve one configured artifact path relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else (Path(project_root) / path).resolve()


def load_hybrid_config(config_path, project_root=None):
    """Load strict hybrid YAML and resolve its local artifact paths."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Hybrid config does not exist: {config_path}")

    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
    if not isinstance(raw_config, dict) or not raw_config:
        raise ValueError(f"Hybrid config must contain a YAML mapping: {config_path}")

    config = HybridConfig.model_validate(raw_config)
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


def configured_qdrant_url(config):
    """Return the config, environment, or stable local Qdrant URL."""
    qdrant_url = config.dense.qdrant_url or os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
    qdrant_url = qdrant_url.strip().rstrip("/")
    return qdrant_url or DEFAULT_QDRANT_URL


def _validate_positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _validate_rank_constant(value):
    if isinstance(value, bool):
        raise ValueError("rank_constant must be a positive finite number.")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("rank_constant must be a positive finite number.") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("rank_constant must be a positive finite number.")
    return value


def _coerce_ranked_chunk(result, retriever_name, position):
    try:
        chunk = result if isinstance(result, RetrievedChunk) else RetrievedChunk.model_validate(result)
    except Exception as error:
        raise ValueError(f"{retriever_name} result at position {position} is invalid: {error}") from error
    if not chunk.chunk_id.strip():
        raise ValueError(f"{retriever_name} result at position {position} has an empty chunk ID.")
    if chunk.rank != position:
        raise ValueError(f"{retriever_name} result {chunk.chunk_id} has rank {chunk.rank}; expected {position}.")
    if not math.isfinite(float(chunk.score)):
        raise ValueError(f"{retriever_name} result {chunk.chunk_id} has a non-finite score.")
    return chunk


def _validate_matching_payload(existing, candidate):
    if existing.document_id != candidate.document_id:
        raise ValueError(f"Chunk {existing.chunk_id} has conflicting document IDs across retrievers.")
    if existing.text != candidate.text:
        raise ValueError(f"Chunk {existing.chunk_id} has conflicting text across retrievers.")


def reciprocal_rank_fusion(rankings, top_k=DEFAULT_HYBRID_TOP_K, rank_constant=DEFAULT_RRF_CONSTANT):
    """Fuse named rankings with score=sum(1/(rank_constant + rank))."""
    top_k = _validate_positive_integer(top_k, "top_k")
    rank_constant = _validate_rank_constant(rank_constant)
    if not isinstance(rankings, Mapping) or not rankings:
        raise ValueError("rankings must be a non-empty mapping of retriever names to ranked results.")

    fused = {}
    for retriever_name, raw_results in rankings.items():
        if not isinstance(retriever_name, str) or not retriever_name.strip():
            raise ValueError("Retriever names must be non-empty strings.")
        retriever_name = retriever_name.strip()
        if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
            raise ValueError(f"Ranking {retriever_name} must be a sequence.")

        seen_chunk_ids = set()
        for position, raw_result in enumerate(raw_results, start=1):
            chunk = _coerce_ranked_chunk(raw_result, retriever_name, position)
            if chunk.chunk_id in seen_chunk_ids:
                raise ValueError(f"Ranking {retriever_name} contains duplicate chunk ID {chunk.chunk_id}.")
            seen_chunk_ids.add(chunk.chunk_id)

            contribution = 1.0 / (rank_constant + position)
            entry = fused.get(chunk.chunk_id)
            if entry is None:
                entry = {
                    "chunk": chunk,
                    "score": 0.0,
                    "best_rank": position,
                    "sources": {},
                }
                fused[chunk.chunk_id] = entry
            else:
                _validate_matching_payload(entry["chunk"], chunk)
                if position < entry["best_rank"]:
                    entry["chunk"] = chunk
                    entry["best_rank"] = position

            entry["score"] += contribution
            entry["sources"][retriever_name] = {
                "rank": position,
                "score": float(chunk.score),
                "rrf_contribution": contribution,
            }

    if not fused:
        return []

    ordered = sorted(
        fused.values(),
        key=lambda entry: (-entry["score"], -len(entry["sources"]), entry["best_rank"], entry["chunk"].chunk_id),
    )[:top_k]
    results = []
    for final_rank, entry in enumerate(ordered, start=1):
        original = entry["chunk"]
        metadata = dict(original.metadata)
        metadata[FUSION_METADATA_KEY] = {
            "method": "rrf",
            "rank_constant": rank_constant,
            "sources": entry["sources"],
        }
        results.append(
            RetrievedChunk(
                chunk_id=original.chunk_id,
                document_id=original.document_id,
                text=original.text,
                score=entry["score"],
                rank=final_rank,
                metadata=metadata,
                source_url=original.source_url,
            )
        )
    return results


class HybridRetriever(Retriever):
    """Configured dense-plus-sparse RRF pipeline using the common interface."""

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        dense_top_k=DEFAULT_DENSE_CANDIDATES,
        bm25_top_k=DEFAULT_BM25_CANDIDATES,
        default_top_k=DEFAULT_HYBRID_TOP_K,
        rank_constant=DEFAULT_RRF_CONSTANT,
        clock=time.perf_counter,
    ):
        super().__init__(default_top_k)
        if not callable(getattr(dense_retriever, "retrieve", None)):
            raise ValueError("dense_retriever must implement retrieve(query, top_k, timings).")
        if not callable(getattr(bm25_retriever, "retrieve", None)):
            raise ValueError("bm25_retriever must implement retrieve(query, top_k, timings).")
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_top_k = validate_top_k(dense_top_k, "dense_top_k")
        self.bm25_top_k = validate_top_k(bm25_top_k, "bm25_top_k")
        self.rank_constant = _validate_rank_constant(rank_constant)
        self.clock = clock
        if self.dense_top_k < self.default_top_k or self.bm25_top_k < self.default_top_k:
            raise ValueError("Candidate depths must each be at least default_top_k.")

    def retrieve(self, query, top_k=None, timings=None):
        query = validate_query(query)
        top_k = resolve_top_k(top_k, self.default_top_k)
        validate_timings(timings)
        if self.dense_top_k < top_k or self.bm25_top_k < top_k:
            raise ValueError("Candidate depths must each be at least top_k.")

        dense_stage_timings = {}
        dense_started_at = self.clock() if timings is not None else None
        try:
            dense_results = list(
                self.dense_retriever.retrieve(
                    query,
                    top_k=self.dense_top_k,
                    timings=dense_stage_timings if timings is not None else None,
                )
            )
        except Exception as error:
            raise RuntimeError(f"Dense candidate retrieval failed: {error}") from error
        finally:
            if dense_started_at is not None:
                dense_stage_timings.setdefault("dense_ms", max(0.0, (self.clock() - dense_started_at) * 1000))
                timings.update(dense_stage_timings)

        bm25_stage_timings = {}
        bm25_started_at = self.clock() if timings is not None else None
        try:
            bm25_results = list(
                self.bm25_retriever.retrieve(
                    query,
                    top_k=self.bm25_top_k,
                    timings=bm25_stage_timings if timings is not None else None,
                )
            )
        except Exception as error:
            raise RuntimeError(f"BM25 candidate retrieval failed: {error}") from error
        finally:
            if bm25_started_at is not None:
                bm25_stage_timings.setdefault("bm25_ms", max(0.0, (self.clock() - bm25_started_at) * 1000))
                timings.update(bm25_stage_timings)

        fusion_started_at = self.clock() if timings is not None else None
        try:
            return reciprocal_rank_fusion(
                {"dense": dense_results, "bm25": bm25_results},
                top_k=top_k,
                rank_constant=self.rank_constant,
            )
        finally:
            if fusion_started_at is not None:
                timings["fusion_ms"] = max(0.0, (self.clock() - fusion_started_at) * 1000)


def build_hybrid_retriever(
    config,
    client,
    index,
    dense_retriever=retrieve_dense,
    bm25_retriever=retrieve_bm25,
    clock=time.perf_counter,
    query_embedder=None,
):
    """Build one common-interface hybrid pipeline from validated config."""
    if dense_retriever is retrieve_dense:
        parameters = {}
        if query_embedder is not None:
            parameters["query_embedder"] = query_embedder
        dense = DenseRetriever(
            client,
            collection_name=config.dense.collection_name,
            embedding_model=config.dense.embedding_model,
            default_top_k=config.dense.top_k,
            clock=clock,
            **parameters,
        )
    else:
        dense = FunctionRetriever(
            dense_retriever,
            config.dense.top_k,
            client=client,
            collection_name=config.dense.collection_name,
            embedding_model=config.dense.embedding_model,
        )
    if bm25_retriever is retrieve_bm25:
        sparse = BM25Retriever(index, default_top_k=config.bm25.top_k, clock=clock)
    else:
        sparse = FunctionRetriever(bm25_retriever, config.bm25.top_k, index=index)
    return HybridRetriever(
        dense,
        sparse,
        dense_top_k=config.dense.top_k,
        bm25_top_k=config.bm25.top_k,
        default_top_k=config.fusion.top_k,
        rank_constant=config.fusion.rank_constant,
        clock=clock,
    )


def retrieve_hybrid(
    query,
    client,
    index,
    dense_top_k=DEFAULT_DENSE_CANDIDATES,
    bm25_top_k=DEFAULT_BM25_CANDIDATES,
    top_k=DEFAULT_HYBRID_TOP_K,
    rank_constant=DEFAULT_RRF_CONSTANT,
    collection_name=DEFAULT_COLLECTION_NAME,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    dense_retriever=retrieve_dense,
    bm25_retriever=retrieve_bm25,
    clock=time.perf_counter,
    timings=None,
):
    """Retrieve dense and sparse candidates, then fuse them into one ranking."""
    dense = FunctionRetriever(
        dense_retriever,
        dense_top_k,
        client=client,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    sparse = FunctionRetriever(bm25_retriever, bm25_top_k, index=index)
    retriever = HybridRetriever(
        dense,
        sparse,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
        default_top_k=top_k,
        rank_constant=rank_constant,
        clock=clock,
    )
    return retriever.retrieve(query, timings=timings)


def retrieve_hybrid_config(
    query,
    config,
    client,
    index,
    dense_retriever=retrieve_dense,
    bm25_retriever=retrieve_bm25,
    clock=time.perf_counter,
    timings=None,
):
    """Run hybrid retrieval using one validated Day 24 configuration."""
    retriever = build_hybrid_retriever(config, client, index, dense_retriever=dense_retriever, bm25_retriever=bm25_retriever, clock=clock)
    return retriever.retrieve(query, timings=timings)
