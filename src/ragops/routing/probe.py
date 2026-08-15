import math
import re
import time
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragops.retrieval.dense import RetrievedChunk, validate_query

INITIAL_PROBE_TOP_K = 2
LONG_TOKEN_MIN_CHARACTERS = 8
FEATURE_SCHEMA_VERSION = 1

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][^\W_]+)*", re.UNICODE)
CLAUSE_MARKERS = frozenset(
    {
        "although",
        "and",
        "because",
        "but",
        "if",
        "or",
        "that",
        "though",
        "unless",
        "when",
        "whereas",
        "which",
        "while",
    }
)
COMPLEXITY_MARKERS = frozenset(
    {
        "analyse",
        "analyze",
        "compare",
        "contrast",
        "difference",
        "differences",
        "evaluate",
        "explain",
        "relationship",
        "relationships",
        "step-by-step",
        "trade-off",
        "trade-offs",
        "tradeoff",
        "tradeoffs",
        "versus",
        "why",
    }
)


class StrictFeatureModel(BaseModel):
    """Base model that rejects silent feature-schema drift."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class QueryLengthFeatures(StrictFeatureModel):
    """Stable query-size signals available before route selection."""

    character_count: int = Field(ge=1)
    token_count: int = Field(ge=1)


class LexicalComplexityFeatures(StrictFeatureModel):
    """Cheap deterministic lexical signals; no model or corpus access is used."""

    unique_token_count: int = Field(ge=1)
    unique_token_ratio: float = Field(ge=0, le=1)
    average_token_length: float = Field(gt=0)
    maximum_token_length: int = Field(ge=1)
    long_token_count: int = Field(ge=0)
    long_token_ratio: float = Field(ge=0, le=1)
    clause_marker_count: int = Field(ge=0)
    complexity_marker_count: int = Field(ge=0)

    @field_validator("unique_token_ratio", "average_token_length", "long_token_ratio")
    @classmethod
    def require_finite_ratios(cls, value):
        if not math.isfinite(value):
            raise ValueError("Lexical feature values must be finite.")
        return value


class RetrievalConfidenceFeatures(StrictFeatureModel):
    """Confidence signals extracted from the ordered dense top-two results."""

    requested_top_k: Literal[2] = INITIAL_PROBE_TOP_K
    result_count: int = Field(ge=0, le=INITIAL_PROBE_TOP_K)
    top_score: float | None = None
    score_gap: float | None = Field(default=None, ge=0)

    @field_validator("top_score", "score_gap")
    @classmethod
    def require_finite_scores(cls, value):
        if value is not None and not math.isfinite(value):
            raise ValueError("Retrieval confidence scores must be finite when present.")
        return value

    @model_validator(mode="after")
    def require_score_shape_for_result_count(self):
        if self.result_count == 0 and (self.top_score is not None or self.score_gap is not None):
            raise ValueError("An empty probe cannot contain a top score or score gap.")
        if self.result_count == 1 and (self.top_score is None or self.score_gap is not None):
            raise ValueError("A one-result probe requires a top score and cannot contain a score gap.")
        if self.result_count >= 2 and (self.top_score is None or self.score_gap is None):
            raise ValueError("A two-result probe requires both a top score and score gap.")
        return self


class InitialRetrievalFeatures(StrictFeatureModel):
    """The complete Day 37 input contract for the future rule-based router."""

    schema_version: Literal[1] = FEATURE_SCHEMA_VERSION
    query_length: QueryLengthFeatures
    lexical_complexity: LexicalComplexityFeatures
    retrieval_confidence: RetrievalConfidenceFeatures


class ProbeTimings(StrictFeatureModel):
    """Probe-only timings kept separate from route-selection features."""

    total_ms: float = Field(ge=0)
    embedding_ms: float | None = Field(default=None, ge=0)
    dense_ms: float | None = Field(default=None, ge=0)

    @field_validator("total_ms", "embedding_ms", "dense_ms")
    @classmethod
    def require_finite_timings(cls, value):
        if value is not None and not math.isfinite(value):
            raise ValueError("Probe timings must be finite when present.")
        return value


class InitialProbeResult(StrictFeatureModel):
    """Structured features plus reusable dense evidence from one cheap probe."""

    query: str = Field(min_length=1)
    features: InitialRetrievalFeatures
    chunks: tuple[RetrievedChunk, ...]
    timings: ProbeTimings


def tokenize_query(query):
    """Return normalized Unicode word tokens used by all lexical features."""
    query = validate_query(query)
    tokens = TOKEN_PATTERN.findall(query)
    if not tokens:
        raise ValueError("query must contain at least one word or number token.")
    return tuple(token.casefold() for token in tokens)


def extract_lexical_complexity(query):
    """Extract deterministic complexity signals without retrieval or model calls."""
    tokens = tokenize_query(query)
    token_lengths = [len(token) for token in tokens]
    unique_tokens = set(tokens)
    long_token_count = sum(length >= LONG_TOKEN_MIN_CHARACTERS for length in token_lengths)
    return LexicalComplexityFeatures(
        unique_token_count=len(unique_tokens),
        unique_token_ratio=len(unique_tokens) / len(tokens),
        average_token_length=sum(token_lengths) / len(tokens),
        maximum_token_length=max(token_lengths),
        long_token_count=long_token_count,
        long_token_ratio=long_token_count / len(tokens),
        clause_marker_count=sum(token in CLAUSE_MARKERS for token in tokens),
        complexity_marker_count=sum(token in COMPLEXITY_MARKERS for token in tokens),
    )


def _validated_probe_chunks(chunks):
    chunks = tuple(chunk if isinstance(chunk, RetrievedChunk) else RetrievedChunk.model_validate(chunk) for chunk in chunks)
    if len(chunks) > INITIAL_PROBE_TOP_K:
        raise ValueError(f"Initial dense probe returned {len(chunks)} results; expected at most {INITIAL_PROBE_TOP_K}.")

    expected_ranks = list(range(1, len(chunks) + 1))
    ranks = [chunk.rank for chunk in chunks]
    if ranks != expected_ranks:
        raise ValueError(f"Initial dense probe ranks must be contiguous and one-based; received {ranks}.")

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Initial dense probe returned duplicate chunk IDs.")

    scores = [chunk.score for chunk in chunks]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Initial dense probe returned a non-finite score.")
    if any(left < right for left, right in zip(scores, scores[1:], strict=False)):
        raise ValueError("Initial dense probe results must be ordered by descending score.")
    return chunks


def build_initial_retrieval_features(query, chunks):
    """Build the router feature contract from a cleaned query and dense evidence."""
    query = validate_query(query)
    tokens = tokenize_query(query)
    chunks = _validated_probe_chunks(chunks)
    scores = [chunk.score for chunk in chunks]
    top_score = scores[0] if scores else None
    score_gap = scores[0] - scores[1] if len(scores) >= 2 else None
    return InitialRetrievalFeatures(
        query_length=QueryLengthFeatures(character_count=len(query), token_count=len(tokens)),
        lexical_complexity=extract_lexical_complexity(query),
        retrieval_confidence=RetrievalConfidenceFeatures(
            result_count=len(chunks),
            top_score=top_score,
            score_gap=score_gap,
        ),
    )


def _timing_value(timings, name):
    value = timings.get(name)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Initial dense probe recorded a non-numeric {name} value.") from error
    return value


def run_initial_retrieval_probe(
    query: str,
    retrieve: Callable[..., Sequence[RetrievedChunk]],
    *,
    clock: Callable[[], float] = time.perf_counter,
):
    """Run dense top-two retrieval once and return validated reusable router input."""
    query = validate_query(query)
    if not callable(retrieve):
        raise ValueError("retrieve must be callable.")
    if not callable(clock):
        raise ValueError("clock must be callable.")

    timings = {}
    started_at = clock()
    chunks = retrieve(query=query, top_k=INITIAL_PROBE_TOP_K, timings=timings)
    total_ms = max(0.0, (clock() - started_at) * 1000)
    chunks = _validated_probe_chunks(chunks)
    return InitialProbeResult(
        query=query,
        features=build_initial_retrieval_features(query, chunks),
        chunks=chunks,
        timings=ProbeTimings(
            total_ms=total_ms,
            embedding_ms=_timing_value(timings, "embedding_ms"),
            dense_ms=_timing_value(timings, "dense_ms"),
        ),
    )
