import math
import time
from contextlib import contextmanager

from pydantic import BaseModel, ConfigDict, Field, field_validator

COMPONENT_TIMING_FIELDS = (
    "embedding_ms",
    "dense_ms",
    "bm25_ms",
    "fusion_ms",
    "reranker_ms",
    "generation_ms",
)


class ComponentLatencies(BaseModel):
    """Validated request-stage latencies in milliseconds."""

    model_config = ConfigDict(extra="forbid")

    embedding_ms: float | None = Field(default=None, ge=0)
    dense_ms: float | None = Field(default=None, ge=0)
    bm25_ms: float | None = Field(default=None, ge=0)
    fusion_ms: float | None = Field(default=None, ge=0)
    reranker_ms: float | None = Field(default=None, ge=0)
    generation_ms: float | None = Field(default=None, ge=0)

    @field_validator(*COMPONENT_TIMING_FIELDS)
    @classmethod
    def validate_finite_latency(cls, value):
        if value is not None and not math.isfinite(value):
            raise ValueError("Component latencies must be finite when provided.")
        return value

    @classmethod
    def from_mapping(cls, timings):
        """Build a snapshot from a retriever timing sink, ignoring aggregate-only keys."""
        timings = timings or {}
        if not isinstance(timings, dict):
            raise ValueError("Component timings must be a dictionary.")
        return cls.model_validate({field: timings[field] for field in COMPONENT_TIMING_FIELDS if field in timings})

    def recorded(self):
        """Return only stages that ran for the selected pipeline."""
        return self.model_dump(exclude_none=True)


class TraceContext:
    """Request-scoped monotonic timer and mutable retriever timing sink."""

    def __init__(self, clock=time.perf_counter):
        if not callable(clock):
            raise ValueError("Trace clock must be callable.")
        self.clock = clock
        self.started_at = self.clock()
        self.timings = {}

    @contextmanager
    def measure(self, component):
        """Measure one component, retaining elapsed time when the stage raises."""
        timing_field = f"{component}_ms"
        if timing_field not in COMPONENT_TIMING_FIELDS:
            supported = ", ".join(field.removesuffix("_ms") for field in COMPONENT_TIMING_FIELDS)
            raise ValueError(f"Unsupported trace component {component!r}; choose one of: {supported}.")
        started_at = self.clock()
        try:
            yield self
        finally:
            elapsed = max(0.0, (self.clock() - started_at) * 1000)
            previous = self.timings.get(timing_field, 0.0)
            self.timings[timing_field] = previous + elapsed

    def snapshot(self):
        """Return validated component latencies without aggregate retriever keys."""
        return ComponentLatencies.from_mapping(self.timings)

    def total_ms(self):
        """Return non-negative whole-request elapsed time in milliseconds."""
        return max(0.0, (self.clock() - self.started_at) * 1000)
