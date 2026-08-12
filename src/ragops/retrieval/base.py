import time
from abc import ABC, abstractmethod

COMMON_RETRIEVER_INTERFACE = "common_v1"


def validate_top_k(top_k, name="top_k"):
    """Return a positive retrieval depth or raise a stable validation error."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return top_k


def resolve_top_k(top_k, default_top_k):
    """Resolve an optional per-query depth against a retriever default."""
    return validate_top_k(default_top_k if top_k is None else top_k)


def validate_timings(timings):
    """Validate the optional mutable timing sink shared by all retrievers."""
    if timings is not None and not isinstance(timings, dict):
        raise ValueError("timings must be a dictionary when provided.")
    return timings


class Retriever(ABC):
    """Common interface implemented by every retrieval pipeline."""

    interface = COMMON_RETRIEVER_INTERFACE

    def __init__(self, default_top_k):
        self.default_top_k = validate_top_k(default_top_k, "default_top_k")

    @abstractmethod
    def retrieve(self, query, top_k=None, timings=None):
        """Return ranked chunks for one query."""

    def __call__(self, query, top_k=None, timings=None):
        return self.retrieve(query, top_k=top_k, timings=timings)


class FunctionRetriever(Retriever):
    """Adapt an existing keyword-based retrieval function to Retriever."""

    def __init__(self, function, default_top_k, timing_name=None, clock=time.perf_counter, **parameters):
        if not callable(function):
            raise ValueError("function must be callable.")
        super().__init__(default_top_k)
        self.function = function
        self.parameters = parameters
        self.timing_name = timing_name
        self.clock = clock

    def retrieve(self, query, top_k=None, timings=None):
        top_k = resolve_top_k(top_k, self.default_top_k)
        validate_timings(timings)
        started_at = self.clock() if timings is not None and self.timing_name else None
        results = list(self.function(query=query, top_k=top_k, **self.parameters))
        if started_at is not None:
            timings[self.timing_name] = max(0.0, (self.clock() - started_at) * 1000)
        return results
