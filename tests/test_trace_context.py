import pytest
from pydantic import ValidationError

from ragops.tracing.context import ComponentLatencies, TraceContext


def test_trace_context_measures_components_and_total_with_monotonic_clock():
    clock_values = iter([10.0, 10.1, 10.13, 10.2])
    context = TraceContext(clock=lambda: next(clock_values))

    with context.measure("embedding"):
        pass

    assert context.snapshot().recorded() == {"embedding_ms": pytest.approx(30.0)}
    assert context.total_ms() == pytest.approx(200.0)


def test_trace_context_retains_failed_stage_time_and_accumulates_repeated_stages():
    clock_values = iter([0.0, 0.1, 0.15, 0.2, 0.23])
    context = TraceContext(clock=lambda: next(clock_values))

    with pytest.raises(RuntimeError, match="provider failed"):
        with context.measure("generation"):
            raise RuntimeError("provider failed")
    with context.measure("generation"):
        pass

    assert context.snapshot().generation_ms == pytest.approx(80.0)


def test_trace_context_rejects_unknown_components():
    context = TraceContext(clock=lambda: 0.0)

    with pytest.raises(ValueError, match="Unsupported trace component"):
        with context.measure("routing"):
            pass


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_component_latencies_reject_negative_or_non_finite_values(value):
    with pytest.raises(ValidationError):
        ComponentLatencies(embedding_ms=value)


def test_component_latency_snapshot_ignores_retriever_aggregate_and_keeps_null_stage_shape():
    snapshot = ComponentLatencies.from_mapping({"embedding_ms": 1.0, "dense_ms": 2.0, "total_ms": 4.0})

    assert snapshot.recorded() == {"embedding_ms": 1.0, "dense_ms": 2.0}
    assert snapshot.model_dump() == {
        "embedding_ms": 1.0,
        "dense_ms": 2.0,
        "bm25_ms": None,
        "fusion_ms": None,
        "reranker_ms": None,
        "generation_ms": None,
    }
