from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from dashboard import app


def test_get_api_url_uses_default(monkeypatch):
    monkeypatch.delenv("RAGOPS_API_URL", raising=False)

    assert app.get_api_url() == app.DEFAULT_API_URL


def test_get_api_url_strips_whitespace_and_trailing_slash(monkeypatch):
    monkeypatch.setenv("RAGOPS_API_URL", "  http://api:8000/  ")

    assert app.get_api_url() == "http://api:8000"


def test_query_api_posts_expected_payload(monkeypatch):
    calls = {}

    def fake_post(endpoint, json, timeout):
        calls["request"] = (endpoint, json, timeout)
        return SimpleNamespace(ok=True, status_code=200, json=lambda: {"answer": "Use FastAPI. [1]", "citations": [], "chunks": [], "latency_ms": 12.5})

    monkeypatch.setattr(app.requests, "post", fake_post)

    result = app.query_api("How do I create an app?", 3, "http://api:8000", config="hybrid_rrf")

    assert calls["request"] == (
        "http://api:8000/query",
        {"query": "How do I create an app?", "top_k": 3, "config": "hybrid_rrf", "debug": True},
        app.REQUEST_TIMEOUT_SECONDS,
    )
    assert result["answer"] == "Use FastAPI. [1]"


def test_route_api_posts_expected_payload(monkeypatch):
    calls = {}

    def fake_post(endpoint, json, timeout):
        calls["request"] = (endpoint, json, timeout)
        return SimpleNamespace(ok=True, status_code=200, json=lambda: {"decision": {"route": "FAST"}})

    monkeypatch.setattr(app.requests, "post", fake_post)

    app.route_api("What is FastAPI?", "http://api:8000")

    assert calls["request"] == (
        "http://api:8000/route",
        {"query": "What is FastAPI?"},
        app.REQUEST_TIMEOUT_SECONDS,
    )


def test_query_api_uses_api_error_detail(monkeypatch):
    response = SimpleNamespace(ok=False, status_code=503, json=lambda: {"detail": "Unable to generate answer."})
    monkeypatch.setattr(app.requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="Unable to generate answer"):
        app.query_api("question", 5, "http://api:8000")


def test_query_api_reports_connection_failure(monkeypatch):
    def fake_post(*args, **kwargs):
        raise app.requests.ConnectionError("connection refused")

    monkeypatch.setattr(app.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="Make sure the FastAPI server is running"):
        app.query_api("question", 5, "http://api:8000")


def _route_result(route="CAREFUL", pipeline_config="hybrid_rrf_cross_encoder", maximum_top_k=5):
    return {
        "decision": {
            "route": route,
            "reason": "The query requires careful retrieval.",
            "reason_code": "score_gap_below_careful_threshold",
            "matched_reason_codes": ["score_gap_below_careful_threshold"],
            "pipeline_config": pipeline_config,
            "maximum_top_k": maximum_top_k,
            "router_id": "rule_router@0.2.0",
            "router_status": "draft",
        },
        "features": {"retrieval_confidence": {"top_score": 0.7, "score_gap": 0.01}},
        "probe_chunks": [{"chunk_id": "probe-1", "score": 0.7, "rank": 1}],
        "probe_timings": {"total_ms": 12.5, "embedding_ms": 10.0, "dense_ms": 2.5},
        "refusal": None,
    }


def test_execute_routed_query_honors_pipeline_and_top_k_cap(monkeypatch):
    calls = {}
    monkeypatch.setattr(app, "route_api", lambda query, api_url: _route_result())

    def fake_query(query, top_k, api_url, config):
        calls["query"] = (query, top_k, api_url, config)
        return {
            "query": query,
            "config": config,
            "config_version": "1.0.0",
            "route": "reranked",
            "latency_ms": 40.0,
            "answer": "Answer [1]",
            "citations": [],
            "chunks": [],
            "cost": {"amount_usd": 0.0001},
        }

    monkeypatch.setattr(app, "query_api", fake_query)

    result = app.execute_routed_query("Explain the tradeoff", 10, "http://api:8000")

    assert calls["query"] == (
        "Explain the tradeoff",
        5,
        "http://api:8000",
        "hybrid_rrf_cross_encoder",
    )
    assert result["routing"]["route"] == "CAREFUL"
    assert result["effective_top_k"] == 5
    assert result["total_latency_ms"] == pytest.approx(52.5)


def test_execute_routed_query_returns_refusal_without_query_call(monkeypatch):
    route_result = _route_result(route="NO_ANSWER", pipeline_config=None, maximum_top_k=0)
    route_result["decision"].update(
        {
            "reason": "The best result is below the threshold.",
            "reason_code": "top_score_below_no_answer_threshold",
        }
    )
    route_result["refusal"] = {"answer": "I cannot answer from the indexed documentation."}
    monkeypatch.setattr(app, "route_api", lambda query, api_url: route_result)
    monkeypatch.setattr(
        app,
        "query_api",
        lambda *args, **kwargs: pytest.fail("NO_ANSWER must not call /query"),
    )

    result = app.execute_routed_query("What is the weather?", 5, "http://api:8000")

    assert result["routing"]["route"] == "NO_ANSWER"
    assert result["answer"] == "I cannot answer from the indexed documentation."
    assert result["total_latency_ms"] == 12.5
    assert result["trace_id"] is None
    assert result["cost"] is None


def test_execute_routed_query_rejects_malformed_decision(monkeypatch):
    monkeypatch.setattr(app, "route_api", lambda query, api_url: {"decision": {"route": "UNKNOWN"}})

    with pytest.raises(RuntimeError, match="unsupported route"):
        app.execute_routed_query("question", 5, "http://api:8000")


def test_reranked_chunk_separates_retrieval_and_cross_encoder_scores():
    chunk = {
        "score": 8.75,
        "metadata": {"_reranker": {"candidate_rank": 4, "candidate_score": 0.03125}},
    }

    assert app.chunk_score_labels(chunk) == {
        "retrieval": "0.0312",
        "reranker": "8.7500",
        "candidate_rank": 4,
    }


def test_engineering_artifacts_expose_required_measured_values():
    benchmark = app.load_final_benchmark(app.PROJECT_ROOT / app.DEFAULT_FINAL_BENCHMARK_PATH)
    rows = app.benchmark_rows(benchmark)
    routed = app.load_routed_summary(app.PROJECT_ROOT / app.DEFAULT_ROUTED_REPORT_PATH)
    failures = app.load_failure_examples(app.PROJECT_ROOT / app.DEFAULT_FAILURE_CONFIG_PATH)

    assert [row["pipeline"] for row in rows] == ["Dense", "BM25", "Hybrid (RRF)", "Hybrid + reranker", "Routed"]
    assert next(row for row in rows if row["pipeline"] == "Hybrid + reranker")["recall_at_5"] == pytest.approx(0.81)
    assert routed["counts"] == {"CAREFUL": 29, "FAST": 2, "NO_ANSWER": 7, "STANDARD": 12}
    assert routed["no_answer_rate"] == pytest.approx(0.14)
    assert len(failures) == 15


def test_dashboard_renders_exactly_two_primary_tabs_and_engineering_story():
    dashboard = AppTest.from_file(str(app.PROJECT_ROOT / "dashboard/app.py"), default_timeout=120)

    dashboard.run()

    assert not dashboard.exception
    assert [tab.label for tab in dashboard.tabs] == ["Query Playground", "Engineering"]
    assert [header.value for header in dashboard.header] == ["Query Playground", "Engineering"]
    assert [subheader.value for subheader in dashboard.subheader] == [
        "Final benchmark",
        "Quality vs. latency",
        "Route distribution",
        "Recent traces",
        "Selected failure examples",
    ]
    metrics = {metric.label: metric.value for metric in dashboard.metric}
    assert metrics["Routed p50"] == "3,109.8 ms"
    assert metrics["Routed p95"] == "7,821.4 ms"
    assert metrics["Avg estimated cost"] == "$0.00007603"
    assert metrics["Supported NO_ANSWER rate"] == "14.0%"
