from types import SimpleNamespace

import pytest

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

    result = app.query_api("How do I create an app?", 3, "http://api:8000")

    assert calls["request"] == ("http://api:8000/query", {"query": "How do I create an app?", "top_k": 3}, app.REQUEST_TIMEOUT_SECONDS)
    assert result["answer"] == "Use FastAPI. [1]"


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
