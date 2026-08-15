from datetime import UTC, datetime

import pytest

from ragops import __version__
from ragops.evaluation.api_runner import (
    compare_reference_report,
    configured_api_url,
    load_reference_report,
    run_api_evaluation,
    summarize_component_latencies,
)
from ragops.evaluation.retrieval_labels import RetrievalLabel
from ragops.evaluation.runner import EvaluationDatasetConfig, EvaluationOutputConfig, RetrievalEvaluationConfig
from ragops.tracing.store import RetrievedChunkTrace, TraceRecord, TraceStore

TRACE_ID = "12345678-1234-4123-8123-123456789abc"


class FakeResponse:

    def __init__(self, payload, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = str(payload)

    def json(self):
        return self.payload


class FakeClient:

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.closed = False

    def get(self, path):
        self.requests.append(("GET", path, None))
        return FakeResponse({"status": "ok", "version": __version__})

    def post(self, path, json):
        self.requests.append(("POST", path, json))
        return next(self.responses)

    def close(self):
        self.closed = True


class FakeClock:

    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def make_config(tmp_path):
    return RetrievalEvaluationConfig(
        name="dense_baseline",
        version="1.0.0",
        status="approved",
        retriever={
            "type": "dense",
            "collection_name": "rag_chunks",
            "embedding_model": "test-model",
            "top_k": 2,
        },
        evaluation=EvaluationDatasetConfig(
            labels_path=tmp_path / "labels.jsonl",
            k_values=[1, 2],
            minimum_labels=1,
        ),
        output=EvaluationOutputConfig(directory=tmp_path / "reports"),
    )


def make_label():
    return RetrievalLabel(
        question_id="q1",
        question="How does the documented API evaluation work?",
        relevant_chunk_ids=["relevant"],
        expected_source="docs/api.md",
        metadata={"label_method": "manual", "reviewed_by": "test-reviewer"},
    )


def response_payload(trace_id=TRACE_ID, chunks=None, latency_ms=10.0):
    chunks = chunks or [
        {
            "chunk_id": "relevant",
            "document_id": "doc-1",
            "text": "Relevant evidence.",
            "score": 0.9,
            "rank": 1,
            "metadata": {"title": "API"},
            "source_url": "docs/api.md",
        },
        {
            "chunk_id": "other",
            "document_id": "doc-2",
            "text": "Other evidence.",
            "score": 0.5,
            "rank": 2,
            "metadata": {"title": "Other"},
            "source_url": "docs/other.md",
        },
    ]
    return {
        "trace_id": trace_id,
        "route": "dense",
        "config": "dense_baseline",
        "config_version": "1.0.0",
        "query": "How does the documented API evaluation work?",
        "answer": "The API evaluates retrieved evidence. [1]",
        "citations": [
            {
                "citation_id": "[1]",
                "document_id": "doc-1",
                "title": "API",
                "url": "docs/api.md",
                "metadata": {},
                "chunk_ids": ["relevant"],
            }
        ],
        "citation_text": "[1] API - docs/api.md",
        "chunks": chunks,
        "used_chunk_ids": ["relevant"],
        "latency_ms": latency_ms,
        "component_latencies": {
            "embedding_ms": 1.0,
            "dense_ms": 2.0,
            "bm25_ms": None,
            "fusion_ms": None,
            "reranker_ms": None,
            "generation_ms": 1.0,
        },
        "cost": {
            "amount_usd": 0.0,
            "currency": "USD",
            "status": "zero_cost",
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
        "debug": {
            "pipeline_id": "dense_baseline@1.0.0",
            "pipeline_status": "approved",
            "retriever_interface": "common_v1",
            "requested_top_k": 2,
            "returned_chunks": len(chunks),
            "configured_depths": {"dense": 10},
            "generation_provider": "template",
            "generation_model": None,
            "resource_cache_hits": {},
        },
    }


def make_response(**overrides):
    payload = response_payload(**overrides)
    return FakeResponse(payload, headers={"X-Trace-ID": str(payload["trace_id"])})


def record_matching_trace(store, payload, total_latency_ms=None):
    now = datetime.now(UTC)
    store.record_trace(
        TraceRecord(
            trace_id=payload["trace_id"],
            created_at=now,
            completed_at=now,
            endpoint="query",
            query=payload["query"],
            requested_top_k=2,
            pipeline_name="dense_baseline",
            pipeline_version="1.0.0",
            status="success",
            retrieved_chunk_count=len(payload["chunks"]),
            answer=payload["answer"],
            total_latency_ms=payload["latency_ms"] if total_latency_ms is None else total_latency_ms,
            embedding_ms=1.0,
            dense_ms=2.0,
            generation_ms=1.0,
        ),
        [
            RetrievedChunkTrace(
                rank=chunk["rank"],
                chunk_id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                text=chunk["text"],
                score=chunk["score"],
                source_url=chunk["source_url"],
                metadata=chunk["metadata"],
                used_for_generation=chunk["chunk_id"] in payload["used_chunk_ids"],
            )
            for chunk in payload["chunks"]
        ],
    )


def reference_report():
    return {
        "run_name": "dense_baseline",
        "metrics": {
            "question_count": 1,
            "k_values": [1, 2],
            "mrr": 1.0,
            "recall_at_k": {"1": 1.0, "2": 1.0},
            "hit_rate_at_k": {"1": 1.0, "2": 1.0},
            "ndcg_at_k": {"1": 1.0, "2": 1.0},
        },
        "questions": [
            {
                "question_id": "q1",
                "question": "How does the documented API evaluation work?",
                "expected_source": "docs/api.md",
                "relevant_chunk_ids": ["relevant"],
                "retrieved_chunk_ids": ["relevant", "other"],
            }
        ],
    }


def test_run_api_evaluation_checks_health_metrics_reference_and_trace(tmp_path):
    config = make_config(tmp_path)
    label = make_label()
    payload = response_payload()
    response = FakeResponse(payload, headers={"x-trace-id": TRACE_ID})
    client = FakeClient([response])
    store = TraceStore(tmp_path / "traces.sqlite3").initialize()
    record_matching_trace(store, payload)
    progress = []

    report = run_api_evaluation(
        config,
        [label],
        client=client,
        trace_store=store,
        reference_report=reference_report(),
        clock=FakeClock([1.0, 1.025]),
        progress=progress.append,
    )

    assert client.requests[0] == ("GET", "/health", None)
    assert client.requests[1][2] == {
        "query": label.question,
        "top_k": 2,
        "config": "dense_baseline",
        "debug": True,
    }
    assert not client.closed
    assert report["run_name"] == "dense_baseline_api"
    assert report["metrics"]["mrr"] == 1.0
    assert report["questions"][0]["trace_id"] == TRACE_ID
    assert report["client_latency_ms"]["average"] == pytest.approx(25.0)
    assert report["component_latency_ms"]["embedding_ms"]["average"] == 1.0
    assert report["component_latency_ms"]["bm25_ms"] is None
    assert report["reference_comparison"]["exact_ranking_match_count"] == 1
    assert report["trace_verification"]["verified_trace_count"] == 1
    assert progress[0]["trace_id"] == TRACE_ID


def test_run_api_evaluation_rejects_http_failure(tmp_path):
    client = FakeClient([FakeResponse({"detail": "unavailable"}, status_code=503)])

    with pytest.raises(RuntimeError, match="HTTP 503: unavailable"):
        run_api_evaluation(make_config(tmp_path), [make_label()], client=client, clock=FakeClock([0.0, 0.1]))


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        (
            [
                {
                    "chunk_id": "same",
                    "document_id": "doc",
                    "text": "one",
                    "score": 1.0,
                    "rank": 1,
                    "metadata": {},
                    "source_url": None,
                },
                {
                    "chunk_id": "same",
                    "document_id": "doc",
                    "text": "two",
                    "score": 0.5,
                    "rank": 2,
                    "metadata": {},
                    "source_url": None,
                },
            ],
            "duplicate chunks",
        ),
        (
            [
                {
                    "chunk_id": "one",
                    "document_id": "doc",
                    "text": "one",
                    "score": 1.0,
                    "rank": 2,
                    "metadata": {},
                    "source_url": None,
                }
            ],
            "non-contiguous ranks",
        ),
    ],
)
def test_run_api_evaluation_rejects_invalid_rankings(tmp_path, chunks, message):
    response = make_response(chunks=chunks)

    with pytest.raises(ValueError, match=message):
        run_api_evaluation(make_config(tmp_path), [make_label()], client=FakeClient([response]), clock=FakeClock([0.0, 0.1]))


def test_run_api_evaluation_rejects_trace_mismatch(tmp_path):
    payload = response_payload()
    store = TraceStore(tmp_path / "traces.sqlite3").initialize()
    record_matching_trace(store, payload, total_latency_ms=11.0)

    with pytest.raises(ValueError, match="total latency differs"):
        run_api_evaluation(
            make_config(tmp_path),
            [make_label()],
            client=FakeClient([make_response()]),
            trace_store=store,
            clock=FakeClock([0.0, 0.1]),
        )


def test_compare_reference_report_reports_and_can_reject_ranking_drift():
    report = {
        "metrics": reference_report()["metrics"],
        "questions": [
            {
                **reference_report()["questions"][0],
                "retrieved_chunk_ids": ["other", "relevant"],
            }
        ],
    }

    comparison = compare_reference_report(report, reference_report(), require_exact=False)

    assert comparison["ranking_mismatch_count"] == 1
    assert comparison["metrics_match"]
    with pytest.raises(ValueError, match="ranking_mismatches=1"):
        compare_reference_report(report, reference_report())


def test_load_reference_report_and_url_validation(tmp_path, monkeypatch):
    path = tmp_path / "reference.json"
    path.write_text('{"metrics": {}, "questions": []}', encoding="utf-8")
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text('{"metrics": []}', encoding="utf-8")
    monkeypatch.setenv("RAGOPS_API_URL", "http://localhost:9000/")

    assert load_reference_report(path)["questions"] == []
    assert configured_api_url() == "http://localhost:9000"
    with pytest.raises(ValueError, match="http"):
        configured_api_url("localhost:8000")
    with pytest.raises(ValueError, match="invalid structure"):
        load_reference_report(invalid_path)


def test_summarize_component_latencies_counts_only_populated_values():
    questions = [
        {"component_latencies": {field: (1.0 if field in {"embedding_ms", "dense_ms"} else None) for field in ("embedding_ms", "dense_ms", "bm25_ms", "fusion_ms", "reranker_ms", "generation_ms")}},
        {"component_latencies": {field: (3.0 if field == "embedding_ms" else None) for field in ("embedding_ms", "dense_ms", "bm25_ms", "fusion_ms", "reranker_ms", "generation_ms")}},
    ]

    summary = summarize_component_latencies(questions)

    assert summary["embedding_ms"] == {"count": 2, "total": 4.0, "average": 2.0, "minimum": 1.0, "maximum": 3.0}
    assert summary["dense_ms"]["count"] == 1
    assert summary["generation_ms"] is None
