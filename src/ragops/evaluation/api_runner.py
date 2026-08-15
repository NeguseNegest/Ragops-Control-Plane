import json
import math
import os
import time
from pathlib import Path

import httpx

from ragops import __version__
from ragops.api.schemas import QueryResponse
from ragops.evaluation.retrieval_labels import RetrievalLabel
from ragops.evaluation.retrieval_metrics import evaluate_retrieval_metrics
from ragops.evaluation.runner import build_question_metrics, latency_summary

DEFAULT_API_URL = "http://127.0.0.1:8000"
API_ROUTES = {
    "dense_baseline": "dense",
    "hybrid_rrf": "hybrid",
    "hybrid_rrf_cross_encoder": "reranked",
}
COMPONENT_LATENCY_FIELDS = (
    "embedding_ms",
    "dense_ms",
    "bm25_ms",
    "fusion_ms",
    "reranker_ms",
    "generation_ms",
)


def configured_api_url(value=None):
    """Resolve an explicit URL, the API environment override, or the local default."""
    value = value if value is not None else os.getenv("RAGOPS_API_URL", DEFAULT_API_URL)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("API URL must be a non-empty string.")
    value = value.strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        raise ValueError("API URL must use http:// or https://.")
    return value


def load_reference_report(path):
    """Load one prior evaluation report used for exact API/offline parity."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reference evaluation report does not exist: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Reference evaluation report is invalid JSON: {path}: {error}") from error
    if not isinstance(report, dict) or not isinstance(report.get("questions"), list) or not isinstance(report.get("metrics"), dict):
        raise ValueError(f"Reference evaluation report has an invalid structure: {path}")
    return report


def _response_detail(response):
    try:
        payload = response.json()
    except Exception:
        text = getattr(response, "text", "")
        return str(text).strip() or "no response body"
    if isinstance(payload, dict) and payload.get("detail"):
        return str(payload["detail"])
    return str(payload)


def _require_http_success(response, operation):
    status_code = getattr(response, "status_code", None)
    if status_code != 200:
        raise RuntimeError(f"{operation} returned HTTP {status_code}: {_response_detail(response)}")


def _response_headers(response):
    return {str(key).casefold(): str(value) for key, value in getattr(response, "headers", {}).items()}


def validate_health_response(response):
    """Validate the live service identity before sending evaluation traffic."""
    _require_http_success(response, "GET /health")
    try:
        payload = response.json()
    except Exception as error:
        raise ValueError("GET /health did not return JSON.") from error
    if payload != {"status": "ok", "version": __version__}:
        raise ValueError(f"GET /health returned an unexpected service identity: {payload}")
    return payload


def _validate_query_response(response, label, config):
    _require_http_success(response, f"POST /query for {label.question_id}")
    try:
        body = QueryResponse.model_validate(response.json())
    except Exception as error:
        raise ValueError(f"POST /query returned an invalid response for {label.question_id}: {error}") from error

    expected_route = API_ROUTES.get(config.name)
    if expected_route is None:
        raise ValueError(f"Evaluation config {config.name!r} is not exposed by POST /query.")
    if body.query != label.question:
        raise ValueError(f"API response query mismatch for {label.question_id}.")
    if body.config != config.name or body.config_version != config.version or body.route != expected_route:
        raise ValueError(f"API pipeline identity mismatch for {label.question_id}.")
    if len(body.chunks) > config.retriever.top_k:
        raise ValueError(f"API returned more than top_k chunks for {label.question_id}.")

    chunk_ids = [chunk.chunk_id for chunk in body.chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(f"API returned duplicate chunks for {label.question_id}.")
    if [chunk.rank for chunk in body.chunks] != list(range(1, len(body.chunks) + 1)):
        raise ValueError(f"API returned non-contiguous ranks for {label.question_id}.")
    if any(not math.isfinite(chunk.score) for chunk in body.chunks):
        raise ValueError(f"API returned a non-finite score for {label.question_id}.")
    if not math.isfinite(body.latency_ms) or body.latency_ms < 0:
        raise ValueError(f"API returned invalid total latency for {label.question_id}.")

    trace_id = str(body.trace_id)
    if _response_headers(response).get("x-trace-id") != trace_id:
        raise ValueError(f"API trace header/body mismatch for {label.question_id}.")
    if not set(body.used_chunk_ids).issubset(chunk_ids):
        raise ValueError(f"API generation referenced an unknown chunk for {label.question_id}.")
    cited_chunk_ids = {chunk_id for citation in body.citations for chunk_id in citation.chunk_ids}
    if not cited_chunk_ids.issubset(chunk_ids):
        raise ValueError(f"API citation referenced an unknown chunk for {label.question_id}.")
    if body.debug is None:
        raise ValueError(f"API omitted requested debug metadata for {label.question_id}.")
    if body.debug.pipeline_id != f"{config.name}@{config.version}":
        raise ValueError(f"API debug pipeline identity mismatch for {label.question_id}.")
    if body.debug.requested_top_k != config.retriever.top_k or body.debug.returned_chunks != len(body.chunks):
        raise ValueError(f"API debug depth mismatch for {label.question_id}.")
    return body


def _equal_latency(left, right):
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def verify_response_trace(trace_store, body, label, config):
    """Cross-check one HTTP response against its durable SQLite trace and chunks."""
    trace_id = str(body.trace_id)
    trace = trace_store.get_trace(trace_id)
    if trace is None:
        raise ValueError(f"API trace {trace_id} was not found for {label.question_id}.")
    expected = {
        "endpoint": "query",
        "query": label.question,
        "requested_top_k": config.retriever.top_k,
        "pipeline_name": config.name,
        "pipeline_version": str(config.version),
        "status": "success",
        "retrieved_chunk_count": len(body.chunks),
        "answer": body.answer,
    }
    mismatches = [field for field, value in expected.items() if getattr(trace, field) != value]
    if mismatches:
        raise ValueError(f"API trace {trace_id} has mismatched fields for {label.question_id}: {mismatches}")
    if not _equal_latency(trace.total_latency_ms, body.latency_ms):
        raise ValueError(f"API trace {trace_id} total latency differs from its response.")
    for field, value in body.component_latencies.model_dump().items():
        if not _equal_latency(getattr(trace, field), value):
            raise ValueError(f"API trace {trace_id} component {field} differs from its response.")

    stored_chunks = trace_store.list_retrieved_chunks(trace_id)
    response_chunk_ids = [chunk.chunk_id for chunk in body.chunks]
    if [chunk.chunk_id for chunk in stored_chunks] != response_chunk_ids:
        raise ValueError(f"API trace {trace_id} chunk ordering differs from its response.")
    used_chunk_ids = set(body.used_chunk_ids)
    if any(chunk.used_for_generation != (chunk.chunk_id in used_chunk_ids) for chunk in stored_chunks):
        raise ValueError(f"API trace {trace_id} generation-use flags differ from its response.")
    return trace_id


def summarize_component_latencies(question_results):
    """Summarize every populated component without inventing values for unused stages."""
    summaries = {}
    for field in COMPONENT_LATENCY_FIELDS:
        values = [question["component_latencies"][field] for question in question_results if question["component_latencies"][field] is not None]
        summaries[field] = (
            None
            if not values
            else {
                "count": len(values),
                "total": sum(values),
                "average": sum(values) / len(values),
                "minimum": min(values),
                "maximum": max(values),
            }
        )
    return summaries


def compare_reference_report(report, reference_report, require_exact=True):
    """Compare API rankings with a prior offline run over the same labels."""
    reference_questions = reference_report["questions"]
    api_questions = report["questions"]
    reference_ids = [question.get("question_id") for question in reference_questions]
    api_ids = [question["question_id"] for question in api_questions]
    if reference_ids != api_ids:
        raise ValueError("API and reference reports do not contain the same ordered question IDs.")

    mismatches = []
    for api_question, reference_question in zip(api_questions, reference_questions, strict=True):
        question_id = api_question["question_id"]
        for field in ("question", "expected_source", "relevant_chunk_ids"):
            if api_question[field] != reference_question.get(field):
                raise ValueError(f"API and reference report field {field!r} differs for {question_id}.")
        api_ranking = api_question["retrieved_chunk_ids"]
        reference_ranking = reference_question.get("retrieved_chunk_ids")
        if api_ranking != reference_ranking:
            mismatches.append(
                {
                    "question_id": question_id,
                    "api_ranking": api_ranking,
                    "reference_ranking": reference_ranking,
                }
            )

    metrics_match = report["metrics"] == reference_report["metrics"]
    comparison = {
        "reference_run_name": reference_report.get("run_name"),
        "question_count": len(api_questions),
        "exact_ranking_match_count": len(api_questions) - len(mismatches),
        "ranking_mismatch_count": len(mismatches),
        "metrics_match": metrics_match,
        "mismatches": mismatches,
    }
    if require_exact and (mismatches or not metrics_match):
        raise ValueError(
            f"API evaluation differs from the reference report: ranking_mismatches={len(mismatches)}, metrics_match={metrics_match}."
        )
    return comparison


def run_api_evaluation(
    config,
    labels,
    api_url=None,
    client=None,
    trace_store=None,
    reference_report=None,
    timeout_seconds=120.0,
    clock=time.perf_counter,
    progress=None,
    require_exact_reference=True,
):
    """Evaluate retrieval through the live POST /query HTTP and trace boundaries."""
    labels = [label if isinstance(label, RetrievalLabel) else RetrievalLabel.model_validate(label) for label in labels]
    if not labels:
        raise ValueError("At least one retrieval label is required.")
    question_ids = [label.question_id for label in labels]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Retrieval labels must not contain duplicate question IDs.")
    if config.name != "dense_baseline":
        raise ValueError("The Week 5 API evaluator requires the dense_baseline retrieval config.")
    if config.retriever.top_k > 20:
        raise ValueError("API evaluation top_k must not exceed the POST /query maximum of 20.")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise ValueError("API timeout must be a positive finite number.")
    timeout_seconds = float(timeout_seconds)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("API timeout must be a positive finite number.")

    api_url = configured_api_url(api_url)
    owns_client = client is None
    if owns_client:
        client = httpx.Client(base_url=api_url, timeout=timeout_seconds)

    trace_counts_before = trace_store.counts() if trace_store is not None else None
    rankings = {}
    question_results = []
    verified_trace_ids = []
    try:
        health = validate_health_response(client.get("/health"))
        for index, label in enumerate(labels, start=1):
            started_at = clock()
            try:
                response = client.post(
                    "/query",
                    json={
                        "query": label.question,
                        "top_k": config.retriever.top_k,
                        "config": config.name,
                        "debug": True,
                    },
                )
            except Exception as error:
                raise RuntimeError(f"POST /query failed for {label.question_id}: {error}") from error
            client_latency_ms = max(0.0, (clock() - started_at) * 1000)
            body = _validate_query_response(response, label, config)
            if trace_store is not None:
                verified_trace_ids.append(verify_response_trace(trace_store, body, label, config))

            retrieved_chunk_ids = [chunk.chunk_id for chunk in body.chunks]
            retrieved_scores = [chunk.score for chunk in body.chunks]
            rankings[label.question_id] = retrieved_chunk_ids
            question_result = {
                "question_id": label.question_id,
                "question": label.question,
                "expected_source": label.expected_source,
                "relevant_chunk_ids": list(label.relevant_chunk_ids),
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieved_scores": retrieved_scores,
                "latency_ms": body.latency_ms,
                "client_latency_ms": client_latency_ms,
                "component_latencies": body.component_latencies.model_dump(),
                "trace_id": str(body.trace_id),
                "route": body.route,
                "config": body.config,
                "config_version": str(body.config_version),
                "citation_count": len(body.citations),
                "used_chunk_ids": list(body.used_chunk_ids),
                "cost": body.cost.model_dump(mode="json"),
                **build_question_metrics(retrieved_chunk_ids, label.relevant_chunk_ids, config.evaluation.k_values),
            }
            question_results.append(question_result)
            if progress:
                progress(
                    {
                        "index": index,
                        "total": len(labels),
                        "question_id": label.question_id,
                        "latency_ms": body.latency_ms,
                        "client_latency_ms": client_latency_ms,
                        "trace_id": str(body.trace_id),
                    }
                )
    finally:
        if owns_client:
            client.close()

    metrics = evaluate_retrieval_metrics(rankings, labels, k_values=config.evaluation.k_values)
    report = {
        "schema_version": 1,
        "run_name": f"{config.name}_api",
        "run_source": "live_api_evaluation",
        "configuration": config.model_dump(mode="json"),
        "api": {
            "base_url": api_url,
            "endpoint": "/query",
            "service_version": health["version"],
            "route": API_ROUTES[config.name],
            "config": config.name,
            "config_version": str(config.version),
            "top_k": config.retriever.top_k,
            "debug": True,
        },
        "metrics": metrics,
        "latency_ms": latency_summary(question_results),
        "client_latency_ms": latency_summary(
            [{"latency_ms": question["client_latency_ms"]} for question in question_results]
        ),
        "component_latency_ms": summarize_component_latencies(question_results),
        "trace_verification": {
            "enabled": trace_store is not None,
            "verified_trace_count": len(verified_trace_ids),
            "counts_before": trace_counts_before,
            "counts_after": trace_store.counts() if trace_store is not None else None,
        },
        "questions": question_results,
    }
    if reference_report is not None:
        report["reference_comparison"] = compare_reference_report(
            report,
            reference_report,
            require_exact=require_exact_reference,
        )
    else:
        report["reference_comparison"] = None
    return report
