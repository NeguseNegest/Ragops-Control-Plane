"""Compact two-tab Streamlit interface for the completed RAGOps system."""

import json
import math
import os
from pathlib import Path

import requests
import streamlit as st

from ragops.evaluation.failure_analysis import build_failure_analysis, load_failure_analysis_config
from ragops.tracing.store import TraceStore

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TOP_K = 5
REQUEST_TIMEOUT_SECONDS = 120
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINAL_BENCHMARK_PATH = Path("reports/evaluations/final_benchmark.json")
DEFAULT_ROUTED_REPORT_PATH = Path("reports/evaluations/final_routed.json")
DEFAULT_FAILURE_CONFIG_PATH = Path("configs/failure_analysis.yaml")
DEFAULT_TRACE_DB_PATH = Path("data/traces/ragops_traces.sqlite3")
ROUTE_NAMES = frozenset({"FAST", "STANDARD", "CAREFUL", "NO_ANSWER"})


class DashboardDataError(RuntimeError):
    """A checked dashboard artifact is missing or malformed."""


def get_api_url():
    """Return the normalized FastAPI base URL."""
    api_url = os.getenv("RAGOPS_API_URL", DEFAULT_API_URL).strip()
    return (api_url or DEFAULT_API_URL).rstrip("/")


def configured_artifact_path(environment_name, default_path):
    """Resolve a dashboard artifact override against the configured project root."""
    configured_root = os.getenv("RAGOPS_PROJECT_ROOT")
    project_root = Path(configured_root.strip()).resolve() if configured_root and configured_root.strip() else PROJECT_ROOT
    configured_path = os.getenv(environment_name)
    path = Path(configured_path.strip()) if configured_path and configured_path.strip() else Path(default_path)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _post_json(endpoint, payload, api_url):
    try:
        response = requests.post(endpoint, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as error:
        raise RuntimeError(f"Could not reach the API at {api_url}. Make sure the FastAPI server is running.") from error

    try:
        result = response.json()
    except ValueError as error:
        message = "The API returned an invalid response." if response.ok else f"The API request failed with status {response.status_code}."
        raise RuntimeError(message) from error

    if not response.ok:
        detail = result.get("detail") if isinstance(result, dict) else None
        raise RuntimeError(str(detail or f"The API request failed with status {response.status_code}."))
    if not isinstance(result, dict):
        raise RuntimeError("The API returned an unexpected response.")
    return result


def route_api(query, api_url):
    """Run the decision-only router and return its evidence."""
    return _post_json(f"{api_url}/route", {"query": query}, api_url)


def query_api(query, top_k, api_url, config="dense_baseline"):
    """Execute one explicit registered pipeline with debug evidence enabled."""
    payload = {"query": query, "top_k": top_k, "config": config, "debug": True}
    return _post_json(f"{api_url}/query", payload, api_url)


def _required_non_negative_number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"The API returned an invalid {label}.") from error
    if not math.isfinite(number) or number < 0:
        raise RuntimeError(f"The API returned an invalid {label}.")
    return number


def execute_routed_query(query, top_k, api_url):
    """Route first, then execute the selected pipeline or deterministic refusal.

    Automatic route dispatch is not implemented inside ``POST /query``. This
    dashboard deliberately makes the two HTTP calls explicit and reports their
    summed server-side latency as its route-plus-query total.
    """
    route_result = route_api(query, api_url)
    decision = route_result.get("decision")
    if not isinstance(decision, dict):
        raise RuntimeError("The routing API returned no decision.")
    selected_route = decision.get("route")
    if selected_route not in ROUTE_NAMES:
        raise RuntimeError("The routing API returned an unsupported route.")
    reason = decision.get("reason")
    reason_code = decision.get("reason_code")
    if not isinstance(reason, str) or not reason.strip() or not isinstance(reason_code, str):
        raise RuntimeError("The routing API returned an incomplete reason.")

    probe_timings = route_result.get("probe_timings") or {}
    probe_latency_ms = _required_non_negative_number(probe_timings.get("total_ms"), "router probe latency")
    routing = {
        "route": selected_route,
        "reason": reason,
        "reason_code": reason_code,
        "pipeline_config": decision.get("pipeline_config"),
        "maximum_top_k": decision.get("maximum_top_k"),
        "router_id": decision.get("router_id"),
        "router_status": decision.get("router_status"),
        "matched_reason_codes": decision.get("matched_reason_codes") or [],
        "probe_latency_ms": probe_latency_ms,
        "features": route_result.get("features") or {},
        "probe_chunks": route_result.get("probe_chunks") or [],
    }

    if selected_route == "NO_ANSWER":
        refusal = route_result.get("refusal")
        if not isinstance(refusal, dict) or not isinstance(refusal.get("answer"), str) or not refusal["answer"].strip():
            raise RuntimeError("The NO_ANSWER route returned no deterministic refusal.")
        return {
            "query": query,
            "answer": refusal["answer"],
            "citations": [],
            "chunks": [],
            "trace_id": None,
            "route": None,
            "config": None,
            "latency_ms": 0.0,
            "total_latency_ms": probe_latency_ms,
            "component_latencies": {
                "embedding_ms": probe_timings.get("embedding_ms"),
                "dense_ms": probe_timings.get("dense_ms"),
            },
            "cost": None,
            "debug": None,
            "routing": routing,
            "execution_note": "The router selected a deterministic refusal, so no answer pipeline ran and no query trace was created.",
        }

    pipeline_config = decision.get("pipeline_config")
    maximum_top_k = decision.get("maximum_top_k")
    if not isinstance(pipeline_config, str) or isinstance(maximum_top_k, bool) or not isinstance(maximum_top_k, int) or maximum_top_k < 1:
        raise RuntimeError("The routing API returned an invalid execution intent.")
    effective_top_k = min(top_k, maximum_top_k)
    result = dict(query_api(query, effective_top_k, api_url, config=pipeline_config))
    if result.get("config") != pipeline_config:
        raise RuntimeError("The query API executed a different pipeline than the router selected.")
    result["routing"] = routing
    result["requested_top_k"] = top_k
    result["effective_top_k"] = effective_top_k
    query_latency_ms = _required_non_negative_number(result.get("latency_ms"), "query latency")
    result["total_latency_ms"] = probe_latency_ms + query_latency_ms
    result["execution_note"] = (
        "The dashboard called /route and then /query with the selected explicit config; "
        "the current backend does not dispatch non-refusal routes inside /query."
    )
    return result


def _load_json_object(path, label):
    path = Path(path)
    if not path.is_file():
        raise DashboardDataError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DashboardDataError(f"Unable to read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise DashboardDataError(f"{label} must contain a JSON object.")
    return payload


def load_final_benchmark(path):
    """Load the completed five-pipeline Day 47 benchmark."""
    benchmark = _load_json_object(path, "Final benchmark")
    pipelines = benchmark.get("pipelines")
    if benchmark.get("benchmark_id") != "final_benchmark@1.0.0" or benchmark.get("status") != "evaluated":
        raise DashboardDataError("Final benchmark is not the completed final_benchmark@1.0.0 artifact.")
    if not isinstance(pipelines, list) or len(pipelines) != 5:
        raise DashboardDataError("Final benchmark must contain exactly five pipeline summaries.")
    expected_pipelines = {"dense", "bm25", "hybrid", "reranked", "routed"}
    observed_pipelines = {pipeline.get("pipeline") for pipeline in pipelines if isinstance(pipeline, dict)}
    if observed_pipelines != expected_pipelines:
        raise DashboardDataError("Final benchmark does not contain the expected five pipeline identities.")
    numeric_fields = (
        "recall_at_5",
        "mrr_at_5",
        "faithfulness",
        "answer_relevance",
        "p50_latency_ms",
        "p95_latency_ms",
        "estimated_cost_per_query_usd",
    )
    for pipeline in pipelines:
        for field in numeric_fields:
            value = pipeline.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise DashboardDataError(f"Final benchmark pipeline {pipeline['pipeline']} has invalid {field}.")
        if pipeline["p95_latency_ms"] < pipeline["p50_latency_ms"]:
            raise DashboardDataError(f"Final benchmark pipeline {pipeline['pipeline']} has p95 below p50.")
    return benchmark


def benchmark_rows(benchmark):
    """Return compact comparable metrics for the engineering table and chart."""
    rows = []
    for pipeline in benchmark["pipelines"]:
        refusal = pipeline.get("refusal_correctness") or {}
        rows.append(
            {
                "pipeline": pipeline.get("label") or pipeline.get("pipeline"),
                "recall_at_5": pipeline.get("recall_at_5"),
                "mrr_at_5": pipeline.get("mrr_at_5"),
                "faithfulness_5": pipeline.get("faithfulness"),
                "answer_relevance_5": pipeline.get("answer_relevance"),
                "refusal_accuracy": refusal.get("accuracy"),
                "p50_latency_ms": pipeline.get("p50_latency_ms"),
                "p95_latency_ms": pipeline.get("p95_latency_ms"),
                "estimated_cost_usd": pipeline.get("estimated_cost_per_query_usd"),
            }
        )
    return rows


def load_routed_summary(path):
    """Load routed latency and supported-query route distribution."""
    report = _load_json_object(path, "Routed evaluation report")
    counts = report.get("route_counts_supported")
    latency = report.get("latency_ms")
    if (
        not isinstance(counts, dict)
        or set(counts) != ROUTE_NAMES
        or any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts.values())
    ):
        raise DashboardDataError("Routed evaluation report has an invalid supported route distribution.")
    if not isinstance(latency, dict) or not isinstance(latency.get("count"), int):
        raise DashboardDataError("Routed evaluation report has no valid latency summary.")
    total = sum(counts.values())
    if total <= 0 or total != latency["count"]:
        raise DashboardDataError("Routed route counts do not match the measured question count.")
    for percentile in ("p50", "p95"):
        value = latency.get(percentile)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise DashboardDataError(f"Routed evaluation report has invalid {percentile} latency.")
    if latency["p95"] < latency["p50"]:
        raise DashboardDataError("Routed evaluation report has p95 below p50.")
    return {
        "router_id": report.get("router_id"),
        "counts": counts,
        "total": total,
        "no_answer_rate": counts["NO_ANSWER"] / total,
        "p50_latency_ms": latency.get("p50"),
        "p95_latency_ms": latency.get("p95"),
    }


def load_failure_examples(config_path):
    """Re-verify and return the complete curated Day 48 failure inventory."""
    try:
        config = load_failure_analysis_config(config_path, project_root=PROJECT_ROOT)
        analysis = build_failure_analysis(config)
    except Exception as error:
        raise DashboardDataError(f"Unable to verify Day 48 failure examples: {error}") from error
    return analysis["cases"]


def load_recent_traces(path, limit=20):
    """Read recent traces without creating or migrating a database."""
    path = Path(path)
    if not path.is_file():
        return []
    try:
        store = TraceStore(path)
        store.validate_schema()
        traces = store.list_traces(limit=limit)
    except Exception as error:
        raise DashboardDataError(f"Unable to read recent traces: {error}") from error
    return [trace.model_dump(mode="json") for trace in traces]


def _score_text(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.4f}" if math.isfinite(number) else "n/a"


def chunk_score_labels(chunk):
    """Separate retrieval and cross-encoder scores when reranking was used."""
    metadata = chunk.get("metadata") or {}
    reranker = metadata.get("_reranker") if isinstance(metadata, dict) else None
    if isinstance(reranker, dict):
        return {
            "retrieval": _score_text(reranker.get("candidate_score")),
            "reranker": _score_text(chunk.get("score")),
            "candidate_rank": reranker.get("candidate_rank"),
        }
    return {"retrieval": _score_text(chunk.get("score")), "reranker": None, "candidate_rank": None}


def render_citations(citations):
    st.subheader("Citations")
    if not citations:
        st.caption("No citations were returned.")
        return
    for citation in citations:
        citation_id = citation.get("citation_id") or "[?]"
        title = citation.get("title") or citation.get("document_id") or "Unknown source"
        url = citation.get("url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            st.markdown(f"{citation_id} [{title}]({url})")
        elif url:
            st.write(f"{citation_id} {title} — {url}")
        else:
            st.write(f"{citation_id} {title}")


def render_chunks(chunks):
    st.subheader("Retrieved chunks")
    if not chunks:
        st.caption("No answer-pipeline chunks were returned.")
        return
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        scores = chunk_score_labels(chunk)
        rank = chunk.get("rank", "?")
        source = chunk.get("source_url") or metadata.get("relative_path") or metadata.get("source_path") or "Unknown source"
        score_summary = f"retrieval {scores['retrieval']}"
        if scores["reranker"] is not None:
            score_summary += f" · reranker {scores['reranker']}"
        with st.expander(f"#{rank} · {score_summary} · {source}"):
            st.write(chunk.get("text") or "No chunk text was returned.")
            details = []
            if scores["candidate_rank"] is not None:
                details.append(f"Pre-rerank candidate rank: {scores['candidate_rank']}")
            fusion = metadata.get("_fusion") if isinstance(metadata, dict) else None
            if isinstance(fusion, dict):
                source_ranks = ", ".join(
                    f"{name} #{values.get('rank')}" for name, values in (fusion.get("sources") or {}).items()
                )
                if source_ranks:
                    details.append(f"RRF inputs: {source_ranks}")
            if metadata.get("heading"):
                details.append(f"Heading: {metadata['heading']}")
            if metadata.get("relative_path"):
                details.append(f"Path: {metadata['relative_path']}")
            if chunk.get("chunk_id"):
                details.append(f"Chunk ID: {chunk['chunk_id']}")
            if details:
                st.caption(" · ".join(details))


def _latency_text(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:,.1f} ms" if math.isfinite(number) and number >= 0 else "n/a"


def _cost_text(cost):
    if not isinstance(cost, dict):
        return "not incurred"
    amount = cost.get("amount_usd")
    if amount is None:
        return "unavailable"
    try:
        number = float(amount)
    except (TypeError, ValueError):
        return "unavailable"
    return f"${number:.8f}" if math.isfinite(number) and number >= 0 else "unavailable"


def render_query_result(result):
    routing = result.get("routing") or {}
    st.markdown("**User query**")
    st.write(result.get("query") or "Unknown query")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected route", routing.get("route") or "n/a")
    metric_columns[1].metric("Total latency", _latency_text(result.get("total_latency_ms")))
    metric_columns[2].metric("Estimated cost", _cost_text(result.get("cost")))
    metric_columns[3].metric("Trace ID", str(result.get("trace_id") or "not created"))
    st.info(f"Router reason: {routing.get('reason', 'unavailable')} ({routing.get('reason_code', 'n/a')})")

    st.subheader("Generated answer")
    st.write(result.get("answer") or "No answer was returned.")
    st.caption(result.get("execution_note") or "")

    debug = result.get("debug") or {}
    if result.get("config"):
        st.caption(
            f"Executed config: {result['config']}@{result.get('config_version', 'unknown')} · "
            f"pipeline route: {result.get('route', 'unknown')} · effective top-k: {result.get('effective_top_k', 'unknown')}"
        )
    if isinstance(result.get("cost"), dict):
        cost = result["cost"]
        st.caption(
            f"Cost status: {cost.get('status', 'unknown')} · provider/model: "
            f"{cost.get('provider', 'unknown')}/{cost.get('model') or 'n/a'} · token source: {cost.get('token_source', 'unknown')}"
        )
    if debug:
        with st.expander("Pipeline and component diagnostics"):
            st.json({"debug": debug, "component_latencies": result.get("component_latencies") or {}, "routing": routing})

    render_citations(result.get("citations") or [])
    render_chunks(result.get("chunks") or [])

    probe_chunks = routing.get("probe_chunks") or []
    if probe_chunks:
        with st.expander("Router probe evidence"):
            st.dataframe(probe_chunks, width="stretch", hide_index=True)


def render_query_playground():
    st.header("Query Playground")
    st.caption("The router chooses FAST, STANDARD, CAREFUL, or a deterministic NO_ANSWER refusal before execution.")
    with st.form("query_form"):
        query = st.text_area("Question", placeholder="How do I create a FastAPI app?")
        top_k = st.number_input("Maximum returned chunks", min_value=1, max_value=20, value=DEFAULT_TOP_K, step=1)
        submitted = st.form_submit_button("Route and ask")
    if not submitted:
        return
    query = query.strip()
    if not query:
        st.warning("Enter a question before submitting.")
        return
    try:
        with st.spinner("Routing, retrieving, and generating..."):
            result = execute_routed_query(query, int(top_k), get_api_url())
    except RuntimeError as error:
        st.error(str(error))
        return
    render_query_result(result)


def _format_benchmark_rows(rows):
    formatted = []
    for row in rows:
        formatted.append(
            {
                "Pipeline": row["pipeline"],
                "Recall@5": row["recall_at_5"],
                "MRR@5": row["mrr_at_5"],
                "Faithfulness / 5": row["faithfulness_5"],
                "Answer relevance / 5": row["answer_relevance_5"],
                "Refusal accuracy": row["refusal_accuracy"],
                "p50 ms": row["p50_latency_ms"],
                "p95 ms": row["p95_latency_ms"],
                "Avg estimated cost USD": row["estimated_cost_usd"],
            }
        )
    return formatted


def render_benchmark(benchmark):
    rows = benchmark_rows(benchmark)
    st.subheader("Final benchmark")
    st.caption("Day 47: 50 paired retrieval questions and 10 cross-provider answer judgments per pipeline.")
    st.dataframe(
        _format_benchmark_rows(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Recall@5": st.column_config.NumberColumn(format="%.3f"),
            "MRR@5": st.column_config.NumberColumn(format="%.3f"),
            "Faithfulness / 5": st.column_config.NumberColumn(format="%.1f"),
            "Answer relevance / 5": st.column_config.NumberColumn(format="%.1f"),
            "Refusal accuracy": st.column_config.NumberColumn(format="%.1%%"),
            "p50 ms": st.column_config.NumberColumn(format="%.1f"),
            "p95 ms": st.column_config.NumberColumn(format="%.1f"),
            "Avg estimated cost USD": st.column_config.NumberColumn(format="$%.8f"),
        },
    )

    st.subheader("Quality vs. latency")
    st.caption("Recall@5 is plotted against p95 retrieval/routed latency; lower latency and higher recall are better.")
    st.scatter_chart(rows, x="p95_latency_ms", y="recall_at_5", color="pipeline", size=120)
    return rows


def render_route_summary(summary, benchmark_rows_data):
    routed = next(row for row in benchmark_rows_data if row["pipeline"] == "Routed")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Routed p50", _latency_text(summary["p50_latency_ms"]))
    metric_columns[1].metric("Routed p95", _latency_text(summary["p95_latency_ms"]))
    metric_columns[2].metric("Avg estimated cost", f"${routed['estimated_cost_usd']:.8f}")
    metric_columns[3].metric("Supported NO_ANSWER rate", f"{summary['no_answer_rate']:.1%}")

    st.subheader("Route distribution")
    st.caption(f"{summary['router_id']} decisions over {summary['total']} supported benchmark questions.")
    route_rows = [
        {"route": route, "count": summary["counts"][route], "share": summary["counts"][route] / summary["total"]}
        for route in ("FAST", "STANDARD", "CAREFUL", "NO_ANSWER")
    ]
    st.bar_chart(route_rows, x="route", y="count")
    st.dataframe(route_rows, width="stretch", hide_index=True)


def _trace_table_rows(traces):
    rows = []
    for trace in traces:
        rows.append(
            {
                "Created (UTC)": trace.get("created_at"),
                "Endpoint": trace.get("endpoint"),
                "Status": trace.get("status"),
                "Pipeline": f"{trace.get('pipeline_name')}@{trace.get('pipeline_version')}",
                "Query": trace.get("query"),
                "Latency ms": trace.get("total_latency_ms"),
                "Cost USD": trace.get("cost_amount_usd"),
                "Trace ID": trace.get("trace_id"),
            }
        )
    return rows


def render_recent_traces(path):
    st.subheader("Recent traces")
    try:
        traces = load_recent_traces(path)
    except DashboardDataError as error:
        st.warning(str(error))
        return
    if not traces:
        st.caption(f"No traces are available in {path}. Successful or failed /query requests will appear here.")
        return
    st.dataframe(_trace_table_rows(traces), width="stretch", hide_index=True)


def _failure_label(case):
    return f"{case['id']} · {case['category'].replace('_', ' ')} · {case['question_id']}"


def render_failures(config_path):
    st.subheader("Selected failure examples")
    try:
        failures = load_failure_examples(config_path)
    except DashboardDataError as error:
        st.warning(str(error))
        return
    st.caption(f"Day 48 verified {len(failures)} measured failures; choose one to inspect its evidence and proposed fix.")
    labels = {case["id"]: _failure_label(case) for case in failures}
    selected_id = st.selectbox("Failure example", list(labels), format_func=labels.__getitem__)
    selected = next(case for case in failures if case["id"] == selected_id)
    st.markdown(f"**Query:** {selected['query']}")
    detail_columns = st.columns(3)
    detail_columns[0].metric("Severity", selected["severity"])
    detail_columns[1].metric("Component", selected["affected_component"])
    detail_columns[2].metric("Regression case", "yes" if selected["regression_test"] else "analysis only")
    st.write(f"**Expected:** {selected['expected_behavior']}")
    st.write(f"**Observed:** {selected['actual_behavior']}")
    st.write(f"**Diagnosis:** {selected['root_cause']}")
    st.write(f"**Proposed fix:** {selected['proposed_fix']}")
    with st.expander("Verified evidence"):
        st.json(selected["evidence"])


def render_engineering():
    st.header("Engineering")
    st.caption("Frozen evaluation evidence is kept separate from live local traces.")
    benchmark_path = configured_artifact_path("RAGOPS_FINAL_BENCHMARK_PATH", DEFAULT_FINAL_BENCHMARK_PATH)
    routed_path = configured_artifact_path("RAGOPS_ROUTED_REPORT_PATH", DEFAULT_ROUTED_REPORT_PATH)
    failure_config_path = configured_artifact_path("RAGOPS_FAILURE_CONFIG_PATH", DEFAULT_FAILURE_CONFIG_PATH)
    trace_path = configured_artifact_path("RAGOPS_TRACE_DB_PATH", DEFAULT_TRACE_DB_PATH)
    try:
        benchmark = load_final_benchmark(benchmark_path)
        summary = load_routed_summary(routed_path)
    except DashboardDataError as error:
        st.error(str(error))
    else:
        rows = render_benchmark(benchmark)
        render_route_summary(summary, rows)
    render_recent_traces(trace_path)
    render_failures(failure_config_path)


def main():
    """Render exactly two primary tabs with no custom state-management layer."""
    st.set_page_config(page_title="RAGOps Control Plane", layout="wide")
    st.title("RAGOps Control Plane")
    st.caption("Explainable routing, grounded answers, measured tradeoffs, and failure evidence in one compact view.")
    query_tab, engineering_tab = st.tabs(["Query Playground", "Engineering"])
    with query_tab:
        render_query_playground()
    with engineering_tab:
        render_engineering()


if __name__ == "__main__":
    main()
