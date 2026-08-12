import json
import time
from collections import defaultdict
from pathlib import Path

from ragops.evaluation.bm25_runner import classify_query_type
from ragops.evaluation.retrieval_labels import RetrievalLabel
from ragops.evaluation.retrieval_metrics import evaluate_retrieval_metrics
from ragops.evaluation.runner import atomic_write_text, build_question_metrics, chunk_id_and_score, close_client, latency_summary
from ragops.indexing.qdrant import create_qdrant_client
from ragops.retrieval.bm25 import BM25Index, load_bm25_index, validate_bm25_index
from ragops.retrieval.hybrid import FUSION_METADATA_KEY, configured_qdrant_url, retrieve_hybrid_config

HYBRID_COMPARISON_SCHEMA_VERSION = 1
RETRIEVER_NAMES = ("dense", "bm25", "hybrid")
RETRIEVER_LABELS = {"dense": "Dense", "bm25": "BM25", "hybrid": "RRF hybrid"}


def require_hybrid_evaluation_settings(config):
    """Reject a retrieval-only hybrid config before evaluation starts."""
    if config.evaluation is None or config.output is None:
        raise ValueError("Hybrid config must include evaluation and output settings for Day 25.")
    return config


def component_latency_summary(question_results):
    """Aggregate per-stage hybrid latency recorded for every question."""
    summary = {}
    for component in ("dense_ms", "bm25_ms", "fusion_ms"):
        values = [question["component_latency_ms"][component] for question in question_results]
        summary[component.removesuffix("_ms")] = {
            "total": sum(values),
            "average": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return summary


def _fusion_sources(result, question_id):
    metadata = getattr(result, "metadata", None)
    fusion = metadata.get(FUSION_METADATA_KEY) if isinstance(metadata, dict) else None
    if not isinstance(fusion, dict) or fusion.get("method") != "rrf":
        raise ValueError(f"Hybrid result {result.chunk_id} for question {question_id} is missing RRF provenance.")
    sources = fusion.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError(f"Hybrid result {result.chunk_id} for question {question_id} has no contributing rankings.")
    return sources


def run_hybrid_evaluation(
    config,
    labels,
    client,
    index,
    retriever=retrieve_hybrid_config,
    clock=time.perf_counter,
    progress=None,
    dense_index=None,
):
    """Run RRF hybrid retrieval for every verified label and compute metrics."""
    require_hybrid_evaluation_settings(config)
    if not isinstance(index, BM25Index):
        raise ValueError("index must be a loaded BM25Index.")

    labels = [label if isinstance(label, RetrievalLabel) else RetrievalLabel.model_validate(label) for label in labels]
    if not labels:
        raise ValueError("At least one retrieval label is required.")
    question_ids = [label.question_id for label in labels]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Retrieval labels must not contain duplicate question IDs.")

    rankings = {}
    question_results = []
    for position, label in enumerate(labels, start=1):
        component_timings = {}
        started_at = clock()
        try:
            retrieved_chunks = list(
                retriever(
                    query=label.question,
                    config=config,
                    client=client,
                    index=index,
                    clock=clock,
                    timings=component_timings,
                )
            )
        except Exception as error:
            raise RuntimeError(f"Hybrid retrieval failed for question {label.question_id}: {error}") from error
        latency_ms = max(0.0, (clock() - started_at) * 1000)

        if len(retrieved_chunks) > config.fusion.top_k:
            raise ValueError(f"Hybrid retriever returned more than top_k results for question {label.question_id}.")
        missing_timings = {"dense_ms", "bm25_ms", "fusion_ms"} - set(component_timings)
        if missing_timings:
            raise ValueError(f"Hybrid retriever did not record component timings for question {label.question_id}: {sorted(missing_timings)}")

        retrieved_chunk_ids = []
        retrieved_scores = []
        retrieved_fusion_sources = []
        for expected_rank, retrieved_chunk in enumerate(retrieved_chunks, start=1):
            chunk_id, score = chunk_id_and_score(retrieved_chunk)
            if chunk_id in retrieved_chunk_ids:
                raise ValueError(f"Hybrid retriever returned duplicate chunk ID {chunk_id} for question {label.question_id}.")
            if getattr(retrieved_chunk, "rank", expected_rank) != expected_rank:
                raise ValueError(f"Hybrid result {chunk_id} has a non-contiguous rank for question {label.question_id}.")
            retrieved_chunk_ids.append(chunk_id)
            retrieved_scores.append(score)
            retrieved_fusion_sources.append(_fusion_sources(retrieved_chunk, label.question_id))

        normalized_timings = {name: float(component_timings[name]) for name in ("dense_ms", "bm25_ms", "fusion_ms")}
        if any(value < 0 or value == float("inf") or value != value for value in normalized_timings.values()):
            raise ValueError(f"Hybrid retriever recorded invalid component latency for question {label.question_id}.")

        rankings[label.question_id] = retrieved_chunk_ids
        question_result = {
            "question_id": label.question_id,
            "question": label.question,
            "expected_source": label.expected_source,
            "relevant_chunk_ids": list(label.relevant_chunk_ids),
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_fusion_sources": retrieved_fusion_sources,
            "latency_ms": latency_ms,
            "component_latency_ms": normalized_timings,
            **build_question_metrics(retrieved_chunk_ids, label.relevant_chunk_ids, config.evaluation.k_values),
        }
        question_results.append(question_result)
        if progress:
            progress({"index": position, "total": len(labels), "question_id": label.question_id, "latency_ms": latency_ms})

    payload = index.payload
    metrics = evaluate_retrieval_metrics(rankings, labels, k_values=config.evaluation.k_values)
    return {
        "schema_version": 1,
        "run_name": config.name,
        "configuration": config.model_dump(mode="json"),
        "dense_index": dense_index or {"collection_name": config.dense.collection_name, "points_count": None},
        "bm25_index": {
            "schema_version": payload.schema_version,
            "tokenizer": payload.tokenizer,
            "parameters": payload.parameters.model_dump(mode="json"),
            "source_sha256": payload.source_sha256,
            "source_record_count": payload.source_record_count,
            "skipped_document_count": payload.skipped_document_count,
            "document_count": payload.document_count,
        },
        "metrics": metrics,
        "latency_ms": latency_summary(question_results),
        "component_latency_ms": component_latency_summary(question_results),
        "questions": question_results,
    }


def _dense_index_metadata(client, collection_name):
    metadata = {"collection_name": collection_name, "points_count": None}
    get_collection = getattr(client, "get_collection", None)
    if get_collection:
        collection = get_collection(collection_name=collection_name)
        points_count = getattr(collection, "points_count", None)
        if points_count is not None:
            metadata["points_count"] = int(points_count)
    return metadata


def evaluate_hybrid_config(
    config,
    labels,
    client_factory=create_qdrant_client,
    index_loader=load_bm25_index,
    retriever=None,
    clock=time.perf_counter,
    progress=None,
):
    """Validate both indexes, run the hybrid benchmark, and close Qdrant."""
    require_hybrid_evaluation_settings(config)
    index = index_loader(config.bm25.index_path)
    payload = validate_bm25_index(index, config.bm25_validation_config())
    client = client_factory(configured_qdrant_url(config))
    try:
        if not client.collection_exists(collection_name=config.dense.collection_name):
            raise RuntimeError(f"Qdrant collection does not exist: {config.dense.collection_name}")
        dense_index = _dense_index_metadata(client, config.dense.collection_name)
        if dense_index["points_count"] is not None and dense_index["points_count"] != payload.source_record_count:
            raise RuntimeError(
                f"Dense collection contains {dense_index['points_count']} points but the shared chunk artifact contains {payload.source_record_count} records."
            )
        if retriever is None:
            from ragops.retrieval.factory import build_retriever

            configured_retriever = build_retriever(config, client=client, index=index, clock=clock)

            def retriever(query, timings, **kwargs):
                return configured_retriever.retrieve(query, timings=timings)

        return run_hybrid_evaluation(
            config,
            labels,
            client,
            index,
            retriever=retriever,
            clock=clock,
            progress=progress,
            dense_index=dense_index,
        )
    finally:
        close_client(client)


def _questions_by_id(report, retriever_name):
    questions = report.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{retriever_name} evaluation report has no question results.")
    by_id = {}
    for question in questions:
        question_id = question.get("question_id") if isinstance(question, dict) else None
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"{retriever_name} evaluation contains a result without a valid question ID.")
        if question_id in by_id:
            raise ValueError(f"{retriever_name} evaluation contains duplicate question ID {question_id}.")
        by_id[question_id] = question
    if report.get("metrics", {}).get("question_count") != len(by_id):
        raise ValueError(f"{retriever_name} aggregate question count does not match its results.")
    return by_id


def _retriever_type(report, name):
    configuration = report.get("configuration", {})
    if name == "hybrid":
        return configuration.get("fusion", {}).get("type")
    return configuration.get("retriever", {}).get("type")


def _first_relevant_rank(question):
    relevant = set(question.get("relevant_chunk_ids") or [])
    for rank, chunk_id in enumerate(question.get("retrieved_chunk_ids") or [], start=1):
        if chunk_id in relevant:
            return rank
    return None


def _reciprocal_rank(rank):
    return 0.0 if rank is None else 1.0 / rank


def _paired_outcome(candidate_rr, baseline_rr):
    if candidate_rr > baseline_rr:
        return "hybrid"
    if baseline_rr > candidate_rr:
        return "baseline"
    return "tie"


def _paired_summary(questions, baseline_name):
    outcomes = [question[f"hybrid_vs_{baseline_name}"] for question in questions]
    return {
        "hybrid_wins": outcomes.count("hybrid"),
        f"{baseline_name}_wins": outcomes.count("baseline"),
        "ties": outcomes.count("tie"),
        "hybrid_recovers_baseline_miss": sum(
            question[f"{baseline_name}_rank"] is None and question["hybrid_rank"] is not None for question in questions
        ),
        "hybrid_loses_baseline_hit": sum(
            question[f"{baseline_name}_rank"] is not None and question["hybrid_rank"] is None for question in questions
        ),
    }


def _metric_comparison(reports, k_values):
    metrics = {
        "mrr": {name: float(report["metrics"]["mrr"]) for name, report in reports.items()},
    }
    metrics["mrr"].update(
        {
            "hybrid_delta_dense": metrics["mrr"]["hybrid"] - metrics["mrr"]["dense"],
            "hybrid_delta_bm25": metrics["mrr"]["hybrid"] - metrics["mrr"]["bm25"],
        }
    )
    for metric_name in ("recall_at_k", "hit_rate_at_k", "ndcg_at_k"):
        metrics[metric_name] = {}
        for k in k_values:
            key = str(k)
            values = {name: float(report["metrics"][metric_name][key]) for name, report in reports.items()}
            values.update(
                {
                    "hybrid_delta_dense": values["hybrid"] - values["dense"],
                    "hybrid_delta_bm25": values["hybrid"] - values["bm25"],
                }
            )
            metrics[metric_name][key] = values
    return metrics


def _warm_latency(report):
    latencies = [float(question["latency_ms"]) for question in report["questions"]][1:]
    if not latencies:
        return None
    return {
        "question_count": len(latencies),
        "total": sum(latencies),
        "average": sum(latencies) / len(latencies),
        "minimum": min(latencies),
        "maximum": max(latencies),
    }


def _relevance_group_summary(questions):
    groups = defaultdict(list)
    for question in questions:
        relevant_ids = tuple(sorted(question["relevant_chunk_ids"]))
        groups[relevant_ids].append(question)

    group_rows = []
    for relevant_ids, grouped_questions in groups.items():
        mrr = {
            name: sum(question[f"{name}_reciprocal_rank"] for question in grouped_questions) / len(grouped_questions)
            for name in RETRIEVER_NAMES
        }
        group_rows.append(
            {
                "relevant_chunk_ids": list(relevant_ids),
                "question_count": len(grouped_questions),
                **{f"{name}_mrr": mrr[name] for name in RETRIEVER_NAMES},
                "hybrid_delta_dense": mrr["hybrid"] - mrr["dense"],
                "hybrid_delta_bm25": mrr["hybrid"] - mrr["bm25"],
            }
        )

    macro_mrr = {
        name: sum(group[f"{name}_mrr"] for group in group_rows) / len(group_rows)
        for name in RETRIEVER_NAMES
    }
    return {
        "group_count": len(group_rows),
        "macro_mrr": {
            **macro_mrr,
            "hybrid_delta_dense": macro_mrr["hybrid"] - macro_mrr["dense"],
            "hybrid_delta_bm25": macro_mrr["hybrid"] - macro_mrr["bm25"],
        },
        "hybrid_vs_bm25": {
            "hybrid_wins": sum(group["hybrid_delta_bm25"] > 0 for group in group_rows),
            "bm25_wins": sum(group["hybrid_delta_bm25"] < 0 for group in group_rows),
            "ties": sum(group["hybrid_delta_bm25"] == 0 for group in group_rows),
        },
        "groups": sorted(group_rows, key=lambda group: group["relevant_chunk_ids"]),
    }


def _relevant_fusion_sources(question):
    relevant = set(question.get("relevant_chunk_ids") or [])
    sources = {}
    for chunk_id, fusion_sources in zip(
        question.get("retrieved_chunk_ids") or [],
        question.get("retrieved_fusion_sources") or [],
        strict=True,
    ):
        if chunk_id in relevant:
            sources[chunk_id] = fusion_sources
    return sources


def compare_hybrid_reports(dense_report, bm25_report, hybrid_report, report_paths=None):
    """Build a strict paired dense/BM25/hybrid benchmark comparison."""
    if _retriever_type(dense_report, "dense") != "dense":
        raise ValueError("Dense baseline report must use dense retrieval.")
    if _retriever_type(bm25_report, "bm25") != "bm25":
        raise ValueError("BM25 baseline report must use BM25 retrieval.")
    if _retriever_type(hybrid_report, "hybrid") != "rrf":
        raise ValueError("Hybrid report must use RRF fusion.")

    dense_config = dense_report["configuration"]["retriever"]
    bm25_config = bm25_report["configuration"]["retriever"]
    hybrid_config = hybrid_report["configuration"]
    for field_name in ("collection_name", "embedding_model"):
        if dense_config.get(field_name) != hybrid_config.get("dense", {}).get(field_name):
            raise ValueError(f"Hybrid and dense baseline configuration differ for {field_name}.")
    for field_name in ("tokenizer", "k1", "b", "epsilon"):
        if bm25_config.get(field_name) != hybrid_config.get("bm25", {}).get(field_name):
            raise ValueError(f"Hybrid and BM25 baseline configuration differ for {field_name}.")

    reports = {"dense": dense_report, "bm25": bm25_report, "hybrid": hybrid_report}
    questions_by_run = {name: _questions_by_id(report, name) for name, report in reports.items()}
    expected_ids = set(questions_by_run["dense"])
    for name in ("bm25", "hybrid"):
        if set(questions_by_run[name]) != expected_ids:
            raise ValueError(f"{name} question IDs differ from the dense baseline.")

    k_values = list(dense_report["metrics"].get("k_values") or [])
    if not k_values or any(list(report["metrics"].get("k_values") or []) != k_values for report in (bm25_report, hybrid_report)):
        raise ValueError("Dense, BM25, and hybrid metric cutoffs must match.")

    bm25_hash = bm25_report.get("index", {}).get("source_sha256")
    hybrid_hash = hybrid_report.get("bm25_index", {}).get("source_sha256")
    if not bm25_hash or bm25_hash != hybrid_hash:
        raise ValueError("BM25 and hybrid reports do not share the same source artifact SHA256.")
    dense_points = hybrid_report.get("dense_index", {}).get("points_count")
    source_records = hybrid_report.get("bm25_index", {}).get("source_record_count")
    if dense_points is not None and dense_points != source_records:
        raise ValueError("Hybrid dense and BM25 indexes do not contain the same source record count.")

    questions = []
    for question_id, dense_question in questions_by_run["dense"].items():
        run_questions = {name: by_id[question_id] for name, by_id in questions_by_run.items()}
        for field_name in ("question", "expected_source", "relevant_chunk_ids"):
            values = {json.dumps(question[field_name], sort_keys=True) for question in run_questions.values()}
            if len(values) != 1:
                raise ValueError(f"Evaluation field {field_name} differs for question {question_id}.")

        ranks = {name: _first_relevant_rank(question) for name, question in run_questions.items()}
        reciprocal_ranks = {name: _reciprocal_rank(rank) for name, rank in ranks.items()}
        best_component_rr = max(reciprocal_ranks["dense"], reciprocal_ranks["bm25"])
        best_components = [name for name in ("dense", "bm25") if reciprocal_ranks[name] == best_component_rr]
        questions.append(
            {
                "question_id": question_id,
                "question": dense_question["question"],
                "expected_source": dense_question["expected_source"],
                "relevant_chunk_ids": list(dense_question["relevant_chunk_ids"]),
                "query_type": classify_query_type(dense_question["question"]),
                **{f"{name}_rank": ranks[name] for name in RETRIEVER_NAMES},
                **{f"{name}_reciprocal_rank": reciprocal_ranks[name] for name in RETRIEVER_NAMES},
                "best_components": best_components,
                "best_component_reciprocal_rank": best_component_rr,
                "hybrid_vs_dense": _paired_outcome(reciprocal_ranks["hybrid"], reciprocal_ranks["dense"]),
                "hybrid_vs_bm25": _paired_outcome(reciprocal_ranks["hybrid"], reciprocal_ranks["bm25"]),
                "hybrid_vs_best_component": _paired_outcome(reciprocal_ranks["hybrid"], best_component_rr),
                "relevant_fusion_sources": _relevant_fusion_sources(run_questions["hybrid"]),
            }
        )

    cohorts = defaultdict(list)
    for question in questions:
        cohorts[question["query_type"]].append(question)
    query_types = {}
    for query_type in sorted(cohorts):
        cohort = cohorts[query_type]
        mrr = {name: sum(question[f"{name}_reciprocal_rank"] for question in cohort) / len(cohort) for name in RETRIEVER_NAMES}
        query_types[query_type] = {
            "question_count": len(cohort),
            **{f"{name}_mrr": mrr[name] for name in RETRIEVER_NAMES},
            "hybrid_delta_dense": mrr["hybrid"] - mrr["dense"],
            "hybrid_delta_bm25": mrr["hybrid"] - mrr["bm25"],
            "hybrid_vs_bm25_wins": sum(question["hybrid_vs_bm25"] == "hybrid" for question in cohort),
            "bm25_vs_hybrid_wins": sum(question["hybrid_vs_bm25"] == "baseline" for question in cohort),
            "ties": sum(question["hybrid_vs_bm25"] == "tie" for question in cohort),
        }

    hybrid_vs_best = [question["hybrid_vs_best_component"] for question in questions]
    metrics = _metric_comparison(reports, k_values)
    return {
        "schema_version": HYBRID_COMPARISON_SCHEMA_VERSION,
        "comparison_name": "dense_vs_bm25_vs_hybrid",
        "run_names": {name: report["run_name"] for name, report in reports.items()},
        "report_paths": report_paths or {},
        "question_count": len(questions),
        "k_values": k_values,
        "corpus": {
            "source_sha256": hybrid_hash,
            "source_record_count": source_records,
            "dense_points_count": dense_points,
            "bm25_document_count": hybrid_report.get("bm25_index", {}).get("document_count"),
            "bm25_skipped_document_count": hybrid_report.get("bm25_index", {}).get("skipped_document_count"),
        },
        "metrics": metrics,
        "latency_ms": {
            "dense_baseline": dense_report["latency_ms"],
            "bm25_baseline": bm25_report["latency_ms"],
            "hybrid": hybrid_report["latency_ms"],
            "dense_after_first": _warm_latency(dense_report),
            "bm25_after_first": _warm_latency(bm25_report),
            "hybrid_after_first": _warm_latency(hybrid_report),
            "hybrid_components": hybrid_report["component_latency_ms"],
        },
        "paired": {
            "hybrid_vs_dense": _paired_summary(questions, "dense"),
            "hybrid_vs_bm25": _paired_summary(questions, "bm25"),
            "hybrid_vs_best_component": {
                "hybrid_wins": hybrid_vs_best.count("hybrid"),
                "best_component_wins": hybrid_vs_best.count("baseline"),
                "ties": hybrid_vs_best.count("tie"),
            },
        },
        "decision": {
            "primary_metric": "mrr",
            "best_retriever": max(RETRIEVER_NAMES, key=lambda name: metrics["mrr"][name]),
            "hybrid_improves_over_dense": metrics["mrr"]["hybrid_delta_dense"] > 0,
            "hybrid_improves_over_bm25": metrics["mrr"]["hybrid_delta_bm25"] > 0,
        },
        "query_type_method": "Deterministic wording cohorts inherited from the Day 23 paired comparison.",
        "query_types": query_types,
        "relevance_groups": _relevance_group_summary(questions),
        "questions": questions,
    }


def _percent(value):
    return f"{100 * value:.1f}%"


def _signed_percent(value):
    return f"{100 * value:+.1f} pp"


def _rank(value):
    return "miss" if value is None else str(value)


def _escape(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_hybrid_comparison_markdown(comparison):
    """Render the Day 25 benchmark, paired outcomes, and failure analysis."""
    metrics = comparison["metrics"]
    decision = comparison["decision"]
    hybrid_mrr = metrics["mrr"]["hybrid"]
    bm25_mrr = metrics["mrr"]["bm25"]
    dense_mrr = metrics["mrr"]["dense"]
    if decision["hybrid_improves_over_bm25"]:
        verdict = "Hybrid improves on both recorded component baselines by MRR."
    elif decision["hybrid_improves_over_dense"]:
        verdict = "Hybrid improves on dense retrieval but does not improve on the stronger BM25 baseline by MRR."
    else:
        verdict = "Hybrid does not improve on either component baseline by MRR."

    lines = [
        "# Dense vs BM25 vs RRF Hybrid Benchmark",
        "",
        "## Executive summary",
        "",
        f"{verdict} On {comparison['question_count']} paired verified questions, MRR is {_percent(dense_mrr)} dense, "
        f"{_percent(bm25_mrr)} BM25, and {_percent(hybrid_mrr)} hybrid. Hybrid changes MRR by "
        f"{_signed_percent(metrics['mrr']['hybrid_delta_dense'])} versus dense and {_signed_percent(metrics['mrr']['hybrid_delta_bm25'])} versus BM25.",
        "",
        "## Benchmark table",
        "",
        "| Metric | Dense | BM25 | RRF hybrid | Hybrid − dense | Hybrid − BM25 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| MRR | {_percent(dense_mrr)} | {_percent(bm25_mrr)} | {_percent(hybrid_mrr)} | "
        f"{_signed_percent(metrics['mrr']['hybrid_delta_dense'])} | {_signed_percent(metrics['mrr']['hybrid_delta_bm25'])} |",
    ]
    for metric_name, label in (("hit_rate_at_k", "Hit rate"), ("recall_at_k", "Recall"), ("ndcg_at_k", "nDCG")):
        for k in comparison["k_values"]:
            values = metrics[metric_name][str(k)]
            lines.append(
                f"| {label}@{k} | {_percent(values['dense'])} | {_percent(values['bm25'])} | {_percent(values['hybrid'])} | "
                f"{_signed_percent(values['hybrid_delta_dense'])} | {_signed_percent(values['hybrid_delta_bm25'])} |"
            )

    dense_pair = comparison["paired"]["hybrid_vs_dense"]
    bm25_pair = comparison["paired"]["hybrid_vs_bm25"]
    best_pair = comparison["paired"]["hybrid_vs_best_component"]
    lines.extend(
        [
            "",
            "Recall and hit rate are identical because each current question has one labeled relevant chunk.",
            "",
            "## Paired rank outcomes",
            "",
            "| Comparison | Hybrid wins | Other wins | Ties | Hybrid recovers miss | Hybrid loses hit |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Hybrid vs dense | {dense_pair['hybrid_wins']} | {dense_pair['dense_wins']} | {dense_pair['ties']} | "
            f"{dense_pair['hybrid_recovers_baseline_miss']} | {dense_pair['hybrid_loses_baseline_hit']} |",
            f"| Hybrid vs BM25 | {bm25_pair['hybrid_wins']} | {bm25_pair['bm25_wins']} | {bm25_pair['ties']} | "
            f"{bm25_pair['hybrid_recovers_baseline_miss']} | {bm25_pair['hybrid_loses_baseline_hit']} |",
            "",
            f"Against the better component rank on each individual question, hybrid wins {best_pair['hybrid_wins']}, "
            f"loses {best_pair['best_component_wins']}, and ties {best_pair['ties']}.",
            "",
            "## Wording cohorts",
            "",
            "| Query type | Questions | Dense MRR | BM25 MRR | Hybrid MRR | Hybrid − BM25 | Hybrid wins | BM25 wins | Ties |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for query_type, values in comparison["query_types"].items():
        lines.append(
            f"| {query_type} | {values['question_count']} | {_percent(values['dense_mrr'])} | {_percent(values['bm25_mrr'])} | "
            f"{_percent(values['hybrid_mrr'])} | {_signed_percent(values['hybrid_delta_bm25'])} | "
            f"{values['hybrid_vs_bm25_wins']} | {values['bm25_vs_hybrid_wins']} | {values['ties']} |"
        )

    gains = [question for question in comparison["questions"] if question["hybrid_vs_bm25"] == "hybrid"]
    gains.sort(key=lambda question: (question["hybrid_reciprocal_rank"] - question["bm25_reciprocal_rank"], question["question_id"]), reverse=True)
    lines.extend(["", "## Hybrid gains over BM25", ""])
    if gains:
        lines.extend(["| Question | Type | Dense | BM25 | Hybrid |", "|---|---|---:|---:|---:|"])
        for question in gains:
            lines.append(
                f"| {_escape(question['question'])} | {question['query_type']} | {_rank(question['dense_rank'])} | "
                f"{_rank(question['bm25_rank'])} | {_rank(question['hybrid_rank'])} |"
            )
    else:
        lines.append("Hybrid does not improve the first-relevant rank of any question relative to BM25.")

    regressions = [question for question in comparison["questions"] if question["hybrid_vs_bm25"] == "baseline"]
    regressions.sort(key=lambda question: (question["bm25_reciprocal_rank"] - question["hybrid_reciprocal_rank"], question["question_id"]), reverse=True)
    lines.extend(["", "## Hybrid regressions versus BM25", ""])
    if regressions:
        lines.extend(["| Question | Type | Dense | BM25 | Hybrid |", "|---|---|---:|---:|---:|"])
        for question in regressions:
            lines.append(
                f"| {_escape(question['question'])} | {question['query_type']} | {_rank(question['dense_rank'])} | "
                f"{_rank(question['bm25_rank'])} | {_rank(question['hybrid_rank'])} |"
            )
    else:
        lines.append("Hybrid does not regress any first-relevant rank relative to BM25.")

    misses = [question for question in comparison["questions"] if question["hybrid_rank"] is None]
    lines.extend(["", "## Hybrid top-10 failures", ""])
    if misses:
        lines.extend(["| Question | Source | Dense | BM25 | Hybrid |", "|---|---|---:|---:|---:|"])
        for question in misses:
            lines.append(
                f"| {_escape(question['question'])} | {_escape(question['expected_source'])} | {_rank(question['dense_rank'])} | "
                f"{_rank(question['bm25_rank'])} | miss |"
            )
    else:
        lines.append("Hybrid retrieves every labeled chunk in the top 10.")

    latency = comparison["latency_ms"]
    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| Recorded run | Average | After first query | Minimum | Maximum |",
            "|---|---:|---:|---:|---:|",
            f"| Dense baseline | {latency['dense_baseline']['average']:.1f} ms | {latency['dense_after_first']['average']:.1f} ms | {latency['dense_baseline']['minimum']:.1f} ms | {latency['dense_baseline']['maximum']:.1f} ms |",
            f"| BM25 baseline | {latency['bm25_baseline']['average']:.1f} ms | {latency['bm25_after_first']['average']:.1f} ms | {latency['bm25_baseline']['minimum']:.1f} ms | {latency['bm25_baseline']['maximum']:.1f} ms |",
            f"| RRF hybrid | {latency['hybrid']['average']:.1f} ms | {latency['hybrid_after_first']['average']:.1f} ms | {latency['hybrid']['minimum']:.1f} ms | {latency['hybrid']['maximum']:.1f} ms |",
            "",
            "Hybrid component timings from the same run:",
            "",
            "| Stage | Average | Minimum | Maximum |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("dense", "bm25", "fusion"):
        values = latency["hybrid_components"][name]
        stage_label = RETRIEVER_LABELS.get(name, name.title())
        lines.append(f"| {stage_label} | {values['average']:.1f} ms | {values['minimum']:.1f} ms | {values['maximum']:.1f} ms |")

    corpus = comparison["corpus"]
    relevance_groups = comparison["relevance_groups"]
    lines.extend(
        [
            "",
            "The historical dense and BM25 latency rows come from separate runs and are not controlled head-to-head latency measurements. "
            "Hybrid total latency is end-to-end sequential dense retrieval, BM25 scoring, and fusion; its component table is internally comparable. "
            "First-query model warm-up can dominate averages.",
            "",
            "## Validity and decision",
            "",
            f"- The live hybrid run used {corpus['dense_points_count']} dense points and the BM25 source artifact contains {corpus['source_record_count']} records "
            f"({corpus['bm25_document_count']} searchable, {corpus['bm25_skipped_document_count']} tokenless skips).",
            f"- BM25 source SHA256: `{corpus['source_sha256']}`.",
            "- All three reports use identical question IDs, question text, expected sources, relevant chunk IDs, and metric cutoffs.",
            "- The historical dense report does not persist a corpus hash; live dense/BM25 record-count parity reduces but does not eliminate snapshot uncertainty.",
            "- The 45 source-derived questions map to only 20 labeled chunks, have one relevance judgment each, and retain lexical overlap that can favor BM25.",
            f"- Giving each of the {relevance_groups['group_count']} unique relevance groups equal weight still leaves hybrid MRR "
            f"{_signed_percent(relevance_groups['macro_mrr']['hybrid_delta_bm25'])} versus BM25; hybrid wins "
            f"{relevance_groups['hybrid_vs_bm25']['hybrid_wins']} groups, BM25 wins {relevance_groups['hybrid_vs_bm25']['bm25_wins']}, "
            f"and {relevance_groups['hybrid_vs_bm25']['ties']} tie.",
            "- Wording cohorts are deterministic heuristics, not independent human query-type labels.",
            "- Unweighted RRF rewards cross-retriever consensus. On this lexically aligned set, that can demote strong BM25-only evidence below weaker chunks appearing in both lists.",
            "",
            f"Decision on the Day 25 primary metric: **{RETRIEVER_LABELS[decision['best_retriever']]} has the highest MRR**. "
            f"Hybrid {'does' if decision['hybrid_improves_over_bm25'] else 'does not'} improve on BM25 and "
            f"{'does' if decision['hybrid_improves_over_dense'] else 'does not'} improve on dense retrieval.",
            "",
        ]
    )
    return "\n".join(lines)


def write_hybrid_comparison_artifacts(comparison, comparison_path, report_path):
    """Atomically write the three-way JSON comparison and Markdown report."""
    comparison_path = Path(comparison_path)
    report_path = Path(report_path)
    atomic_write_text(comparison_path, json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_write_text(report_path, render_hybrid_comparison_markdown(comparison))
    return comparison_path, report_path
