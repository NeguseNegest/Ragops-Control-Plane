import json
import math
import time
from collections import defaultdict
from pathlib import Path

from ragops.evaluation.bm25_runner import classify_query_type
from ragops.evaluation.retrieval_labels import RetrievalLabel
from ragops.evaluation.retrieval_metrics import hit_rate_at_k, mean_ndcg_at_k, mean_recall_at_k, mean_reciprocal_rank
from ragops.evaluation.runner import atomic_write_text, build_question_metrics, chunk_id_and_score, close_client, latency_summary
from ragops.indexing.qdrant import create_qdrant_client
from ragops.reranking.cross_encoder import (
    RERANKER_METADATA_KEY,
    build_cross_encoder_reranker,
    configured_qdrant_url,
)
from ragops.retrieval.bm25 import BM25Index, load_bm25_index, validate_bm25_index
from ragops.retrieval.hybrid import FUSION_METADATA_KEY

RERANKER_COMPARISON_SCHEMA_VERSION = 1
OFFICIAL_RETRIEVERS = ("dense", "bm25", "hybrid", "reranked")
COMPARISON_RETRIEVERS = (*OFFICIAL_RETRIEVERS[:-1], "pre_rerank", "reranked")
RETRIEVER_LABELS = {
    "dense": "Dense",
    "bm25": "BM25",
    "hybrid": "RRF hybrid",
    "pre_rerank": "RRF-25 before reranking",
    "reranked": "Hybrid + cross-encoder",
}


def require_reranker_evaluation_settings(config):
    """Reject a retrieval-only reranker config before evaluation starts."""
    if config.evaluation is None or config.output is None:
        raise ValueError("Hybrid reranker config must include evaluation and output settings for Day 27.")
    return config


def _summary(values):
    return {
        "total": sum(values),
        "average": sum(values) / len(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def component_latency_summary(question_results, after_first=False):
    """Aggregate dense, sparse, fusion, and reranker stage latency."""
    selected = question_results[1:] if after_first else question_results
    if not selected:
        return None
    summary = {}
    for component in ("dense_ms", "bm25_ms", "fusion_ms", "reranker_ms"):
        summary[component.removesuffix("_ms")] = _summary([question["component_latency_ms"][component] for question in selected])
    return summary


def retrieve_reranker_candidates(query, config, client, index, reranker, clock=time.perf_counter, timings=None):
    """Return the RRF candidate ranking and its cross-encoder-reranked output."""
    from ragops.retrieval.factory import build_retriever

    configured_retriever = build_retriever(config, client=client, index=index, reranker=reranker, clock=clock)
    return configured_retriever.retrieve_with_candidates(query, timings=timings)


def _fusion_sources(result, question_id):
    metadata = getattr(result, "metadata", None)
    fusion = metadata.get(FUSION_METADATA_KEY) if isinstance(metadata, dict) else None
    if not isinstance(fusion, dict) or fusion.get("method") != "rrf":
        raise ValueError(f"Candidate {result.chunk_id} for question {question_id} is missing RRF provenance.")
    sources = fusion.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError(f"Candidate {result.chunk_id} for question {question_id} has no contributing rankings.")
    return sources


def _reranker_provenance(result, candidate_by_id, config, question_id):
    metadata = getattr(result, "metadata", None)
    provenance = metadata.get(RERANKER_METADATA_KEY) if isinstance(metadata, dict) else None
    if not isinstance(provenance, dict) or provenance.get("method") != "cross_encoder":
        raise ValueError(f"Reranked result {result.chunk_id} for question {question_id} is missing cross-encoder provenance.")
    if provenance.get("model") != config.reranker.model:
        raise ValueError(f"Reranked result {result.chunk_id} for question {question_id} has unexpected model provenance.")
    candidate = candidate_by_id.get(result.chunk_id)
    if candidate is None:
        raise ValueError(f"Reranked result {result.chunk_id} for question {question_id} is absent from the candidate pool.")
    if provenance.get("candidate_rank") != candidate.rank:
        raise ValueError(f"Reranked result {result.chunk_id} for question {question_id} has incorrect candidate rank provenance.")
    try:
        candidate_score = float(provenance.get("candidate_score"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Reranked result {result.chunk_id} for question {question_id} has invalid candidate score provenance.") from error
    if not math.isclose(candidate_score, float(candidate.score), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Reranked result {result.chunk_id} for question {question_id} has incorrect candidate score provenance.")
    return provenance


def _normalize_ranked_results(results, maximum, question_id, result_name):
    results = list(results)
    if len(results) > maximum:
        raise ValueError(f"{result_name} returned more than its configured depth for question {question_id}.")
    chunk_ids = []
    scores = []
    for expected_rank, result in enumerate(results, start=1):
        chunk_id, score = chunk_id_and_score(result)
        if chunk_id in chunk_ids:
            raise ValueError(f"{result_name} returned duplicate chunk ID {chunk_id} for question {question_id}.")
        if getattr(result, "rank", expected_rank) != expected_rank:
            raise ValueError(f"{result_name} result {chunk_id} has a non-contiguous rank for question {question_id}.")
        chunk_ids.append(chunk_id)
        scores.append(score)
    return results, chunk_ids, scores


def run_reranker_evaluation(config, labels, client, index, reranker, retriever=retrieve_reranker_candidates, clock=time.perf_counter, progress=None, dense_index=None, model_load_ms=0.0):
    """Run the fixed reranker candidate over every verified retrieval label."""
    require_reranker_evaluation_settings(config)
    if not isinstance(index, BM25Index):
        raise ValueError("index must be a loaded BM25Index.")

    labels = [label if isinstance(label, RetrievalLabel) else RetrievalLabel.model_validate(label) for label in labels]
    if not labels:
        raise ValueError("At least one retrieval label is required.")
    question_ids = [label.question_id for label in labels]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Retrieval labels must not contain duplicate question IDs.")
    if not math.isfinite(float(model_load_ms)) or model_load_ms < 0:
        raise ValueError("model_load_ms must be a non-negative finite number.")

    final_rankings = {}
    pre_rerank_rankings = {}
    question_results = []
    for position, label in enumerate(labels, start=1):
        component_timings = {}
        started_at = clock()
        try:
            candidates, reranked = retriever(
                query=label.question,
                config=config,
                client=client,
                index=index,
                reranker=reranker,
                clock=clock,
                timings=component_timings,
            )
        except Exception as error:
            raise RuntimeError(f"Reranked retrieval failed for question {label.question_id}: {error}") from error
        latency_ms = max(0.0, (clock() - started_at) * 1000)

        candidates, candidate_ids, candidate_scores = _normalize_ranked_results(candidates, config.reranker.candidate_top_k, label.question_id, "Hybrid candidate retrieval")
        reranked, retrieved_ids, retrieved_scores = _normalize_ranked_results(reranked, config.reranker.top_k, label.question_id, "Cross-encoder reranker")
        missing_timings = {"dense_ms", "bm25_ms", "fusion_ms", "reranker_ms"} - set(component_timings)
        if missing_timings:
            raise ValueError(f"Reranked retriever did not record component timings for question {label.question_id}: {sorted(missing_timings)}")
        normalized_timings = {name: float(component_timings[name]) for name in ("dense_ms", "bm25_ms", "fusion_ms", "reranker_ms")}
        if any(not math.isfinite(value) or value < 0 for value in normalized_timings.values()):
            raise ValueError(f"Reranked retriever recorded invalid component latency for question {label.question_id}.")

        candidate_sources = [_fusion_sources(candidate, label.question_id) for candidate in candidates]
        candidate_by_id = {candidate.chunk_id: candidate for candidate in candidates}
        reranker_provenance = [_reranker_provenance(result, candidate_by_id, config, label.question_id) for result in reranked]
        final_rankings[label.question_id] = retrieved_ids
        pre_rerank_ids = candidate_ids[: config.reranker.top_k]
        pre_rerank_rankings[label.question_id] = pre_rerank_ids

        question_results.append(
            {
                "question_id": label.question_id,
                "question": label.question,
                "expected_source": label.expected_source,
                "relevant_chunk_ids": list(label.relevant_chunk_ids),
                "candidate_chunk_ids": candidate_ids,
                "candidate_scores": candidate_scores,
                "candidate_fusion_sources": candidate_sources,
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved_scores": retrieved_scores,
                "retrieved_reranker_provenance": reranker_provenance,
                "latency_ms": latency_ms,
                "component_latency_ms": normalized_timings,
                "pre_rerank_metrics": build_question_metrics(pre_rerank_ids, label.relevant_chunk_ids, config.evaluation.k_values),
                **build_question_metrics(retrieved_ids, label.relevant_chunk_ids, config.evaluation.k_values),
            }
        )
        if progress:
            progress({"index": position, "total": len(labels), "question_id": label.question_id, "latency_ms": latency_ms})

    relevance = {label.question_id: label.relevant_chunk_ids for label in labels}
    payload = index.payload
    metrics = _metrics_from_rankings(final_rankings, relevance, config.evaluation.k_values, config.reranker.top_k)
    pre_rerank_metrics = _metrics_from_rankings(pre_rerank_rankings, relevance, config.evaluation.k_values, config.reranker.top_k)
    return {
        "schema_version": 1,
        "run_name": config.name,
        "configuration": config.model_dump(mode="json"),
        "model": {
            "type": config.reranker.type,
            "name": config.reranker.model,
            "batch_size": config.reranker.batch_size,
            "max_length": config.reranker.max_length,
            "device": config.reranker.device,
            "load_latency_ms": float(model_load_ms),
        },
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
        "pre_rerank_metrics": pre_rerank_metrics,
        "latency_ms": latency_summary(question_results),
        "latency_after_first_ms": _warm_latency(question_results),
        "component_latency_ms": component_latency_summary(question_results),
        "component_latency_after_first_ms": component_latency_summary(question_results, after_first=True),
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


def evaluate_reranker_config(config, labels, client_factory=create_qdrant_client, index_loader=load_bm25_index, reranker_factory=build_cross_encoder_reranker, retriever=None, clock=time.perf_counter, progress=None):
    """Validate indexes, load one model, evaluate all labels, and close Qdrant."""
    require_reranker_evaluation_settings(config)
    index = index_loader(config.bm25.index_path)
    payload = validate_bm25_index(index, config.bm25_validation_config())
    client = client_factory(configured_qdrant_url(config))
    try:
        if not client.collection_exists(collection_name=config.dense.collection_name):
            raise RuntimeError(f"Qdrant collection does not exist: {config.dense.collection_name}")
        dense_index = _dense_index_metadata(client, config.dense.collection_name)
        if dense_index["points_count"] is not None and dense_index["points_count"] != payload.source_record_count:
            raise RuntimeError(f"Dense collection contains {dense_index['points_count']} points but the shared chunk artifact contains {payload.source_record_count} records.")
        model_started_at = clock()
        reranker = reranker_factory(config.reranker)
        model_load_ms = max(0.0, (clock() - model_started_at) * 1000)
        if retriever is None:
            from ragops.retrieval.factory import build_retriever

            configured_retriever = build_retriever(config, client=client, index=index, reranker=reranker, clock=clock)

            def retriever(query, timings, **kwargs):
                return configured_retriever.retrieve_with_candidates(query, timings=timings)

        return run_reranker_evaluation(
            config,
            labels,
            client,
            index,
            reranker,
            retriever=retriever,
            clock=clock,
            progress=progress,
            dense_index=dense_index,
            model_load_ms=model_load_ms,
        )
    finally:
        close_client(client)


def _questions_by_id(report, name):
    questions = report.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{name} report has no question results.")
    by_id = {}
    for question in questions:
        question_id = question.get("question_id") if isinstance(question, dict) else None
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"{name} report contains an invalid question ID.")
        if question_id in by_id:
            raise ValueError(f"{name} report contains duplicate question ID {question_id}.")
        by_id[question_id] = question
    if report.get("metrics", {}).get("question_count") != len(by_id):
        raise ValueError(f"{name} aggregate question count does not match its results.")
    return by_id


def _metrics_from_rankings(rankings, relevance, k_values, depth):
    truncated = {question_id: list(chunk_ids)[:depth] for question_id, chunk_ids in rankings.items()}
    return {
        "question_count": len(relevance),
        "k_values": list(k_values),
        "depth": depth,
        "mrr": mean_reciprocal_rank(truncated, relevance, k=depth),
        "recall_at_k": {str(k): mean_recall_at_k(truncated, relevance, k) for k in k_values},
        "hit_rate_at_k": {str(k): hit_rate_at_k(truncated, relevance, k) for k in k_values},
        "ndcg_at_k": {str(k): mean_ndcg_at_k(truncated, relevance, k) for k in k_values},
    }


def _metrics_from_questions(questions_by_id, k_values, depth, ranking_field="retrieved_chunk_ids"):
    rankings = {question_id: question.get(ranking_field) or [] for question_id, question in questions_by_id.items()}
    relevance = {question_id: question.get("relevant_chunk_ids") or [] for question_id, question in questions_by_id.items()}
    return _metrics_from_rankings(rankings, relevance, k_values, depth)


def _validate_metrics(stored, recomputed, name):
    if stored.get("question_count") != recomputed["question_count"] or stored.get("k_values") != recomputed["k_values"]:
        raise ValueError(f"{name} stored metric dimensions do not match its question results.")
    if not math.isclose(float(stored.get("mrr")), recomputed["mrr"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name} stored MRR does not match its question results.")
    for metric_name in ("recall_at_k", "hit_rate_at_k", "ndcg_at_k"):
        for k in recomputed["k_values"]:
            if not math.isclose(float(stored[metric_name][str(k)]), recomputed[metric_name][str(k)], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{name} stored {metric_name}@{k} does not match its question results.")


def _first_relevant_rank(question, ranking_field="retrieved_chunk_ids", depth=5):
    relevant = set(question.get("relevant_chunk_ids") or [])
    for rank, chunk_id in enumerate((question.get(ranking_field) or [])[:depth], start=1):
        if chunk_id in relevant:
            return rank
    return None


def _reciprocal_rank(rank):
    return 0.0 if rank is None else 1.0 / rank


def _paired_outcome(candidate_rr, baseline_rr):
    if candidate_rr > baseline_rr:
        return "reranked"
    if baseline_rr > candidate_rr:
        return "baseline"
    return "tie"


def _paired_summary(questions, baseline_name):
    outcomes = [question[f"reranked_vs_{baseline_name}"] for question in questions]
    return {
        "reranked_wins": outcomes.count("reranked"),
        f"{baseline_name}_wins": outcomes.count("baseline"),
        "ties": outcomes.count("tie"),
        "reranked_recovers_baseline_miss": sum(question[f"{baseline_name}_rank"] is None and question["reranked_rank"] is not None for question in questions),
        "reranked_loses_baseline_hit": sum(question[f"{baseline_name}_rank"] is not None and question["reranked_rank"] is None for question in questions),
    }


def _warm_latency(question_results):
    values = [float(question["latency_ms"]) for question in question_results][1:]
    if not values:
        return None
    return {"question_count": len(values), **_summary(values)}


def _metric_comparison(run_metrics, k_values, depth):
    metrics = {"depth": depth, "mrr_at_depth": {name: values["mrr"] for name, values in run_metrics.items()}}
    metrics["mrr_at_depth"].update(
        {
            "reranked_delta_dense": metrics["mrr_at_depth"]["reranked"] - metrics["mrr_at_depth"]["dense"],
            "reranked_delta_bm25": metrics["mrr_at_depth"]["reranked"] - metrics["mrr_at_depth"]["bm25"],
            "reranked_delta_hybrid": metrics["mrr_at_depth"]["reranked"] - metrics["mrr_at_depth"]["hybrid"],
            "reranked_delta_pre_rerank": metrics["mrr_at_depth"]["reranked"] - metrics["mrr_at_depth"]["pre_rerank"],
        }
    )
    for metric_name in ("recall_at_k", "hit_rate_at_k", "ndcg_at_k"):
        metrics[metric_name] = {}
        for k in k_values:
            values = {name: run_metrics[name][metric_name][str(k)] for name in COMPARISON_RETRIEVERS}
            values.update(
                {
                    "reranked_delta_dense": values["reranked"] - values["dense"],
                    "reranked_delta_bm25": values["reranked"] - values["bm25"],
                    "reranked_delta_hybrid": values["reranked"] - values["hybrid"],
                    "reranked_delta_pre_rerank": values["reranked"] - values["pre_rerank"],
                }
            )
            metrics[metric_name][str(k)] = values
    return metrics


def _relevance_group_summary(questions):
    groups = defaultdict(list)
    for question in questions:
        groups[tuple(sorted(question["relevant_chunk_ids"]))].append(question)
    rows = []
    for relevant_ids, grouped in groups.items():
        mrr = {name: sum(question[f"{name}_reciprocal_rank"] for question in grouped) / len(grouped) for name in COMPARISON_RETRIEVERS}
        rows.append(
            {
                "relevant_chunk_ids": list(relevant_ids),
                "question_count": len(grouped),
                **{f"{name}_mrr_at_depth": value for name, value in mrr.items()},
                "reranked_delta_bm25": mrr["reranked"] - mrr["bm25"],
                "reranked_delta_hybrid": mrr["reranked"] - mrr["hybrid"],
                "reranked_delta_pre_rerank": mrr["reranked"] - mrr["pre_rerank"],
            }
        )
    macro = {name: sum(row[f"{name}_mrr_at_depth"] for row in rows) / len(rows) for name in COMPARISON_RETRIEVERS}
    return {
        "group_count": len(rows),
        "macro_mrr_at_depth": {
            **macro,
            "reranked_delta_bm25": macro["reranked"] - macro["bm25"],
            "reranked_delta_hybrid": macro["reranked"] - macro["hybrid"],
            "reranked_delta_pre_rerank": macro["reranked"] - macro["pre_rerank"],
        },
        "reranked_vs_pre_rerank": {
            "reranked_wins": sum(row["reranked_delta_pre_rerank"] > 0 for row in rows),
            "pre_rerank_wins": sum(row["reranked_delta_pre_rerank"] < 0 for row in rows),
            "ties": sum(row["reranked_delta_pre_rerank"] == 0 for row in rows),
        },
        "groups": sorted(rows, key=lambda row: row["relevant_chunk_ids"]),
    }


def compare_reranker_reports(dense_report, bm25_report, hybrid_report, reranked_report, report_paths=None):
    """Build a strict common-depth comparison and controlled reranker ablation."""
    if dense_report.get("configuration", {}).get("retriever", {}).get("type") != "dense":
        raise ValueError("Dense baseline report must use dense retrieval.")
    if bm25_report.get("configuration", {}).get("retriever", {}).get("type") != "bm25":
        raise ValueError("BM25 baseline report must use BM25 retrieval.")
    if hybrid_report.get("configuration", {}).get("fusion", {}).get("type") != "rrf":
        raise ValueError("Hybrid baseline report must use RRF fusion.")
    reranked_config = reranked_report.get("configuration", {})
    if reranked_config.get("fusion", {}).get("type") != "rrf" or reranked_config.get("reranker", {}).get("type") != "cross_encoder":
        raise ValueError("Reranked report must use RRF fusion followed by a cross-encoder.")

    dense_config = dense_report["configuration"]["retriever"]
    bm25_config = bm25_report["configuration"]["retriever"]
    for field_name in ("collection_name", "embedding_model"):
        expected = dense_config.get(field_name)
        if reranked_config.get("dense", {}).get(field_name) != expected or hybrid_report.get("configuration", {}).get("dense", {}).get(field_name) != expected:
            raise ValueError(f"Dense, hybrid, and reranked configurations differ for {field_name}.")
    for field_name in ("tokenizer", "k1", "b", "epsilon"):
        expected = bm25_config.get(field_name)
        if reranked_config.get("bm25", {}).get(field_name) != expected or hybrid_report.get("configuration", {}).get("bm25", {}).get(field_name) != expected:
            raise ValueError(f"BM25, hybrid, and reranked configurations differ for {field_name}.")

    reports = {"dense": dense_report, "bm25": bm25_report, "hybrid": hybrid_report, "reranked": reranked_report}
    questions_by_run = {name: _questions_by_id(report, name) for name, report in reports.items()}
    expected_ids = set(questions_by_run["dense"])
    for name in ("bm25", "hybrid", "reranked"):
        if set(questions_by_run[name]) != expected_ids:
            raise ValueError(f"{name} question IDs differ from the dense baseline.")

    k_values = list(reranked_report.get("metrics", {}).get("k_values") or [])
    depth = int(reranked_config.get("reranker", {}).get("top_k") or 0)
    if not k_values or depth <= 0 or max(k_values) > depth:
        raise ValueError("Reranked metric cutoffs must fit within the final reranker depth.")
    for name in ("dense", "bm25", "hybrid"):
        if not set(k_values).issubset(set(reports[name].get("metrics", {}).get("k_values") or [])):
            raise ValueError(f"{name} baseline does not contain every requested metric cutoff.")

    bm25_hash = bm25_report.get("index", {}).get("source_sha256")
    hybrid_hash = hybrid_report.get("bm25_index", {}).get("source_sha256")
    reranked_hash = reranked_report.get("bm25_index", {}).get("source_sha256")
    if not bm25_hash or len({bm25_hash, hybrid_hash, reranked_hash}) != 1:
        raise ValueError("BM25, hybrid, and reranked reports do not share the same source artifact SHA256.")
    reranked_points = reranked_report.get("dense_index", {}).get("points_count")
    source_records = reranked_report.get("bm25_index", {}).get("source_record_count")
    hybrid_points = hybrid_report.get("dense_index", {}).get("points_count")
    if reranked_points is not None and reranked_points != source_records:
        raise ValueError("Reranked dense and BM25 indexes do not contain the same source record count.")
    if hybrid_points is not None and reranked_points is not None and hybrid_points != reranked_points:
        raise ValueError("Hybrid and reranked live dense point counts differ.")

    relevance = {}
    questions = []
    for question_id, dense_question in questions_by_run["dense"].items():
        run_questions = {name: by_id[question_id] for name, by_id in questions_by_run.items()}
        for field_name in ("question", "expected_source", "relevant_chunk_ids"):
            values = {json.dumps(question[field_name], sort_keys=True) for question in run_questions.values()}
            if len(values) != 1:
                raise ValueError(f"Evaluation field {field_name} differs for question {question_id}.")
        relevance[question_id] = list(dense_question["relevant_chunk_ids"])
        rank_questions = {**run_questions, "pre_rerank": run_questions["reranked"]}
        ranks = {
            name: _first_relevant_rank(rank_questions[name], ranking_field="candidate_chunk_ids" if name == "pre_rerank" else "retrieved_chunk_ids", depth=depth)
            for name in COMPARISON_RETRIEVERS
        }
        reciprocal_ranks = {name: _reciprocal_rank(rank) for name, rank in ranks.items()}
        candidate_full_rank = _first_relevant_rank(run_questions["reranked"], ranking_field="candidate_chunk_ids", depth=reranked_config["reranker"]["candidate_top_k"])
        questions.append(
            {
                "question_id": question_id,
                "question": dense_question["question"],
                "expected_source": dense_question["expected_source"],
                "relevant_chunk_ids": list(dense_question["relevant_chunk_ids"]),
                "query_type": classify_query_type(dense_question["question"]),
                **{f"{name}_rank": ranks[name] for name in COMPARISON_RETRIEVERS},
                **{f"{name}_reciprocal_rank": reciprocal_ranks[name] for name in COMPARISON_RETRIEVERS},
                "candidate_pool_rank": candidate_full_rank,
                **{
                    f"reranked_vs_{name}": _paired_outcome(reciprocal_ranks["reranked"], reciprocal_ranks[name])
                    for name in ("dense", "bm25", "hybrid", "pre_rerank")
                },
            }
        )

    run_metrics = {
        name: _metrics_from_questions(questions_by_run[name], k_values, depth)
        for name in ("dense", "bm25", "hybrid")
    }
    run_metrics["reranked"] = _metrics_from_questions(questions_by_run["reranked"], k_values, depth)
    run_metrics["pre_rerank"] = _metrics_from_questions(questions_by_run["reranked"], k_values, depth, ranking_field="candidate_chunk_ids")
    _validate_metrics(reranked_report["metrics"], run_metrics["reranked"], "Reranked report")
    _validate_metrics(reranked_report["pre_rerank_metrics"], run_metrics["pre_rerank"], "Pre-rerank report")
    metrics = _metric_comparison(run_metrics, k_values, depth)

    cohorts = defaultdict(list)
    for question in questions:
        cohorts[question["query_type"]].append(question)
    query_types = {}
    for query_type in sorted(cohorts):
        cohort = cohorts[query_type]
        mrr = {name: sum(question[f"{name}_reciprocal_rank"] for question in cohort) / len(cohort) for name in COMPARISON_RETRIEVERS}
        query_types[query_type] = {
            "question_count": len(cohort),
            **{f"{name}_mrr_at_depth": value for name, value in mrr.items()},
            "reranked_delta_bm25": mrr["reranked"] - mrr["bm25"],
            "reranked_delta_hybrid": mrr["reranked"] - mrr["hybrid"],
            "reranked_delta_pre_rerank": mrr["reranked"] - mrr["pre_rerank"],
            "reranked_vs_pre_rerank_wins": sum(question["reranked_vs_pre_rerank"] == "reranked" for question in cohort),
            "pre_rerank_vs_reranked_wins": sum(question["reranked_vs_pre_rerank"] == "baseline" for question in cohort),
            "ties": sum(question["reranked_vs_pre_rerank"] == "tie" for question in cohort),
        }

    official_mrr = {name: metrics["mrr_at_depth"][name] for name in OFFICIAL_RETRIEVERS}
    pre_pipeline_values = [
        sum(question["component_latency_ms"][key] for key in ("dense_ms", "bm25_ms", "fusion_ms"))
        for question in reranked_report["questions"]
    ]
    return {
        "schema_version": RERANKER_COMPARISON_SCHEMA_VERSION,
        "comparison_name": "dense_vs_bm25_vs_hybrid_vs_reranked",
        "run_names": {name: report["run_name"] for name, report in reports.items()},
        "report_paths": report_paths or {},
        "question_count": len(questions),
        "k_values": k_values,
        "comparison_depth": depth,
        "metric_scope": f"All five rankings are truncated to the common final depth of {depth}; MRR is therefore MRR@{depth}.",
        "corpus": {
            "source_sha256": reranked_hash,
            "source_record_count": source_records,
            "dense_points_count": reranked_points,
            "bm25_document_count": reranked_report.get("bm25_index", {}).get("document_count"),
            "bm25_skipped_document_count": reranked_report.get("bm25_index", {}).get("skipped_document_count"),
        },
        "candidate": {
            "dense_top_k": reranked_config["dense"]["top_k"],
            "bm25_top_k": reranked_config["bm25"]["top_k"],
            "fusion_top_k": reranked_config["fusion"]["top_k"],
            "fusion_rank_constant": reranked_config["fusion"]["rank_constant"],
            "reranker_model": reranked_config["reranker"]["model"],
            "reranker_top_k": depth,
            "batch_size": reranked_config["reranker"]["batch_size"],
            "max_length": reranked_config["reranker"]["max_length"],
        },
        "metrics": metrics,
        "paired": {name: _paired_summary(questions, name) for name in ("dense", "bm25", "hybrid", "pre_rerank")},
        "decision": {
            "primary_metric": f"mrr_at_{depth}",
            "best_retriever": max(OFFICIAL_RETRIEVERS, key=lambda name: official_mrr[name]),
            "reranked_improves_over_dense": metrics["mrr_at_depth"]["reranked_delta_dense"] > 0,
            "reranked_improves_over_bm25": metrics["mrr_at_depth"]["reranked_delta_bm25"] > 0,
            "reranked_improves_over_hybrid": metrics["mrr_at_depth"]["reranked_delta_hybrid"] > 0,
            "reranking_ablation_improves": metrics["mrr_at_depth"]["reranked_delta_pre_rerank"] > 0,
        },
        "latency_ms": {
            "dense_baseline": dense_report["latency_ms"],
            "bm25_baseline": bm25_report["latency_ms"],
            "hybrid_baseline": hybrid_report["latency_ms"],
            "reranked": reranked_report["latency_ms"],
            "reranked_after_first": reranked_report.get("latency_after_first_ms"),
            "model_load": reranked_report["model"]["load_latency_ms"],
            "in_run_pre_rerank_estimate": _summary(pre_pipeline_values),
            "reranked_components": reranked_report["component_latency_ms"],
            "reranked_components_after_first": reranked_report.get("component_latency_after_first_ms"),
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


def render_reranker_comparison_markdown(comparison):
    """Render the Day 27 four-way benchmark and controlled reranker ablation."""
    depth = comparison["comparison_depth"]
    metrics = comparison["metrics"]
    mrr = metrics["mrr_at_depth"]
    decision = comparison["decision"]
    verdict = (
        f"{RETRIEVER_LABELS[decision['best_retriever']]} has the highest MRR@{depth}. "
        f"The cross-encoder {'improves' if decision['reranking_ablation_improves'] else 'does not improve'} the controlled RRF-25 ablation."
    )
    lines = [
        "# Dense vs BM25 vs RRF Hybrid vs Cross-Encoder Reranker",
        "",
        "## Executive summary",
        "",
        f"{verdict} On {comparison['question_count']} paired verified questions, hybrid plus reranking reaches {_percent(mrr['reranked'])} MRR@{depth}, "
        f"changing {_signed_percent(mrr['reranked_delta_bm25'])} versus BM25, {_signed_percent(mrr['reranked_delta_hybrid'])} versus the Day 25 RRF baseline, "
        f"and {_signed_percent(mrr['reranked_delta_pre_rerank'])} versus its own pre-rerank candidate order.",
        "",
        f"All headline rankings are truncated to the common final depth of {depth}. This avoids comparing a five-result reranker against MRR computed over ten baseline results.",
        "",
        "## Four-way benchmark table",
        "",
        "| Metric | Dense | BM25 | RRF hybrid | Hybrid + reranker | Reranker − BM25 | Reranker − hybrid |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| MRR@{depth} | {_percent(mrr['dense'])} | {_percent(mrr['bm25'])} | {_percent(mrr['hybrid'])} | {_percent(mrr['reranked'])} | {_signed_percent(mrr['reranked_delta_bm25'])} | {_signed_percent(mrr['reranked_delta_hybrid'])} |",
    ]
    for metric_name, label in (("hit_rate_at_k", "Hit rate"), ("recall_at_k", "Recall"), ("ndcg_at_k", "nDCG")):
        for k in comparison["k_values"]:
            values = metrics[metric_name][str(k)]
            lines.append(
                f"| {label}@{k} | {_percent(values['dense'])} | {_percent(values['bm25'])} | {_percent(values['hybrid'])} | {_percent(values['reranked'])} | "
                f"{_signed_percent(values['reranked_delta_bm25'])} | {_signed_percent(values['reranked_delta_hybrid'])} |"
            )

    ablation = comparison["paired"]["pre_rerank"]
    lines.extend(
        [
            "",
            "Recall and hit rate are identical because each current question has one labeled relevant chunk.",
            "",
            "## Controlled reranker ablation",
            "",
            "The official Day 25 RRF run used dense/BM25 top 20 and returned 10. The controlled ablation below instead compares the first five positions of the exact RRF-25 candidate ranking used by the reranker with its final top five.",
            "",
            "| Metric | RRF-25 before reranking | Hybrid + reranker | Delta |",
            "|---|---:|---:|---:|",
            f"| MRR@{depth} | {_percent(mrr['pre_rerank'])} | {_percent(mrr['reranked'])} | {_signed_percent(mrr['reranked_delta_pre_rerank'])} |",
        ]
    )
    for metric_name, label in (("hit_rate_at_k", "Hit rate"), ("ndcg_at_k", "nDCG")):
        for k in comparison["k_values"]:
            values = metrics[metric_name][str(k)]
            lines.append(f"| {label}@{k} | {_percent(values['pre_rerank'])} | {_percent(values['reranked'])} | {_signed_percent(values['reranked_delta_pre_rerank'])} |")
    lines.extend(
        [
            "",
            f"At the question level, reranking wins {ablation['reranked_wins']}, loses {ablation['pre_rerank_wins']}, and ties {ablation['ties']} against its own candidate order. "
            f"It recovers {ablation['reranked_recovers_baseline_miss']} top-{depth} misses and loses {ablation['reranked_loses_baseline_hit']} prior top-{depth} hits.",
            "",
            "## Paired rank outcomes",
            "",
            "| Comparison | Reranker wins | Other wins | Ties | Recovers miss | Loses hit |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("dense", "bm25", "hybrid", "pre_rerank"):
        values = comparison["paired"][name]
        lines.append(
            f"| Reranker vs {RETRIEVER_LABELS[name]} | {values['reranked_wins']} | {values[f'{name}_wins']} | {values['ties']} | "
            f"{values['reranked_recovers_baseline_miss']} | {values['reranked_loses_baseline_hit']} |"
        )

    lines.extend(
        [
            "",
            "## Wording cohorts",
            "",
            f"| Query type | Questions | BM25 MRR@{depth} | Hybrid MRR@{depth} | Pre-rerank MRR@{depth} | Reranked MRR@{depth} | Reranker − pre | Wins | Losses | Ties |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for query_type, values in comparison["query_types"].items():
        lines.append(
            f"| {query_type} | {values['question_count']} | {_percent(values['bm25_mrr_at_depth'])} | {_percent(values['hybrid_mrr_at_depth'])} | "
            f"{_percent(values['pre_rerank_mrr_at_depth'])} | {_percent(values['reranked_mrr_at_depth'])} | {_signed_percent(values['reranked_delta_pre_rerank'])} | "
            f"{values['reranked_vs_pre_rerank_wins']} | {values['pre_rerank_vs_reranked_wins']} | {values['ties']} |"
        )

    gains = [question for question in comparison["questions"] if question["reranked_vs_pre_rerank"] == "reranked"]
    gains.sort(key=lambda question: (question["reranked_reciprocal_rank"] - question["pre_rerank_reciprocal_rank"], question["question_id"]), reverse=True)
    lines.extend(["", "## Reranking gains", ""])
    if gains:
        lines.extend(["| Question | Type | Candidate-pool rank | Pre-rerank top 5 | Reranked |", "|---|---|---:|---:|---:|"])
        for question in gains:
            lines.append(
                f"| {_escape(question['question'])} | {question['query_type']} | {_rank(question['candidate_pool_rank'])} | "
                f"{_rank(question['pre_rerank_rank'])} | {_rank(question['reranked_rank'])} |"
            )
    else:
        lines.append("Reranking does not improve the first-relevant rank of any question relative to its own candidate order.")

    regressions = [question for question in comparison["questions"] if question["reranked_vs_pre_rerank"] == "baseline"]
    regressions.sort(key=lambda question: (question["pre_rerank_reciprocal_rank"] - question["reranked_reciprocal_rank"], question["question_id"]), reverse=True)
    lines.extend(["", "## Failure cases where reranking hurts", ""])
    if regressions:
        lines.extend(["| Question | Type | Candidate-pool rank | Pre-rerank top 5 | Reranked |", "|---|---|---:|---:|---:|"])
        for question in regressions:
            lines.append(
                f"| {_escape(question['question'])} | {question['query_type']} | {_rank(question['candidate_pool_rank'])} | "
                f"{_rank(question['pre_rerank_rank'])} | {_rank(question['reranked_rank'])} |"
            )
    else:
        lines.append("Reranking does not regress any first-relevant rank relative to its own candidate order.")

    misses = [question for question in comparison["questions"] if question["reranked_rank"] is None]
    lines.extend(["", f"## Reranked top-{depth} failures", ""])
    if misses:
        lines.extend(["| Question | Source | BM25 | Hybrid | Candidate pool | Reranked |", "|---|---|---:|---:|---:|---:|"])
        for question in misses:
            lines.append(
                f"| {_escape(question['question'])} | {_escape(question['expected_source'])} | {_rank(question['bm25_rank'])} | "
                f"{_rank(question['hybrid_rank'])} | {_rank(question['candidate_pool_rank'])} | miss |"
            )
    else:
        lines.append(f"The reranked pipeline retrieves every labeled chunk in its top {depth}.")

    latency = comparison["latency_ms"]
    warm = latency["reranked_after_first"]
    lines.extend(
        [
            "",
            "## Quality and latency tradeoff",
            "",
            "| Recorded run | Average | After first query | Minimum | Maximum |",
            "|---|---:|---:|---:|---:|",
            f"| Dense baseline | {latency['dense_baseline']['average']:.1f} ms | not recomputed | {latency['dense_baseline']['minimum']:.1f} ms | {latency['dense_baseline']['maximum']:.1f} ms |",
            f"| BM25 baseline | {latency['bm25_baseline']['average']:.1f} ms | not recomputed | {latency['bm25_baseline']['minimum']:.1f} ms | {latency['bm25_baseline']['maximum']:.1f} ms |",
            f"| Day 25 RRF hybrid | {latency['hybrid_baseline']['average']:.1f} ms | not recomputed | {latency['hybrid_baseline']['minimum']:.1f} ms | {latency['hybrid_baseline']['maximum']:.1f} ms |",
            f"| Hybrid + reranker | {latency['reranked']['average']:.1f} ms | {warm['average']:.1f} ms | {latency['reranked']['minimum']:.1f} ms | {latency['reranked']['maximum']:.1f} ms |",
            f"| In-run retrieval + fusion estimate | {latency['in_run_pre_rerank_estimate']['average']:.1f} ms | not separately reported | {latency['in_run_pre_rerank_estimate']['minimum']:.1f} ms | {latency['in_run_pre_rerank_estimate']['maximum']:.1f} ms |",
            "",
            f"The cross-encoder model loaded once in {latency['model_load']:.1f} ms before the question loop. That cost is excluded from per-query totals.",
            "",
            "Reranked-run component timings:",
            "",
            "| Stage | Average | After first query | Minimum | Maximum |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("dense", "bm25", "fusion", "reranker"):
        values = latency["reranked_components"][name]
        after = latency["reranked_components_after_first"][name]
        lines.append(f"| {RETRIEVER_LABELS.get(name, name.title())} | {values['average']:.1f} ms | {after['average']:.1f} ms | {values['minimum']:.1f} ms | {values['maximum']:.1f} ms |")

    corpus = comparison["corpus"]
    candidate = comparison["candidate"]
    groups = comparison["relevance_groups"]
    lines.extend(
        [
            "",
            "Historical baseline timings come from separate processes and are contextual, not controlled head-to-head latency measurements. "
            "The in-run estimate sums the dense, BM25, and fusion stages measured immediately before the reranker and is the cleanest latency ablation, "
            "though it excludes small orchestration overhead.",
            "",
            "## Validity and decision",
            "",
            f"- The live reranker run used {corpus['dense_points_count']} dense points and {corpus['source_record_count']} BM25 source records ({corpus['bm25_document_count']} searchable, {corpus['bm25_skipped_document_count']} tokenless skips).",
            f"- BM25 source SHA256: `{corpus['source_sha256']}`.",
            f"- Candidate: dense top {candidate['dense_top_k']} + BM25 top {candidate['bm25_top_k']} -> RRF top {candidate['fusion_top_k']} at k={candidate['fusion_rank_constant']:g} -> {candidate['reranker_model']} top {candidate['reranker_top_k']}.",
            f"- Every headline ranking is truncated to {depth}; MRR is reported as MRR@{depth}. The RRF-25 ablation uses the exact candidate order from the reranked run.",
            "- All reports share question IDs, wording, expected sources, relevant chunk IDs, component model/tokenizer settings, and the persisted BM25 source hash.",
            "- The historical dense report has no corpus hash. Live point/source count parity and agreement with the Day 25 live run reduce but do not eliminate snapshot uncertainty.",
            "- The configured Hugging Face model name is recorded, but a model repository revision is not pinned; a changed upstream snapshot could affect exact reproduction.",
            "- The 45 source-derived questions map to one labeled chunk each and only 20 unique relevance groups. Unjudged chunks may also be useful, while retained lexical overlap can favor BM25.",
            f"- Equal weighting across {groups['group_count']} relevance groups changes reranked MRR@{depth} by {_signed_percent(groups['macro_mrr_at_depth']['reranked_delta_bm25'])} versus BM25 and {_signed_percent(groups['macro_mrr_at_depth']['reranked_delta_pre_rerank'])} versus pre-rerank RRF-25.",
            "- Wording cohorts are deterministic heuristics, not independent human query-type labels.",
            "",
            f"Decision on the Day 27 primary metric: **{RETRIEVER_LABELS[decision['best_retriever']]} has the highest MRR@{depth}**. "
            f"Hybrid plus reranking {'does' if decision['reranked_improves_over_bm25'] else 'does not'} improve on BM25, "
            f"{'does' if decision['reranked_improves_over_hybrid'] else 'does not'} improve on the Day 25 RRF baseline, and "
            f"{'does' if decision['reranking_ablation_improves'] else 'does not'} improve on its controlled pre-rerank ordering.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reranker_comparison_artifacts(comparison, comparison_path, report_path):
    """Atomically write the Day 27 JSON comparison and Markdown report."""
    comparison_path = Path(comparison_path)
    report_path = Path(report_path)
    atomic_write_text(comparison_path, json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_write_text(report_path, render_reranker_comparison_markdown(comparison))
    return comparison_path, report_path
