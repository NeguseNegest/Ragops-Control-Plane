import json
import re
import time
from collections import defaultdict
from pathlib import Path

from ragops.evaluation.retrieval_labels import RetrievalLabel
from ragops.evaluation.retrieval_metrics import evaluate_retrieval_metrics
from ragops.evaluation.runner import atomic_write_text, build_question_metrics, chunk_id_and_score, latency_summary
from ragops.retrieval.bm25 import BM25Index, load_bm25_index, retrieve_bm25, validate_bm25_index

COMPARISON_SCHEMA_VERSION = 1

_EXACT_REFERENCE_RE = re.compile(
    r"(?:<[^>]+>|\b(?:exact|command|endpoint|host and port|method and path|field key|field|key|parameter|"
    r"data type|file format|library|package|configuration|properties|point id)\b|[a-z][a-z0-9]*_[a-z0-9_]+|\w+:/\S+)",
    re.IGNORECASE,
)
_BEHAVIORAL_RE = re.compile(
    r"^(?:how|when)\b|\b(?:what happens|what occurs|in what order|trade-off|difference|determine|forbidden|exceeded|always present)\b",
    re.IGNORECASE,
)


def require_evaluation_settings(config):
    """Reject an index-only BM25 config before starting an evaluation."""
    if config.evaluation is None or config.output is None:
        raise ValueError("BM25 config must include evaluation and output settings for Day 23.")
    return config


def classify_query_type(question):
    """Assign a reproducible wording cohort for retrieval win analysis."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be non-empty text.")
    if _EXACT_REFERENCE_RE.search(question):
        return "exact-reference"
    if _BEHAVIORAL_RE.search(question):
        return "behavioral/procedural"
    return "conceptual/descriptive"


def run_bm25_evaluation(config, labels, index, retriever=retrieve_bm25, clock=time.perf_counter, progress=None):
    """Run sparse retrieval for every label using the dense baseline metrics."""
    require_evaluation_settings(config)
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
    total_questions = len(labels)

    for position, label in enumerate(labels, start=1):
        started_at = clock()
        try:
            retrieved_chunks = list(retriever(query=label.question, index=index, top_k=config.retriever.top_k))
        except Exception as error:
            raise RuntimeError(f"BM25 retrieval failed for question {label.question_id}: {error}") from error
        latency_ms = max(0.0, (clock() - started_at) * 1000)

        if len(retrieved_chunks) > config.retriever.top_k:
            raise ValueError(f"Retriever returned more than top_k results for question {label.question_id}.")

        retrieved_chunk_ids = []
        retrieved_scores = []
        for retrieved_chunk in retrieved_chunks:
            chunk_id, score = chunk_id_and_score(retrieved_chunk)
            if chunk_id in retrieved_chunk_ids:
                raise ValueError(f"Retriever returned duplicate chunk ID {chunk_id} for question {label.question_id}.")
            retrieved_chunk_ids.append(chunk_id)
            retrieved_scores.append(score)

        rankings[label.question_id] = retrieved_chunk_ids
        question_result = {
            "question_id": label.question_id,
            "question": label.question,
            "expected_source": label.expected_source,
            "relevant_chunk_ids": list(label.relevant_chunk_ids),
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_scores": retrieved_scores,
            "latency_ms": latency_ms,
            **build_question_metrics(retrieved_chunk_ids, label.relevant_chunk_ids, config.evaluation.k_values),
        }
        question_results.append(question_result)

        if progress:
            progress({"index": position, "total": total_questions, "question_id": label.question_id, "latency_ms": latency_ms})

    metrics = evaluate_retrieval_metrics(rankings, labels, k_values=config.evaluation.k_values)
    payload = index.payload
    return {
        "schema_version": 1,
        "run_name": config.name,
        "configuration": config.model_dump(mode="json"),
        "index": {
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
        "questions": question_results,
    }


def evaluate_bm25_config(config, labels, index_loader=load_bm25_index, retriever=None, clock=time.perf_counter, progress=None):
    """Load and validate the persisted index once, then run sparse evaluation."""
    require_evaluation_settings(config)
    index = index_loader(config.retriever.index_path)
    validate_bm25_index(index, config)
    if retriever is None:
        from ragops.retrieval.factory import build_retriever

        configured_retriever = build_retriever(config, index=index, clock=clock)

        def retriever(query, top_k, **kwargs):
            return configured_retriever.retrieve(query, top_k=top_k)

    return run_bm25_evaluation(config, labels, index, retriever=retriever, clock=clock, progress=progress)


def load_evaluation_report(path):
    """Load a prior evaluation report for a strict paired comparison."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation report does not exist: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Evaluation report is invalid JSON: {path}") from error
    if not isinstance(report, dict):
        raise ValueError(f"Evaluation report must contain a JSON object: {path}")
    for key in ("run_name", "metrics", "latency_ms", "questions"):
        if key not in report:
            raise ValueError(f"Evaluation report is missing {key}: {path}")
    return report


def _questions_by_id(report):
    questions = report.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"Evaluation run {report.get('run_name', '<unknown>')} has no question results.")
    by_id = {}
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Evaluation question results must be objects.")
        question_id = question.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("Evaluation question result is missing a question_id.")
        if question_id in by_id:
            raise ValueError(f"Evaluation report contains duplicate question ID {question_id}.")
        by_id[question_id] = question
    return by_id


def _first_relevant_rank(question_result):
    relevant = set(question_result.get("relevant_chunk_ids") or [])
    for rank, chunk_id in enumerate(question_result.get("retrieved_chunk_ids") or [], start=1):
        if chunk_id in relevant:
            return rank
    return None


def _winner(dense_rank, bm25_rank):
    dense_reciprocal_rank = 0.0 if dense_rank is None else 1.0 / dense_rank
    bm25_reciprocal_rank = 0.0 if bm25_rank is None else 1.0 / bm25_rank
    if bm25_reciprocal_rank > dense_reciprocal_rank:
        return "bm25"
    if dense_reciprocal_rank > bm25_reciprocal_rank:
        return "dense"
    return "tie"


def _metric_comparison(dense_metrics, bm25_metrics, k_values):
    comparison = {
        "mrr": {
            "dense": float(dense_metrics["mrr"]),
            "bm25": float(bm25_metrics["mrr"]),
            "delta": float(bm25_metrics["mrr"]) - float(dense_metrics["mrr"]),
        }
    }
    for metric_name in ("recall_at_k", "hit_rate_at_k", "ndcg_at_k"):
        comparison[metric_name] = {}
        for k in k_values:
            key = str(k)
            dense_value = float(dense_metrics[metric_name][key])
            bm25_value = float(bm25_metrics[metric_name][key])
            comparison[metric_name][key] = {"dense": dense_value, "bm25": bm25_value, "delta": bm25_value - dense_value}
    return comparison


def compare_evaluation_reports(dense_report, bm25_report, dense_report_path=None, bm25_report_path=None):
    """Create paired aggregate, cohort, and per-question dense/BM25 results."""
    dense_type = dense_report.get("configuration", {}).get("retriever", {}).get("type")
    bm25_type = bm25_report.get("configuration", {}).get("retriever", {}).get("type")
    if dense_type != "dense":
        raise ValueError(f"Dense comparison report must use a dense retriever, got {dense_type!r}.")
    if bm25_type != "bm25":
        raise ValueError(f"BM25 comparison report must use a BM25 retriever, got {bm25_type!r}.")

    dense_questions = _questions_by_id(dense_report)
    bm25_questions = _questions_by_id(bm25_report)
    if dense_report["metrics"].get("question_count") != len(dense_questions):
        raise ValueError("Dense aggregate question count does not match its question results.")
    if bm25_report["metrics"].get("question_count") != len(bm25_questions):
        raise ValueError("BM25 aggregate question count does not match its question results.")
    if set(dense_questions) != set(bm25_questions):
        missing_from_bm25 = sorted(set(dense_questions) - set(bm25_questions))
        missing_from_dense = sorted(set(bm25_questions) - set(dense_questions))
        raise ValueError(f"Evaluation question IDs differ; missing from BM25={missing_from_bm25}, missing from dense={missing_from_dense}.")

    dense_k_values = list(dense_report["metrics"].get("k_values") or [])
    bm25_k_values = list(bm25_report["metrics"].get("k_values") or [])
    if dense_k_values != bm25_k_values or not dense_k_values:
        raise ValueError(f"Evaluation k_values differ: dense={dense_k_values}, BM25={bm25_k_values}.")

    question_comparison = []
    for question_id, dense_question in dense_questions.items():
        bm25_question = bm25_questions[question_id]
        for field_name in ("question", "expected_source", "relevant_chunk_ids"):
            if dense_question.get(field_name) != bm25_question.get(field_name):
                raise ValueError(f"Evaluation field {field_name} differs for question {question_id}.")

        dense_rank = _first_relevant_rank(dense_question)
        bm25_rank = _first_relevant_rank(bm25_question)
        winner = _winner(dense_rank, bm25_rank)
        question_comparison.append(
            {
                "question_id": question_id,
                "question": dense_question["question"],
                "expected_source": dense_question["expected_source"],
                "query_type": classify_query_type(dense_question["question"]),
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "dense_reciprocal_rank": 0.0 if dense_rank is None else 1.0 / dense_rank,
                "bm25_reciprocal_rank": 0.0 if bm25_rank is None else 1.0 / bm25_rank,
                "winner": winner,
            }
        )

    win_counts = {name: sum(question["winner"] == name for question in question_comparison) for name in ("bm25", "dense", "tie")}
    win_counts["bm25_recovered_dense_miss"] = sum(
        question["dense_rank"] is None and question["bm25_rank"] is not None for question in question_comparison
    )
    win_counts["dense_recovered_bm25_miss"] = sum(
        question["bm25_rank"] is None and question["dense_rank"] is not None for question in question_comparison
    )

    cohorts = defaultdict(list)
    for question in question_comparison:
        cohorts[question["query_type"]].append(question)
    query_types = {}
    for query_type in sorted(cohorts):
        cohort = cohorts[query_type]
        dense_mrr = sum(question["dense_reciprocal_rank"] for question in cohort) / len(cohort)
        bm25_mrr = sum(question["bm25_reciprocal_rank"] for question in cohort) / len(cohort)
        query_types[query_type] = {
            "question_count": len(cohort),
            "bm25_wins": sum(question["winner"] == "bm25" for question in cohort),
            "dense_wins": sum(question["winner"] == "dense" for question in cohort),
            "ties": sum(question["winner"] == "tie" for question in cohort),
            "dense_mrr": dense_mrr,
            "bm25_mrr": bm25_mrr,
            "mrr_delta": bm25_mrr - dense_mrr,
        }

    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "comparison_name": "bm25_vs_dense",
        "dense_run_name": dense_report["run_name"],
        "bm25_run_name": bm25_report["run_name"],
        "dense_report_path": None if dense_report_path is None else str(dense_report_path),
        "bm25_report_path": None if bm25_report_path is None else str(bm25_report_path),
        "question_count": len(question_comparison),
        "k_values": dense_k_values,
        "metrics": _metric_comparison(dense_report["metrics"], bm25_report["metrics"], dense_k_values),
        "latency_ms": {"dense": dense_report["latency_ms"], "bm25": bm25_report["latency_ms"]},
        "wins": win_counts,
        "query_type_method": "Deterministic wording cohorts: exact references first, then behavioral/procedural wording, otherwise conceptual/descriptive.",
        "query_types": query_types,
        "questions": question_comparison,
    }


def _percent(value):
    return f"{100 * value:.1f}%"


def _signed_percent(value):
    return f"{100 * value:+.1f} pp"


def _rank(value):
    return "miss" if value is None else str(value)


def _escape_table(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_comparison_markdown(comparison):
    """Render the reproducible Day 23 decision report from comparison JSON."""
    metrics = comparison["metrics"]
    lines = [
        "# Dense vs BM25 Retrieval Baseline",
        "",
        "## Executive summary",
        "",
    ]

    mrr_delta = metrics["mrr"]["delta"]
    better_name = "BM25" if mrr_delta > 0 else "dense retrieval" if mrr_delta < 0 else "Neither retriever"
    lines.append(
        f"{better_name} leads on MRR for this {_escape_table(comparison['question_count'])}-question verified label set "
        f"({_percent(metrics['mrr']['bm25'])} BM25 vs {_percent(metrics['mrr']['dense'])} dense; {_signed_percent(mrr_delta)}). "
        f"BM25 wins {comparison['wins']['bm25']} paired questions, dense wins {comparison['wins']['dense']}, and {comparison['wins']['tie']} tie."
    )
    lines.extend(
        [
            "",
            "## Aggregate retrieval quality",
            "",
            "| Metric | Dense | BM25 | BM25 − dense |",
            "|---|---:|---:|---:|",
            f"| MRR | {_percent(metrics['mrr']['dense'])} | {_percent(metrics['mrr']['bm25'])} | {_signed_percent(metrics['mrr']['delta'])} |",
        ]
    )
    for metric_name, label in (("hit_rate_at_k", "Hit rate"), ("recall_at_k", "Recall"), ("ndcg_at_k", "nDCG")):
        for k in comparison["k_values"]:
            values = metrics[metric_name][str(k)]
            lines.append(f"| {label}@{k} | {_percent(values['dense'])} | {_percent(values['bm25'])} | {_signed_percent(values['delta'])} |")

    lines.extend(
        [
            "",
            "Recall and hit rate are identical here because each verified question has one relevant chunk.",
            "",
            "## Paired wins and query types",
            "",
            "Query types are deterministic wording cohorts, not hand-retrofitted judgments: exact references include identifiers, endpoints, commands, fields, and parameters; behavioral/procedural queries ask how something behaves; all others are conceptual/descriptive.",
            f"BM25 recovers {comparison['wins']['bm25_recovered_dense_miss']} dense top-k misses; dense recovers {comparison['wins']['dense_recovered_bm25_miss']} BM25 top-k misses.",
            "",
            "| Query type | Questions | BM25 wins | Dense wins | Ties | Dense MRR | BM25 MRR | Delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for query_type, values in comparison["query_types"].items():
        lines.append(
            f"| {query_type} | {values['question_count']} | {values['bm25_wins']} | {values['dense_wins']} | {values['ties']} | "
            f"{_percent(values['dense_mrr'])} | {_percent(values['bm25_mrr'])} | {_signed_percent(values['mrr_delta'])} |"
        )

    lines.extend(["", "### Questions where BM25 wins", ""])
    bm25_wins = [question for question in comparison["questions"] if question["winner"] == "bm25"]
    bm25_wins.sort(key=lambda question: (question["dense_reciprocal_rank"] - question["bm25_reciprocal_rank"], question["question_id"]))
    if bm25_wins:
        lines.extend(["| Question | Type | Dense rank | BM25 rank |", "|---|---|---:|---:|"])
        for question in bm25_wins:
            lines.append(
                f"| {_escape_table(question['question'])} | {question['query_type']} | {_rank(question['dense_rank'])} | {_rank(question['bm25_rank'])} |"
            )
    else:
        lines.append("BM25 does not win any paired question in this run.")

    lines.extend(["", "### Questions where dense retrieval wins", ""])
    dense_wins = [question for question in comparison["questions"] if question["winner"] == "dense"]
    dense_wins.sort(key=lambda question: (question["bm25_reciprocal_rank"] - question["dense_reciprocal_rank"], question["question_id"]))
    if dense_wins:
        lines.extend(["| Question | Type | Dense rank | BM25 rank |", "|---|---|---:|---:|"])
        for question in dense_wins:
            lines.append(
                f"| {_escape_table(question['question'])} | {question['query_type']} | {_rank(question['dense_rank'])} | {_rank(question['bm25_rank'])} |"
            )
    else:
        lines.append("Dense retrieval does not win any paired question in this run.")

    dense_latency = comparison["latency_ms"]["dense"]
    bm25_latency = comparison["latency_ms"]["bm25"]
    lines.extend(
        [
            "",
            "## Latency context",
            "",
            "| Retriever | Average | Minimum | Maximum |",
            "|---|---:|---:|---:|",
            f"| Dense | {dense_latency['average']:.1f} ms | {dense_latency['minimum']:.1f} ms | {dense_latency['maximum']:.1f} ms |",
            f"| BM25 | {bm25_latency['average']:.1f} ms | {bm25_latency['minimum']:.1f} ms | {bm25_latency['maximum']:.1f} ms |",
            "",
            "These timings are diagnostic, not a controlled benchmark. Dense latency includes query embedding and Qdrant search, with its recorded first-query model warm-up; BM25 latency measures in-memory scoring after the persisted index is loaded.",
            "",
            "## Reproduction and limits",
            "",
            f"- Both runs use the same {comparison['question_count']} verified questions and cutoffs {comparison['k_values']}.",
            "- Per-question wins compare the rank of the first relevant chunk; two misses or equal ranks are ties.",
            "- The verified set is intentionally small and source-balanced enough for iteration, not statistical proof of production superiority.",
            "- Questions were generated from and verified against exact source chunks, so retained source vocabulary can favor lexical retrieval.",
            "- Use the machine-readable comparison artifact for downstream hybrid-retrieval experiments.",
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison_artifacts(comparison, comparison_path, report_path):
    """Atomically write the paired JSON artifact and narrative Markdown report."""
    comparison_path = Path(comparison_path)
    report_path = Path(report_path)
    atomic_write_text(comparison_path, json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_write_text(report_path, render_comparison_markdown(comparison))
    return comparison_path, report_path
