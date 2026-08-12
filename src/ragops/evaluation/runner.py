import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragops.evaluation.retrieval_labels import RetrievalLabel, load_retrieval_labels
from ragops.evaluation.retrieval_metrics import evaluate_retrieval_metrics, hit_at_k, ndcg_at_k, normalize_k_values, recall_at_k, reciprocal_rank
from ragops.indexing.qdrant import DEFAULT_COLLECTION_NAME, DEFAULT_QDRANT_URL, create_qdrant_client
from ragops.retrieval.base import COMMON_RETRIEVER_INTERFACE
from ragops.retrieval.dense import DEFAULT_EMBEDDING_MODEL, retrieve_dense


class DenseRetrieverEvaluationConfig(BaseModel):
    """Dense retrieval settings used during one evaluation run."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["dense"] = "dense"
    collection_name: str = DEFAULT_COLLECTION_NAME
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = Field(default=10, gt=0)
    qdrant_url: str | None = None

    @field_validator("collection_name", "embedding_model")
    @classmethod
    def clean_required_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Value must not be empty.")
        return value

    @field_validator("qdrant_url")
    @classmethod
    def clean_optional_url(cls, value):
        if value is None:
            return None
        value = value.strip().rstrip("/")
        return value or None


class EvaluationDatasetConfig(BaseModel):
    """Retrieval label input and metric cutoffs."""

    model_config = ConfigDict(extra="forbid")

    labels_path: Path = Path("data/eval/retrieval_labels.jsonl")
    k_values: list[int] = [1, 3, 5, 10]
    minimum_labels: int = Field(default=40, gt=0)

    @field_validator("k_values")
    @classmethod
    def validate_metric_cutoffs(cls, values):
        return normalize_k_values(values)


class EvaluationOutputConfig(BaseModel):
    """Artifact output settings."""

    model_config = ConfigDict(extra="forbid")

    directory: Path = Path("reports/evaluations")


class RetrievalEvaluationConfig(BaseModel):
    """Complete validated Day 19 evaluation configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    retriever_interface: Literal["common_v1"] = COMMON_RETRIEVER_INTERFACE
    retriever: DenseRetrieverEvaluationConfig
    evaluation: EvaluationDatasetConfig
    output: EvaluationOutputConfig

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        value = value.strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
            raise ValueError("Evaluation name must use lowercase letters, numbers, underscores, or hyphens.")
        return value

    @model_validator(mode="after")
    def validate_retrieval_depth(self):
        largest_cutoff = max(self.evaluation.k_values)
        if self.retriever.top_k < largest_cutoff:
            raise ValueError(f"retriever.top_k must be at least the largest evaluation cutoff ({largest_cutoff}).")
        return self


def resolve_project_path(path, project_root):
    """Resolve a configured path relative to the project root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path(project_root) / path).resolve()


def load_evaluation_config(config_path, project_root=None):
    """Load, validate, and resolve one YAML evaluation configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Evaluation config does not exist: {config_path}")

    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error

    if not isinstance(raw_config, dict) or not raw_config:
        raise ValueError(f"Evaluation config must contain a YAML mapping: {config_path}")

    config = RetrievalEvaluationConfig.model_validate(raw_config)
    project_root = Path(project_root or Path.cwd()).resolve()
    evaluation = config.evaluation.model_copy(update={"labels_path": resolve_project_path(config.evaluation.labels_path, project_root)})
    output = config.output.model_copy(update={"directory": resolve_project_path(config.output.directory, project_root)})
    return config.model_copy(update={"evaluation": evaluation, "output": output})


def load_evaluation_labels(config):
    """Load the configured verified label set and enforce its minimum size."""
    labels_path = config.evaluation.labels_path
    if not labels_path.exists():
        raise FileNotFoundError(f"Retrieval label file does not exist: {labels_path}")

    labels = load_retrieval_labels(labels_path)
    if len(labels) < config.evaluation.minimum_labels:
        raise ValueError(f"Retrieval label set contains {len(labels)} labels; at least {config.evaluation.minimum_labels} are required.")
    return labels


def chunk_id_and_score(retrieved_chunk):
    """Extract a validated chunk ID and numeric score from one result."""
    if isinstance(retrieved_chunk, dict):
        chunk_id = retrieved_chunk.get("chunk_id")
        score = retrieved_chunk.get("score", 0.0)
    else:
        chunk_id = getattr(retrieved_chunk, "chunk_id", None)
        score = getattr(retrieved_chunk, "score", 0.0)

    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("Retriever returned a result without a valid chunk_id.")

    try:
        score = float(score)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Retriever returned a non-numeric score for chunk {chunk_id}.") from error

    if not (float("-inf") < score < float("inf")):
        raise ValueError(f"Retriever returned a non-finite score for chunk {chunk_id}.")

    return chunk_id.strip(), score


def build_question_metrics(retrieved_chunk_ids, relevant_chunk_ids, k_values):
    """Compute one question's metric values."""
    return {
        "reciprocal_rank": reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids),
        "recall_at_k": {str(k): recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, k) for k in k_values},
        "hit_at_k": {str(k): hit_at_k(retrieved_chunk_ids, relevant_chunk_ids, k) for k in k_values},
        "ndcg_at_k": {str(k): ndcg_at_k(retrieved_chunk_ids, relevant_chunk_ids, k) for k in k_values},
    }


def latency_summary(question_results):
    """Summarize retrieval-only latency across questions."""
    latencies = [result["latency_ms"] for result in question_results]
    return {
        "total": sum(latencies),
        "average": sum(latencies) / len(latencies),
        "minimum": min(latencies),
        "maximum": max(latencies),
    }


def run_retrieval_evaluation(config, labels, client, retriever=retrieve_dense, clock=time.perf_counter, progress=None):
    """Run dense retrieval for every label and compute aggregate metrics."""
    labels = [label if isinstance(label, RetrievalLabel) else RetrievalLabel.model_validate(label) for label in labels]
    if not labels:
        raise ValueError("At least one retrieval label is required.")
    question_ids = [label.question_id for label in labels]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Retrieval labels must not contain duplicate question IDs.")

    rankings = {}
    question_results = []
    total_questions = len(labels)

    for index, label in enumerate(labels, start=1):
        started_at = clock()
        try:
            retrieved_chunks = list(
                retriever(
                    query=label.question,
                    client=client,
                    top_k=config.retriever.top_k,
                    collection_name=config.retriever.collection_name,
                    embedding_model=config.retriever.embedding_model,
                )
            )
        except Exception as error:
            raise RuntimeError(f"Retrieval failed for question {label.question_id}: {error}") from error
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
        question_metrics = build_question_metrics(retrieved_chunk_ids, label.relevant_chunk_ids, config.evaluation.k_values)
        question_result = {
            "question_id": label.question_id,
            "question": label.question,
            "expected_source": label.expected_source,
            "relevant_chunk_ids": list(label.relevant_chunk_ids),
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_scores": retrieved_scores,
            "latency_ms": latency_ms,
            **question_metrics,
        }
        question_results.append(question_result)

        if progress:
            progress({"index": index, "total": total_questions, "question_id": label.question_id, "latency_ms": latency_ms})

    metrics = evaluate_retrieval_metrics(rankings, labels, k_values=config.evaluation.k_values)
    return {
        "schema_version": 1,
        "run_name": config.name,
        "configuration": config.model_dump(mode="json"),
        "metrics": metrics,
        "latency_ms": latency_summary(question_results),
        "questions": question_results,
    }


def configured_qdrant_url(config):
    """Return the config, environment, or stable local Qdrant URL."""
    qdrant_url = config.retriever.qdrant_url or os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
    qdrant_url = qdrant_url.strip().rstrip("/")
    if not qdrant_url:
        return DEFAULT_QDRANT_URL
    return qdrant_url


def close_client(client):
    """Close a client when it exposes a close method."""
    close = getattr(client, "close", None)
    if close:
        close()


def evaluate_dense_config(config, labels, client_factory=create_qdrant_client, retriever=None, clock=time.perf_counter, progress=None):
    """Create one Qdrant client, validate the collection, run, and close."""
    client = client_factory(configured_qdrant_url(config))

    try:
        if not client.collection_exists(collection_name=config.retriever.collection_name):
            raise RuntimeError(f"Qdrant collection does not exist: {config.retriever.collection_name}")
        if retriever is None:
            from ragops.retrieval.factory import build_retriever

            configured_retriever = build_retriever(config, client=client, clock=clock)

            def retriever(query, top_k, **kwargs):
                return configured_retriever.retrieve(query, top_k=top_k)

        return run_retrieval_evaluation(config, labels, client, retriever=retriever, clock=clock, progress=progress)
    finally:
        close_client(client)


def atomic_write_text(path, text):
    """Atomically write text to a destination path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def csv_fieldnames(k_values):
    """Return stable CSV columns for aggregate and per-question results."""
    fields = [
        "run_name",
        "question_id",
        "question",
        "expected_source",
        "relevant_chunk_ids",
        "retrieved_chunk_ids",
        "retrieved_scores",
        "latency_ms",
        "reciprocal_rank",
        "aggregate_mrr",
    ]
    for k in k_values:
        fields.extend(
            [
                f"recall_at_{k}",
                f"hit_at_{k}",
                f"ndcg_at_{k}",
                f"aggregate_recall_at_{k}",
                f"aggregate_hit_rate_at_{k}",
                f"aggregate_ndcg_at_{k}",
            ]
        )
    return fields


def question_csv_row(report, question_result):
    """Flatten one question result and aggregate metrics into a CSV row."""
    metrics = report["metrics"]
    row = {
        "run_name": report["run_name"],
        "question_id": question_result["question_id"],
        "question": question_result["question"],
        "expected_source": question_result["expected_source"],
        "relevant_chunk_ids": json.dumps(question_result["relevant_chunk_ids"], ensure_ascii=False, separators=(",", ":")),
        "retrieved_chunk_ids": json.dumps(question_result["retrieved_chunk_ids"], ensure_ascii=False, separators=(",", ":")),
        "retrieved_scores": json.dumps(question_result["retrieved_scores"], separators=(",", ":")),
        "latency_ms": question_result["latency_ms"],
        "reciprocal_rank": question_result["reciprocal_rank"],
        "aggregate_mrr": metrics["mrr"],
    }

    for k in metrics["k_values"]:
        key = str(k)
        row[f"recall_at_{k}"] = question_result["recall_at_k"][key]
        row[f"hit_at_{k}"] = question_result["hit_at_k"][key]
        row[f"ndcg_at_{k}"] = question_result["ndcg_at_k"][key]
        row[f"aggregate_recall_at_{k}"] = metrics["recall_at_k"][key]
        row[f"aggregate_hit_rate_at_{k}"] = metrics["hit_rate_at_k"][key]
        row[f"aggregate_ndcg_at_{k}"] = metrics["ndcg_at_k"][key]

    return row


def write_csv_report(report, output_path):
    """Write stable per-question CSV rows with aggregate metric columns."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    fieldnames = csv_fieldnames(report["metrics"]["k_values"])

    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for question_result in report["questions"]:
            writer.writerow(question_csv_row(report, question_result))

    temporary_path.replace(output_path)


def write_evaluation_artifacts(report, output_directory=None):
    """Write the full JSON report and flat CSV question results."""
    output_directory = Path(output_directory or report["configuration"]["output"]["directory"])
    json_path = output_directory / f"{report['run_name']}.json"
    csv_path = output_directory / f"{report['run_name']}.csv"
    json_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(json_path, json_text)
    write_csv_report(report, csv_path)
    return json_path, csv_path
