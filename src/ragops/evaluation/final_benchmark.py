import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragops.evaluation.bm25_runner import classify_query_type
from ragops.evaluation.llm_judge import (
    AutomaticJudgment,
    build_judge_prompt,
    evidence_from_chunks,
    load_golden_questions,
    parse_judge_response,
)
from ragops.evaluation.retrieval_labels import load_retrieval_labels
from ragops.evaluation.retrieval_metrics import recall_at_k, reciprocal_rank
from ragops.evaluation.runner import atomic_write_text
from ragops.generation.client import GenerationUsage, generate_answer
from ragops.generation.cost import estimate_generation_cost, load_model_cost_table
from ragops.generation.no_answer import NO_ANSWER_RESPONSE
from ragops.pipeline_registry import PipelineStatus, PipelineVersion
from ragops.retrieval.dense import RetrievedChunk, source_url_from_metadata
from ragops.routing.config import load_router_config
from ragops.routing.probe import build_initial_retrieval_features
from ragops.routing.router import RuleBasedRouter
from ragops.schemas import DocumentChunk

FINAL_BENCHMARK_SCHEMA_VERSION = 1
PIPELINE_NAMES = ("dense", "bm25", "hybrid", "reranked", "routed")
FIXED_PIPELINES = PIPELINE_NAMES[:-1]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FinalDatasetConfig(StrictModel):
    golden_path: Path
    retrieval_labels_path: Path
    adversarial_path: Path
    chunks_path: Path
    expected_retrieval_questions: int = Field(gt=0)
    expected_adversarial_questions: int = Field(gt=0)


class PipelineArtifactConfig(StrictModel):
    label: str = Field(min_length=1)
    config_path: Path
    retrieval_report_path: Path
    judgments_path: Path

    @field_validator("label")
    @classmethod
    def clean_label(cls, value):
        return value.strip()


class FinalPipelinesConfig(StrictModel):
    dense: PipelineArtifactConfig
    bm25: PipelineArtifactConfig
    hybrid: PipelineArtifactConfig
    reranked: PipelineArtifactConfig
    routed: PipelineArtifactConfig

    def items(self):
        return [(name, getattr(self, name)) for name in PIPELINE_NAMES]


class ProviderConfig(StrictModel):
    provider: Literal["openai", "gemini"]
    model: str = Field(min_length=1)

    @field_validator("model")
    @classmethod
    def clean_model(cls, value):
        return value.strip()


class AnswerQualityConfig(StrictModel):
    sample_question_ids: list[str] = Field(min_length=1)
    generation: ProviderConfig
    judge: ProviderConfig
    require_cross_provider_judge: bool = True

    @field_validator("sample_question_ids")
    @classmethod
    def unique_sample(cls, values):
        values = [value.strip() for value in values]
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError("Answer-quality sample IDs must be non-empty and unique.")
        return values

    @model_validator(mode="after")
    def cross_provider(self):
        if self.require_cross_provider_judge and self.generation.provider == self.judge.provider:
            raise ValueError("Final benchmark generation and judge providers must differ.")
        return self


class RetrievalBenchmarkConfig(StrictModel):
    cutoff: Literal[5] = 5
    routed_fast_depth: Literal[2] = 2
    routed_standard_depth: Literal[10] = 10
    routed_careful_depth: Literal[5] = 5


class LatencyBenchmarkConfig(StrictModel):
    unit: Literal["ms"] = "ms"
    percentile_method: Literal["linear_interpolation"]
    scope: Literal["measured_retrieval_only"]
    routed_method: Literal["measured_serial_artifact_replay"]
    include_cold_start: Literal[True] = True


class CostProjectionConfig(StrictModel):
    provider: Literal["openai", "gemini"]
    model: str = Field(min_length=1)
    model_cost_config: Path
    token_basis: Literal["exact_prompt_plus_verified_reference_answer"]
    scope: Literal["supported_retrieval_questions"]


class FinalMLflowConfig(StrictModel):
    tracking_uri: str = Field(min_length=1)
    experiment_name: str = Field(min_length=1)

    @field_validator("tracking_uri", "experiment_name")
    @classmethod
    def clean_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("MLflow values must not be empty.")
        return value.rstrip("/") if "://" in value else value


class FinalOutputConfig(StrictModel):
    json_path: Path
    csv_path: Path
    markdown_path: Path


class FinalBenchmarkConfig(StrictModel):
    schema_version: Literal[1] = FINAL_BENCHMARK_SCHEMA_VERSION
    name: Literal["final_benchmark"]
    version: PipelineVersion
    status: PipelineStatus
    mode: Literal["measured_artifact_replay"]
    datasets: FinalDatasetConfig
    pipelines: FinalPipelinesConfig
    answer_quality: AnswerQualityConfig
    retrieval: RetrievalBenchmarkConfig
    latency: LatencyBenchmarkConfig
    cost_projection: CostProjectionConfig
    mlflow: FinalMLflowConfig
    output: FinalOutputConfig

    @model_validator(mode="after")
    def distinct_outputs(self):
        outputs = {self.output.json_path, self.output.csv_path, self.output.markdown_path}
        if len(outputs) != 3:
            raise ValueError("Final benchmark JSON, CSV, and Markdown outputs must be distinct.")
        return self


class BenchmarkJudgment(StrictModel):
    schema_version: Literal[1] = 1
    run_name: str
    pipeline: Literal["dense", "bm25", "hybrid", "reranked", "routed"]
    question_id: str
    question: str
    retrieved_chunk_ids: list[str]
    route: str | None = None
    generated_answer: str
    generation_usage: GenerationUsage | None = None
    generation_ms: float = Field(ge=0)
    judge_ms: float = Field(ge=0)
    automatic_judgment: AutomaticJudgment
    created_at: str


def _resolve(path, root):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def load_final_benchmark_config(config_path, project_root=None):
    """Load the strict Day 47 benchmark contract and resolve every artifact path."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Final benchmark config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid final benchmark YAML in {config_path}: {error}") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Final benchmark config must contain a YAML mapping: {config_path}")
    config = FinalBenchmarkConfig.model_validate(payload)
    root = Path(project_root or Path.cwd()).resolve()
    datasets = config.datasets.model_copy(
        update={
            name: _resolve(getattr(config.datasets, name), root)
            for name in ("golden_path", "retrieval_labels_path", "adversarial_path", "chunks_path")
        }
    )
    pipeline_updates = {}
    for name, pipeline in config.pipelines.items():
        pipeline_updates[name] = pipeline.model_copy(
            update={
                field: _resolve(getattr(pipeline, field), root)
                for field in ("config_path", "retrieval_report_path", "judgments_path")
            }
        )
    pipelines = config.pipelines.model_copy(update=pipeline_updates)
    cost = config.cost_projection.model_copy(
        update={"model_cost_config": _resolve(config.cost_projection.model_cost_config, root)}
    )
    output = config.output.model_copy(
        update={name: _resolve(getattr(config.output, name), root) for name in ("json_path", "csv_path", "markdown_path")}
    )
    return config.model_copy(update={"datasets": datasets, "pipelines": pipelines, "cost_projection": cost, "output": output})


def _read_json(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} contains invalid JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _read_jsonl(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{label} contains invalid JSON on line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{label} line {line_number} must be a JSON object.")
            rows.append(row)
    return rows


def _pipeline_config_name(path):
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid pipeline YAML in {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        raise ValueError(f"Pipeline config must contain a name: {path}")
    return payload["name"]


def _question_index(report, label):
    questions = report.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{label} must contain non-empty question results.")
    indexed = {}
    for row in questions:
        question_id = row.get("question_id") if isinstance(row, dict) else None
        if not isinstance(question_id, str) or not question_id or question_id in indexed:
            raise ValueError(f"{label} contains invalid or duplicate question IDs.")
        indexed[question_id] = row
    return indexed


def validate_retrieval_reports(config, reports=None):
    """Require four frozen, paired retrieval reports and exact final-label provenance."""
    labels = load_retrieval_labels(config.datasets.retrieval_labels_path)
    if len(labels) != config.datasets.expected_retrieval_questions:
        raise ValueError(
            f"Final benchmark has {len(labels)} retrieval labels; expected {config.datasets.expected_retrieval_questions}."
        )
    labels_by_id = {label.question_id: label for label in labels}
    reports = dict(reports or {})
    indexed = {}
    for pipeline in FIXED_PIPELINES:
        artifact = getattr(config.pipelines, pipeline)
        report = reports.get(pipeline) or _read_json(artifact.retrieval_report_path, f"{pipeline} retrieval report")
        expected_name = _pipeline_config_name(artifact.config_path)
        if report.get("run_name") != expected_name:
            raise ValueError(f"{pipeline} report run_name must be {expected_name!r}.")
        by_id = _question_index(report, f"{pipeline} retrieval report")
        if set(by_id) != set(labels_by_id):
            raise ValueError(f"{pipeline} retrieval report does not exactly cover the frozen final labels.")
        for question_id, label in labels_by_id.items():
            row = by_id[question_id]
            expected = (label.question, label.expected_source, list(label.relevant_chunk_ids))
            actual = (row.get("question"), row.get("expected_source"), row.get("relevant_chunk_ids"))
            if actual != expected:
                raise ValueError(f"{pipeline} report provenance differs for {question_id}.")
            ids = row.get("retrieved_chunk_ids")
            scores = row.get("retrieved_scores")
            if not isinstance(ids, list) or len(ids) < config.retrieval.cutoff or len(ids) != len(set(ids)):
                raise ValueError(f"{pipeline} report has an invalid top-{config.retrieval.cutoff} ranking for {question_id}.")
            if not isinstance(scores, list) or len(scores) != len(ids):
                raise ValueError(f"{pipeline} report has incomplete scores for {question_id}.")
            latency = row.get("latency_ms")
            if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
                raise ValueError(f"{pipeline} report has invalid latency for {question_id}.")
        reports[pipeline] = report
        indexed[pipeline] = by_id
    return labels, reports, indexed


def validate_final_benchmark_inputs(config, require_reports=True, require_judgments=False):
    """Validate dataset sizes, sample coverage, model pricing, configs, and optional run artifacts."""
    golden = load_golden_questions(config.datasets.golden_path)
    golden_by_id = {question.id: question for question in golden}
    sample_ids = config.answer_quality.sample_question_ids
    missing = sorted(set(sample_ids) - set(golden_by_id))
    if missing:
        raise ValueError(f"Answer-quality sample IDs are missing from the final golden set: {missing}.")
    if any(golden_by_id[question_id].query_type != "supported" for question_id in sample_ids):
        raise ValueError("Answer-quality sample must contain supported questions only; refusal is measured separately.")

    labels = load_retrieval_labels(config.datasets.retrieval_labels_path)
    label_ids = {label.question_id for label in labels}
    if len(labels) != config.datasets.expected_retrieval_questions or not set(sample_ids) <= label_ids:
        raise ValueError("Final retrieval labels do not have the configured size or answer-quality sample coverage.")
    adversarial = _read_jsonl(config.datasets.adversarial_path, "Final adversarial dataset")
    if len(adversarial) != config.datasets.expected_adversarial_questions:
        raise ValueError(
            f"Final adversarial dataset has {len(adversarial)} rows; expected {config.datasets.expected_adversarial_questions}."
        )
    if any(row.get("query_type") != "unsupported" or row.get("expected_behavior") != "refusal" for row in adversarial):
        raise ValueError("Every final adversarial row must require unsupported-query refusal behavior.")
    adversarial_ids = [row.get("id") for row in adversarial]
    if any(not isinstance(value, str) or not value for value in adversarial_ids) or len(adversarial_ids) != len(set(adversarial_ids)):
        raise ValueError("Final adversarial IDs must be non-empty and unique.")

    for _, pipeline in config.pipelines.items():
        if not pipeline.config_path.is_file():
            raise FileNotFoundError(f"Pipeline config does not exist: {pipeline.config_path}")
    if not config.datasets.chunks_path.is_file():
        raise FileNotFoundError(f"Processed chunks do not exist: {config.datasets.chunks_path}")
    cost_table = load_model_cost_table(config.cost_projection.model_cost_config)
    if cost_table.pricing_for(config.cost_projection.provider, config.cost_projection.model) is None:
        raise ValueError("Final benchmark cost model has no exact entry in the configured model cost table.")

    reports = indexed = None
    if require_reports:
        labels, reports, indexed = validate_retrieval_reports(config)
    if require_judgments:
        for pipeline, artifact in config.pipelines.items():
            records = load_benchmark_judgments(artifact.judgments_path)
            validate_benchmark_judgments(records, config, pipeline)
    return {
        "golden": golden,
        "golden_by_id": golden_by_id,
        "labels": labels,
        "adversarial": adversarial,
        "cost_table": cost_table,
        "reports": reports,
        "indexed": indexed,
    }


def _probe_features(question, row):
    scores = row["retrieved_scores"]
    ids = row["retrieved_chunk_ids"]
    chunks = [
        RetrievedChunk(
            chunk_id=ids[index],
            document_id="final-benchmark-probe",
            text="Probe evidence.",
            score=float(scores[index]),
            rank=index + 1,
            metadata={},
        )
        for index in range(2)
    ]
    return build_initial_retrieval_features(question, chunks, requested_top_k=2)


def _rank_metrics(ids, relevant, cutoff=5):
    ids = list(ids)[:cutoff]
    return {
        "reciprocal_rank_at_5": reciprocal_rank(ids, relevant),
        "recall_at_5": recall_at_k(ids, relevant, cutoff),
    }


def _percentile(values, quantile):
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("A percentile requires at least one value.")
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def numeric_summary(values):
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Summary values must be finite, non-negative, and non-empty.")
    return {
        "count": len(values),
        "average": sum(values) / len(values),
        "minimum": min(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
    }


def build_routed_report(config, labels, indexed, adversarial, route_query, clock=time.perf_counter, progress=None):
    """Replay supported routes and run all held-out unsupported policy probes."""
    router_config = load_router_config(config.pipelines.routed.config_path, project_root=config.pipelines.routed.config_path.parents[1])
    router = RuleBasedRouter(router_config)
    dense_by_id = indexed["dense"]
    reranked_by_id = indexed["reranked"]
    supported_rows = []
    for label in labels:
        dense = dense_by_id[label.question_id]
        reranked = reranked_by_id[label.question_id]
        decision = router.select(_probe_features(label.question, dense))
        if decision.route == "FAST":
            depth = config.retrieval.routed_fast_depth
            source = dense
            latency_ms = dense["latency_ms"]
        elif decision.route == "STANDARD":
            depth = config.retrieval.routed_standard_depth
            source = dense
            latency_ms = 2 * dense["latency_ms"]
        elif decision.route == "CAREFUL":
            depth = config.retrieval.routed_careful_depth
            source = reranked
            latency_ms = dense["latency_ms"] + reranked["latency_ms"]
        else:
            depth = 0
            source = dense
            latency_ms = dense["latency_ms"]
        ids = list(source["retrieved_chunk_ids"][:depth])
        scores = list(source["retrieved_scores"][:depth])
        supported_rows.append(
            {
                "question_id": label.question_id,
                "question": label.question,
                "expected_source": label.expected_source,
                "relevant_chunk_ids": list(label.relevant_chunk_ids),
                "retrieved_chunk_ids": ids,
                "retrieved_scores": scores,
                "latency_ms": float(latency_ms),
                "route": decision.route,
                "reason_code": decision.reason_code,
                "source_pipeline": None if decision.route == "NO_ANSWER" else decision.pipeline_config,
                **_rank_metrics(ids, label.relevant_chunk_ids),
            }
        )

    policy_rows = []
    for index, example in enumerate(adversarial, start=1):
        started = clock()
        result = route_query(example["question"])
        latency_ms = max(0.0, (clock() - started) * 1000)
        decision = result.decision
        refused = decision.route == "NO_ANSWER"
        policy_rows.append(
            {
                "question_id": example["id"],
                "question": example["question"],
                "query_type": example["query_type"],
                "difficulty": example["difficulty"],
                "category": example["category"],
                "expected_behavior": example["expected_behavior"],
                "route": decision.route,
                "reason_code": decision.reason_code,
                "top_score": result.probe.features.retrieval_confidence.top_score,
                "score_gap": result.probe.features.retrieval_confidence.score_gap,
                "latency_ms": latency_ms,
                "refused": refused,
                "correct": refused,
            }
        )
        if progress:
            progress({"index": index, "total": len(adversarial), "question_id": example["id"], "route": decision.route})

    quality = [_rank_metrics(row["retrieved_chunk_ids"], row["relevant_chunk_ids"]) for row in supported_rows]
    correct = sum(row["correct"] for row in policy_rows)
    return {
        "schema_version": 1,
        "run_name": "final_routed",
        "router_id": f"{router_config.name}@{router_config.version}",
        "configuration": router_config.model_dump(mode="json"),
        "measurement": {
            "supported": "route decisions replayed from final dense scores; route latency is serial measured-artifact composition",
            "adversarial": "live dense top-2 probe and deterministic router decision",
            "standard_latency": "dense top-10 latency counted twice as probe-plus-full-retrieval proxy",
            "careful_latency": "dense top-10 probe proxy plus full reranked retrieval latency",
        },
        "metrics": {
            "question_count": len(supported_rows),
            "mrr_at_5": sum(row["reciprocal_rank_at_5"] for row in quality) / len(quality),
            "recall_at_5": sum(row["recall_at_5"] for row in quality) / len(quality),
        },
        "latency_ms": numeric_summary(row["latency_ms"] for row in supported_rows),
        "route_counts_supported": dict(sorted(Counter(row["route"] for row in supported_rows).items())),
        "refusal_correctness": {
            "question_count": len(policy_rows),
            "correct": correct,
            "incorrect": len(policy_rows) - correct,
            "accuracy": correct / len(policy_rows),
        },
        "questions": supported_rows,
        "policy_questions": policy_rows,
    }


def write_routed_report(report, config, overwrite=False):
    path = config.pipelines.routed.retrieval_report_path
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite routed report: {path}")
    atomic_write_text(path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return path


def _load_chunks(path):
    chunks = {}
    for row in _read_jsonl(path, "Processed chunks"):
        chunk = DocumentChunk.model_validate(row)
        if chunk.chunk_id in chunks:
            raise ValueError(f"Processed chunks contain duplicate ID {chunk.chunk_id}.")
        chunks[chunk.chunk_id] = chunk
    return chunks


def _retrieved_chunks(row, chunks_by_id, depth=None):
    ids = row["retrieved_chunk_ids"] if depth is None else row["retrieved_chunk_ids"][:depth]
    scores = row["retrieved_scores"] if depth is None else row["retrieved_scores"][:depth]
    chunks = []
    for rank, (chunk_id, score) in enumerate(zip(ids, scores, strict=True), start=1):
        try:
            chunk = chunks_by_id[chunk_id]
        except KeyError as error:
            raise ValueError(f"Processed chunk artifact is missing retrieved chunk {chunk_id}.") from error
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=float(score),
                rank=rank,
                metadata=chunk.metadata,
                source_url=source_url_from_metadata(chunk.metadata),
            )
        )
    return chunks


def _quality_row(pipeline, indexed, question_id):
    row = indexed[pipeline][question_id]
    depth = None if pipeline == "routed" else 5
    return row, depth


def run_answer_quality_pipeline(
    config,
    pipeline,
    indexed,
    golden_by_id,
    chunks_by_id,
    generator_client,
    judge_client,
    clock=time.perf_counter,
    timestamp_factory=None,
    progress=None,
    existing_records=None,
    checkpoint=None,
):
    """Generate and cross-provider judge the identical supported sample for one pipeline."""
    if pipeline not in PIPELINE_NAMES:
        raise ValueError(f"Unknown final benchmark pipeline: {pipeline}")
    timestamp_factory = timestamp_factory or (lambda: datetime.now(UTC).isoformat())
    records = validate_benchmark_judgment_prefix(existing_records or [], config, pipeline)
    remaining_ids = config.answer_quality.sample_question_ids[len(records) :]
    for position, question_id in enumerate(remaining_ids, start=len(records) + 1):
        question = golden_by_id[question_id]
        row, depth = _quality_row(pipeline, indexed, question_id)
        chunks = _retrieved_chunks(row, chunks_by_id, depth=depth)
        if pipeline == "routed" and row.get("route") == "NO_ANSWER":
            answer = NO_ANSWER_RESPONSE
            usage = None
            generation_ms = 0.0
        else:
            if not chunks:
                raise ValueError(f"{pipeline} returned no generation context for supported question {question_id}.")
            started = clock()
            generation = generate_answer(question.question, chunks, client=generator_client)
            generation_ms = max(0.0, (clock() - started) * 1000)
            answer = generation.answer
            usage = generation.usage
        judge_prompt = build_judge_prompt(question, answer, evidence_from_chunks(chunks))
        started = clock()
        judgment = parse_judge_response(judge_client.generate(judge_prompt), question.query_type)
        judge_ms = max(0.0, (clock() - started) * 1000)
        record = BenchmarkJudgment(
            run_name=f"final_{pipeline}_answer_quality",
            pipeline=pipeline,
            question_id=question.id,
            question=question.question,
            retrieved_chunk_ids=[chunk.chunk_id for chunk in chunks],
            route=row.get("route"),
            generated_answer=answer,
            generation_usage=usage,
            generation_ms=generation_ms,
            judge_ms=judge_ms,
            automatic_judgment=judgment,
            created_at=timestamp_factory(),
        )
        records.append(record)
        if checkpoint:
            checkpoint(records)
        if progress:
            progress({"pipeline": pipeline, "index": position, "total": len(config.answer_quality.sample_question_ids), "question_id": question_id, "generation_ms": generation_ms, "judge_ms": judge_ms})
    return records


def validate_benchmark_judgments(records, config, pipeline):
    records = [record if isinstance(record, BenchmarkJudgment) else BenchmarkJudgment.model_validate(record) for record in records]
    expected_ids = config.answer_quality.sample_question_ids
    if [record.question_id for record in records] != expected_ids:
        raise ValueError(f"{pipeline} judgments do not exactly match the ordered answer-quality sample.")
    if any(record.pipeline != pipeline or record.run_name != f"final_{pipeline}_answer_quality" for record in records):
        raise ValueError(f"{pipeline} judgment provenance is inconsistent.")
    return records


def validate_benchmark_judgment_prefix(records, config, pipeline):
    records = [record if isinstance(record, BenchmarkJudgment) else BenchmarkJudgment.model_validate(record) for record in records]
    expected_ids = config.answer_quality.sample_question_ids[: len(records)]
    if len(records) > len(config.answer_quality.sample_question_ids) or [record.question_id for record in records] != expected_ids:
        raise ValueError(f"{pipeline} judgment checkpoint is not an ordered prefix of the answer-quality sample.")
    if any(record.pipeline != pipeline or record.run_name != f"final_{pipeline}_answer_quality" for record in records):
        raise ValueError(f"{pipeline} judgment checkpoint provenance is inconsistent.")
    return records


def load_benchmark_judgments(path):
    return [BenchmarkJudgment.model_validate(row) for row in _read_jsonl(path, "Benchmark judgments")]


def write_benchmark_judgments(records, config, pipeline, overwrite=False):
    records = validate_benchmark_judgments(records, config, pipeline)
    path = getattr(config.pipelines, pipeline).judgments_path
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite benchmark judgments: {path}")
    text = "".join(
        json.dumps(record.model_dump(mode="json", exclude_none=True), ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    atomic_write_text(path, text)
    return path


def write_benchmark_judgment_checkpoint(records, config, pipeline):
    """Atomically persist one valid ordered prefix so throttled runs can resume."""
    records = validate_benchmark_judgment_prefix(records, config, pipeline)
    path = getattr(config.pipelines, pipeline).judgments_path
    text = "".join(
        json.dumps(record.model_dump(mode="json", exclude_none=True), ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    atomic_write_text(path, text)
    return path


def _first_relevant_rank(row, depth=5, field="retrieved_chunk_ids"):
    relevant = set(row["relevant_chunk_ids"])
    for rank, chunk_id in enumerate(row[field][:depth], start=1):
        if chunk_id in relevant:
            return rank
    return None


def _fixed_metrics(rows):
    values = [_rank_metrics(row["retrieved_chunk_ids"], row["relevant_chunk_ids"]) for row in rows.values()]
    return {
        "question_count": len(values),
        "mrr_at_5": sum(row["reciprocal_rank_at_5"] for row in values) / len(values),
        "recall_at_5": sum(row["recall_at_5"] for row in values) / len(values),
    }


def _quality_summary(records):
    faithfulness = [record.automatic_judgment.faithfulness.score for record in records]
    relevance = [record.automatic_judgment.answer_relevance.score for record in records]
    return {
        "question_count": len(records),
        "mean_faithfulness": sum(faithfulness) / len(faithfulness),
        "mean_answer_relevance": sum(relevance) / len(relevance),
        "generation_latency_ms": numeric_summary(record.generation_ms for record in records),
        "judge_latency_ms": numeric_summary(record.judge_ms for record in records),
    }


def _project_costs(config, indexed, golden_by_id, chunks_by_id, cost_table):
    client = type("CostClient", (), {"provider": config.cost_projection.provider, "model": config.cost_projection.model})()
    projected = {pipeline: {} for pipeline in PIPELINE_NAMES}
    for pipeline in PIPELINE_NAMES:
        for question_id, question in golden_by_id.items():
            if question_id not in indexed[pipeline]:
                continue
            row, depth = _quality_row(pipeline, indexed, question_id)
            chunks = _retrieved_chunks(row, chunks_by_id, depth=depth)
            if pipeline == "routed" and row.get("route") == "NO_ANSWER":
                amount = 0.0
                details = {"status": "zero_cost", "reason": "deterministic_no_answer_policy"}
            else:
                if not chunks:
                    raise ValueError(f"Cannot project generation cost without context for {pipeline}/{question_id}.")
                generation = generate_answer(question.question, chunks)
                cost = estimate_generation_cost(
                    client,
                    usage=None,
                    cost_table=cost_table,
                    input_text=generation.prompt,
                    output_text=question.expected_answer,
                )
                if cost.status != "estimated" or cost.amount_usd is None:
                    raise ValueError(f"Cost projection is unavailable for {pipeline}/{question_id}.")
                amount = cost.amount_usd
                details = cost.model_dump(mode="json")
            projected[pipeline][question_id] = {"amount_usd": amount, "details": details}
    return projected


def _wins_and_ablations(indexed, costs):
    question_ids = list(indexed["dense"])
    questions = []
    for question_id in question_ids:
        ranks = {pipeline: _first_relevant_rank(indexed[pipeline][question_id]) for pipeline in PIPELINE_NAMES}
        scores = {pipeline: 0.0 if rank is None else 1 / rank for pipeline, rank in ranks.items()}
        best = max(scores.values())
        winners = [pipeline for pipeline in PIPELINE_NAMES if scores[pipeline] == best]
        reranked_row = indexed["reranked"][question_id]
        pre_rank = _first_relevant_rank(reranked_row, field="candidate_chunk_ids")
        pre_score = 0.0 if pre_rank is None else 1 / pre_rank
        reranked_score = scores["reranked"]
        routing_delta = scores["routed"] - scores["reranked"]
        cost_delta = costs["routed"][question_id]["amount_usd"] - costs["reranked"][question_id]["amount_usd"]
        questions.append(
            {
                "question_id": question_id,
                "question": indexed["dense"][question_id]["question"],
                "golden_query_type": "supported",
                "retrieval_wording_cohort": classify_query_type(indexed["dense"][question_id]["question"]),
                "first_relevant_ranks_at_5": ranks,
                "winner": winners[0] if len(winners) == 1 else "tie",
                "winning_pipelines": winners,
                "bm25_beats_dense": scores["bm25"] > scores["dense"],
                "pre_rerank_rank_at_5": pre_rank,
                "reranking_effect": "helps" if reranked_score > pre_score else "hurts" if reranked_score < pre_score else "tie",
                "routing_quality_delta_vs_reranked": routing_delta,
                "routing_harms_quality": routing_delta < 0,
                "routing_cost_delta_usd_vs_reranked": cost_delta,
                "routing_reduces_cost": cost_delta < 0,
            }
        )
    cohort_rows = {}
    for field in ("golden_query_type", "retrieval_wording_cohort"):
        groups = defaultdict(list)
        for row in questions:
            groups[row[field]].append(row)
        cohort_rows[field] = {
            cohort: {
                "question_count": len(rows),
                "unique_wins": {pipeline: sum(row["winner"] == pipeline for row in rows) for pipeline in PIPELINE_NAMES},
                "ties": sum(row["winner"] == "tie" for row in rows),
            }
            for cohort, rows in sorted(groups.items())
        }
    return {
        "retrieval_wins": cohort_rows,
        "bm25_beats_dense": [row for row in questions if row["bm25_beats_dense"]],
        "reranking_helps": [row for row in questions if row["reranking_effect"] == "helps"],
        "reranking_hurts": [row for row in questions if row["reranking_effect"] == "hurts"],
        "routing_reduces_cost": [row for row in questions if row["routing_reduces_cost"]],
        "routing_harms_quality": [row for row in questions if row["routing_harms_quality"]],
        "questions": questions,
    }


def build_final_benchmark_report(config):
    """Build the complete five-way benchmark from frozen measured and judged artifacts."""
    inputs = validate_final_benchmark_inputs(config, require_reports=True, require_judgments=False)
    routed_report = _read_json(config.pipelines.routed.retrieval_report_path, "Routed retrieval report")
    routed_by_id = _question_index(routed_report, "Routed retrieval report")
    if set(routed_by_id) != {label.question_id for label in inputs["labels"]}:
        raise ValueError("Routed retrieval report does not exactly cover the final labels.")
    indexed = {**inputs["indexed"], "routed": routed_by_id}
    records = {}
    for pipeline, artifact in config.pipelines.items():
        records[pipeline] = validate_benchmark_judgments(
            load_benchmark_judgments(artifact.judgments_path), config, pipeline
        )
    chunks_by_id = _load_chunks(config.datasets.chunks_path)
    costs = _project_costs(config, indexed, inputs["golden_by_id"], chunks_by_id, inputs["cost_table"])
    pipeline_rows = []
    for pipeline, artifact in config.pipelines.items():
        metrics = (
            routed_report["metrics"]
            if pipeline == "routed"
            else _fixed_metrics(indexed[pipeline])
        )
        latency = numeric_summary(row["latency_ms"] for row in indexed[pipeline].values())
        quality = _quality_summary(records[pipeline])
        amounts = [entry["amount_usd"] for entry in costs[pipeline].values()]
        refusal = (
            routed_report["refusal_correctness"]
            if pipeline == "routed"
            else {"status": "not_applicable", "reason": "fixed retrieval pipelines have no explicit refusal policy"}
        )
        pipeline_rows.append(
            {
                "pipeline": pipeline,
                "label": artifact.label,
                "recall_at_5": metrics["recall_at_5"],
                "mrr_at_5": metrics["mrr_at_5"],
                "faithfulness": quality["mean_faithfulness"],
                "answer_relevance": quality["mean_answer_relevance"],
                "refusal_correctness": refusal,
                "p50_latency_ms": latency["p50"],
                "p95_latency_ms": latency["p95"],
                "estimated_cost_per_query_usd": sum(amounts) / len(amounts),
                "retrieval_question_count": metrics["question_count"],
                "answer_quality_question_count": quality["question_count"],
                "cost_question_count": len(amounts),
                "latency": latency,
                "answer_quality": quality,
                "config_path": str(artifact.config_path),
                "retrieval_report_path": str(artifact.retrieval_report_path),
                "judgments_path": str(artifact.judgments_path),
            }
        )
    ablations = _wins_and_ablations(indexed, costs)
    return {
        "schema_version": FINAL_BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": f"{config.name}@{config.version}",
        "status": config.status,
        "created_at": datetime.now(UTC).isoformat(),
        "measurement_contract": {
            "retrieval_quality": "50 paired reviewed supported questions; all rankings truncated to top 5; MRR is MRR@5",
            "answer_quality": f"cross-provider LLM judge on the same {len(config.answer_quality.sample_question_ids)} explicit supported questions per pipeline; scores are 1-5 rubric means",
            "refusal_correctness": f"routed policy only, across all {len(inputs['adversarial'])} reviewed unsupported/adversarial questions; fixed pipelines are not applicable",
            "latency": "retrieval-only wall-clock measurements including cold start; routed values use documented serial artifact replay",
            "cost": "paired heuristic token projection over all supported retrieval questions using exact prompts and the same verified reference answer",
            "percentiles": "linear interpolation over ordered per-question measurements",
        },
        "datasets": {
            "golden_path": str(config.datasets.golden_path),
            "retrieval_labels_path": str(config.datasets.retrieval_labels_path),
            "adversarial_path": str(config.datasets.adversarial_path),
            "retrieval_questions": len(inputs["labels"]),
            "adversarial_questions": len(inputs["adversarial"]),
            "answer_quality_sample_ids": config.answer_quality.sample_question_ids,
        },
        "providers": {
            "generation": config.answer_quality.generation.model_dump(),
            "judge": config.answer_quality.judge.model_dump(),
            "cost_projection": {
                "provider": config.cost_projection.provider,
                "model": config.cost_projection.model,
                "model_cost_config": str(config.cost_projection.model_cost_config),
                "price_table_id": inputs["cost_table"].identity,
            },
        },
        "pipelines": pipeline_rows,
        "ablations": ablations,
        "mlflow_runs": {},
        "limitations": [
            "Answer-quality judge scores are estimates on a fixed 10-question supported sample, not human ground truth.",
            "Unsupported queries have refusal labels but no relevant-chunk labels, so retrieval Recall/MRR apply only to supported questions.",
            "Routed latency is a serial composition of measured artifacts; STANDARD uses dense top-10 latency as both probe and full-retrieval proxy.",
            "Cost excludes local embedding, sparse retrieval, reranking compute, judge calls, caching, and infrastructure; it estimates generation token charges only.",
            "The routed policy remains draft and its false-refusal behavior is visible in the supported retrieval metrics.",
        ],
    }


def _format_percent(value):
    return f"{100 * value:.1f}%"


def render_final_benchmark_markdown(report):
    lines = [
        "# Final Benchmark and Ablation Run",
        "",
        "## Central result",
        "",
        "| Pipeline | Recall@5 | MRR@5 | Faithfulness (1-5) | Answer relevance (1-5) | Refusal correctness | p50 retrieval | p95 retrieval | Estimated generation cost/query | MLflow run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["pipelines"]:
        refusal = row["refusal_correctness"]
        refusal_text = "N/A" if refusal.get("status") == "not_applicable" else _format_percent(refusal["accuracy"])
        run = report.get("mlflow_runs", {}).get(row["pipeline"])
        run_text = "pending" if not run else f"[{run['run_id'][:8]}]({run['run_url']})"
        lines.append(
            f"| {row['label']} | {_format_percent(row['recall_at_5'])} | {_format_percent(row['mrr_at_5'])} | "
            f"{row['faithfulness']:.2f} | {row['answer_relevance']:.2f} | {refusal_text} | "
            f"{row['p50_latency_ms']:.1f} ms | {row['p95_latency_ms']:.1f} ms | ${row['estimated_cost_per_query_usd']:.6f} | {run_text} |"
        )
    lines.extend(
        [
            "",
            "MRR is intentionally reported as MRR@5 so every pipeline is compared at the same final depth. Fixed retrieval pipelines show refusal correctness as N/A because they do not implement the router's explicit no-answer policy.",
            "",
            "## Ablation counts",
            "",
            f"- BM25 beats dense: {len(report['ablations']['bm25_beats_dense'])} questions.",
            f"- Reranking helps its own RRF-25 candidate order: {len(report['ablations']['reranking_helps'])} questions.",
            f"- Reranking hurts its own RRF-25 candidate order: {len(report['ablations']['reranking_hurts'])} questions.",
            f"- Routing reduces projected generation cost versus always-reranked: {len(report['ablations']['routing_reduces_cost'])} questions.",
            f"- Routing harms top-5 reciprocal-rank quality versus always-reranked: {len(report['ablations']['routing_harms_quality'])} questions.",
            "",
            "## Retrieval wins by query type",
            "",
            "All retrieval-labeled questions are supported, so the golden query-type view has one cohort. The second view uses the predeclared deterministic wording cohorts from the BM25 evaluation; unsupported behavior is evaluated separately as refusal correctness.",
            "",
        ]
    )
    for view, cohorts in report["ablations"]["retrieval_wins"].items():
        lines.extend(
            [
                f"### {view.replace('_', ' ').title()}",
                "",
                "| Cohort | Questions | Dense wins | BM25 wins | Hybrid wins | Reranked wins | Routed wins | Ties |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for cohort, values in cohorts.items():
            wins = values["unique_wins"]
            lines.append(
                f"| {cohort} | {values['question_count']} | {wins['dense']} | {wins['bm25']} | {wins['hybrid']} | "
                f"{wins['reranked']} | {wins['routed']} | {values['ties']} |"
            )
        lines.append("")

    case_sections = (
        ("Cases where BM25 beats dense", "bm25_beats_dense"),
        ("Cases where reranking helps", "reranking_helps"),
        ("Cases where reranking hurts", "reranking_hurts"),
        ("Cases where routing reduces cost", "routing_reduces_cost"),
        ("Cases where routing harms quality", "routing_harms_quality"),
    )
    for title, key in case_sections:
        rows = report["ablations"][key]
        lines.extend([f"## {title}", ""])
        if not rows:
            lines.extend(["No cases in this run.", ""])
            continue
        lines.extend(["| ID | Question | Wording cohort |", "|---|---|---|"])
        for row in rows:
            question = row["question"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['question_id']} | {question} | {row['retrieval_wording_cohort']} |")
        lines.append("")

    lines.extend(["## Measurement contract and limitations", ""])
    for name, description in report["measurement_contract"].items():
        lines.append(f"- {name.replace('_', ' ').title()}: {description}.")
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def write_final_benchmark_artifacts(report, config, overwrite=False):
    paths = (config.output.json_path, config.output.csv_path, config.output.markdown_path)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite final benchmark artifacts: {', '.join(map(str, existing))}")
    atomic_write_text(config.output.json_path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    rows = []
    for pipeline in report["pipelines"]:
        refusal = pipeline["refusal_correctness"]
        run = report.get("mlflow_runs", {}).get(pipeline["pipeline"], {})
        rows.append(
            {
                "pipeline": pipeline["pipeline"],
                "label": pipeline["label"],
                "recall_at_5": pipeline["recall_at_5"],
                "mrr_at_5": pipeline["mrr_at_5"],
                "faithfulness": pipeline["faithfulness"],
                "answer_relevance": pipeline["answer_relevance"],
                "refusal_correctness": "" if refusal.get("status") == "not_applicable" else refusal["accuracy"],
                "p50_latency_ms": pipeline["p50_latency_ms"],
                "p95_latency_ms": pipeline["p95_latency_ms"],
                "estimated_cost_per_query_usd": pipeline["estimated_cost_per_query_usd"],
                "config_path": pipeline["config_path"],
                "retrieval_report_path": pipeline["retrieval_report_path"],
                "judgments_path": pipeline["judgments_path"],
                "mlflow_run_id": run.get("run_id", ""),
                "mlflow_run_url": run.get("run_url", ""),
            }
        )
    fieldnames = list(rows[0])
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(config.output.csv_path, output.getvalue())
    atomic_write_text(config.output.markdown_path, render_final_benchmark_markdown(report))
    return paths


def _sha256_files(paths):
    digest = hashlib.sha256()
    for path in sorted((Path(path) for path in paths), key=str):
        digest.update(str(path).encode())
        with path.open("rb") as input_file:
            while block := input_file.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def log_final_benchmark_runs(config, report, client=None):
    """Log one immutable MLflow run per benchmark row and return exact run links."""
    try:
        if client is None:
            from mlflow import MlflowClient, set_tracking_uri

            set_tracking_uri(config.mlflow.tracking_uri)
            client = MlflowClient()
        experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(config.mlflow.experiment_name)
        else:
            experiment_id = experiment.experiment_id
        references = {}
        timestamp = int(time.time() * 1000)
        rows = {row["pipeline"]: row for row in report["pipelines"]}
        for pipeline, artifact in config.pipelines.items():
            source_paths = [artifact.config_path, artifact.retrieval_report_path, artifact.judgments_path]
            digest = _sha256_files(source_paths)
            matches = client.search_runs(
                [experiment_id],
                filter_string=(
                    f"tags.`ragops.final_benchmark.digest` = '{digest}' and "
                    f"tags.`ragops.final_benchmark.pipeline` = '{pipeline}'"
                ),
                max_results=10,
            )
            finished = [run for run in matches if run.info.status == "FINISHED"]
            if finished:
                run = finished[0]
            else:
                tags = {
                    "mlflow.runName": f"final-benchmark-{pipeline}-{digest[:8]}",
                    "ragops.final_benchmark.id": report["benchmark_id"],
                    "ragops.final_benchmark.pipeline": pipeline,
                    "ragops.final_benchmark.digest": digest,
                    "ragops.measurement.retrieval": "paired_final_labels_top5",
                    "ragops.measurement.answer_quality": "cross_provider_judge_supported_sample",
                }
                run = client.create_run(experiment_id, tags=tags)
                try:
                    from mlflow.entities import Metric, Param

                    row = rows[pipeline]
                    metrics = {
                        "recall_at_5": row["recall_at_5"],
                        "mrr_at_5": row["mrr_at_5"],
                        "faithfulness": row["faithfulness"],
                        "answer_relevance": row["answer_relevance"],
                        "p50_latency_ms": row["p50_latency_ms"],
                        "p95_latency_ms": row["p95_latency_ms"],
                        "estimated_cost_per_query_usd": row["estimated_cost_per_query_usd"],
                    }
                    refusal = row["refusal_correctness"]
                    if refusal.get("accuracy") is not None:
                        metrics["refusal_correctness"] = refusal["accuracy"]
                    client.log_batch(
                        run.info.run_id,
                        metrics=[Metric(key, float(value), timestamp, 0) for key, value in metrics.items()],
                        params=[
                            Param("pipeline", pipeline),
                            Param("config_path", str(artifact.config_path)),
                            Param("retrieval_questions", str(row["retrieval_question_count"])),
                            Param("answer_quality_questions", str(row["answer_quality_question_count"])),
                            Param("cost_questions", str(row["cost_question_count"])),
                            Param("generator", f"{config.answer_quality.generation.provider}/{config.answer_quality.generation.model}"),
                            Param("judge", f"{config.answer_quality.judge.provider}/{config.answer_quality.judge.model}"),
                        ],
                    )
                    for path, directory in (
                        (artifact.config_path, "config"),
                        (artifact.retrieval_report_path, "retrieval"),
                        (artifact.judgments_path, "answer_quality"),
                    ):
                        client.log_artifact(run.info.run_id, str(path), artifact_path=directory)
                    client.set_terminated(run.info.run_id, status="FINISHED")
                except Exception:
                    client.set_terminated(run.info.run_id, status="FAILED")
                    raise
            base = config.mlflow.tracking_uri.rstrip("/")
            references[pipeline] = {
                "experiment_id": str(experiment_id),
                "run_id": run.info.run_id,
                "run_url": f"{base}/#/experiments/{experiment_id}/runs/{run.info.run_id}",
                "artifact_digest_sha256": digest,
            }
        return references, client
    except Exception as error:
        raise RuntimeError(f"Unable to log final benchmark runs to MLflow at {config.mlflow.tracking_uri}: {error}") from error


def attach_final_artifacts_to_runs(config, report, client):
    for reference in report["mlflow_runs"].values():
        for path in (config.output.json_path, config.output.csv_path, config.output.markdown_path):
            client.log_artifact(reference["run_id"], str(path), artifact_path="final_benchmark")


def verify_final_benchmark_runs(config, report, client=None):
    """Verify exact metrics, digests, status, links, and artifacts for all five MLflow runs."""
    try:
        if client is None:
            from mlflow import MlflowClient, set_tracking_uri

            set_tracking_uri(config.mlflow.tracking_uri)
            client = MlflowClient()
        experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
        if experiment is None:
            raise ValueError(f"MLflow experiment does not exist: {config.mlflow.experiment_name}")
        references = report.get("mlflow_runs")
        if not isinstance(references, dict) or set(references) != set(PIPELINE_NAMES):
            raise ValueError("Final report must link exactly five MLflow pipeline runs.")
        rows = {row["pipeline"]: row for row in report["pipelines"]}
        verified = {}
        for pipeline, artifact in config.pipelines.items():
            reference = references[pipeline]
            run = client.get_run(reference["run_id"])
            if str(run.info.experiment_id) != str(experiment.experiment_id) or run.info.status != "FINISHED":
                raise ValueError(f"MLflow run for {pipeline} is not a finished run in the configured experiment.")
            source_paths = [artifact.config_path, artifact.retrieval_report_path, artifact.judgments_path]
            digest = _sha256_files(source_paths)
            expected_tags = {
                "ragops.final_benchmark.id": report["benchmark_id"],
                "ragops.final_benchmark.pipeline": pipeline,
                "ragops.final_benchmark.digest": digest,
            }
            if any(run.data.tags.get(key) != value for key, value in expected_tags.items()):
                raise ValueError(f"MLflow tags or artifact digest differ for {pipeline}.")
            if reference.get("artifact_digest_sha256") != digest:
                raise ValueError(f"Final report artifact digest differs from the {pipeline} source bundle.")
            expected_url = (
                f"{config.mlflow.tracking_uri.rstrip('/')}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"
            )
            if reference.get("run_url") != expected_url:
                raise ValueError(f"Final report MLflow link differs for {pipeline}.")
            row = rows[pipeline]
            metrics = {
                "recall_at_5": row["recall_at_5"],
                "mrr_at_5": row["mrr_at_5"],
                "faithfulness": row["faithfulness"],
                "answer_relevance": row["answer_relevance"],
                "p50_latency_ms": row["p50_latency_ms"],
                "p95_latency_ms": row["p95_latency_ms"],
                "estimated_cost_per_query_usd": row["estimated_cost_per_query_usd"],
            }
            refusal = row["refusal_correctness"]
            if refusal.get("accuracy") is not None:
                metrics["refusal_correctness"] = refusal["accuracy"]
            mismatched = [
                name
                for name, value in metrics.items()
                if name not in run.data.metrics
                or not math.isclose(float(run.data.metrics[name]), float(value), rel_tol=1e-12, abs_tol=1e-12)
            ]
            if mismatched:
                raise ValueError(f"MLflow metrics differ for {pipeline}: {mismatched}.")
            expected_artifacts = {
                "config": {artifact.config_path.name},
                "retrieval": {artifact.retrieval_report_path.name},
                "answer_quality": {artifact.judgments_path.name},
                "final_benchmark": {config.output.json_path.name, config.output.csv_path.name, config.output.markdown_path.name},
            }
            for directory, names in expected_artifacts.items():
                actual = {Path(item.path).name for item in client.list_artifacts(run.info.run_id, directory)}
                if not names <= actual:
                    raise ValueError(f"MLflow artifacts are incomplete for {pipeline}/{directory}: {sorted(names - actual)}.")
            verified[pipeline] = run.info.run_id
        return verified
    except Exception as error:
        raise RuntimeError(f"Unable to verify final benchmark runs in MLflow at {config.mlflow.tracking_uri}: {error}") from error
