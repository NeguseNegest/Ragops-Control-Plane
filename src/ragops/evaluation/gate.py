import hashlib
import json
import math
import re
import time
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Literal
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from qdrant_client import QdrantClient

from ragops.evaluation.retrieval_metrics import recall_at_k, reciprocal_rank
from ragops.evaluation.runner import load_evaluation_config
from ragops.generation.client import LocalTemplateGenerationClient, generate_answer
from ragops.indexing.qdrant import create_collection, upsert_records
from ragops.pipeline_registry import PipelineVersion
from ragops.retrieval.factory import build_retriever
from ragops.routing.config import load_router_config
from ragops.routing.probe import build_initial_retrieval_features
from ragops.routing.router import RuleBasedRouter

GATE_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
SHA256_PATTERN = r"^[0-9a-f]{64}$"
CITATION_PATTERN = re.compile(r"\[(\d+)]")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GateCandidateConfig(StrictModel):
    pipeline_type: Literal["dense"]
    config_path: Path
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    expected_name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    expected_version: PipelineVersion
    expected_status: Literal["evaluated", "approved"]


class GateRouterConfig(StrictModel):
    config_path: Path
    config_sha256: str = Field(pattern=SHA256_PATTERN)


class GateCorpusConfig(StrictModel):
    records_path: Path
    records_sha256: str = Field(pattern=SHA256_PATTERN)
    collection_name: str = Field(min_length=1)
    vector_size: int = Field(gt=0)

    @field_validator("collection_name")
    @classmethod
    def clean_collection_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Gate corpus collection_name must not be empty.")
        return value


class GateDatasetConfig(StrictModel):
    cases_path: Path
    cases_sha256: str = Field(pattern=SHA256_PATTERN)
    minimum_supported_cases: int = Field(gt=0)
    minimum_unsupported_cases: int = Field(gt=0)
    retrieval_k: int = Field(gt=0)


GenerationMetric = Literal["answer_presence_rate", "citation_coverage", "citation_precision"]


class GateGenerationConfig(StrictModel):
    provider: Literal["template"]
    available_metrics: tuple[GenerationMetric, ...] = Field(min_length=1)

    @field_validator("available_metrics")
    @classmethod
    def require_unique_metrics(cls, values):
        if len(values) != len(set(values)):
            raise ValueError("Gate generation metrics must be unique.")
        required = {"answer_presence_rate", "citation_coverage", "citation_precision"}
        if set(values) != required:
            raise ValueError("The offline gate must declare answer presence, citation coverage, and citation precision exactly once.")
        return values


class GateThresholds(StrictModel):
    baseline_recall_at_k: float = Field(ge=0, le=1)
    minimum_recall_at_k: float = Field(ge=0, le=1)
    maximum_recall_regression: float = Field(ge=0, le=1)
    minimum_mrr: float = Field(ge=0, le=1)
    minimum_answer_presence_rate: float = Field(ge=0, le=1)
    minimum_citation_coverage: float = Field(ge=0, le=1)
    minimum_citation_precision: float = Field(ge=0, le=1)
    minimum_refusal_correctness: float = Field(ge=0, le=1)
    maximum_p95_latency_ms: float = Field(gt=0)
    maximum_error_count: int = Field(ge=0)

    @field_validator(
        "baseline_recall_at_k",
        "minimum_recall_at_k",
        "maximum_recall_regression",
        "minimum_mrr",
        "minimum_answer_presence_rate",
        "minimum_citation_coverage",
        "minimum_citation_precision",
        "minimum_refusal_correctness",
        "maximum_p95_latency_ms",
    )
    @classmethod
    def require_finite_thresholds(cls, value):
        if not math.isfinite(value):
            raise ValueError("Evaluation-gate thresholds must be finite.")
        return value

    @model_validator(mode="after")
    def require_coherent_recall_thresholds(self):
        if self.minimum_recall_at_k > self.baseline_recall_at_k:
            raise ValueError("minimum_recall_at_k cannot exceed baseline_recall_at_k.")
        return self


class GateLatencyCalibration(StrictModel):
    environment: str = Field(min_length=1)
    measured_on: date
    sample_runs: int = Field(gt=0)
    observed_p95_latency_ms: float = Field(ge=0)
    threshold_headroom_reason: str = Field(min_length=1)

    @field_validator("observed_p95_latency_ms")
    @classmethod
    def require_finite_latency(cls, value):
        if not math.isfinite(value):
            raise ValueError("Observed p95 latency must be finite.")
        return value


class EvaluationGateConfig(StrictModel):
    schema_version: Literal[1] = GATE_SCHEMA_VERSION
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: PipelineVersion
    candidate: GateCandidateConfig
    router: GateRouterConfig
    corpus: GateCorpusConfig
    dataset: GateDatasetConfig
    generation: GateGenerationConfig
    thresholds: GateThresholds
    latency_calibration: GateLatencyCalibration

    @model_validator(mode="after")
    def require_latency_headroom(self):
        if self.latency_calibration.observed_p95_latency_ms > self.thresholds.maximum_p95_latency_ms:
            raise ValueError("The configured latency threshold is below its recorded calibration observation.")
        return self


class GateCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    query: str = Field(min_length=1)
    query_type: Literal["supported", "unsupported"]
    expected_behavior: Literal["answer", "refusal"]
    vector: tuple[float, ...] = Field(min_length=1)
    relevant_chunk_ids: tuple[str, ...]

    @field_validator("query")
    @classmethod
    def clean_query(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Gate-case query must not be empty.")
        return value

    @field_validator("vector")
    @classmethod
    def require_finite_vector(cls, values):
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Gate-case vectors must contain only finite values.")
        if not any(value != 0 for value in values):
            raise ValueError("Gate-case vectors must not be all zero.")
        return values

    @field_validator("relevant_chunk_ids")
    @classmethod
    def require_unique_relevance(cls, values):
        if len(values) != len(set(values)):
            raise ValueError("Gate-case relevant_chunk_ids must be unique.")
        return values

    @model_validator(mode="after")
    def require_behavior_shape(self):
        expected = "answer" if self.query_type == "supported" else "refusal"
        if self.expected_behavior != expected:
            raise ValueError(f"{self.query_type} gate cases must expect {expected} behavior.")
        if self.query_type == "supported" and not self.relevant_chunk_ids:
            raise ValueError("Supported gate cases require at least one relevant chunk ID.")
        if self.query_type == "unsupported" and self.relevant_chunk_ids:
            raise ValueError("Unsupported gate cases cannot declare relevant chunk IDs.")
        return self


class GateCorpusRecord(StrictModel):
    chunk_id: str
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    chunk_hash: str = Field(pattern=SHA256_PATTERN)
    embedding: tuple[float, ...] = Field(min_length=1)
    metadata: dict

    @field_validator("chunk_id")
    @classmethod
    def canonicalize_chunk_id(cls, value):
        return str(UUID(value))

    @field_validator("embedding")
    @classmethod
    def require_finite_embedding(cls, values):
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Gate corpus embeddings must contain only finite values.")
        if not any(value != 0 for value in values):
            raise ValueError("Gate corpus embeddings must not be all zero.")
        return values

    @model_validator(mode="after")
    def require_text_hash(self):
        actual = hashlib.sha256(self.text.encode()).hexdigest()
        if actual != self.chunk_hash:
            raise ValueError(f"Gate corpus chunk {self.chunk_id} has a stale text hash.")
        return self


class GateObservation(StrictModel):
    observed_behavior: Literal["answer", "refusal"]
    route: str = Field(min_length=1)
    retrieved_chunk_ids: tuple[str, ...] = ()
    cited_chunk_ids: tuple[str, ...] = ()
    answer: str | None = None

    @model_validator(mode="after")
    def require_behavior_payload(self):
        if self.observed_behavior == "refusal":
            if self.route != "NO_ANSWER" or self.retrieved_chunk_ids or self.cited_chunk_ids or self.answer is not None:
                raise ValueError("Refusal observations cannot contain final retrieval or generation output.")
        elif self.route == "NO_ANSWER" or not self.answer or not self.retrieved_chunk_ids:
            raise ValueError("Answer observations require a non-refusal route, retrieved chunks, and an answer.")
        return self


class GateCaseResult(StrictModel):
    case_id: str
    query_type: Literal["supported", "unsupported"]
    expected_behavior: Literal["answer", "refusal"]
    observed_behavior: Literal["answer", "refusal", "error"]
    route: str | None
    relevant_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    cited_chunk_ids: tuple[str, ...]
    answer: str | None
    latency_ms: float = Field(ge=0)
    error: str | None = None

    @field_validator("latency_ms")
    @classmethod
    def require_finite_latency(cls, value):
        if not math.isfinite(value):
            raise ValueError("Gate-case latency must be finite.")
        return value


class LoadedGate:
    """Validated, hash-pinned inputs needed by one compact gate run."""

    def __init__(self, config, candidate_config, router_config, cases, records, project_root):
        self.config = config
        self.candidate_config = candidate_config
        self.router_config = router_config
        self.cases = tuple(cases)
        self.records = tuple(records)
        self.project_root = Path(project_root)


def _resolve_path(path, project_root):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(project_root) / path).resolve()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file_hash(path, expected, label):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} checksum mismatch: expected {expected}, received {actual}.")


def _read_jsonl(path, model, label):
    rows = []
    with Path(path).open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(model.model_validate(json.loads(line)))
            except Exception as error:
                raise ValueError(f"Invalid {label} row on line {line_number}: {error}") from error
    if not rows:
        raise ValueError(f"{label} must contain at least one row.")
    return rows


def load_evaluation_gate(config_path, project_root=None):
    """Load and cross-validate the complete hash-pinned Day 44 gate."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Evaluation-gate config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid evaluation-gate YAML in {config_path}: {error}") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Evaluation-gate config must contain a YAML mapping.")

    config = EvaluationGateConfig.model_validate(payload)
    root = Path(project_root or Path.cwd()).resolve()
    candidate = config.candidate.model_copy(update={"config_path": _resolve_path(config.candidate.config_path, root)})
    router = config.router.model_copy(update={"config_path": _resolve_path(config.router.config_path, root)})
    corpus = config.corpus.model_copy(update={"records_path": _resolve_path(config.corpus.records_path, root)})
    dataset = config.dataset.model_copy(update={"cases_path": _resolve_path(config.dataset.cases_path, root)})
    config = config.model_copy(update={"candidate": candidate, "router": router, "corpus": corpus, "dataset": dataset})

    _require_file_hash(candidate.config_path, candidate.config_sha256, "Candidate config")
    _require_file_hash(router.config_path, router.config_sha256, "Router config")
    _require_file_hash(corpus.records_path, corpus.records_sha256, "Gate corpus")
    _require_file_hash(dataset.cases_path, dataset.cases_sha256, "Gate dataset")

    candidate_config = load_evaluation_config(candidate.config_path, project_root=root)
    if candidate_config.name != candidate.expected_name or candidate_config.version != candidate.expected_version or candidate_config.status != candidate.expected_status:
        raise ValueError("Candidate config identity or lifecycle status differs from the gate pin.")
    if candidate_config.retriever.type != candidate.pipeline_type:
        raise ValueError("Candidate pipeline type differs from the gate pin.")
    if candidate_config.retriever.collection_name != corpus.collection_name:
        raise ValueError("Candidate and compact corpus must use the same collection name.")
    if candidate_config.retriever.top_k < dataset.retrieval_k:
        raise ValueError("Candidate top_k must be at least the gate retrieval_k.")

    router_config = load_router_config(router.config_path, project_root=root)
    cases = _read_jsonl(dataset.cases_path, GateCase, "evaluation-gate case")
    records = _read_jsonl(corpus.records_path, GateCorpusRecord, "evaluation-gate corpus")

    case_ids = [case.case_id for case in cases]
    queries = [case.query.casefold() for case in cases]
    if len(case_ids) != len(set(case_ids)) or len(queries) != len(set(queries)):
        raise ValueError("Evaluation-gate case IDs and queries must be unique.")
    supported_count = sum(case.query_type == "supported" for case in cases)
    unsupported_count = sum(case.query_type == "unsupported" for case in cases)
    if supported_count < dataset.minimum_supported_cases or unsupported_count < dataset.minimum_unsupported_cases:
        raise ValueError("Evaluation-gate dataset does not meet its supported/unsupported minimums.")
    if any(len(case.vector) != corpus.vector_size for case in cases):
        raise ValueError("Every gate-case vector must match corpus.vector_size.")

    chunk_ids = [record.chunk_id for record in records]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Evaluation-gate corpus chunk IDs must be unique.")
    if any(len(record.embedding) != corpus.vector_size for record in records):
        raise ValueError("Every gate-corpus embedding must match corpus.vector_size.")
    known_chunk_ids = set(chunk_ids)
    for case in cases:
        if not set(case.relevant_chunk_ids).issubset(known_chunk_ids):
            raise ValueError(f"Gate case {case.case_id} references a missing corpus chunk.")

    return LoadedGate(config, candidate_config, router_config, cases, records, root)


def referenced_citation_chunk_ids(answer, citations):
    """Map only citation IDs actually present in an answer back to chunk IDs."""
    citation_map = {citation["citation_id"]: tuple(citation.get("chunk_ids") or ()) for citation in citations}
    referenced_ids = []
    for number in CITATION_PATTERN.findall(answer or ""):
        for chunk_id in citation_map.get(f"[{number}]", ()):
            if chunk_id not in referenced_ids:
                referenced_ids.append(chunk_id)
    return tuple(referenced_ids)


class CompactCandidateRuntime:
    """Run the selected dense config, router, and template generator in memory."""

    def __init__(self, loaded_gate, query_vectors=None):
        self.loaded_gate = loaded_gate
        self.client = QdrantClient(location=":memory:")
        create_collection(self.client, loaded_gate.config.corpus.collection_name, loaded_gate.config.corpus.vector_size)
        upsert_records(self.client, loaded_gate.config.corpus.collection_name, [record.model_dump() for record in loaded_gate.records])
        default_vectors = {case.query: case.vector for case in loaded_gate.cases}
        self.query_vectors = dict(default_vectors if query_vectors is None else query_vectors)
        if set(self.query_vectors) != set(default_vectors):
            raise ValueError("Runtime query vectors must cover every configured gate case exactly once.")
        if any(len(vector) != loaded_gate.config.corpus.vector_size for vector in self.query_vectors.values()):
            raise ValueError("Runtime query vectors must match the configured vector size.")
        if any(not all(math.isfinite(value) for value in vector) for vector in self.query_vectors.values()):
            raise ValueError("Runtime query vectors must be finite.")
        if any(not any(value != 0 for value in vector) for vector in self.query_vectors.values()):
            raise ValueError("Runtime query vectors must not be all zero.")
        self.retriever = build_retriever(loaded_gate.candidate_config, client=self.client, query_embedder=self._embed_query)
        self.router = RuleBasedRouter(loaded_gate.router_config)
        self.generation_client = LocalTemplateGenerationClient()

    def _embed_query(self, query, embedding_model):
        try:
            return list(self.query_vectors[query])
        except KeyError as error:
            raise ValueError(f"The compact gate has no deterministic vector for query: {query}") from error

    def execute(self, case):
        case = case if isinstance(case, GateCase) else GateCase.model_validate(case)
        probe_depth = self.loaded_gate.router_config.probe.top_k
        probe_chunks = self.retriever.retrieve(case.query, top_k=probe_depth)
        features = build_initial_retrieval_features(case.query, probe_chunks, requested_top_k=probe_depth)
        decision = self.router.select(features)
        if decision.route == "NO_ANSWER":
            return GateObservation(observed_behavior="refusal", route=decision.route)

        final_chunks = self.retriever.retrieve(case.query, top_k=self.loaded_gate.candidate_config.retriever.top_k)
        generation = generate_answer(case.query, final_chunks, client=self.generation_client)
        cited_chunk_ids = referenced_citation_chunk_ids(generation.answer, generation.citations)
        return GateObservation(
            observed_behavior="answer",
            route=decision.route,
            retrieved_chunk_ids=tuple(chunk.chunk_id for chunk in final_chunks),
            cited_chunk_ids=cited_chunk_ids,
            answer=generation.answer,
        )

    def close(self):
        self.client.close()


def percentile_nearest_rank(values, percentile):
    """Return the nearest-rank percentile used by the compact latency gate."""
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Latency percentiles require non-empty, finite, non-negative values.")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be greater than zero and at most one.")
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def _case_result(case, observation, latency_ms, error=None):
    if error is not None:
        return GateCaseResult(
            case_id=case.case_id,
            query_type=case.query_type,
            expected_behavior=case.expected_behavior,
            observed_behavior="error",
            route=None,
            relevant_chunk_ids=case.relevant_chunk_ids,
            retrieved_chunk_ids=(),
            cited_chunk_ids=(),
            answer=None,
            latency_ms=latency_ms,
            error=f"{type(error).__name__}: {error}",
        )
    observation = observation if isinstance(observation, GateObservation) else GateObservation.model_validate(observation)
    return GateCaseResult(
        case_id=case.case_id,
        query_type=case.query_type,
        expected_behavior=case.expected_behavior,
        observed_behavior=observation.observed_behavior,
        route=observation.route,
        relevant_chunk_ids=case.relevant_chunk_ids,
        retrieved_chunk_ids=observation.retrieved_chunk_ids,
        cited_chunk_ids=observation.cited_chunk_ids,
        answer=observation.answer,
        latency_ms=latency_ms,
    )


def run_gate_cases(cases, executor, clock=time.perf_counter):
    """Execute every case independently so one runtime error becomes gate evidence."""
    if not callable(executor) or not callable(clock):
        raise ValueError("Gate executor and clock must be callable.")
    results = []
    for case in cases:
        started_at = clock()
        observation = None
        error = None
        try:
            observation = executor(case)
        except Exception as execution_error:
            error = execution_error
        latency_ms = max(0.0, (clock() - started_at) * 1000)
        results.append(_case_result(case, observation, latency_ms, error=error))
    return results


def calculate_gate_metrics(results, retrieval_k):
    """Calculate retrieval, deterministic generation, refusal, latency, and error metrics."""
    supported = [result for result in results if result.query_type == "supported"]
    if not supported:
        raise ValueError("Evaluation-gate metrics require supported cases.")
    retrieval_recalls = [recall_at_k(result.retrieved_chunk_ids, result.relevant_chunk_ids, retrieval_k) for result in supported]
    reciprocal_ranks = [reciprocal_rank(result.retrieved_chunk_ids, result.relevant_chunk_ids) for result in supported]
    relevant_cited = sum(len(set(result.cited_chunk_ids) & set(result.relevant_chunk_ids)) for result in supported)
    relevant_total = sum(len(set(result.relevant_chunk_ids)) for result in supported)
    cited_total = sum(len(set(result.cited_chunk_ids)) for result in supported)
    behavior_correct = sum(result.observed_behavior == result.expected_behavior for result in results)
    latencies = [result.latency_ms for result in results]
    return {
        "retrieval": {
            "evaluated_count": len(supported),
            "k": retrieval_k,
            "recall_at_k": sum(retrieval_recalls) / len(retrieval_recalls),
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        },
        "generation": {
            "evaluated_count": len(supported),
            "answer_presence_rate": sum(result.observed_behavior == "answer" and bool(result.answer) for result in supported) / len(supported),
            "citation_coverage": relevant_cited / relevant_total,
            "citation_precision": relevant_cited / cited_total if cited_total else 0.0,
            "faithfulness": None,
            "faithfulness_status": "not_available_without_external_judge",
        },
        "refusal": {
            "evaluated_count": len(results),
            "correct_count": behavior_correct,
            "correctness": behavior_correct / len(results),
        },
        "latency_ms": {
            "evaluated_count": len(latencies),
            "p50": percentile_nearest_rank(latencies, 0.50),
            "p95": percentile_nearest_rank(latencies, 0.95),
            "maximum": max(latencies),
        },
        "errors": {"count": sum(result.error is not None for result in results)},
    }


def _minimum_check(check_id, actual, threshold, label):
    passed = actual >= threshold
    return {"id": check_id, "passed": passed, "actual": actual, "operator": ">=", "threshold": threshold, "summary": label}


def _maximum_check(check_id, actual, threshold, label):
    passed = actual <= threshold
    return {"id": check_id, "passed": passed, "actual": actual, "operator": "<=", "threshold": threshold, "summary": label}


def evaluate_thresholds(metrics, thresholds):
    """Compare calculated metrics with every configured regression threshold."""
    recall = metrics["retrieval"]["recall_at_k"]
    regression = max(0.0, thresholds.baseline_recall_at_k - recall)
    checks = [
        _minimum_check("retrieval.minimum_recall_at_k", recall, thresholds.minimum_recall_at_k, "retrieval recall"),
        _maximum_check("retrieval.maximum_recall_regression", regression, thresholds.maximum_recall_regression, "recall regression from compact baseline"),
        _minimum_check("retrieval.minimum_mrr", metrics["retrieval"]["mrr"], thresholds.minimum_mrr, "mean reciprocal rank"),
        _minimum_check("generation.minimum_answer_presence_rate", metrics["generation"]["answer_presence_rate"], thresholds.minimum_answer_presence_rate, "supported answer presence"),
        _minimum_check("generation.minimum_citation_coverage", metrics["generation"]["citation_coverage"], thresholds.minimum_citation_coverage, "relevant-evidence citation coverage"),
        _minimum_check("generation.minimum_citation_precision", metrics["generation"]["citation_precision"], thresholds.minimum_citation_precision, "cited-evidence precision"),
        _minimum_check("refusal.minimum_correctness", metrics["refusal"]["correctness"], thresholds.minimum_refusal_correctness, "answer/refusal behavior correctness"),
        _maximum_check("latency.maximum_p95_ms", metrics["latency_ms"]["p95"], thresholds.maximum_p95_latency_ms, "whole-case p95 latency"),
        _maximum_check("errors.maximum_count", metrics["errors"]["count"], thresholds.maximum_error_count, "runtime error count"),
    ]
    return checks


def build_gate_report(loaded_gate, results):
    """Build a JSON-ready report with explicit checks and input provenance."""
    config = loaded_gate.config
    metrics = calculate_gate_metrics(results, config.dataset.retrieval_k)
    checks = evaluate_thresholds(metrics, config.thresholds)
    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "gate_id": f"{config.name}@{config.version}",
        "candidate_pipeline_id": f"{config.candidate.expected_name}@{config.candidate.expected_version}",
        "status": "pass" if passed else "fail",
        "case_count": len(results),
        "metrics": metrics,
        "checks": checks,
        "cases": [result.model_dump(mode="json") for result in results],
        "provenance": {
            "candidate_config_sha256": config.candidate.config_sha256,
            "router_config_sha256": config.router.config_sha256,
            "corpus_sha256": config.corpus.records_sha256,
            "dataset_sha256": config.dataset.cases_sha256,
            "generation_provider": config.generation.provider,
            "available_generation_metrics": list(config.generation.available_metrics),
            "latency_calibration": config.latency_calibration.model_dump(mode="json"),
        },
    }


def execute_evaluation_gate(loaded_gate, query_vectors=None, clock=time.perf_counter):
    """Execute the real compact runtime and return its threshold report."""
    runtime = CompactCandidateRuntime(loaded_gate, query_vectors=query_vectors)
    try:
        results = run_gate_cases(loaded_gate.cases, runtime.execute, clock=clock)
    finally:
        runtime.close()
    return build_gate_report(loaded_gate, results)


def render_gate_summary(report):
    """Render one stable, human-readable pass/fail summary."""
    lines = [
        f"Evaluation gate: {report['gate_id']}",
        f"Candidate: {report['candidate_pipeline_id']}",
        f"Cases: {report['case_count']}",
        "",
    ]
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        actual = check["actual"]
        threshold = check["threshold"]
        if isinstance(actual, float):
            actual = f"{actual:.6f}"
        if isinstance(threshold, float):
            threshold = f"{threshold:.6f}"
        lines.append(f"[{status}] {check['id']}: {actual} {check['operator']} {threshold} ({check['summary']})")
    passed_count = sum(check["passed"] for check in report["checks"])
    lines.extend(["", f"Overall: {report['status'].upper()} ({passed_count}/{len(report['checks'])} checks passed)"])
    if report["metrics"]["generation"]["faithfulness"] is None:
        lines.append("Note: faithfulness is not scored by the offline template gate; external-judge evidence remains a separate evaluation.")
    return "\n".join(lines)


def gate_exit_code(report):
    """Map a validated gate outcome to a shell-compatible status code."""
    if not isinstance(report, Mapping) or report.get("status") not in {"pass", "fail"}:
        raise ValueError("Gate report status must be 'pass' or 'fail'.")
    return 0 if report["status"] == "pass" else 1
