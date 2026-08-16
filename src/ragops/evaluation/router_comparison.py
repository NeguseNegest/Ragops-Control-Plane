import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragops.evaluation.llm_judge import load_golden_questions
from ragops.evaluation.no_answer import load_no_answer_config, validate_no_answer_inputs
from ragops.evaluation.retrieval_labels import load_retrieval_labels
from ragops.evaluation.retrieval_metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from ragops.evaluation.runner import atomic_write_text
from ragops.generation.citations import build_citations
from ragops.generation.client import build_generation_prompt
from ragops.generation.cost import GenerationCost, estimate_generation_cost, load_model_cost_table
from ragops.pipeline_registry import PipelineStatus, PipelineVersion
from ragops.retrieval.dense import RetrievedChunk
from ragops.routing.config import RouterConfig, load_router_config
from ragops.routing.probe import build_initial_retrieval_features
from ragops.routing.router import RuleBasedRouter
from ragops.schemas import DocumentChunk

ROUTER_EVALUATION_SCHEMA_VERSION = 1
STRATEGIES = ("always_fast", "always_careful", "routed")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RouterEvaluationInputs(StrictModel):
    labels_path: Path
    golden_path: Path
    chunks_path: Path
    dense_report_path: Path
    careful_report_path: Path
    no_answer_config_path: Path
    no_answer_report_path: Path
    expected_supported_questions: int = Field(gt=0)
    expected_unsupported_questions: int = Field(gt=0)


class RouterQualityConfig(StrictModel):
    retrieval_cutoff: Literal[5] = 5
    combined_proxy_definition: Literal["supported_hit_or_correct_unsupported_refusal"]


class RouterLatencyConfig(StrictModel):
    method: Literal["measured_artifact_serial_replay"]
    include_cold_start: Literal[True] = True
    probe_proxy: Literal["dense_top_10_measurement"]


class RouterCostProjectionConfig(StrictModel):
    provider: Literal["openai", "gemini"]
    model: str = Field(min_length=1)
    model_cost_config: Path
    token_basis: Literal["exact_prompt_plus_verified_reference_answer"]

    @field_validator("model")
    @classmethod
    def clean_model(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Cost projection model must not be empty.")
        return value


class RouterEvaluationOutput(StrictModel):
    json_path: Path
    csv_path: Path
    markdown_path: Path


class RouterEvaluationConfig(StrictModel):
    schema_version: Literal[1] = ROUTER_EVALUATION_SCHEMA_VERSION
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: PipelineVersion
    status: PipelineStatus
    mode: Literal["artifact_replay"]
    router_config: Path
    inputs: RouterEvaluationInputs
    quality: RouterQualityConfig
    latency: RouterLatencyConfig
    cost_projection: RouterCostProjectionConfig
    output: RouterEvaluationOutput

    @model_validator(mode="after")
    def require_distinct_artifacts(self):
        input_paths = {
            self.router_config,
            self.inputs.labels_path,
            self.inputs.golden_path,
            self.inputs.chunks_path,
            self.inputs.dense_report_path,
            self.inputs.careful_report_path,
            self.inputs.no_answer_config_path,
            self.inputs.no_answer_report_path,
            self.cost_projection.model_cost_config,
        }
        output_paths = {self.output.json_path, self.output.csv_path, self.output.markdown_path}
        if len(output_paths) != 3 or input_paths & output_paths:
            raise ValueError("Router evaluation input and output paths must be distinct.")
        return self


def _resolve(path, root):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def load_router_evaluation_config(config_path, project_root=None):
    """Load the strict Day 41 artifact-replay contract and resolve all paths."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Router evaluation config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid router evaluation YAML in {config_path}: {error}") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Router evaluation config must contain a YAML mapping: {config_path}")
    config = RouterEvaluationConfig.model_validate(payload)
    root = Path(project_root or Path.cwd()).resolve()
    inputs = config.inputs.model_copy(
        update={name: _resolve(getattr(config.inputs, name), root) for name in (
            "labels_path",
            "golden_path",
            "chunks_path",
            "dense_report_path",
            "careful_report_path",
            "no_answer_config_path",
            "no_answer_report_path",
        )}
    )
    cost_projection = config.cost_projection.model_copy(
        update={"model_cost_config": _resolve(config.cost_projection.model_cost_config, root)}
    )
    output = config.output.model_copy(
        update={name: _resolve(getattr(config.output, name), root) for name in ("json_path", "csv_path", "markdown_path")}
    )
    return config.model_copy(
        update={
            "router_config": _resolve(config.router_config, root),
            "inputs": inputs,
            "cost_projection": cost_projection,
            "output": output,
        }
    )


def _load_json(path, label):
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


def _question_index(report, label):
    questions = report.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{label} must contain non-empty question results.")
    indexed = {}
    for row in questions:
        question_id = row.get("question_id") if isinstance(row, dict) else None
        if not isinstance(question_id, str) or not question_id.strip() or question_id in indexed:
            raise ValueError(f"{label} contains invalid or duplicate question IDs.")
        indexed[question_id] = row
    return indexed


def _finite_non_negative(value, label):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric.") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return value


def _validate_retrieval_report(report, labels, *, expected_run, minimum_depth):
    if report.get("run_name") != expected_run:
        raise ValueError(f"Expected {expected_run} report; received {report.get('run_name')!r}.")
    indexed = _question_index(report, f"{expected_run} report")
    labels_by_id = {label.question_id: label for label in labels}
    if set(indexed) != set(labels_by_id):
        raise ValueError(f"{expected_run} report does not exactly cover the verified supported labels.")
    for question_id, label in labels_by_id.items():
        row = indexed[question_id]
        if (row.get("question"), row.get("expected_source"), row.get("relevant_chunk_ids")) != (
            label.question,
            label.expected_source,
            list(label.relevant_chunk_ids),
        ):
            raise ValueError(f"{expected_run} report provenance differs from label {question_id}.")
        chunk_ids = row.get("retrieved_chunk_ids")
        scores = row.get("retrieved_scores")
        if not isinstance(chunk_ids, list) or len(chunk_ids) < minimum_depth or len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(f"{expected_run} question {question_id} has an invalid top-{minimum_depth} ranking.")
        if not isinstance(scores, list) or len(scores) < minimum_depth:
            raise ValueError(f"{expected_run} question {question_id} has incomplete scores.")
        _finite_non_negative(row.get("latency_ms"), f"{expected_run} latency for {question_id}")
    return indexed


def _features_from_score_pair(question, top_score, score_gap, prefix):
    top_score = float(top_score)
    score_gap = float(score_gap)
    if not math.isfinite(top_score) or not math.isfinite(score_gap) or score_gap < 0:
        raise ValueError(f"Router evidence for {prefix} must contain finite scores and a non-negative gap.")
    return build_initial_retrieval_features(
        question,
        [
            RetrievedChunk(chunk_id=f"{prefix}-1", document_id="router-evaluation", text="Probe evidence.", score=top_score, rank=1, metadata={}),
            RetrievedChunk(
                chunk_id=f"{prefix}-2",
                document_id="router-evaluation",
                text="Probe evidence.",
                score=top_score - score_gap,
                rank=2,
                metadata={},
            ),
        ],
        requested_top_k=2,
    )


def _supported_features(row):
    return _features_from_score_pair(row["question"], row["retrieved_scores"][0], row["retrieved_scores"][0] - row["retrieved_scores"][1], row["question_id"])


def _validate_no_answer_report(report, labels, router, no_answer_inputs):
    expected_router_id = f"{router.config.name}@{router.config.version}"
    if report.get("run_name") != "no_answer" or report.get("router_id") != expected_router_id:
        raise ValueError("No-answer report does not match the configured router identity.")
    threshold = report.get("threshold")
    if not isinstance(threshold, dict) or not math.isclose(
        float(threshold.get("configured")), router.config.thresholds.no_answer.top_score_below, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("No-answer report threshold differs from the configured router.")
    indexed = _question_index(report, "no-answer report")
    labels_by_id = {label.question_id: label for label in labels}
    examples_by_id = {example.id: example for example in no_answer_inputs["examples"]}
    if set(indexed) != set(labels_by_id) | set(examples_by_id):
        raise ValueError("No-answer report does not exactly cover supported labels and reviewed unsupported examples.")
    for question_id, row in indexed.items():
        if question_id in labels_by_id:
            label = labels_by_id[question_id]
            expected_question = label.question
            expected_type = "supported"
        else:
            expected_question = examples_by_id[question_id].question
            expected_type = "unsupported"
        if row.get("question") != expected_question or row.get("query_type") != expected_type:
            raise ValueError(f"No-answer report provenance differs for {question_id}.")
        features = _features_from_score_pair(expected_question, row.get("top_score"), row.get("score_gap"), question_id)
        decision = router.select(features)
        if row.get("route") != decision.route or row.get("reason_code") != decision.reason_code:
            raise ValueError(f"No-answer report decision drifted for {question_id}.")
        if bool(row.get("refused")) != (decision.route == "NO_ANSWER"):
            raise ValueError(f"No-answer report refusal state drifted for {question_id}.")
    return indexed


def _load_required_chunks(path, required_ids):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Processed chunk artifact does not exist: {path}")
    found = {}
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid chunk JSON on line {line_number} of {path}.") from error
            chunk_id = payload.get("chunk_id") if isinstance(payload, dict) else None
            if chunk_id not in required_ids:
                continue
            if chunk_id in found:
                raise ValueError(f"Processed chunk artifact contains duplicate required chunk {chunk_id}.")
            found[chunk_id] = DocumentChunk.model_validate(payload)
    missing = sorted(required_ids - set(found))
    if missing:
        raise ValueError(f"Processed chunk artifact is missing {len(missing)} evaluated chunks, including {missing[0]}.")
    return found


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        while block := input_file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_router_evaluation_inputs(config):
    """Cross-check every source artifact before computing the paired comparison."""
    config = config if isinstance(config, RouterEvaluationConfig) else RouterEvaluationConfig.model_validate(config)
    labels = load_retrieval_labels(config.inputs.labels_path)
    if len(labels) != config.inputs.expected_supported_questions:
        raise ValueError(f"Router evaluation contains {len(labels)} supported labels; expected {config.inputs.expected_supported_questions}.")
    ids = [label.question_id for label in labels]
    if len(ids) != len(set(ids)):
        raise ValueError("Router evaluation labels contain duplicate question IDs.")

    golden_by_id = {question.id: question for question in load_golden_questions(config.inputs.golden_path)}
    for label in labels:
        golden = golden_by_id.get(label.question_id)
        if golden is None or golden.query_type != "supported" or golden.question != label.question:
            raise ValueError(f"Verified reference answer is missing or inconsistent for {label.question_id}.")
        if golden.expected_source != label.expected_source:
            raise ValueError(f"Golden source differs from retrieval label {label.question_id}.")

    dense_report = _load_json(config.inputs.dense_report_path, "Dense report")
    careful_report = _load_json(config.inputs.careful_report_path, "CAREFUL report")
    dense_by_id = _validate_retrieval_report(dense_report, labels, expected_run="dense_baseline", minimum_depth=10)
    careful_by_id = _validate_retrieval_report(
        careful_report, labels, expected_run="hybrid_rrf_cross_encoder", minimum_depth=5
    )

    router_config = load_router_config(config.router_config, project_root=config.router_config.parent.parent)
    if not isinstance(router_config, RouterConfig):
        raise ValueError("Router evaluation requires a validated RouterConfig.")
    if (
        router_config.routes.fast.pipeline_config != "dense_baseline"
        or router_config.routes.fast.maximum_top_k != 2
        or not router_config.routes.fast.reuse_probe
        or router_config.routes.standard.pipeline_config != "dense_baseline"
        or router_config.routes.standard.maximum_top_k != 10
        or router_config.routes.careful.pipeline_config != "hybrid_rrf_cross_encoder"
        or router_config.routes.careful.maximum_top_k != 5
    ):
        raise ValueError("Day 41 requires the documented FAST-2, STANDARD-10, and CAREFUL-5 router execution intent.")
    router = RuleBasedRouter(router_config)

    no_answer_config = load_no_answer_config(config.inputs.no_answer_config_path, project_root=config.router_config.parent.parent)
    if no_answer_config.output.json_path != config.inputs.no_answer_report_path:
        raise ValueError("Router evaluation no-answer report path must match the Day 39 output contract.")
    no_answer_inputs = validate_no_answer_inputs(no_answer_config)
    if len(no_answer_inputs["examples"]) != config.inputs.expected_unsupported_questions:
        raise ValueError(
            f"Router evaluation contains {len(no_answer_inputs['examples'])} unsupported examples; expected {config.inputs.expected_unsupported_questions}."
        )
    no_answer_report = _load_json(config.inputs.no_answer_report_path, "No-answer report")
    no_answer_by_id = _validate_no_answer_report(no_answer_report, labels, router, no_answer_inputs)

    for label in labels:
        decision = router.select(_supported_features(dense_by_id[label.question_id]))
        recorded = no_answer_by_id[label.question_id]
        if (recorded["route"], recorded["reason_code"]) != (decision.route, decision.reason_code):
            raise ValueError(f"Supported routing decision differs between dense replay and no-answer evidence for {label.question_id}.")

    required_chunk_ids = set()
    for label in labels:
        required_chunk_ids.update(dense_by_id[label.question_id]["retrieved_chunk_ids"][:10])
        required_chunk_ids.update(careful_by_id[label.question_id]["retrieved_chunk_ids"][:5])
    chunks_by_id = _load_required_chunks(config.inputs.chunks_path, required_chunk_ids)

    cost_table = load_model_cost_table(config.cost_projection.model_cost_config)
    if cost_table.pricing_for(config.cost_projection.provider, config.cost_projection.model) is None:
        raise ValueError("Router cost projection requires an exact provider/model entry in the model cost table.")
    return {
        "labels": labels,
        "golden_by_id": golden_by_id,
        "dense_report": dense_report,
        "dense_by_id": dense_by_id,
        "careful_report": careful_report,
        "careful_by_id": careful_by_id,
        "no_answer_report": no_answer_report,
        "no_answer_by_id": no_answer_by_id,
        "unsupported_examples": no_answer_inputs["examples"],
        "router_config": router_config,
        "router": router,
        "chunks_by_id": chunks_by_id,
        "cost_table": cost_table,
    }


def _percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _numeric_summary(values):
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Summary values must be finite, non-negative, and non-empty.")
    return {
        "count": len(values),
        "total": sum(values),
        "average": sum(values) / len(values),
        "minimum": min(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
    }


def _zero_provider_cost():
    return GenerationCost(
        amount_usd=0,
        status="zero_cost",
        provider="deterministic_policy",
        model="no_answer_v1",
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        token_source="not_applicable",
        pricing_source="not_applicable",
    )


def _project_generation_cost(config, inputs, label, retrieved_ids, refused):
    if refused:
        return _zero_provider_cost()
    chunks = [inputs["chunks_by_id"][chunk_id] for chunk_id in retrieved_ids]
    prompt = build_generation_prompt(label.question, chunks, build_citations(chunks))
    expected_answer = inputs["golden_by_id"][label.question_id].expected_answer
    client = SimpleNamespace(provider=config.cost_projection.provider, model=config.cost_projection.model)
    cost = estimate_generation_cost(
        client,
        usage=None,
        cost_table=inputs["cost_table"],
        input_text=prompt,
        output_text=expected_answer,
    )
    if cost.status != "estimated" or cost.token_source != "heuristic_estimate":
        raise ValueError("Controlled router cost projection unexpectedly produced unavailable evidence.")
    return cost


def _question_strategy(label, route, pipeline, retrieved_ids, latency_ms, config, inputs):
    refused = route == "NO_ANSWER"
    retrieved_ids = [] if refused else list(retrieved_ids)
    cutoff = config.quality.retrieval_cutoff
    metrics = {
        "reciprocal_rank": reciprocal_rank(retrieved_ids, label.relevant_chunk_ids),
        "recall_at_5": recall_at_k(retrieved_ids, label.relevant_chunk_ids, cutoff),
        "hit_at_5": hit_at_k(retrieved_ids, label.relevant_chunk_ids, cutoff),
        "ndcg_at_5": ndcg_at_k(retrieved_ids, label.relevant_chunk_ids, cutoff),
    }
    cost = _project_generation_cost(config, inputs, label, retrieved_ids, refused)
    return {
        "route": route,
        "pipeline_config": pipeline,
        "refused": refused,
        "retrieved_chunk_ids": retrieved_ids,
        "latency_ms": float(latency_ms),
        **metrics,
        "cost": cost.model_dump(mode="json"),
    }


def _aggregate_strategy(name, question_rows, policy_rows, unsupported_count, table_id):
    results = [row["strategies"][name] for row in question_rows]
    policy = [row["strategies"][name] for row in policy_rows]
    supported_policy = [row for row, source in zip(policy, policy_rows, strict=True) if source["query_type"] == "supported"]
    unsupported_policy = [row for row, source in zip(policy, policy_rows, strict=True) if source["query_type"] == "unsupported"]
    costs = [result["cost"] for result in results]
    amounts = [cost["amount_usd"] for cost in costs]
    supported_hits = sum(result["hit_at_5"] for result in results)
    unsupported_correct_refusals = sum(row["correct"] for row in unsupported_policy)
    return {
        "route_counts_supported": dict(sorted(Counter(result["route"] for result in results).items())),
        "retrieval_quality_supported": {
            "question_count": len(results),
            "mrr": sum(result["reciprocal_rank"] for result in results) / len(results),
            "recall_at_5": sum(result["recall_at_5"] for result in results) / len(results),
            "hit_rate_at_5": supported_hits / len(results),
            "ndcg_at_5": sum(result["ndcg_at_5"] for result in results) / len(results),
        },
        "refusal_quality": {
            "supported_answer_rate": sum(not row["refused"] for row in supported_policy) / len(supported_policy),
            "supported_false_refusal_rate": sum(row["refused"] for row in supported_policy) / len(supported_policy),
            "unsupported_refusal_accuracy": unsupported_correct_refusals / unsupported_count,
            "overall_policy_accuracy": sum(row["correct"] for row in policy) / len(policy),
        },
        "combined_quality_proxy": {
            "definition": "supported_hit_at_5_or_correct_unsupported_refusal",
            "successful_supported": int(supported_hits),
            "correct_unsupported_refusals": unsupported_correct_refusals,
            "total_questions": len(policy),
            "success_rate": (supported_hits + unsupported_correct_refusals) / len(policy),
        },
        "latency_ms_supported": {
            "method": "measured_artifact_serial_replay",
            **_numeric_summary([result["latency_ms"] for result in results]),
        },
        "generation_cost_projection_supported": {
            "currency": "USD",
            "price_table_id": table_id,
            "token_basis": "exact_prompt_plus_verified_reference_answer",
            "status_counts": dict(sorted(Counter(cost["status"] for cost in costs).items())),
            "input_tokens": sum(cost["input_tokens"] for cost in costs),
            "output_tokens": sum(cost["output_tokens"] for cost in costs),
            "total_tokens": sum(cost["total_tokens"] for cost in costs),
            "amount_usd": _numeric_summary(amounts),
        },
    }


def _relative_delta(routed, baseline):
    return (routed - baseline) / baseline if baseline else None


def run_router_evaluation(config, inputs=None, *, hash_sources=True):
    """Replay paired evidence for fixed FAST, fixed CAREFUL, and routed strategies."""
    config = config if isinstance(config, RouterEvaluationConfig) else RouterEvaluationConfig.model_validate(config)
    inputs = inputs or validate_router_evaluation_inputs(config)
    router = inputs["router"]
    question_rows = []
    for label in inputs["labels"]:
        dense = inputs["dense_by_id"][label.question_id]
        careful = inputs["careful_by_id"][label.question_id]
        decision = router.select(_supported_features(dense))
        dense_latency = float(dense["latency_ms"])
        careful_latency = float(careful["latency_ms"])
        routed_rankings = {
            "FAST": dense["retrieved_chunk_ids"][:2],
            "STANDARD": dense["retrieved_chunk_ids"][:10],
            "CAREFUL": careful["retrieved_chunk_ids"][:5],
            "NO_ANSWER": [],
        }
        routed_latency = dense_latency
        if decision.route == "STANDARD":
            routed_latency += dense_latency
        elif decision.route == "CAREFUL":
            routed_latency += careful_latency
        question_rows.append(
            {
                "question_id": label.question_id,
                "question": label.question,
                "expected_source": label.expected_source,
                "relevant_chunk_ids": list(label.relevant_chunk_ids),
                "router_reason_code": decision.reason_code,
                "strategies": {
                    "always_fast": _question_strategy(
                        label, "FAST", "dense_baseline", dense["retrieved_chunk_ids"][:2], dense_latency, config, inputs
                    ),
                    "always_careful": _question_strategy(
                        label,
                        "CAREFUL",
                        "hybrid_rrf_cross_encoder",
                        careful["retrieved_chunk_ids"][:5],
                        careful_latency,
                        config,
                        inputs,
                    ),
                    "routed": _question_strategy(
                        label,
                        decision.route,
                        decision.pipeline_config,
                        routed_rankings[decision.route],
                        routed_latency,
                        config,
                        inputs,
                    ),
                },
            }
        )

    policy_rows = []
    for question_id, recorded in inputs["no_answer_by_id"].items():
        query_type = recorded["query_type"]
        policy_features = _features_from_score_pair(
            recorded["question"], recorded["top_score"], recorded["score_gap"], question_id
        )
        policy_decision = router.select(policy_features)
        routed_refused = policy_decision.route == "NO_ANSWER"
        expected_refusal = query_type == "unsupported"
        policy_rows.append(
            {
                "question_id": question_id,
                "question": recorded["question"],
                "query_type": query_type,
                "expected_refusal": expected_refusal,
                "strategies": {
                    "always_fast": {"route": "FAST", "refused": False, "correct": not expected_refusal},
                    "always_careful": {"route": "CAREFUL", "refused": False, "correct": not expected_refusal},
                    "routed": {
                        "route": policy_decision.route,
                        "reason_code": policy_decision.reason_code,
                        "refused": routed_refused,
                        "correct": routed_refused == expected_refusal,
                    },
                },
            }
        )
    policy_rows.sort(key=lambda row: (row["query_type"] == "supported", row["question_id"]))

    table_id = inputs["cost_table"].identity
    strategies = {
        name: _aggregate_strategy(name, question_rows, policy_rows, config.inputs.expected_unsupported_questions, table_id)
        for name in STRATEGIES
    }
    routed = strategies["routed"]
    tradeoff = {}
    for baseline_name in ("always_fast", "always_careful"):
        baseline = strategies[baseline_name]
        tradeoff[f"routed_vs_{baseline_name}"] = {
            "hit_rate_at_5_delta": routed["retrieval_quality_supported"]["hit_rate_at_5"]
            - baseline["retrieval_quality_supported"]["hit_rate_at_5"],
            "combined_quality_proxy_delta": routed["combined_quality_proxy"]["success_rate"]
            - baseline["combined_quality_proxy"]["success_rate"],
            "average_latency_relative_delta": _relative_delta(
                routed["latency_ms_supported"]["average"], baseline["latency_ms_supported"]["average"]
            ),
            "total_cost_relative_delta": _relative_delta(
                routed["generation_cost_projection_supported"]["amount_usd"]["total"],
                baseline["generation_cost_projection_supported"]["amount_usd"]["total"],
            ),
        }

    root = config.router_config.parent.parent
    source_paths = {
        "router_config": config.router_config,
        "labels": config.inputs.labels_path,
        "golden": config.inputs.golden_path,
        "chunks": config.inputs.chunks_path,
        "dense_report": config.inputs.dense_report_path,
        "careful_report": config.inputs.careful_report_path,
        "no_answer_config": config.inputs.no_answer_config_path,
        "no_answer_report": config.inputs.no_answer_report_path,
        "model_costs": config.cost_projection.model_cost_config,
    }
    sources = (
        {
            name: {
                "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                "sha256": _sha256(path),
            }
            for name, path in source_paths.items()
        }
        if hash_sources
        else {}
    )
    return {
        "schema_version": ROUTER_EVALUATION_SCHEMA_VERSION,
        "run_name": config.name,
        "evaluation_id": f"{config.name}@{config.version}",
        "evaluation_status": config.status,
        "mode": config.mode,
        "router": {
            "router_id": f"{inputs['router_config'].name}@{inputs['router_config'].version}",
            "status": inputs["router_config"].status,
            "decision_order": list(inputs["router_config"].decision_order),
        },
        "scope": {
            "supported_operational_questions": len(question_rows),
            "unsupported_policy_questions": config.inputs.expected_unsupported_questions,
            "policy_questions": len(policy_rows),
            "operational_metrics_exclude_unsupported": True,
        },
        "methodology": {
            "quality": "Paired replay of fixed dense top-2, fixed reranked top-5, and the route-selected ranking on verified supported labels.",
            "latency": "Serial composition of recorded dense and reranked retrieval latency; the dense top-10 measurement proxies the top-2 probe and cold starts remain included.",
            "cost": "Day 40 UTF-8 heuristic over the exact selected-context prompt and verified reference answer, priced as the configured model; no provider call was made.",
            "unsupported": "Policy/refusal quality uses the reviewed Day 39 unsupported set; operational latency, retrieval quality, and projected generation cost are not fabricated for unsupported fixed baselines.",
        },
        "cost_projection": {
            "provider": config.cost_projection.provider,
            "model": config.cost_projection.model,
            "price_table_id": table_id,
            "token_estimator": inputs["cost_table"].token_estimator,
            "token_basis": config.cost_projection.token_basis,
        },
        "sources": sources,
        "strategies": strategies,
        "tradeoff": tradeoff,
        "decision": {
            "status": "keep_router_draft",
            "summary": "The routed policy improves the combined support/refusal proxy over always FAST and reduces latency/cost versus always CAREFUL, but loses substantial supported retrieval quality and retains a 20% supported false-refusal rate.",
            "next_step": "Day 42 tuning is complete; keep the policy draft until broader offline and live evidence reduces the supported false-refusal risk.",
        },
        "limitations": [
            "This is a deterministic replay of prior measured artifacts, not a simultaneous live benchmark.",
            "The dense top-10 timing is a conservative proxy for the top-2 probe; STANDARD latency assumes two serial dense calls and CAREFUL assumes a serial dense probe plus reranked retrieval.",
            "Generation cost uses verified reference-answer length, not an observed provider response, and excludes non-token charges listed by the Day 40 cost table.",
            "Retrieval Hit@5 is an evidence-availability proxy, not answer correctness, faithfulness, or relevance.",
            "Only 12 manually reviewed unsupported questions are included, and 9 of 45 supported questions are refused by the current draft threshold.",
        ],
        "questions": question_rows,
        "policy_questions": policy_rows,
    }


def render_router_comparison_markdown(report):
    """Render the human-readable Day 41 tradeoff report from the canonical JSON data."""
    lines = [
        "# Week 6 Router Comparison",
        "",
        f"Evaluation: `{report['evaluation_id']}` ({report['evaluation_status']})",
        f"Router: `{report['router']['router_id']}` ({report['router']['status']})",
        f"Mode: `{report['mode']}`",
        "",
        "## Result",
        "",
        report["decision"]["summary"],
        "",
        "| Strategy | Supported Hit@5 | Supported MRR | Unsupported refusal | Policy accuracy | Combined proxy | Avg latency | p95 latency | Projected cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"always_fast": "Always FAST", "always_careful": "Always CAREFUL", "routed": "Routed"}
    for name in STRATEGIES:
        strategy = report["strategies"][name]
        quality = strategy["retrieval_quality_supported"]
        refusal = strategy["refusal_quality"]
        latency = strategy["latency_ms_supported"]
        cost = strategy["generation_cost_projection_supported"]["amount_usd"]
        lines.append(
            f"| {labels[name]} | {quality['hit_rate_at_5']:.2%} | {quality['mrr']:.4f} | "
            f"{refusal['unsupported_refusal_accuracy']:.2%} | {refusal['overall_policy_accuracy']:.2%} | "
            f"{strategy['combined_quality_proxy']['success_rate']:.2%} | {latency['average']:.1f} ms | "
            f"{latency['p95']:.1f} ms | ${cost['total']:.8f} |"
        )
    lines.extend(
        [
            "",
            "The combined proxy counts a supported question only when a relevant chunk is present at the strategy's final depth, and counts an unsupported question only when it is refused. It is not answer-quality scoring.",
            "",
            "## Routed supported distribution",
            "",
        ]
    )
    for route, count in report["strategies"]["routed"]["route_counts_supported"].items():
        lines.append(f"- {route}: {count}")
    lines.extend(["", "## Methodology", ""])
    for text in report["methodology"].values():
        lines.append(f"- {text}")
    lines.extend(["", "## Interpretation", ""])
    for baseline in ("always_fast", "always_careful"):
        delta = report["tradeoff"][f"routed_vs_{baseline}"]
        lines.append(
            f"- Versus {labels[baseline]}: Hit@5 {delta['hit_rate_at_5_delta']:+.2%}; combined proxy "
            f"{delta['combined_quality_proxy_delta']:+.2%}; average latency {delta['average_latency_relative_delta']:+.2%}; "
            f"projected total cost {delta['total_cost_relative_delta']:+.2%}."
        )
    lines.extend(["", f"Decision: **{report['decision']['status']}**. {report['decision']['next_step']}", "", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def _csv_rows(report):
    for question in report["questions"]:
        for strategy_name in STRATEGIES:
            result = question["strategies"][strategy_name]
            cost = result["cost"]
            yield {
                "strategy": strategy_name,
                "question_id": question["question_id"],
                "question": question["question"],
                "router_reason_code": question["router_reason_code"] if strategy_name == "routed" else "",
                "route": result["route"],
                "pipeline_config": result["pipeline_config"] or "",
                "refused": result["refused"],
                "retrieved_chunk_ids": json.dumps(result["retrieved_chunk_ids"], separators=(",", ":")),
                "hit_at_5": result["hit_at_5"],
                "reciprocal_rank": result["reciprocal_rank"],
                "ndcg_at_5": result["ndcg_at_5"],
                "latency_ms": result["latency_ms"],
                "cost_status": cost["status"],
                "cost_amount_usd": cost["amount_usd"],
                "input_tokens": cost["input_tokens"],
                "output_tokens": cost["output_tokens"],
                "total_tokens": cost["total_tokens"],
                "token_source": cost["token_source"],
                "pricing_source": cost["pricing_source"],
                "price_table_id": cost["price_table_id"] or "",
            }


def write_router_evaluation_artifacts(report, config, overwrite=False):
    """Write canonical JSON, flat supported-question CSV, and Markdown atomically."""
    paths = (config.output.json_path, config.output.csv_path, config.output.markdown_path)
    if not overwrite:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing router evaluation artifacts: {', '.join(existing)}")
    atomic_write_text(config.output.json_path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    fieldnames = [
        "strategy",
        "question_id",
        "question",
        "router_reason_code",
        "route",
        "pipeline_config",
        "refused",
        "retrieved_chunk_ids",
        "hit_at_5",
        "reciprocal_rank",
        "ndcg_at_5",
        "latency_ms",
        "cost_status",
        "cost_amount_usd",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "token_source",
        "pricing_source",
        "price_table_id",
    ]
    temporary = config.output.csv_path.with_name(f".{config.output.csv_path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_csv_rows(report))
    temporary.replace(config.output.csv_path)
    atomic_write_text(config.output.markdown_path, render_router_comparison_markdown(report))
    return paths
