"""Deterministic Day 48 failure analysis over the frozen final benchmark evidence."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragops.evaluation.runner import atomic_write_text

FAILURE_ANALYSIS_SCHEMA_VERSION = 1
PipelineName = Literal["dense", "bm25", "hybrid", "reranked", "routed"]
RankSource = Literal["dense", "bm25", "hybrid", "reranked", "routed", "pre_rerank"]
FailureCategory = Literal[
    "bad_dense_retrieval",
    "lexical_retrieval_miss",
    "hybrid_fusion_failure",
    "reranker_regression",
    "incorrect_refusal",
    "router_mistake",
    "high_latency_query",
    "missing_or_weak_citation",
    "unexpected_generation_behavior",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FailureSources(StrictModel):
    final_benchmark_path: Path
    dense_report_path: Path
    bm25_report_path: Path
    hybrid_report_path: Path
    reranked_report_path: Path
    routed_report_path: Path
    dense_judgments_path: Path
    bm25_judgments_path: Path
    hybrid_judgments_path: Path
    reranked_judgments_path: Path
    routed_judgments_path: Path


class FailureOutputs(StrictModel):
    report_path: Path
    regression_cases_path: Path


class RetrievalRankEvidence(StrictModel):
    kind: Literal["retrieval_rank"]
    pipeline: RankSource
    observed_rank_at_5: int | None
    baseline_pipeline: RankSource
    observed_baseline_rank_at_5: int | None

    @model_validator(mode="after")
    def require_real_regression(self):
        if self.pipeline == self.baseline_pipeline:
            raise ValueError("Retrieval failure evidence requires two different pipeline views.")
        if self.observed_baseline_rank_at_5 is None:
            raise ValueError("The comparison pipeline must retrieve a relevant chunk at top five.")
        if self.observed_rank_at_5 is not None and self.observed_rank_at_5 <= self.observed_baseline_rank_at_5:
            raise ValueError("Configured retrieval evidence does not describe a worse rank.")
        return self


class RouteDecisionEvidence(StrictModel):
    kind: Literal["route_decision"]
    dataset: Literal["supported", "adversarial"]
    observed_route: Literal["FAST", "STANDARD", "CAREFUL", "NO_ANSWER"]
    observed_reason_code: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_route_failure(self):
        if self.dataset == "supported" and self.observed_route != "NO_ANSWER":
            raise ValueError("A supported route failure must record an observed NO_ANSWER route.")
        if self.dataset == "adversarial" and self.observed_route == "NO_ANSWER":
            raise ValueError("An adversarial route failure must record a non-refusal route.")
        return self


class LatencyEvidence(StrictModel):
    kind: Literal["latency"]
    pipeline: Literal["reranked", "routed"]
    minimum_latency_ms: float = Field(gt=0)


class JudgmentEvidence(StrictModel):
    kind: Literal["judgment"]
    pipeline: PipelineName
    observed_behavior: Literal["answer", "refusal"]
    faithfulness: int = Field(ge=1, le=5)
    answer_relevance: int = Field(ge=1, le=5)
    refusal_verdict: Literal["correct", "incorrect", "not_applicable"]


FailureEvidence = Annotated[
    RetrievalRankEvidence | RouteDecisionEvidence | LatencyEvidence | JudgmentEvidence,
    Field(discriminator="kind"),
]


class FailureCaseConfig(StrictModel):
    id: str = Field(pattern=r"^day48-[0-9]{3}$")
    category: FailureCategory
    severity: Literal["high", "medium"]
    question_id: str = Field(min_length=1)
    expected_behavior: str = Field(min_length=20)
    actual_behavior: str = Field(min_length=20)
    root_cause: str = Field(min_length=30)
    affected_component: str = Field(min_length=3)
    proposed_fix: str = Field(min_length=30)
    regression_test: bool
    evidence: FailureEvidence

    @field_validator(
        "question_id",
        "expected_behavior",
        "actual_behavior",
        "root_cause",
        "affected_component",
        "proposed_fix",
    )
    @classmethod
    def clean_text(cls, value):
        return value.strip()


class FailureAcceptance(StrictModel):
    minimum_failures: int = Field(ge=10)
    maximum_failures: int = Field(le=20)
    minimum_regression_cases: int = Field(gt=0)
    required_categories: list[FailureCategory] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.minimum_failures > self.maximum_failures:
            raise ValueError("Failure-analysis minimum cannot exceed its maximum.")
        if len(self.required_categories) != len(set(self.required_categories)):
            raise ValueError("Required failure categories must be unique.")
        return self


class FailureAnalysisConfig(StrictModel):
    schema_version: Literal[1] = FAILURE_ANALYSIS_SCHEMA_VERSION
    analysis_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_@.-]*$")
    status: Literal["reviewed"]
    reviewer: str = Field(min_length=1)
    reviewed_on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    sources: FailureSources
    outputs: FailureOutputs
    acceptance: FailureAcceptance
    cases: list[FailureCaseConfig]

    @model_validator(mode="after")
    def validate_cases(self):
        case_ids = [case.id for case in self.cases]
        targets = [(case.question_id, case.category) for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Failure case IDs must be unique.")
        if len(targets) != len(set(targets)):
            raise ValueError("Question/category failure targets must be unique.")
        count = len(self.cases)
        if not self.acceptance.minimum_failures <= count <= self.acceptance.maximum_failures:
            raise ValueError("Configured failure count is outside the accepted 10-20 range.")
        regression_count = sum(case.regression_test for case in self.cases)
        if regression_count < self.acceptance.minimum_regression_cases:
            raise ValueError("Too few selected failures are marked for the regression suite.")
        categories = {case.category for case in self.cases}
        missing = set(self.acceptance.required_categories) - categories
        if missing:
            raise ValueError(f"Required failure categories are missing: {sorted(missing)}")
        return self


def _resolve_path(path, project_root):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(project_root) / path).resolve()


def load_failure_analysis_config(config_path, project_root=None):
    """Load and resolve the strict Day 48 analysis contract."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Failure-analysis config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid failure-analysis YAML in {config_path}: {error}") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Failure-analysis config must contain a YAML mapping: {config_path}")
    config = FailureAnalysisConfig.model_validate(payload)
    root = Path(project_root or Path.cwd()).resolve()
    sources = config.sources.model_copy(
        update={field: _resolve_path(value, root) for field, value in config.sources.model_dump().items()}
    )
    outputs = config.outputs.model_copy(
        update={field: _resolve_path(value, root) for field, value in config.outputs.model_dump().items()}
    )
    return config.model_copy(update={"sources": sources, "outputs": outputs})


def _load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Failure-analysis source does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON failure-analysis source {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Failure-analysis JSON source must be an object: {path}")
    return value


def _load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Failure-analysis source does not exist: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Failure-analysis JSONL row must be an object: {path}:{line_number}")
        rows.append(row)
    return rows


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_rows(report, section, path):
    rows = report.get(section)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Failure-analysis source has no non-empty {section} list: {path}")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("question_id"), str):
            raise ValueError(f"Malformed {section} row in failure-analysis source: {path}")
        if row["question_id"] in indexed:
            raise ValueError(f"Duplicate question ID in {path}: {row['question_id']}")
        indexed[row["question_id"]] = row
    return indexed


def _relevant_rank(row, retrieved_key="retrieved_chunk_ids", cutoff=5):
    relevant = set(row.get("relevant_chunk_ids", []))
    retrieved = row.get(retrieved_key)
    if not relevant or not isinstance(retrieved, list):
        raise ValueError(f"Question {row.get('question_id')} lacks ranked relevance evidence.")
    return next((rank for rank, chunk_id in enumerate(retrieved[:cutoff], start=1) if chunk_id in relevant), None)


def _load_sources(config):
    source_paths = config.sources.model_dump()
    loaded = {name: _load_json(path) for name, path in source_paths.items() if name.endswith("_report_path") or name == "final_benchmark_path"}
    benchmark = loaded["final_benchmark_path"]
    if benchmark.get("benchmark_id") != "final_benchmark@1.0.0" or benchmark.get("status") != "evaluated":
        raise ValueError("Day 48 requires the completed final_benchmark@1.0.0 evidence.")
    reports = {
        pipeline: loaded[f"{pipeline}_report_path"]
        for pipeline in ("dense", "bm25", "hybrid", "reranked", "routed")
    }
    question_rows = {
        pipeline: _index_rows(report, "questions", getattr(config.sources, f"{pipeline}_report_path"))
        for pipeline, report in reports.items()
    }
    policy_rows = _index_rows(reports["routed"], "policy_questions", config.sources.routed_report_path)
    judgments = {}
    for pipeline in ("dense", "bm25", "hybrid", "reranked", "routed"):
        path = getattr(config.sources, f"{pipeline}_judgments_path")
        judgments[pipeline] = _index_rows({"judgments": _load_jsonl(path)}, "judgments", path)
    return source_paths, reports, question_rows, policy_rows, judgments


def _retrieval_observation(case, evidence, question_rows):
    def row_and_rank(source):
        pipeline = "reranked" if source == "pre_rerank" else source
        row = question_rows[pipeline].get(case.question_id)
        if row is None:
            raise ValueError(f"Question {case.question_id} is absent from {pipeline} retrieval evidence.")
        key = "candidate_chunk_ids" if source == "pre_rerank" else "retrieved_chunk_ids"
        return row, _relevant_rank(row, retrieved_key=key)

    row, rank = row_and_rank(evidence.pipeline)
    baseline_row, baseline_rank = row_and_rank(evidence.baseline_pipeline)
    if row.get("question") != baseline_row.get("question"):
        raise ValueError(f"Retrieval evidence question text differs for {case.question_id}.")
    if rank != evidence.observed_rank_at_5 or baseline_rank != evidence.observed_baseline_rank_at_5:
        raise ValueError(
            f"Retrieval evidence drift for {case.id}: expected ranks "
            f"{evidence.pipeline}={evidence.observed_rank_at_5}, "
            f"{evidence.baseline_pipeline}={evidence.observed_baseline_rank_at_5}; "
            f"found {rank} and {baseline_rank}."
        )
    return row["question"], {
        "kind": evidence.kind,
        "pipeline": evidence.pipeline,
        "rank_at_5": rank,
        "baseline_pipeline": evidence.baseline_pipeline,
        "baseline_rank_at_5": baseline_rank,
    }


def _route_observation(case, evidence, question_rows, policy_rows):
    source = question_rows["routed"] if evidence.dataset == "supported" else policy_rows
    row = source.get(case.question_id)
    if row is None:
        raise ValueError(f"Question {case.question_id} is absent from routed {evidence.dataset} evidence.")
    if row.get("route") != evidence.observed_route or row.get("reason_code") != evidence.observed_reason_code:
        raise ValueError(f"Route evidence drift for {case.id}.")
    if evidence.dataset == "adversarial" and row.get("correct") is not False:
        raise ValueError(f"Adversarial route evidence is no longer an incorrect refusal decision: {case.id}")
    return row["question"], {
        "kind": evidence.kind,
        "dataset": evidence.dataset,
        "route": row["route"],
        "reason_code": row["reason_code"],
        "top_score": row.get("top_score"),
        "score_gap": row.get("score_gap"),
    }


def _latency_observation(case, evidence, question_rows):
    row = question_rows[evidence.pipeline].get(case.question_id)
    if row is None:
        raise ValueError(f"Question {case.question_id} is absent from {evidence.pipeline} latency evidence.")
    latency = row.get("latency_ms")
    if not isinstance(latency, (int, float)) or isinstance(latency, bool) or not math.isfinite(latency):
        raise ValueError(f"Latency evidence is invalid for {case.id}.")
    if latency < evidence.minimum_latency_ms:
        raise ValueError(
            f"Latency evidence drift for {case.id}: {latency:.3f} ms is below "
            f"{evidence.minimum_latency_ms:.3f} ms."
        )
    return row["question"], {
        "kind": evidence.kind,
        "pipeline": evidence.pipeline,
        "latency_ms": latency,
        "minimum_latency_ms": evidence.minimum_latency_ms,
    }


def _judgment_observation(case, evidence, judgments):
    row = judgments[evidence.pipeline].get(case.question_id)
    if row is None:
        raise ValueError(f"Question {case.question_id} is absent from {evidence.pipeline} judgment evidence.")
    judgment = row.get("automatic_judgment", {})
    faithfulness = judgment.get("faithfulness", {}).get("score")
    relevance = judgment.get("answer_relevance", {}).get("score")
    refusal = judgment.get("refusal_correctness", {})
    observed = refusal.get("observed_behavior")
    verdict = refusal.get("verdict")
    expected = (evidence.faithfulness, evidence.answer_relevance, evidence.observed_behavior, evidence.refusal_verdict)
    actual = (faithfulness, relevance, observed, verdict)
    if actual != expected:
        raise ValueError(f"Judgment evidence drift for {case.id}: expected {expected}; found {actual}.")
    return row["question"], {
        "kind": evidence.kind,
        "pipeline": evidence.pipeline,
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "observed_behavior": observed,
        "refusal_verdict": verdict,
        "generated_answer": row.get("generated_answer"),
    }


def _verify_case(case, question_rows, policy_rows, judgments):
    evidence = case.evidence
    if isinstance(evidence, RetrievalRankEvidence):
        query, observation = _retrieval_observation(case, evidence, question_rows)
    elif isinstance(evidence, RouteDecisionEvidence):
        query, observation = _route_observation(case, evidence, question_rows, policy_rows)
    elif isinstance(evidence, LatencyEvidence):
        query, observation = _latency_observation(case, evidence, question_rows)
    else:
        query, observation = _judgment_observation(case, evidence, judgments)
    return {
        "schema_version": FAILURE_ANALYSIS_SCHEMA_VERSION,
        "id": case.id,
        "category": case.category,
        "severity": case.severity,
        "question_id": case.question_id,
        "query": query,
        "expected_behavior": case.expected_behavior,
        "actual_behavior": case.actual_behavior,
        "root_cause": case.root_cause,
        "affected_component": case.affected_component,
        "proposed_fix": case.proposed_fix,
        "regression_test": case.regression_test,
        "evidence": observation,
    }


def _regression_record(case, analysis_id):
    return {
        "schema_version": FAILURE_ANALYSIS_SCHEMA_VERSION,
        "id": f"regression-{case['id']}",
        "analysis_id": analysis_id,
        "source_failure_id": case["id"],
        "question_id": case["question_id"],
        "query": case["query"],
        "category": case["category"],
        "severity": case["severity"],
        "affected_component": case["affected_component"],
        "expected_behavior": case["expected_behavior"],
        "forbidden_behavior": case["actual_behavior"],
        "evidence_guard": case["evidence"],
        "proposed_fix": case["proposed_fix"],
        "provenance": "Day 47 final_benchmark@1.0.0 measured evidence",
        "review_status": "verified",
    }


def build_failure_analysis(config):
    """Verify curated failures against Day 47 and construct deterministic outputs."""
    source_paths, _, question_rows, policy_rows, judgments = _load_sources(config)
    cases = [_verify_case(case, question_rows, policy_rows, judgments) for case in config.cases]
    regressions = [_regression_record(case, config.analysis_id) for case in cases if case["regression_test"]]
    category_counts = dict(sorted(Counter(case["category"] for case in cases).items()))
    component_counts = dict(sorted(Counter(case["affected_component"] for case in cases).items()))
    source_artifacts = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in sorted(source_paths.items())
    }
    analysis = {
        "schema_version": FAILURE_ANALYSIS_SCHEMA_VERSION,
        "analysis_id": config.analysis_id,
        "status": "complete",
        "review": {"reviewer": config.reviewer, "reviewed_on": config.reviewed_on},
        "summary": {
            "failure_count": len(cases),
            "regression_case_count": len(regressions),
            "non_regression_case_count": len(cases) - len(regressions),
            "category_counts": category_counts,
            "affected_component_counts": component_counts,
        },
        "source_artifacts": source_artifacts,
        "cases": cases,
        "regression_cases": regressions,
        "limitations": [
            "Root-cause statements are engineering diagnoses from frozen artifacts, not controlled causal experiments.",
            "Latency is host- and cold-start-dependent; the latency outlier is retained for analysis but not selected as a deterministic regression gate.",
            "Generation failures reflect one fixed cross-provider sample and should be re-judged when prompts, models, or evidence change.",
        ],
    }
    if len(cases) < config.acceptance.minimum_failures or len(cases) > config.acceptance.maximum_failures:
        raise ValueError("Verified failure count is outside the configured acceptance range.")
    if len(regressions) < config.acceptance.minimum_regression_cases:
        raise ValueError("Verified regression-case count is below the configured acceptance minimum.")
    return analysis


def _rank_text(value):
    return "miss" if value is None else str(value)


def _evidence_text(evidence):
    if evidence["kind"] == "retrieval_rank":
        return (
            f"{evidence['pipeline']} rank@5={_rank_text(evidence['rank_at_5'])}; "
            f"{evidence['baseline_pipeline']} rank@5={_rank_text(evidence['baseline_rank_at_5'])}"
        )
    if evidence["kind"] == "route_decision":
        return f"{evidence['dataset']} route={evidence['route']}; reason={evidence['reason_code']}"
    if evidence["kind"] == "latency":
        return f"{evidence['pipeline']} latency={evidence['latency_ms']:.1f} ms"
    return (
        f"{evidence['pipeline']} faithfulness={evidence['faithfulness']}/5, "
        f"relevance={evidence['answer_relevance']}/5, behavior={evidence['observed_behavior']}"
    )


def render_failure_analysis_markdown(analysis):
    """Render the reviewed Day 48 report."""
    summary = analysis["summary"]
    lines = [
        "# Failure Analysis and Regression Cases",
        "",
        "## Outcome",
        "",
        f"This review verifies **{summary['failure_count']} real failures** from the completed Day 47 benchmark and promotes **{summary['regression_case_count']}** stable cases into the regression dataset. One host-dependent latency outlier remains analysis-only.",
        "",
        "Root causes below are engineering diagnoses supported by the recorded ranks, routes, timings, and judgments. They are proposed explanations, not claims from a controlled causal experiment.",
        "",
        "## Evidence provenance",
        "",
        "| Logical source | SHA256 |",
        "|---|---|",
    ]
    for name, source in analysis["source_artifacts"].items():
        lines.append(f"| `{name}` | `{source['sha256']}` |")
    lines.extend(
        [
            "",
        "## Failure inventory",
        "",
        "| ID | Severity | Category | Question ID | Component | Evidence | Regression |",
        "|---|---|---|---|---|---|---|",
        ]
    )
    for case in analysis["cases"]:
        lines.append(
            f"| {case['id']} | {case['severity']} | {case['category']} | `{case['question_id']}` | "
            f"{case['affected_component']} | {_evidence_text(case['evidence'])} | "
            f"{'yes' if case['regression_test'] else 'no'} |"
        )
    lines.extend(["", "## Detailed findings", ""])
    for case in analysis["cases"]:
        lines.extend(
            [
                f"### {case['id']} — {case['category']}",
                "",
                f"- **Query:** {case['query']}",
                f"- **Expected behavior:** {case['expected_behavior']}",
                f"- **Actual behavior:** {case['actual_behavior']}",
                f"- **Verified evidence:** {_evidence_text(case['evidence'])}.",
                f"- **Root cause:** {case['root_cause']}",
                f"- **Affected component:** {case['affected_component']}",
                f"- **Proposed fix:** {case['proposed_fix']}",
                f"- **Regression decision:** {'Promoted to `data/eval/regression_cases.jsonl`.' if case['regression_test'] else 'Analysis-only because the measurement is host-dependent.'}",
                "",
            ]
        )
    lines.extend(
        [
            "## Regression-suite contract",
            "",
            (
                "Each promoted JSONL row retains the reviewed expected behavior, the forbidden measured behavior, "
                "an evidence guard, proposed fix, and Day 47 provenance. The validation command reconstructs these "
                "rows from the curated contract and frozen benchmark artifacts; changed ranks, routes, judgments, "
                "missing questions, or manual output edits fail validation."
            ),
            "",
            "This is a regression-case dataset and evidence guard, not a claim that every proposed remediation has already been implemented. Future pipeline candidates should execute these cases and replace the forbidden behavior with the expected behavior before promotion.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in analysis["limitations"])
    return "\n".join(lines) + "\n"


def _jsonl_text(records):
    return "".join(f"{json.dumps(record, ensure_ascii=False, separators=(',', ':'))}\n" for record in records)


def write_failure_analysis_outputs(config, analysis=None, overwrite=False):
    """Write the deterministic Day 48 report and regression dataset."""
    analysis = analysis or build_failure_analysis(config)
    outputs = {
        config.outputs.report_path: render_failure_analysis_markdown(analysis),
        config.outputs.regression_cases_path: _jsonl_text(analysis["regression_cases"]),
    }
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Failure-analysis outputs already exist: {', '.join(map(str, existing))}")
    for path, content in outputs.items():
        atomic_write_text(path, content)
    return analysis


def validate_failure_analysis_outputs(config, analysis=None):
    """Reject stale or manually edited Day 48 outputs."""
    analysis = analysis or build_failure_analysis(config)
    expected = {
        config.outputs.report_path: render_failure_analysis_markdown(analysis),
        config.outputs.regression_cases_path: _jsonl_text(analysis["regression_cases"]),
    }
    for path, content in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"Failure-analysis output does not exist: {path}")
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Failure-analysis output is stale or manually edited: {path}")
    return analysis
