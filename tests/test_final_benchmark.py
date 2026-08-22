import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragops.evaluation.final_benchmark import (
    _probe_features,
    build_routed_report,
    load_final_benchmark_config,
    numeric_summary,
    run_answer_quality_pipeline,
    validate_benchmark_judgments,
    validate_final_benchmark_inputs,
)
from ragops.evaluation.llm_judge import load_golden_questions
from ragops.evaluation.retrieval_labels import RetrievalLabel
from ragops.generation.client import LocalTemplateGenerationClient
from ragops.routing.config import load_router_config
from ragops.routing.router import RuleBasedRouter
from ragops.schemas import DocumentChunk
from scripts.final_benchmark import RetryingGenerationClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FixedJudge:
    def generate(self, prompt):
        assert "FAITHFULNESS RUBRIC" in prompt
        return json.dumps(
            {
                "faithfulness": {"score": 5, "rationale": "Every factual claim is supported by the supplied evidence."},
                "answer_relevance": {"score": 4, "rationale": "The response directly addresses the supported documentation question."},
                "refusal_correctness": {
                    "observed_behavior": "answer",
                    "verdict": "not_applicable",
                    "rationale": "A supported question should be answered rather than refused.",
                },
            }
        )


class FlakyClient:
    provider = "gemini"
    model = "test-model"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("429 too_many_requests; please retry in 0.5s")
        return "completed"

    def generate_with_metadata(self, prompt):
        return self.generate(prompt)


def _label(question_id, question, relevant):
    return RetrievalLabel(
        question_id=question_id,
        question=question,
        relevant_chunk_ids=[relevant],
        expected_source="docs/example.md",
        metadata={"label_method": "manual", "review_status": "verified", "reviewed_by": "pytest"},
    )


def _row(label, ids, scores, latency):
    return {
        "question_id": label.question_id,
        "question": label.question,
        "expected_source": label.expected_source,
        "relevant_chunk_ids": list(label.relevant_chunk_ids),
        "retrieved_chunk_ids": ids,
        "retrieved_scores": scores,
        "latency_ms": latency,
    }


def test_checked_in_final_benchmark_contract_covers_frozen_day46_outputs():
    config = load_final_benchmark_config(PROJECT_ROOT / "configs/final_benchmark.yaml", project_root=PROJECT_ROOT)
    inputs = validate_final_benchmark_inputs(config, require_reports=False)

    assert len(inputs["labels"]) == 50
    assert len(inputs["adversarial"]) == 30
    assert len(config.answer_quality.sample_question_ids) == 10
    assert set(config.answer_quality.sample_question_ids) <= {label.question_id for label in inputs["labels"]}
    assert config.answer_quality.generation.provider != config.answer_quality.judge.provider


def test_numeric_summary_uses_linear_p50_and_p95():
    summary = numeric_summary([0, 10, 20, 30, 40])
    assert summary["p50"] == 20
    assert summary["p95"] == pytest.approx(38)
    with pytest.raises(ValueError, match="finite"):
        numeric_summary([1, float("nan")])


def test_routed_report_replays_supported_routes_and_measures_all_policy_rows():
    config = load_final_benchmark_config(PROJECT_ROOT / "configs/final_benchmark.yaml", project_root=PROJECT_ROOT)
    fast = _label("q-fast", "What is FastAPI?", "fast-relevant")
    careful = _label("q-careful", "Explain why retrieval differs between systems.", "careful-relevant")
    dense = {
        fast.question_id: _row(fast, ["fast-relevant", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10"], [0.80, 0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64, 0.63, 0.62], 10),
        careful.question_id: _row(careful, ["d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20"], [0.60, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51, 0.50], 20),
    }
    reranked = {
        fast.question_id: _row(fast, ["fast-relevant", "r2", "r3", "r4", "r5"], [5, 4, 3, 2, 1], 100),
        careful.question_id: _row(careful, ["careful-relevant", "r7", "r8", "r9", "r10"], [5, 4, 3, 2, 1], 200),
    }
    router_config = load_router_config(config.pipelines.routed.config_path, project_root=PROJECT_ROOT)
    router = RuleBasedRouter(router_config)

    def route_query(question):
        row = {"retrieved_chunk_ids": ["p1", "p2"], "retrieved_scores": [0.40, 0.39]}
        features = _probe_features(question, row)
        return SimpleNamespace(probe=SimpleNamespace(features=features), decision=router.select(features))

    adversarial = [
        {
            "id": "u1",
            "question": "How should I configure an unrelated framework migration?",
            "query_type": "unsupported",
            "difficulty": "hard",
            "category": "near_domain_technology",
            "expected_behavior": "refusal",
        }
    ]
    clock = iter([0.0, 0.01]).__next__
    report = build_routed_report(config, [fast, careful], {"dense": dense, "reranked": reranked}, adversarial, route_query, clock=clock)

    assert report["route_counts_supported"] == {"CAREFUL": 1, "FAST": 1}
    assert report["metrics"] == {"question_count": 2, "mrr_at_5": 1.0, "recall_at_5": 1.0}
    assert report["questions"][0]["latency_ms"] == 10
    assert report["questions"][1]["latency_ms"] == 220
    assert report["refusal_correctness"]["accuracy"] == 1


def test_answer_quality_pipeline_uses_fixed_retrieved_evidence_and_strict_judgment():
    config = load_final_benchmark_config(PROJECT_ROOT / "configs/final_benchmark.yaml", project_root=PROJECT_ROOT)
    question = next(question for question in load_golden_questions(config.datasets.golden_path) if question.id == "gqa-001")
    answer_quality = config.answer_quality.model_copy(update={"sample_question_ids": [question.id]})
    config = config.model_copy(update={"answer_quality": answer_quality})
    chunk = DocumentChunk(
        chunk_id="relevant",
        document_id="doc-1",
        text="Run the app with fastapi dev main.py and open /docs for interactive documentation.",
        token_count=15,
        chunk_hash="hash",
        metadata={"relative_path": question.expected_source},
    )
    row = {
        "question_id": question.id,
        "question": question.question,
        "expected_source": question.expected_source,
        "relevant_chunk_ids": [chunk.chunk_id],
        "retrieved_chunk_ids": [chunk.chunk_id],
        "retrieved_scores": [0.9],
        "latency_ms": 1,
    }
    clock = iter([0.0, 0.01, 1.0, 1.02]).__next__
    records = run_answer_quality_pipeline(
        config,
        "dense",
        {"dense": {question.id: row}},
        {question.id: question},
        {chunk.chunk_id: chunk},
        LocalTemplateGenerationClient(),
        FixedJudge(),
        clock=clock,
        timestamp_factory=lambda: "2026-08-21T00:00:00+00:00",
    )
    records = validate_benchmark_judgments(records, config, "dense")

    assert records[0].retrieved_chunk_ids == ["relevant"]
    assert records[0].automatic_judgment.faithfulness.score == 5
    assert records[0].generation_ms == 10
    assert records[0].judge_ms == pytest.approx(20)

    resumed = run_answer_quality_pipeline(
        config,
        "dense",
        {"dense": {question.id: row}},
        {question.id: question},
        {chunk.chunk_id: chunk},
        LocalTemplateGenerationClient(),
        FixedJudge(),
        existing_records=records,
    )
    assert resumed == records


def test_provider_retry_is_bounded_to_transient_throttles(monkeypatch):
    client = FlakyClient()
    delays = []
    monkeypatch.setattr("scripts.final_benchmark.time.sleep", delays.append)
    retrying = RetryingGenerationClient(client, maximum_attempts=3)

    assert retrying.generate("prompt") == "completed"
    assert client.calls == 2
    assert delays == [3.0]
