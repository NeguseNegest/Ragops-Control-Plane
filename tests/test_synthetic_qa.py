import json
from pathlib import Path

import pytest

from ragops.evaluation.synthetic_qa import (
    SyntheticQACandidate,
    apply_review_decisions,
    build_candidate_prompt,
    generate_synthetic_candidates,
    load_source_chunks,
    load_synthetic_candidates,
    merge_approved_candidates,
    parse_generated_pairs,
    read_jsonl,
    select_source_chunks,
    write_jsonl,
)


def make_chunk(chunk_id="chunk-1", source_name="fastapi", relative_path="fastapi/docs/example.md", text=None):
    if text is None:
        text = " ".join(f"source-word-{index}" for index in range(100))
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc-{chunk_id}",
        "text": text,
        "metadata": {
            "relative_path": relative_path,
            "source_name": source_name,
            "heading": "Example",
        },
    }


def generated_response(prefix, count):
    return json.dumps(
        {
            "candidates": [
                {
                    "question": f"How does {prefix} example number {index} work?",
                    "expected_answer": f"The source explains {prefix} answer number {index} in detail.",
                    "difficulty": "Medium",
                }
                for index in range(count)
            ]
        }
    )


class FakeGenerationClient:

    def __init__(self, model, responses):
        self.model = model
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_parse_generated_pairs_accepts_fenced_json_and_skips_invalid_rows():
    response = """```json
    {"candidates":[
      {"question":"How is a valid example configured?","expected_answer":"Configure the valid example with the documented setting.","difficulty":"EASY"},
      {"question":"short","expected_answer":"also short","difficulty":"unknown"}
    ]}
    ```"""

    pairs = parse_generated_pairs(response)

    assert len(pairs) == 1
    assert pairs[0].difficulty == "easy"


def test_parse_generated_pairs_rejects_non_json_and_empty_candidate_lists():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_generated_pairs("not JSON")

    with pytest.raises(ValueError, match="no valid QA"):
        parse_generated_pairs('{"candidates":[]}')


def test_select_source_chunks_is_deterministic_and_balanced():
    chunks = [
        make_chunk(f"fastapi-{index}", "fastapi", f"fastapi/docs/{index}.md")
        for index in range(4)
    ] + [
        make_chunk(f"mlflow-{index}", "mlflow", f"mlflow/docs/{index}.md")
        for index in range(4)
    ] + [
        make_chunk(f"qdrant-{index}", "qdrant", f"qdrant/docs/{index}.md")
        for index in range(4)
    ]

    first = select_source_chunks(chunks, count=6, seed=4)
    second = select_source_chunks(chunks, count=6, seed=4)

    assert [chunk["chunk_id"] for chunk in first] == [chunk["chunk_id"] for chunk in second]
    assert [chunk["metadata"]["source_name"] for chunk in first] == [
        "fastapi",
        "mlflow",
        "qdrant",
        "fastapi",
        "mlflow",
        "qdrant",
    ]


def test_build_candidate_prompt_contains_source_and_strict_output_shape():
    chunk = make_chunk(text="FastAPI validates typed path parameters using Python annotations. " * 10)

    prompt = build_candidate_prompt(chunk, candidate_count=3)

    assert "Generate exactly 3" in prompt
    assert "FastAPI validates typed path parameters" in prompt
    assert "Source chunk ID: chunk-1" in prompt
    assert '"candidates"' in prompt


def test_generate_synthetic_candidates_balances_providers_and_tracks_provenance():
    chunks = [make_chunk(f"chunk-{index}") for index in range(12)]
    openai = FakeGenerationClient("openai-test-model", [generated_response("openai-first", 2), generated_response("openai-second", 1)])
    gemini = FakeGenerationClient("gemini-test-model", [generated_response("gemini-first", 2)])

    candidates = generate_synthetic_candidates(
        chunks,
        {"openai": openai, "gemini": gemini},
        count=5,
        pairs_per_chunk=2,
        existing_questions=["How does an unrelated existing question work?"],
    )

    assert len(candidates) == 5
    assert [candidate.metadata.provider for candidate in candidates] == ["openai", "openai", "openai", "gemini", "gemini"]
    assert candidates[0].metadata.model == "openai-test-model"
    assert candidates[-1].metadata.model == "gemini-test-model"
    assert all(candidate.metadata.review_status == "pending" for candidate in candidates)
    assert all(candidate.expected_source == "fastapi/docs/example.md" for candidate in candidates)
    assert len({candidate.id for candidate in candidates}) == 5


def test_generate_synthetic_candidates_skips_bad_response_and_duplicate_question():
    chunks = [make_chunk(f"chunk-{index}") for index in range(6)]
    duplicate = generated_response("duplicate", 1)
    client = FakeGenerationClient("test-model", ["invalid", duplicate, duplicate, generated_response("unique", 1)])

    candidates = generate_synthetic_candidates(chunks, {"openai": client}, count=2, pairs_per_chunk=1)

    assert len(candidates) == 2
    assert len(client.prompts) == 4
    assert candidates[0].question != candidates[1].question


def test_source_chunk_loader_drops_embeddings_and_short_or_excluded_chunks(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    valid_chunk = make_chunk()
    valid_chunk["embedding"] = [0.1, 0.2]
    short_chunk = make_chunk("short", text="too short")
    excluded_chunk = make_chunk("excluded", relative_path="fastapi/docs/_llm-test.md")
    write_jsonl([valid_chunk, short_chunk, excluded_chunk], chunks_path)

    chunks = load_source_chunks(chunks_path)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "chunk-1"
    assert "embedding" not in chunks[0]


def test_candidate_jsonl_round_trip_and_overwrite_guard(tmp_path):
    candidate_path = tmp_path / "candidates.jsonl"
    client = FakeGenerationClient("test-model", [generated_response("round-trip", 1)])
    candidates = generate_synthetic_candidates([make_chunk()], {"openai": client}, count=1, pairs_per_chunk=1)
    write_jsonl(candidates, candidate_path)

    loaded = load_synthetic_candidates(candidate_path)

    assert loaded == candidates
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_jsonl(candidates, candidate_path)


def test_read_jsonl_reports_invalid_line(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"valid":true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        read_jsonl(path)


def test_review_decisions_and_merge_only_add_approved_candidates():
    client = FakeGenerationClient("test-model", [generated_response("review", 2)])
    candidates = generate_synthetic_candidates([make_chunk()], {"openai": client}, count=2, pairs_per_chunk=2)
    decisions = {
        candidates[0].id: "approved",
        candidates[1].id: "rejected",
    }
    reviewed = apply_review_decisions(candidates, decisions, reviewer="test-reviewer")
    golden = [
        {
            "id": "gqa-001",
            "question": "What is already in the golden dataset?",
            "expected_answer": "An existing manually reviewed example.",
            "expected_source": "fastapi/docs/existing.md",
            "query_type": "supported",
            "difficulty": "easy",
        }
    ]

    merged, added = merge_approved_candidates(golden, reviewed)
    merged_again, added_again = merge_approved_candidates(merged, reviewed)

    assert reviewed[0].metadata.reviewed_by == "test-reviewer"
    assert reviewed[1].metadata.review_status == "rejected"
    assert len(added) == 1
    assert len(merged) == 2
    assert added[0]["metadata"]["origin"] == "synthetic"
    assert len(merged_again) == 2
    assert added_again == []


def test_invalid_review_status_is_rejected():
    candidate = SyntheticQACandidate(
        id="sqa-test",
        question="How does this documented feature behave?",
        expected_answer="The documented feature behaves according to the supplied source.",
        expected_source="fastapi/docs/example.md",
        difficulty="easy",
        metadata={"provider": "openai", "model": "test", "source_chunk_id": "chunk-1"},
    )

    with pytest.raises(ValueError, match="Unsupported review statuses"):
        apply_review_decisions([candidate], {candidate.id: "maybe"})


def test_day_16_datasets_meet_acceptance_criteria():
    project_root = Path(__file__).resolve().parents[1]
    candidates = load_synthetic_candidates(project_root / "data/eval/synthetic_qa_candidates.jsonl")
    golden_records = read_jsonl(project_root / "data/eval/golden_qa.jsonl")
    approved_candidates = [candidate for candidate in candidates if candidate.metadata.review_status == "approved"]
    rejected_candidates = [candidate for candidate in candidates if candidate.metadata.review_status == "rejected"]
    golden_ids = {record["id"] for record in golden_records}

    assert len(candidates) == 100
    assert 40 <= len(approved_candidates) <= 60
    assert len(approved_candidates) + len(rejected_candidates) == 100
    assert {candidate.metadata.provider for candidate in candidates} == {"openai", "gemini"}
    assert all(candidate.metadata.source_chunk_id for candidate in candidates)
    assert all(candidate.id in golden_ids for candidate in approved_candidates)
    assert len(golden_records) >= 75
