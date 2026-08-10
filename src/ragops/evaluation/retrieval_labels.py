import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ragops.evaluation.synthetic_qa import SyntheticQACandidate, normalize_question


class RetrievalLabelMetadata(BaseModel):
    """Review provenance for one retrieval relevance label."""

    label_method: Literal["manual", "verified_synthetic_source"]
    review_status: Literal["verified"] = "verified"
    reviewed_by: str = Field(min_length=1)


class RetrievalLabel(BaseModel):
    """Relevant chunk IDs for one supported golden question."""

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=10, max_length=500)
    relevant_chunk_ids: list[str] = Field(min_length=1)
    expected_source: str = Field(min_length=1)
    metadata: RetrievalLabelMetadata

    @field_validator("question_id", "question", "expected_source")
    @classmethod
    def clean_text(cls, value):
        return value.strip()

    @field_validator("relevant_chunk_ids")
    @classmethod
    def validate_chunk_ids(cls, chunk_ids):
        cleaned_ids = []
        for chunk_id in chunk_ids:
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("Relevant chunk IDs must be non-empty strings.")
            cleaned_ids.append(chunk_id.strip())

        if len(cleaned_ids) != len(set(cleaned_ids)):
            raise ValueError("Relevant chunk IDs must not contain duplicates.")

        return cleaned_ids


def load_retrieval_labels(path):
    """Load and validate retrieval label rows from JSONL."""
    from ragops.evaluation.synthetic_qa import read_jsonl

    return [RetrievalLabel.model_validate(record) for record in read_jsonl(path)]


def chunk_source_path(chunk):
    """Return the stable relative source path stored on a chunk."""
    metadata = chunk.get("metadata") or {}
    return metadata.get("relative_path") or metadata.get("source_path")


def index_by_id(records, id_field, record_name):
    """Build an ID index and reject missing or duplicate IDs."""
    indexed_records = {}

    for record in records:
        record_id = record.get(id_field)
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"Every {record_name} must have a non-empty {id_field}.")
        if record_id in indexed_records:
            raise ValueError(f"Duplicate {record_name} {id_field}: {record_id}")
        indexed_records[record_id] = record

    return indexed_records


def validate_retrieval_labels(labels, golden_records, chunks, minimum_count=0):
    """Validate labels against the golden questions and processed chunks."""
    labels = [label if isinstance(label, RetrievalLabel) else RetrievalLabel.model_validate(label) for label in labels]

    if len(labels) < minimum_count:
        raise ValueError(f"Retrieval label set contains {len(labels)} labels; at least {minimum_count} are required.")

    golden_by_id = index_by_id(golden_records, "id", "golden question")
    chunks_by_id = index_by_id(chunks, "chunk_id", "chunk")
    seen_question_ids = set()

    for label in labels:
        if label.question_id in seen_question_ids:
            raise ValueError(f"Duplicate retrieval label question_id: {label.question_id}")
        seen_question_ids.add(label.question_id)

        golden_record = golden_by_id.get(label.question_id)
        if golden_record is None:
            raise ValueError(f"Retrieval label references unknown golden question: {label.question_id}")
        if golden_record.get("query_type") != "supported":
            raise ValueError(f"Retrieval label question must be supported: {label.question_id}")
        if normalize_question(golden_record.get("question", "")) != normalize_question(label.question):
            raise ValueError(f"Retrieval label question text does not match golden question: {label.question_id}")
        if golden_record.get("expected_source") != label.expected_source:
            raise ValueError(f"Retrieval label expected source does not match golden question: {label.question_id}")

        for chunk_id in label.relevant_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                raise ValueError(f"Retrieval label references unknown chunk ID: {chunk_id}")
            if chunk_source_path(chunk) != label.expected_source:
                raise ValueError(f"Relevant chunk {chunk_id} does not belong to expected source {label.expected_source}.")

    return {
        "label_count": len(labels),
        "question_count": len(seen_question_ids),
        "relevant_chunk_count": sum(len(label.relevant_chunk_ids) for label in labels),
    }


def build_retrieval_label(golden_record, relevant_chunk_ids, reviewer="manual-review", label_method="manual"):
    """Build a verified retrieval label from a supported golden row."""
    if golden_record.get("query_type") != "supported":
        raise ValueError("Only supported golden questions can receive retrieval labels.")

    return RetrievalLabel(
        question_id=golden_record["id"],
        question=golden_record["question"],
        relevant_chunk_ids=relevant_chunk_ids,
        expected_source=golden_record["expected_source"],
        metadata=RetrievalLabelMetadata(
            label_method=label_method,
            reviewed_by=reviewer,
        ),
    )


def bootstrap_labels_from_approved_candidates(golden_records, candidates, chunks, reviewer="source-audit"):
    """Build labels from source-audited synthetic candidates."""
    golden_by_id = index_by_id(golden_records, "id", "golden question")
    chunks_by_id = index_by_id(chunks, "chunk_id", "chunk")
    labels = []

    for raw_candidate in candidates:
        candidate = raw_candidate if isinstance(raw_candidate, SyntheticQACandidate) else SyntheticQACandidate.model_validate(raw_candidate)
        if candidate.metadata.review_status != "approved":
            continue

        golden_record = golden_by_id.get(candidate.id)
        if golden_record is None:
            raise ValueError(f"Approved candidate is missing from the golden set: {candidate.id}")
        if normalize_question(golden_record.get("question", "")) != normalize_question(candidate.question):
            raise ValueError(f"Approved candidate question differs from the golden set: {candidate.id}")
        if golden_record.get("expected_source") != candidate.expected_source:
            raise ValueError(f"Approved candidate source differs from the golden set: {candidate.id}")

        source_chunk = chunks_by_id.get(candidate.metadata.source_chunk_id)
        if source_chunk is None:
            raise ValueError(f"Approved candidate references an unknown source chunk: {candidate.metadata.source_chunk_id}")
        if chunk_source_path(source_chunk) != candidate.expected_source:
            raise ValueError(f"Approved candidate source chunk does not match expected source: {candidate.id}")

        labels.append(
            build_retrieval_label(
                golden_record,
                [candidate.metadata.source_chunk_id],
                reviewer=reviewer,
                label_method="verified_synthetic_source",
            )
        )

    validate_retrieval_labels(labels, golden_records, chunks)
    return labels


def merge_retrieval_labels(existing_labels, new_labels):
    """Merge labels idempotently and reject conflicting decisions."""
    merged_labels = list(existing_labels)
    existing_by_question = {label.question_id: label for label in merged_labels}
    added_labels = []

    for label in new_labels:
        existing_label = existing_by_question.get(label.question_id)
        if existing_label is not None:
            if existing_label.model_dump() != label.model_dump():
                raise ValueError(f"Conflicting retrieval label for question: {label.question_id}")
            continue

        merged_labels.append(label)
        added_labels.append(label)
        existing_by_question[label.question_id] = label

    return merged_labels, added_labels


def lexical_terms(text):
    """Return useful lowercase terms for offline chunk ranking."""
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
    }
    return {term for term in re.findall(r"[a-z0-9_]+", text.casefold()) if len(term) > 1 and term not in stop_words}


def rank_candidate_chunks(question, expected_source, chunks, preferred_chunk_ids=None, limit=8):
    """Rank chunks from the expected source for manual inspection."""
    if limit <= 0:
        raise ValueError("Chunk display limit must be greater than zero.")

    preferred_chunk_ids = set(preferred_chunk_ids or [])
    question_terms = lexical_terms(question)
    ranked_chunks = []

    for chunk in chunks:
        if chunk_source_path(chunk) != expected_source:
            continue

        metadata = chunk.get("metadata") or {}
        searchable_text = f"{metadata.get('heading', '')} {chunk.get('text', '')}"
        overlap_count = len(question_terms & lexical_terms(searchable_text))
        preferred = chunk.get("chunk_id") in preferred_chunk_ids
        chunk_index = metadata.get("chunk_index", 0)
        ranked_chunks.append((chunk, preferred, overlap_count, chunk_index))

    ranked_chunks.sort(key=lambda item: (-int(item[1]), -item[2], item[3], item[0].get("chunk_id", "")))
    return [(chunk, overlap_count) for chunk, _, overlap_count, _ in ranked_chunks[:limit]]


def resolve_chunk_selection(selection, displayed_chunks):
    """Resolve comma-separated display numbers or exact chunk IDs."""
    tokens = [token.strip() for token in selection.split(",") if token.strip()]
    if not tokens:
        raise ValueError("Select at least one chunk.")

    displayed_by_id = {chunk["chunk_id"]: chunk for chunk, _ in displayed_chunks}
    selected_ids = []

    for token in tokens:
        if token.isdigit():
            position = int(token)
            if position < 1 or position > len(displayed_chunks):
                raise ValueError(f"Chunk number is outside the displayed range: {token}")
            chunk_id = displayed_chunks[position - 1][0]["chunk_id"]
        else:
            if token not in displayed_by_id:
                raise ValueError(f"Chunk ID is not in the displayed candidates: {token}")
            chunk_id = token

        if chunk_id in selected_ids:
            raise ValueError(f"Chunk was selected more than once: {chunk_id}")
        selected_ids.append(chunk_id)

    return selected_ids
