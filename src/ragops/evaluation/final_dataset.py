import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragops.evaluation.llm_judge import GoldenQuestion
from ragops.evaluation.retrieval_labels import (
    RetrievalLabel,
    build_retrieval_label,
    chunk_source_path,
    validate_retrieval_labels,
)
from ragops.evaluation.synthetic_qa import normalize_question, read_jsonl, write_jsonl
from ragops.generation.no_answer import NO_ANSWER_RESPONSE

FINAL_DATASET_SCHEMA_VERSION = 1
Difficulty = Literal["easy", "medium", "hard"]
AdversarialCategory = Literal[
    "near_domain_technology",
    "high_stakes_out_of_scope",
    "instruction_injection",
    "false_premise",
    "general_out_of_scope",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CountRange(StrictModel):
    minimum: int = Field(gt=0)
    maximum: int = Field(gt=0)
    expected: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self):
        if not self.minimum <= self.expected <= self.maximum:
            raise ValueError("Expected count must be inside the accepted range.")
        return self


class FinalDatasetSources(StrictModel):
    historical_golden_path: Path
    historical_retrieval_labels_path: Path
    historical_adversarial_path: Path
    additions_path: Path
    chunks_path: Path


class FinalDatasetOutputs(StrictModel):
    golden_path: Path
    retrieval_labels_path: Path
    adversarial_path: Path
    report_path: Path


class ExcludedGoldenQuestion(StrictModel):
    id: str = Field(min_length=1)
    reason: Literal["ambiguous_without_context", "semantic_duplicate", "weak_or_trivial"]
    rationale: str = Field(min_length=20)


class ManualRetrievalDecision(StrictModel):
    question_id: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(min_length=1)

    @field_validator("relevant_chunk_ids")
    @classmethod
    def validate_chunk_ids(cls, values):
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("Manual retrieval chunk IDs must not be empty.")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Manual retrieval chunk IDs must be unique per question.")
        return cleaned


class FinalReviewConfig(StrictModel):
    reviewer: str = Field(min_length=1)
    completed_on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    excluded_golden: list[ExcludedGoldenQuestion] = Field(min_length=1)
    manual_retrieval_labels: list[ManualRetrievalDecision] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_decisions(self):
        excluded_ids = [item.id for item in self.excluded_golden]
        label_ids = [item.question_id for item in self.manual_retrieval_labels]
        if len(excluded_ids) != len(set(excluded_ids)):
            raise ValueError("Excluded golden IDs must be unique.")
        if len(label_ids) != len(set(label_ids)):
            raise ValueError("Manual retrieval question IDs must be unique.")
        return self


class FinalAcceptanceConfig(StrictModel):
    golden: CountRange
    retrieval_labels: CountRange
    adversarial: CountRange
    minimum_supported: int = Field(gt=0)
    minimum_ambiguous: int = Field(gt=0)
    minimum_unsupported: int = Field(gt=0)
    minimum_hard_golden: int = Field(gt=0)
    minimum_source_families: int = Field(gt=0)
    minimum_adversarial_categories: int = Field(gt=0)
    minimum_manual_retrieval_labels: int = Field(gt=0)


class FinalEvaluationDatasetConfig(StrictModel):
    schema_version: Literal[1] = FINAL_DATASET_SCHEMA_VERSION
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    status: Literal["reviewed"]
    sources: FinalDatasetSources
    outputs: FinalDatasetOutputs
    review: FinalReviewConfig
    acceptance: FinalAcceptanceConfig

    @model_validator(mode="after")
    def require_distinct_outputs(self):
        source_paths = set(self.sources.model_dump().values())
        output_paths = list(self.outputs.model_dump().values())
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("Final dataset output paths must be distinct.")
        if source_paths & set(output_paths):
            raise ValueError("Historical sources and final outputs must be distinct.")
        return self


class SupportedAddition(StrictModel):
    record_type: Literal["supported"]
    id: str = Field(pattern=r"^day46-supported-[0-9]{3}$")
    question: str = Field(min_length=10, max_length=500)
    expected_answer: str = Field(min_length=10)
    expected_source: str = Field(min_length=1)
    query_type: Literal["supported"] = "supported"
    difficulty: Difficulty
    source_chunk_ids: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=20)


class AdversarialAddition(StrictModel):
    record_type: Literal["adversarial"]
    id: str = Field(pattern=r"^day46-adv-[0-9]{3}$")
    question: str = Field(min_length=10, max_length=500)
    difficulty: Difficulty
    category: AdversarialCategory
    split: Literal["evaluation"] = "evaluation"
    rationale: str = Field(min_length=20)


class FinalAdversarialExample(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    question: str = Field(min_length=10, max_length=500)
    query_type: Literal["unsupported"] = "unsupported"
    difficulty: Difficulty
    split: Literal["calibration", "evaluation"]
    category: AdversarialCategory
    expected_behavior: Literal["refusal"] = "refusal"
    provenance: str = Field(min_length=1)
    review_status: Literal["verified"] = "verified"
    reviewed_by: str = Field(min_length=1)

    @field_validator("question", "provenance", "reviewed_by")
    @classmethod
    def clean_text(cls, value):
        return value.strip()


class FinalDatasetSnapshot(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    golden: list[GoldenQuestion]
    retrieval_labels: list[RetrievalLabel]
    adversarial: list[FinalAdversarialExample]
    report: dict


def _resolve_path(path, project_root):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(project_root) / path).resolve()


def load_final_dataset_config(config_path, project_root=None):
    """Load the strict Day 46 curation contract and resolve project-relative paths."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Final dataset config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid final dataset YAML in {config_path}: {error}") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Final dataset config must contain a YAML mapping: {config_path}")

    config = FinalEvaluationDatasetConfig.model_validate(payload)
    root = Path(project_root or Path.cwd()).resolve()
    sources = config.sources.model_copy(
        update={field: _resolve_path(value, root) for field, value in config.sources.model_dump().items()}
    )
    outputs = config.outputs.model_copy(
        update={field: _resolve_path(value, root) for field, value in config.outputs.model_dump().items()}
    )
    return config.model_copy(update={"sources": sources, "outputs": outputs})


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_bytes(records):
    lines = []
    for record in records:
        if isinstance(record, BaseModel):
            record = record.model_dump(exclude_none=True)
        lines.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _sha256_records(records):
    return hashlib.sha256(_jsonl_bytes(records)).hexdigest()


def _source_family(source):
    return source.split("/", maxsplit=1)[0]


def _normalized_questions(records):
    return [normalize_question(record.question if isinstance(record, BaseModel) else record["question"]) for record in records]


def _validate_unique_questions(records, label):
    normalized = _normalized_questions(records)
    if len(normalized) != len(set(normalized)):
        duplicates = sorted(question for question, count in Counter(normalized).items() if count > 1)
        raise ValueError(f"{label} contains duplicate normalized questions: {duplicates}")


def _in_range(value, accepted_range, label):
    if not accepted_range.minimum <= value <= accepted_range.maximum:
        raise ValueError(
            f"{label} contains {value} rows; expected {accepted_range.minimum}-{accepted_range.maximum}."
        )
    if value != accepted_range.expected:
        raise ValueError(f"{label} contains {value} rows; the reviewed snapshot expects {accepted_range.expected}.")


def _reviewed_metadata(existing, config, origin):
    metadata = dict(existing or {})
    metadata.setdefault("origin", origin)
    metadata.update(
        {
            "final_review_status": "verified",
            "final_reviewed_by": config.review.reviewer,
            "final_reviewed_on": config.review.completed_on,
        }
    )
    return metadata


def _load_additions(path):
    supported = []
    adversarial = []
    for row in read_jsonl(path):
        record_type = row.get("record_type")
        if record_type == "supported":
            supported.append(SupportedAddition.model_validate(row))
        elif record_type == "adversarial":
            adversarial.append(AdversarialAddition.model_validate(row))
        else:
            raise ValueError(f"Unknown Day 46 addition record_type: {record_type!r}")
    return supported, adversarial


def _build_golden(config, historical_golden, supported_additions, adversarial_additions):
    excluded_ids = {item.id for item in config.review.excluded_golden}
    historical_by_id = {question.id: question for question in historical_golden}
    missing_exclusions = excluded_ids - set(historical_by_id)
    if missing_exclusions:
        raise ValueError(f"Excluded golden IDs do not exist in the historical set: {sorted(missing_exclusions)}")

    final_questions = []
    for question in historical_golden:
        if question.id in excluded_ids:
            continue
        origin = "synthetic" if question.id.startswith("sqa-") else "manual"
        final_questions.append(
            question.model_copy(update={"metadata": _reviewed_metadata(question.metadata, config, origin)})
        )

    for addition in supported_additions:
        final_questions.append(
            GoldenQuestion(
                id=addition.id,
                question=addition.question,
                expected_answer=addition.expected_answer,
                expected_source=addition.expected_source,
                query_type="supported",
                difficulty=addition.difficulty,
                metadata=_reviewed_metadata(
                    {
                        "origin": "manual_day46",
                        "source_chunk_ids": addition.source_chunk_ids,
                        "review_rationale": addition.rationale,
                    },
                    config,
                    "manual_day46",
                ),
            )
        )

    for addition in adversarial_additions:
        final_questions.append(
            GoldenQuestion(
                id=addition.id,
                question=addition.question,
                expected_answer=NO_ANSWER_RESPONSE,
                expected_source=None,
                query_type="unsupported",
                difficulty=addition.difficulty,
                metadata=_reviewed_metadata(
                    {
                        "origin": "manual_day46",
                        "adversarial_category": addition.category,
                        "review_rationale": addition.rationale,
                    },
                    config,
                    "manual_day46",
                ),
            )
        )

    ids = [question.id for question in final_questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Final golden questions must have unique IDs.")
    _validate_unique_questions(final_questions, "Final golden set")
    return final_questions


def _build_retrieval_labels(config, historical_labels, golden, chunks):
    golden_by_id = {question.id: question for question in golden}
    labels = [label for label in historical_labels if label.question_id in golden_by_id]
    existing_ids = {label.question_id for label in labels}

    for decision in config.review.manual_retrieval_labels:
        if decision.question_id in existing_ids:
            raise ValueError(f"Manual retrieval decision duplicates an existing label: {decision.question_id}")
        question = golden_by_id.get(decision.question_id)
        if question is None:
            raise ValueError(f"Manual retrieval decision references an unknown golden ID: {decision.question_id}")
        label = build_retrieval_label(
            question.model_dump(),
            decision.relevant_chunk_ids,
            reviewer=config.review.reviewer,
            label_method="manual",
        )
        labels.append(label)
        existing_ids.add(label.question_id)

    validate_retrieval_labels(
        labels,
        [question.model_dump() for question in golden],
        chunks,
        minimum_count=config.acceptance.retrieval_labels.minimum,
    )
    return labels


def _build_adversarial(config, historical_rows, golden, additions):
    golden_by_id = {question.id: question for question in golden}
    final_rows = []
    for row in historical_rows:
        golden_question = golden_by_id.get(row["id"])
        difficulty = golden_question.difficulty if golden_question else (
            "hard" if row["category"] == "high_stakes_out_of_scope" else "medium"
        )
        final_rows.append(
            FinalAdversarialExample(
                id=row["id"],
                question=row["question"],
                difficulty=difficulty,
                split=row["split"],
                category=row["category"],
                provenance=row["provenance"],
                reviewed_by=row["reviewed_by"],
            )
        )

    for addition in additions:
        final_rows.append(
            FinalAdversarialExample(
                id=addition.id,
                question=addition.question,
                difficulty=addition.difficulty,
                split=addition.split,
                category=addition.category,
                provenance="data/eval/day46_additions.jsonl",
                reviewed_by=config.review.reviewer,
            )
        )

    ids = [row.id for row in final_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Final adversarial examples must have unique IDs.")
    _validate_unique_questions(final_rows, "Final adversarial set")
    return final_rows


def _validate_supported_source_provenance(golden, chunks):
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    available_sources = {chunk_source_path(chunk) for chunk in chunks}
    for question in golden:
        if question.query_type != "supported":
            if question.expected_source is not None:
                raise ValueError(f"Non-supported question must not name an expected source: {question.id}")
            continue
        if question.expected_source not in available_sources:
            raise ValueError(f"Supported question source is missing from processed chunks: {question.id}")
        source_chunk_ids = question.metadata.get("source_chunk_ids", [])
        for chunk_id in source_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                raise ValueError(f"Supported addition references an unknown source chunk: {question.id}/{chunk_id}")
            if chunk_source_path(chunk) != question.expected_source:
                raise ValueError(f"Supported addition chunk/source mismatch: {question.id}/{chunk_id}")


def _validate_final_acceptance(config, golden, labels, adversarial):
    acceptance = config.acceptance
    _in_range(len(golden), acceptance.golden, "Final golden set")
    _in_range(len(labels), acceptance.retrieval_labels, "Final retrieval label set")
    _in_range(len(adversarial), acceptance.adversarial, "Final adversarial set")

    query_types = Counter(question.query_type for question in golden)
    required_query_counts = {
        "supported": acceptance.minimum_supported,
        "ambiguous": acceptance.minimum_ambiguous,
        "unsupported": acceptance.minimum_unsupported,
    }
    for query_type, minimum in required_query_counts.items():
        if query_types[query_type] < minimum:
            raise ValueError(f"Final golden set needs at least {minimum} {query_type} questions.")

    hard_count = sum(question.difficulty == "hard" for question in golden)
    if hard_count < acceptance.minimum_hard_golden:
        raise ValueError(f"Final golden set needs at least {acceptance.minimum_hard_golden} hard questions.")

    source_families = {_source_family(question.expected_source) for question in golden if question.expected_source}
    if len(source_families) < acceptance.minimum_source_families:
        raise ValueError("Final golden set does not cover enough source families.")

    adversarial_categories = {row.category for row in adversarial}
    if len(adversarial_categories) < acceptance.minimum_adversarial_categories:
        raise ValueError("Final adversarial set does not cover enough categories.")

    manual_count = sum(label.metadata.label_method == "manual" for label in labels)
    if manual_count < acceptance.minimum_manual_retrieval_labels:
        raise ValueError("Final retrieval labels do not include enough independent manual decisions.")

    for question in golden:
        metadata = question.metadata
        if metadata.get("final_review_status") != "verified":
            raise ValueError(f"Golden question lacks final verified review metadata: {question.id}")
        if metadata.get("final_reviewed_by") != config.review.reviewer:
            raise ValueError(f"Golden question reviewer metadata is inconsistent: {question.id}")


def _build_report(config, golden, labels, adversarial):
    query_types = Counter(question.query_type for question in golden)
    difficulties = Counter(question.difficulty for question in golden)
    sources = Counter(_source_family(question.expected_source) for question in golden if question.expected_source)
    label_methods = Counter(label.metadata.label_method for label in labels)
    label_sources = Counter(_source_family(label.expected_source) for label in labels)
    adversarial_categories = Counter(row.category for row in adversarial)
    adversarial_difficulties = Counter(row.difficulty for row in adversarial)

    source_paths = config.sources.model_dump()
    output_paths = config.outputs.model_dump()
    all_paths = [str(Path(path).resolve()) for path in (*source_paths.values(), *output_paths.values())]
    project_root = Path(os.path.commonpath(all_paths))
    contract_payload = {
        "schema_version": config.schema_version,
        "name": config.name,
        "version": config.version,
        "status": config.status,
        "review": config.review.model_dump(mode="json"),
        "acceptance": config.acceptance.model_dump(mode="json"),
    }
    contract_sha256 = hashlib.sha256(
        json.dumps(contract_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    def display_path(path):
        path = Path(path)
        try:
            return str(path.relative_to(project_root))
        except ValueError:
            return str(path)

    return {
        "schema_version": FINAL_DATASET_SCHEMA_VERSION,
        "name": config.name,
        "version": config.version,
        "status": config.status,
        "review": {
            "completed_on": config.review.completed_on,
            "reviewer": config.review.reviewer,
            "historical_questions_reviewed": sum(1 for _ in read_jsonl(config.sources.historical_golden_path)),
            "excluded_question_count": len(config.review.excluded_golden),
            "excluded_questions": [item.model_dump() for item in config.review.excluded_golden],
            "manual_retrieval_decision_count": len(config.review.manual_retrieval_labels),
            "curation_contract_sha256": contract_sha256,
        },
        "construction": {
            "policy": "immutable_historical_inputs_plus_explicit_reviewed_curation",
            "historical_inputs_are_unchanged": True,
            "unsupported_overlap_policy": "Unsupported golden questions intentionally overlap the adversarial set so generation and refusal behavior can be evaluated on the same reviewed prompts.",
        },
        "counts": {
            "golden": len(golden),
            "retrieval_labels": len(labels),
            "adversarial": len(adversarial),
        },
        "golden_distribution": {
            "query_type": dict(sorted(query_types.items())),
            "difficulty": dict(sorted(difficulties.items())),
            "supported_source_family": dict(sorted(sources.items())),
        },
        "retrieval_distribution": {
            "label_method": dict(sorted(label_methods.items())),
            "source_family": dict(sorted(label_sources.items())),
            "unique_question_count": len({label.question_id for label in labels}),
            "relevant_chunk_count": sum(len(label.relevant_chunk_ids) for label in labels),
        },
        "adversarial_distribution": {
            "category": dict(sorted(adversarial_categories.items())),
            "difficulty": dict(sorted(adversarial_difficulties.items())),
        },
        "source_provenance": {
            field: {"path": display_path(path), "sha256": _sha256_file(path)}
            for field, path in source_paths.items()
        },
        "artifacts": {
            "golden": {
                "path": display_path(output_paths["golden_path"]),
                "sha256": _sha256_records(golden),
            },
            "retrieval_labels": {
                "path": display_path(output_paths["retrieval_labels_path"]),
                "sha256": _sha256_records(labels),
            },
            "adversarial": {
                "path": display_path(output_paths["adversarial_path"]),
                "sha256": _sha256_records(adversarial),
            },
        },
        "acceptance": {
            "passed": True,
            "golden_range": config.acceptance.golden.model_dump(),
            "retrieval_label_range": config.acceptance.retrieval_labels.model_dump(),
            "adversarial_range": config.acceptance.adversarial.model_dump(),
            "minimum_hard_golden": config.acceptance.minimum_hard_golden,
            "minimum_source_families": config.acceptance.minimum_source_families,
            "minimum_adversarial_categories": config.acceptance.minimum_adversarial_categories,
            "minimum_manual_retrieval_labels": config.acceptance.minimum_manual_retrieval_labels,
        },
    }


def build_final_dataset_snapshot(config):
    """Rebuild and validate the reviewed final datasets entirely from pinned local inputs."""
    historical_golden = [GoldenQuestion.model_validate(row) for row in read_jsonl(config.sources.historical_golden_path)]
    historical_labels = [RetrievalLabel.model_validate(row) for row in read_jsonl(config.sources.historical_retrieval_labels_path)]
    historical_adversarial = read_jsonl(config.sources.historical_adversarial_path)
    chunks = read_jsonl(config.sources.chunks_path)
    supported_additions, adversarial_additions = _load_additions(config.sources.additions_path)

    golden = _build_golden(config, historical_golden, supported_additions, adversarial_additions)
    _validate_supported_source_provenance(golden, chunks)
    labels = _build_retrieval_labels(config, historical_labels, golden, chunks)
    adversarial = _build_adversarial(config, historical_adversarial, golden, adversarial_additions)
    _validate_final_acceptance(config, golden, labels, adversarial)
    report = _build_report(config, golden, labels, adversarial)
    return FinalDatasetSnapshot(golden=golden, retrieval_labels=labels, adversarial=adversarial, report=report)


def _record_dicts(records):
    return [record.model_dump(exclude_none=True) if isinstance(record, BaseModel) else record for record in records]


def validate_final_dataset_outputs(config, snapshot=None):
    """Reject missing, malformed, edited, or stale final Day 46 artifacts."""
    snapshot = snapshot or build_final_dataset_snapshot(config)
    expected = {
        config.outputs.golden_path: _record_dicts(snapshot.golden),
        config.outputs.retrieval_labels_path: _record_dicts(snapshot.retrieval_labels),
        config.outputs.adversarial_path: _record_dicts(snapshot.adversarial),
    }
    for path, expected_rows in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"Final dataset artifact does not exist: {path}")
        actual_rows = read_jsonl(path)
        if actual_rows != expected_rows:
            raise ValueError(f"Final dataset artifact is stale or edited outside the curation contract: {path}")

    if not config.outputs.report_path.is_file():
        raise FileNotFoundError(f"Final dataset report does not exist: {config.outputs.report_path}")
    try:
        actual_report = json.loads(config.outputs.report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid final dataset report JSON: {config.outputs.report_path}: {error}") from error
    if actual_report != snapshot.report:
        raise ValueError(f"Final dataset report is stale: {config.outputs.report_path}")
    return snapshot.report


def write_final_dataset_outputs(config, snapshot=None, overwrite=False):
    """Atomically write the three reviewed snapshots and their deterministic audit report."""
    snapshot = snapshot or build_final_dataset_snapshot(config)
    paths = (
        config.outputs.golden_path,
        config.outputs.retrieval_labels_path,
        config.outputs.adversarial_path,
        config.outputs.report_path,
    )
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite final dataset artifacts: {', '.join(map(str, existing))}")

    write_jsonl(snapshot.golden, config.outputs.golden_path, overwrite=overwrite)
    write_jsonl(snapshot.retrieval_labels, config.outputs.retrieval_labels_path, overwrite=overwrite)
    write_jsonl(snapshot.adversarial, config.outputs.adversarial_path, overwrite=overwrite)

    report_path = config.outputs.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_name(f".{report_path.name}.tmp")
    temporary_path.write_text(json.dumps(snapshot.report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(report_path)
    return snapshot.report
