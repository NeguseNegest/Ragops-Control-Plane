import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.evaluation.llm_judge import (  # noqa: E402
    ANSWER_RELEVANCE_RUBRIC,
    FAITHFULNESS_RUBRIC,
    REFUSAL_CORRECTNESS_RUBRIC,
    apply_manual_review,
    judgment_artifact_paths,
    load_generation_judge_config,
    load_judged_answers,
    summarize_judgments,
    validate_judgment_set,
    write_judgment_artifacts,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Manually spot-check automatic Day 20 LLM judgments.")
    parser.add_argument("--config", type=Path, default=Path("configs/generation_judge.yaml"), help="Generation-judge YAML configuration.")
    parser.add_argument("--input", type=Path, help="Optional judgments JSONL path override.")
    parser.add_argument("--reviewer", default="manual-review", help="Reviewer identity recorded with each decision.")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing automatic judgments without interactive review.")
    parser.add_argument("--require-reviewed", action="store_true", help="Require the configured minimum number of completed spot-checks.")
    args = parser.parse_args()
    if not args.reviewer.strip():
        parser.error("--reviewer must not be empty.")
    return args


def print_rubrics():
    print("\nFaithfulness rubric")
    for score, description in sorted(FAITHFULNESS_RUBRIC.items()):
        print(f"  {score}: {description}")
    print("\nAnswer relevance rubric")
    for score, description in sorted(ANSWER_RELEVANCE_RUBRIC.items()):
        print(f"  {score}: {description}")
    print("\nRefusal correctness rubric")
    for query_type, description in REFUSAL_CORRECTNESS_RUBRIC.items():
        print(f"  {query_type}: {description}")


def print_record(record, index, total):
    automatic = record.automatic_judgment
    print("\n" + "=" * 100)
    print(f"Judgment {index}/{total}: {record.question_id} ({record.query_type}, expected {record.expected_behavior})")
    print(f"Question: {record.question}")
    print(f"Expected answer: {record.expected_answer}")
    print(f"Expected source: {record.expected_source}")
    print(f"Generator: {record.generator.provider}/{record.generator.model}")
    print(f"Judge: {record.judge.provider}/{record.judge.model}")
    print(f"\nGenerated answer:\n{record.generated_answer}")
    print("\nRetrieved evidence:")
    for evidence in record.retrieved_evidence:
        print("\n" + "-" * 100)
        print(f"rank={evidence.rank} score={evidence.score:.4f} chunk={evidence.chunk_id} source={evidence.source or 'unknown'}")
        print(evidence.text)
    print("\nAutomatic judgment:")
    print(f"  Faithfulness {automatic.faithfulness.score}/5: {automatic.faithfulness.rationale}")
    print(f"  Answer relevance {automatic.answer_relevance.score}/5: {automatic.answer_relevance.rationale}")
    refusal = automatic.refusal_correctness
    print(f"  Refusal correctness {refusal.verdict} (observed {refusal.observed_behavior}): {refusal.rationale}")


def main():
    args = parse_args()
    config = load_generation_judge_config(args.config, project_root=PROJECT_ROOT)
    default_input, _ = judgment_artifact_paths(config)
    input_path = args.input.resolve() if args.input else default_input
    if not input_path.is_file():
        raise FileNotFoundError(f"Judgments file does not exist: {input_path}")
    records = load_judged_answers(input_path)
    minimum_reviewed = config.manual_review.minimum_spot_checks if args.require_reviewed else 0

    if args.validate_only:
        records = validate_judgment_set(records, expected_count=config.dataset.sample_size, minimum_reviewed=minimum_reviewed)
        summary = summarize_judgments(records)
        print(
            f"Valid judgment set: {len(records)} records, "
            f"{summary['manual_review']['reviewed_count']} manually spot-checked."
        )
        return

    if input_path != default_input:
        raise ValueError("Interactive review with --input is read-only; use the configured artifact path to persist decisions.")

    print_rubrics()
    pending = [record for record in records if record.manual_review.status == "pending"]
    for index, record in enumerate(pending, start=1):
        print_record(record, index, len(pending))
        while True:
            decision = input("\nDoes the automatic judgment match the rubric? [a]gree, [d]isagree, [s]kip, [q]uit: ").strip().lower()
            if decision in {"a", "d", "s", "q"}:
                break
            print("Enter a, d, s, or q.")
        if decision == "q":
            break
        if decision == "s":
            continue
        notes = input("Reviewer notes (required for disagreement, optional for agreement): ").strip() or None
        try:
            records = apply_manual_review(
                records,
                record.question_id,
                "agree" if decision == "a" else "disagree",
                args.reviewer,
                notes=notes,
            )
        except ValueError as error:
            print(error)
            continue
        write_judgment_artifacts(records, config)
        print(f"Saved manual review for {record.question_id}.")

    summary = summarize_judgments(records)
    reviewed_count = summary["manual_review"]["reviewed_count"]
    print(f"Manual spot-checks complete: {reviewed_count}/{config.manual_review.minimum_spot_checks} required.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Judgment review failed: {error}") from error
