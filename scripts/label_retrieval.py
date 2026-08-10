import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.evaluation.retrieval_labels import (  # noqa: E402
    bootstrap_labels_from_approved_candidates,
    build_retrieval_label,
    load_retrieval_labels,
    merge_retrieval_labels,
    rank_candidate_chunks,
    resolve_chunk_selection,
    validate_retrieval_labels,
)
from ragops.evaluation.synthetic_qa import load_source_chunks, load_synthetic_candidates, read_jsonl, write_jsonl  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect source chunks and build retrieval relevance labels.")
    parser.add_argument("--golden", type=Path, default=Path("data/eval/golden_qa.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--candidates", type=Path, default=Path("data/eval/synthetic_qa_candidates.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/retrieval_labels.jsonl"))
    parser.add_argument("--target", type=int, default=40, help="Required number of verified question labels.")
    parser.add_argument("--top-k", type=int, default=8, help="Candidate chunks shown for each question.")
    parser.add_argument("--preview-chars", type=int, default=1200, help="Maximum source characters shown per chunk.")
    parser.add_argument("--reviewer", default="manual-review", help="Reviewer label stored with new decisions.")
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--bootstrap-approved-synthetic", action="store_true", help="Create labels from approved, source-audited Day 16 candidates before interactive labeling.")
    action_group.add_argument("--validate-only", action="store_true", help="Validate the existing label file and exit.")
    args = parser.parse_args()

    if args.target <= 0:
        parser.error("--target must be greater than zero.")
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero.")
    if args.preview_chars <= 0:
        parser.error("--preview-chars must be greater than zero.")
    if not args.reviewer.strip():
        parser.error("--reviewer must not be empty.")

    return args


def preview_text(text, max_chars):
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def print_question(golden_record, ranked_chunks, current_count, target, preview_chars):
    print("\n" + "=" * 100)
    print(f"Labels: {current_count}/{target}")
    print(f"Question ID: {golden_record['id']}")
    print(f"Question: {golden_record['question']}")
    print(f"Expected answer: {golden_record['expected_answer']}")
    print(f"Expected source: {golden_record['expected_source']}")

    for index, (chunk, score) in enumerate(ranked_chunks, start=1):
        metadata = chunk.get("metadata") or {}
        heading = metadata.get("heading") or "[no heading]"
        print("\n" + "-" * 100)
        print(f"[{index}] chunk_id={chunk['chunk_id']} lexical_overlap={score} heading={heading}")
        print(preview_text(chunk.get("text", ""), preview_chars))


def preferred_chunk_ids(golden_record):
    metadata = golden_record.get("metadata") or {}
    source_chunk_id = metadata.get("source_chunk_id")
    return [source_chunk_id] if source_chunk_id else []


def save_labels(labels, output_path):
    write_jsonl(labels, output_path, overwrite=True)


def main():
    args = parse_args()
    golden_records = read_jsonl(args.golden)
    chunks = load_source_chunks(args.chunks, min_words=1)
    labels = load_retrieval_labels(args.output) if args.output.exists() else []

    validate_retrieval_labels(labels, golden_records, chunks)

    if args.validate_only:
        summary = validate_retrieval_labels(labels, golden_records, chunks, minimum_count=args.target)
        print(f"Valid retrieval label set: {summary}")
        return

    if args.bootstrap_approved_synthetic:
        candidates = load_synthetic_candidates(args.candidates)
        bootstrapped_labels = bootstrap_labels_from_approved_candidates(
            golden_records,
            candidates,
            chunks,
            reviewer=args.reviewer,
        )
        labels, added_labels = merge_retrieval_labels(labels, bootstrapped_labels)
        validate_retrieval_labels(labels, golden_records, chunks)
        save_labels(labels, args.output)
        print(f"Added {len(added_labels)} verified labels from approved Day 16 candidates.")

    if len(labels) >= args.target:
        summary = validate_retrieval_labels(labels, golden_records, chunks, minimum_count=args.target)
        print(f"Retrieval label target already satisfied: {summary}")
        return

    labeled_question_ids = {label.question_id for label in labels}
    supported_questions = [
        record
        for record in golden_records
        if record.get("query_type") == "supported" and record.get("id") not in labeled_question_ids
    ]

    for golden_record in supported_questions:
        if len(labels) >= args.target:
            break

        ranked_chunks = rank_candidate_chunks(
            golden_record["question"],
            golden_record["expected_source"],
            chunks,
            preferred_chunk_ids=preferred_chunk_ids(golden_record),
            limit=args.top_k,
        )

        if not ranked_chunks:
            print(f"No chunks found for {golden_record['id']} at {golden_record['expected_source']}; skipping.")
            continue

        print_question(golden_record, ranked_chunks, len(labels), args.target, args.preview_chars)

        while True:
            selection = input("Select chunk numbers/IDs separated by commas, [s]kip, or [q]uit: ").strip()
            if selection.lower() in {"s", "q"}:
                break
            try:
                selected_chunk_ids = resolve_chunk_selection(selection, ranked_chunks)
                label = build_retrieval_label(
                    golden_record,
                    selected_chunk_ids,
                    reviewer=args.reviewer.strip(),
                )
                validate_retrieval_labels([label], golden_records, chunks)
            except ValueError as error:
                print(error)
                continue

            labels.append(label)
            save_labels(labels, args.output)
            print(f"Saved label {label.question_id} with {len(label.relevant_chunk_ids)} relevant chunk(s).")
            break

        if selection.lower() == "q":
            break

    summary = validate_retrieval_labels(labels, golden_records, chunks)
    print(f"Saved retrieval labels: {summary}")
    if len(labels) < args.target:
        print(f"Target not yet reached: {len(labels)}/{args.target}. Run the command again to resume.")


if __name__ == "__main__":
    main()
