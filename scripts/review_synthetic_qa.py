import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.evaluation.synthetic_qa import load_source_chunks, load_synthetic_candidates, merge_approved_candidates, read_jsonl, write_jsonl  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Manually review synthetic QA candidates and merge approved examples.")
    parser.add_argument("--candidates", type=Path, default=Path("data/eval/synthetic_qa_candidates.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--golden", type=Path, default=Path("data/eval/golden_qa.jsonl"))
    parser.add_argument("--target-approvals", type=int, default=40, help="Stop after this many candidates are approved in total.")
    parser.add_argument("--reviewer", default="manual-review", help="Reviewer label stored in approved and rejected metadata.")
    args = parser.parse_args()

    if args.target_approvals <= 0:
        parser.error("--target-approvals must be greater than zero.")

    return args


def print_candidate(candidate, source_text, index, total):
    print("\n" + "=" * 80)
    print(f"Candidate {index}/{total}: {candidate.id}")
    print(f"Provider: {candidate.metadata.provider} / {candidate.metadata.model}")
    print(f"Source: {candidate.expected_source}")
    print(f"Difficulty: {candidate.difficulty}")
    print(f"Question: {candidate.question}")
    print(f"Expected answer: {candidate.expected_answer}")
    print("\nSource chunk:")
    print(source_text)


def main():
    args = parse_args()
    candidates = load_synthetic_candidates(args.candidates)
    golden_records = read_jsonl(args.golden) if args.golden.exists() else []
    required_chunk_ids = {candidate.metadata.source_chunk_id for candidate in candidates}
    chunks = load_source_chunks(args.chunks, min_words=1)
    source_text_by_id = {chunk["chunk_id"]: chunk["text"] for chunk in chunks if chunk["chunk_id"] in required_chunk_ids}

    approved_count = sum(candidate.metadata.review_status == "approved" for candidate in candidates)
    pending_candidates = [candidate for candidate in candidates if candidate.metadata.review_status == "pending"]

    for index, candidate in enumerate(pending_candidates, start=1):
        if approved_count >= args.target_approvals:
            break

        source_text = source_text_by_id.get(candidate.metadata.source_chunk_id, "[Source chunk not found]")
        print_candidate(candidate, source_text, index, len(pending_candidates))

        while True:
            decision = input("[a]pprove, [r]eject, [s]kip, or [q]uit: ").strip().lower()
            if decision in {"a", "r", "s", "q"}:
                break
            print("Enter a, r, s, or q.")

        if decision == "q":
            break
        if decision == "s":
            continue

        candidate.metadata.review_status = "approved" if decision == "a" else "rejected"
        candidate.metadata.reviewed_by = args.reviewer
        if decision == "a":
            approved_count += 1

        write_jsonl(candidates, args.candidates, overwrite=True)

    write_jsonl(candidates, args.candidates, overwrite=True)
    merged_records, added_records = merge_approved_candidates(golden_records, candidates)
    write_jsonl(merged_records, args.golden, overwrite=True)

    rejected_count = sum(candidate.metadata.review_status == "rejected" for candidate in candidates)
    pending_count = sum(candidate.metadata.review_status == "pending" for candidate in candidates)
    print(f"Approved candidates: {approved_count}")
    print(f"Rejected candidates: {rejected_count}")
    print(f"Pending candidates: {pending_count}")
    print(f"Added to golden set: {len(added_records)}")
    print(f"Golden set size: {len(merged_records)}")


if __name__ == "__main__":
    main()
