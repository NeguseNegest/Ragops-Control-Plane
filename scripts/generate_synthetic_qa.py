import argparse
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragops.evaluation.synthetic_qa import generate_synthetic_candidates, load_source_chunks, read_jsonl, write_jsonl  # noqa: E402
from ragops.generation.factory import create_generation_client  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Generate source-grounded synthetic QA candidates with multiple LLM providers.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/chunks.jsonl"), help="Input chunk JSONL path.")
    parser.add_argument("--golden", type=Path, default=Path("data/eval/golden_qa.jsonl"), help="Existing golden QA JSONL used for duplicate detection.")
    parser.add_argument("--output", type=Path, default=Path("data/eval/synthetic_qa_candidates.jsonl"), help="Output candidate JSONL path.")
    parser.add_argument("--providers", nargs="+", choices=("openai", "gemini"), default=["openai", "gemini"], help="Providers to use, in allocation order.")
    parser.add_argument("--count", type=int, default=100, help="Total number of valid unique candidates.")
    parser.add_argument("--pairs-per-chunk", type=int, default=5, help="Candidates requested in each provider call.")
    parser.add_argument("--min-source-words", type=int, default=80, help="Minimum source chunk length.")
    parser.add_argument("--seed", type=int, default=16, help="Deterministic source sampling seed.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional local environment file containing API keys.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing candidate output file.")
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be greater than zero.")
    if args.pairs_per_chunk <= 0:
        parser.error("--pairs-per-chunk must be greater than zero.")
    if args.min_source_words <= 0:
        parser.error("--min-source-words must be greater than zero.")
    if len(set(args.providers)) != len(args.providers):
        parser.error("--providers must not contain duplicates.")

    return args


def load_env_file(path):
    """Load simple KEY=VALUE entries without replacing exported values."""
    path = Path(path)
    if not path.exists():
        return

    with path.open(encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


def print_progress(event):
    if event["status"] == "skipped":
        print(f"[{event['provider']}] skipped {event['source']}: {event['reason']}", flush=True)
        return

    print(
        f"[{event['provider']}] +{event['added']} from {event['source']} "
        f"({event['provider_total']}/{event['provider_target']})",
        flush=True,
    )


def main():
    args = parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {args.output}. Pass --overwrite to replace it.")

    load_env_file(args.env_file)
    chunks = load_source_chunks(args.input, min_words=args.min_source_words)
    golden_records = read_jsonl(args.golden) if args.golden.exists() else []
    existing_questions = [record.get("question", "") for record in golden_records]
    provider_clients = {provider: create_generation_client(provider) for provider in args.providers}

    print(f"Loaded {len(chunks)} eligible source chunks.")
    print(f"Generating {args.count} candidates with: {', '.join(args.providers)}")
    candidates = generate_synthetic_candidates(
        chunks,
        provider_clients,
        count=args.count,
        pairs_per_chunk=args.pairs_per_chunk,
        seed=args.seed,
        existing_questions=existing_questions,
        progress=print_progress,
    )
    write_jsonl(candidates, args.output, overwrite=args.overwrite)

    provider_counts = Counter(candidate.metadata.provider for candidate in candidates)
    source_counts = Counter(candidate.expected_source.split("/", 1)[0] for candidate in candidates)
    print(f"Wrote {len(candidates)} candidates to {args.output}")
    print(f"Providers: {dict(sorted(provider_counts.items()))}")
    print(f"Sources: {dict(sorted(source_counts.items()))}")
    print("All candidates are pending review; review them before merging into the golden set.")


if __name__ == "__main__":
    main()
