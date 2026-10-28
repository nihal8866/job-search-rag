"""Command-line access to ingestion and reranked job search."""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import json
from pathlib import Path
import sys

from job_rag.config import PROJECT_ROOT, Settings


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and search the LF Jobs dataset.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    commands = parser.add_subparsers(dest="command", required=True)

    search = commands.add_parser("search", help="Return reranked job evidence as JSON.")
    search.add_argument("query")
    search.add_argument("--top-k", type=positive_integer, default=5)
    search.add_argument("--location", help="Literal text to match in recorded locations.")

    ingest = commands.add_parser("ingest", help="Rebuild cleaned data, embeddings, and Chroma.")
    ingest.add_argument("--dataset", type=Path, help="Defaults to data/raw/LF Jobs.xlsx.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "search":
        if not args.query.strip():
            parser.error("The query must contain non-whitespace text.")
        if args.location is not None and not args.location.strip():
            parser.error("The location filter must contain non-whitespace text.")
    root = args.project_root.resolve()
    settings = replace(
        Settings(), device=args.device,
        chunks_path=root / "data" / "processed" / "job_chunks.jsonl",
        database_path=root / "data" / "indexes" / "chroma",
    )

    try:
        # Keep model/progress messages off stdout so stdout remains valid JSON.
        with redirect_stdout(sys.stderr):
            if args.command == "search":
                from job_rag.pipeline import JobSearch
                from job_rag.results import serialize_jobs

                pipeline = JobSearch(settings)
                jobs = pipeline.search(args.query, top_k=args.top_k, location=args.location)
                payload = {
                    "query": args.query.strip(), "location_filter": args.location,
                    "jobs": serialize_jobs(jobs),
                }
            else:
                from job_rag.ingestion import ingest_dataset

                dataset = args.dataset or root / "data" / "raw" / "LF Jobs.xlsx"
                payload = ingest_dataset(dataset, settings)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
    return 0
