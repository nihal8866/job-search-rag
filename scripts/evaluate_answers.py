"""Generate local answers for fixed evidence contexts for manual review."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import time

import pandas as pd

from job_rag.answers import GroundedAnswerService, SYSTEM_PROMPT
from job_rag.config import PROJECT_ROOT, Settings
from job_rag.ollama_local import OllamaGenerator


class FixedEvidence:
    """Control retrieval context; this experiment measures generation only."""

    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, top_k=None, location=None):
        return self.chunks


class RecordingGenerator:
    """Retain model output for review even if citation validation rejects it."""

    def __init__(self, generator):
        self.generator = generator
        self.last = None

    def generate(self, system_prompt, user_prompt):
        self.last = self.generator.generate(system_prompt, user_prompt)
        return self.last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["G01", "G02", "G03", "G04", "G05"])
    parser.add_argument("--output", default="evaluation/answer_results_current.json")
    args = parser.parse_args()
    settings = Settings()
    chunks = pd.read_json(settings.chunks_path, lines=True, convert_dates=False).set_index("chunk_id")
    cases = [
        ("G01", "Python backend developer working with APIs and databases",
         ["LF0265_chunk_000", "LF0247_chunk_001"],
         "Separate Python/GenAI from backend/database evidence; do not invent API work or classify messaging frameworks as databases."),
        ("G02", "What is the exact annual salary for the Senior Python Developer at EPAM?",
         ["LF0265_chunk_000"], "No salary amount in excerpt; say it is not established without inventing a range."),
        ("G03", "Is the EPAM Senior Python Developer vacancy still accepting applications today?",
         ["LF0265_chunk_000"], "Historical listing date cannot confirm current application availability."),
        ("G04", "What annual salary range is listed for the DIR, Equities Quant job?",
         ["LF0001_chunk_001"], "Quote the stated $225,000.00 - $300,000.00 annualized range and preserve offer qualifiers."),
        ("G05", "What is the recorded job location for the EPAM Senior Python Developer?",
         ["LF0265_chunk_000"], "Use the provided location metadata: Brezno, Slovakia; do not infer remote eligibility."),
    ]
    results = []
    for case_id, query, ids, rubric in cases:
        if args.case is not None and args.case != case_id:
            continue
        generator = RecordingGenerator(OllamaGenerator(settings))
        selected = chunks.loc[ids].copy()
        selected.insert(0, "rank", range(1, len(selected) + 1))
        service = GroundedAnswerService(FixedEvidence(selected), generator)
        start = time.perf_counter()
        try:
            payload = service.answer(query)
            row = {"response": payload, "reference_validation": "passed"}
        except (ValueError, RuntimeError) as error:
            row = {"error_type": type(error).__name__, "reference_validation": "not passed"}
            if generator.last is not None:
                row["rejected_generated_output"] = asdict(generator.last)
        results.append({"case_id": case_id, "query": query, "source_chunk_ids": ids,
                        "review_rubric": rubric, "seconds": time.perf_counter() - start, **row})
        print(f"Completed {case_id}: {row['reference_validation']}", flush=True)
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": settings.llm_model, "temperature": 0,
        "generation_mode": "LLM source-line selection with application-rendered quotations and citations",
        "system_prompt": SYSTEM_PROMPT,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "chunks_sha256": hashlib.sha256(settings.chunks_path.read_bytes()).hexdigest(),
        "method": "Fixed source chunks with actual local generation; not an end-to-end retrieval evaluation.",
        "limitations": "Fixed-evidence development cases, one run each. Semantic judgments require manual source review; citation consistency alone is insufficient.",
        "results": results,
    }
    path = PROJECT_ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
