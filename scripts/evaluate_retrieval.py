"""Compare vector, hybrid, and reranked search on known target jobs."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import pandas as pd

from job_rag.config import PROJECT_ROOT, Settings
from job_rag.evaluation import known_item_metrics, summarize_known_items
from job_rag.retrieval import reciprocal_rank_fusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "evaluation/known_item_queries.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evaluation/retrieval_results.json")
    args = parser.parse_args()
    settings = replace(Settings(), device=args.device)
    suite_bytes = args.queries.read_bytes()
    suite = json.loads(suite_bytes)
    cleaned = pd.read_json(PROJECT_ROOT / "data/processed/jobs_cleaned.jsonl", lines=True, convert_dates=False).set_index("ID")
    cases = suite["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Cases must have distinct IDs.")
    for case in cases:
        source = cleaned.loc[case["target_job_id"], "Clean Description"]
        if not all(quote in source for quote in case["evidence"]):
            raise ValueError(f"Source evidence does not match case {case['id']}.")

    from job_rag.pipeline import JobSearch

    start = time.perf_counter()
    pipeline = JobSearch(settings)
    load_seconds = time.perf_counter() - start
    index = pipeline.retriever.index
    model = pipeline.retriever.embedding_model
    rows = []
    for case in cases:
        query = case["query"]
        start = time.perf_counter()
        embedding = model.encode(["query: " + query], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        vectors = index.search_vectors(embedding, limit=settings.candidate_count)
        vector_seconds = time.perf_counter() - start
        start = time.perf_counter()
        keywords = index.search_keywords(query, limit=settings.candidate_count)
        fused = reciprocal_rank_fusion([[i for i, _ in vectors], [i for i, _ in keywords]], settings.rrf_k)
        candidates = index.chunks.loc[[i for i, _ in fused]].copy()
        candidates["rrf_score"] = [score for _, score in fused]
        hybrid_seconds = vector_seconds + time.perf_counter() - start
        start = time.perf_counter()
        reranked = pipeline.reranker.rerank(query, candidates)
        reranked_seconds = hybrid_seconds + time.perf_counter() - start
        vector_chunks = index.chunks.loc[[i for i, _ in vectors]]
        for name, chunks, elapsed in [("vector", vector_chunks, vector_seconds),
                                     ("hybrid", candidates, hybrid_seconds),
                                     ("reranked", reranked, reranked_seconds)]:
            jobs = chunks.drop_duplicates("ID").head(5)
            ids = jobs["ID"].tolist()
            rows.append({"case_id": case["id"], "method": name,
                         "job_ids": ids, "chunk_ids": jobs.index.tolist(),
                         "seconds": elapsed, **known_item_metrics(ids, case["target_job_id"], 5)})
        print(f"Completed {case['id']}: " + ", ".join(
            f"{r['method']} rank={r['target_rank_at_k']}" for r in rows[-3:]), flush=True)

    summary = {}
    for method in ["vector", "hybrid", "reranked"]:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {**summarize_known_items(selected),
                           "mean_seconds": statistics.mean(r["seconds"] for r in selected),
                           "median_seconds": statistics.median(r["seconds"] for r in selected)}
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff": 5, "device": args.device, "candidate_count": settings.candidate_count,
        "rrf_k": settings.rrf_k, "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model, "load_seconds": load_seconds,
        "query_sha256": hashlib.sha256(suite_bytes).hexdigest(),
        "chunks_sha256": hashlib.sha256(settings.chunks_path.read_bytes()).hexdigest(),
        "limitations": ["Source-derived development queries; not held out or independently labeled.",
                        "Only target discovery is judged; other results can also be relevant.",
                        "One sequential run; timings reuse common stages and exclude model load and generation.",
                        "All methods have up to 20 chunks per retriever; unique-job counts can be below 5."],
        "summary": summary, "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
