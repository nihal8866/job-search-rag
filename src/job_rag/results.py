"""Format ranked jobs and evidence for JSON output."""

import pandas as pd


def serialize_jobs(jobs: pd.DataFrame) -> list[dict[str, object]]:
    """Keep source IDs and text; convert missing values to JSON null."""
    def optional_text(value: object) -> str | None:
        return None if pd.isna(value) else str(value)

    def optional_score(value: object) -> float | None:
        return None if pd.isna(value) else float(value)

    records = []
    for chunk_id, row in jobs.iterrows():
        records.append({
            "rank": int(row["rank"]),
            "job_id": str(row["ID"]),
            "title": str(row["Job Title"]),
            "company": str(row["Company Name"]),
            "location": optional_text(row.get("Job Location")),
            "level": optional_text(row.get("Job Level")),
            "publication_date": optional_text(row.get("Publication Date")),
            "source_chunk_id": str(chunk_id),
            "evidence": str(row["chunk_text"]),
            "scores": {
                "rrf": optional_score(row.get("rrf_score")),
                "reranker": optional_score(row.get("reranker_score")),
                "vector_distance": optional_score(row.get("vector_distance")),
                "bm25": optional_score(row.get("bm25_score")),
            },
        })
    return records
