"""Known-item metrics: unjudged jobs are not assumed irrelevant."""


def known_item_metrics(job_ids: list[str], target: str, k: int = 5) -> dict[str, float | int | None]:
    if k < 1 or not target:
        raise ValueError("Use a positive cutoff and a nonempty target ID.")
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("Rankings must contain distinct job IDs.")
    top = job_ids[:k]
    rank = top.index(target) + 1 if target in top else None
    return {
        "target_rank_at_k": rank,
        "hit_at_1": float(rank == 1),
        "hit_at_k": float(rank is not None),
        "reciprocal_rank_at_k": 1.0 / rank if rank else 0.0,
    }


def summarize_known_items(records: list[dict]) -> dict:
    if not records:
        raise ValueError("At least one evaluation record is required.")
    return {
        "case_count": len(records),
        **{key: sum(row[key] for row in records) / len(records)
           for key in ["hit_at_1", "hit_at_k", "reciprocal_rank_at_k"]},
    }
