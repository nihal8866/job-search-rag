"""Keyword tokenization and rank fusion shared by retrieval components."""

import re


def tokenize_keywords(text: str) -> list[str]:
    """Normalize terms, preserving C++/C# and separating common CJK characters."""
    text = text.casefold()
    text = re.sub(r"([\u4e00-\u9fff\u3040-\u30ff])", r" \1 ", text)
    return re.findall(r"c\+\+|c#|[^\W_]+", text)


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int,
) -> list[tuple[str, float]]:
    """Return unique chunk IDs and fused scores, ordered by score then ID.

    Each input list must contain unique chunk IDs in best-first order.
    A chunk can appear in multiple lists and receives one contribution per list.
    """
    if k < 0:
        raise ValueError("The RRF constant must be non-negative.")

    scores: dict[str, float] = {}

    for ranked_ids in rankings:
        if len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError("Each ranking must contain unique chunk IDs.")

        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
