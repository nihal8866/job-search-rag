"""Combine hybrid search, reranking, and job-level results."""

import pandas as pd

from job_rag.config import Settings
from job_rag.reranking import CrossEncoderReranker
from job_rag.search import HybridRetriever


class JobSearch:
    """Load the search models once and reuse them across queries."""

    def __init__(self, settings: Settings):
        if settings.top_k < 1:
            raise ValueError("The default result limit must be positive.")
        self.settings = settings
        self.retriever = HybridRetriever(settings)
        self.reranker = CrossEncoderReranker(settings)

    def search(
        self, query: str, top_k: int | None = None, location: str | None = None
    ) -> pd.DataFrame:
        """Rank jobs by their highest-scoring candidate chunk."""
        limit = self.settings.top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("The result limit must be positive.")

        candidates = self.retriever.search(query, location=location)
        reranked = self.reranker.rerank(query, candidates)
        jobs = reranked.drop_duplicates(subset="ID", keep="first").head(limit).copy()
        jobs.insert(0, "rank", range(1, len(jobs) + 1))
        return jobs
