"""Combine model-based vector retrieval with BM25 and rank fusion."""

import pandas as pd
from sentence_transformers import SentenceTransformer

from job_rag.config import Settings
from job_rag.index import RetrievalIndex
from job_rag.retrieval import reciprocal_rank_fusion


class HybridRetriever:
    """Reuse loaded resources across queries and return ranked candidate chunks."""

    def __init__(self, settings: Settings):
        if settings.candidate_count < 1:
            raise ValueError("The candidate count must be positive.")
        if settings.rrf_k < 0:
            raise ValueError("The RRF constant must be non-negative.")

        self.settings = settings
        self.index = RetrievalIndex(settings)
        self.embedding_model = SentenceTransformer(
            settings.embedding_model,
            device=settings.device,
        )

    def search(self, query: str, location: str | None = None) -> pd.DataFrame:
        """Retrieve fused chunks; job deduplication belongs after reranking."""
        if not query.strip():
            raise ValueError("The query must contain non-whitespace text.")

        query_text = "query: " + query.strip()
        token_ids = self.embedding_model.tokenizer.encode(
            query_text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
        if len(token_ids) > self.embedding_model.max_seq_length:
            raise ValueError("The query exceeds the embedding model's token limit.")

        allowed_ids = self.index.matching_chunk_ids(location)
        if allowed_ids == []:
            candidates = self.index.chunks.iloc[:0].copy()
            for column in ["rrf_score", "vector_distance", "bm25_score"]:
                candidates[column] = pd.Series(index=candidates.index, dtype=float)
            return candidates

        query_embedding = self.embedding_model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vector_matches = self.index.search_vectors(
            query_embedding, limit=self.settings.candidate_count, allowed_ids=allowed_ids
        )
        keyword_matches = self.index.search_keywords(
            query.strip(), limit=self.settings.candidate_count, allowed_ids=allowed_ids
        )

        fused = reciprocal_rank_fusion(
            [
                [chunk_id for chunk_id, _ in vector_matches],
                [chunk_id for chunk_id, _ in keyword_matches],
            ],
            k=self.settings.rrf_k,
        )

        candidates = self.index.chunks.loc[
            [chunk_id for chunk_id, _ in fused]
        ].copy()
        candidates["rrf_score"] = [score for _, score in fused]
        candidates["vector_distance"] = candidates.index.map(dict(vector_matches))
        candidates["bm25_score"] = candidates.index.map(dict(keyword_matches))
        return candidates
