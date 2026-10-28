"""Load the persisted vector collection and its matching keyword corpus."""

import chromadb
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from job_rag.config import Settings
from job_rag.retrieval import tokenize_keywords


KEYWORD_COLUMNS = [
    "Job Title",
    "Company Name",
    "Job Category",
    "Job Level",
    "Job Location",
    "Tags",
    "chunk_text",
]


class RetrievalIndex:
    """Keep keyword scores and vector-store records linked by chunk ID."""

    def __init__(self, settings: Settings):
        self.settings = settings

        if not settings.chunks_path.is_file():
            raise FileNotFoundError("Chunk records are missing. Run notebook 02 first.")
        if not settings.database_path.is_dir():
            raise FileNotFoundError("The vector database is missing. Run notebook 03 first.")

        chunks = pd.read_json(settings.chunks_path, lines=True, convert_dates=False)
        required = set(KEYWORD_COLUMNS) | {"chunk_id", "ID", "embedding_text"}
        missing = required - set(chunks.columns)
        if missing:
            raise ValueError(f"Missing chunk columns: {', '.join(sorted(missing))}")
        if chunks.empty:
            raise ValueError("The chunk file contains no records.")
        valid_ids = chunks["chunk_id"].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        if not valid_ids.all():
            raise ValueError("Every chunk must have a non-empty string ID.")
        if not chunks["chunk_id"].is_unique:
            raise ValueError("Chunk IDs must be unique.")
        valid_text = chunks["chunk_text"].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        if not valid_text.all():
            raise ValueError("Every chunk must contain non-empty text.")

        self.client = chromadb.PersistentClient(path=str(settings.database_path))
        self.collection = self.client.get_collection(
            name=settings.collection_name,
            embedding_function=None,
        )

        model_name = (self.collection.metadata or {}).get("embedding_model")
        if model_name != settings.embedding_model:
            raise ValueError("The collection's embedding model does not match the settings.")

        self.chunk_ids = chunks["chunk_id"].tolist()
        stored_ids = set(self.collection.get(include=[])["ids"])
        if stored_ids != set(self.chunk_ids):
            raise ValueError("The saved chunks and vector collection contain different IDs.")

        self.chunks = chunks.set_index("chunk_id", drop=False)
        texts = chunks[KEYWORD_COLUMNS].fillna("").astype(str).agg("\n".join, axis=1)
        self.bm25 = BM25Okapi([tokenize_keywords(text) for text in texts])

    def matching_chunk_ids(self, location: str | None) -> list[str] | None:
        """Match literal location text, ignoring case; None means no filter."""
        if location is None:
            return None
        if not location.strip():
            raise ValueError("The location filter must contain non-whitespace text.")

        locations = self.chunks["Job Location"].fillna("").str.casefold()
        matches = locations.str.contains(location.strip().casefold(), regex=False)
        return self.chunks.index[matches].tolist()

    def search_keywords(
        self, query: str, limit: int, allowed_ids: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Return positive-scoring chunk IDs in descending BM25 score order."""
        if limit < 1:
            raise ValueError("The result limit must be positive.")

        tokens = tokenize_keywords(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        positions = np.argsort(-scores, kind="stable")
        positions = positions[scores[positions] > 0]
        if allowed_ids is not None:
            positions = positions[np.isin(
                [self.chunk_ids[position] for position in positions], allowed_ids
            )]
        positions = positions[:limit]
        return [(self.chunk_ids[position], float(scores[position])) for position in positions]

    def search_vectors(
        self, query_embedding: np.ndarray, limit: int,
        allowed_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return chunk IDs and cosine distances for one query vector."""
        if limit < 1:
            raise ValueError("The result limit must be positive.")

        vector = np.asarray(query_embedding, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.ndim != 2 or vector.shape[0] != 1 or vector.shape[1] == 0:
            raise ValueError("Supply exactly one non-empty query vector.")
        if not np.isfinite(vector).all():
            raise ValueError("Query vector values must be finite.")
        if allowed_ids == []:
            return []

        available = len(self.chunk_ids) if allowed_ids is None else len(allowed_ids)
        result = self.collection.query(
            query_embeddings=vector.tolist(),
            ids=allowed_ids,
            n_results=min(limit, available),
            include=["distances"],
        )
        distances = result["distances"]
        if distances is None:
            raise RuntimeError("The vector store did not return distances.")
        return [
            (chunk_id, float(distance))
            for chunk_id, distance in zip(result["ids"][0], distances[0], strict=True)
        ]
