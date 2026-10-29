"""Score hybrid candidates with a multilingual cross-encoder."""

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from job_rag.config import Settings


class CrossEncoderReranker:
    """Load the reranker once and check pair lengths before scoring."""

    def __init__(self, settings: Settings):
        if settings.reranker_batch_size < 1 or settings.reranker_max_length < 1:
            raise ValueError("Reranker batch size and token limit must be positive.")
        self.settings = settings
        self.model = CrossEncoder(
            settings.reranker_model,
            device=settings.device,
            max_length=settings.reranker_max_length,
        )

    def rerank(self, query: str, candidates: pd.DataFrame) -> pd.DataFrame:
        """Sort candidate chunks by reranker score and keep their metadata."""
        if not query.strip():
            raise ValueError("The query must contain non-whitespace text.")

        result = candidates.copy()
        if result.empty:
            result["reranker_score"] = pd.Series(index=result.index, dtype=float)
            return result

        texts = result["embedding_text"].str.removeprefix("passage: ").tolist()
        pairs = [(query.strip(), text) for text in texts]
        for chunk_id, (question, text) in zip(result.index, pairs, strict=True):
            tokens = self.model.tokenizer(
                question,
                text,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )["input_ids"]
            if len(tokens) > self.settings.reranker_max_length:
                raise ValueError(f"Query and chunk {chunk_id} exceed the reranker token limit.")

        scores = self.model.predict(
            pairs,
            batch_size=self.settings.reranker_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(scores).reshape(-1)
        if len(scores) != len(result) or not np.isfinite(scores).all():
            raise RuntimeError("The reranker returned invalid or missing scores.")

        result["reranker_score"] = scores
        return result.sort_values("reranker_score", ascending=False, kind="stable")
