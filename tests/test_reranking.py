"""Check pair limits and evidence/score alignment without model downloads."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from job_rag.config import Settings
from job_rag.reranking import CrossEncoderReranker


class RerankingTests(unittest.TestCase):
    def setUp(self):
        model_patch = patch("job_rag.reranking.CrossEncoder")
        self.addCleanup(model_patch.stop)
        self.model = model_patch.start().return_value
        self.model.tokenizer.return_value = {"input_ids": [0, 1, 2]}
        self.model.predict.return_value = np.array([0.1, 0.9])
        self.reranker = CrossEncoderReranker(Settings())
        self.candidates = pd.DataFrame({
            "ID": ["job_a", "job_b"],
            "embedding_text": ["passage: Python backend", "passage: SQL services"],
            "chunk_text": ["Python backend", "SQL services"],
            "rrf_score": [0.03, 0.02],
        }, index=["a", "b"])

    def test_scores_stay_attached_to_their_original_evidence(self):
        result = self.reranker.rerank(" backend ", self.candidates)
        self.assertEqual(result.index.tolist(), ["b", "a"])
        self.assertEqual(result.loc["b", "chunk_text"], "SQL services")
        self.assertEqual(result.loc["b", "rrf_score"], 0.02)
        self.assertEqual(self.candidates.index.tolist(), ["a", "b"])
        self.assertEqual(self.model.predict.call_args.args[0],
                         [("backend", "Python backend"), ("backend", "SQL services")])

    def test_oversized_pair_is_rejected_before_scoring(self):
        self.model.tokenizer.return_value = {"input_ids": list(range(513))}
        with self.assertRaisesRegex(ValueError, "token limit"):
            self.reranker.rerank("query", self.candidates)
        self.model.predict.assert_not_called()

    def test_empty_candidates_do_not_run_inference(self):
        result = self.reranker.rerank("query", self.candidates.iloc[:0])
        self.assertTrue(result.empty)
        self.assertIn("reranker_score", result)
        self.model.predict.assert_not_called()

    def test_tied_scores_keep_the_original_fused_order(self):
        self.model.predict.return_value = np.array([0.5, 0.5])
        result = self.reranker.rerank("query", self.candidates)
        self.assertEqual(result.index.tolist(), ["a", "b"])

    def test_nonfinite_scores_are_rejected(self):
        self.model.predict.return_value = np.array([np.nan, 0.9])
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            self.reranker.rerank("query", self.candidates)

    def test_missing_scores_are_rejected(self):
        self.model.predict.return_value = np.array([0.9])
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.reranker.rerank("query", self.candidates)
