"""Check job-level ranking and result limits without inference."""

import unittest
from unittest.mock import patch

import pandas as pd

from job_rag.config import Settings
from job_rag.pipeline import JobSearch


class PipelineTests(unittest.TestCase):
    def setUp(self):
        retriever_patch = patch("job_rag.pipeline.HybridRetriever")
        reranker_patch = patch("job_rag.pipeline.CrossEncoderReranker")
        self.addCleanup(retriever_patch.stop)
        self.addCleanup(reranker_patch.stop)
        self.retriever = retriever_patch.start().return_value
        self.reranker = reranker_patch.start().return_value
        self.reranker.rerank.return_value = pd.DataFrame({
            "ID": ["job_a", "job_a", "job_b"],
            "chunk_text": ["best a", "other a", "best b"],
            "reranker_score": [0.9, 0.8, 0.7],
        }, index=["a1", "a2", "b1"])
        self.pipeline = JobSearch(Settings())

    def test_duplicate_jobs_keep_the_best_candidate_evidence(self):
        result = self.pipeline.search("query")
        self.assertEqual(result["ID"].tolist(), ["job_a", "job_b"])
        self.assertEqual(result.index.tolist(), ["a1", "b1"])
        self.assertEqual(result["rank"].tolist(), [1, 2])

    def test_requested_limit_is_applied_after_deduplication(self):
        result = self.pipeline.search("query", top_k=2)
        self.assertEqual(len(result), 2)

    def test_nonpositive_limit_does_not_start_retrieval(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            self.pipeline.search("query", top_k=0)
        self.retriever.search.assert_not_called()

    def test_empty_candidates_return_an_empty_job_table(self):
        self.reranker.rerank.return_value = self.reranker.rerank.return_value.iloc[:0]
        result = self.pipeline.search("query")
        self.assertTrue(result.empty)
        self.assertIn("rank", result)

    def test_location_constraint_is_forwarded_to_candidate_retrieval(self):
        self.pipeline.search("query", location="Tokyo")
        self.retriever.search.assert_called_once_with("query", location="Tokyo")
