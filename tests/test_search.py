"""Check query handling and candidate assembly without loading model weights."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from job_rag.config import Settings
from job_rag.search import HybridRetriever


class HybridSearchTests(unittest.TestCase):
    def setUp(self):
        index_patch = patch("job_rag.search.RetrievalIndex")
        model_patch = patch("job_rag.search.SentenceTransformer")
        self.addCleanup(index_patch.stop)
        self.addCleanup(model_patch.stop)
        self.index = index_patch.start().return_value
        self.model = model_patch.start().return_value
        self.model.max_seq_length = 512
        self.model.tokenizer.encode.return_value = [0, 1, 2]
        self.model.encode.return_value = np.array([[1.0, 0.0]])
        self.index.chunks = pd.DataFrame(
            {"chunk_id": ["a", "b", "c"], "ID": ["job_a", "job_b", "job_c"]}
        ).set_index("chunk_id", drop=False)
        self.index.search_vectors.return_value = [("a", 0.1), ("b", 0.2)]
        self.index.matching_chunk_ids.return_value = None
        self.index.search_keywords.return_value = [("b", 3.0), ("c", 2.0)]
        self.retriever = HybridRetriever(Settings())

    def test_combines_candidates_and_preserves_source_score_alignment(self):
        result = self.retriever.search(" Python backend ")
        self.assertEqual(result.index.tolist(), ["b", "a", "c"])
        self.assertEqual(result.loc["b", "ID"], "job_b")
        self.assertEqual(result.loc["b", "vector_distance"], 0.2)
        self.assertEqual(result.loc["b", "bm25_score"], 3.0)
        self.assertTrue(pd.isna(result.loc["a", "bm25_score"]))
        self.assertTrue(pd.isna(result.loc["c", "vector_distance"]))
        self.assertEqual(self.model.encode.call_args.args[0], ["query: Python backend"])

    def test_blank_query_does_not_run_inference(self):
        with self.assertRaisesRegex(ValueError, "non-whitespace"):
            self.retriever.search(" \n ")
        self.model.encode.assert_not_called()
        self.index.search_vectors.assert_not_called()

    def test_oversized_query_is_rejected_instead_of_truncated(self):
        self.model.tokenizer.encode.return_value = list(range(513))
        with self.assertRaisesRegex(ValueError, "token limit"):
            self.retriever.search("long query")
        self.model.encode.assert_not_called()

    def test_no_keyword_hits_does_not_remove_vector_candidates(self):
        self.index.search_keywords.return_value = []
        result = self.retriever.search("semantic query")
        self.assertEqual(result.index.tolist(), ["a", "b"])
        self.assertTrue(result["bm25_score"].isna().all())

    def test_same_location_subset_is_applied_to_both_retrievers(self):
        self.index.matching_chunk_ids.return_value = ["b"]
        self.index.search_vectors.return_value = [("b", 0.2)]
        self.index.search_keywords.return_value = [("b", 3.0)]
        result = self.retriever.search("backend", location="Tokyo")
        self.index.matching_chunk_ids.assert_called_once_with("Tokyo")
        self.assertEqual(self.index.search_vectors.call_args.kwargs["allowed_ids"], ["b"])
        self.assertEqual(self.index.search_keywords.call_args.kwargs["allowed_ids"], ["b"])
        self.assertEqual(result.index.tolist(), ["b"])

    def test_unmatched_location_does_not_fall_back_to_unfiltered_results(self):
        self.index.matching_chunk_ids.return_value = []
        result = self.retriever.search("backend", location="No such location")
        self.assertTrue(result.empty)
        self.model.encode.assert_not_called()
        self.index.search_vectors.assert_not_called()
        self.index.search_keywords.assert_not_called()


if __name__ == "__main__":
    unittest.main()
