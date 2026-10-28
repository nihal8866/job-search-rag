"""Vector-query boundary checks without opening a database."""

import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from job_rag.index import RetrievalIndex


class VectorSearchTests(unittest.TestCase):
    def setUp(self):
        self.index = RetrievalIndex.__new__(RetrievalIndex)
        self.index.chunk_ids = ["a", "b"]
        self.index.collection = Mock()
        self.index.collection.query.return_value = {
            "ids": [["a", "b"]], "distances": [[0.1, 0.2]]
        }

    def test_single_vector_is_accepted_and_request_is_capped_to_collection_size(self):
        result = self.index.search_vectors(np.array([1.0, 0.0]), limit=20)
        self.assertEqual(result, [("a", 0.1), ("b", 0.2)])
        self.assertEqual(self.index.collection.query.call_args.kwargs["n_results"], 2)

    def test_batch_of_multiple_queries_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.index.search_vectors(np.ones((2, 3)), limit=2)
        self.index.collection.query.assert_not_called()

    def test_nonfinite_query_vector_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.index.search_vectors(np.array([np.nan, 1.0]), limit=2)
        self.index.collection.query.assert_not_called()

    def test_nonpositive_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            self.index.search_vectors(np.ones(3), limit=0)

    def test_location_subset_is_sent_to_vector_store(self):
        self.index.collection.query.return_value = {"ids": [["b"]], "distances": [[0.2]]}
        self.index.search_vectors(np.ones(3), limit=20, allowed_ids=["b"])
        self.assertEqual(self.index.collection.query.call_args.kwargs["ids"], ["b"])
        self.assertEqual(self.index.collection.query.call_args.kwargs["n_results"], 1)

    def test_empty_location_subset_never_queries_vector_store(self):
        self.assertEqual(self.index.search_vectors(np.ones(3), 5, allowed_ids=[]), [])
        self.index.collection.query.assert_not_called()


class LocationTests(unittest.TestCase):
    def setUp(self):
        self.index = RetrievalIndex.__new__(RetrievalIndex)
        self.index.chunks = pd.DataFrame({
            "Job Location": ["New York, NY, Tokyo, Japan", "York, UK", None, "A+B City"]
        }, index=["a", "b", "c", "d"])

    def test_case_insensitive_match_inside_multi_location_field(self):
        self.assertEqual(self.index.matching_chunk_ids(" new YORK "), ["a"])

    def test_missing_locations_do_not_match(self):
        self.assertEqual(self.index.matching_chunk_ids("Japan"), ["a"])

    def test_filter_is_literal_not_a_regular_expression(self):
        self.assertEqual(self.index.matching_chunk_ids("A+B"), ["d"])

    def test_unknown_location_returns_an_empty_subset(self):
        self.assertEqual(self.index.matching_chunk_ids("Atlantis"), [])

    def test_omitted_filter_is_distinct_from_no_matches(self):
        self.assertIsNone(self.index.matching_chunk_ids(None))

    def test_blank_filter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-whitespace"):
            self.index.matching_chunk_ids("  ")

    def test_keyword_filter_is_applied_before_result_limit(self):
        self.index.chunk_ids = ["a", "b", "c"]
        self.index.bm25 = Mock()
        self.index.bm25.get_scores.return_value = np.array([10.0, 2.0, 1.0])
        result = self.index.search_keywords("Python", limit=1, allowed_ids=["b", "c"])
        self.assertEqual(result, [("b", 2.0)])


if __name__ == "__main__":
    unittest.main()
