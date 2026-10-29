"""Check JSON output, including missing metadata and scores."""

import json
import unittest

import numpy as np
import pandas as pd

from job_rag.results import serialize_jobs


def sample_jobs():
    return pd.DataFrame({
        "rank": [1], "ID": ["LF0001"], "Job Title": ["Python 工程师"],
        "Company Name": ["Example"], "Job Location": [np.nan],
        "Job Level": ["Mid Level"], "Publication Date": ["2025-01-01T00:00:00Z"],
        "chunk_text": ["Python and SQL experience."], "rrf_score": [0.03],
        "reranker_score": [0.4], "vector_distance": [0.1], "bm25_score": [np.nan],
    }, index=["LF0001_chunk_000"])


class ResultTests(unittest.TestCase):
    def test_missing_values_become_json_null_and_evidence_is_retained(self):
        records = serialize_jobs(sample_jobs())
        self.assertIsNone(records[0]["location"])
        self.assertIsNone(records[0]["scores"]["bm25"])
        self.assertEqual(records[0]["source_chunk_id"], "LF0001_chunk_000")
        self.assertEqual(records[0]["evidence"], "Python and SQL experience.")
        decoded = json.loads(json.dumps(records, ensure_ascii=False, allow_nan=False))
        self.assertEqual(decoded[0]["title"], "Python 工程师")

    def test_empty_table_produces_empty_list(self):
        self.assertEqual(serialize_jobs(sample_jobs().iloc[:0]), [])
