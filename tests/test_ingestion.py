"""Chunk-boundary and source-data checks without model downloads."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from job_rag.config import Settings
from job_rag.ingestion import SOURCE_COLUMNS, chunk_jobs, clean_description, prepare_jobs


class CharacterTokenizer:
    """Tiny deterministic tokenizer for boundary tests, not a real model."""

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def encode(self, text, add_special_tokens=True, **kwargs):
        return list(range(len(text) + (2 if add_special_tokens else 0)))


def source_rows():
    return pd.DataFrame([{
        "ID": "LF0001", "Job Category": "Software Engineering", "Job Title": "Engineer",
        "Company Name": "Example", "Publication Date": "2025-01-01T00:00:00Z",
        "Job Location": None, "Job Level": "Mid Level", "Tags": None,
        "Job Description": "<p>软件开发</p><p>C++ and Python</p>",
    }], columns=SOURCE_COLUMNS)


class IngestionTests(unittest.TestCase):
    def test_html_cleaning_preserves_multilingual_and_technical_text(self):
        self.assertEqual(clean_description("<p>软件</p><p>C++ &amp; C#</p>"), "软件\nC++ & C#")

    def prepare_fixture(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"fixture.xlsx"
            path.touch()
            with patch("job_rag.ingestion.pd.read_excel", return_value=rows):
                return prepare_jobs(path)

    def test_preparation_preserves_original_fields_and_missing_metadata(self):
        rows = source_rows()
        result = self.prepare_fixture(rows)
        self.assertEqual(result.loc[0, "Job Description"], rows.loc[0, "Job Description"])
        self.assertTrue(pd.isna(result.loc[0, "Job Location"]))
        self.assertEqual(result.loc[0, "Clean Description"], "软件开发\nC++ and Python")

    def test_duplicate_job_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self.prepare_fixture(pd.concat([source_rows(), source_rows()], ignore_index=True))

    def test_invalid_dates_are_rejected(self):
        rows = source_rows()
        rows.loc[0, "Publication Date"] = "not a date"
        with self.assertRaisesRegex(ValueError, "dates"):
            self.prepare_fixture(rows)

    def test_overlap_covers_the_tail_without_a_redundant_final_window(self):
        rows = source_rows()
        rows["Clean Description"] = ["abcdefghij"]
        settings = replace(Settings(), chunk_size=6, chunk_overlap=2)
        chunks = chunk_jobs(rows, CharacterTokenizer(), settings, max_length=1000)
        self.assertEqual(chunks["chunk_text"].tolist(), ["abcdef", "efghij"])
        self.assertEqual(chunks["chunk_id"].tolist(), ["LF0001_chunk_000", "LF0001_chunk_001"])

    def test_metadata_and_prefix_are_included_in_length_limit(self):
        rows = source_rows()
        rows["Clean Description"] = ["short"]
        with self.assertRaisesRegex(ValueError, "model limit"):
            chunk_jobs(rows, CharacterTokenizer(), Settings(), max_length=10)

    def test_overlap_cannot_prevent_forward_progress(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_jobs(source_rows(), CharacterTokenizer(),
                       replace(Settings(), chunk_size=10, chunk_overlap=10), max_length=512)
