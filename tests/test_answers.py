"""Provider-independent checks; fake outputs are not live LLM validation."""

import json
import unittest
from unittest.mock import Mock

from job_rag.answers import GeneratedAnswer, GroundedAnswerService, SYSTEM_PROMPT
from test_results import sample_jobs


class GroundedAnswerTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Mock()
        self.retriever.search.return_value = sample_jobs()
        self.generator = Mock()
        self.generator.generate.return_value = GeneratedAnswer(
            "The excerpt mentions Python and SQL [LF0001].", ["LF0001"]
        )
        self.service = GroundedAnswerService(self.retriever, self.generator)

    def test_prompt_has_evidence_and_answer_preserves_sources(self):
        result = self.service.answer("Python jobs", top_k=3, location="Tokyo")
        self.retriever.search.assert_called_once_with("Python jobs", top_k=3, location="Tokyo")
        system, prompt = self.generator.generate.call_args.args
        self.assertEqual(system, SYSTEM_PROMPT)
        payload = json.loads(prompt)
        self.assertEqual(payload["sources"][0]["evidence"], "Python and SQL experience.")
        self.assertNotIn("scores", payload["sources"][0])
        self.assertEqual(result["cited_job_ids"], ["LF0001"])

    def test_unknown_source_reference_is_rejected(self):
        self.generator.generate.return_value = GeneratedAnswer("A job [LF9999].", ["LF9999"])
        with self.assertRaisesRegex(RuntimeError, "source references"):
            self.service.answer("query")

    def test_inline_references_must_match_declared_references(self):
        self.generator.generate.return_value = GeneratedAnswer("A job [LF0001].", ["LF0002"])
        with self.assertRaisesRegex(RuntimeError, "source references"):
            self.service.answer("query")

    def test_uncited_answer_is_rejected(self):
        self.generator.generate.return_value = GeneratedAnswer("Python role.", [])
        with self.assertRaisesRegex(RuntimeError, "source references"):
            self.service.answer("query")

    def test_empty_generated_text_is_rejected(self):
        self.generator.generate.return_value = GeneratedAnswer("  ", ["LF0001"])
        with self.assertRaisesRegex(RuntimeError, "empty text"):
            self.service.answer("query")

    def test_no_jobs_does_not_call_generator(self):
        self.retriever.search.return_value = sample_jobs().iloc[:0]
        result = self.service.answer("query", location="unknown place")
        self.assertEqual(result["jobs"], [])
        self.assertEqual(result["cited_job_ids"], [])
        self.generator.generate.assert_not_called()

    def test_brackets_in_blockquoted_evidence_are_not_citations(self):
        self.generator.generate.return_value = GeneratedAnswer(
            "Selected evidence [LF0001]:\n> [Pay transparency] Python and SQL.", ["LF0001"],
        )
        result = self.service.answer("query")
        self.assertEqual(result["cited_job_ids"], ["LF0001"])
