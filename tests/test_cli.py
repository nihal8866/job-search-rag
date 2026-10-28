"""CLI parsing and JSON output checks without model inference."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

from job_rag.cli import build_parser, main
from test_results import sample_jobs


class CliTests(unittest.TestCase):
    def test_search_accepts_explicit_location_and_limit(self):
        args = build_parser().parse_args(["search", "Python", "--location", "Tokyo", "--top-k", "3"])
        self.assertEqual(args.location, "Tokyo")
        self.assertEqual(args.top_k, 3)

    def test_invalid_limit_is_rejected_before_loading_models(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["search", "query", "--top-k", "0"])

    def test_blank_query_is_rejected_before_loading_models(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["search", "  "])

    def test_blank_location_is_rejected_before_loading_models(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["search", "Python", "--location", "  "])

    def test_ask_uses_cpu_retrieval_by_default_and_local_generator(self):
        with patch("job_rag.pipeline.JobSearch") as pipeline, \
                patch("job_rag.ollama_local.OllamaGenerator"), \
                patch("job_rag.answers.GroundedAnswerService") as service:
            service.return_value.answer.return_value = {"answer": "text", "jobs": [], "cited_job_ids": []}
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(["ask", "Python"])
        self.assertEqual(status, 0)
        self.assertEqual(pipeline.call_args.args[0].device, "cpu")
        service.return_value.answer.assert_called_once_with("Python", top_k=2, location=None)

    def test_model_logs_do_not_corrupt_json_stdout(self):
        with patch("job_rag.pipeline.JobSearch") as factory:
            factory.return_value.search.side_effect = lambda *a, **kw: (print("model log"), sample_jobs())[1]
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(["search", "Python"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout.getvalue())["jobs"][0]["job_id"], "LF0001")
        self.assertIn("model log", stderr.getvalue())

    def test_failed_search_has_nonzero_status_and_no_fake_result(self):
        with patch("job_rag.pipeline.JobSearch", side_effect=FileNotFoundError("missing index")):
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(["search", "Python"])
        self.assertEqual(status, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("missing index", stderr.getvalue())
