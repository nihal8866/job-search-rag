"""HTTP contract checks with injected services, without model loading."""

import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from job_rag.api import create_app
from job_rag.results import serialize_jobs
from test_results import sample_jobs


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.service = Mock()
        self.service.answer.return_value = {
            "answer": "Python and SQL are mentioned [LF0001].",
            "cited_job_ids": ["LF0001"], "jobs": serialize_jobs(sample_jobs()),
        }
        self.factory = Mock(return_value=self.service)
        self.app = create_app(self.factory)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_success_normalizes_input_and_preserves_evidence(self):
        response = self.client.post("/api/query", json={
            "query": " Python jobs ", "top_k": 2, "location": " Tokyo ",
        })
        self.assertEqual(response.status_code, 200)
        self.service.answer.assert_called_once_with("Python jobs", top_k=2, location="Tokyo")
        self.assertEqual(response.json()["jobs"][0]["evidence"], "Python and SQL experience.")
        self.assertEqual(response.json()["cited_job_ids"], ["LF0001"])
        self.assertEqual(response.json()["location_filter"], "Tokyo")

    def test_default_limit_and_service_reuse(self):
        for _ in range(2):
            self.assertEqual(self.client.post("/api/query", json={"query": "Python"}).status_code, 200)
        self.factory.assert_called_once_with()
        self.service.answer.assert_called_with("Python", top_k=2, location=None)

    def test_invalid_inputs_do_not_run_pipeline(self):
        for payload in [{}, {"query": " "}, {"query": 5}, {"query": "x" * 1001},
                        {"query": "x", "top_k": 0}, {"query": "x", "top_k": 6},
                        {"query": "x", "top_k": True}, {"query": "x", "top_k": "2"},
                        {"query": "x", "location": " "},
                        {"query": "x", "location": "x" * 301},
                        {"query": "x", "model": "other"}]:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/api/query", json=payload).status_code, 422)
        self.service.answer.assert_not_called()

    def test_malformed_json(self):
        response = self.client.post("/api/query", content="{", headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 422)

    def test_no_match_is_a_successful_empty_result(self):
        self.service.answer.return_value = {"answer": "No jobs were retrieved.", "cited_job_ids": [], "jobs": []}
        response = self.client.post("/api/query", json={"query": "Python", "location": "unknown"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["jobs"], [])

    def test_errors_are_sanitized_and_lock_is_released(self):
        for error, status in [(ValueError("private detail"), 422),
                              (RuntimeError("private detail"), 503),
                              (OSError("private detail"), 503),
                              (KeyError("private detail"), 500)]:
            with self.subTest(status=status, error=type(error)):
                self.service.answer.side_effect = error
                response = self.client.post("/api/query", json={"query": "Python"})
                self.assertEqual(response.status_code, status)
                self.assertNotIn("private detail", response.text)
                self.assertFalse(self.app.state.inference_lock.locked())

    def test_busy_returns_retry_without_running_pipeline(self):
        self.app.state.inference_lock.acquire()
        try:
            response = self.client.post("/api/query", json={"query": "Python"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.headers["Retry-After"], "5")
            self.service.answer.assert_not_called()
        finally:
            self.app.state.inference_lock.release()

    def test_invalid_service_response_is_a_server_error(self):
        self.service.answer.return_value = {"answer": "x", "cited_job_ids": [], "jobs": [{}]}
        response = self.client.post("/api/query", json={"query": "Python"})
        self.assertEqual(response.status_code, 500)
        self.assertFalse(self.app.state.inference_lock.locked())

    def test_health_and_openapi(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ready")
        schema = self.client.get("/openapi.json").json()
        self.assertIn("post", schema["paths"]["/api/query"])
        self.service.answer.assert_not_called()

    def test_startup_failure_is_not_reported_as_ready(self):
        with self.assertRaisesRegex(RuntimeError, "missing index"):
            with TestClient(create_app(Mock(side_effect=RuntimeError("missing index")))):
                pass
