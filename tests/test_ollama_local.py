"""Validate local-only routing and response handling without network calls."""

from dataclasses import replace
import json
import unittest
from unittest.mock import MagicMock
from urllib.error import HTTPError, URLError

from job_rag.config import Settings
from job_rag.ollama_local import NoRedirects, OllamaGenerator


class OllamaTests(unittest.TestCase):
    def setUp(self):
        self.generator = OllamaGenerator(Settings())
        self.generator.opener = MagicMock()
        self.response = self.generator.opener.open.return_value.__enter__.return_value
        self.response.read.return_value = self.response_bytes()

    def response_bytes(self, generated=None, **kwargs):
        generated = generated or {"text": "Python is listed [LF0001].", "cited_job_ids": ["LF0001"]}
        return json.dumps({
            "done": True, "done_reason": "stop",
            "message": {"content": json.dumps(generated)}, **kwargs,
        }).encode()

    def test_local_request_uses_schema_without_credentials(self):
        result = self.generator.generate("system", "question")
        request = self.generator.opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertNotIn("Authorization", request.headers)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["keep_alive"], 0)
        self.assertEqual(payload["format"]["required"], ["text", "cited_job_ids"])
        self.assertEqual(result.cited_job_ids, ["LF0001"])

    def test_remote_hosts_credentials_and_proxy_style_urls_are_rejected(self):
        for url in ["https://ollama.com", "http://example.com", "http://127.0.0.1.example.com",
                    "http://user:password@127.0.0.1:11434", "http://127.0.0.1:11434?redirect=1"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                OllamaGenerator(replace(Settings(), llm_base_url=url))

    def test_cloud_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "local Qwen"):
            OllamaGenerator(replace(Settings(), llm_model="example:cloud"))

    def test_redirects_are_not_followed(self):
        self.assertIsNone(NoRedirects().redirect_request(None, None, 302, "", {}, "https://example.com"))

    def test_large_context_is_rejected_before_network_call(self):
        with self.assertRaisesRegex(ValueError, "context is too large"):
            self.generator.generate("system", "x" * 9000)
        self.generator.opener.open.assert_not_called()

    def test_unicode_context_budget_counts_bytes(self):
        with self.assertRaisesRegex(ValueError, "context is too large"):
            self.generator.generate("system", "界" * 3000)

    def test_server_error_does_not_echo_its_body(self):
        self.generator.opener.open.side_effect = HTTPError("local", 500, "private body", {}, None)
        with self.assertRaisesRegex(RuntimeError, "HTTP 500") as error:
            self.generator.generate("system", "question")
        self.assertNotIn("private body", str(error.exception))

    def test_server_unavailable_is_a_clear_error(self):
        self.generator.opener.open.side_effect = URLError("connection refused")
        with self.assertRaisesRegex(RuntimeError, "local Ollama"):
            self.generator.generate("system", "question")

    def test_output_token_limit_is_not_treated_as_a_complete_answer(self):
        self.response.read.return_value = self.response_bytes(done_reason="length")
        with self.assertRaisesRegex(RuntimeError, "output limit"):
            self.generator.generate("system", "question")

    def test_malformed_json_is_not_accepted(self):
        self.response.read.return_value = b"not JSON"
        with self.assertRaisesRegex(RuntimeError, "structured response"):
            self.generator.generate("system", "question")

    def test_wrong_schema_is_rejected(self):
        self.response.read.return_value = self.response_bytes({"text": "answer", "cited_job_ids": [1]})
        with self.assertRaisesRegex(RuntimeError, "schema"):
            self.generator.generate("system", "question")
