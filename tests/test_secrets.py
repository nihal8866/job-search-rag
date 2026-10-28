"""Use constructed dummy strings; never use live credentials as test fixtures."""

import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "repository_secret_check", Path(__file__).resolve().parents[1] / "scripts/check_secrets.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SecretCheckTests(unittest.TestCase):
    def test_private_key_header_is_detected(self):
        header = b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----"
        self.assertIn("private key block", MODULE.detect_secrets("notes.txt", header))

    def test_token_value_is_not_part_of_report(self):
        dummy = b"ghp_" + b"A" * 36
        issues = MODULE.detect_secrets("code.py", dummy)
        self.assertIn("GitHub token", issues)
        self.assertNotIn(dummy.decode(), str(issues))

    def test_private_filename_is_blocked_even_for_binary_content(self):
        self.assertIn("private credential filename", MODULE.detect_secrets("key.pfx", b"\x00abc"))

    def test_empty_environment_example_is_allowed(self):
        self.assertEqual(MODULE.detect_secrets(".env.example", b"LLM_API_KEY=\n"), [])

    def test_environment_secret_file_is_blocked(self):
        self.assertIn("private credential filename", MODULE.detect_secrets(".env.local", b""))

    def test_normal_source_is_allowed(self):
        self.assertEqual(MODULE.detect_secrets("src/job_rag/config.py", b'device = "cuda"'), [])

    def test_model_download_token_is_detected(self):
        dummy = b"hf_" + b"A" * 34
        self.assertIn("Hugging Face token", MODULE.detect_secrets("notes.txt", dummy))
