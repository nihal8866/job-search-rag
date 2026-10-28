import json
import unittest

from job_rag.evidence_answers import render_selection, selection_request


class EvidenceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.context = {"question": "Python", "sources": [
            {"job_id": "LF0001", "title": "Developer", "location": "Tokyo",
             "evidence": "Java, Golang, or Python.\n\nDatabases and messaging frameworks.\n[Source heading]"},
            {"job_id": "LF0002", "title": "Designer", "evidence": "Prototyping experience."},
        ]}

    def test_schema_restricts_lines_to_their_own_source(self):
        prompt, schema, units = selection_request(json.dumps(self.context))
        self.assertEqual(schema["properties"]["LF0002"]["items"]["enum"], [0])
        self.assertEqual(units["LF0001"][3], "Location: Tokyo")
        self.assertEqual(json.loads(prompt)["sources"][0]["lines"][0]["text"], "Java, Golang, or Python.")

    def test_renderer_preserves_alternatives_and_source_order(self):
        _, _, units = selection_request(json.dumps(self.context))
        result = render_selection({"LF0001": [1, 0], "LF0002": []}, units)
        self.assertIn("> Java, Golang, or Python.\n> Databases and messaging frameworks.", result.text)
        self.assertEqual(result.cited_job_ids, ["LF0001", "LF0002"])
        self.assertIn("not established", result.text)

    def test_invalid_or_freeform_model_outputs_are_rejected(self):
        _, _, units = selection_request(json.dumps(self.context))
        for selected in [{"text": "invented answer"}, {"LF0001": [], "LF9999": []},
                         {"LF0001": [-1], "LF0002": []},
                         {"LF0001": [True], "LF0002": []},
                         {"LF0001": [0, 0], "LF0002": []},
                         {"LF0001": [0, 1, 2, 3], "LF0002": []},
                         {"LF0001": [], "LF0002": [1]}]:
            with self.subTest(selected=selected), self.assertRaises(RuntimeError):
                render_selection(selected, units)

    def test_multiline_metadata_remains_inside_the_source_quote(self):
        result = render_selection({"LF0001": [0]}, {"LF0001": ["Location: Tokyo\n[Recorded note]"]})
        self.assertIn("> Location: Tokyo\n> [Recorded note]", result.text)

    def test_duplicate_invalid_and_empty_sources_are_rejected(self):
        for sources in [[], [self.context["sources"][0]] * 2,
                        [{"job_id": "other", "title": "x", "evidence": "x"}],
                        [{"job_id": "LF0001", "title": "x", "evidence": " "}]]:
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                selection_request(json.dumps({"question": "x", "sources": sources}))
