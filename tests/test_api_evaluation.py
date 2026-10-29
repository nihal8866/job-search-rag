from copy import deepcopy
import unittest

from job_rag.api_evaluation import inspect_response


class ApiEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.case = {"status": 200, "request": {"query": "Python", "location": "tOkYo", "top_k": 1}, "target_job_id": "LF0001"}
        self.payload = {
            "query": "Python", "location_filter": "tOkYo",
            "answer": "Selected evidence [LF0001]:\n> Python or Java.",
            "cited_job_ids": ["LF0001"],
            "jobs": [{"job_id": "LF0001", "location": "Tokyo, Japan", "evidence": "Python or Java."}],
        }

    def test_valid_response_and_separate_rank_observation(self):
        result = inspect_response(self.case, 200, self.payload)
        self.assertTrue(all(result["contract_checks"].values()))
        self.assertEqual(result["quality_observations"]["target_rank"], 1)

    def test_fabricated_quote_fails_fidelity(self):
        self.payload["answer"] = "Selected evidence [LF0001]:\n> Mandatory Python."
        self.assertFalse(inspect_response(self.case, 200, self.payload)["contract_checks"]["quoted_lines_match_cited_evidence"])

    def test_unquoted_claim_is_not_treated_as_grounded(self):
        self.payload["answer"] = "This vacancy is definitely open today [LF0001]."
        result = inspect_response(self.case, 200, self.payload)
        self.assertFalse(result["contract_checks"]["constrained_answer_format"])

    def test_wrong_location_and_unknown_citation_fail(self):
        self.payload["jobs"][0]["location"] = "Singapore"
        self.payload["cited_job_ids"] = ["LF9999"]
        checks = inspect_response(self.case, 200, self.payload)["contract_checks"]
        self.assertFalse(checks["literal_location_filter"])
        self.assertFalse(checks["citations_match_jobs"])

    def test_target_miss_is_an_observation_not_a_contract_error(self):
        case = deepcopy(self.case)
        case["target_job_id"] = "LF0002"
        result = inspect_response(case, 200, self.payload)
        self.assertTrue(all(result["contract_checks"].values()))
        self.assertIsNone(result["quality_observations"]["target_rank"])

    def test_abstention_is_reported_separately(self):
        self.case["expect_no_quotes"] = True
        self.payload["answer"] = "An answer is not established by the selected evidence for this job [LF0001]."
        self.assertTrue(inspect_response(self.case, 200, self.payload)["quality_observations"]["expected_abstention"])

    def test_metadata_quote_and_status_failures(self):
        self.payload["answer"] = "Selected evidence [LF0001]:\n> Location: Tokyo, Japan"
        self.assertTrue(all(inspect_response(self.case, 200, self.payload)["contract_checks"].values()))
        result = inspect_response(self.case, 503, {"detail": "unavailable"})
        self.assertFalse(result["contract_checks"]["expected_status"])
