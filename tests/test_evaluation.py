import unittest

from job_rag.evaluation import known_item_metrics, summarize_known_items


class KnownItemTests(unittest.TestCase):
    def test_first_target(self):
        result = known_item_metrics(["A", "B"], "A")
        self.assertEqual(result["hit_at_1"], 1)
        self.assertEqual(result["reciprocal_rank_at_k"], 1)

    def test_later_target(self):
        result = known_item_metrics(["A", "B", "C"], "C")
        self.assertEqual(result["target_rank_at_k"], 3)
        self.assertEqual(result["hit_at_1"], 0)
        self.assertAlmostEqual(result["reciprocal_rank_at_k"], 1 / 3)

    def test_target_outside_cutoff_is_not_found(self):
        result = known_item_metrics(["A", "B", "C"], "C", k=2)
        self.assertIsNone(result["target_rank_at_k"])
        self.assertEqual(result["hit_at_k"], 0)

    def test_empty_ranking(self):
        self.assertEqual(known_item_metrics([], "A")["reciprocal_rank_at_k"], 0)

    def test_bad_rankings_or_cutoff(self):
        for ids, target, cutoff in [(["A", "A"], "A", 5), ([], "A", 0), ([], "", 5)]:
            with self.assertRaises(ValueError):
                known_item_metrics(ids, target, cutoff)

    def test_macro_average(self):
        result = summarize_known_items([known_item_metrics(["A"], "A"), known_item_metrics([], "B")])
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["hit_at_k"], 0.5)
        self.assertEqual(result["reciprocal_rank_at_k"], 0.5)
        with self.assertRaises(ValueError):
            summarize_known_items([])
