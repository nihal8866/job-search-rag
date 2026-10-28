"""Checks for technical keywords and fusion ranking behavior."""

import unittest

from job_rag.retrieval import reciprocal_rank_fusion, tokenize_keywords


class KeywordTests(unittest.TestCase):
    def test_technical_terms_and_multilingual_characters(self):
        self.assertEqual(
            tokenize_keywords("Python APIs, MySQL, C++ and C# 软件开发"),
            ["python", "apis", "mysql", "c++", "and", "c#", "软", "件", "开", "发"],
        )

    def test_empty_or_punctuation_only_text(self):
        self.assertEqual(tokenize_keywords(""), [])
        self.assertEqual(tokenize_keywords(" , ; ! "), [])


class FusionTests(unittest.TestCase):
    def test_shared_candidate_can_outrank_single_source_leaders(self):
        result = reciprocal_rank_fusion(
            [["vector_only", "shared"], ["keyword_only", "shared"]], k=60
        )
        self.assertEqual([chunk_id for chunk_id, _ in result],
                         ["shared", "keyword_only", "vector_only"])
        self.assertGreater(result[0][1], result[1][1])

    def test_empty_rankings_return_no_candidates(self):
        self.assertEqual(reciprocal_rank_fusion([[], []], k=60), [])

    def test_one_retriever_can_have_no_candidates(self):
        result = reciprocal_rank_fusion([[], ["b", "a"]], k=60)
        self.assertEqual([chunk_id for chunk_id, _ in result], ["b", "a"])

    def test_duplicate_ids_cannot_inflate_a_single_retrievers_vote(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            reciprocal_rank_fusion([["a", "a"]], k=60)

    def test_negative_constant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            reciprocal_rank_fusion([["a"]], k=-1)


if __name__ == "__main__":
    unittest.main()
