"""Tests for benchmark_eval.scoring — EM/F1/retrieval metrics/LLM-judge."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_autorag.benchmark_eval.scoring import (
    best_em,
    best_f1,
    exact_match,
    llm_judge,
    normalize_answer,
    retrieval_metrics,
    token_f1,
)


class TestNormalizeAnswer:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("The Beatles", "beatles"),
            ("The   Beatles", "beatles"),
            ("A quick brown fox", "quick brown fox"),
            ("New York, NY!", "new york ny"),
            ("  leading and trailing  ", "leading and trailing"),
            ("AN apple", "apple"),
            ("", ""),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert normalize_answer(raw) == expected


class TestExactMatch:
    def test_case_insensitive(self) -> None:
        assert exact_match("The Beatles", "the beatles") == 1.0

    def test_punctuation_stripped(self) -> None:
        assert exact_match("New York!", "new york") == 1.0

    def test_article_stripped(self) -> None:
        assert exact_match("The Beatles", "Beatles") == 1.0

    def test_mismatch(self) -> None:
        assert exact_match("Paris", "London") == 0.0


class TestTokenF1:
    def test_full_overlap(self) -> None:
        assert token_f1("New York City", "New York City") == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        f1 = token_f1("New York City", "New York")
        # pred=3 tokens, gold=2 tokens, common=2 → P=2/3 R=2/2 F1≈0.8
        assert f1 == pytest.approx(0.8)

    def test_yes_no_exact(self) -> None:
        assert token_f1("yes", "yes") == 1.0
        assert token_f1("Yes.", "yes") == 1.0
        assert token_f1("no", "yes") == 0.0
        # Partial overlap on yes/no must not be credited
        assert token_f1("yes I think", "yes") == 0.0

    def test_no_overlap(self) -> None:
        assert token_f1("Paris", "London") == 0.0

    def test_empty(self) -> None:
        assert token_f1("", "something") == 0.0
        assert token_f1("something", "") == 0.0


class TestBestAgainstAliases:
    def test_best_em_across_aliases(self) -> None:
        assert best_em("Obama", ["Barack Obama", "Obama", "BHO"]) == 1.0

    def test_best_f1_across_aliases(self) -> None:
        # Best alias gives F1=1, pick that one.
        assert best_f1("New York City", ["NYC", "New York City"]) == pytest.approx(1.0)


class TestRetrievalMetrics:
    def test_all_gold_found_top_ranks(self) -> None:
        m = retrieval_metrics(
            retrieved_doc_ids=["a", "b", "c", "d", "e"],
            supporting_doc_ids=["a", "b"],
        )
        assert m.recall_at_k[1] == 0.5
        assert m.recall_at_k[2] == 1.0
        assert m.recall_at_k[5] == 1.0
        # Joint recall: top-1 has only 1 of 2 gold → 0; top-2 has both → 1.
        assert m.joint_recall_at_k[1] == 0.0
        assert m.joint_recall_at_k[2] == 1.0
        assert m.joint_recall_at_k[5] == 1.0
        assert m.first_gold_rank == 1
        assert m.complete_rank == 2

    def test_first_gold_at_rank_3(self) -> None:
        m = retrieval_metrics(
            retrieved_doc_ids=["x", "y", "a", "b", "z"],
            supporting_doc_ids=["a", "b"],
        )
        assert m.first_gold_rank == 3
        assert m.complete_rank == 4
        assert m.recall_at_k[1] == 0.0
        assert m.recall_at_k[2] == 0.0  # top-2 = [x, y], no gold yet
        assert m.recall_at_k[5] == 1.0  # top-5 contains both gold
        assert m.joint_recall_at_k[2] == 0.0
        assert m.joint_recall_at_k[5] == 1.0

    def test_no_gold_retrieved(self) -> None:
        m = retrieval_metrics(
            retrieved_doc_ids=["x", "y", "z"],
            supporting_doc_ids=["a", "b"],
        )
        assert m.first_gold_rank is None
        assert m.complete_rank is None
        assert all(v == 0.0 for v in m.recall_at_k.values())
        assert all(v == 0.0 for v in m.joint_recall_at_k.values())

    def test_duplicates_deduplicated(self) -> None:
        m = retrieval_metrics(
            retrieved_doc_ids=["a", "a", "b", "c"],
            supporting_doc_ids=["a", "b"],
        )
        # After dedup: a, b, c — rank 1 is a, rank 2 is b.
        assert m.first_gold_rank == 1
        assert m.complete_rank == 2
        assert m.recall_at_k[2] == 1.0
        assert m.joint_recall_at_k[2] == 1.0

    def test_no_supporting_docs(self) -> None:
        m = retrieval_metrics(
            retrieved_doc_ids=["a", "b"],
            supporting_doc_ids=[],
        )
        assert m.first_gold_rank is None
        assert m.complete_rank is None
        assert all(v == 0.0 for v in m.recall_at_k.values())
        assert all(v == 0.0 for v in m.joint_recall_at_k.values())

    def test_single_hop_degradation(self) -> None:
        # On single-gold (single-hop) questions, joint_recall_at_k must equal
        # recall_at_k and complete_rank must equal first_gold_rank for every k.
        # This is the property that lets us use joint_recall as the headline
        # retrieval metric across both single- and multi-hop benchmarks.
        for retrieved, gold, ks in [
            (["a", "b", "c"], ["a"], (1, 2, 5)),
            (["x", "y", "a"], ["a"], (1, 2, 5, 10)),
            (["x", "y", "z"], ["a"], (1, 2, 5, 10)),
            (["a", "b", "c"], ["b"], (1, 2, 3)),
        ]:
            m = retrieval_metrics(retrieved, gold, ks=ks)
            for k in ks:
                assert m.joint_recall_at_k[k] == m.recall_at_k[k], (
                    f"degradation broken at k={k}: {m}"
                )
            assert m.complete_rank == m.first_gold_rank, (
                f"degradation broken on ranks: {m}"
            )

    def test_three_hop_complete_rank(self) -> None:
        # 3-hop generalisation: complete_rank is the rank where ALL gold appear.
        m = retrieval_metrics(
            retrieved_doc_ids=["a", "x", "b", "y", "c"],
            supporting_doc_ids=["a", "b", "c"],
        )
        assert m.first_gold_rank == 1
        assert m.complete_rank == 5
        assert m.recall_at_k[1] == pytest.approx(1 / 3)
        assert m.recall_at_k[5] == 1.0
        assert m.joint_recall_at_k[1] == 0.0
        assert m.joint_recall_at_k[2] == 0.0
        assert m.joint_recall_at_k[5] == 1.0


class TestLLMJudge:
    async def test_returns_1_on_yes(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "YES"
        with patch(
            "agentic_autorag.litellm_runtime.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await llm_judge("gemini/flash", "Q?", "Paris", ["Paris"])
        assert result == 1

    async def test_returns_0_on_no(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "No"
        with patch(
            "agentic_autorag.litellm_runtime.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await llm_judge("gemini/flash", "Q?", "London", ["Paris"])
        assert result == 0

    async def test_handles_trailing_text(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Yes - the answers match."
        with patch(
            "agentic_autorag.litellm_runtime.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await llm_judge("gemini/flash", "Q?", "Paris", ["Paris"])
        assert result == 1

    async def test_returns_none_on_parse_fail(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "maybe"
        with patch(
            "agentic_autorag.litellm_runtime.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await llm_judge("gemini/flash", "Q?", "x", ["y"])
        assert result is None

    async def test_returns_minus_one_on_no_answer(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "NO_ANSWER"
        with patch(
            "agentic_autorag.litellm_runtime.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await llm_judge("gemini/flash", "Q?", "I don't know.", ["Paris"])
        assert result == -1

    async def test_no_answer_does_not_collide_with_no(self) -> None:
        # Sanity check that the parse regex picks NO_ANSWER, not NO, when
        # the model emits the longer token. Without leftmost-longest order
        # NO would win.
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  no_answer  "
        with patch(
            "agentic_autorag.litellm_runtime.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await llm_judge("gemini/flash", "Q?", "...", ["Paris"])
        assert result == -1

    async def test_returns_none_on_api_error(self) -> None:
        with patch(
            "agentic_autorag.litellm_runtime.litellm.acompletion",
            new=AsyncMock(side_effect=RuntimeError("rate limit")),
        ):
            result = await llm_judge("gemini/flash", "Q?", "x", ["y"])
        assert result is None
