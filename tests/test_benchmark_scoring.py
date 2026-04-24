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
        recalls, rank = retrieval_metrics(
            retrieved_doc_ids=["a", "b", "c", "d", "e"],
            supporting_doc_ids=["a", "b"],
        )
        assert recalls[1] == 0.5
        assert recalls[2] == 1.0
        assert recalls[5] == 1.0
        assert rank == 1

    def test_first_gold_at_rank_3(self) -> None:
        recalls, rank = retrieval_metrics(
            retrieved_doc_ids=["x", "y", "a", "b", "z"],
            supporting_doc_ids=["a", "b"],
        )
        assert rank == 3
        assert recalls[1] == 0.0
        assert recalls[2] == 0.0  # top-2 = [x, y], no gold yet
        assert recalls[5] == 1.0  # top-5 contains both gold

    def test_no_gold_retrieved(self) -> None:
        recalls, rank = retrieval_metrics(
            retrieved_doc_ids=["x", "y", "z"],
            supporting_doc_ids=["a", "b"],
        )
        assert rank is None
        assert all(v == 0.0 for v in recalls.values())

    def test_duplicates_deduplicated(self) -> None:
        recalls, rank = retrieval_metrics(
            retrieved_doc_ids=["a", "a", "b", "c"],
            supporting_doc_ids=["a", "b"],
        )
        # After dedup: a, b, c — rank 1 is a, rank 2 is b.
        assert rank == 1
        assert recalls[2] == 1.0

    def test_no_supporting_docs(self) -> None:
        recalls, rank = retrieval_metrics(
            retrieved_doc_ids=["a", "b"],
            supporting_doc_ids=[],
        )
        assert rank is None
        assert all(v == 0.0 for v in recalls.values())


class TestLLMJudge:
    async def test_returns_1_on_yes(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "YES"
        with patch(
            "agentic_autorag.benchmark_eval.scoring.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await llm_judge("gemini/flash", "Q?", "Paris", ["Paris"])
        assert result == 1

    async def test_returns_0_on_no(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "No"
        with patch(
            "agentic_autorag.benchmark_eval.scoring.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await llm_judge("gemini/flash", "Q?", "London", ["Paris"])
        assert result == 0

    async def test_handles_trailing_text(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Yes - the answers match."
        with patch(
            "agentic_autorag.benchmark_eval.scoring.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await llm_judge("gemini/flash", "Q?", "Paris", ["Paris"])
        assert result == 1

    async def test_returns_none_on_parse_fail(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "maybe"
        with patch(
            "agentic_autorag.benchmark_eval.scoring.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await llm_judge("gemini/flash", "Q?", "x", ["y"])
        assert result is None

    async def test_returns_none_on_api_error(self) -> None:
        with patch(
            "agentic_autorag.benchmark_eval.scoring.litellm.acompletion",
            new=AsyncMock(side_effect=RuntimeError("rate limit")),
        ):
            result = await llm_judge("gemini/flash", "Q?", "x", ["y"])
        assert result is None
