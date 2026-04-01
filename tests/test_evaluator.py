"""Tests for the MCQ evaluator module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_autorag.config.models import MCQQuestion
from agentic_autorag.engine.pipeline import RetrievalResult, RetrievedDocument
from agentic_autorag.examiner.evaluator import ExamResult, MCQEvaluator, QuestionResult

FOUR_KEYS = {"A", "B", "C", "D"}
THREE_KEYS = {"A", "B", "C"}


class TestParseAnswer:
    """Test the regex-based answer extraction."""

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("B", "B"),
            ("b", "B"),
            ("B)", "B"),
            ("B.", "B"),
            ("B: something", "B"),
            ("The answer is B", "B"),
            ("The answer is: C", "C"),
            ("answer: D", "D"),
            ("A) First option", "A"),
            ("  D  ", "D"),
        ],
    )
    def test_common_formats(self, response: str, expected: str) -> None:
        result = MCQEvaluator._parse_answer(response, FOUR_KEYS)

        assert result == expected

    def test_invalid_response(self) -> None:
        result1 = MCQEvaluator._parse_answer("I don't know", FOUR_KEYS)
        result2 = MCQEvaluator._parse_answer("", FOUR_KEYS)

        assert result1 == "INVALID"
        assert result2 == "INVALID"

    def test_three_options_rejects_d(self) -> None:
        result = MCQEvaluator._parse_answer("D", THREE_KEYS)

        assert result == "INVALID"

    def test_three_options_accepts_valid(self) -> None:
        result = MCQEvaluator._parse_answer("C", THREE_KEYS)

        assert result == "C"


class TestExamResult:
    def test_failed_questions_returns_incorrect_only(self) -> None:
        results = [
            QuestionResult(
                question_id="q1",
                correct=True,
                selected_answer="A",
                correct_answer="A",
                retrieved_context="ctx",
                generated_response="A",
            ),
            QuestionResult(
                question_id="q2",
                correct=False,
                selected_answer="B",
                correct_answer="C",
                retrieved_context="ctx",
                generated_response="B",
            ),
        ]
        exam_result = ExamResult(
            score=0.5,
            n_correct=1,
            n_total=2,
            question_results=results,
        )

        failed = exam_result.failed_questions()

        assert len(failed) == 1
        assert failed[0].question_id == "q2"

    def test_all_correct(self) -> None:
        result = ExamResult(
            score=1.0,
            n_correct=1,
            n_total=1,
            question_results=[
                QuestionResult(
                    question_id="q1",
                    correct=True,
                    selected_answer="A",
                    correct_answer="A",
                    retrieved_context="",
                    generated_response="A",
                ),
            ],
        )

        failed = result.failed_questions()

        assert failed == []


def _make_question(qid: str, correct: str) -> MCQQuestion:
    return MCQQuestion(
        id=qid,
        question="What is X?",
        options={"A": "opt a", "B": "opt b", "C": "opt c", "D": "opt d"},
        correct_answer=correct,
        source_doc_ids=["doc_0"],
        cluster_id=0,
    )


def _mock_pipeline(answer_text: str) -> MagicMock:
    """Return a mock RAGPipeline that always retrieves one doc and generates *answer_text*."""
    pipeline = MagicMock()
    pipeline.retrieve = AsyncMock(
        return_value=RetrievalResult(
            documents=[RetrievedDocument(id="d0", text="some context", score=1.0)],
        ),
    )
    pipeline.generate = AsyncMock(return_value=answer_text)
    return pipeline


class TestEvaluate:
    async def test_all_correct(self) -> None:
        exam = [_make_question("q1", "B"), _make_question("q2", "B")]
        pipeline = _mock_pipeline("B")

        result = await MCQEvaluator().evaluate(pipeline, exam)

        assert result.score == 1.0
        assert result.n_correct == 2
        assert result.n_total == 2
        assert result.failed_questions() == []

    async def test_mixed_results(self) -> None:
        exam = [_make_question("q1", "A"), _make_question("q2", "C")]
        pipeline = _mock_pipeline("A")  # always answers A

        result = await MCQEvaluator().evaluate(pipeline, exam)

        assert result.n_correct == 1
        assert result.n_total == 2
        assert result.score == pytest.approx(0.5)
        assert len(result.failed_questions()) == 1
        assert result.failed_questions()[0].question_id == "q2"

    async def test_invalid_answer(self) -> None:
        exam = [_make_question("q1", "A")]
        pipeline = _mock_pipeline("I have no idea")

        result = await MCQEvaluator().evaluate(pipeline, exam)

        assert result.n_correct == 0
        assert result.question_results[0].selected_answer == "INVALID"

    async def test_empty_exam(self) -> None:
        pipeline = _mock_pipeline("A")

        result = await MCQEvaluator().evaluate(pipeline, [])

        assert result.score == 0.0
        assert result.n_total == 0


class TestRetryOnTransientFailure:
    async def test_retries_after_transient_error(self) -> None:
        """A question that fails on the first pass should succeed after retry."""
        exam = [_make_question("q1", "B"), _make_question("q2", "B")]
        pipeline = MagicMock()
        pipeline.retrieve = AsyncMock(
            return_value=RetrievalResult(
                documents=[RetrievedDocument(id="d0", text="ctx", score=1.0)],
            ),
        )
        # q2 fails once then succeeds; q1 always succeeds
        call_count = 0

        async def _generate_side_effect(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ConnectionError("503 Service Unavailable")
            return "B"

        pipeline.generate = AsyncMock(side_effect=_generate_side_effect)

        with patch("agentic_autorag.examiner.evaluator.asyncio.sleep", new_callable=AsyncMock):
            result = await MCQEvaluator().evaluate(pipeline, exam)

        assert result.n_correct == 2
        assert result.n_total == 2
        assert result.score == 1.0
        assert all(qr.selected_answer == "B" for qr in result.question_results)

    async def test_permanent_failure_after_all_retries(self) -> None:
        """A question that always fails stays as INVALID after all retry rounds."""
        exam = [_make_question("q1", "A")]
        pipeline = MagicMock()
        pipeline.retrieve = AsyncMock(
            return_value=RetrievalResult(
                documents=[RetrievedDocument(id="d0", text="ctx", score=1.0)],
            ),
        )
        pipeline.generate = AsyncMock(side_effect=ConnectionError("always fails"))

        with patch("agentic_autorag.examiner.evaluator.asyncio.sleep", new_callable=AsyncMock):
            result = await MCQEvaluator().evaluate(pipeline, exam)

        assert result.n_correct == 0
        assert result.question_results[0].selected_answer == "INVALID"

    async def test_no_retry_when_all_pass(self) -> None:
        """When all questions succeed, no retry rounds should happen."""
        exam = [_make_question("q1", "A")]
        pipeline = _mock_pipeline("A")

        with patch("agentic_autorag.examiner.evaluator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await MCQEvaluator().evaluate(pipeline, exam)

        mock_sleep.assert_not_called()
        assert result.n_correct == 1
