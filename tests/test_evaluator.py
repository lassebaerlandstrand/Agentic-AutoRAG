"""Tests for the MCQ evaluator module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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
        assert MCQEvaluator._parse_answer(response, FOUR_KEYS) == expected

    def test_invalid_response(self) -> None:
        assert MCQEvaluator._parse_answer("I don't know", FOUR_KEYS) == "INVALID"
        assert MCQEvaluator._parse_answer("", FOUR_KEYS) == "INVALID"

    def test_three_options_rejects_d(self) -> None:
        assert MCQEvaluator._parse_answer("D", THREE_KEYS) == "INVALID"

    def test_three_options_accepts_valid(self) -> None:
        assert MCQEvaluator._parse_answer("C", THREE_KEYS) == "C"

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
        assert result.failed_questions() == []




def _make_question(qid: str, correct: str) -> MCQQuestion:
    return MCQQuestion(
        id=qid,
        question="What is X?",
        options={"A": "opt a", "B": "opt b", "C": "opt c", "D": "opt d"},
        correct_answer=correct,
        source_chunk_id="chunk_0",
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
