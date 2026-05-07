"""Tests for the open-ended evaluator's scoring stack."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentic_autorag.config.models import OpenEndedQuestion
from agentic_autorag.engine.pipeline import RetrievedDocument
from agentic_autorag.examiner.evaluator import OpenEndedEvaluator, QuestionResult


def _make_question() -> OpenEndedQuestion:
    return OpenEndedQuestion(
        id="q1",
        question="Who founded Beta Inc?",
        canonical_answer="Sarah Smith",
        answer_variants=["S. Smith", "Smith"],
        reasoning_type="bridge",
        source_chunk_ids=["a::0", "b::0"],
        source_doc_ids=["doc_a", "doc_b"],
        source_spans=["some span A", "some span B"],
    )


class _FakeTiming:
    model_s = 0.0


class _FakeRetrieval:
    def __init__(self, docs: list[RetrievedDocument]) -> None:
        self.documents = docs
        self.timing = _FakeTiming()


class _FakePipelineConfig:
    llm_timeout_s = 10.0


class _FakePipeline:
    def __init__(self, retrieval, generation_response: str) -> None:
        self._retrieval = retrieval
        self._gen = generation_response
        self.config = _FakePipelineConfig()

    async def retrieve(self, _q: str):
        return self._retrieval

    async def generate(self, _prompt: str) -> str:
        return self._gen


@pytest.mark.asyncio
class TestEvaluatorScoring:
    async def test_em_match_marks_correct(self) -> None:
        evaluator = OpenEndedEvaluator(concurrency=1, retrieval_quality_alpha=1.0)
        pipeline = _FakePipeline(_FakeRetrieval([]), "Sarah Smith")
        result = await evaluator.evaluate(pipeline, [_make_question()])
        assert result.n_correct == 1
        assert result.question_results[0].em == 1.0
        assert result.question_results[0].correct is True

    async def test_paraphrase_uses_f1_threshold(self) -> None:
        evaluator = OpenEndedEvaluator(concurrency=1, retrieval_quality_alpha=1.0)
        pipeline = _FakePipeline(_FakeRetrieval([]), "Smith")  # token F1 high vs variant
        result = await evaluator.evaluate(pipeline, [_make_question()])
        # "Smith" matches variant exactly under normalised EM.
        assert result.question_results[0].correct is True

    async def test_judge_fallback_invoked_only_for_low_f1(self) -> None:
        evaluator = OpenEndedEvaluator(
            concurrency=1,
            retrieval_quality_alpha=1.0,
            judge_model="test/judge",
        )
        # Bogus answer: EM=0, F1≈0 → judge invoked. Stub it to say YES.
        pipeline = _FakePipeline(_FakeRetrieval([]), "completely different phrasing")
        with patch(
            "agentic_autorag.examiner.evaluator.llm_judge",
            new=AsyncMock(return_value=1),
        ):
            result = await evaluator.evaluate(pipeline, [_make_question()])
        qr = result.question_results[0]
        assert qr.em == 0.0
        assert qr.judge == 1
        assert qr.correct is True

    async def test_judge_not_invoked_when_em_already_passed(self) -> None:
        judge_mock = AsyncMock(return_value=1)
        evaluator = OpenEndedEvaluator(
            concurrency=1,
            retrieval_quality_alpha=1.0,
            judge_model="test/judge",
        )
        pipeline = _FakePipeline(_FakeRetrieval([]), "Sarah Smith")
        with patch("agentic_autorag.examiner.evaluator.llm_judge", new=judge_mock):
            result = await evaluator.evaluate(pipeline, [_make_question()])
        qr = result.question_results[0]
        assert qr.correct is True
        assert qr.judge is None
        judge_mock.assert_not_awaited()


class TestQuestionResultProperties:
    def test_context_sufficient_only_when_all_spans_found(self) -> None:
        qr = QuestionResult(
            question_id="q1",
            correct=False,
            selected_answer="x",
            correct_answer="Sarah Smith",
            retrieved_context="",
            generated_response="x",
            chunk_precision=0.4,
            retrieved_spans=1,
            n_spans=2,
        )
        assert qr.context_sufficient is False
        assert qr.retrieval_status == "partial 1/2"

        qr_neither = qr.model_copy(update={"retrieved_spans": 0, "chunk_precision": 0.0})
        assert qr_neither.context_sufficient is False
        assert qr_neither.retrieval_status == "none"

        qr_both = qr.model_copy(update={"retrieved_spans": 2})
        assert qr_both.context_sufficient is True
        assert qr_both.retrieval_status == "complete"


@pytest.mark.asyncio
class TestRefusalDetection:
    """``refused`` is set from the three-way judge verdict (NO_ANSWER) plus
    empty predictions, replacing the previous regex-based detector. These
    tests exercise the judge-driven path end-to-end via the evaluator."""

    async def test_judge_no_answer_marks_refused(self) -> None:
        evaluator = OpenEndedEvaluator(concurrency=1, judge_model="judge/test")
        # EM=0 (pred != gold) → judge called → returns -1 (NO_ANSWER).
        pipeline = _FakePipeline(_FakeRetrieval([]), "I cannot determine the answer.")
        with patch(
            "agentic_autorag.examiner.evaluator.llm_judge",
            new=AsyncMock(return_value=-1),
        ):
            result = await evaluator.evaluate(pipeline, [_make_question()])
        qr = result.question_results[0]
        assert qr.refused is True
        assert qr.judge == -1
        assert result.n_refused == 1
        assert result.n_judge_no_answer == 1

    async def test_judge_no_marks_not_refused(self) -> None:
        evaluator = OpenEndedEvaluator(concurrency=1, judge_model="judge/test")
        pipeline = _FakePipeline(_FakeRetrieval([]), "Bob")
        with patch(
            "agentic_autorag.examiner.evaluator.llm_judge",
            new=AsyncMock(return_value=0),
        ):
            result = await evaluator.evaluate(pipeline, [_make_question()])
        qr = result.question_results[0]
        assert qr.refused is False
        assert qr.judge == 0
        assert result.n_refused == 0

    async def test_empty_pred_marks_refused_without_judge_call(self) -> None:
        evaluator = OpenEndedEvaluator(concurrency=1, judge_model="judge/test")
        pipeline = _FakePipeline(_FakeRetrieval([]), "")
        judge_mock = AsyncMock(return_value=1)
        with patch("agentic_autorag.examiner.evaluator.llm_judge", new=judge_mock):
            result = await evaluator.evaluate(pipeline, [_make_question()])
        qr = result.question_results[0]
        assert qr.refused is True
        assert qr.judge is None
        judge_mock.assert_not_called()
        assert result.n_no_answer == 1


@pytest.mark.asyncio
class TestExamResultAggregates:
    async def test_score_equals_accuracy(self) -> None:
        evaluator = OpenEndedEvaluator(concurrency=1)
        pipeline = _FakePipeline(_FakeRetrieval([]), "Sarah Smith")
        result = await evaluator.evaluate(pipeline, [_make_question()])
        assert result.score == result.answer_accuracy

    async def test_failure_mode_counters_populate(self) -> None:
        evaluator = OpenEndedEvaluator(concurrency=1)
        pipeline = _FakePipeline(_FakeRetrieval([]), "Sarah Smith")
        result = await evaluator.evaluate(pipeline, [_make_question()])
        # No documents retrieved → 0 of 2 spans found.
        assert result.n_retrieval_miss == 1
        assert result.n_retrieval_complete == 0
        assert result.n_refused == 0
