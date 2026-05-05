"""Tests for the open-ended evaluator's scoring stack."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentic_autorag.config.models import OpenEndedQuestion
from agentic_autorag.engine.pipeline import RetrievedDocument
from agentic_autorag.examiner.evaluator import OpenEndedEvaluator, QuestionResult, _detect_refusal


def _make_question() -> OpenEndedQuestion:
    return OpenEndedQuestion(
        id="q1",
        question="Who founded Beta Inc?",
        canonical_answer="Sarah Smith",
        answer_variants=["S. Smith", "Smith"],
        chunk_A_id="a::0",
        chunk_B_id="b::0",
        source_span_A="some span A",
        source_span_B="some span B",
        source_doc_ids=["doc_a", "doc_b"],
        bridge_entity="beta inc",
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
    def test_context_sufficient_only_when_both_spans_found(self) -> None:
        qr = QuestionResult(
            question_id="q1",
            correct=False,
            selected_answer="x",
            correct_answer="Sarah Smith",
            retrieved_context="",
            generated_response="x",
            chunk_precision=0.4,
            retrieval_status="only_A",
        )
        assert qr.context_sufficient is False

        qr_b = qr.model_copy(update={"retrieval_status": "only_B"})
        assert qr_b.context_sufficient is False

        qr_neither = qr.model_copy(update={"retrieval_status": "neither", "chunk_precision": 0.0})
        assert qr_neither.context_sufficient is False

        qr_both = qr.model_copy(update={"retrieval_status": "both"})
        assert qr_both.context_sufficient is True


class TestRefusalDetection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Cannot answer based on provided context.", True),
            ("I can't answer this question.", True),
            ("The context does not contain relevant information.", True),
            ("There is no information about this in the documents.", True),
            ("I don't have enough information to answer.", True),
            ("Insufficient context to determine the answer.", True),
            ("Unable to determine from the provided context.", True),
            ("I don't know who founded Beta Inc.", True),
            ("Sarah Smith founded Beta Inc.", False),
            ("Based on the provided context, Sarah Smith is the founder.", False),
            ("", False),
            (None, False),
        ],
    )
    def test_detect_refusal(self, text, expected) -> None:
        assert _detect_refusal(text) is expected


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
        # No documents retrieved → retrieval_status == "neither"
        assert result.n_retrieval_miss == 1
        assert result.n_retrieval_complete == 0
        assert result.n_refused == 0
