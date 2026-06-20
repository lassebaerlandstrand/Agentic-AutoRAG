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
    def __init__(
        self,
        docs: list[RetrievedDocument],
        expansion_cost_usd: float = 0.0,
    ) -> None:
        self.documents = docs
        self.timing = _FakeTiming()
        self.expansion_cost = {
            "usd": expansion_cost_usd,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


class _FakePipelineConfig:
    llm_timeout_s = 10.0


class _FakePipeline:
    def __init__(self, retrieval, generation_response: str, generation_cost_usd: float = 0.0) -> None:
        self._retrieval = retrieval
        self._gen = generation_response
        self._gen_cost = {
            "usd": generation_cost_usd,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        self.config = _FakePipelineConfig()

    async def retrieve(self, _q: str):
        return self._retrieval

    async def prepare_context(self, _q: str, retrieval) -> tuple[str, dict[str, float | int]]:
        return "\n".join(doc.text for doc in retrieval.documents), {
            "usd": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    async def generate(self, _prompt: str) -> tuple[str, dict[str, float | int]]:
        return self._gen, dict(self._gen_cost)


@pytest.mark.asyncio
class TestEvaluatorScoring:
    async def test_em_match_marks_correct(self) -> None:
        evaluator = OpenEndedEvaluator(concurrency=1)
        pipeline = _FakePipeline(_FakeRetrieval([]), "Sarah Smith")
        result = await evaluator.evaluate(pipeline, [_make_question()])
        assert result.n_correct == 1
        assert result.question_results[0].em == 1.0
        assert result.question_results[0].correct is True

    async def test_paraphrase_uses_f1_threshold(self) -> None:
        evaluator = OpenEndedEvaluator(concurrency=1)
        pipeline = _FakePipeline(_FakeRetrieval([]), "Smith")  # token F1 high vs variant
        result = await evaluator.evaluate(pipeline, [_make_question()])
        # "Smith" matches variant exactly under normalised EM.
        assert result.question_results[0].correct is True

    async def test_judge_fallback_invoked_only_for_low_f1(self) -> None:
        evaluator = OpenEndedEvaluator(
            concurrency=1,
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
        evaluator = OpenEndedEvaluator(
            concurrency=1,
            judge_model="test/judge",
        )
        pipeline = _FakePipeline(_FakeRetrieval([]), "Sarah Smith")
        with patch("agentic_autorag.examiner.evaluator.llm_judge", new=AsyncMock(return_value=1)):
            result = await evaluator.evaluate(pipeline, [_make_question()])
        qr = result.question_results[0]
        assert qr.correct is True
        # judge stays None iff the judge was never consulted (it records its
        # verdict whenever it runs), so this output proves the EM short-circuit.
        assert qr.judge is None


@pytest.mark.asyncio
class TestCostAggregation:
    async def test_per_question_cost_captured_and_rolled_up(self) -> None:
        evaluator = OpenEndedEvaluator(concurrency=1)
        # Generation cost $0.0050; expansion cost $0.0010 (e.g., HyDE).
        pipeline = _FakePipeline(
            _FakeRetrieval([], expansion_cost_usd=0.001),
            "Sarah Smith",
            generation_cost_usd=0.005,
        )
        result = await evaluator.evaluate(pipeline, [_make_question()])

        qr = result.question_results[0]
        assert abs(qr.llm_cost_usd - 0.006) < 1e-9
        # Roll-up: one valid question, both cost components included.
        assert abs(result.mean_llm_cost_per_query_usd - 0.006) < 1e-9
        assert abs(result.total_llm_cost_usd - 0.006) < 1e-9

    async def test_zero_cost_for_local_only_models(self) -> None:
        """When LiteLLM has no pricing (cost=0), aggregates to 0 cleanly."""
        evaluator = OpenEndedEvaluator(concurrency=1)
        pipeline = _FakePipeline(_FakeRetrieval([]), "Sarah Smith")  # both costs 0
        result = await evaluator.evaluate(pipeline, [_make_question()])

        assert result.mean_llm_cost_per_query_usd == 0.0
        assert result.total_llm_cost_usd == 0.0


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
        with patch("agentic_autorag.examiner.evaluator.llm_judge", new=AsyncMock(return_value=1)):
            result = await evaluator.evaluate(pipeline, [_make_question()])
        qr = result.question_results[0]
        assert qr.refused is True
        # judge is None ⇒ the empty-prediction short-circuit refused without
        # ever consulting the judge (it records a verdict whenever it runs).
        assert qr.judge is None
        assert result.n_no_answer == 1


@pytest.mark.asyncio
class TestExamResultAggregates:
    async def test_answer_accuracy_is_objective(self) -> None:
        evaluator = OpenEndedEvaluator(concurrency=1)
        pipeline = _FakePipeline(_FakeRetrieval([]), "Sarah Smith")
        result = await evaluator.evaluate(pipeline, [_make_question()])
        # Single EM-correct question over one valid question → accuracy 1.0.
        assert result.answer_accuracy == 1.0

    async def test_failure_mode_counters_populate(self) -> None:
        evaluator = OpenEndedEvaluator(concurrency=1)
        pipeline = _FakePipeline(_FakeRetrieval([]), "Sarah Smith")
        result = await evaluator.evaluate(pipeline, [_make_question()])
        # No documents retrieved → 0 of 2 spans found.
        assert result.n_retrieval_miss == 1
        assert result.n_retrieval_complete == 0
        assert result.n_refused == 0

    async def test_all_errored_flags_when_every_question_hits_sentinel(self) -> None:
        """When every question's response is an error sentinel, the result
        carries ``all_errored=True`` and ``error_sentinel`` is populated. This
        is what the orchestrator promotes to AllQuestionsErrored so the
        proposer routes through failure recovery."""
        from agentic_autorag.examiner._errors import PERMANENT_ERROR_SENTINEL

        evaluator = OpenEndedEvaluator(concurrency=1)

        class _FailingPipeline(_FakePipeline):
            async def generate(self, _prompt: str):
                raise RuntimeError("BadRequestError: Operation not allowed")

        pipeline = _FailingPipeline(_FakeRetrieval([]), "ignored")
        with patch(
            "agentic_autorag.examiner.evaluator.is_permanent_llm_error",
            return_value=True,
        ):
            result = await evaluator.evaluate(pipeline, [_make_question(), _make_question()])
        assert result.n_valid == 0
        assert result.all_errored is True
        assert result.error_sentinel == PERMANENT_ERROR_SENTINEL
