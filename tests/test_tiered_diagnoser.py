"""Tests for the tier-aware evaluator: doc-level (tier B) and gated diagnosis
judge (tier A) failure attribution, plus the consolidated ``failure_mode``."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentic_autorag.config.models import OpenEndedQuestion
from agentic_autorag.engine.pipeline import RetrievedDocument
from agentic_autorag.examiner.evaluator import OpenEndedEvaluator, QuestionResult, failure_mode


class _FakeTiming:
    model_s = 0.0


class _FakeRetrieval:
    def __init__(self, docs: list[RetrievedDocument]) -> None:
        self.documents = docs
        self.timing = _FakeTiming()
        self.expansion_cost = {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}


class _FakePipelineConfig:
    llm_timeout_s = 10.0


class _FakePipeline:
    def __init__(self, retrieval, generation_response: str) -> None:
        self._retrieval = retrieval
        self._gen = generation_response
        self.config = _FakePipelineConfig()

    async def retrieve(self, _q: str):
        return self._retrieval

    async def prepare_context(self, _q: str, retrieval):
        return "\n".join(d.text for d in retrieval.documents), {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    async def generate(self, _prompt: str):
        return self._gen, {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}


def _doc(doc_id: str, text: str = "text") -> RetrievedDocument:
    return RetrievedDocument(id=doc_id, text=text, score=1.0, metadata={"doc_id": doc_id})


def _tier_b_q() -> OpenEndedQuestion:
    return OpenEndedQuestion(
        id="b1", question="Who founded Beta?", canonical_answer="Sarah Smith", supporting_doc_ids=["doc_a", "doc_b"]
    )


def _tier_a_q() -> OpenEndedQuestion:
    return OpenEndedQuestion(id="a1", question="Who founded Beta?", canonical_answer="Sarah Smith")


# --- tier B: doc-level retrieval fork (no judge) --------------------------


@pytest.mark.asyncio
async def test_tier_b_gold_retrieved_but_wrong_is_generation() -> None:
    # Both gold docs retrieved; wrong answer ⇒ generation failure, no judge.
    pipeline = _FakePipeline(_FakeRetrieval([_doc("doc_a"), _doc("doc_b")]), "Wrong Person")
    ev = OpenEndedEvaluator(concurrency=1)  # no judge_model
    with patch("agentic_autorag.examiner.evaluator.llm_diagnose_failure", new=AsyncMock()) as diag:
        res = await ev.evaluate(pipeline, [_tier_b_q()])
    qr = res.question_results[0]
    assert qr.failure_class == "generation_wrong"
    assert qr.retrieved_spans == 2 and qr.n_spans == 2
    assert qr.retrieval_complete_rank == 2
    diag.assert_not_called()  # tier B never calls the diagnosis judge


@pytest.mark.asyncio
async def test_tier_b_gold_missing_is_retrieval_miss() -> None:
    pipeline = _FakePipeline(_FakeRetrieval([_doc("noise_1"), _doc("noise_2")]), "Wrong Person")
    ev = OpenEndedEvaluator(concurrency=1)
    res = await ev.evaluate(pipeline, [_tier_b_q()])
    qr = res.question_results[0]
    assert qr.failure_class == "retrieval_miss"
    assert qr.retrieved_spans == 0 and qr.retrieval_first_gold_rank == 0


@pytest.mark.asyncio
async def test_tier_b_partial_retrieval_is_partial() -> None:
    pipeline = _FakePipeline(_FakeRetrieval([_doc("doc_a"), _doc("noise")]), "Wrong Person")
    ev = OpenEndedEvaluator(concurrency=1)
    res = await ev.evaluate(pipeline, [_tier_b_q()])
    qr = res.question_results[0]
    assert qr.failure_class == "retrieval_partial"
    assert qr.retrieved_spans == 1 and qr.retrieval_first_gold_rank == 1


# --- tier A: gated diagnosis judge ---------------------------------------


@pytest.mark.asyncio
async def test_tier_a_wrong_fires_diagnosis_judge_context_insufficient() -> None:
    pipeline = _FakePipeline(_FakeRetrieval([_doc("x")]), "Wrong Person")
    ev = OpenEndedEvaluator(concurrency=1, judge_model="test/judge")
    with (
        patch("agentic_autorag.examiner.evaluator.llm_judge", new=AsyncMock(return_value=0)),
        patch(
            "agentic_autorag.examiner.evaluator.llm_diagnose_failure",
            new=AsyncMock(return_value="context_insufficient"),
        ) as diag,
    ):
        res = await ev.evaluate(pipeline, [_tier_a_q()])
    qr = res.question_results[0]
    diag.assert_awaited_once()
    assert qr.failure_class == "retrieval_miss"


@pytest.mark.asyncio
async def test_tier_a_wrong_context_present_is_generation() -> None:
    pipeline = _FakePipeline(_FakeRetrieval([_doc("x")]), "Wrong Person")
    ev = OpenEndedEvaluator(concurrency=1, judge_model="test/judge")
    with (
        patch("agentic_autorag.examiner.evaluator.llm_judge", new=AsyncMock(return_value=0)),
        patch(
            "agentic_autorag.examiner.evaluator.llm_diagnose_failure",
            new=AsyncMock(return_value="context_present_but_wrong"),
        ),
    ):
        res = await ev.evaluate(pipeline, [_tier_a_q()])
    assert res.question_results[0].failure_class == "generation_wrong"


@pytest.mark.asyncio
async def test_tier_a_correct_does_not_fire_diagnosis_judge() -> None:
    pipeline = _FakePipeline(_FakeRetrieval([_doc("x")]), "Sarah Smith")
    ev = OpenEndedEvaluator(concurrency=1, judge_model="test/judge")
    with patch("agentic_autorag.examiner.evaluator.llm_diagnose_failure", new=AsyncMock()) as diag:
        res = await ev.evaluate(pipeline, [_tier_a_q()])
    assert res.question_results[0].correct is True
    diag.assert_not_called()


@pytest.mark.asyncio
async def test_tier_a_no_judge_model_leaves_unattributed() -> None:
    pipeline = _FakePipeline(_FakeRetrieval([_doc("x")]), "Wrong Person")
    ev = OpenEndedEvaluator(concurrency=1)  # no judge
    res = await ev.evaluate(pipeline, [_tier_a_q()])
    assert res.question_results[0].failure_class == "unattributed"


@pytest.mark.asyncio
async def test_tier_c_never_fires_diagnosis_judge() -> None:
    q = OpenEndedQuestion(
        id="c1",
        question="q?",
        canonical_answer="Sarah Smith",
        reasoning_type="bridge",
        source_doc_ids=["doc_a", "doc_b"],
        source_spans=["span one", "span two"],
    )
    pipeline = _FakePipeline(_FakeRetrieval([]), "Wrong Person")
    ev = OpenEndedEvaluator(concurrency=1, judge_model="test/judge")
    with (
        patch("agentic_autorag.examiner.evaluator.llm_judge", new=AsyncMock(return_value=0)),
        patch("agentic_autorag.examiner.evaluator.llm_diagnose_failure", new=AsyncMock()) as diag,
    ):
        res = await ev.evaluate(pipeline, [q])
    # No docs retrieved ⇒ span miss, forked mechanically; judge untouched.
    assert res.question_results[0].failure_class == "retrieval_miss"
    diag.assert_not_called()


# --- consolidated failure_mode -------------------------------------------


def test_failure_mode_prefers_failure_class() -> None:
    qr = QuestionResult(
        question_id="x",
        correct=False,
        selected_answer="w",
        correct_answer="a",
        retrieved_context="",
        generated_response="w",
        failure_class="generation_wrong",
        n_spans=0,
    )
    assert failure_mode(qr) == "generation_wrong"


def test_failure_mode_fallback_for_correct() -> None:
    qr = QuestionResult(
        question_id="x",
        correct=True,
        selected_answer="a",
        correct_answer="a",
        retrieved_context="",
        generated_response="a",
        n_spans=2,
        retrieved_spans=2,
    )
    assert failure_mode(qr) == "retrieval_complete"


def test_cross_tab_and_attribution_handle_none_reasoning_type() -> None:
    # Failures across tiers, some with reasoning_type=None (tier A/B), must not
    # break the cross-tab (sorts by reasoning_type) or the attribution.
    from agentic_autorag.optimizer.state import build_failure_attribution, build_failure_cross_tab

    questions = [
        OpenEndedQuestion(
            id="c1",
            question="q?",
            canonical_answer="a",
            reasoning_type="bridge",
            source_doc_ids=["d"],
            source_spans=["s"],
        ),
        OpenEndedQuestion(id="b1", question="q?", canonical_answer="a", supporting_doc_ids=["d"]),
        OpenEndedQuestion(id="a1", question="q?", canonical_answer="a"),
    ]
    results = [
        QuestionResult(
            question_id="c1",
            correct=False,
            selected_answer="w",
            correct_answer="a",
            retrieved_context="",
            generated_response="w",
            n_spans=1,
            retrieved_spans=0,
            failure_class="retrieval_miss",
        ),
        QuestionResult(
            question_id="b1",
            correct=False,
            selected_answer="w",
            correct_answer="a",
            retrieved_context="",
            generated_response="w",
            n_spans=1,
            retrieved_spans=1,
            supporting_doc_ids=["d"],
            failure_class="generation_wrong",
        ),
        QuestionResult(
            question_id="a1",
            correct=False,
            selected_answer="w",
            correct_answer="a",
            retrieved_context="",
            generated_response="w",
            n_spans=0,
            failure_class="unattributed",
        ),
    ]
    cross_tab = build_failure_cross_tab(results, questions)  # must not raise
    assert "unknown" in cross_tab  # tier-A/B reasoning_type rendered as "unknown"
    attribution = build_failure_attribution(results)
    assert attribution.retrieval + attribution.generation == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_mixed_tier_exam_aggregates_without_error() -> None:
    # Tier C (no docs → miss), tier B (partial), tier A (judge) in one exam.
    tier_c = OpenEndedQuestion(
        id="c1",
        question="q?",
        canonical_answer="X",
        reasoning_type="bridge",
        source_doc_ids=["doc_a"],
        source_spans=["span"],
    )
    pipeline = _FakePipeline(_FakeRetrieval([_doc("doc_a")]), "Wrong")
    ev = OpenEndedEvaluator(concurrency=1, judge_model="test/judge")
    with (
        patch("agentic_autorag.examiner.evaluator.llm_judge", new=AsyncMock(return_value=0)),
        patch(
            "agentic_autorag.examiner.evaluator.llm_diagnose_failure",
            new=AsyncMock(return_value="context_insufficient"),
        ),
    ):
        res = await ev.evaluate(pipeline, [tier_c, _tier_b_q(), _tier_a_q()])
    # Aggregation and per-question classes computed without KeyError.
    assert res.n_total == 3
    classes = {qr.question_id: qr.failure_class for qr in res.question_results}
    assert classes["a1"] == "retrieval_miss"  # tier A via judge
