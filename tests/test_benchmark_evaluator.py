"""End-to-end test for FreeFormEvaluator with a stubbed RAGPipeline."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_autorag.benchmark_eval.evaluator import FreeFormEvaluator, is_error_sentinel
from agentic_autorag.benchmarks.schema import BenchmarkQAPair
from agentic_autorag.engine.pipeline import RetrievalResult, RetrievalTiming, RetrievedDocument


@dataclass
class _FakeRuntimeConfig:
    llm_timeout_s: float = 30.0


_FAKE_GEN_COST = {"usd": 0.001, "prompt_tokens": 100, "completion_tokens": 20}
_FAKE_EXPANSION_COST = {"usd": 0.0002, "prompt_tokens": 30, "completion_tokens": 5}


class _FakePipeline:
    """Minimal stand-in for RAGPipeline: deterministic retrieve + scripted generate."""

    def __init__(self, answers_by_question: dict[str, str], doc_ids: list[str]) -> None:
        self._answers = answers_by_question
        self._doc_ids = doc_ids
        self.config = _FakeRuntimeConfig()

    async def retrieve(self, query: str) -> RetrievalResult:
        docs = [
            RetrievedDocument(id=f"chunk_{i}", text=f"ctx for {d}", score=1.0 - i * 0.1, metadata={"doc_id": d})
            for i, d in enumerate(self._doc_ids)
        ]
        return RetrievalResult(documents=docs, timing=RetrievalTiming(), expansion_cost=dict(_FAKE_EXPANSION_COST))

    async def generate(self, prompt: str) -> tuple[str, dict[str, float | int]]:
        for q, ans in self._answers.items():
            if q in prompt:
                return ans, dict(_FAKE_GEN_COST)
        return "", dict(_FAKE_GEN_COST)


async def test_evaluator_populates_core_metrics() -> None:
    qa_pairs = [
        BenchmarkQAPair(
            id="q1",
            question="Who directed Doctor Strange?",
            gold_answers=["Scott Derrickson"],
            supporting_doc_ids=["scott_derrickson"],
        ),
        BenchmarkQAPair(
            id="q2",
            question="What is the capital of France?",
            gold_answers=["Paris"],
            supporting_doc_ids=["paris"],
        ),
        BenchmarkQAPair(
            id="q3",
            question="Who wrote Hamlet?",
            gold_answers=["Shakespeare", "William Shakespeare"],
            supporting_doc_ids=["shakespeare"],
        ),
    ]
    # q1 retrieval miss, correct generation; q2 retrieval hit, correct; q3 retrieval miss, alias hit.
    pipeline = _FakePipeline(
        answers_by_question={
            "Doctor Strange": "Scott Derrickson",
            "capital of France": "Paris",
            "Hamlet": "William Shakespeare",
        },
        doc_ids=["paris", "other", "filler"],  # same for every query
    )
    evaluator = FreeFormEvaluator(concurrency=2, judge_model=None)

    results = await evaluator.evaluate(pipeline, qa_pairs)

    assert len(results) == 3
    assert all(not is_error_sentinel(r) for r in results)

    by_id = {r.id: r for r in results}

    # EM hits — Scott Derrickson (case match after normalization), Paris, alias match.
    assert by_id["q1"].em == 1.0
    assert by_id["q2"].em == 1.0
    assert by_id["q3"].em == 1.0
    assert by_id["q1"].f1 == 1.0

    # Retrieval: q2 is the only question with its gold in retrieved_doc_ids.
    assert by_id["q2"].retrieval_rank_of_first_gold == 1
    assert by_id["q1"].retrieval_rank_of_first_gold is None
    assert by_id["q3"].retrieval_rank_of_first_gold is None


async def test_judge_populates_when_enabled() -> None:
    qa_pairs = [
        BenchmarkQAPair(
            id="q1",
            question="Q?",
            gold_answers=["Paris"],
            supporting_doc_ids=["paris"],
        )
    ]
    pipeline = _FakePipeline({"Q?": "The answer is Paris."}, doc_ids=["paris"])

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "YES"

    with patch(
        "agentic_autorag.litellm_runtime.litellm.acompletion",
        new=AsyncMock(return_value=mock_response),
    ):
        evaluator = FreeFormEvaluator(concurrency=1, judge_model="gemini/flash")
        results = await evaluator.evaluate(pipeline, qa_pairs)

    assert results[0].judge == 1


async def test_cost_and_tokens_propagate_to_qa_result() -> None:
    """Pipeline.generate returns ``(text, cost)`` and retrieval carries
    ``expansion_cost``; both must surface on the per-question result so
    callers can sum a benchmark-wide dollar figure."""
    qa_pairs = [BenchmarkQAPair(id="q1", question="Q?", gold_answers=["x"], supporting_doc_ids=["a"])]
    pipeline = _FakePipeline({"Q?": "x"}, doc_ids=["a"])

    evaluator = FreeFormEvaluator(concurrency=1, judge_model=None)
    results = await evaluator.evaluate(pipeline, qa_pairs)

    expected_usd = _FAKE_GEN_COST["usd"] + _FAKE_EXPANSION_COST["usd"]
    expected_prompt = _FAKE_GEN_COST["prompt_tokens"] + _FAKE_EXPANSION_COST["prompt_tokens"]
    expected_completion = _FAKE_GEN_COST["completion_tokens"] + _FAKE_EXPANSION_COST["completion_tokens"]
    assert results[0].llm_cost_usd == expected_usd
    assert results[0].prompt_tokens == expected_prompt
    assert results[0].completion_tokens == expected_completion


async def test_permanent_error_not_retried() -> None:
    qa_pairs = [BenchmarkQAPair(id="q1", question="Q?", gold_answers=["x"])]

    class _ContentPolicy(Exception):
        pass

    _ContentPolicy.__name__ = "ContentPolicyViolationError"

    pipeline = _FakePipeline({}, doc_ids=["a"])

    # Monkey-patch retrieve to raise a permanent error.
    async def _raise(_query: str):
        raise _ContentPolicy("blocked")

    pipeline.retrieve = _raise  # type: ignore[method-assign]

    evaluator = FreeFormEvaluator(concurrency=1, judge_model=None)
    # Patch RETRY_COOLDOWNS_S so this test doesn't sleep in real time.
    with patch("agentic_autorag.benchmark_eval.evaluator.RETRY_COOLDOWNS_S", ()):
        results = await evaluator.evaluate(pipeline, qa_pairs)

    assert is_error_sentinel(results[0])
    assert results[0].error == "PERMANENT_LLM_ERROR"
