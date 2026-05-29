"""Tests for the LLM-written end-of-run optimization summary."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_autorag.config.models import OpenEndedQuestion, TrialConfig
from agentic_autorag.cost_ledger import CostLedger, reset_active_ledger, set_active_ledger
from agentic_autorag.optimizer.diagnosis import Diagnosis, ProposalMeta, TrialMetrics
from agentic_autorag.optimizer.final_report import (
    _build_context,
    _strip_code_fence,
    generate_final_report,
)
from agentic_autorag.optimizer.history import TrialRecord
from agentic_autorag.optimizer.pareto import SelectionPolicy


def _metrics() -> TrialMetrics:
    return TrialMetrics(
        answer_accuracy=0.5,
        retrieval_complete=0.4,
        retrieval_partial=0.3,
        retrieval_miss=0.3,
        refusal_rate=0.1,
    )


def _record(
    n: int,
    score: float,
    cost: float,
    *,
    rationale: str = "changed top_k",
    narrative: str = "bottleneck was retrieval",
) -> TrialRecord:
    metrics = _metrics()
    return TrialRecord(
        trial_number=n,
        config=TrialConfig(generator_llm="gemini/gemini-3-flash-preview"),
        score=score,
        question_results=[],
        mean_llm_cost_per_query_usd=cost,
        trial_metrics=metrics,
        diagnosis=Diagnosis(trial_metrics=metrics, narrative=narrative, confirmed_findings=["finding A"]),
        meta=ProposalMeta(rationale=rationale),
    )


def _exam(n: int = 3) -> list[OpenEndedQuestion]:
    return [
        OpenEndedQuestion(
            id=f"q{i}",
            question="?",
            canonical_answer="a",
            reasoning_type="bridge",
            source_chunk_ids=["c"],
            source_doc_ids=["d"],
            source_spans=["s"],
        )
        for i in range(n)
    ]


def _ledger_with(*pairs: tuple[str, float]) -> CostLedger:
    ledger = CostLedger()
    for category, usd in pairs:
        ledger.record(category, usd, 100, 50)
    return ledger


def _mock_completion(content: str) -> MagicMock:
    """A litellm-shaped response with explicit usage so token extraction
    doesn't trip over auto-created MagicMock attributes."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = 1000
    response.usage.completion_tokens = 300
    response.usage.cache_read_input_tokens = 0
    response.usage.cache_creation_input_tokens = 0
    response.usage.prompt_tokens_details = None
    return response


class TestBuildContext:
    def test_includes_trajectory_recommendation_exam_and_cost(self) -> None:
        records = [
            _record(1, 0.4, 0.001, rationale="baseline config", narrative="retrieval miss high"),
            _record(2, 0.7, 0.002, rationale="raised top_k to 8", narrative="accuracy recovered"),
        ]

        ctx = _build_context(
            records=records,
            recommended=records[1],
            objective=SelectionPolicy.parse("max_score"),
            exam=_exam(3),
            ledger=_ledger_with(("rag_eval", 0.05), ("agent_proposal", 0.02)),
            cost_aware=True,
            include_graph=False,
            corpus_description="Company filings.",
        )

        assert "Trial 1" in ctx and "Trial 2" in ctx
        assert "raised top_k to 8" in ctx
        assert "retrieval miss high" in ctx
        assert "finding A" in ctx
        assert "Recommended configuration" in ctx
        assert "Trial 2 (score=0.700" in ctx
        assert "generator_llm: gemini/gemini-3-flash-preview" in ctx
        assert "Corpus: Company filings." in ctx
        assert "Total questions: 3" in ctx
        assert "bridge: 3" in ctx
        assert "rag_eval: $0.0500" in ctx
        assert "total: $0.0700" in ctx

    def test_recommended_none_when_policy_unmet(self) -> None:
        ctx = _build_context(
            records=[_record(1, 0.4, 0.001)],
            recommended=None,
            objective=SelectionPolicy.parse("cheapest_above:0.9"),
            exam=_exam(1),
            ledger=_ledger_with(("rag_eval", 0.01)),
            cost_aware=True,
            include_graph=False,
            corpus_description="docs",
        )

        assert "no frontier member satisfied the selection policy" in ctx


class TestGenerateFinalReport:
    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_uses_optimizer_model_and_credits_bucket(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_completion("## Summary\nGood run."))
        mock_litellm.completion_cost = MagicMock(return_value=0.012)
        ledger = CostLedger()
        token = set_active_ledger(ledger)
        try:
            body = await generate_final_report(
                model="gemini/gemini-3-flash-preview",
                records=[_record(1, 0.4, 0.001), _record(2, 0.7, 0.002)],
                recommended=_record(2, 0.7, 0.002),
                objective=SelectionPolicy.parse("max_score"),
                exam=_exam(2),
                ledger=ledger,
                cost_aware=True,
                include_graph=False,
                corpus_description="Company filings.",
            )
        finally:
            reset_active_ledger(token)

        assert body == "## Summary\nGood run."
        assert ledger.buckets["final_report"].usd == pytest.approx(0.012)
        assert ledger.buckets["final_report"].n_calls == 1
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["model"] == "gemini/gemini-3-flash-preview"
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"


class TestStripCodeFence:
    def test_removes_wrapping_fence(self) -> None:
        assert _strip_code_fence("```markdown\n## Summary\nx\n```") == "## Summary\nx"

    def test_leaves_unfenced_text(self) -> None:
        assert _strip_code_fence("## Summary\nx") == "## Summary\nx"
