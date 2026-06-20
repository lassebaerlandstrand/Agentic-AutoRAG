"""Tests for the LLM-written end-of-run optimization summary."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from agentic_autorag.config.models import OpenEndedQuestion, TrialConfig
from agentic_autorag.cost_ledger import CostLedger
from agentic_autorag.optimizer.diagnosis import Diagnosis, ProposalMeta, TrialMetrics
from agentic_autorag.optimizer.final_report import (
    _build_context,
    _strip_code_fence,
    assemble_summary,
    generate_final_report,
)
from agentic_autorag.optimizer.history import TrialRecord


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
        answer_accuracy=score,
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
    def test_renders_trajectory_recommendation_exam_and_cost(self) -> None:
        records = [
            _record(1, 0.4, 0.001, rationale="baseline config", narrative="retrieval miss high"),
            _record(2, 0.7, 0.002, rationale="raised top_k to 8", narrative="accuracy recovered"),
        ]
        ctx = _build_context(
            records=records,
            recommended_trial=2,
            exam=_exam(3),
            ledger=_ledger_with(("rag_eval", 0.05), ("agent_proposal", 0.02)),
            cost_aware=False,
            include_graph=False,
            corpus_description="Company filings.",
        )
        assert "Trial 1" in ctx and "Trial 2" in ctx
        assert "raised top_k to 8" in ctx
        assert "retrieval miss high" in ctx
        assert "finding A" in ctx
        assert "## Recommended configuration" in ctx
        assert "Trial 2 (accuracy=0.700" in ctx
        assert "generator_llm: gemini/gemini-3-flash-preview" in ctx
        assert "Corpus: Company filings." in ctx
        assert "Total questions: 3" in ctx
        assert "bridge: 3" in ctx
        assert "rag_eval: $0.0500" in ctx
        assert "total: $0.0700" in ctx

    def test_cost_aware_omits_preselected_recommendation(self) -> None:
        # In cost-aware mode the model picks from the frontier, so the digest
        # carries no pre-selected "Recommended configuration" section.
        ctx = _build_context(
            records=[_record(1, 0.4, 0.001), _record(2, 0.7, 0.002)],
            recommended_trial=None,
            exam=_exam(1),
            ledger=_ledger_with(("rag_eval", 0.01)),
            cost_aware=True,
            include_graph=False,
            corpus_description="docs",
        )
        assert "## Recommended configuration" not in ctx
        assert "## Pareto frontier" in ctx


class TestStripCodeFence:
    def test_removes_wrapping_fence(self) -> None:
        assert _strip_code_fence("```markdown\n## Summary\nx\n```") == "## Summary\nx"

    def test_leaves_unfenced_text(self) -> None:
        assert _strip_code_fence("## Summary\nx") == "## Summary\nx"


def _patch_completion(return_value=None, *, side_effect=None) -> AsyncMock:
    """Patch the report's single LLM call. ``acompletion_with_cost`` returns
    ``(response, cost)``; tests only care about the response content."""
    mock = AsyncMock(return_value=return_value, side_effect=side_effect)
    return patch("agentic_autorag.optimizer.final_report.acompletion_with_cost", new=mock)


class TestGenerateFinalReport:
    """``generate_final_report`` narrates score-only runs and, in cost-aware mode,
    picks the recommended trial from the Pareto frontier (validate → retry →
    max-score fallback)."""

    @staticmethod
    def _frontier_records() -> list[TrialRecord]:
        # Three non-dominated trials: score and cost both rise, so none dominates
        # another and all three land on the frontier. Max score is trial 3.
        return [_record(1, 0.5, 0.001), _record(2, 0.7, 0.005), _record(3, 0.9, 0.02)]

    async def test_score_only_returns_fallback_and_uses_score_prompt(self) -> None:
        records = self._frontier_records()
        with _patch_completion((_mock_completion("## Recommendation\nGood run."), 0.0)) as mock_call:
            trial, body = await generate_final_report(
                model="test/model",
                records=records,
                fallback_trial=3,
                exam=_exam(),
                ledger=_ledger_with(("rag_eval", 0.1)),
                cost_aware=False,
                include_graph=False,
                corpus_description="A corpus.",
            )
        assert trial == 3
        assert body == "## Recommendation\nGood run."
        # Score-only mode does no validation retry loop — exactly one LLM call;
        # the call count isn't reflected in the returned (trial, body).
        mock_call.assert_awaited_once()
        system_prompt = mock_call.await_args.kwargs["messages"][0]["content"]
        assert "optimized exam score only" in system_prompt

    async def test_cost_aware_picks_valid_frontier_trial(self) -> None:
        records = self._frontier_records()
        content = "recommended_trial: 2\n\n## Recommendation\nTrial 2 balances cost and capability."
        with _patch_completion((_mock_completion(content), 0.0)) as mock_call:
            trial, body = await generate_final_report(
                model="test/model",
                records=records,
                fallback_trial=3,
                exam=_exam(),
                ledger=_ledger_with(("rag_eval", 0.1)),
                cost_aware=True,
                include_graph=False,
                corpus_description="A corpus.",
            )
        assert trial == 2
        assert body.startswith("## Recommendation")
        assert "recommended_trial" not in body
        # A valid first pick must not trigger the retry path (contrast with the
        # retries test); trial==2 alone wouldn't prove no retry occurred.
        mock_call.assert_awaited_once()

    async def test_cost_aware_retries_then_accepts(self) -> None:
        records = self._frontier_records()
        invalid = _mock_completion("recommended_trial: 99\n\n## Recommendation\nNot on the frontier.")
        valid = _mock_completion("recommended_trial: 1\n\n## Recommendation\nCheapest viable config.")
        with _patch_completion(side_effect=[(invalid, 0.0), (valid, 0.0)]) as mock_call:
            trial, _body = await generate_final_report(
                model="test/model",
                records=records,
                fallback_trial=3,
                exam=_exam(),
                ledger=_ledger_with(("rag_eval", 0.1)),
                cost_aware=True,
                include_graph=False,
                corpus_description="A corpus.",
            )
        assert trial == 1
        assert mock_call.await_count == 2

    async def test_cost_aware_falls_back_when_pick_never_valid(self) -> None:
        records = self._frontier_records()
        bad = _mock_completion("recommended_trial: 99\n\n## Recommendation\nStill off-frontier.")
        with _patch_completion((bad, 0.0)) as mock_call:
            trial, body = await generate_final_report(
                model="test/model",
                records=records,
                fallback_trial=3,
                exam=_exam(),
                ledger=_ledger_with(("rag_eval", 0.1)),
                cost_aware=True,
                include_graph=False,
                corpus_description="A corpus.",
            )
        assert trial == 3  # max-score fallback after exhausting attempts
        assert mock_call.await_count == 2
        assert "recommended_trial" not in body

    async def test_cost_aware_falls_back_when_line_missing(self) -> None:
        records = self._frontier_records()
        no_line = _mock_completion("## Recommendation\nNo machine-readable pick here.")
        with _patch_completion((no_line, 0.0)) as mock_call:
            trial, _body = await generate_final_report(
                model="test/model",
                records=records,
                fallback_trial=3,
                exam=_exam(),
                ledger=_ledger_with(("rag_eval", 0.1)),
                cost_aware=True,
                include_graph=False,
                corpus_description="A corpus.",
            )
        assert trial == 3
        assert mock_call.await_count == 2


class TestAssembleSummary:
    """``assemble_summary`` stitches the LLM prose with the deterministic
    frontier blocks into the single ``optimization_summary.md`` body."""

    @staticmethod
    def _records() -> list[TrialRecord]:
        return [_record(1, 0.5, 0.001), _record(2, 0.7, 0.005), _record(3, 0.9, 0.02)]

    _PROSE = "## Recommendation\nTrial 2 is the pick.\n\n## What the search found\nRetrieval bottleneck."

    def test_cost_aware_embeds_table_chart_and_configs(self) -> None:
        md = assemble_summary(
            project_name="proj",
            records=self._records(),
            recommended_trial=2,
            prose_body=self._PROSE,
            include_graph=False,
            cost_aware=True,
        )
        assert md.startswith("# Optimization summary: proj")
        assert "**Recommended: trial 2**" in md
        assert "non-dominated config(s)" in md
        assert "## Pareto frontier" in md
        assert "## Recommendation" in md
        assert "## What the search found" in md
        assert "## Frontier details" in md
        assert "### Accuracy vs cost" in md
        assert "### Tradeoffs" in md
        assert "#### Trial 2" in md
        # Trimmed sections must not reappear.
        assert "## Summary" not in md
        assert "What to try next" not in md

    def test_score_only_uses_leaderboard_without_frontier_sections(self) -> None:
        md = assemble_summary(
            project_name="proj",
            records=self._records(),
            recommended_trial=3,
            prose_body=self._PROSE,
            include_graph=False,
            cost_aware=False,
        )
        assert "## Trials (by accuracy)" in md
        assert "best accuracy" in md
        assert "## Recommended config" in md
        # Cost is not an objective here — no Pareto framing, chart, or tradeoffs.
        assert "## Pareto frontier" not in md
        assert "### Accuracy vs cost" not in md
        assert "### Tradeoffs" not in md
