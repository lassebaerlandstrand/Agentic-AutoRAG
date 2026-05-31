"""Tests for agentic_autorag.optimizer.frontier_report — markdown rendering."""

from __future__ import annotations

from agentic_autorag.config.models import IndexType, TrialConfig
from agentic_autorag.optimizer.frontier_report import render_report
from agentic_autorag.optimizer.history import TrialRecord


def _make_config(**overrides) -> TrialConfig:
    defaults = dict(
        chunking_strategy="recursive",
        chunk_token_size=512,
        chunk_token_overlap=64,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_type=IndexType.VECTOR_ONLY,
        top_k=5,
        reranker="none",
        reranker_top_n=5,
        generator_llm="ollama/llama3.2",
        temperature=0.0,
    )
    defaults.update(overrides)
    return TrialConfig(**defaults)


def _make_record(trial: int, score: float, cost: float) -> TrialRecord:
    return TrialRecord(
        trial_number=trial,
        config=_make_config(),
        score=score,
        mean_llm_cost_per_query_usd=cost,
        total_llm_cost_usd=cost * 100.0,
        question_results=[],
    )


class TestRenderReport:
    """``render_report`` marks the LLM-chosen recommended member and the score
    leader, and renders the frontier table / chart / per-member configs."""

    @staticmethod
    def _frontier() -> list[TrialRecord]:
        # Three non-dominated trials (score and cost both rise); max score is 3.
        return [_make_record(1, 0.5, 0.001), _make_record(2, 0.7, 0.005), _make_record(3, 0.9, 0.02)]

    def test_empty_records(self) -> None:
        out = render_report(records=[], recommended_trial=None, include_graph=False)
        assert "No trials completed." in out

    def test_marks_recommended_and_max_score(self) -> None:
        out = render_report(records=self._frontier(), recommended_trial=2, include_graph=False)
        assert "# Pareto Frontier Report" in out
        assert "Recommended trial**: #2" in out
        assert "**recommended**" in out
        assert "max score" in out

    def test_chart_uses_recommended_star(self) -> None:
        out = render_report(records=self._frontier(), recommended_trial=1, include_graph=False)
        assert "★" in out

    def test_full_configs_lists_every_frontier_member(self) -> None:
        out = render_report(records=self._frontier(), recommended_trial=2, include_graph=False)
        assert "## Per-frontier-member configs" in out
        for n in (1, 2, 3):
            assert f"### Trial {n}" in out
        assert "generator_llm" in out  # config fields render in the embedded YAML

    def test_dominated_record_excluded_from_frontier(self) -> None:
        # Trial 4 is dominated by trial 3 (same cost, lower score) → not on frontier.
        records = [*self._frontier(), _make_record(4, 0.4, 0.02)]
        out = render_report(records=records, recommended_trial=3, include_graph=False)
        assert "### Trial 4" not in out

    def test_single_member_frontier_falls_back_on_chart(self) -> None:
        out = render_report(records=[_make_record(1, 0.8, 0.01)], recommended_trial=1, include_graph=False)
        assert "Recommended trial**: #1" in out
        assert "too few frontier members" in out
