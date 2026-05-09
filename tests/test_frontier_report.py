"""Tests for agentic_autorag.optimizer.frontier_report — markdown rendering."""

from __future__ import annotations

from agentic_autorag.config.models import IndexType, TrialConfig
from agentic_autorag.optimizer import pareto
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
        llm_model="ollama/llama3.2",
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
    def test_empty_records_renders_placeholder(self) -> None:
        text = render_report(
            records=[],
            policy=pareto.SelectionPolicy.parse("max_score"),
            recommended_trial=None,
            include_graph=False,
        )
        assert "No trials completed" in text

    def test_table_lists_every_frontier_member(self) -> None:
        records = [
            _make_record(1, 0.50, 0.001),
            _make_record(2, 0.70, 0.005),
            _make_record(3, 0.90, 0.020),
        ]
        text = render_report(
            records=records,
            policy=pareto.SelectionPolicy.parse("max_score"),
            recommended_trial=3,
            include_graph=False,
        )
        # Header.
        assert "# Pareto Frontier Report" in text
        # Frontier table.
        assert "## Frontier" in text
        for trial_num in (1, 2, 3):
            assert f"| {trial_num} |" in text
        # Knee + max-score + recommended annotations.
        assert "knee" in text
        assert "max score" in text
        assert "recommended" in text

    def test_recommended_trial_appears_in_summary(self) -> None:
        records = [
            _make_record(1, 0.60, 0.001),
            _make_record(2, 0.85, 0.010),
        ]
        text = render_report(
            records=records,
            policy=pareto.SelectionPolicy.parse("knee"),
            recommended_trial=1,
            include_graph=False,
        )
        assert "Recommended trial**: #1" in text
        assert "knee" in text

    def test_full_configs_section_renders_yaml(self) -> None:
        records = [
            _make_record(1, 0.50, 0.001),
            _make_record(2, 0.80, 0.010),
        ]
        text = render_report(
            records=records,
            policy=pareto.SelectionPolicy.parse("max_score"),
            recommended_trial=2,
            include_graph=False,
        )
        assert "## Per-frontier-member configs" in text
        # Each frontier member's config block is fenced YAML and renders
        # every TrialConfig field name at least once across the report.
        for field_name in (
            "embedding_model",
            "chunking_strategy",
            "chunk_token_size",
            "top_k",
            "llm_model",
            "temperature",
            "reasoning",
        ):
            assert field_name in text

    def test_unmet_policy_explains_no_recommendation(self) -> None:
        records = [
            _make_record(1, 0.40, 0.001),
            _make_record(2, 0.50, 0.005),
        ]
        text = render_report(
            records=records,
            policy=pareto.SelectionPolicy.parse("cheapest_above:0.99"),
            recommended_trial=None,
            include_graph=False,
        )
        assert "no frontier member satisfies the policy" in text.lower()

    def test_dominated_records_omitted_from_frontier_table(self) -> None:
        # Trial 3 is dominated by trial 2 (lower score, equal cost).
        records = [
            _make_record(1, 0.50, 0.001),
            _make_record(2, 0.80, 0.010),
            _make_record(3, 0.40, 0.010),
        ]
        text = render_report(
            records=records,
            policy=pareto.SelectionPolicy.parse("max_score"),
            recommended_trial=2,
            include_graph=False,
        )
        # The frontier table cell uses "| <trial_number> |"; trial 3 should not
        # appear there even though it appears in the records list.
        assert "| 3 |" not in text
