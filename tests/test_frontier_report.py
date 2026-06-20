"""Tests for agentic_autorag.optimizer.frontier_report — section renderers."""

from __future__ import annotations

from agentic_autorag.config.models import IndexType, TrialConfig
from agentic_autorag.optimizer.frontier_report import (
    build_members,
    render_frontier_chart,
    render_frontier_table,
    render_full_configs,
    render_recommended_config,
    render_tradeoffs,
    render_trials_leaderboard,
)
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
        answer_accuracy=score,
        mean_llm_cost_per_query_usd=cost,
        total_llm_cost_usd=cost * 100.0,
        question_results=[],
    )


def _frontier() -> list[TrialRecord]:
    # Three non-dominated trials (score and cost both rise); max score is trial 3.
    return [_make_record(1, 0.5, 0.001), _make_record(2, 0.7, 0.005), _make_record(3, 0.9, 0.02)]


class TestBuildMembers:
    def test_empty_records(self) -> None:
        assert build_members([], recommended_trial=None) == []

    def test_tags_recommended_and_max_accuracy(self) -> None:
        members = build_members(_frontier(), recommended_trial=2)
        by_trial = {m.record.trial_number: m for m in members}
        assert by_trial[2].is_recommended
        assert by_trial[3].is_max_accuracy
        assert not by_trial[1].is_recommended

    def test_dominated_record_excluded(self) -> None:
        # Trial 4 is dominated by trial 3 (same cost, lower score) → not on frontier.
        records = [*_frontier(), _make_record(4, 0.4, 0.02)]
        trials = {m.record.trial_number for m in build_members(records, recommended_trial=3)}
        assert 4 not in trials

    def test_sorted_by_accuracy_ascending(self) -> None:
        members = build_members(_frontier(), recommended_trial=2)
        assert [m.record.trial_number for m in members] == [1, 2, 3]


class TestRenderTable:
    def test_marks_recommended_and_max_accuracy(self) -> None:
        out = render_frontier_table(build_members(_frontier(), recommended_trial=2))
        assert "## Pareto frontier" in out
        assert "**recommended**" in out
        assert "max accuracy" in out


class TestRenderChart:
    def test_uses_recommended_star(self) -> None:
        out = render_frontier_chart(build_members(_frontier(), recommended_trial=1))
        assert "### Accuracy vs cost" in out
        assert "★" in out

    def test_single_member_falls_back(self) -> None:
        out = render_frontier_chart(build_members([_make_record(1, 0.8, 0.01)], recommended_trial=1))
        assert "too few frontier members" in out


class TestRenderTradeoffs:
    def test_lists_deltas_vs_leader(self) -> None:
        out = render_tradeoffs(build_members(_frontier(), recommended_trial=2))
        assert "### Tradeoffs" in out
        assert "max accuracy" in out
        assert "% accuracy" in out

    def test_single_member_falls_back(self) -> None:
        out = render_tradeoffs(build_members([_make_record(1, 0.8, 0.01)], recommended_trial=1))
        assert "only one frontier member" in out


class TestRenderFullConfigs:
    def test_lists_every_frontier_member(self) -> None:
        out = render_full_configs(build_members(_frontier(), recommended_trial=2), include_graph=False)
        assert "### Per-frontier-member configs" in out
        for n in (1, 2, 3):
            assert f"#### Trial {n}" in out
        assert "generator_llm" in out  # config fields render in the embedded YAML


class TestRenderRecommendedConfig:
    def test_renders_yaml_block(self) -> None:
        out = render_recommended_config(_make_record(1, 0.8, 0.01), include_graph=False)
        assert "## Recommended config" in out
        assert "```yaml" in out
        assert "generator_llm" in out


class TestRenderLeaderboard:
    def test_ranks_by_accuracy_and_marks_best(self) -> None:
        out = render_trials_leaderboard(_frontier(), recommended_trial=3)
        assert "## Trials (by accuracy)" in out
        assert "best accuracy" in out
        assert "**recommended**" in out
        # Highest-accuracy trial (3) appears before the lowest (1) in the table body.
        body = out.split("|---", 1)[1]
        assert body.index("| 3 |") < body.index("| 1 |")
