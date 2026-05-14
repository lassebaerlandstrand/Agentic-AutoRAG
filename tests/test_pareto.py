"""Tests for agentic_autorag.optimizer.pareto — pure 2-objective Pareto helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_autorag.optimizer import pareto


def _r(trial: int, score: float, cost: float) -> SimpleNamespace:
    """Minimal record-shaped object for the Pareto helpers' Protocol."""
    return SimpleNamespace(
        trial_number=trial,
        score=score,
        mean_llm_cost_per_query_usd=cost,
    )


class TestDominates:
    def test_strictly_better_on_both_axes_dominates(self) -> None:
        a = _r(1, 0.9, 0.01)
        b = _r(2, 0.5, 0.10)
        assert pareto.dominates(a, b) is True
        assert pareto.dominates(b, a) is False

    def test_equal_score_strictly_lower_cost_dominates(self) -> None:
        a = _r(1, 0.7, 0.01)
        b = _r(2, 0.7, 0.05)
        assert pareto.dominates(a, b) is True

    def test_identical_does_not_dominate(self) -> None:
        a = _r(1, 0.7, 0.05)
        b = _r(2, 0.7, 0.05)
        assert pareto.dominates(a, b) is False
        assert pareto.dominates(b, a) is False

    def test_better_score_worse_cost_does_not_dominate(self) -> None:
        a = _r(1, 0.9, 0.10)
        b = _r(2, 0.5, 0.01)
        assert pareto.dominates(a, b) is False
        assert pareto.dominates(b, a) is False


class TestComputeFrontier:
    def test_empty_returns_empty(self) -> None:
        assert pareto.compute_frontier([]) == []

    def test_single_record_is_on_frontier(self) -> None:
        r = _r(1, 0.5, 0.05)
        assert pareto.compute_frontier([r]) == [r]

    def test_all_dominated_by_one(self) -> None:
        leader = _r(1, 0.9, 0.01)
        rest = [_r(2, 0.5, 0.05), _r(3, 0.6, 0.10), _r(4, 0.4, 0.20)]
        frontier = pareto.compute_frontier([leader, *rest])
        assert frontier == [leader]

    def test_three_non_dominated(self) -> None:
        # Score-cost mix: (0.9, $0.10) leader, (0.7, $0.05), (0.5, $0.01)
        # cheap-low. Two dominated points slot above each frontier member.
        r_top = _r(1, 0.9, 0.10)
        r_mid = _r(2, 0.7, 0.05)
        r_low = _r(3, 0.5, 0.01)
        d_high_cost = _r(4, 0.6, 0.10)  # dominated by r_top (lower score, equal cost)
        d_low_score = _r(5, 0.4, 0.05)  # dominated by r_mid (lower score, equal cost)

        frontier = pareto.compute_frontier([r_top, r_mid, r_low, d_high_cost, d_low_score])
        # Sorted by (score, cost) ascending:
        assert frontier == [r_low, r_mid, r_top]


class TestComputeRanks:
    def test_frontier_records_have_rank_zero(self) -> None:
        r1 = _r(1, 0.9, 0.10)
        r2 = _r(2, 0.5, 0.01)
        # d is dominated by BOTH r1 (better score, equal cost) and r2 (better
        # score AND better cost) → rank 2.
        d = _r(3, 0.4, 0.10)
        ranks = pareto.compute_ranks([r1, r2, d])
        assert ranks[1] == 0
        assert ranks[2] == 0
        assert ranks[3] == 2


class TestHypervolume:
    def test_empty_frontier_zero_hv(self) -> None:
        assert pareto.compute_hypervolume([], ref_point=(0.0, 1.0)) == 0.0

    def test_single_point_hv_is_box_area(self) -> None:
        r = _r(1, 0.8, 0.02)
        hv = pareto.compute_hypervolume([r], ref_point=(0.0, 0.10))
        # Box: width = 0.10 − 0.02 = 0.08; height = 0.8 − 0.0 = 0.8
        assert abs(hv - 0.08 * 0.8) < 1e-9

    def test_adding_dominated_record_does_not_increase_hv(self) -> None:
        r1 = _r(1, 0.9, 0.10)
        r2 = _r(2, 0.5, 0.01)
        ref = (0.0, 0.20)
        hv_before = pareto.compute_hypervolume(pareto.compute_frontier([r1, r2]), ref_point=ref)
        d = _r(3, 0.3, 0.05)  # dominated
        hv_after = pareto.compute_hypervolume(
            pareto.compute_frontier([r1, r2, d]),
            ref_point=ref,
        )
        assert hv_after == hv_before

    def test_adding_non_dominated_strictly_increases_hv(self) -> None:
        r1 = _r(1, 0.9, 0.10)
        ref = (0.0, 0.20)
        hv_before = pareto.compute_hypervolume([r1], ref_point=ref)
        r2 = _r(2, 0.5, 0.01)
        hv_after = pareto.compute_hypervolume(
            pareto.compute_frontier([r1, r2]),
            ref_point=ref,
        )
        assert hv_after > hv_before


class TestFindKnee:
    def test_empty_returns_none(self) -> None:
        assert pareto.find_knee([]) is None

    def test_concave_frontier_picks_score_per_cost_max(self) -> None:
        # Three points, score/cost: 0.5/0.01=50, 0.7/0.05=14, 0.9/0.10=9
        # Knee = highest ratio = first point.
        r_low = _r(1, 0.5, 0.01)
        r_mid = _r(2, 0.7, 0.05)
        r_top = _r(3, 0.9, 0.10)
        knee = pareto.find_knee([r_low, r_mid, r_top])
        assert knee is r_low

    def test_zero_cost_does_not_crash_and_picks_high_score(self) -> None:
        r1 = _r(1, 0.5, 0.0)
        r2 = _r(2, 0.9, 0.0)
        knee = pareto.find_knee([r1, r2])
        # With zero cost both ratios use ε; the higher score wins.
        assert knee is r2


class TestNearestDominator:
    def test_no_dominator_returns_none(self) -> None:
        r = _r(1, 0.9, 0.01)
        assert pareto.nearest_dominator(r, [r]) is None

    def test_picks_minimum_normalised_distance(self) -> None:
        target = _r(1, 0.5, 0.05)
        far = _r(2, 0.95, 0.001)  # dominates by a lot
        near = _r(3, 0.55, 0.04)  # dominates by little
        # `near` has both axes close to target — should win the distance race.
        chosen = pareto.nearest_dominator(target, [target, far, near])
        assert chosen is near


class TestSelectionPolicyParse:
    def test_max_score_no_args(self) -> None:
        p = pareto.SelectionPolicy.parse("max_score")
        assert p.kind == "max_score"

    def test_knee_no_args(self) -> None:
        p = pareto.SelectionPolicy.parse("knee")
        assert p.kind == "knee"

    def test_cheapest_above_with_threshold(self) -> None:
        p = pareto.SelectionPolicy.parse("cheapest_above:0.7")
        assert p.kind == "cheapest_above"
        assert p.score_threshold == 0.7

    def test_cheapest_above_missing_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="cheapest_above requires"):
            pareto.SelectionPolicy.parse("cheapest_above")

    def test_closest_to_two_args(self) -> None:
        p = pareto.SelectionPolicy.parse("closest_to:0.7,0.005")
        assert p.kind == "closest_to"
        assert p.target_score == 0.7
        assert p.target_cost == 0.005

    def test_closest_to_one_arg_raises(self) -> None:
        with pytest.raises(ValueError, match="closest_to requires"):
            pareto.SelectionPolicy.parse("closest_to:0.7")

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown selection policy"):
            pareto.SelectionPolicy.parse("magical_unicorn")

    def test_whitespace_tolerated(self) -> None:
        p = pareto.SelectionPolicy.parse("  cheapest_above : 0.6  ")
        assert p.kind == "cheapest_above"
        assert p.score_threshold == 0.6


class TestSelectionFunctions:
    def _frontier_records(self) -> list[SimpleNamespace]:
        # Three non-dominated trials forming a clean frontier.
        return [
            _r(1, 0.50, 0.001),  # cheap, low score
            _r(2, 0.70, 0.005),  # knee (best score-per-dollar)
            _r(3, 0.90, 0.020),  # expensive, max score
        ]

    def test_select_max_score_returns_score_leader(self) -> None:
        recs = self._frontier_records()
        chosen = pareto.select_max_score(recs)
        assert chosen is not None
        assert chosen.trial_number == 3

    def test_select_knee_returns_knee(self) -> None:
        # knee maximises score / cost: 0.50/0.001=500, 0.70/0.005=140, 0.90/0.020=45.
        # Cheapest record wins — same as find_knee directly.
        recs = self._frontier_records()
        chosen = pareto.select_knee(recs)
        assert chosen is not None
        assert chosen.trial_number == 1

    def test_select_cheapest_above_picks_cheapest_passing(self) -> None:
        recs = self._frontier_records()
        chosen = pareto.select_cheapest_above(recs, score_threshold=0.65)
        assert chosen is not None
        assert chosen.trial_number == 2  # 0.7 ≥ 0.65 and cheaper than 0.9

    def test_select_cheapest_above_returns_none_when_threshold_unmet(self) -> None:
        recs = self._frontier_records()
        chosen = pareto.select_cheapest_above(recs, score_threshold=0.99)
        assert chosen is None

    def test_select_closest_to_picks_minimum_distance(self) -> None:
        recs = self._frontier_records()
        # Aim for high-score region.
        chosen = pareto.select_closest_to(recs, target_score=0.88, target_cost=0.018)
        assert chosen is not None
        assert chosen.trial_number == 3

    def test_select_excludes_dominated_records(self) -> None:
        frontier = self._frontier_records()
        dominated = _r(4, 0.40, 0.020)  # dominated by both r2 and r3
        chosen = pareto.select_max_score([*frontier, dominated])
        # max_score should never return a dominated record.
        assert chosen is not None
        assert chosen.trial_number == 3

    def test_select_dispatches_via_policy(self) -> None:
        recs = self._frontier_records()
        knee_policy = pareto.SelectionPolicy.parse("knee")
        cheapest_policy = pareto.SelectionPolicy.parse("cheapest_above:0.5")
        max_policy = pareto.SelectionPolicy.parse("max_score")
        assert pareto.select(recs, policy=knee_policy).trial_number == 1
        assert pareto.select(recs, policy=cheapest_policy).trial_number == 1
        assert pareto.select(recs, policy=max_policy).trial_number == 3

    def test_select_empty_returns_none(self) -> None:
        for policy_str in ("max_score", "knee", "cheapest_above:0.5", "closest_to:0.5,0.01"):
            policy = pareto.SelectionPolicy.parse(policy_str)
            assert pareto.select([], policy=policy) is None
