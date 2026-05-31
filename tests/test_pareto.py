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

    def test_value_equal_copies_of_same_trial_do_not_self_dominate(self) -> None:
        """Two record objects with the same ``trial_number`` are the same
        trial — they must not knock each other off the frontier."""
        original = _r(1, 0.9, 0.10)
        copy = _r(1, 0.9, 0.10)
        frontier = pareto.compute_frontier([original, copy])
        # The frontier dedupes by ``trial_number``: both stay because neither
        # dominates the other (filtered by trial_number identity).
        assert len(frontier) == 2
        assert all(r.trial_number == 1 for r in frontier)

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


class TestCostReference:
    def test_returns_twice_worst_observed_cost(self) -> None:
        assert pareto.cost_reference([0.01, 0.05, 0.02]) == pytest.approx(0.10)

    def test_falls_back_to_one_when_no_positive_cost(self) -> None:
        assert pareto.cost_reference([]) == 1.0
        assert pareto.cost_reference([0.0, 0.0]) == 1.0

    def test_costliest_frontier_point_contributes_unlike_max_reference(self) -> None:
        """The score leader is usually the priciest. With ``ref == max cost`` it
        sits on the reference and sweeps zero width; ``cost_reference`` pushes the
        reference beyond it so its score finally counts toward the hypervolume."""
        cheap = _r(1, 0.5, 0.01)
        leader = _r(2, 0.9, 0.05)
        frontier = pareto.compute_frontier([cheap, leader])
        costs = [0.01, 0.05]

        hv_ref_at_max = pareto.compute_hypervolume(frontier, ref_point=(0.0, max(costs)))
        hv_cheap_only = pareto.compute_hypervolume([cheap], ref_point=(0.0, max(costs)))
        hv_ref_beyond = pareto.compute_hypervolume(frontier, ref_point=(0.0, pareto.cost_reference(costs)))

        # With ref on the leader, the frontier's HV equals the cheap point alone.
        assert hv_ref_at_max == pytest.approx(hv_cheap_only)
        # Pushing the reference beyond the worst cost makes the leader count.
        assert hv_ref_beyond > hv_ref_at_max

    def test_raising_ceiling_beats_cheaper_cut_in_hv_gain(self) -> None:
        """Under the cost_reference scheme, raising the score ceiling adds more
        hypervolume than a same-score cheaper cut — inverting the old incentive."""
        base = [_r(1, 0.5, 0.01), _r(2, 0.7, 0.05)]
        raise_ceiling = [*base, _r(3, 0.85, 0.08)]
        cheaper_cut = [*base, _r(3, 0.5, 0.005)]

        # Shared reference (beyond the priciest cost in either move) so the
        # comparison isolates frontier geometry from the moving reference.
        ref = (0.0, pareto.cost_reference([0.005, 0.01, 0.05, 0.08]))

        def hv(records: list[SimpleNamespace]) -> float:
            return pareto.compute_hypervolume(pareto.compute_frontier(records), ref_point=ref)

        hv_base = hv(base)
        gain_ceiling = hv(raise_ceiling) - hv_base
        gain_cut = hv(cheaper_cut) - hv_base
        assert gain_ceiling > gain_cut


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
