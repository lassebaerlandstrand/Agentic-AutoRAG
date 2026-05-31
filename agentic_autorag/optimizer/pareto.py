"""Pareto-frontier helpers for two-objective (score↑, cost↓) optimization.

Pure functions used by ``state.build_state_card`` and the orchestrator.
None of these touch an LLM; the agent reads their rendered output via the
state card. Aggregations here are arithmetic (dominance, hypervolume)
— never interpretive averages over heterogeneous trial baselines.
"""

from __future__ import annotations

from typing import Protocol

_RANGE_EPSILON = 1e-9

# Hypervolume cost reference = this multiple of the worst observed cost. Must be
# > 1 so the costliest frontier member sits strictly inside the reference and
# sweeps a non-zero box; otherwise the score leader (usually the priciest) adds
# no hypervolume and cheap points dominate the signal.
_HV_COST_REF_MULTIPLIER = 2.0


class _ScoreCostRecord(Protocol):
    """Minimal protocol any record passed to these helpers must satisfy."""

    trial_number: int
    score: float
    mean_llm_cost_per_query_usd: float


def dominates(a: _ScoreCostRecord, b: _ScoreCostRecord) -> bool:
    """Return True if ``a`` Pareto-dominates ``b`` on (score↑, cost↓).

    Domination requires ``a`` weakly better on both axes and strictly better on
    at least one. Two records with identical (score, cost) do NOT dominate
    each other and both stay on the frontier.
    """
    a_score, a_cost = float(a.score), float(a.mean_llm_cost_per_query_usd)
    b_score, b_cost = float(b.score), float(b.mean_llm_cost_per_query_usd)
    weakly_better = a_score >= b_score and a_cost <= b_cost
    strictly_better = a_score > b_score or a_cost < b_cost
    return weakly_better and strictly_better


def compute_frontier(records: list[_ScoreCostRecord]) -> list[_ScoreCostRecord]:
    """Return the non-dominated subset, sorted by score ascending then cost ascending.

    Identity is keyed on ``trial_number``, not ``is``, so value-equal copies of
    the same trial (e.g. records re-hydrated from disk) are not treated as
    distinct competitors that dominate the original. Trial numbers are unique
    by construction, so two records with the same ``trial_number`` are the
    same trial regardless of object identity.
    """
    frontier: list[_ScoreCostRecord] = []
    for r in records:
        if any(dominates(other, r) for other in records if other.trial_number != r.trial_number):
            continue
        frontier.append(r)
    frontier.sort(
        key=lambda r: (
            float(r.score),
            float(r.mean_llm_cost_per_query_usd),
        )
    )
    return frontier


def compute_ranks(records: list[_ScoreCostRecord]) -> dict[int, int]:
    """Map ``trial_number`` → number of records that dominate it (0 = on frontier)."""
    return {
        int(r.trial_number): sum(1 for other in records if other.trial_number != r.trial_number and dominates(other, r))
        for r in records
    }


def cost_reference(cost_values: list[float]) -> float:
    """Cost axis of the hypervolume reference point.

    The reference must strictly exceed the worst observed cost; otherwise the
    costliest frontier member (usually the score leader) sits on the reference
    and sweeps zero width, so raising the score ceiling adds no hypervolume.
    Twice the worst cost makes the leader's box proportional to its score, so
    ceiling gains dominate the HV signal. The multiplier only scales absolute
    HV — not which trajectory has the larger delta. Falls back to 1.0 when no
    positive cost has been observed (e.g. local-only models).
    """
    positive = [c for c in cost_values if c > 0.0]
    if not positive:
        return 1.0
    return _HV_COST_REF_MULTIPLIER * max(positive)


def compute_hypervolume(
    frontier: list[_ScoreCostRecord],
    *,
    ref_point: tuple[float, float],
) -> float:
    """2D hypervolume of the (score↑, cost↓) frontier vs ``ref_point=(score_ref, cost_ref)``.

    For our setup ``score_ref=0.0`` (worst score) and ``cost_ref`` comes from
    ``cost_reference`` (strictly above the worst observed cost, so the costliest
    frontier member still sweeps a non-zero box). HV is the staircase area swept
    by the frontier above and to the left of the reference. Records worse than
    the reference on either axis contribute nothing. Cost is clamped at
    ``ref_point[1]`` so the box never has negative width.
    """
    if not frontier:
        return 0.0
    score_ref, cost_ref = ref_point
    points = sorted(
        ((float(r.score), min(float(r.mean_llm_cost_per_query_usd), cost_ref)) for r in frontier),
        key=lambda p: p[0],
        reverse=True,
    )
    hv = 0.0
    prev_cost = cost_ref
    for score, cost in points:
        if score <= score_ref or cost >= cost_ref:
            continue
        width = max(0.0, prev_cost - cost)
        height = score - score_ref
        hv += width * height
        prev_cost = cost
    return hv


def nearest_dominator(
    record: _ScoreCostRecord,
    records: list[_ScoreCostRecord],
) -> _ScoreCostRecord | None:
    """Return the dominating record with the smallest normalised Manhattan distance.

    Distance is computed in ``(score, cost)`` space normalised by the observed
    range across ``records``, so the two axes contribute comparably regardless
    of unit scale. Returns ``None`` if no record dominates ``record``.
    """
    dominators = [r for r in records if r.trial_number != record.trial_number and dominates(r, record)]
    if not dominators:
        return None
    score_values = [float(r.score) for r in records]
    cost_values = [float(r.mean_llm_cost_per_query_usd) for r in records]
    score_range = max(max(score_values) - min(score_values), _RANGE_EPSILON)
    cost_range = max(max(cost_values) - min(cost_values), _RANGE_EPSILON)
    target_score = float(record.score)
    target_cost = float(record.mean_llm_cost_per_query_usd)

    def _distance(r: _ScoreCostRecord) -> float:
        ds = abs(float(r.score) - target_score) / score_range
        dc = abs(float(r.mean_llm_cost_per_query_usd) - target_cost) / cost_range
        return ds + dc

    return min(dominators, key=_distance)
