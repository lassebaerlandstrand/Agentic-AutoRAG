"""Pareto-frontier helpers for two-objective (score↑, cost↓) optimization.

Pure functions used by ``state.build_state_card`` and the orchestrator.
None of these touch an LLM; the agent reads their rendered output via the
state card. Aggregations here are arithmetic (dominance, hypervolume, knee)
— never interpretive averages over heterogeneous trial baselines.
"""

from __future__ import annotations

from typing import Literal, Protocol

_KNEE_EPSILON = 1e-9

# Polish-phase defaults. ``polish_fraction`` is the tail share of the trial
# budget eligible for cost reduction; ``polish_score_floor`` gates polish on
# having a working config; ``polish_score_tolerance`` is the score band
# around the leader the agent is asked to hold while polishing.
DEFAULT_POLISH_FRACTION = 0.3
DEFAULT_POLISH_SCORE_FLOOR = 0.5
DEFAULT_POLISH_SCORE_TOLERANCE = 0.05


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
    """Return the non-dominated subset, sorted by score ascending then cost ascending."""
    frontier: list[_ScoreCostRecord] = []
    for r in records:
        if any(dominates(other, r) for other in records if other is not r):
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
    return {int(r.trial_number): sum(1 for other in records if other is not r and dominates(other, r)) for r in records}


def compute_hypervolume(
    frontier: list[_ScoreCostRecord],
    *,
    ref_point: tuple[float, float],
) -> float:
    """2D hypervolume of the (score↑, cost↓) frontier vs ``ref_point=(score_ref, cost_ref)``.

    For our setup ``score_ref=0.0`` (worst score) and ``cost_ref=max_observed_cost``
    (worst cost). HV is the staircase area swept by the frontier above and to the
    left of the reference. Records worse than the reference on either axis
    contribute nothing. Cost is clamped at ``ref_point[1]`` so the box never
    has negative width.
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


def find_knee(frontier: list[_ScoreCostRecord]) -> _ScoreCostRecord | None:
    """Return the frontier record maximising score / max(cost, ε).

    The knee is the most score-efficient point. If all frontier records have
    cost == 0 (e.g. local-only models), returns the record with the highest
    score so the agent still has a coherent reference.
    """
    if not frontier:
        return None
    return max(frontier, key=lambda r: float(r.score) / max(float(r.mean_llm_cost_per_query_usd), _KNEE_EPSILON))


def nearest_dominator(
    record: _ScoreCostRecord,
    records: list[_ScoreCostRecord],
) -> _ScoreCostRecord | None:
    """Return the dominating record with the smallest normalised Manhattan distance.

    Distance is computed in ``(score, cost)`` space normalised by the observed
    range across ``records``, so the two axes contribute comparably regardless
    of unit scale. Returns ``None`` if no record dominates ``record``.
    """
    dominators = [r for r in records if r is not record and dominates(r, record)]
    if not dominators:
        return None
    score_values = [float(r.score) for r in records]
    cost_values = [float(r.mean_llm_cost_per_query_usd) for r in records]
    score_range = max(max(score_values) - min(score_values), _KNEE_EPSILON)
    cost_range = max(max(cost_values) - min(cost_values), _KNEE_EPSILON)
    target_score = float(record.score)
    target_cost = float(record.mean_llm_cost_per_query_usd)

    def _distance(r: _ScoreCostRecord) -> float:
        ds = abs(float(r.score) - target_score) / score_range
        dc = abs(float(r.mean_llm_cost_per_query_usd) - target_cost) / cost_range
        return ds + dc

    return min(dominators, key=_distance)


def phase_label(
    *,
    trial_number: int,
    max_trials: int,
    best_score: float,
    polish_fraction: float = DEFAULT_POLISH_FRACTION,
    polish_score_floor: float = DEFAULT_POLISH_SCORE_FLOOR,
) -> Literal["search", "polish"]:
    """Decide phase from trial budget and best score so far.

    Mechanical split: the last ``polish_fraction`` of the budget is *eligible*
    for polish. Score-floor gate: polish only engages once a working config
    exists (``best_score >= polish_score_floor``); below that, the agent stays
    in search even past the eligibility line, because polishing a broken
    config wastes budget. ``polish_fraction=0.0`` recovers pure score-only
    optimisation; ``polish_fraction=1.0`` is polish from trial 1 (whenever
    score floor is met).
    """
    if polish_fraction <= 0.0 or max_trials <= 0:
        return "search"
    if best_score < polish_score_floor:
        return "search"
    eligible_at = max_trials * (1.0 - polish_fraction)
    return "polish" if trial_number > eligible_at else "search"
