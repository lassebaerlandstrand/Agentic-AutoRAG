"""Pydantic models for the structured Diagnoser/Proposer hand-off."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TrialMetrics(BaseModel):
    """Per-trial signals derived mechanically from QuestionResults. Rates are
    over questions that did not hit a system-error sentinel.
    ``answer_correct_given_complete_retrieval`` is 0.0 when no question
    had complete retrieval; read alongside ``retrieval_complete``."""

    answer_accuracy: float = 0.0
    retrieval_complete: float = 0.0
    retrieval_partial: float = 0.0
    retrieval_miss: float = 0.0
    refusal_rate: float = 0.0
    answer_correct_given_complete_retrieval: float = 0.0
    n_valid: int = 0
    mean_llm_cost_per_query_usd: float = 0.0


class BundleEffectDelta(BaseModel):
    """Combined effect of all lever changes between a baseline and current trial.

    When ``len(changes) > 1``, the delta is the bundled effect and cannot be
    attributed to any individual lever from observation alone.
    """

    changes: list[str]
    score_delta: float = 0.0
    acc_given_complete_delta: float = 0.0
    retrieval_complete_delta: float = 0.0
    cost_delta_usd: float = 0.0


class Diagnosis(BaseModel):
    """Evidence-extractor output: narrative + confirmed findings + notable
    deltas + illustrative qids. Does not prescribe levers. Mechanical
    failure-mode counts are attached separately to the Proposer's state
    card; the Diagnoser does not restate them."""

    trial_metrics: TrialMetrics
    narrative: str = Field(default="", max_length=2000)
    confirmed_findings: list[str] = Field(default_factory=list, max_length=5)
    notable_deltas: list[str] = Field(default_factory=list, max_length=4)
    illustrative_qids: list[str] = Field(default_factory=list, max_length=5)


Stance = Literal["explore", "refine"]


class Strategy(BaseModel):
    """Agent-owned stance + journal carried across trials.

    ``stance`` is a self-label with no machine enforcement; the agent can
    switch any trial. In score-only mode (``cost_aware=False``) stance is
    always ``None`` — the orchestrator validates the pairing at parse time.

    ``journal`` is working memory rewritten each trial — drop falsified
    beliefs, keep what's still load-bearing.
    """

    stance: Stance | None = None
    journal: str = Field(default="", max_length=6000)


class ProposalMeta(BaseModel):
    """Structured output the Proposer emits alongside the new TrialConfig.

    ``strategy`` is required on normal proposals; ``None`` on failure-
    recovery records (the orchestrator preserves the prior trial's strategy
    verbatim there and never persists the recovery's own meta).
    """

    rationale: str = ""
    strategy: Strategy | None = None


class FrontierContext(BaseModel):
    """Frontier-relative summary of the current trial — lets the Diagnoser
    reason about trajectory (am I dominated? by what? on which axes?) rather
    than only per-trial bottlenecks."""

    is_on_frontier: bool = False
    nearest_dominator_trial: int | None = None
    nearest_dominator_score: float | None = None
    nearest_dominator_cost_usd: float | None = None
    nearest_dominator_config_diff: list[str] = Field(default_factory=list)
    score_gap_to_dominator: float | None = None
    cost_gap_to_dominator_usd: float | None = None


class StateCard(BaseModel):
    """Mechanical optimizer-state summary fed to both agents. The phase is
    owned by the agent via ``Strategy.stance``; this card hands over the
    data (Pareto frontier, hypervolume, best-score trial, prior carry-over).
    When ``cost_aware=False`` every Pareto/cost field stays at its
    zero/empty default and renderers strip the cost sections entirely."""

    cost_aware: bool = True
    trial_number: int
    trials_remaining: int
    best_score_so_far: float
    best_trial_number: int | None
    last_trial_delta: float
    # ``trial_number - best_trial_number``; 0 when the current trial set the best.
    trials_since_best_score: int = 0
    # One entry per surveyed lever, e.g. ``{"label": "generators", "tried": 3,
    # "total": 13}``. Empty when ``search_space_sizes`` wasn't supplied.
    coverage: list[dict] = Field(default_factory=list)
    trial_summaries: list[dict] = Field(default_factory=list)
    # Pareto state (score↑ × cost↓). All zero/empty when ``cost_aware=False``.
    pareto_frontier: list[dict] = Field(default_factory=list)
    hypervolume: float = 0.0
    hypervolume_delta_last_3: float = 0.0
    # Trailing trials whose config is not on the current frontier; 0 when the
    # latest trial landed a non-dominated point. Climbing → recent moves aren't
    # extending the frontier.
    trials_since_frontier_improved: int = 0
    current_trial_cost_usd: float = 0.0
    previous_strategy: Strategy | None = None
    # ``(trial_number, stance)`` for every prior stance, chronological order.
    # Rendered RLE-encoded in the carry-over block.
    stance_history: list[tuple[int, str]] = Field(default_factory=list)
