"""Pydantic models for the structured Diagnoser/Proposer hand-off."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationInfo, field_validator


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
    accuracy_delta: float = 0.0
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
    notable_deltas: list[str] = Field(default_factory=list, max_length=5)
    illustrative_qids: list[str] = Field(default_factory=list, max_length=4)


# Seed phase for the first carried-over plan: every run starts by establishing
# the achievable score ceiling before optimizing anything else.
INITIAL_PHASE = "ceiling"


# Per-field character caps for the campaign plan. Generous on purpose: a good
# plan is a few sentences, but the plan is echoed back every trial, so unbounded
# fields would bloat the carried-forward state. Over-length fields are truncated
# to these caps at parse time (see ``_truncate_to_max``) rather than rejected, so
# a verbose plan never wastes a Proposer retry.
_STRATEGY_FIELD_MAX: dict[str, int] = {"phase": 40, "plan": 4000, "notes": 3000}


class Strategy(BaseModel):
    """Agent-owned campaign plan, carried across trials and re-authored each
    trial. The agent is shown its own prior plan and either honors it or
    revises it deliberately — that continuity is what lets a per-trial loop
    reason about the whole trial budget instead of reacting locally.

    All three fields are working memory the agent rewrites (never appends) each
    trial:
      ``phase`` — the campaign part it is in (e.g. ``ceiling`` then
                  ``frontier``/``refine``); also drives the phase trajectory
                  shown back in the state card.
      ``plan``  — how it is spending the remaining trials: the budget across the
                  campaign, where it stands against it, which pipeline stage
                  currently limits the score, and the next move.
      ``notes`` — durable beliefs worth carrying; drop anything falsified.
    """

    phase: str = Field(default="", max_length=_STRATEGY_FIELD_MAX["phase"])
    plan: str = Field(default="", max_length=_STRATEGY_FIELD_MAX["plan"])
    notes: str = Field(default="", max_length=_STRATEGY_FIELD_MAX["notes"])

    @field_validator("phase", "plan", "notes", mode="before")
    @classmethod
    def _truncate_to_max(cls, v: object, info: ValidationInfo) -> str:
        """Coerce to str and truncate to the field cap, so an over-long plan is
        clipped rather than raising and burning a Proposer retry."""
        text = "" if v is None else str(v)
        limit = _STRATEGY_FIELD_MAX.get(info.field_name or "", len(text))
        return text[:limit]


class ProposalMeta(BaseModel):
    """Structured output the Proposer emits alongside the new TrialConfig.

    ``strategy`` is required on normal proposals; ``None`` on failure-
    recovery records (the orchestrator preserves the prior trial's strategy
    verbatim there and never persists the recovery's own meta).
    """

    rationale: str = ""
    strategy: Strategy | None = None


class StateCard(BaseModel):
    """Mechanical optimizer-state summary fed to both agents. The campaign
    phase is owned by the agent via ``Strategy.phase``; this card hands over the
    data (component ceilings, Pareto frontier, hypervolume, best-accuracy trial,
    prior plan carry-over).
    When ``cost_aware=False`` every Pareto/cost field stays at its
    zero/empty default and renderers strip the cost sections entirely."""

    cost_aware: bool = True
    trial_number: int
    trials_remaining: int
    best_accuracy_so_far: float
    best_trial_number: int | None
    last_trial_delta: float
    # ``trial_number - best_trial_number``; 0 when the current trial set the best.
    trials_since_best_accuracy: int = 0
    # Component ceilings observed so far — the objective decomposed into the two
    # measured, largely-separable stages. ``retrieval_complete`` is a pure
    # retrieval-stack property; ``acc_given_complete`` is a pure generator
    # property. The limiting stage is whichever is lower. Cost-neutral, rendered
    # in both modes.
    best_retrieval_complete: float = 0.0
    best_retrieval_complete_trial: int | None = None
    best_acc_given_complete: float = 0.0
    best_acc_given_complete_trial: int | None = None
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
    # ``(trial_number, phase)`` for every prior trial that declared a phase,
    # chronological order. Rendered RLE-encoded in the plan carry-over block.
    phase_history: list[tuple[int, str]] = Field(default_factory=list)
