"""Pydantic models for the structured Diagnoser/Proposer hand-off.

The Diagnoser emits a ``Diagnosis`` (per-trial metrics, failure attribution,
lever-effect deltas, evidence findings, regression flags, and illustrative
qids). The Proposer emits a ``ProposalMeta`` (changes, rationale, and a
``Strategy`` block) alongside the new ``TrialConfig``. The Strategy persists
across trials and ratchets one-way through ``search → polish → done``; the
agent owns when to transition.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TrialMetrics(BaseModel):
    """Per-trial signals derived mechanically from open-ended QuestionResults.

    All rates are over questions that did not hit a system-error sentinel.
    ``answer_correct_given_complete_retrieval`` is undefined (0.0) when no
    question had complete retrieval; the diagnoser should read it alongside
    ``retrieval_complete`` to distinguish "no signal" from "model is bad".
    """

    answer_accuracy: float = 0.0
    retrieval_complete: float = 0.0
    retrieval_partial: float = 0.0
    retrieval_miss: float = 0.0
    refusal_rate: float = 0.0
    answer_correct_given_complete_retrieval: float = 0.0
    n_valid: int = 0
    mean_llm_cost_per_query_usd: float = 0.0


class FailureAttribution(BaseModel):
    """Fraction of this trial's failures attributable to each pipeline stage.

    Computed orchestrator-side from failure-mode counts then re-emitted by the
    Diagnoser so the agent must look at the numbers (and can disagree, with
    evidence, in the narrative). Sums to ~1.0 — small rounding drift is fine.
    """

    retrieval: float = Field(default=0.0, ge=0.0, le=1.0)
    ranking: float = Field(default=0.0, ge=0.0, le=1.0)
    generation: float = Field(default=0.0, ge=0.0, le=1.0)
    composition: float = Field(default=0.0, ge=0.0, le=1.0)


RegressionAxis = Literal["score", "acc_given_complete", "retrieval_complete", "cost"]


class LeverEffectDelta(BaseModel):
    """One lever-change effect, computed orchestrator-side, surfaced to Diagnoser.

    ``change`` mirrors the ``"field: old → new"`` rendering used in
    ``ProposalMeta.changes``. The four ``*_delta`` fields are current_metrics
    minus the anchor trial's metrics on each axis.
    """

    change: str
    score_delta: float = 0.0
    acc_given_complete_delta: float = 0.0
    retrieval_complete_delta: float = 0.0
    cost_delta_usd: float = 0.0


class Diagnosis(BaseModel):
    """Structured output of the Diagnoser agent.

    The Diagnoser is an evidence-extractor: it compiles failure attribution,
    confirmed findings (grounded in lever-effect numbers or failure-mode
    counts), open questions, regression flags, and a small set of illustrative
    question IDs the Proposer can read raw. It does not prescribe levers.

    ``regression_detected`` is validated numerically by the orchestrator
    against ``LeverEffectDelta`` magnitudes — a hallucinated regression claim
    that doesn't match any axis-crossing delta is rejected and the agent must
    re-emit.
    """

    trial_metrics: TrialMetrics
    failure_attribution: FailureAttribution = Field(default_factory=FailureAttribution)
    narrative: str = Field(default="", max_length=2000)
    confirmed_findings: list[str] = Field(default_factory=list, max_length=5)
    open_questions: list[str] = Field(default_factory=list, max_length=5)
    regression_detected: bool = False
    regression_axes: list[RegressionAxis] = Field(default_factory=list)
    notable_deltas: list[str] = Field(default_factory=list, max_length=4)
    illustrative_qids: list[str] = Field(default_factory=list, max_length=5)


_DONE_REASONS = ("converged_no_new_information", "score_plateau_at_target", "budget_efficient_stop")
DoneReason = Literal["converged_no_new_information", "score_plateau_at_target", "budget_efficient_stop"]
Stance = Literal["search", "polish", "done"]


class Strategy(BaseModel):
    """Agent-owned optimization stance carried across trials.

    The agent commits to a ``stance`` and a free-text ``intent`` describing
    *what* it is currently pursuing. ``stance`` advances one-way through
    ``search → polish → done``; the validator (``_validate_strategy_transition``
    in ``reasoning_agent.py``) rejects illegal ratchet moves and lock-in
    violations. ``intent`` is the agent's narrative and is allowed to evolve
    freely inside a stance.

    ``journal`` is mutable working memory rewritten each trial — drop
    falsified beliefs, keep what's still load-bearing, ≤4000 chars (~800
    tokens). The prompt asks the agent to compress, not append.

    ``committed_at_trial`` and ``revision_count`` are filled by the
    orchestrator (not the LLM) so the agent can't fake its own commitment
    history.
    """

    stance: Stance
    intent: str = Field(default="", max_length=200)
    anchor_trial: int | None = None
    committed_at_trial: int = Field(default=1, ge=1)
    revision_count: int = Field(default=0, ge=0)
    journal: str = Field(default="", max_length=4000)
    done_reason: DoneReason | None = None
    regression_reason: str | None = None

    @model_validator(mode="after")
    def _check_done_reason(self) -> Strategy:
        """A ``done`` stance must carry a ``done_reason``; other stances must not."""
        if self.stance == "done" and self.done_reason is None:
            raise ValueError("strategy.stance='done' requires a done_reason")
        if self.stance != "done" and self.done_reason is not None:
            raise ValueError(f"strategy.done_reason is only valid when stance='done' (got stance={self.stance!r})")
        return self


class ProposalMeta(BaseModel):
    """Structured output the Proposer emits alongside the new TrialConfig.

    ``strategy`` carries the agent's stance, intent, journal, and (when
    applicable) ``anchor_trial`` of the frontier member being perturbed.
    ``changes`` is the diff against ``strategy.anchor_trial``'s config when
    set, else the diff against the most recent trial — checked by the
    orchestrator against the actual config diff for auditability.

    Required on normal proposals (the proposer enforces this); allowed to
    be ``None`` on failure-recovery records, where the orchestrator
    preserves the prior trial's strategy verbatim and never persists the
    recovery's own meta.
    """

    changes: list[str] = Field(default_factory=list)
    rationale: str = ""
    strategy: Strategy | None = None


class FrontierContext(BaseModel):
    """Frontier-relative summary of the current trial.

    Computed once per trial after evaluation and rendered into the diagnostic
    prompt so the diagnoser can reason about *trajectory* (am I dominated?
    by what? on which axes?) rather than only per-trial bottlenecks.
    """

    is_on_frontier: bool = False
    nearest_dominator_trial: int | None = None
    nearest_dominator_score: float | None = None
    nearest_dominator_cost_usd: float | None = None
    nearest_dominator_config_diff: list[str] = Field(default_factory=list)
    score_gap_to_dominator: float | None = None
    cost_gap_to_dominator_usd: float | None = None


class StateCard(BaseModel):
    """Mechanical optimizer-state summary fed to both agents.

    The optimizer phase is owned by the agent via ``Strategy.stance`` — this
    card just hands the agent the data (Pareto frontier, knee, hypervolume,
    cheapest-in-band) and the orchestrator-computed eligibility for early
    exit. The Pareto fields are arithmetic: dominance and the knee point are
    direct computations over (score, cost), not interpretive aggregates.

    ``previous_strategy``, ``strategy_history_summary``, and
    ``revision_count_this_run`` make the agent's own trajectory visible so
    every trial sees its prior commitments rather than starting fresh.
    """

    trial_number: int
    trials_remaining: int
    best_score_so_far: float
    best_trial_number: int | None
    last_trial_delta: float
    trial_summaries: list[dict] = Field(default_factory=list)
    # Pareto state (score↑ × cost↓)
    pareto_frontier: list[dict] = Field(default_factory=list)
    hypervolume: float = 0.0
    hypervolume_delta_last_3: float = 0.0
    knee_trial_number: int | None = None
    nearest_dominator_trial: int | None = None
    current_trial_cost_usd: float = 0.0
    cheapest_at_score_threshold_usd: float | None = None
    cheapest_at_score_threshold_trial: int | None = None
    # Agent-owned strategy carry-over
    previous_strategy: Strategy | None = None
    strategy_history_summary: list[dict] = Field(default_factory=list)
    revision_count_this_run: int = 0
    # Early-exit gate (orchestrator-computed; agent can request done only when True)
    done_eligible: bool = False
    done_blocked_reason: str | None = None
