"""Pydantic models for the structured Diagnoser/Proposer hand-off.

The Diagnoser emits a ``Diagnosis`` (per-trial metrics, ordered list of
bottlenecks). The Proposer emits a ``ProposalMeta`` (changes, rationale,
durable memo) alongside the new ``TrialConfig``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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


class Bottleneck(BaseModel):
    """One bottleneck named by the Diagnoser.

    ``stage`` is one of the four pipeline locations a fix can target.
    ``composition`` flags that the exam itself is malformed — useful evidence
    for the user even though no Proposer lever addresses it.
    """

    stage: Literal["retrieval", "ranking", "generation", "composition"]
    severity: Literal["primary", "secondary", "minor"]
    evidence: str = ""


class Diagnosis(BaseModel):
    """Structured output of the Diagnoser agent."""

    trial_metrics: TrialMetrics
    bottlenecks: list[Bottleneck] = Field(default_factory=list)
    narrative: str = ""


class ProposalMeta(BaseModel):
    """Structured output the Proposer emits alongside the new TrialConfig."""

    changes: list[str] = Field(default_factory=list)
    rationale: str = ""
    memo: list[str] = Field(default_factory=list)


class StateCard(BaseModel):
    """Mechanical optimizer-state summary fed to both agents.

    ``phase`` is a hint, not a constraint — the Proposer prompt frames it as
    guidance ("search → maximise score; polish → hold score and reduce cost").
    The Pareto fields are arithmetic: dominance and the knee point are direct
    computations over (score, cost), not interpretive aggregates.
    """

    trial_number: int
    trials_remaining: int
    best_score_so_far: float
    best_trial_number: int | None
    last_trial_delta: float
    phase: Literal["search", "polish"] = "search"
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
