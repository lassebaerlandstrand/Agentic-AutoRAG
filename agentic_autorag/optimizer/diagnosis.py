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
    retrieval_partial_a_only: float = 0.0
    retrieval_partial_b_only: float = 0.0
    retrieval_miss: float = 0.0
    refusal_rate: float = 0.0
    answer_correct_given_complete_retrieval: float = 0.0
    n_valid: int = 0


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
    guidance ("explore → diverse moves; exploit → tighten the leader").
    """

    trial_number: int
    trials_remaining: int
    best_score_so_far: float
    best_trial_number: int | None
    last_trial_delta: float
    phase: Literal["explore", "exploit"] = "explore"
    trial_summaries: list[dict] = Field(default_factory=list)
