"""Pydantic models for the structured Diagnoser/Proposer hand-off.

The Diagnoser emits a ``Diagnosis`` (stage metrics, bottleneck, ranked interventions,
hypothesis check against the prior trial's ``ProposalMeta``). The Proposer emits a
``ProposalMeta`` (move type, chosen intervention, next hypothesis, working memo)
alongside the new ``TrialConfig``.

Every structured field is falsifiable at the next trial boundary by comparing
stored metrics — no LLM is involved in the hypothesis check.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Stage(StrEnum):
    RETRIEVAL = "retrieval"
    RANKING = "ranking"
    GENERATION = "generation"


class MoveType(StrEnum):
    PROBE = "PROBE"
    REFINE = "REFINE"
    PIVOT = "PIVOT"
    COMPOUND = "COMPOUND"
    REVERT = "REVERT"


# Bottleneck thresholds. Ranking is only a meaningful bottleneck when retrieval
# is already succeeding; if retrieval_success is below this ceiling the retriever
# is upstream and masks ranking problems. gold_in_reranker_window below its
# ceiling (with retrieval already clearing the first) indicates the right chunks
# exist but are ordered outside the window the LLM actually sees.
BOTTLENECK_RETRIEVAL_CEILING = 0.6
BOTTLENECK_RANKING_CEILING = 0.7


class StageMetrics(BaseModel):
    """Per-stage quality signals derived mechanically from QuestionResult fields.

    These replace the single scalar ``score`` as the diagnostic signal. The scalar
    score remains the selection objective (``HistoryLog.get_best``).
    """

    retrieval_success: float = 0.0  # rate of context_sufficient=True
    ranking_quality: float = 0.0  # mean 1/source_fact_rank over sufficient cases
    gold_in_reranker_window: float = 0.0  # rate of source_fact_rank in [1, reranker_top_n]
    generation_given_context: float = 0.0  # MCQ accuracy conditional on context_sufficient=True
    n_eligible_for_generation: int = 0  # denominator for generation_given_context

    def bottleneck(self) -> Stage:
        """Return the stage with the lowest quality signal."""
        if self.retrieval_success < BOTTLENECK_RETRIEVAL_CEILING:
            return Stage.RETRIEVAL
        if self.gold_in_reranker_window < BOTTLENECK_RANKING_CEILING and self.retrieval_success > 0:
            return Stage.RANKING
        return Stage.GENERATION


# Per-metric polarity: +1 means "higher is better", -1 means "lower is better".
# Used by ``check_prior_hypothesis`` to judge whether the metric moved in the
# improving direction, and by the proposal validator to reject hypotheses that
# predict a regression on the target metric. All current metrics are quality
# signals where higher = better.
METRIC_POLARITY: dict[str, int] = {
    "retrieval_success": +1,
    "ranking_quality": +1,
    "gold_in_reranker_window": +1,
    "generation_given_context": +1,
}


class HypothesisCheck(BaseModel):
    """Mechanical verification of the prior trial's hypothesis."""

    prior_hypothesis: str | None = None
    target_metric: str | None = None
    expected_delta: float | None = None
    observed_delta: float | None = None
    verdict: Literal["confirmed", "falsified", "inconclusive", "n/a"] = "n/a"


class Diagnosis(BaseModel):
    """Structured output of the Diagnoser agent.

    The Diagnoser identifies the bottleneck stage and names the levers that
    could address it. It deliberately does NOT propose specific model
    swaps or parameter values — that's the Proposer's job, because only
    the Proposer has the knowledge base in its prompt.
    """

    stage_metrics: StageMetrics
    bottleneck: Stage
    confidence: Literal["high", "medium", "low"] = "medium"
    hypothesis_check: HypothesisCheck
    applicable_levers: list[str] = Field(default_factory=list)
    narrative: str = ""  # ≤ 300 words of prose reasoning


class ProposalMeta(BaseModel):
    """Structured output the Proposer emits alongside the new TrialConfig."""

    move_type: MoveType
    primary_lever: str  # which primary lever is the focus of this move
    hypothesis: str = ""
    target_metric: str = ""
    expected_delta: float = 0.0
    rationale: str = ""  # short prose: why this move, why this lever
    memo: list[str] = Field(default_factory=list)  # ≤ 5 durable findings
    revert_to_trial: int | None = None  # set only for REVERT moves


class StateCard(BaseModel):
    """Pure mechanical summary of optimizer state, fed to both agents.

    Computed from HistoryLog after each trial. No LLM involvement.
    """

    trial_number: int  # number of the *just-completed* trial
    trials_remaining: int
    best_score_so_far: float
    best_trial_number: int | None
    last_trial_delta: float  # current score minus previous trial's score
    consecutive_non_improvements: int
    current_bottleneck: Stage
    bottleneck_stable: bool  # same bottleneck as previous trial
    # (lever, value_from, value_to, verdict). value_from/to are stringified so the tuple
    # stays JSON-serialisable across heterogeneous lever types (enums, ints, floats, strings).
    interventions_tried: list[tuple[str, str, str, str]] = Field(default_factory=list)
    top_trials: list[dict] = Field(default_factory=list)  # small serializable dicts for prompt
    suggested_move_type: MoveType
