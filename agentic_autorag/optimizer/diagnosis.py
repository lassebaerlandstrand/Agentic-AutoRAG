"""Pydantic models for the structured Diagnoser/Proposer hand-off.

The Diagnoser emits a ``Diagnosis`` (per-trial metrics, narrative,
confirmed findings, notable deltas, illustrative qids). The Proposer
emits a ``ProposalMeta`` (changes, rationale, and a ``Strategy`` block)
alongside the new ``TrialConfig``. ``Strategy`` carries the agent's
self-declared stance (``explore`` or ``refine`` in cost-aware mode;
absent in score-only mode) and a journal it rewrites each trial.
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


class BundleEffectDelta(BaseModel):
    """Combined effect of all lever changes between a baseline and current trial.

    ``changes`` is a list of ``"field: old → new"`` strings computed by
    ``_config_diff_summary`` from the two ``TrialConfig`` objects. The four
    ``*_delta`` fields are current_metrics minus the baseline trial's metrics
    on each axis. When ``len(changes) > 1``, the delta is the BUNDLED effect
    of all changes simultaneously and cannot be attributed to any individual
    lever from observation alone.
    """

    changes: list[str]
    score_delta: float = 0.0
    acc_given_complete_delta: float = 0.0
    retrieval_complete_delta: float = 0.0
    cost_delta_usd: float = 0.0


class Diagnosis(BaseModel):
    """Structured output of the Diagnoser agent.

    The Diagnoser is an evidence-extractor: it compiles narrative,
    confirmed findings (grounded in failure-mode counts or lever-effect
    numbers), notable deltas, and a small set of illustrative question IDs
    the Proposer can read raw. It does not prescribe levers.

    The orchestrator separately attaches mechanical failure-mode counts to
    the Proposer's state card; the Diagnoser does not restate them in YAML.
    """

    trial_metrics: TrialMetrics
    narrative: str = Field(default="", max_length=2000)
    confirmed_findings: list[str] = Field(default_factory=list, max_length=5)
    notable_deltas: list[str] = Field(default_factory=list, max_length=4)
    illustrative_qids: list[str] = Field(default_factory=list, max_length=5)


Stance = Literal["explore", "refine"]


class Strategy(BaseModel):
    """Agent-owned optimization stance + working memory carried across trials.

    ``stance`` is the agent's self-label for the trial's primary objective:
    ``explore`` (score-chasing) or ``refine`` (cost-chasing). It has NO
    machine enforcement — the agent can switch any trial when evidence
    supports it. In score-only mode (``cost_aware=False``) the stance is
    always ``None``: there is no cost objective to chase, so the run is
    implicitly score-chasing and no stance is declared. The orchestrator
    validates the ``cost_aware``/``stance`` pairing at parse time.

    ``journal`` is mutable working memory rewritten each trial — drop
    falsified beliefs, keep what's still load-bearing, ≤6000 chars
    (~1500 tokens). The prompt frames it as agent-to-self notes, not a
    log; the agent re-authors it in full rather than appending.
    """

    stance: Stance | None = None
    journal: str = Field(default="", max_length=6000)


class ProposalMeta(BaseModel):
    """Structured output the Proposer emits alongside the new TrialConfig.

    ``strategy`` carries the agent's stance (when cost-aware) and journal.
    The diff against the prior trial is computed mechanically from the two
    ``TrialConfig`` objects by the renderers; the agent does not restate it.

    Required on normal proposals; allowed to be ``None`` on failure-recovery
    records, where the orchestrator preserves the prior trial's strategy
    verbatim and never persists the recovery's own meta.
    """

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
    card hands the agent the data (Pareto frontier, hypervolume, best-score
    trial, prior strategy carry-over). The agent decides when to flip stance
    and when no further trials are productive (orchestrator handles
    termination via ``trials_remaining``).

    ``cost_aware`` mirrors ``MetaConfig.cost_aware`` for the renderers. When
    False, every Pareto/cost-axis field is left at its zero/empty default and
    the prompt renderers strip the cost-related sections entirely.
    """

    cost_aware: bool = True
    trial_number: int
    trials_remaining: int
    best_score_so_far: float
    best_trial_number: int | None
    last_trial_delta: float
    # ``trial_number - best_trial_number`` (0 when the current trial just set
    # the best). Rendered alongside ``best_score_so_far`` as a single integer
    # so the agent doesn't have to count plateau length from scattered trial
    # blocks.
    trials_since_best_score: int = 0
    # Search-space coverage. One entry per surveyed lever, e.g.
    # ``{"label": "generators", "tried": 3, "total": 13}``. Empty when the
    # caller didn't supply ``search_space_sizes``. Always rendered; the nudge
    # is visible regardless of stance.
    coverage: list[dict] = Field(default_factory=list)
    trial_summaries: list[dict] = Field(default_factory=list)
    # Pareto state (score↑ × cost↓). All zero/empty when ``cost_aware=False``.
    pareto_frontier: list[dict] = Field(default_factory=list)
    hypervolume: float = 0.0
    hypervolume_delta_last_3: float = 0.0
    current_trial_cost_usd: float = 0.0
    # Agent-owned strategy carry-over (slimmed: stance + journal only)
    previous_strategy: Strategy | None = None
    # ``(trial_number, stance)`` for every prior trial whose ProposalMeta
    # carried a stance, in chronological order. Empty in score-only mode and
    # for the first proposal of a run. Rendered as a run-length-encoded
    # trajectory in the carry-over block so the agent doesn't have to scan
    # every trial block to reconstruct it.
    stance_history: list[tuple[int, str]] = Field(default_factory=list)
