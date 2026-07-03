"""Two-stage reasoning agent: Diagnoser interprets the just-completed
trial; Proposer picks the next ``TrialConfig``. No hard move-type validators
— guidance lives in the prompt."""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import litellm
import yaml

from agentic_autorag.config.knowledge_base import KnowledgeBase
from agentic_autorag.config.models import OpenEndedQuestion, ProjectConfig, TrialConfig
from agentic_autorag.examiner._errors import ERROR_SENTINELS
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult, failure_mode
from agentic_autorag.examiner.exam_validator import _fold_unicode
from agentic_autorag.litellm_runtime import acompletion_with_cost
from agentic_autorag.optimizer.diagnosis import (
    INITIAL_PHASE,
    BundleEffectDelta,
    Diagnosis,
    ProposalMeta,
    StateCard,
    Strategy,
    TrialMetrics,
)
from agentic_autorag.optimizer.history import HistoryLog, TrialRecord, _compact_history, _config_signature
from agentic_autorag.optimizer.state import (
    FailureAttribution,
    _top_stages_from_attribution,
    build_failure_attribution,
    build_failure_cross_tab,
    build_state_card,
    compute_bundle_effect,
    compute_trial_metrics,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

DIAGNOSTIC_PROMPT = (_PROMPTS_DIR / "diagnostic.txt").read_text(encoding="utf-8")
PROPOSAL_PROMPT = (_PROMPTS_DIR / "proposal.txt").read_text(encoding="utf-8")
INITIAL_PROPOSAL_PROMPT = (_PROMPTS_DIR / "initial_proposal.txt").read_text(encoding="utf-8")
FAILURE_RECOVERY_PROMPT = (_PROMPTS_DIR / "failure_recovery.txt").read_text(encoding="utf-8")
# OPRO (``compact_history``) baseline: dedicated, self-contained templates with
# NO knowledge base, NO diagnosis, NO campaign plan — only the search space, a
# compact config→accuracy trajectory, and a budget line. Kept as separate files
# (rather than blanking slots out of the rich templates) so the OPRO prompt is
# auditable and can't silently regain a leaked section.
OPRO_PROPOSAL_PROMPT = (_PROMPTS_DIR / "opro_proposal.txt").read_text(encoding="utf-8")
OPRO_INITIAL_PROPOSAL_PROMPT = (_PROMPTS_DIR / "opro_initial_proposal.txt").read_text(encoding="utf-8")


# The one place the two modes diverge: a single self-contained "objective +
# campaign" block per mode, rendered into the proposal prompt's static prefix.
# The score-only block carries NO cost language, so a score-only run never
# reasons about price; the cost-aware block adds the cost axes and the frontier.
# Everything else in the prompt (role, RAG mental model, plan instructions,
# search space, output schema) is shared and cost-neutral.

_CAMPAIGN_BLOCK_COST_AWARE = """
## Your objective: map the score–cost frontier

You optimize two things at once: exam score (higher is better) and
cost-per-query (lower is better). The result is a Pareto frontier — the set of
configs where nothing scores higher at the same or lower cost. The goal is the
whole curve: the best score reachable at every cost, from the cheapest config
worth running up to the ceiling — a populated frontier, not one tall point.

What a query costs: the generator is billed per token, so a query's price is
(the tokens in its context + the tokens it generates) × the generator's
per-token price, plus any separate LLM calls (query_expansion, a compressor).
The size of that context is set on the retrieval side — how many chunks reach
the generator (reranker_top_n, or top_k when there is no reranker) and how large
each is (chunk_token_size) — not by the generator alone. The state card and
history report in_tok/out_tok per trial so you can see, for any config, how much
of its price is model rate and how much is tokens read.

One reading note: the diagnosis you are shown each trial reports why questions
were missed — it measures score, not cost. A config you deliberately made
cheaper will show more misses; that is the trade-off you chose, not automatically
a regression to undo.

The run has a natural arc, and you set the budget for each part yourself from
`trials_remaining`:
- phase `ceiling`: find the highest score the corpus allows, cost aside. Until
  you know the ceiling you cannot tell what a cheaper config gives up.
- phase `frontier`: find configs that hold as much of that score as possible for
  less. Every point on the curve — cheap or mid-range — is part of the
  deliverable; a probe that lands dominated still earns its trial if it tells you
  what a region of the space is worth.

You decide how much budget each phase gets, from the facts in the state card
(trials_remaining, the component ceilings, coverage, and the frontier) and your
own prior plan — there is no fixed split. Linger on the ceiling and you have no
trials left to map the frontier; leave it too early and you map a frontier
beneath the real ceiling.
"""

_CAMPAIGN_BLOCK_SCORE_ONLY = """
## Your objective: maximize exam score

One objective: the highest exam score the corpus allows. Nothing else is
optimized — pursue score alone.

The run has a natural arc, and you set the budget for each part yourself from
`trials_remaining`:
- phase `ceiling`: range widely over strong, genuinely different approaches to
  find the highest score the corpus allows. A single strong model scoring low is
  usually a fit issue, not proof the approach is dead.
- phase `refine`: once higher scores stop coming across several different strong
  approaches, spend the rest tightening the best region — while still checking
  the ceiling is real and not a plateau you accepted too early.

You decide how much budget each phase gets, from the facts in the state card
(trials_remaining, the component ceilings, coverage) and your own prior plan —
there is no fixed split.
"""


# Initial-proposer conditional sections. The initial proposer always runs
# before any frontier exists, so framing it around score (and only score in
# score-only mode) prevents the LLM from anchoring on cheap models for
# reasons that don't apply to the active objective.
_INITIAL_PREAMBLE_COST_AWARE = (
    "Pick a strong, capable starting config. The run builds a score-against-cost "
    "Pareto frontier from the first trial; with no frontier yet, a strong first "
    "point is the biggest possible frontier improvement and anchors its top."
)
_INITIAL_PREAMBLE_SCORE_ONLY = (
    "Pick a strong starting config aimed at score. This run optimizes score only — cost is not a target."
)

_INITIAL_LLM_PICK_COST_AWARE = (
    "the most capable LLM for the corpus type, stepping down to the next-strongest "
    "only when the top model is far more expensive for little capability gain."
)
_INITIAL_LLM_PICK_SCORE_ONLY = (
    "the most capable LLM for the corpus type. Disregard price — this run optimizes score only."
)


def _initial_proposal_template_sections(cost_aware: bool) -> dict[str, str]:
    """Return the conditional-section substitutions for the initial proposer."""
    if cost_aware:
        return {
            "initial_preamble": _INITIAL_PREAMBLE_COST_AWARE,
            "initial_llm_pick_guidance": _INITIAL_LLM_PICK_COST_AWARE,
            "baseline_stance": _BASELINE_STANCE_COST_AWARE,
        }
    return {
        "initial_preamble": _INITIAL_PREAMBLE_SCORE_ONLY,
        "initial_llm_pick_guidance": _INITIAL_LLM_PICK_SCORE_ONLY,
        "baseline_stance": _BASELINE_STANCE_SCORE_ONLY,
    }


def _proposal_template_sections(cost_aware: bool) -> dict[str, str]:
    """Return the conditional-section substitutions for the proposal prompt.

    Exactly one slot varies by mode: the self-contained ``campaign_block``
    (objective + how to run the campaign for it). The score-only block carries
    no cost language."""
    return {
        "campaign_block": _CAMPAIGN_BLOCK_COST_AWARE if cost_aware else _CAMPAIGN_BLOCK_SCORE_ONLY,
    }


MAX_RETRIES = 3


def _config_retry_message(error: Exception) -> str:
    """Re-prompt after a malformed or out-of-space config proposal.

    On a search-space violation, nudges the agent to reconsider its plan rather
    than clamp the single offending value: a clamp often will not deliver the
    intended improvement, and a valid fix may require changing the related
    levers (or a different config). The violation text already lists the legal
    range/options, so this is purely a framing change.
    """
    return (
        f"Your response had an error: {error}\n\n"
        "Output a corrected ```yaml block matching the schema in the original prompt. "
        "If a value you wanted is out of range or not allowed, do NOT just snap it to "
        "the nearest legal value — that may not achieve the improvement you intended. "
        "Reconsider whether your goal is reachable within the search space and adjust "
        "the related levers (or choose a different config) so the result still pursues "
        "your stated rationale."
    )


# Max qids the regression-vs-best band may contribute to the stratified
# failure sample. Score is often non-monotonic across trials, so flagging
# items the run already solved but is now regressing on gives the Diagnoser
# a long-horizon signal the flip-vs-prev band cannot capture.
_REGRESSION_VS_BEST_BAND_SIZE = 5

# Max times the Proposer is re-prompted after emitting a config identical to a
# prior trial. The retry message shows the duplicated config and points at the
# complete "configs already tried" index so the agent can diverge. After these
# are exhausted the Proposer falls back to a programmatic non-duplicate
# perturbation rather than re-running an identical trial.
MAX_DUPLICATE_RETRIES = 3

# Max random-perturbation attempts before the Proposer fallback gives up
# and re-uses the current config unchanged. Each attempt picks one lever
# and validates against the search space; 20 is comfortably above the
# expected pool of mutable levers.
_PROPOSER_FALLBACK_PERTURBATION_ATTEMPTS = 20

# Levers the Proposer-fallback may mutate. Curated to exclude levers with
# tight cross-field dependencies (e.g. reranker_top_n vs top_k, graph_*
# fields gated on index_type) so the perturbed config validates without
# rerunning the agent.
_PROPOSER_FALLBACK_SAFE_LEVERS: tuple[str, ...] = (
    "chunking_strategy",
    "embedding_model",
    "generator_llm",
    "reranker",
    "top_k",
    "chunk_token_size",
    "chunk_token_overlap",
    "temperature",
)

_DEEP_FAILURE_SAMPLE = 12
_DEEP_SUCCESS_SAMPLE = 2
_KEY_EVIDENCE_SAMPLE = 4
_CHUNK_PREFIX_CHARS = 160  # ~40 tokens at 4 chars/token
_SPAN_WINDOW_CHARS = 160  # ~40 tokens before/after gold span


_GRAPH_RULES = """\
   - Switching to graph_only or hybrid_graph_vector changes what query
     expansion is useful (HyDE works well with graph retrieval).
   - graph_query_mode "local" is better for specific entity lookups;
     "global" for broad thematic questions; "hybrid" for balanced retrieval.
   - graph_top_k controls how many graph nodes are explored — increase it
     when the error trace shows entity_gap or relationship_missing failures.
   - graph_query_mode and graph_top_k are ONLY relevant when index_type is
     graph_only or hybrid_graph_vector.
"""


_GRAPH_GUIDANCE = """\
3. If graph-based index types are available (graph_only, hybrid_graph_vector),
   consider whether the content is relationship-rich (e.g. scientific papers
   with many named entities, legal documents with cross-references). If so,
   starting with a graph or hybrid type may be advantageous.
4. When index_type is graph_only or hybrid_graph_vector, set graph_query_mode
   and graph_top_k appropriately. "hybrid" mode generally works best as a
   starting point; larger graph_top_k captures more graph context.
"""

_BASELINE_STANCE_COST_AWARE = """\
2. Start with a strong, capable configuration — with no frontier yet, a
   high-scoring first point is the biggest frontier improvement and anchors
   its top. Keep retrieval levers general-purpose enough that the loop can
   diagnose bottlenecks, but don't hold back on the levers that drive score
   (generator model, reranker)."""

_BASELINE_STANCE_SCORE_ONLY = """\
2. Start with an ambitious configuration aimed at maximizing exam score.
   The optimization loop diagnoses retrieval, ranking, and generation
   bottlenecks each trial and proposes refined configs from there."""

_GRAPH_DIAGNOSTIC_TYPES = """\
When a graph index is in use, additional levers exist: entity-focused retrieval
via ``graph_query_mode`` and ``graph_top_k``. Entity gaps or missing relationships
can be addressed by increasing ``graph_top_k`` or swapping the graph query mode.
"""


@dataclass(frozen=True)
class SelectedFailure:
    """A question surfaced for the Diagnoser, tagged with the band that
    picked it: ``flip_vs_prev`` / ``regression_vs_best`` / ``stratified``.
    ``judge_only`` flags regressions whose best-so-far credit came from the
    judge rather than EM so the Diagnoser can weight them accordingly."""

    result: QuestionResult
    source: str
    judge_only: bool = False


def _pick_numeric_alternative(
    dim,
    current: int | float,
    rng: random.Random,
    *,
    int_type: bool,
) -> int | float | None:
    """Pick a value from a NumericRange or DiscreteValues that differs from ``current``."""
    from agentic_autorag.config.models import DiscreteValues

    if isinstance(dim, DiscreteValues):
        options = [v for v in dim.values if v != current]
        return rng.choice(options) if options else None

    lo, hi = dim.min, dim.max
    if int_type:
        lo_i, hi_i = int(lo), int(hi)
        if lo_i >= hi_i:
            return None
        for _ in range(10):
            v = rng.randint(lo_i, hi_i)
            if v != int(current):
                return v
        return None
    if lo >= hi:
        return None
    for _ in range(10):
        v = rng.uniform(lo, hi)
        if abs(v - float(current)) > 1e-6:
            return round(v, 4)
    return None


def _pick_alternative_value(
    lever: str,
    current_config: TrialConfig,
    project_config: ProjectConfig,
    rng: random.Random,
) -> object | None:
    """Pick a single alternative value for ``lever`` from the search space."""
    ss = project_config.search_space
    current = getattr(current_config, lever)

    if lever == "chunking_strategy":
        options = [s for s in ss.chunking.strategies if s != current]
        return rng.choice(options) if options else None
    if lever == "embedding_model":
        options = [m for m in ss.embedding.models if m != current]
        return rng.choice(options) if options else None
    if lever == "generator_llm":
        options = [m for m in ss.generator.models if m != current]
        return rng.choice(options) if options else None
    if lever == "reranker":
        options = [m for m in ss.reranker.models if m != current]
        return rng.choice(options) if options else None
    if lever == "top_k":
        return _pick_numeric_alternative(ss.retrieval.top_k, current, rng, int_type=True)
    if lever == "chunk_token_size":
        return _pick_numeric_alternative(ss.chunking.chunk_token_size, current, rng, int_type=True)
    if lever == "chunk_token_overlap":
        return _pick_numeric_alternative(ss.chunking.chunk_token_overlap, current, rng, int_type=True)
    if lever == "temperature":
        return _pick_numeric_alternative(ss.temperature, current, rng, int_type=False)
    return None


def _best_score_trial(history_records: list) -> int | None:
    """Trial number of the highest-accuracy prior trial (ties broken by lower
    cost). The universal anchor for lever-effect deltas and ``meta.changes``
    diffs. Empty history → ``None``."""
    if not history_records:
        return None
    leader = max(
        history_records,
        key=lambda r: (
            float(getattr(r, "answer_accuracy", 0.0)),
            -float(getattr(r, "mean_llm_cost_per_query_usd", 0.0)),
        ),
    )
    return int(getattr(leader, "trial_number", 0)) or None


class ReasoningAgent:
    """Two-stage reasoning agent with structured Diagnosis → ProposalMeta
    hand-off. Pure functions in ``state.py`` pre-compute the trial metrics
    and state card so the LLM's job shrinks to interpretation and selection."""

    def __init__(
        self,
        agent_model: str,
        config: ProjectConfig,
        history: HistoryLog,
        knowledge_base: KnowledgeBase | None = None,
        seed: int | None = None,
        use_diagnosis: bool = True,
        compact_history: bool = False,
    ) -> None:
        self.model = agent_model
        self.config = config
        self.history = history
        self.knowledge_base = knowledge_base
        # ``use_diagnosis=False`` is the diagnosis-off ablation: skip the
        # per-question failure-diagnosis LLM step and propose from the state
        # card alone. Isolates the contribution of the diagnosis stage.
        self.use_diagnosis = use_diagnosis
        # ``compact_history=True`` is the OPRO baseline: the proposer sees only a
        # one-line ``config -> accuracy`` trajectory instead of the rich
        # per-trial blocks, and the diagnosis/key-evidence prompt slots are
        # blanked. Combined with ``knowledge_base=None`` and
        # ``use_diagnosis=False`` it reduces the agent to a naive score-history
        # LLM proposer.
        self.compact_history = compact_history
        # Forwarded to litellm as ``seed=`` on every proposer call. Providers
        # that don't accept ``seed`` drop it via ``litellm.drop_params=True``.
        self.seed = seed
        self._include_graph = config.uses_graph()
        self._tunable_levers = config.search_space.tunable_levers()
        self._reasoning_effort = self._resolve_reasoning_effort(agent_model, config.agent.optimizer_reasoning_effort)
        if self._reasoning_effort is not None:
            logger.info("Reasoning agent using reasoning_effort=%s on %s", self._reasoning_effort, agent_model)

    @staticmethod
    def _resolve_reasoning_effort(model: str, effort: str | None) -> str | None:
        if not effort:
            return None
        try:
            supported = bool(litellm.supports_reasoning(model=model))
        except Exception:
            supported = True
        return effort if supported else None

    def _log_exchange(self, stage: str, prompt: str, response: str) -> None:
        """Mirror the full agent prompt/response to ``run.log`` for transparency.

        Logs at DEBUG, so it reaches the run.log file handler (DEBUG) but not
        the console handler (INFO) — the terminal stays readable while the log
        keeps a complete record of what the optimizer asked and got back.
        """
        sep = "═" * 64
        logging.getLogger("agentic_autorag.run").debug(
            "\n%s\n  PROMPT → %s\n%s\n%s\n\n%s\n  RESPONSE ← %s\n%s\n%s\n%s",
            sep,
            stage,
            sep,
            prompt,
            sep,
            stage,
            sep,
            response,
            sep,
        )

    async def propose_initial(self, corpus_description: str) -> TrialConfig:
        """Propose the first configuration based on corpus description."""
        if self.compact_history:
            prompt = OPRO_INITIAL_PROPOSAL_PROMPT.format(
                corpus_description=corpus_description,
                search_space=self.config.to_agent_prompt(),
            )
        else:
            prompt = INITIAL_PROPOSAL_PROMPT.format(
                corpus_description=corpus_description,
                search_space=self.config.to_agent_prompt(),
                knowledge_base=self._kb_text(),
                graph_guidance=_GRAPH_GUIDANCE if self._include_graph else "",
                **_initial_proposal_template_sections(self.config.meta.cost_aware),
            )
        return await self._call_for_config_only(prompt, stage="Initial Proposer")

    async def propose_after_failure(
        self,
        *,
        failed_config: TrialConfig,
        error_summary: str,
        failure_history: list[tuple[TrialConfig, str]],
    ) -> tuple[TrialConfig, ProposalMeta]:
        """Pick a recovery config after a trial failed before producing a
        result. ``failure_history`` is every prior (config, error) pair so
        the agent can avoid re-proposing them."""
        if self.compact_history:
            history_text = _compact_history(self.history.records, self._tunable_levers)
        else:
            history_text = self.history.format_for_agent(tunable=self._tunable_levers)
        prompt = FAILURE_RECOVERY_PROMPT.format(
            failed_config=failed_config.to_prompt_kv(include_graph=self._include_graph),
            error_summary=error_summary,
            failure_history=_format_failure_history(failure_history),
            history=history_text,
            search_space=self.config.to_agent_prompt(),
            knowledge_base=self._kb_text(),
            graph_rules=_GRAPH_RULES if self._include_graph else "",
        )

        messages = [{"role": "user", "content": prompt}]
        last_raw = ""
        for attempt in range(MAX_RETRIES):
            try:
                raw = await self._llm_complete_messages(messages)
                last_raw = raw
                self._log_exchange("Failure Recovery", messages[-1]["content"], raw)
                yaml_dict = self._extract_yaml(raw)
                meta_dict = yaml_dict.pop("meta", None) or {}
                self._inject_pinned(yaml_dict)
                config = TrialConfig.model_validate(yaml_dict)
                violations = self.config.validate_trial(config)
                if violations:
                    raise ValueError("Search space violations:\n" + "\n".join(f"- {v}" for v in violations))
                meta = ProposalMeta.model_validate(meta_dict) if isinstance(meta_dict, dict) else ProposalMeta()
                return config, meta
            except Exception as e:
                logger.warning("Failure-recovery attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, e)
                if attempt < MAX_RETRIES - 1:
                    messages.append({"role": "assistant", "content": last_raw})
                    messages.append({"role": "user", "content": _config_retry_message(e)})

        raise RuntimeError(f"Failure-recovery proposal failed after {MAX_RETRIES} attempts")

    async def analyze_and_propose(
        self,
        exam_result: ExamResult,
        exam_questions: list[OpenEndedQuestion],
        current_config: TrialConfig,
        trial_number: int,
        trials_remaining: int,
        previous_strategy: Strategy | None = None,
    ) -> tuple[TrialMetrics, Diagnosis, TrialConfig, ProposalMeta]:
        """Diagnose the current trial, then propose the next config. Returns
        ``(trial_metrics, diagnosis, next_config, proposal_meta)``.
        ``previous_strategy`` is the agent-owned campaign plan that was
        active during ``trial_number``; the orchestrator only round-trips it."""
        trial_metrics = compute_trial_metrics(exam_result)

        if self.use_diagnosis:
            diagnosis = await self._diagnose(
                exam_result=exam_result,
                exam_questions=exam_questions,
                current_config=current_config,
                trial_metrics=trial_metrics,
                trial_number=trial_number,
                trials_remaining=trials_remaining,
                previous_strategy=previous_strategy,
            )
        else:
            # Diagnosis-off ablation: hand the Proposer an empty diagnosis
            # (metrics only, no narrative/findings/illustrative qids). The
            # Proposer prompt renders the empty diagnosis block cleanly.
            diagnosis = Diagnosis(trial_metrics=trial_metrics)

        mechanical_attribution = build_failure_attribution(exam_result.question_results)
        top_modes = _top_stages_from_attribution(mechanical_attribution, n=2)
        state_card = build_state_card(
            trial_number=trial_number,
            trials_remaining=trials_remaining,
            current_accuracy=exam_result.answer_accuracy,
            history_records=self.history.records,
            current_config=current_config,
            current_top_failure_modes=top_modes,
            current_cost_usd=exam_result.mean_llm_cost_per_query_usd,
            current_retrieval_complete=trial_metrics.retrieval_complete,
            current_acc_given_complete=trial_metrics.answer_correct_given_complete_retrieval,
            current_in_tok=exam_result.mean_prompt_tokens,
            current_out_tok=exam_result.mean_completion_tokens,
            cost_aware=self.config.meta.cost_aware,
            previous_strategy=previous_strategy,
            hv_delta_window=self.config.meta.hv_delta_window,
            search_space_sizes=self._search_space_sizes(),
        )

        # Synthetic TrialRecord for the just-completed trial — not yet in
        # ``self.history.records`` (the orchestrator persists it after this
        # function returns). Passed to the Proposer's history dump so the
        # current trial's full block sits alongside the prior trials.
        current_trial_preview = TrialRecord(
            trial_number=trial_number,
            config=current_config,
            question_results=exam_result.question_results,
            answer_accuracy=exam_result.answer_accuracy,
            mean_retrieval_quality=exam_result.mean_retrieval_quality,
            n_em_correct=exam_result.n_em_correct,
            n_judge_correct=exam_result.n_judge_correct,
            n_judge_rejected=exam_result.n_judge_rejected,
            n_judge_no_answer=exam_result.n_judge_no_answer,
            n_judge_failed=exam_result.n_judge_failed,
            n_no_answer=exam_result.n_no_answer,
            n_judge_calls=exam_result.n_judge_calls,
            mean_em=exam_result.mean_em,
            mean_f1=exam_result.mean_f1,
            mean_llm_cost_per_query_usd=exam_result.mean_llm_cost_per_query_usd,
            total_llm_cost_usd=exam_result.total_llm_cost_usd,
            mean_prompt_tokens=exam_result.mean_prompt_tokens,
            mean_completion_tokens=exam_result.mean_completion_tokens,
            trial_metrics=trial_metrics,
            diagnosis=diagnosis,
        )

        next_config, meta = await self._propose(
            diagnosis=diagnosis,
            exam_questions=exam_questions,
            question_results=exam_result.question_results,
            current_config=current_config,
            state_card=state_card,
            current_trial=current_trial_preview,
            previous_strategy=previous_strategy,
            intended_trial=trial_number + 1,
        )
        return trial_metrics, diagnosis, next_config, meta

    async def _diagnose(
        self,
        *,
        exam_result: ExamResult,
        exam_questions: list[OpenEndedQuestion],
        current_config: TrialConfig,
        trial_metrics: TrialMetrics,
        trial_number: int,
        trials_remaining: int,
        previous_strategy: Strategy | None,
    ) -> Diagnosis:
        """Produce a structured ``Diagnosis``. The Diagnoser stays in
        evidence-extraction mode: narrative + grounded findings + notable
        deltas + illustrative qids; it does not prescribe levers and does not
        restate the mechanical attribution rendered into its prompt."""
        valid_results = [qr for qr in exam_result.question_results if qr.generated_response not in ERROR_SENTINELS]
        real_failures = [qr for qr in valid_results if not qr.correct]
        n_errors = sum(
            1 for q in exam_result.question_results if not q.correct and q.generated_response in ERROR_SENTINELS
        )

        error_note = ""
        if n_errors:
            error_note = (
                f"\n\nNote: {n_errors} question(s) failed due to system errors"
                " (timeouts, API failures) and are excluded from this analysis."
            )

        question_by_id = {q.id: q for q in exam_questions}
        prev_results_by_id = self._prev_trial_correctness()
        best_so_far_by_id = self._best_so_far_correctness()
        sample_seed = self._failure_sample_seed(trial_number)
        sample = self._select_stratified_failures(
            real_failures,
            question_by_id,
            prev_results_by_id,
            best_so_far_by_id,
            n=_DEEP_FAILURE_SAMPLE,
            seed=sample_seed,
        )

        def _render_with_source(sf: SelectedFailure) -> str:
            block = self._render_failure_block(sf.result, question_by_id.get(sf.result.question_id))
            tag = sf.source + (" (judge_only)" if sf.judge_only else "")
            return f"  source: {tag}\n{block}"

        deep_blocks = "\n\n".join(_render_with_source(sf) for sf in sample)
        failed_questions = (deep_blocks or "(no failures this trial)") + error_note

        # Top-confidence success cases so the diagnoser has a calibration
        # anchor for what "the pipeline working as designed" looks like.
        # Sample top-N by chunk_precision among complete-retrieval correct
        # answers — gives the cleanest signal.
        success_sample = self._select_top_successes(valid_results, n=_DEEP_SUCCESS_SAMPLE)
        success_blocks = "\n\n".join(
            self._render_failure_block(qr, question_by_id.get(qr.question_id), mode="retrieval_complete")
            for qr in success_sample
        )
        successes_text = success_blocks or "(no clean successes this trial)"

        failure_crosstab = build_failure_cross_tab(valid_results, exam_questions)
        failure_list = self._render_failure_list(real_failures, question_by_id)
        mechanical_attribution = build_failure_attribution(valid_results)

        # Single anchor = best-score trial. Renders deltas vs the run's
        # best-scoring config (regardless of cost-aware mode). The Pareto
        # frontier is rendered separately in the Proposer's state card.
        anchor_trial = _best_score_trial(self.history.records)
        single_effect = compute_bundle_effect(
            history_records=self.history.records,
            current_config=current_config,
            current_metrics=trial_metrics,
            current_cost_usd=exam_result.mean_llm_cost_per_query_usd,
            anchor_trial=anchor_trial,
        )
        bundle_effects = [(f"best-score trial {anchor_trial}", single_effect)] if single_effect is not None else []
        anchor_summary = f"best-score trial {anchor_trial}" if anchor_trial is not None else "n/a (first trial)"

        config_kv = current_config.to_prompt_kv(include_graph=self._include_graph)
        graph_diag = _GRAPH_DIAGNOSTIC_TYPES if self._include_graph else ""
        prior_trial_signal = self.history.format_for_diagnoser()
        diagnostic_state = (
            f"trial_number={trial_number} trials_remaining={trials_remaining}"
            f" best_accuracy_so_far={self._best_score():.3f}"
        )
        lever_effect_text = _format_bundle_effects(bundle_effects, fallback_label=anchor_summary)
        prompt = DIAGNOSTIC_PROMPT.format(
            trial_metrics=_format_trial_metrics(trial_metrics, show_cost=False),
            state_card=diagnostic_state,
            current_config=config_kv,
            prior_trial_signal=prior_trial_signal,
            failure_crosstab=failure_crosstab,
            failure_list=failure_list,
            mechanical_failure_attribution=_format_failure_attribution(mechanical_attribution),
            lever_effect_deltas=lever_effect_text,
            success_blocks=successes_text,
            failed_questions=failed_questions,
            graph_diagnostic_types=graph_diag,
        )

        exam_qids = {q.id for q in exam_questions}

        messages = [{"role": "user", "content": prompt}]
        raw = ""
        diagnosis: Diagnosis | None = None
        for attempt in range(MAX_RETRIES):
            try:
                raw = await self._llm_complete_messages(messages)
                diagnosis = self._build_diagnosis(
                    raw=raw,
                    trial_metrics=trial_metrics,
                    exam_qids=exam_qids,
                )
                break
            except Exception as e:
                logger.warning("Diagnoser attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, e)
                if attempt == MAX_RETRIES - 1:
                    break
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your response had an error: {e}\n\n"
                            "Please fix the issue and output a corrected ```yaml block with"
                            " a `narrative` string and the lists `confirmed_findings`,"
                            " `notable_deltas`, and `illustrative_qids`."
                        ),
                    }
                )

        if diagnosis is None:
            logger.error("Diagnoser returned unparseable output after %d attempts; falling back", MAX_RETRIES)
            diagnosis = Diagnosis(
                trial_metrics=trial_metrics,
                narrative=_extract_narrative(raw)[:300],
            )

        self._log_exchange("Diagnoser", prompt, raw)
        return diagnosis

    async def _propose(
        self,
        *,
        diagnosis: Diagnosis,
        exam_questions: list[OpenEndedQuestion],
        question_results: list[QuestionResult],
        current_config: TrialConfig,
        state_card: StateCard,
        previous_strategy: Strategy | None,
        intended_trial: int,
        current_trial: TrialRecord | None = None,
    ) -> tuple[TrialConfig, ProposalMeta]:
        """Produce the next (TrialConfig, ProposalMeta). Requires only that a
        campaign-plan Strategy is emitted — no ratchet, no lock-in, no done
        gate; the agent owns its phase and budget."""
        if self.compact_history:
            prompt = OPRO_PROPOSAL_PROMPT.format(
                search_space=self.config.to_agent_prompt(),
                progress=_format_opro_progress(state_card),
                current_config=current_config.to_prompt_kv(include_graph=self._include_graph),
                history=_compact_history(self.history.records, self._tunable_levers, current_trial=current_trial),
            )
        else:
            prompt = PROPOSAL_PROMPT.format(
                diagnosis=_format_diagnosis(diagnosis, show_cost=self.config.meta.cost_aware),
                state_card=_format_state_card(state_card),
                current_config=current_config.to_prompt_kv(include_graph=self._include_graph),
                history=self.history.format_for_agent(
                    tunable=self._tunable_levers,
                    current_trial=current_trial,
                    show_cost=self.config.meta.cost_aware,
                ),
                key_evidence=self._format_key_evidence(diagnosis, exam_questions, question_results),
                search_space=self.config.to_agent_prompt(),
                knowledge_base=self._kb_text(),
                graph_rules=_GRAPH_RULES if self._include_graph else "",
                **_proposal_template_sections(self.config.meta.cost_aware),
            )

        messages = [{"role": "user", "content": prompt}]
        last_raw = ""
        parse_failures = 0
        duplicate_retries = 0
        while parse_failures < MAX_RETRIES:
            try:
                raw = await self._llm_complete_messages(messages)
                last_raw = raw
                self._log_exchange("Proposer", messages[-1]["content"], raw)
                yaml_dict = self._extract_yaml(raw)
                meta_dict = yaml_dict.pop("meta", None)
                if not isinstance(meta_dict, dict):
                    raise ValueError("proposal YAML must include a 'meta' dict")
                self._inject_pinned(yaml_dict)
                config = TrialConfig.model_validate(yaml_dict)

                violations = self.config.validate_trial(config)
                if violations:
                    raise ValueError("Search space violations:\n" + "\n".join(f"- {v}" for v in violations))

                meta = ProposalMeta.model_validate(meta_dict)
                # The OPRO baseline (compact_history) must not maintain a
                # cross-trial campaign plan — that is agentic working memory.
                # Don't require a strategy block from it; any it emits is never
                # fed back (the state card's carry-over is suppressed).
                if not self.compact_history and meta.strategy is None:
                    raise ValueError(
                        "proposal `meta.strategy` is required. Emit a `strategy:` block with"
                        " `phase`, `plan`, and `notes` — your campaign plan,"
                        " re-authored this trial."
                    )
            except Exception as e:
                parse_failures += 1
                logger.warning("Proposer parse attempt %d/%d failed: %s", parse_failures, MAX_RETRIES, e)
                if parse_failures >= MAX_RETRIES:
                    break
                messages.append({"role": "assistant", "content": last_raw})
                messages.append({"role": "user", "content": _config_retry_message(e)})
                continue

            dup_trial = self._find_duplicate_in_history(config)
            if dup_trial is None:
                return config, meta
            if duplicate_retries >= MAX_DUPLICATE_RETRIES:
                logger.warning(
                    "Proposer kept proposing duplicates (last = trial %d) after %d retries; "
                    "perturbing to a non-duplicate config",
                    dup_trial,
                    duplicate_retries,
                )
                return self._proposer_perturbation_fallback(
                    current_config=current_config,
                    previous_strategy=previous_strategy,
                    intended_trial=intended_trial,
                    reason=f"duplicate of trial {dup_trial}",
                )
            duplicate_retries += 1
            logger.warning(
                "Proposer emitted a duplicate of trial %d; retry %d/%d",
                dup_trial,
                duplicate_retries,
                MAX_DUPLICATE_RETRIES,
            )
            messages.append({"role": "assistant", "content": last_raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The config you emitted is identical to trial {dup_trial}:\n"
                        f"  {_config_signature(config, self._tunable_levers)}\n"
                        "Every lever matches a config in the 'Configs already tried' list. "
                        "Change at least one lever to a value that does not appear there — "
                        "prefer levers your diagnosis implicates — and re-emit the YAML block."
                    ),
                }
            )

        return self._proposer_perturbation_fallback(
            current_config=current_config,
            previous_strategy=previous_strategy,
            intended_trial=intended_trial,
            reason=f"parse failed {MAX_RETRIES}x",
        )

    def _find_duplicate_in_history(self, config: TrialConfig) -> int | None:
        """Return the trial_number of an earlier trial with an identical config, or None."""
        for record in self.history.records:
            if record.config == config:
                return record.trial_number
        return None

    def _proposer_perturbation_fallback(
        self,
        *,
        current_config: TrialConfig,
        previous_strategy: Strategy | None,
        intended_trial: int,
        reason: str,
    ) -> tuple[TrialConfig, ProposalMeta]:
        """Minimal single-lever perturbation when the Proposer cannot deliver a
        usable config — either it never emitted valid YAML or it kept emitting
        duplicates. Picks one lever from a curated safe list, validates against
        ``project_config``, and rejects duplicates of any prior trial, so this
        never re-runs an identical trial. RNG seeded from ``intended_trial`` +
        ``failure_sample_seed`` for reproducibility from run.log alone.
        ``reason`` is surfaced in the log and the fallback rationale."""
        seed_source = self.config.meta.failure_sample_seed or 0
        rng = random.Random(seed_source ^ intended_trial)
        levers = list(_PROPOSER_FALLBACK_SAFE_LEVERS)
        rng.shuffle(levers)

        chosen_config: TrialConfig | None = None
        chosen_lever: str | None = None
        old_value: object | None = None
        new_value: object | None = None

        for _ in range(_PROPOSER_FALLBACK_PERTURBATION_ATTEMPTS):
            if not levers:
                break
            lever = levers.pop()
            new_val = _pick_alternative_value(lever, current_config, self.config, rng)
            if new_val is None:
                continue
            try:
                candidate = current_config.model_copy(update={lever: new_val})
            except Exception:  # noqa: BLE001
                continue
            if self.config.validate_trial(candidate):
                continue
            if self._find_duplicate_in_history(candidate) is not None:
                continue
            chosen_config = candidate
            chosen_lever = lever
            old_value = getattr(current_config, lever)
            new_value = new_val
            break

        if chosen_config is None:
            logger.warning(
                "Proposer fallback: no valid non-duplicate single-lever perturbation found; reusing current config"
            )
            chosen_config = current_config
            change_note = "no perturbation found"
        else:
            change_note = f"{chosen_lever}: {old_value} -> {new_value}"

        logger.warning(
            "Proposer fallback (%s); minimal single-lever perturbation: %s",
            reason,
            change_note,
        )

        if previous_strategy is not None:
            fallback_strategy = previous_strategy.model_copy()
        else:
            fallback_strategy = Strategy(phase=INITIAL_PHASE)
        meta = ProposalMeta(
            rationale=(f"Proposer fallback ({reason}); minimal perturbation to keep the run alive ({change_note})."),
            strategy=fallback_strategy,
        )
        return chosen_config, meta

    def _build_diagnosis(
        self,
        *,
        raw: str,
        trial_metrics: TrialMetrics,
        exam_qids: set[str] | None = None,
    ) -> Diagnosis:
        """Parse the diagnoser's YAML and validate.

        Raises ``ValueError`` so the retry loop in ``_diagnose`` can re-prompt
        the agent. Validation: ``illustrative_qids`` must be a subset of this
        trial's exam.
        """
        yaml_dict = self._extract_yaml(raw)
        narrative = yaml_dict.get("narrative") or _extract_narrative(raw)

        confirmed = _coerce_str_list(yaml_dict.get("confirmed_findings"))
        notable = _coerce_str_list(yaml_dict.get("notable_deltas"))
        qids = _coerce_str_list(yaml_dict.get("illustrative_qids"))

        if exam_qids is not None:
            bad_qids = [q for q in qids if q not in exam_qids]
            if bad_qids:
                raise ValueError(
                    f"illustrative_qids contains qids not in this trial's exam: {bad_qids}. "
                    "Use only question_ids from the failed-question blocks above."
                )

        return Diagnosis(
            trial_metrics=trial_metrics,
            narrative=narrative,
            confirmed_findings=confirmed[:5],
            notable_deltas=notable[:5],
            illustrative_qids=qids[:_KEY_EVIDENCE_SAMPLE],
        )

    async def _call_for_config_only(self, prompt: str, *, stage: str) -> TrialConfig:
        """Call LLM, extract a TrialConfig-shaped YAML, validate, retry on failure."""
        messages = [{"role": "user", "content": prompt}]
        raw = ""
        for attempt in range(MAX_RETRIES):
            try:
                raw = await self._llm_complete_messages(messages)
                self._log_exchange(stage, messages[-1]["content"], raw)
                yaml_dict = self._extract_yaml(raw)
                yaml_dict.pop("meta", None)
                self._inject_pinned(yaml_dict)
                config = TrialConfig.model_validate(yaml_dict)

                violations = self.config.validate_trial(config)
                if violations:
                    raise ValueError("Search space violations:\n" + "\n".join(f"- {v}" for v in violations))

                return config

            except Exception as e:
                logger.warning("%s attempt %d/%d failed: %s", stage, attempt + 1, MAX_RETRIES, e)
                if attempt < MAX_RETRIES - 1:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": _config_retry_message(e)})

        raise RuntimeError(f"Failed to get valid config after {MAX_RETRIES} attempts")

    def _prev_trial_correctness(self) -> dict[str, bool]:
        """``question_id → correct`` from the most recent prior trial. Empty
        when there is no prior trial. Feeds the stratified sampler's
        "flipped since last trial" tier."""
        if not self.history.records:
            return {}
        prev = self.history.records[-1]
        return {qr.question_id: bool(qr.correct) for qr in prev.question_results}

    def _best_so_far_correctness(self) -> dict[str, tuple[bool, bool]]:
        """``question_id → (was_correct, judge_only)`` from the best-so-far
        trial. Best-so-far is the prior trial with the highest ``answer_accuracy``
        (ties broken by trial_number, first wins). ``judge_only`` flags
        credit awarded by the judge with ``em == 0``."""
        if not self.history.records:
            return {}
        best = max(self.history.records, key=lambda r: (r.answer_accuracy, -r.trial_number))
        out: dict[str, tuple[bool, bool]] = {}
        for qr in best.question_results:
            was_correct = bool(qr.correct)
            judge_only = was_correct and qr.em == 0.0
            out[qr.question_id] = (was_correct, judge_only)
        return out

    def _failure_sample_seed(self, trial_number: int) -> int:
        """Seed for the stratified failure sampler. Honours
        ``MetaConfig.failure_sample_seed``; otherwise derives from the trial
        number — deterministic per trial, varying across trials."""
        configured = self.config.meta.failure_sample_seed
        return int(configured) if configured is not None else int(trial_number)

    def _format_key_evidence(
        self,
        diagnosis: Diagnosis,
        exam_questions: list[OpenEndedQuestion],
        question_results: list[QuestionResult],
    ) -> str:
        """Render the Diagnoser-selected ``illustrative_qids`` as raw blocks.
        The format matches what the Diagnoser saw so the Proposer can
        verify claims against ground truth."""
        qids = diagnosis.illustrative_qids[:_KEY_EVIDENCE_SAMPLE]
        if not qids:
            return "(diagnosis emitted no illustrative_qids)"
        results_by_id = {qr.question_id: qr for qr in question_results}
        questions_by_id = {q.id: q for q in exam_questions}
        blocks: list[str] = []
        for qid in qids:
            qr = results_by_id.get(qid)
            if qr is None:
                continue
            blocks.append(self._render_failure_block(qr, questions_by_id.get(qid)))
        return "\n\n".join(blocks) if blocks else "(no matching question results for illustrative_qids)"

    async def _llm_complete_messages(self, messages: list[dict]) -> str:
        kwargs: dict = {"model": self.model, "messages": messages}
        if self._reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._reasoning_effort
        if self.seed is not None:
            kwargs["seed"] = self.seed
        response, _ = await acompletion_with_cost(cost_category="agent_proposal", **kwargs)
        return response.choices[0].message.content or ""

    def _best_score(self) -> float:
        if not self.history.records:
            return 0.0
        return max(float(r.answer_accuracy) for r in self.history.records)

    def _search_space_sizes(self) -> dict[str, int]:
        """Pool sizes for the three component-pool levers shown in the state
        card's coverage line. Numeric ranges and boolean strategies aren't
        useful as coverage signals."""
        ss = self.config.search_space
        return {
            "generator_llm": len(ss.generator.models),
            "embedding_model": len(ss.embedding.models),
            "reranker": len(ss.reranker.models),
        }

    def _kb_text(self) -> str:
        if self.knowledge_base is None:
            return ""
        ss = self.config.search_space
        # Only generator-stage LLMs are eligible to toggle reasoning. Force
        # ``False`` for non-generator stages regardless of litellm catalog
        # claims so the proposer isn't misled.
        all_llms = ss.all_llm_models()
        generator_set = set(ss.generator.models)
        reasoning_allowed = {m: ss.is_reasoning_allowed(m) if m in generator_set else False for m in all_llms}
        # Skip parameter-guide entries for every pinned lever (already in the
        # "Fixed values" block) and the derived stage LLMs (compressor_llm /
        # expander_llm are resolved at injection time from the strategy
        # choice — a guide entry would contradict the "Derived values" block).
        skip_params = set(ss.pinned_field_values().keys())
        if ss.compressor_llm_is_derived():
            skip_params.add("compressor_llm")
        if ss.expander_llm_is_derived():
            skip_params.add("expander_llm")
        option_filter: dict[str, set[str]] = {
            "chunking_strategy": set(ss.chunking.strategies),
            "index_type": {t.value for t in ss.retrieval.index_types},
            "bm25_vector_fusion": set(ss.retrieval.bm25_vector_fusion),
            "passage_compressor": set(ss.passage_compressor.strategies),
            "query_expansion": set(ss.query_expansion.strategies),
        }
        if ss.graph_retrieval is not None:
            option_filter["graph_query_mode"] = set(ss.graph_retrieval.graph_query_modes)
        return self.knowledge_base.format_for_prompt(
            llm_models=all_llms,
            embedding_models=ss.embedding.models,
            reranker_models=ss.reranker.models,
            reasoning_allowed=reasoning_allowed,
            reasoning_enabled=ss.generator.reasoning,
            reasoning_effort=ss.generator.reasoning_effort,
            cost_aware=self.config.meta.cost_aware,
            include_graph=self._include_graph,
            skip_params=skip_params,
            option_filter=option_filter,
        )

    @staticmethod
    def _extract_yaml(text: str) -> dict:
        match = re.search(r"```ya?ml\n(.*?)```", text, re.DOTALL)
        if not match:
            match = re.search(r"```\n(.*?)```", text, re.DOTALL)
        if not match:
            raise ValueError("No YAML block found in agent response")
        raw_yaml = match.group(1)
        try:
            parsed = yaml.safe_load(raw_yaml)
        except yaml.YAMLError:
            repaired = re.sub(
                r"^(\s*)([A-Za-z_][A-Za-z0-9_-]*):(?=\S)",
                r"\1\2: ",
                raw_yaml,
                flags=re.MULTILINE,
            )
            parsed = yaml.safe_load(repaired)
        if not isinstance(parsed, dict):
            raise ValueError("YAML block must be a mapping")
        return parsed

    def _inject_pinned(self, yaml_dict: dict) -> None:
        """Overwrite pinned-field entries with their single legal value, then
        resolve any *derived* stage-LLM fields from the proposer's strategy.

        The proposer prompt instructs the agent to omit pinned fields; this
        injection is the defensive backstop that makes the parse succeed even
        if the agent ignores those instructions. Pinned wins unconditionally.

        ``compressor_llm`` and ``expander_llm`` may also be *derived*: when
        their stage's strategy list mixes ``"none"`` with non-``"none"`` and
        their LLM pool is size 1, the validator requires ``None`` when the
        proposer chose ``"none"`` and the pool's lone model otherwise. We
        resolve this after the static-pin pass so the strategy field is
        already in its final form.

        When the agent's emitted value *differs* from the pinned (or derived)
        value we log a warning — that's a search-space violation that the new
        prompt structure was supposed to prevent.
        """
        ss = self.config.search_space
        pinned = ss.pinned_field_values()
        for field, value in pinned.items():
            emitted = yaml_dict.get(field)
            if emitted is not None and emitted != value:
                logger.warning(
                    "Proposer emitted %s=%r for a pinned field; overriding with %r. "
                    "Prompt instructed the agent to omit pinned fields — this is a "
                    "search-space violation that injection caught.",
                    field,
                    emitted,
                    value,
                )
            yaml_dict[field] = value

        self._resolve_stage_llms(yaml_dict)

    def _resolve_stage_llms(self, yaml_dict: dict) -> None:
        """Derive ``compressor_llm`` / ``expander_llm`` from the resolved
        strategy choice when their pool is size 1 and the stage may be off.

        Runs after ``_inject_pinned`` has applied the static pins, so the
        ``passage_compressor`` / ``query_expansion`` fields are in their final
        form (whether emitted by the proposer or injected). When the pool is
        multi-LLM the proposer is responsible for picking — we still force
        ``None`` when the chosen strategy is explicitly ``"none"``, but
        otherwise leave the agent's emitted value alone.

        When the strategy field is absent from ``yaml_dict``, do nothing: a
        missing strategy means the proposer didn't emit it (tunable case) or
        the proposer pipeline hasn't filled it in yet. The trial validator
        will surface the missing field; the resolver shouldn't paper over it.
        """
        ss = self.config.search_space

        if ss.compressor_llm_is_derived():
            strategy = yaml_dict.get("passage_compressor", "none")
            yaml_dict["compressor_llm"] = None if strategy == "none" else ss.passage_compressor.models[0]
        elif (
            not ss.compressor_llm_is_dead()
            and len(ss.passage_compressor.models) > 1
            and "passage_compressor" in yaml_dict
            and yaml_dict["passage_compressor"] == "none"
        ):
            yaml_dict["compressor_llm"] = None

        if ss.expander_llm_is_derived():
            strategy = yaml_dict.get("query_expansion", "none")
            yaml_dict["expander_llm"] = None if strategy == "none" else ss.query_expansion.models[0]
        elif (
            not ss.expander_llm_is_dead()
            and len(ss.query_expansion.models) > 1
            and "query_expansion" in yaml_dict
            and yaml_dict["query_expansion"] == "none"
        ):
            yaml_dict["expander_llm"] = None

    @staticmethod
    def _render_failure_list(
        failures: list[QuestionResult],
        questions_by_id: dict[str, OpenEndedQuestion],
    ) -> str:
        """One line per failure (Tier 2): qid | mode | reasoning_type(n) | gold | pred."""
        if not failures:
            return "(no failures this trial)"
        lines: list[str] = []
        for qr in failures:
            mode = failure_mode(qr)
            q = questions_by_id.get(qr.question_id)
            rt = (q.reasoning_type if q is not None else None) or "unknown"
            gold = (q.canonical_answer if q is not None else qr.correct_answer) or ""
            pred = qr.selected_answer or ""
            lines.append(
                f"  {qr.question_id} | {mode:18s} | {rt:12s}(n={qr.n_spans}) | "
                f"gold={_truncate_for_list(gold, 40)!r} | pred={_truncate_for_list(pred, 40)!r}"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_failure_block(
        qr: QuestionResult,
        question: OpenEndedQuestion | None,
        *,
        mode: str | None = None,
    ) -> str:
        """Render a single failure with failure-mode-adaptive chunk windowing.

        Chunk-rendering policy (corpus-agnostic — no doc-title dependence):
          - ``retrieval_miss``: first ~60 tokens of the top-2 retrieved chunks.
          - ``retrieval_partial``: ~60-token window around each retrieved gold
            span, plus the first retrieved chunk's prefix as a distractor.
          - ``retrieval_complete + generation_wrong``: window around each gold
            span only; no distractors.
          - ``refused``: window around each gold span (if any) plus one
            distractor prefix.

        When ``qr.retrieved_chunks`` is empty (e.g. legacy records), falls back
        to splitting ``retrieved_context`` on ``\\n``.
        """
        mode = mode or failure_mode(qr)
        q = question
        question_text = q.question if q else "<question text unavailable>"
        gold_text = q.canonical_answer if q else qr.correct_answer
        spans: list[str] = list(q.source_spans) if q else []
        # Tier C carries gold docs in the span-aligned source_doc_ids; tier B
        # carries them doc-level in supporting_doc_ids. Fall back so doc-level
        # gt_coverage renders for both (spans is empty for tier B, so the
        # span-zip below is skipped).
        source_doc_ids: list[str] = []
        if q:
            source_doc_ids = list(q.source_doc_ids) if q.source_doc_ids else list(q.supporting_doc_ids)
        source_docs_text = ", ".join(source_doc_ids) if source_doc_ids else "<unknown>"
        retrieved_doc_ids: list[str] = list(qr.retrieved_doc_ids or [])
        n_retrieved = len(retrieved_doc_ids)

        gt_set = set(source_doc_ids)
        gt_hits = sum(1 for d in retrieved_doc_ids if d in gt_set) if gt_set else 0
        gt_coverage = f"{gt_hits}/{n_retrieved}" if n_retrieved else "0/0"

        header_lines = [
            f"### {mode}  q_id={qr.question_id}",
            f"  question: {question_text}",
            f"  gold: {gold_text!r}",
            f"  pred: {qr.selected_answer!r}",
            (
                f"  retrieval_status: {qr.retrieval_status} | refused: {qr.refused} | "
                f"chunk_precision={qr.chunk_precision:.2f} | source_span_rank={qr.source_fact_rank}"
            ),
            f"  source_docs: {source_docs_text} | gt_coverage: {gt_coverage}",
        ]
        if spans:
            header_lines.append("  source_spans:")
            for i, (span, doc_id) in enumerate(zip(spans, source_doc_ids, strict=True), start=1):
                header_lines.append(f"    span_{i} (doc={doc_id}): {span!r}")

        chunks = list(qr.retrieved_chunks) if qr.retrieved_chunks else _split_legacy_context(qr.retrieved_context)
        # Per-chunk × per-span evaluator ground truth. When present, this is
        # authoritative: the evaluator's matcher already decided which chunks
        # satisfy which spans (via char-range overlap, unicode-folded find, OR
        # n-gram coverage). For each (chunk, span) pair on the satisfies list
        # we render a window or — if the text doesn't contain the span
        # verbatim under unicode-fold — a chunk prefix tagged as an
        # approximate match. Without this table the renderer fell through to
        # text-only search and missed chunks that the evaluator credited via
        # n-gram coverage on non-source-doc chunks.
        satisfies = list(qr.chunk_satisfies_spans) if qr.chunk_satisfies_spans else []
        chunk_lines = _render_chunks_for_mode(
            mode=mode,
            chunks=chunks,
            retrieved_doc_ids=retrieved_doc_ids,
            gold_spans=spans,
            chunk_satisfies_spans=satisfies,
        )

        return "\n".join(header_lines + chunk_lines)

    @staticmethod
    def _select_top_successes(
        valid_results: list[QuestionResult],
        *,
        n: int = _DEEP_SUCCESS_SAMPLE,
    ) -> list[QuestionResult]:
        """Top-confidence calibration anchor for the diagnoser.

        Filters to ``correct AND context_sufficient`` (the pipeline did exactly
        what it was designed to do — retrieved every gold span and answered
        correctly), sorts by ``chunk_precision`` descending, returns the top N.
        Top-confidence beats random: a precision-1.0 retrieval is the cleanest
        possible "this is what success looks like" example.
        """
        candidates = [qr for qr in valid_results if qr.correct and qr.context_sufficient]
        if not candidates:
            # Ungrounded (tier-A) exams have no context-sufficiency signal, so
            # the strict filter empties out. Fall back to correct-only so the
            # diagnoser still gets a "what success looks like" anchor.
            candidates = [qr for qr in valid_results if qr.correct]
        candidates.sort(key=lambda qr: qr.chunk_precision, reverse=True)
        return candidates[:n]

    @staticmethod
    def _select_stratified_failures(
        failures: list[QuestionResult],
        questions_by_id: dict[str, OpenEndedQuestion],
        prev_results_by_id: dict[str, bool] | None,
        best_so_far_results_by_id: dict[str, tuple[bool, bool]] | None = None,
        *,
        n: int = _DEEP_FAILURE_SAMPLE,
        seed: int = 0,
    ) -> list[SelectedFailure]:
        """Stratified sample across ``(failure_mode, reasoning_type)`` cells.

        Three bands feed the sample in priority order:
          1. ``regression_vs_best`` — correct in best-so-far, wrong now. Up to
             ``_REGRESSION_VS_BEST_BAND_SIZE`` qids; tagged with ``judge_only``
             when best-so-far's credit came from the judge (not EM).
          2. ``flip_vs_prev`` — correct in the most recent trial, wrong now.
          3. ``stratified`` — round-robin across remaining
             (failure_mode, reasoning_type) cells.
        """
        if not failures:
            return []

        rng = random.Random(seed)

        def cell_key(qr: QuestionResult) -> tuple[str, str]:
            mode = failure_mode(qr)
            q = questions_by_id.get(qr.question_id)
            return mode, ((q.reasoning_type if q is not None else None) or "unknown")

        flipped: list[QuestionResult] = []
        regression: list[tuple[QuestionResult, bool]] = []
        steady: list[QuestionResult] = []
        best_map = best_so_far_results_by_id or {}
        for qr in failures:
            was_correct_in_best, judge_only = best_map.get(qr.question_id, (False, False))
            was_correct_prev = bool(prev_results_by_id and prev_results_by_id.get(qr.question_id, False))
            if was_correct_in_best and not was_correct_prev:
                regression.append((qr, judge_only))
            elif was_correct_prev:
                flipped.append(qr)
            else:
                steady.append(qr)

        cells: dict[tuple[str, str], list[QuestionResult]] = {}
        for qr in steady:
            cells.setdefault(cell_key(qr), []).append(qr)
        for bucket in cells.values():
            rng.shuffle(bucket)
        rng.shuffle(flipped)
        rng.shuffle(regression)
        cell_order = list(cells.keys())
        rng.shuffle(cell_order)

        picked: list[SelectedFailure] = []
        seen: set[str] = set()

        # Band 1: regression-vs-best, capped at _REGRESSION_VS_BEST_BAND_SIZE.
        for qr, judge_only in regression[:_REGRESSION_VS_BEST_BAND_SIZE]:
            if len(picked) >= n:
                break
            if qr.question_id in seen:
                continue
            picked.append(SelectedFailure(result=qr, source="regression_vs_best", judge_only=judge_only))
            seen.add(qr.question_id)

        # Band 2: flip-vs-prev (no explicit cap; uses remaining budget).
        for qr in flipped:
            if len(picked) >= n:
                break
            if qr.question_id in seen:
                continue
            picked.append(SelectedFailure(result=qr, source="flip_vs_prev"))
            seen.add(qr.question_id)

        # Band 3: round-robin across remaining cells.
        while len(picked) < n and any(cells[k] for k in cell_order):
            progressed = False
            for key in cell_order:
                if not cells[key]:
                    continue
                qr = cells[key].pop()
                if qr.question_id in seen:
                    continue
                picked.append(SelectedFailure(result=qr, source="stratified"))
                seen.add(qr.question_id)
                progressed = True
                if len(picked) >= n:
                    break
            if not progressed:
                break
        return picked


def _format_trial_metrics(tm: TrialMetrics, *, show_cost: bool = True) -> str:
    line = (
        f"answer_accuracy={tm.answer_accuracy:.3f}"
        f" | retrieval: complete={tm.retrieval_complete:.3f}"
        f" partial={tm.retrieval_partial:.3f}"
        f" miss={tm.retrieval_miss:.3f}"
        f" | refusal_rate={tm.refusal_rate:.3f}"
        f" | acc_given_complete={tm.answer_correct_given_complete_retrieval:.3f}"
        f" (n_valid={tm.n_valid})"
    )
    if show_cost:
        line += f" | cost_per_query=${tm.mean_llm_cost_per_query_usd:.4f}"
    return line


def _format_opro_progress(sc: StateCard) -> str:
    """Minimal budget line for the OPRO baseline: trial counter + best-so-far,
    with none of the per-trial summaries / plan / coverage the full state
    card carries (those are agentic working memory and diagnosis-adjacent
    signal the naive score-history proposer must not see)."""
    total_budget = sc.trial_number + sc.trials_remaining
    return (
        f"trial {sc.trial_number} of {total_budget}; "
        f"best accuracy so far {sc.best_accuracy_so_far:.3f} (trial {sc.best_trial_number}); "
        f"last trial delta {sc.last_trial_delta:+.3f}."
    )


def _format_state_card(sc: StateCard) -> str:
    total_budget = sc.trial_number + sc.trials_remaining
    lines = [
        f"trial_number={sc.trial_number} trials_remaining={sc.trials_remaining} (of {total_budget} total)",
        (
            f"best_accuracy_so_far={sc.best_accuracy_so_far:.3f}"
            f" (trial {sc.best_trial_number}; trials_since_best_accuracy={sc.trials_since_best_accuracy})"
        ),
        f"last_trial_delta={sc.last_trial_delta:+.3f}",
        (
            f"component bests so far (best OBSERVED, not the achievable max):"
            f" best_retrieval_complete={sc.best_retrieval_complete:.3f} (trial {sc.best_retrieval_complete_trial});"
            f" best_acc_given_complete={sc.best_acc_given_complete:.3f} (trial {sc.best_acc_given_complete_trial})"
        ),
    ]
    if sc.coverage:
        parts = [f"{c['label']} {c['tried']}/{c['total']}" for c in sc.coverage]
        lines.append("search space coverage: " + "; ".join(parts))
    if sc.trial_summaries:
        lines.append("trial_summaries:")
        for t in sc.trial_summaries[-8:]:
            changes = t.get("what_changed_from_prev") or []
            modes = t.get("top_failure_modes") or []
            change_str = "; ".join(changes) if changes else "<initial>"
            mode_str = ", ".join(modes) if modes else "<none>"
            cost_usd = float(t.get("cost_usd", 0.0))
            if sc.cost_aware:
                cost_str = (
                    f" cost=${cost_usd:.4f}/q"
                    f" in_tok={float(t.get('in_tok', 0.0)):.0f}/q"
                    f" out_tok={float(t.get('out_tok', 0.0)):.0f}/q"
                )
            else:
                cost_str = ""
            retrieval_complete = float(t.get("retrieval_complete", 0.0))
            acc_given_complete = float(t.get("acc_given_complete", 0.0))
            lines.append(
                f"  - trial {t.get('trial_number')}: accuracy={float(t.get('accuracy', 0.0)):.3f}"
                f" retrieval_complete={retrieval_complete:.2f}"
                f" acc_given_complete={acc_given_complete:.2f}"
                f"{cost_str}"
                f" | changes vs prior: {change_str} | top_failure_modes: {mode_str}"
            )

    lines.extend(_format_plan_block(sc))
    if sc.cost_aware:
        lines.extend(_format_pareto_block(sc))
    return "\n".join(lines)


def _format_plan_block(sc: StateCard) -> list[str]:
    """Render the agent's campaign-plan carry-over: its prior phase, plan, and
    notes, plus the run-length-encoded phase trajectory.

    This is the agent's own plan from last trial, shown back so it honors it or
    revises it on purpose rather than drifting. The trajectory (e.g.
    ``ceiling×6, frontier×3``) lets the agent see how its budget has actually
    been spent without scanning every trial block. Rendered in both modes —
    every run has a campaign plan; only the phase vocabulary differs.
    """
    lines: list[str] = ["", "## Your campaign plan (carried from last trial — honor it or revise it on purpose)"]
    prev = sc.previous_strategy
    if prev is None or not (prev.phase or prev.plan or prev.notes):
        lines.append("previous_plan: <none — this is the first plan of the run; author it now>")
        return lines

    if prev.phase:
        lines.append(f"previous_phase: {prev.phase}")
    if sc.phase_history:
        lines.append(f"phase trajectory (oldest → newest): {_rle_phase_history(sc.phase_history)}")
    if prev.plan:
        lines.append(f"  plan: {prev.plan}")
    if prev.notes:
        lines.append(f"  notes: {prev.notes}")
    return lines


def _rle_phase_history(history: list[tuple[int, str]]) -> str:
    """Run-length-encode the phase trajectory as ``phase×N`` chunks.

    Example: ``[(2,'ceiling'),(3,'ceiling'),(4,'frontier'),(5,'ceiling')]``
    renders as ``"ceiling×2 (trials 2-3), frontier×1 (trial 4), ceiling×1 (trial 5)"``.
    The trial-range suffix lets the agent map back to specific trial blocks
    without having to count.
    """
    if not history:
        return "<none>"
    chunks: list[str] = []
    run_phase = history[0][1]
    run_start = history[0][0]
    run_len = 1
    prev_trial = history[0][0]
    for trial_n, phase in history[1:]:
        if phase == run_phase:
            run_len += 1
            prev_trial = trial_n
            continue
        chunks.append(_format_phase_run(run_phase, run_len, run_start, prev_trial))
        run_phase = phase
        run_start = trial_n
        run_len = 1
        prev_trial = trial_n
    chunks.append(_format_phase_run(run_phase, run_len, run_start, prev_trial))
    return ", ".join(chunks)


def _format_phase_run(phase: str, run_len: int, start: int, end: int) -> str:
    span = f"trial {start}" if start == end else f"trials {start}-{end}"
    return f"{phase}×{run_len} ({span})"


def _format_pareto_block(sc: StateCard) -> list[str]:
    """Render the Pareto state for cost-aware reasoning.

    Shows the non-dominated frontier with ``★best`` marker, plus the FULL
    config of every frontier member so the agent can name a specific
    frontier member to perturb. No knee anchor and no cheapest-within-band
    pre-computation — the agent reads the frontier directly and picks.
    """
    lines: list[str] = ["", "## Pareto state"]
    lines.append(
        f"trials_since_frontier_improved={sc.trials_since_frontier_improved}  "
        f"current_trial_cost=${sc.current_trial_cost_usd:.6f}/q"
    )
    if not sc.pareto_frontier:
        lines.append("pareto_frontier: (no trials yet)")
        return lines

    lines.append(f"pareto_frontier ({len(sc.pareto_frontier)} non-dominated, cheapest first):")
    for entry in sorted(sc.pareto_frontier, key=lambda e: float(e.get("cost_usd", 0.0))):
        tn = entry.get("trial_number")
        lines.append(
            f"  - trial {tn}: cost=${float(entry.get('cost_usd', 0.0)):.6f}/q"
            f"  accuracy={float(entry.get('accuracy', 0.0)):.3f}"
            f"  in_tok={float(entry.get('in_tok', 0.0)):.0f}"
            f"  out_tok={float(entry.get('out_tok', 0.0)):.0f}"
            f"  | {entry.get('config_summary', '')}"
        )

    full_lines = _format_frontier_full_configs(sc.pareto_frontier)
    if full_lines:
        lines.append("")
        lines.append("frontier_configs (full config per member, for reference by trial_number):")
        lines.extend(full_lines)
    return lines


def _format_frontier_full_configs(frontier_entries: list[dict]) -> list[str]:
    """Render every field of every frontier member's TrialConfig.

    The proposer is asked to perturb a specific frontier member by trial
    number; without the full config it has to guess. Inapplicable graph
    fields render as ``n/a`` so the agent sees the absence explicitly.
    """
    out: list[str] = []
    graph_index_values = {"graph_only", "hybrid_graph_vector"}
    for entry in frontier_entries:
        cfg = entry.get("config")
        if not isinstance(cfg, dict):
            continue
        tn = entry.get("trial_number")
        is_graph = cfg.get("index_type") in graph_index_values
        graph_mode = cfg.get("graph_query_mode") if is_graph else "n/a"
        graph_top_k: object = cfg.get("graph_top_k") if is_graph else "n/a"
        out.append(
            f"  trial {tn}:"
            f" index_type={cfg.get('index_type')} embedding_model={cfg.get('embedding_model')}"
            f" | chunking={cfg.get('chunking_strategy')}"
            f" size={cfg.get('chunk_token_size')} overlap={cfg.get('chunk_token_overlap')}"
            f" | top_k={cfg.get('top_k')} hybrid_alpha={cfg.get('hybrid_alpha')}"
            f" reranker={cfg.get('reranker')} reranker_top_n={cfg.get('reranker_top_n')}"
            f" | query_expansion={cfg.get('query_expansion')}"
            f" | gen_llm={cfg.get('generator_llm')}"
            f" comp_llm={cfg.get('compressor_llm')}"
            f" exp_llm={cfg.get('expander_llm')}"
            f" temp={cfg.get('temperature')}"
            f" reasoning={str(cfg.get('reasoning')).lower()}"
            f" | graph_query_mode={graph_mode} graph_top_k={graph_top_k}"
        )
    return out


def _format_failure_attribution(fa: FailureAttribution) -> str:
    """Single-line render of failure attribution percentages.

    Only the two observable stages (retrieval, generation) are rendered.
    The ``ranking`` and ``composition`` axes are not derivable from
    ``QuestionResult`` alone — they stay at 0.0 in the mechanical
    attribution and are reasoned about by the Diagnoser in narrative.
    """
    return f"retrieval={fa.retrieval:.2f} generation={fa.generation:.2f}"


def _format_bundle_effects(
    effects: list[tuple[str, BundleEffectDelta]],
    *,
    fallback_label: str,
) -> str:
    """Render one or more ``(anchor_label, BundleEffectDelta)`` blocks back-to-back.

    The current build emits a single block anchored on the best-score trial.
    The list signature is retained so multi-anchor renderings (if ever
    re-introduced) plug in without changing the call site.
    """
    if not effects:
        return f"(no lever changes vs. {fallback_label})"
    return "\n\n".join(_format_bundle_effect(eff, anchor_label=label) for label, eff in effects)


def _format_bundle_effect(effect: BundleEffectDelta | None, *, anchor_label: str) -> str:
    """Render the trial-vs-anchor bundle delta on three axes (score,
    acc_given_complete, retrieval_complete).

    Cost is deliberately omitted — the Diagnoser is the sole consumer of this
    render and is objective-agnostic; cost is the Proposer's concern.

    The anchor is the best-score prior trial. When a single lever changed,
    the delta is cleanly attributable to that lever. When N>1 levers
    changed, the deltas reflect the *bundled* effect — they cannot be split
    per-lever from observation alone. The render makes that distinction
    explicit so the agent doesn't credit/blame any individual lever in a
    multi-change bundle.

    Layout: the changes block is rendered first, then the column header sits
    directly above the data row so the labels stay visually bound to their
    numbers.
    """
    if effect is None or not effect.changes:
        return f"(no lever changes vs. {anchor_label})"
    header_line = "  Δaccuracy   Δacc|complete  Δrcomp"
    delta_row = (
        f"  {effect.accuracy_delta:+.3f}      {effect.acc_given_complete_delta:+.3f}         "
        f"{effect.retrieval_complete_delta:+.3f}"
    )
    if len(effect.changes) == 1:
        return f"vs. {anchor_label}:\n  {effect.changes[0]}\n{header_line}\n{delta_row}"
    change_lines = "\n".join(f"    - {c}" for c in effect.changes)
    return (
        f"vs. {anchor_label}:\n"
        f"  bundle of {len(effect.changes)} levers changed (effect below is the BUNDLE, NOT per-lever):\n"
        f"{change_lines}\n"
        f"{header_line}\n"
        f"{delta_row}"
    )


def _format_diagnosis(d: Diagnosis, *, show_cost: bool = True) -> str:
    lines = [f"trial_metrics: {_format_trial_metrics(d.trial_metrics, show_cost=show_cost)}"]
    if d.confirmed_findings:
        lines.append("confirmed_findings:")
        lines.extend(f"  - {item}" for item in d.confirmed_findings)
    if d.notable_deltas:
        lines.append("notable_deltas:")
        lines.extend(f"  - {item}" for item in d.notable_deltas)
    lines.append(f"narrative: {d.narrative}")
    return "\n".join(lines)


def _extract_narrative(text: str) -> str:
    """Return prose prior to the first ``` fence as the narrative fallback."""
    idx = text.find("```")
    return text[:idx].strip() if idx > 0 else text.strip()


def _truncate_for_list(text: str, n: int) -> str:
    """Single-line truncation used by the Tier-2 failure list."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= n else flat[: n - 1] + "…"


def _split_legacy_context(context: str) -> list[str]:
    """Best-effort split of an old-format ``retrieved_context`` into chunks.

    Used only for records that pre-date ``QuestionResult.retrieved_chunks``.
    Splits on blank lines (which the LLM-facing context tends to use as
    paragraph boundaries) and falls back to a single chunk if none are found.
    """
    if not context:
        return []
    pieces = [p for p in re.split(r"\n\s*\n", context) if p.strip()]
    return pieces if pieces else [context]


def _find_span_in_chunk(chunk: str, span: str) -> int:
    """Return char offset of ``span`` in ``chunk``, or -1 when not present.

    Match progression (each fallback uses a strictly more lenient comparison):
      1. Exact ``str.find``.
      2. Unicode-folded ``str.find`` (en-dash/em-dash/etc. → hyphen, curly
         quotes → straight, non-breaking space → space). The fold table is
         (with one exception) 1-char-to-1-char so offsets are preserved.
      3. Whitespace-collapsed search on the folded text, with offset mapped
         back to the original chunk via a single linear pass.

    Returns the offset into the original ``chunk`` string for slicing. The
    ellipsis fold ("…" → "...") perturbs offsets by 2 chars per ellipsis;
    the window radius (~240 chars) absorbs this without losing context.
    """
    if not chunk or not span:
        return -1
    idx = chunk.find(span)
    if idx >= 0:
        return idx

    folded_chunk = _fold_unicode(chunk)
    folded_span = _fold_unicode(span)
    fidx = folded_chunk.find(folded_span)
    if fidx >= 0:
        return fidx

    norm_chunk = re.sub(r"\s+", " ", folded_chunk)
    norm_span = re.sub(r"\s+", " ", folded_span).strip()
    if not norm_span:
        return -1
    nidx = norm_chunk.find(norm_span)
    if nidx < 0:
        return -1
    seen = 0
    last_was_space = False
    for i, ch in enumerate(folded_chunk):
        if ch.isspace():
            if last_was_space:
                continue
            last_was_space = True
        else:
            last_was_space = False
        if seen == nidx:
            return i
        seen += 1
    return -1


def _window_around(chunk: str, offset: int, span_len: int, radius: int = _SPAN_WINDOW_CHARS) -> str:
    """Slice ``chunk[offset-radius : offset+span_len+radius]`` with elision markers."""
    start = max(0, offset - radius)
    end = min(len(chunk), offset + span_len + radius)
    window = chunk[start:end]
    prefix = "… " if start > 0 else ""
    suffix = " …" if end < len(chunk) else ""
    return f"{prefix}{window}{suffix}"


def _chunk_prefix(chunk: str, n: int = _CHUNK_PREFIX_CHARS) -> str:
    """First ``n`` chars of a chunk, with an elision marker when truncated."""
    if not chunk:
        return ""
    flat = chunk.strip()
    if len(flat) <= n:
        return flat
    return flat[:n].rstrip() + " …"


def _render_chunks_for_mode(
    *,
    mode: str,
    chunks: list[str],
    retrieved_doc_ids: list[str],
    gold_spans: list[str],
    chunk_satisfies_spans: list[list[int]] | None = None,
) -> list[str]:
    """Failure-mode-adaptive per-chunk rendering. Universal across corpora.

    ``chunk_satisfies_spans`` is the evaluator's authoritative per-chunk ×
    per-span match table (see ``QuestionResult.chunk_satisfies_spans``). When
    present, it drives the "this chunk satisfies span_i" decision for the
    ``retrieval_partial`` / ``retrieval_complete`` / ``generation_wrong`` /
    ``refused`` modes; the renderer then tries to locate the span verbatim
    (with unicode-fold + whitespace-collapse fallbacks) for a clean window,
    falling back to a chunk-prefix tagged ``(approximate match)`` when the
    span text isn't directly in the chunk (the evaluator credited it via
    n-gram coverage). When the table is absent (legacy records), the
    renderer falls through to text-only span search.
    """
    if not chunks:
        return ["  retrieved_chunks: <none>"]

    def _label(rank: int) -> str:
        doc_id = retrieved_doc_ids[rank - 1] if rank - 1 < len(retrieved_doc_ids) else "<unknown>"
        return f"[rank={rank} | doc={doc_id}]"

    def _render_span_hit(rank: int, chunk: str, span_idx: int) -> str:
        """Render a (chunk, span_idx) hit as a window or approximate-match prefix."""
        span_text = gold_spans[span_idx] if span_idx < len(gold_spans) else ""
        if not span_text:
            return f"  {_label(rank)} (matched span_{span_idx + 1}, span text unavailable)"
        offset = _find_span_in_chunk(chunk, span_text)
        if offset >= 0:
            window = _window_around(chunk, offset, len(span_text))
            return f"  {_label(rank)} [span_{span_idx + 1}: {span_text!r}] window: {window!r}"
        return f"  {_label(rank)} [span_{span_idx + 1} approximate match]: {_chunk_prefix(chunk)!r}"

    def _ranks_with_span_hits() -> list[tuple[int, list[int]]]:
        """List of (rank, satisfied_span_indices) for every chunk with a hit.

        Uses the evaluator's table when present; otherwise falls back to text
        search and reports the per-span indices found by ``_find_span_in_chunk``.
        """
        out: list[tuple[int, list[int]]] = []
        if chunk_satisfies_spans:
            for rank, indices in enumerate(chunk_satisfies_spans, start=1):
                if indices:
                    out.append((rank, list(indices)))
            return out
        for rank, chunk in enumerate(chunks, start=1):
            indices = [i for i, span in enumerate(gold_spans) if _find_span_in_chunk(chunk, span) >= 0]
            if indices:
                out.append((rank, indices))
        return out

    lines: list[str] = []
    if mode == "retrieval_miss":
        for rank, chunk in enumerate(chunks[:2], start=1):
            lines.append(f"  {_label(rank)} prefix: {_chunk_prefix(chunk)!r}")
        return lines

    hits = _ranks_with_span_hits()

    if mode in ("generation_wrong", "retrieval_complete"):
        for rank, span_indices in hits:
            chunk = chunks[rank - 1]
            for span_idx in span_indices:
                lines.append(_render_span_hit(rank, chunk, span_idx))
        if not lines:
            lines.append("  (no gold spans located in retrieved_chunks; showing top-2 prefixes)")
            for rank, chunk in enumerate(chunks[:2], start=1):
                lines.append(f"  {_label(rank)} prefix: {_chunk_prefix(chunk)!r}")
        return lines

    if mode == "retrieval_partial":
        for rank, span_indices in hits:
            chunk = chunks[rank - 1]
            for span_idx in span_indices:
                lines.append(_render_span_hit(rank, chunk, span_idx))
        if not lines:
            for rank, chunk in enumerate(chunks[:2], start=1):
                lines.append(f"  {_label(rank)} prefix: {_chunk_prefix(chunk)!r}")
        hit_ranks = {r for r, _ in hits}
        distractor_rank = next((r for r in range(1, len(chunks) + 1) if r not in hit_ranks), None)
        if distractor_rank is not None:
            lines.append(
                f"  {_label(distractor_rank)} (distractor) prefix: {_chunk_prefix(chunks[distractor_rank - 1])!r}"
            )
        return lines

    if mode == "refused":
        for rank, span_indices in hits:
            chunk = chunks[rank - 1]
            for span_idx in span_indices:
                lines.append(_render_span_hit(rank, chunk, span_idx))
        if not lines and chunks:
            lines.append(f"  {_label(1)} prefix: {_chunk_prefix(chunks[0])!r}")
        if len(chunks) > 1:
            hit_ranks = {r for r, _ in hits}
            distractor_rank = next((r for r in range(1, len(chunks) + 1) if r not in hit_ranks), None) or 2
            if distractor_rank - 1 < len(chunks):
                lines.append(
                    f"  {_label(distractor_rank)} (distractor) prefix: {_chunk_prefix(chunks[distractor_rank - 1])!r}"
                )
        return lines

    # Unknown mode — show two prefixes defensively.
    for rank, chunk in enumerate(chunks[:2], start=1):
        lines.append(f"  {_label(rank)} prefix: {_chunk_prefix(chunk)!r}")
    return lines


def _coerce_str_list(value: object) -> list[str]:
    """Coerce a YAML field to a list of stripped strings; missing/None becomes []."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def _format_failure_history(failures: list[tuple[TrialConfig, str]]) -> str:
    """Render past failed (config, error) pairs as a deduped, human-readable list."""
    if not failures:
        return "(none yet)"
    lines: list[str] = []
    for i, (cfg, err) in enumerate(failures, 1):
        idx = getattr(cfg.index_type, "value", cfg.index_type)
        summary = (
            f"  - failure {i}: reranker={cfg.reranker} embed={cfg.embedding_model}"
            f" gen_llm={cfg.generator_llm} comp_llm={cfg.compressor_llm}"
            f" exp_llm={cfg.expander_llm} index={idx} chunk={cfg.chunk_token_size}"
            f" top_k={cfg.top_k}"
        )
        first_line = err.strip().splitlines()[0] if err else "<no message>"
        lines.append(summary + f"\n    error: {first_line}")
    return "\n".join(lines)
