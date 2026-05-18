"""Two-stage reasoning agent for RAG optimization.

Stage 1 (diagnose): interpret the just-completed trial's per-question
results and emit a structured ``Diagnosis`` (trial metrics + ordered
bottlenecks).

Stage 2 (propose): pick the next ``TrialConfig`` and emit a structured
``ProposalMeta`` (changes, rationale, durable memo). No hard move-type
validators — guidance lives in the prompt.
"""

from __future__ import annotations

import logging
import math
import random
import re
from pathlib import Path

import litellm
import yaml

from agentic_autorag.config.knowledge_base import KnowledgeBase
from agentic_autorag.config.models import OpenEndedQuestion, ProjectConfig, TrialConfig
from agentic_autorag.examiner._errors import ERROR_SENTINELS
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.examiner.exam_validator import _fold_unicode
from agentic_autorag.litellm_runtime import acompletion_with_cost
from agentic_autorag.optimizer import pareto
from agentic_autorag.optimizer.diagnosis import (
    BundleEffectDelta,
    Diagnosis,
    FailureAttribution,
    FrontierContext,
    ProposalMeta,
    StateCard,
    Strategy,
    TrialMetrics,
)
from agentic_autorag.optimizer.history import HistoryLog
from agentic_autorag.optimizer.state import (
    _top_stages_from_attribution,
    build_failure_attribution,
    build_failure_cross_tab,
    build_frontier_context,
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

MAX_RETRIES = 3

_DEEP_FAILURE_SAMPLE = 12
_KEY_EVIDENCE_SAMPLE = 5
_CHUNK_PREFIX_CHARS = 240  # ~60 tokens at 4 chars/token
_SPAN_WINDOW_CHARS = 240  # ~60 tokens before/after gold span


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


_PIPELINE_RULE_BLOCKS: dict[str, str] = {
    "query_decompose": (
        "   - query_expansion='query_decompose' REPLACES the original query with N\n"
        "     self-contained sub-queries each retrieved independently — does NOT\n"
        "     augment. Strong fit for multi-hop tasks; overhead-only for single-hop.\n"
        "     When the LLM declares the query already atomic, falls back to the\n"
        "     original. Costs one extra LLM call per query and multiplies retrieval\n"
        "     work by N — raise top_k modestly when enabling."
    ),
    "passage_compressor": (
        "   - passage_compressor 'tree_summarize' synthesises retrieved passages\n"
        "     recursively (batch=16) into a single string; 'refine' threads a\n"
        "     running answer through passages serially. Both help when retrieval is\n"
        "     noisy; both can lose exact spans the grader needs. tree_summarize\n"
        "     fans out concurrently per level; refine is serial and N LLM calls."
    ),
    "long_context_reorder": (
        "   - long_context_reorder duplicates the top-scored passage at the END of\n"
        "     the joined context (input order otherwise preserved). Useful when\n"
        "     top_k is large. It is a no-op when passage_compressor != 'none'\n"
        "     (compression collapses retrieval to one string) — don't toggle both."
    ),
    "bm25_vector_fusion": (
        "   - bm25_vector_fusion 'rrf' fuses BM25 and vector by rank reciprocals\n"
        "     (robust to score-scale mismatch); 'alpha' is a smooth tunable\n"
        "     score-blend (use hybrid_alpha). Only meaningful when index_type is\n"
        "     hybrid_bm25_vector."
    ),
}


def _pipeline_rules_for(search_space) -> str:
    """Return only the rule blocks whose levers are active in the search space.

    "Active" means the lever has non-trivial runtime behavior — either tunable
    or pinned to a non-trivial value. Inactive levers (e.g. ``passage_compressor``
    pinned to ``"none"``) need no guidance because they don't affect the run.
    """
    active = search_space.active_levers()
    blocks: list[str] = []
    if "query_decompose" in search_space.query_expansion:
        blocks.append(_PIPELINE_RULE_BLOCKS["query_decompose"])
    if "passage_compressor" in active:
        blocks.append(_PIPELINE_RULE_BLOCKS["passage_compressor"])
    if "long_context_reorder" in active:
        blocks.append(_PIPELINE_RULE_BLOCKS["long_context_reorder"])
    if "bm25_vector_fusion" in active:
        blocks.append(_PIPELINE_RULE_BLOCKS["bm25_vector_fusion"])
    return ("\n".join(blocks) + "\n") if blocks else ""


_GRAPH_GUIDANCE = """\
3. If graph-based index types are available (graph_only, hybrid_graph_vector),
   consider whether the content is relationship-rich (e.g. scientific papers
   with many named entities, legal documents with cross-references). If so,
   starting with a graph or hybrid type may be advantageous.
4. When index_type is graph_only or hybrid_graph_vector, set graph_query_mode
   and graph_top_k appropriately. "hybrid" mode generally works best as a
   starting point; larger graph_top_k captures more graph context.
"""

_REASONING_GUIDANCE = """\
5. Start with reasoning: false unless the corpus clearly requires deep
   multi-step reasoning (e.g. math, logic, complex inference). You can
   enable reasoning in later trials if reasoning_error is the dominant
   failure pattern."""

_GRAPH_DIAGNOSTIC_TYPES = """\
When a graph index is in use, additional levers exist: entity-focused retrieval
via ``graph_query_mode`` and ``graph_top_k``. Entity gaps or missing relationships
can be addressed by increasing ``graph_top_k`` or swapping the graph query mode.
"""


def _failure_mode(qr: QuestionResult) -> str:
    """Categorise a question into one of the open-ended failure modes."""
    if qr.refused:
        return "refused"
    if qr.retrieved_spans == 0:
        return "retrieval_miss"
    if qr.retrieved_spans < qr.n_spans:
        return "retrieval_partial"
    if not qr.correct:
        return "generation_wrong"
    return "retrieval_complete"


def _effective_anchor_trial(history_records: list) -> int | None:
    """Trial number of the current Pareto knee across prior trials.

    The anchor is orchestrator-managed: lever-effect deltas are always
    computed against the run's current score-per-cost knee, not an
    agent-chosen reference. Empty history → None (first trial). Single-record
    history → that record (no frontier yet, fall back to it).
    """
    if not history_records:
        return None
    frontier = pareto.compute_frontier(list(history_records))
    if not frontier:
        return None
    knee = pareto.find_knee(frontier)
    if knee is None:
        knee = max(frontier, key=lambda r: float(getattr(r, "score", 0.0)))
    return int(getattr(knee, "trial_number", 0)) or None


class ReasoningAgent:
    """Two-stage reasoning agent with structured Diagnosis → ProposalMeta hand-off.

    Pure functions in ``state.py`` pre-compute the trial metrics and state
    card so the LLM's job shrinks to interpretation and selection.
    """

    def __init__(
        self,
        agent_model: str,
        config: ProjectConfig,
        history: HistoryLog,
        debug_prompts: bool = False,
        knowledge_base: KnowledgeBase | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = agent_model
        self.config = config
        self.history = history
        self.debug_prompts = debug_prompts
        self.knowledge_base = knowledge_base
        # Forwarded to litellm as ``seed=`` on every proposer call. Providers
        # that don't accept ``seed`` drop it via ``litellm.drop_params=True``.
        self.seed = seed
        self._include_graph = config.uses_graph()
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
        if not self.debug_prompts:
            return
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
        prompt = INITIAL_PROPOSAL_PROMPT.format(
            corpus_description=corpus_description,
            search_space=self.config.to_agent_prompt(),
            knowledge_base=self._kb_text(),
            graph_guidance=_GRAPH_GUIDANCE if self._include_graph else "",
            reasoning_guidance=_REASONING_GUIDANCE if self.config.search_space.reasoning else "",
            module_rules=_pipeline_rules_for(self.config.search_space),
        )
        return await self._call_for_config_only(prompt, stage="Initial Proposer")

    async def propose_after_failure(
        self,
        *,
        failed_config: TrialConfig,
        error_summary: str,
        failure_history: list[tuple[TrialConfig, str]],
    ) -> tuple[TrialConfig, ProposalMeta]:
        """Pick a recovery config after a trial failed before producing a result.

        ``failure_history`` is the list of all (config, error) pairs that have
        failed in this run, so the agent can avoid re-proposing them. The
        returned ``ProposalMeta`` carries the agent's `changes`/`rationale`/
        `memo` so the orchestrator can persist the recovery decision.
        """
        history_text = self.history.format_for_agent(last_n=self.config.agent.max_history_trials)
        prompt = FAILURE_RECOVERY_PROMPT.format(
            failed_config=failed_config.to_prompt_json(include_graph=self._include_graph),
            error_summary=error_summary,
            failure_history=_format_failure_history(failure_history),
            history=history_text,
            search_space=self.config.to_agent_prompt(),
            knowledge_base=self._kb_text(),
            graph_rules=_GRAPH_RULES if self._include_graph else "",
            module_rules=_pipeline_rules_for(self.config.search_space),
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
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your response had an error: {e}\n\n"
                                "Please fix the issue and output a corrected ```yaml block with"
                                " a TrialConfig and a `meta:` dict (changes/rationale/memo)."
                            ),
                        }
                    )

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
        """Diagnose the current trial, then propose the next config.

        Returns ``(trial_metrics, diagnosis, next_config, proposal_meta)``.
        ``trial_metrics`` and ``diagnosis`` describe the just-completed trial;
        ``next_config`` and ``proposal_meta`` describe the next one.

        ``previous_strategy`` is the agent-owned strategy that was active
        during ``trial_number`` — threaded in from the orchestrator's
        ``_active_strategy``. The proposer validates its emitted strategy
        against this previous commitment (ratchet, lock-in, done gate).
        """
        trial_metrics = compute_trial_metrics(exam_result)

        frontier_context = build_frontier_context(
            history_records=self.history.records,
            current_trial_number=trial_number,
            current_score=exam_result.score,
            current_cost_usd=exam_result.mean_llm_cost_per_query_usd,
            current_config=current_config,
        )

        diagnosis = await self._diagnose(
            exam_result=exam_result,
            exam_questions=exam_questions,
            current_config=current_config,
            trial_metrics=trial_metrics,
            trial_number=trial_number,
            trials_remaining=trials_remaining,
            frontier_context=frontier_context,
            previous_strategy=previous_strategy,
        )

        top_modes = _top_stages_from_attribution(diagnosis.failure_attribution, n=2)
        max_trials = trial_number + trials_remaining
        state_card = build_state_card(
            trial_number=trial_number,
            trials_remaining=trials_remaining,
            current_score=exam_result.score,
            history_records=self.history.records,
            max_trials=max_trials,
            current_config=current_config,
            current_top_failure_modes=top_modes,
            current_cost_usd=exam_result.mean_llm_cost_per_query_usd,
            polish_score_tolerance=self.config.meta.polish_score_tolerance,
            previous_strategy=previous_strategy,
            allow_early_exit=self.config.meta.allow_early_exit,
            min_trials_before_done=self.config.meta.min_trials_before_done,
            min_frontier_size_for_done=self.config.meta.min_frontier_size_for_done,
            early_exit_hv_epsilon=self.config.meta.early_exit_hv_epsilon,
        )

        next_config, meta = await self._propose(
            diagnosis=diagnosis,
            exam_questions=exam_questions,
            question_results=exam_result.question_results,
            current_config=current_config,
            state_card=state_card,
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
        frontier_context: FrontierContext,
        previous_strategy: Strategy | None,
    ) -> Diagnosis:
        """Produce a structured ``Diagnosis`` from failed exam questions.

        Evidence pipeline (orchestrator-side, mechanical):
          1. Stratified deep sample of failures (12 by default, seeded).
          2. Tier-1 cross-tab over ALL failures.
          3. Tier-2 one-line-per-failure list over ALL failures.
          4. Mechanical ``failure_attribution`` from per-question modes.
          5. ``lever_effect_deltas`` against the strategy anchor trial.
          6. Decontaminated history (Diagnoser view — no prior Proposer
             fields or Diagnoser interpretive labels).

        The agent re-emits its own ``failure_attribution`` so it must explicitly
        look at the numbers and may disagree in the narrative. Numeric
        validation in ``_build_diagnosis`` rejects hallucinated regression claims.
        """
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
        sample_seed = self._failure_sample_seed(trial_number)
        sample = self._select_stratified_failures(
            real_failures,
            question_by_id,
            prev_results_by_id,
            n=_DEEP_FAILURE_SAMPLE,
            seed=sample_seed,
        )
        deep_blocks = "\n\n".join(self._render_failure_block(qr, question_by_id.get(qr.question_id)) for qr in sample)
        failed_questions = (deep_blocks or "(no failures this trial)") + error_note

        failure_crosstab = build_failure_cross_tab(valid_results, exam_questions)
        failure_list = self._render_failure_list(real_failures, question_by_id)
        mechanical_attribution = build_failure_attribution(valid_results)

        # Anchor is orchestrator-managed: always compute deltas against the
        # current Pareto knee, not against an agent-chosen reference. Keeps
        # the reference frame meaningful as the frontier moves.
        anchor_trial = _effective_anchor_trial(self.history.records)
        bundle_effect = compute_bundle_effect(
            history_records=self.history.records,
            current_config=current_config,
            current_metrics=trial_metrics,
            current_cost_usd=exam_result.mean_llm_cost_per_query_usd,
            anchor_trial=anchor_trial,
        )
        anchor_label = (
            f"current Pareto knee (trial {anchor_trial})" if anchor_trial is not None else "n/a (first trial)"
        )

        config_json = current_config.to_prompt_json(include_graph=self._include_graph)
        graph_diag = _GRAPH_DIAGNOSTIC_TYPES if self._include_graph else ""
        history_text = self.history.format_for_agent(
            last_n=self.config.agent.max_history_trials,
            include_proposer_context=False,
        )
        diagnostic_state = (
            f"trial_number={trial_number} trials_remaining={trials_remaining}"
            f" best_score_so_far={self._best_score():.3f}"
        )
        prompt = DIAGNOSTIC_PROMPT.format(
            trial_metrics=_format_trial_metrics(trial_metrics),
            state_card=diagnostic_state,
            current_config=config_json,
            history_count=self.config.agent.max_history_trials,
            history=history_text,
            failure_crosstab=failure_crosstab,
            failure_list=failure_list,
            mechanical_failure_attribution=_format_failure_attribution(mechanical_attribution),
            lever_effect_deltas=_format_bundle_effect(bundle_effect, anchor_label=anchor_label),
            failed_questions=failed_questions,
            graph_diagnostic_types=graph_diag,
            frontier_signal=_format_frontier_context(frontier_context),
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
                    mechanical_attribution=mechanical_attribution,
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
                            " a `failure_attribution` mapping, a `narrative` string, the lists"
                            " `confirmed_findings` / `open_questions` / `notable_deltas` /"
                            " `illustrative_qids`, a boolean `regression_detected`, and (when true)"
                            " a `regression_axes` list."
                        ),
                    }
                )

        if diagnosis is None:
            logger.error("Diagnoser returned unparseable output after %d attempts; falling back", MAX_RETRIES)
            diagnosis = Diagnosis(
                trial_metrics=trial_metrics,
                failure_attribution=mechanical_attribution,
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
    ) -> tuple[TrialConfig, ProposalMeta]:
        """Produce the next (TrialConfig, ProposalMeta).

        Validates the agent's emitted Strategy against ``previous_strategy``
        (ratchet, lock-in, done gate). On violation, surfaces the broken
        rule in the retry-prompt message so the agent can correct itself.
        Orchestrator-managed Strategy fields (``committed_at_trial``,
        ``revision_count``) are overwritten after validation — the agent
        cannot fake its own commitment history.

        Threads the Diagnoser-selected ``illustrative_qids`` into a
        "## Key evidence" section so the Proposer can verify Diagnoser
        claims against raw failed-question blocks.
        """
        history_text = self.history.format_for_agent(
            last_n=self.config.agent.max_history_trials,
            include_proposer_context=True,
        )
        key_evidence = self._format_key_evidence(diagnosis, exam_questions, question_results)

        prompt = PROPOSAL_PROMPT.format(
            diagnosis=_format_diagnosis(diagnosis),
            state_card=_format_state_card(state_card),
            current_config=current_config.to_prompt_json(include_graph=self._include_graph),
            history=history_text,
            key_evidence=key_evidence,
            search_space=self.config.to_agent_prompt(),
            knowledge_base=self._kb_text(),
            graph_rules=_GRAPH_RULES if self._include_graph else "",
            module_rules=_pipeline_rules_for(self.config.search_space),
        )

        messages = [{"role": "user", "content": prompt}]
        last_raw = ""
        for attempt in range(MAX_RETRIES):
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
                if meta.strategy is None:
                    raise ValueError(
                        "proposal `meta.strategy` is required. Emit a `strategy:` block with "
                        "stance/intent/journal/anchor_trial (and done_reason or regression_reason "
                        "where applicable)."
                    )
                _validate_strategy_transition(
                    previous=previous_strategy,
                    proposed=meta.strategy,
                    intended_trial=intended_trial,
                    last_diagnosis=diagnosis,
                    state_card=state_card,
                    min_stance_lock_trials=self.config.meta.min_stance_lock_trials,
                )
                meta.strategy = _finalize_strategy(
                    proposed=meta.strategy,
                    previous=previous_strategy,
                    intended_trial=intended_trial,
                    effective_anchor=_effective_anchor_trial(self.history.records),
                )
                return config, meta

            except Exception as e:
                logger.warning("Proposer attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, e)
                if attempt < MAX_RETRIES - 1:
                    messages.append({"role": "assistant", "content": last_raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your response had an error: {e}\n\n"
                                "Please fix the issue and output a corrected ```yaml block with BOTH "
                                "the TrialConfig fields AND the `meta:` dict containing `changes`, "
                                "`rationale`, and `strategy` (stance/intent/journal/anchor_trial)."
                            ),
                        }
                    )

        raise RuntimeError(f"Failed to get valid proposal after {MAX_RETRIES} attempts")

    def _build_diagnosis(
        self,
        *,
        raw: str,
        trial_metrics: TrialMetrics,
        mechanical_attribution: FailureAttribution | None = None,
        exam_qids: set[str] | None = None,
    ) -> Diagnosis:
        """Parse the diagnoser's YAML, validate, and merge in mechanical signals.

        Raises ``ValueError`` so the retry loop in ``_diagnose`` can re-prompt
        the agent. Validation:
          - ``illustrative_qids`` must be a subset of this trial's exam.
          - When ``regression_detected=True``, at least one listed axis must
            actually be worse on the current trial than the best-so-far across
            prior trials by ≥ ``regression_threshold``. A regression is defined
            relative to the best the run has ever achieved.
        """
        yaml_dict = self._extract_yaml(raw)
        narrative = yaml_dict.get("narrative") or _extract_narrative(raw)

        attribution_dict = yaml_dict.get("failure_attribution") or {}
        if isinstance(attribution_dict, dict):
            attribution = FailureAttribution.model_validate(attribution_dict)
        else:
            attribution = FailureAttribution()
        # If the agent emitted zeros (or the field was missing) AND we have
        # mechanical attribution available, fall through to the mechanical
        # numbers so downstream consumers still see useful evidence.
        if mechanical_attribution is not None and _attribution_is_empty(attribution):
            attribution = mechanical_attribution

        confirmed = _coerce_str_list(yaml_dict.get("confirmed_findings"))
        open_qs = _coerce_str_list(yaml_dict.get("open_questions"))
        notable = _coerce_str_list(yaml_dict.get("notable_deltas"))
        qids = _coerce_str_list(yaml_dict.get("illustrative_qids"))

        regression = bool(yaml_dict.get("regression_detected", False))
        raw_axes = yaml_dict.get("regression_axes") or []
        valid_axes = {"score", "acc_given_complete", "retrieval_complete", "cost"}
        axes = [a for a in _coerce_str_list(raw_axes) if a in valid_axes] if regression else []

        if exam_qids is not None:
            bad_qids = [q for q in qids if q not in exam_qids]
            if bad_qids:
                raise ValueError(
                    f"illustrative_qids contains qids not in this trial's exam: {bad_qids}. "
                    "Use only question_ids from the failed-question blocks above."
                )

        if regression:
            if not axes:
                raise ValueError(
                    "regression_detected=true requires a non-empty regression_axes list "
                    "(one or more of: score, acc_given_complete, retrieval_complete, cost)."
                )
            if self.history.records:
                threshold = float(self.config.meta.regression_threshold)
                unsupported = [
                    a for a in axes if not _axis_regressed_vs_history(a, trial_metrics, self.history.records, threshold)
                ]
                if unsupported:
                    raise ValueError(
                        "regression_detected=true on axes "
                        f"{unsupported} but none of those axes regressed by ≥ "
                        f"{threshold:.3f} versus the best-so-far across prior trials. "
                        "Drop the unsupported axis or set regression_detected=false."
                    )

        return Diagnosis(
            trial_metrics=trial_metrics,
            failure_attribution=attribution,
            narrative=narrative,
            confirmed_findings=confirmed[:5],
            open_questions=open_qs[:5],
            regression_detected=regression,
            regression_axes=axes,
            notable_deltas=notable[:4],
            illustrative_qids=qids[:5],
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
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your response had an error: {e}\n\n"
                                "Please fix the issue and output a corrected ```yaml block."
                            ),
                        }
                    )

        raise RuntimeError(f"Failed to get valid config after {MAX_RETRIES} attempts")

    def _prev_trial_correctness(self) -> dict[str, bool]:
        """``question_id → correct`` from the most recent prior trial.

        Used to give the stratified sampler a "flipped since last trial" tier.
        Returns an empty dict when there is no prior trial.
        """
        if not self.history.records:
            return {}
        prev = self.history.records[-1]
        return {qr.question_id: bool(qr.correct) for qr in prev.question_results}

    def _failure_sample_seed(self, trial_number: int) -> int:
        """Seed used by the stratified failure sampler.

        Honours ``MetaConfig.failure_sample_seed`` when set; otherwise derives
        from the trial number so the picks are deterministic-per-trial but
        vary across trials.
        """
        configured = self.config.meta.failure_sample_seed
        return int(configured) if configured is not None else int(trial_number)

    def _format_key_evidence(
        self,
        diagnosis: Diagnosis,
        exam_questions: list[OpenEndedQuestion],
        question_results: list[QuestionResult],
    ) -> str:
        """Render the Diagnoser-selected ``illustrative_qids`` as raw blocks.

        These are the questions the Diagnoser thought best showcase the
        observed pattern (most representative / newly failing / newly fixed).
        Re-using ``_render_failure_block`` keeps the format identical to what
        the Diagnoser already saw, so the Proposer can verify Diagnoser claims
        against ground truth.
        """
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
        return max(float(r.score) for r in self.history.records)

    def _kb_text(self) -> str:
        if self.knowledge_base is None:
            return ""
        ss = self.config.search_space
        # ``reasoning_allowed`` keys all stage LLMs the agent might see in the
        # KB table, but only generator-stage LLMs are eligible to toggle
        # reasoning (the reasoning_effort knob applies to the final-answer
        # call). For non-generator stages we report ``False`` regardless of
        # litellm catalog claims to avoid misleading the proposer.
        all_llms = ss.llm_models.all_models()
        generator_set = set(ss.llm_models.generator)
        reasoning_allowed = {m: ss.is_reasoning_allowed(m) if m in generator_set else False for m in all_llms}
        # Skip parameter-guide entries only for pinned-AND-inactive levers.
        # Pinned-but-active levers (e.g. passage_compressor=["tree_summarize"])
        # still need their guide so the agent understands what's running.
        pinned = set(ss.pinned_field_values().keys())
        inactive_pinned = pinned - ss.active_levers()
        return self.knowledge_base.format_for_prompt(
            llm_models=all_llms,
            embedding_models=ss.embedding_models,
            reranker_models=ss.reranker.models,
            reasoning_allowed=reasoning_allowed,
            reasoning_enabled=ss.reasoning,
            include_graph=self._include_graph,
            skip_params=inactive_pinned,
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
        """Overwrite any pinned-field entries with their single legal value.

        The proposer prompt instructs the agent to omit pinned fields; this
        injection is the defensive backstop that makes the parse succeed even
        if the agent ignores those instructions. Pinned wins unconditionally.

        When the agent's emitted value *differs* from the pinned value we log
        a warning — that's a search-space violation that the new prompt
        structure was supposed to prevent, and counting these tells us whether
        the prompt change is landing or whether we're just relying on the
        backstop.
        """
        pinned = self.config.search_space.pinned_field_values()
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
            mode = _failure_mode(qr)
            q = questions_by_id.get(qr.question_id)
            rt = q.reasoning_type if q is not None else "unknown"
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
        mode = mode or _failure_mode(qr)
        q = question
        question_text = q.question if q else "<question text unavailable>"
        gold_text = q.canonical_answer if q else qr.correct_answer
        spans: list[str] = list(q.source_spans) if q else []
        source_doc_ids: list[str] = list(q.source_doc_ids) if q and q.source_doc_ids else []
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
    def _select_stratified_failures(
        failures: list[QuestionResult],
        questions_by_id: dict[str, OpenEndedQuestion],
        prev_results_by_id: dict[str, bool] | None,
        *,
        n: int = _DEEP_FAILURE_SAMPLE,
        seed: int = 0,
    ) -> list[QuestionResult]:
        """Stratified sample across ``(failure_mode, reasoning_type)`` cells.

        Prioritises questions that *flipped* since the previous trial (correct
        last trial, failing now) so the Diagnoser sees the most informative
        deltas. Falls back to a seeded round-robin across cells once flipped
        questions are exhausted.
        """
        if not failures:
            return []

        rng = random.Random(seed)

        def cell_key(qr: QuestionResult) -> tuple[str, str]:
            mode = _failure_mode(qr)
            q = questions_by_id.get(qr.question_id)
            return mode, (q.reasoning_type if q is not None else "unknown")

        flipped: list[QuestionResult] = []
        steady: list[QuestionResult] = []
        for qr in failures:
            was_correct = bool(prev_results_by_id and prev_results_by_id.get(qr.question_id, False))
            (flipped if was_correct else steady).append(qr)

        # Group steady failures by cell so round-robin pulls cover the table.
        cells: dict[tuple[str, str], list[QuestionResult]] = {}
        for qr in steady:
            cells.setdefault(cell_key(qr), []).append(qr)
        for bucket in cells.values():
            rng.shuffle(bucket)
        rng.shuffle(flipped)
        cell_order = list(cells.keys())
        rng.shuffle(cell_order)

        picked: list[QuestionResult] = []
        seen: set[str] = set()
        for qr in flipped:
            if len(picked) >= n:
                break
            if qr.question_id in seen:
                continue
            picked.append(qr)
            seen.add(qr.question_id)
        # Round-robin across cells.
        while len(picked) < n and any(cells[k] for k in cell_order):
            progressed = False
            for key in cell_order:
                if not cells[key]:
                    continue
                qr = cells[key].pop()
                if qr.question_id in seen:
                    continue
                picked.append(qr)
                seen.add(qr.question_id)
                progressed = True
                if len(picked) >= n:
                    break
            if not progressed:
                break
        return picked


def _format_trial_metrics(tm: TrialMetrics) -> str:
    return (
        f"answer_accuracy={tm.answer_accuracy:.3f}"
        f" | retrieval: complete={tm.retrieval_complete:.3f}"
        f" partial={tm.retrieval_partial:.3f}"
        f" miss={tm.retrieval_miss:.3f}"
        f" | refusal_rate={tm.refusal_rate:.3f}"
        f" | acc_given_complete={tm.answer_correct_given_complete_retrieval:.3f}"
        f" (n_valid={tm.n_valid})"
        f" | cost_per_query=${tm.mean_llm_cost_per_query_usd:.4f}"
    )


def _format_state_card(sc: StateCard) -> str:
    lines = [
        f"trial_number={sc.trial_number} trials_remaining={sc.trials_remaining}",
        f"best_score_so_far={sc.best_score_so_far:.3f} (trial {sc.best_trial_number})",
        f"last_trial_delta={sc.last_trial_delta:+.3f}",
    ]
    if sc.trial_summaries:
        lines.append("trial_summaries:")
        for t in sc.trial_summaries[-8:]:
            changes = t.get("what_changed_from_prev") or []
            modes = t.get("top_failure_modes") or []
            change_str = "; ".join(changes) if changes else "<initial>"
            mode_str = ", ".join(modes) if modes else "<none>"
            cost_usd = float(t.get("cost_usd", 0.0))
            lines.append(
                f"  - trial {t.get('trial_number')}: score={float(t.get('score', 0.0)):.3f}"
                f" cost=${cost_usd:.4f}/q"
                f" | changed: {change_str} | top_failure_modes: {mode_str}"
            )

    lines.extend(_format_strategy_block(sc))
    lines.extend(_format_pareto_block(sc))
    return "\n".join(lines)


def _format_strategy_block(sc: StateCard) -> list[str]:
    """Render the agent's own strategy carry-over.

    Shows the previous commitment verbatim, the trajectory of stances so the
    agent sees its own thrashing (or coherence), and the orchestrator-
    computed ``done_eligible`` gate. The ratchet rules are stated once in
    one line — the prompt re-states them; this is just a reminder.
    """
    lines: list[str] = ["", "## Strategy carry-over"]
    prev = sc.previous_strategy
    if prev is None:
        lines.append("previous_strategy: <none — this is trial 1 of the run>")
    else:
        anchor_str = f" anchor_trial={prev.anchor_trial}" if prev.anchor_trial is not None else ""
        lines.append(
            f"previous_strategy: stance={prev.stance} committed_at_trial={prev.committed_at_trial}"
            f" revisions_so_far={prev.revision_count}{anchor_str}"
        )
        lines.append(f"  intent: {prev.intent or '<empty>'}")
        if prev.journal:
            lines.append(f"  journal (rewrite each trial, ≤800 tokens):\n{prev.journal}")
        else:
            lines.append("  journal: <empty — write the first entry>")

    rec_line = _recommended_anchor_line(sc)
    if rec_line is not None:
        lines.append(rec_line)

    if sc.strategy_history_summary:
        lines.append("strategy trajectory:")
        for entry in sc.strategy_history_summary[-8:]:
            lines.append(
                f"  - trial {entry.get('trial_number')}: stance={entry.get('stance')}"
                f" revisions={entry.get('revision_count')} | intent: {entry.get('intent') or '<empty>'}"
            )

    done_str = "true" if sc.done_eligible else f"false ({sc.done_blocked_reason})"
    lines.append(f"done_eligible: {done_str}")
    lines.append(
        "ratchet: search → polish → done (one-way; polish → search allowed only with"
        " regression_reason AND the just-emitted diagnosis.regression_detected=true on a primary axis"
        " (score or acc_given_complete))."
    )
    return lines


def _recommended_anchor_line(sc: StateCard) -> str | None:
    """State the mechanical anchor (current Pareto knee) used for lever-effect
    deltas.

    The anchor is orchestrator-managed and tracks the run's score-per-cost
    knee. Surfaced here so the agent understands what reference frame the
    delta numbers it sees are computed against. The agent's emitted
    ``anchor_trial`` field is overridden at finalize time — there is no
    advisory mode; this is informational only.
    """
    target = sc.knee_trial_number if sc.knee_trial_number is not None else sc.best_trial_number
    if target is None or target == sc.trial_number:
        return None
    label = "knee" if target == sc.knee_trial_number else "best"
    return f"anchor_trial: trial {target} (current Pareto {label}, orchestrator-managed)"


def _format_pareto_block(sc: StateCard) -> list[str]:
    """Render the Pareto state — the load-bearing block for cost-aware reasoning.

    Renders three things the proposer needs to anchor on a specific frontier
    member: the non-dominated frontier with knee/best annotations, the FULL
    config of every frontier member (so a perturbation can name specific
    fields to change), and the nearest dominator of the current trial.
    """
    lines: list[str] = ["", "## Pareto state"]
    cheapest_str = "n/a"
    if sc.cheapest_at_score_threshold_usd is not None:
        cheapest_str = f"${sc.cheapest_at_score_threshold_usd:.4f}/q (trial {sc.cheapest_at_score_threshold_trial})"
    lines.append(
        f"hypervolume={sc.hypervolume:.4f} (Δ_last_3={sc.hypervolume_delta_last_3:+.4f})  "
        f"current_trial_cost=${sc.current_trial_cost_usd:.4f}/q  "
        f"cheapest_within_polish_band={cheapest_str}"
    )
    if not sc.pareto_frontier:
        lines.append("pareto_frontier: (no trials yet)")
        return lines

    knee = sc.knee_trial_number
    best = sc.best_trial_number
    lines.append(f"pareto_frontier ({len(sc.pareto_frontier)} non-dominated):")
    for entry in sc.pareto_frontier:
        tag_parts: list[str] = []
        tn = entry.get("trial_number")
        if tn == knee:
            tag_parts.append("★knee")
        if tn == best:
            tag_parts.append("★best")
        tag_str = "  " + " ".join(tag_parts) if tag_parts else ""
        lines.append(
            f"  - trial {tn}: score={float(entry.get('score', 0.0)):.3f}"
            f"  cost=${float(entry.get('cost_usd', 0.0)):.4f}/q{tag_str}"
            f"  | {entry.get('config_summary', '')}"
        )
    if sc.nearest_dominator_trial is not None:
        lines.append(
            f"nearest dominator of current trial: trial {sc.nearest_dominator_trial}"
            "  (full config in 'frontier_configs' below)"
        )

    full_lines = _format_frontier_full_configs(sc.pareto_frontier)
    if full_lines:
        lines.append("")
        lines.append("frontier_configs (anchor on these by trial_number):")
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


def _format_frontier_context(fc: FrontierContext) -> str:
    """Frontier-relative summary rendered into the diagnostic prompt.

    Empty frontier (first trial, or no dominator) renders one line so the
    diagnostic prompt stays well-formed regardless.
    """
    if fc.is_on_frontier and fc.nearest_dominator_trial is None:
        return "current trial is on the Pareto frontier (not dominated by any prior trial)."
    if fc.nearest_dominator_trial is None:
        return "no Pareto signal available (insufficient history)."
    score_gap = fc.score_gap_to_dominator if fc.score_gap_to_dominator is not None else 0.0
    cost_gap = fc.cost_gap_to_dominator_usd if fc.cost_gap_to_dominator_usd is not None else 0.0
    diff_block = (
        "\n  config diff (current → dominator):\n  - " + "\n  - ".join(fc.nearest_dominator_config_diff)
        if fc.nearest_dominator_config_diff
        else "\n  config diff (current → dominator): (no tracked-field differences)"
    )
    on_frontier_note = " (current trial is also on the frontier)" if fc.is_on_frontier else ""
    return (
        f"current trial dominated by trial {fc.nearest_dominator_trial}"
        f" (score={fc.nearest_dominator_score:.3f}, cost=${fc.nearest_dominator_cost_usd:.4f}/q)"
        f"{on_frontier_note}\n"
        f"  score gap: dominator is +{score_gap:.3f} above current"
        f" | cost gap: current is +${cost_gap:.4f}/q above dominator" + diff_block
    )


def _format_failure_attribution(fa: FailureAttribution) -> str:
    """Single-line render of failure attribution percentages."""
    return (
        f"retrieval={fa.retrieval:.2f} ranking={fa.ranking:.2f} "
        f"generation={fa.generation:.2f} composition={fa.composition:.2f}"
    )


def _format_bundle_effect(effect: BundleEffectDelta | None, *, anchor_label: str) -> str:
    """Render the trial-vs-knee bundle delta on four axes (score,
    acc_given_complete, retrieval_complete, cost).

    The anchor is orchestrator-managed (current Pareto knee); the agent does
    not choose it. When a single lever changed, the delta is cleanly
    attributable to that lever. When N>1 levers changed, the deltas reflect
    the *bundled* effect — they cannot be split per-lever from observation
    alone. The render makes that distinction explicit so the agent doesn't
    credit/blame any individual lever in a multi-change bundle.
    """
    if effect is None or not effect.changes:
        return f"(no lever changes vs. {anchor_label})"
    header = f"vs. {anchor_label}:\n  Δscore   Δacc|complete  Δrcomp   Δcost_usd"
    delta_row = (
        f"  {effect.score_delta:+.3f}   {effect.acc_given_complete_delta:+.3f}         "
        f"{effect.retrieval_complete_delta:+.3f}   {effect.cost_delta_usd:+.5f}"
    )
    if len(effect.changes) == 1:
        return f"{header}\n  {effect.changes[0]}\n{delta_row}"
    change_lines = "\n".join(f"    - {c}" for c in effect.changes)
    return (
        f"{header}\n"
        f"  bundle of {len(effect.changes)} levers changed (effect below is the BUNDLE, NOT per-lever):\n"
        f"{change_lines}\n"
        f"{delta_row}"
    )


def _attribution_is_empty(fa: FailureAttribution) -> bool:
    """All four stage fractions are zero (agent omitted or emitted defaults)."""
    return fa.retrieval == 0.0 and fa.ranking == 0.0 and fa.generation == 0.0 and fa.composition == 0.0


_REGRESSION_FP_TOLERANCE = 1e-9
_REGRESSION_CI_Z = 1.96


def _proportion_ci_halfwidth(p: float, n: int) -> float:
    """Half-width of the 95% confidence interval for a proportion estimate.

    Used to derive a noise-floor for the regression-threshold check on
    proportion-valued axes (score, acc_given_complete, retrieval_complete).
    The static config threshold and this variance term are combined via
    ``max(static, ci_halfwidth)`` so the effective check tolerates the
    observed noise floor at the current exam size while still respecting any
    explicit user override that is stricter than noise.
    """
    if n <= 0:
        return 0.0
    p = min(max(float(p), 0.0), 1.0)
    return _REGRESSION_CI_Z * math.sqrt(p * (1.0 - p) / float(n))


def _axis_regressed_vs_history(
    axis: str,
    current: TrialMetrics,
    history_records: list,
    threshold: float,
) -> bool:
    """Whether the current trial regresses on ``axis`` vs the best-so-far history.

    "Best" is max() for score / acc_given_complete / retrieval_complete and
    min() for cost. The effective threshold is
    ``max(threshold, 1.96 * sqrt(p*(1-p)/n))`` for proportion-valued axes,
    where ``p`` is the best baseline value and ``n`` is the relevant question
    count — so noise-sized drops at small exam sizes never register as
    regressions. Cost uses the static threshold (it's not a proportion).
    Empty history → no regression. A tiny FP tolerance keeps boundary cases
    from being rejected due to subtraction rounding.
    """
    if not history_records:
        return False

    if axis == "score":
        scored = [(float(getattr(r, "score", 0.0)), r) for r in history_records]
        best_p, best_rec = max(scored, key=lambda t: t[0])
        n_valid = _trial_n_valid(best_rec)
        ci = _proportion_ci_halfwidth(best_p, n_valid)
        effective = max(threshold, ci) - _REGRESSION_FP_TOLERANCE
        return (best_p - float(current.answer_accuracy)) >= effective

    if axis == "acc_given_complete":
        candidates = [
            (
                float(getattr(getattr(r, "trial_metrics", None), "answer_correct_given_complete_retrieval", 0.0)),
                r,
            )
            for r in history_records
            if getattr(r, "trial_metrics", None) is not None
        ]
        if not candidates:
            return False
        best_p, best_rec = max(candidates, key=lambda t: t[0])
        # n for this axis is questions with complete retrieval, not the full
        # exam — estimate as n_valid * retrieval_complete for the baseline.
        # When n_valid is unknown (0), n_complete is 0 and the CI half-width
        # falls back to 0, so only the static threshold applies.
        best_tm = best_rec.trial_metrics
        n_complete = int(round(_trial_n_valid(best_rec) * float(best_tm.retrieval_complete)))
        ci = _proportion_ci_halfwidth(best_p, n_complete)
        effective = max(threshold, ci) - _REGRESSION_FP_TOLERANCE
        return (best_p - float(current.answer_correct_given_complete_retrieval)) >= effective

    if axis == "retrieval_complete":
        candidates = [
            (float(getattr(getattr(r, "trial_metrics", None), "retrieval_complete", 0.0)), r)
            for r in history_records
            if getattr(r, "trial_metrics", None) is not None
        ]
        if not candidates:
            return False
        best_p, best_rec = max(candidates, key=lambda t: t[0])
        ci = _proportion_ci_halfwidth(best_p, _trial_n_valid(best_rec))
        effective = max(threshold, ci) - _REGRESSION_FP_TOLERANCE
        return (best_p - float(current.retrieval_complete)) >= effective

    if axis == "cost":
        prior_costs = [float(getattr(r, "mean_llm_cost_per_query_usd", 0.0)) for r in history_records]
        if not prior_costs:
            return False
        cheapest = min(prior_costs)
        effective = threshold - _REGRESSION_FP_TOLERANCE
        return (float(current.mean_llm_cost_per_query_usd) - cheapest) >= effective

    return False


def _trial_n_valid(record) -> int:
    """Extract the exam ``n_valid`` from a history record.

    Returns 0 when missing or zero — callers must treat 0 as "variance term
    unknown" and fall back to the static threshold alone rather than assume
    a specific exam size.
    """
    tm = getattr(record, "trial_metrics", None)
    if tm is None:
        return 0
    return max(0, int(getattr(tm, "n_valid", 0)))


def _format_diagnosis(d: Diagnosis) -> str:
    fa = d.failure_attribution
    lines = [
        f"trial_metrics: {_format_trial_metrics(d.trial_metrics)}",
        (
            f"failure_attribution: retrieval={fa.retrieval:.2f} ranking={fa.ranking:.2f} "
            f"generation={fa.generation:.2f} composition={fa.composition:.2f}"
        ),
    ]
    if d.confirmed_findings:
        lines.append("confirmed_findings:")
        lines.extend(f"  - {item}" for item in d.confirmed_findings)
    if d.open_questions:
        lines.append("open_questions:")
        lines.extend(f"  - {item}" for item in d.open_questions)
    if d.notable_deltas:
        lines.append("notable_deltas:")
        lines.extend(f"  - {item}" for item in d.notable_deltas)
    if d.regression_detected:
        axes_str = ", ".join(d.regression_axes) or "<unspecified>"
        lines.append(f"regression_detected: true (axes: {axes_str})")
    else:
        lines.append("regression_detected: false")
    if d.illustrative_qids:
        lines.append(f"illustrative_qids: {', '.join(d.illustrative_qids)}")
    lines.append(f"narrative: {d.narrative}")
    return "\n".join(lines)


_LEGAL_FORWARD_TRANSITIONS = frozenset(
    {
        ("search", "polish"),
        ("search", "done"),
        ("polish", "done"),
    }
)
_LEGAL_RETREAT_TRANSITIONS = frozenset({("polish", "search")})


_RETREAT_QUALIFYING_AXES = frozenset({"score", "acc_given_complete"})


def _validate_strategy_transition(
    *,
    previous: Strategy | None,
    proposed: Strategy,
    intended_trial: int,
    last_diagnosis: Diagnosis | None,
    state_card: StateCard,
    min_stance_lock_trials: int,
) -> None:
    """Enforce one-way ratchet, stance lock-in, and the done-eligibility gate.

    Raises ``ValueError`` with a precise reason on violation so the retry
    prompt can name the broken rule. Mutation of ``revision_count`` and
    ``committed_at_trial`` is handled by ``_finalize_strategy`` *after* this
    validation succeeds.

    Backward transition (``polish → search``) is permitted only when the
    just-emitted diagnosis flags ``regression_detected=True`` AND at least one
    listed regression axis is a qualifying primary axis (``score`` or
    ``acc_given_complete``). A retreat on cost or retrieval_complete alone
    isn't enough — the run-objective ratchet only unlocks when the answer
    score regressed.
    """
    if proposed.stance == "done" and not state_card.done_eligible:
        raise ValueError(
            f"strategy.stance='done' is not currently allowed: "
            f"{state_card.done_blocked_reason}. Continue the search with stance='search' "
            "or stance='polish' as appropriate."
        )

    if previous is not None and previous.stance == "done":
        raise ValueError("cannot transition out of stance='done' — it is terminal.")

    if previous is None:
        return

    if proposed.stance != previous.stance:
        earliest_transition_trial = previous.committed_at_trial + min_stance_lock_trials + 1
        if intended_trial < earliest_transition_trial:
            raise ValueError(
                f"stance lock: stance={previous.stance!r} was committed at trial "
                f"{previous.committed_at_trial}; lock-in {min_stance_lock_trials} trial(s) "
                f"means the earliest legal transition is at trial {earliest_transition_trial}. "
                f"You proposed transitioning at trial {intended_trial} — hold "
                f"stance={previous.stance!r} for this trial and revisit later."
            )

    if proposed.stance == previous.stance:
        return

    transition = (previous.stance, proposed.stance)
    if transition in _LEGAL_FORWARD_TRANSITIONS:
        return
    if transition in _LEGAL_RETREAT_TRANSITIONS:
        regression = bool(last_diagnosis is not None and last_diagnosis.regression_detected)
        axes = set(last_diagnosis.regression_axes) if last_diagnosis is not None else set()
        qualifying = bool(axes & _RETREAT_QUALIFYING_AXES)
        if not (regression and qualifying):
            raise ValueError(
                f"backward transition {previous.stance!r} → {proposed.stance!r} requires the "
                "just-emitted diagnosis.regression_detected=true AND at least one regression axis "
                f"in {sorted(_RETREAT_QUALIFYING_AXES)} "
                f"(got regression_detected={regression}, regression_axes={sorted(axes)}). "
                "If this trial was not a primary-axis regression, continue the current stance."
            )
        if not (proposed.regression_reason and proposed.regression_reason.strip()):
            raise ValueError(
                f"backward transition {previous.stance!r} → {proposed.stance!r} requires a "
                "non-empty strategy.regression_reason explaining what's being walked back."
            )
        return
    raise ValueError(
        f"illegal stance transition {previous.stance!r} → {proposed.stance!r}. "
        "Lattice: search → polish → done (one-way), with single-step retreat "
        "polish → search allowed only on regression."
    )


def _finalize_strategy(
    *,
    proposed: Strategy,
    previous: Strategy | None,
    intended_trial: int,
    effective_anchor: int | None,
) -> Strategy:
    """Set the orchestrator-managed fields (``committed_at_trial``, ``revision_count``,
    ``anchor_trial``).

    A stance transition resets ``committed_at_trial`` to the trial the new
    stance becomes active for. A same-stance proposal inherits the previous
    commitment timestamp. ``revision_count`` increments whenever stance OR
    intent changed; otherwise it persists, so an agent re-emitting an
    identical strategy doesn't inflate the count.

    ``anchor_trial`` is always set to ``effective_anchor`` — the current
    Pareto knee — overriding whatever the agent emitted. The anchor is the
    reference frame for lever-effect deltas and must track the run's best
    score-per-cost trade-off mechanically.
    """
    if previous is None:
        return proposed.model_copy(
            update={
                "committed_at_trial": intended_trial,
                "revision_count": 0,
                "anchor_trial": effective_anchor,
            }
        )
    if proposed.stance != previous.stance:
        return proposed.model_copy(
            update={
                "committed_at_trial": intended_trial,
                "revision_count": previous.revision_count + 1,
                "anchor_trial": effective_anchor,
            }
        )
    if proposed.intent.strip() != previous.intent.strip():
        return proposed.model_copy(
            update={
                "committed_at_trial": previous.committed_at_trial,
                "revision_count": previous.revision_count + 1,
                "anchor_trial": effective_anchor,
            }
        )
    return proposed.model_copy(
        update={
            "committed_at_trial": previous.committed_at_trial,
            "revision_count": previous.revision_count,
            "anchor_trial": effective_anchor,
        }
    )


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
