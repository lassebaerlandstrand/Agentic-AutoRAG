"""Two-stage reasoning agent for RAG optimization.

Stage 1 (diagnose): analyze why the current trial under-performed and emit a
structured ``Diagnosis`` (per-stage metrics, bottleneck, ranked interventions,
hypothesis check).

Stage 2 (propose): pick the next ``TrialConfig`` and emit a structured
``ProposalMeta`` (move type, primary lever, hypothesis, memo). Move-type lever
constraints are enforced at parse time with the same self-healing retry loop
used for search-space violations.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import litellm
import yaml

from agentic_autorag.config.knowledge_base import KnowledgeBase
from agentic_autorag.config.models import OpenEndedQuestion, ProjectConfig, TrialConfig
from agentic_autorag.examiner.evaluator import (
    _ERROR_SENTINEL,
    _PERMANENT_ERROR_SENTINEL,
    ExamResult,
    QuestionResult,
)
from agentic_autorag.optimizer.diagnosis import (
    METRIC_POLARITY,
    Diagnosis,
    MoveType,
    ProposalMeta,
    Stage,
    StageMetrics,
    StateCard,
)
from agentic_autorag.optimizer.history import HistoryLog
from agentic_autorag.optimizer.state import (
    PRIMARY_LEVERS,
    PRIMARY_LEVERS_BY_STAGE,
    REFINE_SMALL_STEPS,
    build_state_card,
    check_prior_hypothesis,
    compute_stage_metrics,
    prior_bottleneck,
    suggest_move_type,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

DIAGNOSTIC_PROMPT = (_PROMPTS_DIR / "diagnostic.txt").read_text(encoding="utf-8")
PROPOSAL_PROMPT = (_PROMPTS_DIR / "proposal.txt").read_text(encoding="utf-8")
INITIAL_PROPOSAL_PROMPT = (_PROMPTS_DIR / "initial_proposal.txt").read_text(encoding="utf-8")

MAX_RETRIES = 3
_MAX_FAILURE_SAMPLE = 15
_PROBE_MIN_DELTA = 0.03
_STRUCTURAL_LEVERS = frozenset({"index_type", "embedding_model", "llm_model"})

_ERROR_SENTINELS = (_ERROR_SENTINEL, _PERMANENT_ERROR_SENTINEL)


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

_GRAPH_DIAGNOSTIC_TYPES = """\
When a graph index is in use, additional levers exist: entity-focused retrieval
via ``graph_query_mode`` and ``graph_top_k``. Entity gaps or missing relationships
can be addressed by increasing ``graph_top_k`` or swapping the graph query mode.
"""


class ReasoningAgent:
    """Two-stage reasoning agent with structured Diagnosis → ProposalMeta hand-off.

    Uses the shared ``HistoryLog`` as the single source of truth for trial
    history. Pure functions in ``state.py`` pre-compute stage metrics, the
    hypothesis check, and the state card so the LLM's job shrinks to
    interpretation and selection.
    """

    def __init__(
        self,
        agent_model: str,
        config: ProjectConfig,
        history: HistoryLog,
        debug_prompts: bool = False,
        knowledge_base: KnowledgeBase | None = None,
    ) -> None:
        self.model = agent_model
        self.config = config
        self.history = history
        self.debug_prompts = debug_prompts
        self.knowledge_base = knowledge_base
        self._include_graph = config.uses_graph()
        self._reasoning_effort = self._resolve_reasoning_effort(agent_model, config.agent.optimizer_reasoning_effort)
        if self._reasoning_effort is not None:
            logger.info("Reasoning agent using reasoning_effort=%s on %s", self._reasoning_effort, agent_model)

    @staticmethod
    def _resolve_reasoning_effort(model: str, effort: str | None) -> str | None:
        """Return the effort value to pass to litellm, or None to omit the kwarg.

        LiteLLM's ``supports_reasoning`` lets us skip the kwarg for models that
        would reject it (e.g. gpt-4o, nova-lite). Unknown models default to
        passing the effort through — LiteLLM will ignore or warn as appropriate.
        """
        if not effort:
            return None
        try:
            supported = bool(litellm.supports_reasoning(model=model))
        except Exception:
            supported = True
        return effort if supported else None

    def _log_exchange(self, stage: str, prompt: str, response: str) -> None:
        """Write a formatted prompt/response block to run.log at DEBUG level."""
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
        )
        return await self._call_for_config_only(prompt, stage="Initial Proposer")

    async def analyze_and_propose(
        self,
        exam_result: ExamResult,
        exam_questions: list[OpenEndedQuestion],
        current_config: TrialConfig,
        trial_number: int,
        trials_remaining: int,
    ) -> tuple[StageMetrics, Diagnosis, TrialConfig, ProposalMeta]:
        """Run the two-stage loop. Returns the artefacts the caller records.

        - ``StageMetrics``: attach to the current trial's record.
        - ``Diagnosis``: attach to the current trial's record.
        - ``TrialConfig``: the next trial's config.
        - ``ProposalMeta``: attach to the current trial's record (it records
          the decision made after seeing this trial).
        """
        stage_metrics = compute_stage_metrics(exam_result, reranker_top_n=current_config.reranker_top_n)
        prev_meta, prev_metrics = self._previous_meta_and_metrics()
        hypothesis_check = check_prior_hypothesis(prev_meta, prev_metrics, stage_metrics)
        state_card = build_state_card(
            trial_number=trial_number,
            trials_remaining=trials_remaining,
            current_metrics=stage_metrics,
            current_score=exam_result.score,
            history_records=self.history.records,
        )

        diagnosis = await self._diagnose(
            exam_result=exam_result,
            exam_questions=exam_questions,
            current_config=current_config,
            stage_metrics=stage_metrics,
            hypothesis_check=hypothesis_check,
            state_card=state_card,
        )
        # Reconcile: if the Diagnoser overrode the mechanical bottleneck, recompute
        # the suggested move type against the Diagnoser's claim before the Proposer
        # sees the state card. Otherwise the Proposer reads a stale suggestion.
        state_card = self._reconcile_state_card(state_card, diagnosis, self.history.records)
        next_config, meta = await self._propose(
            diagnosis=diagnosis,
            current_config=current_config,
            state_card=state_card,
        )
        return stage_metrics, diagnosis, next_config, meta

    @staticmethod
    def _reconcile_state_card(state_card: StateCard, diagnosis: Diagnosis, history_records: list) -> StateCard:
        """Update the state card's bottleneck + suggested move to match the Diagnoser.

        ``bottleneck_stable`` must be recomputed against the Diagnoser's choice,
        not the mechanical one, or suggest_move_type reads a stale "stable" flag.
        """
        if diagnosis.bottleneck == state_card.current_bottleneck:
            return state_card
        prev = prior_bottleneck(history_records)
        new_bottleneck_stable = prev is not None and prev == diagnosis.bottleneck
        return state_card.model_copy(
            update={
                "current_bottleneck": diagnosis.bottleneck,
                "bottleneck_stable": new_bottleneck_stable,
                "suggested_move_type": suggest_move_type(
                    bottleneck=diagnosis.bottleneck,
                    bottleneck_stable=new_bottleneck_stable,
                    consecutive_non_improvements=state_card.consecutive_non_improvements,
                    last_trial_delta=state_card.last_trial_delta,
                    trials_remaining=state_card.trials_remaining,
                    interventions_tried=state_card.interventions_tried,
                ),
            }
        )

    def _previous_meta_and_metrics(self) -> tuple[ProposalMeta | None, StageMetrics | None]:
        """Read the most recent trial's meta+metrics from history (None if empty)."""
        if not self.history.records:
            return None, None
        last = sorted(self.history.records, key=lambda r: r.trial_number)[-1]
        return last.meta, last.stage_metrics

    async def _diagnose(
        self,
        *,
        exam_result: ExamResult,
        exam_questions: list[OpenEndedQuestion],
        current_config: TrialConfig,
        stage_metrics: StageMetrics,
        hypothesis_check,
        state_card: StateCard,
    ) -> Diagnosis:
        """Produce a structured ``Diagnosis`` from failed exam questions."""
        real_failures = [
            q for q in exam_result.question_results if not q.correct and q.generated_response not in _ERROR_SENTINELS
        ]
        # Retrieval misses the MCQ happened to answer correctly: the retriever
        # returned no chunks overlapping the source_fact, but the LLM guessed
        # right or leaned on parametric knowledge. These are real retrieval
        # failures — surface them with the same diagnostic detail as true
        # failures so the Diagnoser can weigh them properly.
        retrieval_miss_guesses = [q for q in exam_result.question_results if q.correct and not q.context_sufficient]
        n_errors = sum(
            1 for q in exam_result.question_results if not q.correct and q.generated_response in _ERROR_SENTINELS
        )
        # Prioritise real failures first, then fill with retrieval-miss
        # guesses up to the sample cap.
        sample = (real_failures + retrieval_miss_guesses)[:_MAX_FAILURE_SAMPLE]
        miss_ids = {q.question_id for q in retrieval_miss_guesses}
        tags = {
            q.question_id: "Retrieval-miss (correct by guess)" if q.question_id in miss_ids else "Failure"
            for q in sample
        }

        error_note = ""
        if n_errors:
            error_note = (
                f"\n\nNote: {n_errors} question(s) failed due to system errors"
                " (timeouts, API failures) and are excluded from this analysis."
            )

        question_by_id = {q.id: q for q in exam_questions}
        failed_questions = self._format_failures(sample, question_by_id, tags=tags) + error_note

        config_json = current_config.to_prompt_json(include_graph=self._include_graph)
        graph_diag = _GRAPH_DIAGNOSTIC_TYPES if self._include_graph else ""
        history_text = self.history.format_for_agent(last_n=self.config.agent.max_history_trials)
        applicable_levers = sorted(PRIMARY_LEVERS_BY_STAGE[stage_metrics.bottleneck()])
        prompt = DIAGNOSTIC_PROMPT.format(
            stage_metrics=_format_stage_metrics(stage_metrics),
            hypothesis_check=_format_hypothesis_check(hypothesis_check),
            state_card=_format_state_card(state_card),
            current_config=config_json,
            history_count=self.config.agent.max_history_trials,
            history=history_text,
            failed_questions=failed_questions,
            graph_diagnostic_types=graph_diag,
            applicable_levers_hint=", ".join(applicable_levers),
        )

        messages = [{"role": "user", "content": prompt}]
        raw = ""
        diagnosis: Diagnosis | None = None
        for attempt in range(MAX_RETRIES):
            try:
                raw = await self._llm_complete_messages(messages)
                diagnosis = self._build_diagnosis(
                    raw=raw,
                    stage_metrics=stage_metrics,
                    hypothesis_check=hypothesis_check,
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
                            " bottleneck / confidence / applicable_levers / narrative."
                        ),
                    }
                )

        if diagnosis is None:
            logger.error("Diagnoser returned unparseable output after %d attempts; falling back", MAX_RETRIES)
            diagnosis = Diagnosis(
                stage_metrics=stage_metrics,
                bottleneck=stage_metrics.bottleneck(),
                confidence="low",
                hypothesis_check=hypothesis_check,
                applicable_levers=applicable_levers,
                narrative=_extract_narrative(raw)[:300],
            )

        self._log_exchange("Diagnoser", prompt, raw)
        return diagnosis

    async def _propose(
        self,
        *,
        diagnosis: Diagnosis,
        current_config: TrialConfig,
        state_card: StateCard,
    ) -> tuple[TrialConfig, ProposalMeta]:
        """Produce the next (TrialConfig, ProposalMeta), enforcing move-type rules."""
        history_text = self.history.format_for_agent(last_n=self.config.agent.max_history_trials)

        prompt = PROPOSAL_PROMPT.format(
            diagnosis=_format_diagnosis(diagnosis),
            state_card=_format_state_card(state_card),
            current_config=current_config.to_prompt_json(include_graph=self._include_graph),
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
                self._log_exchange("Proposer", messages[-1]["content"], raw)
                yaml_dict = self._extract_yaml(raw)
                meta_dict = yaml_dict.pop("meta", None)
                if not isinstance(meta_dict, dict):
                    raise ValueError("proposal YAML must include a 'meta' dict")
                config = TrialConfig.model_validate(yaml_dict)

                violations = self.config.validate_trial(config)
                if violations:
                    raise ValueError("Search space violations:\n" + "\n".join(f"- {v}" for v in violations))

                meta = ProposalMeta.model_validate(meta_dict)
                self._validate_move(current_config, config, meta, state_card)
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
                                "the TrialConfig fields AND the `meta:` dict."
                            ),
                        }
                    )

        raise RuntimeError(f"Failed to get valid proposal after {MAX_RETRIES} attempts")

    def _build_diagnosis(
        self,
        *,
        raw: str,
        stage_metrics: StageMetrics,
        hypothesis_check,
    ) -> Diagnosis:
        """Pure parser: validate LLM output and merge in the authoritative mechanical fields.

        ``stage_metrics`` and ``hypothesis_check`` are computed outside the LLM and
        always win over any values the LLM might try to include in its YAML.
        ``applicable_levers`` is filtered to the mechanically-valid set for the
        chosen bottleneck so the Proposer can't anchor on stray lever names.
        """
        yaml_dict = self._extract_yaml(raw)
        bottleneck = Stage(yaml_dict.get("bottleneck") or stage_metrics.bottleneck().value)
        confidence = yaml_dict.get("confidence", "medium")
        narrative = yaml_dict.get("narrative") or _extract_narrative(raw)
        raw_levers = yaml_dict.get("applicable_levers") or []
        if not isinstance(raw_levers, list):
            raw_levers = []
        allowed = PRIMARY_LEVERS_BY_STAGE[bottleneck]
        applicable_levers = [str(item) for item in raw_levers if isinstance(item, str) and item in allowed]
        if not applicable_levers:
            applicable_levers = sorted(allowed)
        return Diagnosis(
            stage_metrics=stage_metrics,
            bottleneck=bottleneck,
            confidence=confidence,
            hypothesis_check=hypothesis_check,
            applicable_levers=applicable_levers,
            narrative=narrative,
        )

    async def _call_for_config_only(self, prompt: str, *, stage: str) -> TrialConfig:
        """Call LLM, extract a TrialConfig-shaped YAML, validate, retry on failure.

        Used for ``propose_initial`` where there's no diagnosis yet and we
        only need a starting TrialConfig (no ProposalMeta).
        """
        messages = [{"role": "user", "content": prompt}]
        raw = ""
        for attempt in range(MAX_RETRIES):
            try:
                raw = await self._llm_complete_messages(messages)
                self._log_exchange(stage, messages[-1]["content"], raw)
                yaml_dict = self._extract_yaml(raw)
                yaml_dict.pop("meta", None)  # tolerate but ignore for initial proposal
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

    async def _llm_complete_messages(self, messages: list[dict]) -> str:
        kwargs: dict = {"model": self.model, "messages": messages}
        if self._reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._reasoning_effort
        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content or ""

    def _validate_move(
        self,
        current: TrialConfig,
        proposed: TrialConfig,
        meta: ProposalMeta,
        state_card: StateCard,
    ) -> None:
        """Raise if the declared move type and actual change disagree.

        The error message is fed back to the LLM via the retry loop, so it
        must name the specific violation and the fix.
        """
        changed_primary = _changed_primary_levers(current, proposed)

        if meta.primary_lever and meta.primary_lever not in changed_primary:
            raise ValueError(
                f"meta.primary_lever='{meta.primary_lever}' but that field did not change between "
                f"current and proposed config. Either change that lever or pick a different primary_lever."
            )

        # Reject hypotheses that would be "confirmed" only by making the pipeline worse.
        # Each metric has a fixed polarity: +1 = higher is better, -1 = lower is better.
        # The Proposer's expected_delta must point in the improving direction for its
        # target_metric; otherwise the hypothesis-check loop would reward regressions.
        if meta.target_metric and meta.target_metric in METRIC_POLARITY and meta.expected_delta != 0.0:
            polarity = METRIC_POLARITY[meta.target_metric]
            if meta.expected_delta * polarity <= 0:
                direction = "decrease" if polarity == -1 else "increase"
                raise ValueError(
                    f"meta.expected_delta={meta.expected_delta:+.3f} on target_metric="
                    f"'{meta.target_metric}' predicts a regression: this metric improves when it "
                    f"{direction}s (polarity={polarity:+d}). Flip the sign so expected_delta describes "
                    f"an improvement, or pick a different target_metric."
                )

        if meta.move_type == MoveType.PROBE:
            if len(changed_primary) != 1:
                raise ValueError(
                    f"PROBE requires exactly 1 primary-lever change, got {len(changed_primary)}: "
                    f"{sorted(changed_primary)}. Narrow the change or declare REFINE/PIVOT/COMPOUND."
                )
            if abs(meta.expected_delta) < _PROBE_MIN_DELTA:
                raise ValueError(
                    f"PROBE requires |expected_delta| >= {_PROBE_MIN_DELTA}, got {meta.expected_delta}. "
                    "Either predict a meaningful effect or choose REFINE."
                )

        elif meta.move_type == MoveType.REFINE:
            if len(changed_primary) > 2:
                raise ValueError(
                    f"REFINE allows at most 2 primary-lever changes, got {len(changed_primary)}: "
                    f"{sorted(changed_primary)}."
                )
            # Discrete primary levers cannot change in REFINE
            for lever in changed_primary:
                if lever in _STRUCTURAL_LEVERS or lever in {"chunking_strategy", "reasoning", "reranker"}:
                    raise ValueError(
                        f"REFINE cannot change discrete primary lever '{lever}'. "
                        "Use PROBE or PIVOT for model/index/reranker swaps."
                    )
                if not _within_refine_step(lever, current, proposed):
                    raise ValueError(
                        f"REFINE requires '{lever}' to stay within its small-step bound "
                        f"({REFINE_SMALL_STEPS.get(lever, 'n/a')})."
                    )

        elif meta.move_type == MoveType.PIVOT:
            if not (changed_primary & _STRUCTURAL_LEVERS):
                raise ValueError(
                    f"PIVOT must change at least one structural lever "
                    f"({sorted(_STRUCTURAL_LEVERS)}); changed: {sorted(changed_primary)}."
                )

        elif meta.move_type == MoveType.COMPOUND:
            if len(changed_primary) < 2:
                raise ValueError(
                    f"COMPOUND requires >= 2 primary-lever changes, got {len(changed_primary)}: "
                    f"{sorted(changed_primary)}."
                )
            # COMPOUND requires each changed primary lever to have a confirmed entry
            # that moved to the *same concrete value* we're proposing now. Matching
            # on lever name alone would let the proposer smuggle in fresh choices.
            confirmed_values: dict[str, set[str]] = {}
            for lever, _from, to_val, verdict in state_card.interventions_tried:
                if verdict == "confirmed" and to_val:
                    confirmed_values.setdefault(lever, set()).add(to_val)
            unsupported: list[str] = []
            for lever in changed_primary:
                proposed_val = _lever_value_string(proposed, lever)
                if proposed_val not in confirmed_values.get(lever, set()):
                    unsupported.append(f"{lever}={proposed_val}")
            if unsupported:
                available = (
                    ", ".join(
                        f"{lever}={{{','.join(sorted(vals))}}}" for lever, vals in sorted(confirmed_values.items())
                    )
                    or "(none)"
                )
                raise ValueError(
                    f"COMPOUND requires each changed primary lever's proposed value to have been "
                    f"confirmed in a prior trial. Unsupported: {sorted(unsupported)}. Confirmed "
                    f"values so far: {available}."
                )

        elif meta.move_type == MoveType.REVERT:
            if meta.revert_to_trial is None:
                raise ValueError("REVERT requires meta.revert_to_trial to reference a prior trial_number.")
            baseline = next(
                (r for r in self.history.records if r.trial_number == meta.revert_to_trial),
                None,
            )
            if baseline is None:
                raise ValueError(f"REVERT references trial {meta.revert_to_trial} but no such record in history.")
            changed_vs_baseline = _changed_primary_levers(baseline.config, proposed)
            if len(changed_vs_baseline) != 1:
                raise ValueError(
                    f"REVERT must change exactly 1 primary lever vs trial {meta.revert_to_trial}; "
                    f"changed: {sorted(changed_vs_baseline)}."
                )

    def _kb_text(self) -> str:
        """Return formatted knowledge base text, or empty string if not available."""
        if self.knowledge_base is None:
            return ""
        ss = self.config.search_space
        reasoning_allowed = {m: ss.is_reasoning_allowed(m) for m in ss.llm_models}
        return self.knowledge_base.format_for_prompt(
            llm_models=ss.llm_models,
            embedding_models=ss.embedding_models,
            reranker_models=ss.reranker.models,
            reasoning_allowed=reasoning_allowed,
            include_graph=self._include_graph,
        )

    @staticmethod
    def _extract_yaml(text: str) -> dict:
        """Extract a YAML block from agent response text."""
        match = re.search(r"```ya?ml\n(.*?)```", text, re.DOTALL)
        if not match:
            match = re.search(r"```\n(.*?)```", text, re.DOTALL)
        if not match:
            raise ValueError("No YAML block found in agent response")
        parsed = yaml.safe_load(match.group(1))
        if not isinstance(parsed, dict):
            raise ValueError("YAML block must be a mapping")
        return parsed

    @staticmethod
    def _format_failures(
        failures: list[QuestionResult],
        questions_by_id: dict[str, OpenEndedQuestion],
        max_context_chars: int = 0,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Format failed questions as readable blocks for the diagnostic prompt.

        Each block includes the question text, options, the *ground-truth* source
        document(s), how many retrieved chunks came from those docs, and the usual
        retrieval-quality stats. The distinct-docs list flags Frankenstein
        retrieval; the ground-truth coverage flags wrong-doc retrieval.

        ``tags`` maps question_id to a header label (e.g. ``"Failure"`` or
        ``"Retrieval-miss (correct by guess)"``). Missing entries default to
        ``"Failure"``. The rendered header is ``### {tag} {i}``.
        """
        blocks: list[str] = []
        tags = tags or {}
        for i, qr in enumerate(failures, 1):
            context = qr.retrieved_context
            if max_context_chars and len(context) > max_context_chars:
                context = context[:max_context_chars] + "\n[...truncated]"
            q = questions_by_id.get(qr.question_id)
            question_text = q.question if q else "<question text unavailable>"
            gold_block = ""
            if q:
                gold_lines = [f"  canonical: {q.canonical_answer}"]
                if q.answer_variants:
                    gold_lines.append(f"  variants: {q.answer_variants}")
                gold_lines.append(f"  bridge_entity: {q.bridge_entity}")
                gold_block = "\n".join(gold_lines)
            spans_block = ""
            if q:
                spans_block = (
                    f"  span_A (doc={q.source_doc_ids[0] if q.source_doc_ids else '?'}): {q.source_span_A}\n"
                    f"  span_B (doc={q.source_doc_ids[1] if len(q.source_doc_ids) > 1 else '?'}): {q.source_span_B}"
                )
            source_doc_ids = list(q.source_doc_ids) if q and q.source_doc_ids else []
            source_docs_text = ", ".join(source_doc_ids) if source_doc_ids else "<unknown>"
            unique_docs = sorted({d for d in qr.retrieved_doc_ids if d})
            doc_summary = ", ".join(unique_docs) if unique_docs else "<unknown>"
            gt_set = set(source_doc_ids)
            n_retrieved = len(qr.retrieved_doc_ids)
            gt_hits = sum(1 for d in qr.retrieved_doc_ids if d in gt_set) if gt_set else 0
            gt_coverage = f"{gt_hits}/{n_retrieved}" if n_retrieved else "0/0"
            tag = tags.get(qr.question_id, "Failure")
            em = getattr(qr, "em", 0.0)
            f1 = getattr(qr, "f1", 0.0)
            block = (
                f"### {tag} {i}\n"
                f"Question ID: {qr.question_id}\n"
                f"Question: {question_text}\n"
                f"Gold answer:\n{gold_block or '  <unavailable>'}\n"
                f"Predicted answer: {qr.selected_answer}\n"
                f"Score: em={em:.0f} f1={f1:.2f} correct={qr.correct}\n"
                f"Retrieval quality: context_sufficient={qr.context_sufficient}"
                f" chunk_precision={qr.chunk_precision:.2f}\n"
                f"Source span rank: {qr.source_fact_rank}"
                f" (MRR: {1.0 / qr.source_fact_rank if qr.source_fact_rank else 0.0:.2f})\n"
                f"Source spans:\n{spans_block or '  <unavailable>'}\n"
                f"Ground-truth source doc(s): {source_docs_text}\n"
                f"Ground-truth coverage: {gt_coverage} retrieved chunks from source docs\n"
                f"Retrieved chunks from {len(unique_docs)} distinct document(s): {doc_summary}\n"
                f"Generated response: {qr.generated_response}\n"
                f"Retrieved context:\n{context}\n"
            )
            blocks.append(block)
        return "\n".join(blocks)


def _changed_primary_levers(a: TrialConfig, b: TrialConfig) -> set[str]:
    """Names of primary levers whose values differ between *a* and *b*."""
    changed: set[str] = set()
    for lever in PRIMARY_LEVERS:
        va = getattr(a, lever, None)
        vb = getattr(b, lever, None)
        # index_type is an enum; compare values
        if hasattr(va, "value"):
            va = va.value
        if hasattr(vb, "value"):
            vb = vb.value
        if va != vb:
            changed.add(lever)
    return changed


def _within_refine_step(lever: str, current: TrialConfig, proposed: TrialConfig) -> bool:
    """True when a REFINE move's numeric-lever change respects the small-step bound."""
    step = REFINE_SMALL_STEPS.get(lever)
    if step is None:
        return False
    cur = getattr(current, lever)
    new = getattr(proposed, lever)
    if lever == "chunk_token_size":
        # relative bound
        if cur == 0:
            return new == 0
        return abs(new - cur) / cur <= step
    return abs(new - cur) <= step


def _format_stage_metrics(sm: StageMetrics) -> str:
    return (
        f"retrieval_success={sm.retrieval_success:.3f}"
        f" | ranking_quality={sm.ranking_quality:.3f}"
        f" | gold_in_reranker_window={sm.gold_in_reranker_window:.3f}"
        f" | generation_given_context={sm.generation_given_context:.3f}"
        f" (n_eligible={sm.n_eligible_for_generation})"
    )


def _format_hypothesis_check(hc) -> str:
    if hc is None or hc.verdict == "n/a":
        return "verdict=n/a (first trial or no prior hypothesis)"
    exp = f"{hc.expected_delta:+.3f}" if hc.expected_delta is not None else "n/a"
    obs = f"{hc.observed_delta:+.3f}" if hc.observed_delta is not None else "n/a"
    return (
        f"prior_hypothesis={hc.prior_hypothesis!r}"
        f" target_metric={hc.target_metric}"
        f" expected_delta={exp}"
        f" observed_delta={obs}"
        f" verdict={hc.verdict}"
    )


def _format_state_card(sc: StateCard) -> str:
    lines = [
        f"trial_number={sc.trial_number} trials_remaining={sc.trials_remaining}",
        f"best_score_so_far={sc.best_score_so_far:.3f} (trial {sc.best_trial_number})",
        f"last_trial_delta={sc.last_trial_delta:+.3f}",
        f"consecutive_non_improvements={sc.consecutive_non_improvements}",
        f"current_bottleneck={sc.current_bottleneck.value} (stable={'yes' if sc.bottleneck_stable else 'no'})",
        f"suggested_move_type={sc.suggested_move_type.value}",
    ]
    if sc.interventions_tried:
        lines.append("interventions_tried:")
        for lever, value_from, value_to, verdict in sc.interventions_tried[-8:]:
            transition = f"{value_from or '?'} → {value_to or '?'}"
            lines.append(f"  - {lever}: {transition} [{verdict}]")
    if sc.top_trials:
        lines.append("top_trials_so_far:")
        for t in sc.top_trials:
            lines.append(f"  - trial {t['trial_number']}: score={t['score']:.3f}")
    return "\n".join(lines)


def _format_diagnosis(d: Diagnosis) -> str:
    levers = ", ".join(d.applicable_levers) if d.applicable_levers else "(none provided)"
    return "\n".join(
        [
            f"bottleneck={d.bottleneck.value} confidence={d.confidence}",
            f"stage_metrics: {_format_stage_metrics(d.stage_metrics)}",
            f"hypothesis_check: {_format_hypothesis_check(d.hypothesis_check)}",
            f"narrative: {d.narrative}",
            f"applicable_levers: {levers}",
        ]
    )


def _lever_value_string(config: TrialConfig, lever: str) -> str:
    """Stringify a config lever value, unwrapping enums — shared with state.py."""
    raw = getattr(config, lever, None)
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw))


def _extract_narrative(text: str) -> str:
    """Return prose prior to the first ``` fence as the narrative fallback."""
    idx = text.find("```")
    return text[:idx].strip() if idx > 0 else text.strip()
