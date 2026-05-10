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
import re
from pathlib import Path

import litellm
import yaml

from agentic_autorag.config.knowledge_base import KnowledgeBase
from agentic_autorag.config.models import OpenEndedQuestion, ProjectConfig, TrialConfig
from agentic_autorag.examiner._errors import ERROR_SENTINELS
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.litellm_runtime import acompletion_with_cost
from agentic_autorag.optimizer.diagnosis import (
    Bottleneck,
    Diagnosis,
    FrontierContext,
    ProposalMeta,
    StateCard,
    TrialMetrics,
)
from agentic_autorag.optimizer.history import HistoryLog
from agentic_autorag.optimizer.state import build_frontier_context, build_state_card, compute_trial_metrics

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

DIAGNOSTIC_PROMPT = (_PROMPTS_DIR / "diagnostic.txt").read_text(encoding="utf-8")
PROPOSAL_PROMPT = (_PROMPTS_DIR / "proposal.txt").read_text(encoding="utf-8")
INITIAL_PROPOSAL_PROMPT = (_PROMPTS_DIR / "initial_proposal.txt").read_text(encoding="utf-8")
FAILURE_RECOVERY_PROMPT = (_PROMPTS_DIR / "failure_recovery.txt").read_text(encoding="utf-8")

MAX_RETRIES = 3
_MAX_FAILURE_SAMPLE = 15


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
    ) -> tuple[TrialMetrics, Diagnosis, TrialConfig, ProposalMeta]:
        """Diagnose the current trial, then propose the next config.

        Returns ``(trial_metrics, diagnosis, next_config, proposal_meta)``.
        ``trial_metrics`` and ``diagnosis`` describe the just-completed trial;
        ``next_config`` and ``proposal_meta`` describe the next one.
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
        )

        top_modes = [b.stage for b in diagnosis.bottlenecks[:2]]
        state_card = build_state_card(
            trial_number=trial_number,
            trials_remaining=trials_remaining,
            current_score=exam_result.score,
            history_records=self.history.records,
            max_trials=trial_number + trials_remaining,
            current_config=current_config,
            current_top_failure_modes=top_modes,
            current_cost_usd=exam_result.mean_llm_cost_per_query_usd,
            polish_fraction=self.config.meta.polish_fraction,
            polish_score_floor=self.config.meta.polish_score_floor,
            polish_score_tolerance=self.config.meta.polish_score_tolerance,
        )

        next_config, meta = await self._propose(
            diagnosis=diagnosis,
            current_config=current_config,
            state_card=state_card,
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
    ) -> Diagnosis:
        """Produce a structured ``Diagnosis`` from failed exam questions."""
        real_failures = [
            q for q in exam_result.question_results if not q.correct and q.generated_response not in ERROR_SENTINELS
        ]
        n_errors = sum(
            1 for q in exam_result.question_results if not q.correct and q.generated_response in ERROR_SENTINELS
        )
        sample = real_failures[:_MAX_FAILURE_SAMPLE]
        tags = {q.question_id: _failure_mode(q) for q in sample}

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
            failed_questions=failed_questions,
            graph_diagnostic_types=graph_diag,
            frontier_signal=_format_frontier_context(frontier_context),
        )

        messages = [{"role": "user", "content": prompt}]
        raw = ""
        diagnosis: Diagnosis | None = None
        for attempt in range(MAX_RETRIES):
            try:
                raw = await self._llm_complete_messages(messages)
                diagnosis = self._build_diagnosis(raw=raw, trial_metrics=trial_metrics)
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
                            " a `bottlenecks` list and a `narrative` string."
                        ),
                    }
                )

        if diagnosis is None:
            logger.error("Diagnoser returned unparseable output after %d attempts; falling back", MAX_RETRIES)
            diagnosis = Diagnosis(
                trial_metrics=trial_metrics,
                bottlenecks=[],
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
        """Produce the next (TrialConfig, ProposalMeta)."""
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
                                "the TrialConfig fields AND the `meta:` dict (changes/rationale/memo)."
                            ),
                        }
                    )

        raise RuntimeError(f"Failed to get valid proposal after {MAX_RETRIES} attempts")

    def _build_diagnosis(self, *, raw: str, trial_metrics: TrialMetrics) -> Diagnosis:
        """Parse the diagnoser's YAML and merge in mechanical trial_metrics."""
        yaml_dict = self._extract_yaml(raw)
        narrative = yaml_dict.get("narrative") or _extract_narrative(raw)
        raw_bots = yaml_dict.get("bottlenecks") or []
        bottlenecks: list[Bottleneck] = []
        if isinstance(raw_bots, list):
            for item in raw_bots:
                if isinstance(item, dict):
                    bottlenecks.append(Bottleneck.model_validate(item))
        return Diagnosis(trial_metrics=trial_metrics, bottlenecks=bottlenecks, narrative=narrative)

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
        tags: dict[str, str] | None = None,
    ) -> str:
        """Format failed questions as readable blocks for the diagnostic prompt.

        Each block is headed with the question's failure_mode tag. The
        retrieval-status / refused / failure_mode line is the load-bearing
        per-question diagnostic for the open-ended setup.
        """
        blocks: list[str] = []
        tags = tags or {}
        for i, qr in enumerate(failures, 1):
            context = qr.retrieved_context
            q = questions_by_id.get(qr.question_id)
            question_text = q.question if q else "<question text unavailable>"
            gold_block = ""
            if q:
                gold_lines = [f"  canonical: {q.canonical_answer}"]
                if q.answer_variants:
                    gold_lines.append(f"  variants: {q.answer_variants}")
                gold_block = "\n".join(gold_lines)
            spans_block = ""
            if q:
                span_lines: list[str] = []
                for span_idx, (span, doc_id) in enumerate(
                    zip(q.source_spans, q.source_doc_ids, strict=True),
                    start=1,
                ):
                    span_lines.append(f"  span_{span_idx} (doc={doc_id}): {span}")
                spans_block = "\n".join(span_lines)
            source_doc_ids = list(q.source_doc_ids) if q and q.source_doc_ids else []
            source_docs_text = ", ".join(source_doc_ids) if source_doc_ids else "<unknown>"
            unique_docs = sorted({d for d in qr.retrieved_doc_ids if d})
            doc_summary = ", ".join(unique_docs) if unique_docs else "<unknown>"
            gt_set = set(source_doc_ids)
            n_retrieved = len(qr.retrieved_doc_ids)
            gt_hits = sum(1 for d in qr.retrieved_doc_ids if d in gt_set) if gt_set else 0
            gt_coverage = f"{gt_hits}/{n_retrieved}" if n_retrieved else "0/0"
            tag = tags.get(qr.question_id, "generation_wrong")
            em = getattr(qr, "em", 0.0)
            f1 = getattr(qr, "f1", 0.0)
            block = (
                f"### {tag} {i}\n"
                f"Question ID: {qr.question_id}\n"
                f"Question: {question_text}\n"
                f"Gold answer:\n{gold_block or '  <unavailable>'}\n"
                f"Predicted answer: {qr.selected_answer}\n"
                f"Score: em={em:.0f} f1={f1:.2f} correct={qr.correct}\n"
                f"Retrieval status: {qr.retrieval_status} | refused: {qr.refused} | failure_mode: {tag}\n"
                f"chunk_precision={qr.chunk_precision:.2f}"
                f" source_span_rank={qr.source_fact_rank}"
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
        f"trial_number={sc.trial_number} trials_remaining={sc.trials_remaining} phase={sc.phase}",
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

    lines.extend(_format_pareto_block(sc))
    return "\n".join(lines)


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
            f" | llm={cfg.get('llm_model')} temp={cfg.get('temperature')}"
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


def _format_diagnosis(d: Diagnosis) -> str:
    if d.bottlenecks:
        bot_lines = ["bottlenecks:"]
        for b in d.bottlenecks:
            bot_lines.append(f"  - {b.stage} ({b.severity}): {b.evidence}")
        bot_block = "\n".join(bot_lines)
    else:
        bot_block = "bottlenecks: (none reported)"
    return "\n".join(
        [
            f"trial_metrics: {_format_trial_metrics(d.trial_metrics)}",
            bot_block,
            f"narrative: {d.narrative}",
        ]
    )


def _extract_narrative(text: str) -> str:
    """Return prose prior to the first ``` fence as the narrative fallback."""
    idx = text.find("```")
    return text[:idx].strip() if idx > 0 else text.strip()


def _format_failure_history(failures: list[tuple[TrialConfig, str]]) -> str:
    """Render past failed (config, error) pairs as a deduped, human-readable list."""
    if not failures:
        return "(none yet)"
    lines: list[str] = []
    for i, (cfg, err) in enumerate(failures, 1):
        idx = getattr(cfg.index_type, "value", cfg.index_type)
        summary = (
            f"  - failure {i}: reranker={cfg.reranker} embed={cfg.embedding_model}"
            f" llm={cfg.llm_model} index={idx} chunk={cfg.chunk_token_size}"
            f" top_k={cfg.top_k}"
        )
        first_line = err.strip().splitlines()[0] if err else "<no message>"
        lines.append(summary + f"\n    error: {first_line}")
    return "\n".join(lines)
