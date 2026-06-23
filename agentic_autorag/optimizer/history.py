"""Trial history — JSONL persistence for optimization trial records."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from agentic_autorag.config.models import IndexType, TrialConfig
from agentic_autorag.examiner.evaluator import QuestionResult
from agentic_autorag.optimizer.diagnosis import Diagnosis, ProposalMeta, TrialMetrics

logger = logging.getLogger(__name__)

# Index types that USE graph retrieval — fields like graph_query_mode and
# graph_top_k only render meaningfully for these; everything else gets ``n/a``.
_GRAPH_INDEX_VALUES = frozenset({IndexType.GRAPH_ONLY.value, IndexType.HYBRID_GRAPH_VECTOR.value})

# Trials rendered in FULL detail in the agent history view. Older trials outside
# the keep-set collapse to a single line in the complete "configs already tried"
# index, bounding agent-prompt growth to O(keep-set) instead of O(trials) so the
# loop scales to long runs. The full-detail set is the most recent
# ``_RECENT_FULL_WINDOW`` trials plus the best-accuracy trial, every Pareto
# frontier member, and the ``_TOP_MOVERS_FULL`` largest accuracy movements.
_RECENT_FULL_WINDOW = 8
_TOP_MOVERS_FULL = 2

# Most-recent trials whose per-trial failure cross-tab is shown to the
# Diagnoser so it can narrate failure-mode migration without the full
# per-trial config/cost history (which the Diagnoser cannot cite).
_DIAGNOSER_CROSSTAB_WINDOW = 3


def _fmt_per_stage_llm(c: TrialConfig) -> str:
    """Compact per-stage LLM string — collapses when every active stage uses
    the same LLM. Used by trial summary/history renderers."""
    parts = {"gen": c.generator_llm, "comp": c.compressor_llm, "exp": c.expander_llm}
    active = [v for v in parts.values() if v is not None]
    if active and all(v == active[0] for v in active):
        return active[0]
    return "|".join(f"{k}:{v if v is not None else 'null'}" for k, v in parts.items())


@dataclass
class TrialRecord:
    """A single optimization trial result with JSON serialization.

    ``diagnosis`` is the structured output of the Diagnoser for this trial.
    ``meta`` is the structured output of the Proposer call that *produced*
    this trial's config (changes, rationale, strategy). It is None for the
    initial trial (the seed config has no preceding Proposer call) and for
    trials whose config was reused after a Proposer crash.
    """

    trial_number: int
    config: TrialConfig
    question_results: list[QuestionResult]
    timestamp: datetime = field(default_factory=datetime.now)
    answer_accuracy: float = 0.0
    mean_retrieval_quality: float = 0.0
    n_em_correct: int = 0
    n_judge_correct: int = 0
    n_judge_rejected: int = 0
    n_judge_no_answer: int = 0
    n_judge_failed: int = 0
    n_no_answer: int = 0
    n_judge_calls: int = 0
    mean_em: float = 0.0
    mean_f1: float = 0.0
    mean_llm_cost_per_query_usd: float = 0.0
    total_llm_cost_usd: float = 0.0
    mean_prompt_tokens: float = 0.0
    mean_completion_tokens: float = 0.0
    # Totals over this trial's duration sourced from the cost-ledger delta
    # (every bucket, including embedding_build credits and agent_proposal
    # tokens). Reconciles with ``sum(trial.total_*_tokens for trial in history)``
    # matching the run-level ledger up to first-trial setup activity.
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_embedding_tokens: int = 0
    is_pareto_optimal: bool = False
    trial_metrics: TrialMetrics | None = None
    diagnosis: Diagnosis | None = None
    meta: ProposalMeta | None = None
    cross_tab_snapshot: str = ""

    def summary(self) -> str:
        """One-line summary for agent context."""
        c = self.config
        reasoning_tag = " +reasoning" if c.reasoning else ""
        verdict = f"EM={self.n_em_correct}, judge=yes:{self.n_judge_correct}/no:{self.n_judge_rejected}"
        cost_tag = f" cost=${self.mean_llm_cost_per_query_usd:.4f}/q"
        return (
            f"Trial {self.trial_number}: "
            f"acc={self.answer_accuracy:.3f}{cost_tag} ({verdict}), "
            f"rq={self.mean_retrieval_quality:.3f} | "
            f"chunk={c.chunk_token_size}, "
            f"embed={c.embedding_model}, "
            f"index={c.index_type.value}, "
            f"top_k={c.top_k}, "
            f"reranker={c.reranker}, "
            f"llm={_fmt_per_stage_llm(c)}{reasoning_tag}"
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "trial_number": self.trial_number,
            "config": self.config.model_dump(mode="json"),
            "question_results": [qr.model_dump(mode="json") for qr in self.question_results],
            "timestamp": self.timestamp.isoformat(),
            "answer_accuracy": self.answer_accuracy,
            "mean_retrieval_quality": self.mean_retrieval_quality,
            "n_em_correct": self.n_em_correct,
            "n_judge_correct": self.n_judge_correct,
            "n_judge_rejected": self.n_judge_rejected,
            "n_judge_no_answer": self.n_judge_no_answer,
            "n_judge_failed": self.n_judge_failed,
            "n_no_answer": self.n_no_answer,
            "n_judge_calls": self.n_judge_calls,
            "mean_em": self.mean_em,
            "mean_f1": self.mean_f1,
            "mean_llm_cost_per_query_usd": self.mean_llm_cost_per_query_usd,
            "total_llm_cost_usd": self.total_llm_cost_usd,
            "mean_prompt_tokens": self.mean_prompt_tokens,
            "mean_completion_tokens": self.mean_completion_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_embedding_tokens": self.total_embedding_tokens,
            "is_pareto_optimal": self.is_pareto_optimal,
            "trial_metrics": self.trial_metrics.model_dump(mode="json") if self.trial_metrics else None,
            "diagnosis": self.diagnosis.model_dump(mode="json") if self.diagnosis else None,
            "meta": self.meta.model_dump(mode="json") if self.meta else None,
            "cross_tab_snapshot": self.cross_tab_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrialRecord:
        """Reconstruct a TrialRecord from a stored dict."""
        tm = data.get("trial_metrics")
        diag = data.get("diagnosis")
        meta = data.get("meta")
        return cls(
            trial_number=data["trial_number"],
            config=TrialConfig.model_validate(data["config"]),
            question_results=[QuestionResult.model_validate(qr) for qr in data["question_results"]],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            answer_accuracy=data.get("answer_accuracy", 0.0),
            mean_retrieval_quality=data.get("mean_retrieval_quality", 0.0),
            n_em_correct=data.get("n_em_correct", 0),
            n_judge_correct=data.get("n_judge_correct", 0),
            n_judge_rejected=data.get("n_judge_rejected", 0),
            n_judge_no_answer=data.get("n_judge_no_answer", 0),
            n_judge_failed=data.get("n_judge_failed", 0),
            n_no_answer=data.get("n_no_answer", 0),
            n_judge_calls=data.get("n_judge_calls", 0),
            mean_em=data.get("mean_em", 0.0),
            mean_f1=data.get("mean_f1", 0.0),
            mean_llm_cost_per_query_usd=data.get("mean_llm_cost_per_query_usd", 0.0),
            total_llm_cost_usd=data.get("total_llm_cost_usd", 0.0),
            mean_prompt_tokens=data.get("mean_prompt_tokens", 0.0),
            mean_completion_tokens=data.get("mean_completion_tokens", 0.0),
            total_prompt_tokens=int(data.get("total_prompt_tokens", 0)),
            total_completion_tokens=int(data.get("total_completion_tokens", 0)),
            total_embedding_tokens=int(data.get("total_embedding_tokens", 0)),
            is_pareto_optimal=bool(data.get("is_pareto_optimal", False)),
            trial_metrics=TrialMetrics.model_validate(tm) if tm else None,
            diagnosis=Diagnosis.model_validate(diag) if diag else None,
            meta=ProposalMeta.model_validate(meta) if meta else None,
            cross_tab_snapshot=data.get("cross_tab_snapshot", ""),
        )


class HistoryLog:
    """Persistent trial history stored as JSONL."""

    def __init__(self, path: str = "./experiments/history.jsonl", *, load_existing: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[TrialRecord] = []
        if load_existing:
            self._load_existing()

    def _load_existing(self) -> None:
        """Load existing records from the JSONL file if it exists."""
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self.records.append(TrialRecord.from_dict(data))
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(
                        "Skipping malformed record on line %d (%s: %s)",
                        line_num,
                        type(e).__name__,
                        e,
                    )

    def clear(self) -> None:
        """Remove all records and truncate the backing file.

        Called at the start of a new optimization run so the agent
        never sees stale trials from a previous run.
        """
        self.records.clear()
        if self.path.exists():
            self.path.unlink()

    def add(self, record: TrialRecord) -> None:
        """Append a record to in-memory list and persist to JSONL.

        After writing the full record to disk, large string fields are cleared
        from the in-memory copy to reduce RAM usage. The JSONL file retains
        the complete data; in-memory consumers only need question_id and correct.
        """
        self.records.append(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        for qr in record.question_results:
            qr.retrieved_context = ""
            qr.generated_response = ""
            qr.retrieved_chunks = []
            qr.chunk_satisfies_spans = []

    def get_best(self) -> TrialRecord | None:
        """Return the record with the highest accuracy, or None if empty."""
        if not self.records:
            return None
        return max(self.records, key=lambda r: r.answer_accuracy)

    def rewrite_all(self) -> None:
        """Truncate the JSONL file and rewrite every in-memory record.

        Used when a flag computed over the whole history changes
        (``is_pareto_optimal``); rewriting is cheap at 10–30 trials and keeps
        the on-disk record consistent with the in-memory truth.
        """
        with open(self.path, "w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict()) + "\n")

    def recompute_pareto_flags(self) -> None:
        """Recompute ``is_pareto_optimal`` on every record from current scores+costs.

        Local import of ``pareto`` to avoid an import cycle (pareto reads
        ``TrialRecord`` attributes via Protocol, doesn't import this module).
        """
        from agentic_autorag.optimizer import pareto

        if not self.records:
            return
        frontier = pareto.compute_frontier(list(self.records))
        frontier_ids = {int(r.trial_number) for r in frontier}
        for record in self.records:
            record.is_pareto_optimal = int(record.trial_number) in frontier_ids

    def format_for_agent(
        self,
        *,
        current_trial: TrialRecord | None = None,
        recent_window: int = _RECENT_FULL_WINDOW,
        show_cost: bool = True,
    ) -> str:
        """Format trial history as structured text for the Proposer prompt.

        Trials in the full-detail keep-set (see ``_full_detail_trials``) render
        ALL ``TrialConfig`` fields and the full mechanical metric set (verdict
        breakdown, retrieval rates, retrieval/EM/F1 quality) so the Proposer
        can do its own cross-trial aggregation from raw data. Older trials
        outside the keep-set are represented only by their line in the complete
        "configs already tried" index, which keeps the Proposer's no-repeat
        awareness intact while bounding prompt growth on long runs.

        When ``show_cost`` is False (score-only runs), cost/token columns and
        the Pareto-frontier tag are dropped — cost is not an objective there,
        so the figures are noise. The Diagnoser uses ``format_for_diagnoser``
        instead and never sees this view.

        ``current_trial`` is a synthetic preview record for the just-completed
        trial that is not yet in ``self.records`` (the orchestrator persists it
        after the Proposer returns). When supplied it is appended as the last
        block so the Proposer sees its full detail alongside the prior trials.
        """
        all_records = [*self.records, current_trial] if current_trial is not None else list(self.records)
        if not all_records:
            return "No previous trials."

        best_trial: int = max(all_records, key=lambda r: r.answer_accuracy).trial_number
        keep_full = _full_detail_trials(all_records, best_trial=best_trial, recent_window=recent_window)

        blocks: list[str] = []
        latest_journal: str = ""
        prev_config: TrialConfig | None = None
        for record in all_records:
            if record.trial_number in keep_full:
                blocks.append(
                    _render_trial_block(
                        record,
                        prev_config=prev_config,
                        is_best=(record.trial_number == best_trial),
                        show_cost=show_cost,
                    )
                )
            prev_config = record.config
            strategy = getattr(record.meta, "strategy", None) if record.meta is not None else None
            if strategy is not None and strategy.journal:
                latest_journal = strategy.journal

        result = "\n\n".join(blocks)
        result += "\n\n" + _configs_tried_index(all_records)
        if latest_journal:
            result += f"\n\n### Latest agent journal (rewritten each trial)\n{latest_journal}"
        return result

    def format_for_diagnoser(self, *, crosstab_window: int = _DIAGNOSER_CROSSTAB_WINDOW) -> str:
        """Cost-free, objective-agnostic trajectory view for the Diagnoser.

        One correctness line per prior trial — no configs, no cost, no
        "configs already tried" index: the Diagnoser proposes nothing and its
        grounding rules never cite history, so the per-trial config/cost dump
        the Proposer needs is pure noise here. The most recent
        ``crosstab_window`` per-trial failure cross-tabs are appended so the
        Diagnoser can still narrate failure-mode migration across trials.

        Returns the empty-history sentinel when there are no prior trials.
        """
        if not self.records:
            return "No previous trials."

        strip = "\n".join(_diagnoser_trial_line(r) for r in self.records)

        recent = [r for r in self.records if r.cross_tab_snapshot][-crosstab_window:]
        if recent:
            blocks = []
            for r in recent:
                snap = "\n".join(f"    {ln.strip()}" for ln in r.cross_tab_snapshot.splitlines() if ln.strip())
                blocks.append(f"  trial {r.trial_number}:\n{snap}")
            strip += f"\n\nRecent failure-mode cross-tabs (last {len(recent)} trials):\n" + "\n".join(blocks)
        return strip

    def get_response_matrix(self) -> np.ndarray | None:
        """Build a (n_trials × n_questions) binary matrix from stored results.

        Returns None if fewer than 2 trials exist (IRT needs at least 2).
        Aligns columns by question_id across all trials.
        """
        if len(self.records) < 2:
            return None

        # Collect all unique question IDs in stable order
        question_ids: list[str] = []
        seen: set[str] = set()
        for record in self.records:
            for qr in record.question_results:
                if qr.question_id not in seen:
                    seen.add(qr.question_id)
                    question_ids.append(qr.question_id)

        qid_to_col = {qid: i for i, qid in enumerate(question_ids)}
        n_trials = len(self.records)
        n_questions = len(question_ids)

        # Default to 0 (incorrect) for questions a trial didn't encounter
        matrix = np.zeros((n_trials, n_questions), dtype=int)
        for row, record in enumerate(self.records):
            for qr in record.question_results:
                col = qid_to_col[qr.question_id]
                matrix[row, col] = 1 if qr.correct else 0

        return matrix

    def get_response_matrix_for_exam(self, exam_question_ids: set[str]) -> np.ndarray | None:
        """Build response matrix aligned to current exam questions only.

        Columns are sorted by question ID for deterministic alignment.
        Questions absent in a trial remain 0.
        """
        if len(self.records) < 2 or not exam_question_ids:
            return None

        ordered_ids = sorted(exam_question_ids)
        qid_to_col = {qid: idx for idx, qid in enumerate(ordered_ids)}

        n_trials = len(self.records)
        n_questions = len(ordered_ids)
        matrix = np.zeros((n_trials, n_questions), dtype=int)

        for row, record in enumerate(self.records):
            for qr in record.question_results:
                if qr.question_id not in qid_to_col:
                    continue
                col = qid_to_col[qr.question_id]
                matrix[row, col] = 1 if qr.correct else 0

        return matrix


def _config_lines(config: TrialConfig) -> list[str]:
    """Render every TrialConfig field, two per line, with ``n/a`` for inapplicable fields.

    Inapplicable graph fields render as ``n/a`` so the agent sees the absence
    explicitly rather than guessing a default. Reasoning effort is search-space-
    level (not per trial) and is therefore omitted here — when reasoning=true
    the agent reads the effort from the search space block.
    """
    is_graph_index = getattr(config.index_type, "value", config.index_type) in _GRAPH_INDEX_VALUES
    graph_mode = config.graph_query_mode if is_graph_index else "n/a"
    graph_top_k: int | str = config.graph_top_k if is_graph_index else "n/a"
    return [
        f"  index_type={config.index_type.value}  embedding_model={config.embedding_model}",
        (
            f"  chunking_strategy={config.chunking_strategy}  "
            f"chunk_token_size={config.chunk_token_size}  "
            f"chunk_token_overlap={config.chunk_token_overlap}"
        ),
        (
            f"  top_k={config.top_k}  hybrid_alpha={config.hybrid_alpha}  "
            f"reranker={config.reranker}  reranker_top_n={config.reranker_top_n}"
        ),
        f"  query_expansion={config.query_expansion}",
        f"  generator_llm={config.generator_llm}",
        f"  compressor_llm={config.compressor_llm}",
        f"  expander_llm={config.expander_llm}",
        f"  temperature={config.temperature}  reasoning={str(config.reasoning).lower()}",
        f"  graph_query_mode={graph_mode}  graph_top_k={graph_top_k}",
    ]


def _render_trial_block(
    record: TrialRecord,
    *,
    prev_config: TrialConfig | None = None,
    is_best: bool = False,
    show_cost: bool = True,
) -> str:
    """Render every recorded field of a trial in a single block.

    The renderer backs ``HistoryLog.format_for_agent`` (the Proposer view) for
    every past trial. Fields that were not populated render with sensible zero
    defaults so the agent sees the schema even on early or partial records.

    The "changes vs prior" line is a mechanical diff between ``prev_config``
    and ``record.config`` — pass ``None`` for the very first trial so the
    diff is suppressed.

    When ``show_cost`` is False (score-only runs), the cost/token columns and
    the Pareto-frontier tag are dropped — cost is not an objective there.
    """
    tags: list[str] = []
    if show_cost and record.is_pareto_optimal:
        tags.append("★on Pareto frontier")
    if is_best:
        tags.append("★best accuracy")
    header = f"### Trial {record.trial_number}" + ("  " + "  ".join(tags) if tags else "")

    n_valid = record.trial_metrics.n_valid if record.trial_metrics is not None else 0
    n_em = record.n_em_correct
    n_yes = record.n_judge_correct
    n_no = record.n_judge_rejected
    n_no_ans = record.n_judge_no_answer
    n_failed = record.n_judge_failed
    n_calls = record.n_judge_calls
    if show_cost:
        score_cost_line = (
            f"accuracy={record.answer_accuracy:.3f}  "
            f"cost=${record.mean_llm_cost_per_query_usd:.4f}/q  "
            f"cost_total=${record.total_llm_cost_usd:.3f}  "
            f"in_tok={record.mean_prompt_tokens:.0f}/q  out_tok={record.mean_completion_tokens:.0f}/q"
        )
    else:
        score_cost_line = f"accuracy={record.answer_accuracy:.3f}"
    verdict_line = (
        f"verdicts: EM={n_em}/{n_valid} judge_yes={n_yes}/{n_valid} "
        f"judge_no={n_no}/{n_valid} judge_no_answer={n_no_ans}/{n_valid} "
        f"judge_failed={n_failed}/{n_valid} judge_calls={n_calls}"
    )
    quality_line = (
        f"quality:  retrieval_quality={record.mean_retrieval_quality:.2f}  "
        f"mean_em={record.mean_em:.2f}  mean_f1={record.mean_f1:.2f}"
    )

    rates_line = "retrieval rates: (no metrics recorded)"
    if record.trial_metrics is not None:
        tm = record.trial_metrics
        rates_line = (
            f"retrieval rates: complete={tm.retrieval_complete:.2f} "
            f"partial={tm.retrieval_partial:.2f} "
            f"miss={tm.retrieval_miss:.2f} "
            f"refused={tm.refusal_rate:.2f} "
            f"acc_given_complete={tm.answer_correct_given_complete_retrieval:.2f}  "
            f"(n_valid={tm.n_valid})"
        )

    config_lines = ["config:", *_config_lines(record.config)]

    extra: list[str] = []
    if prev_config is not None:
        from agentic_autorag.optimizer.state import _config_diff_summary

        diff = _config_diff_summary(prev_config, record.config)
        if diff:
            extra.append(f"changes vs prior: {'; '.join(diff)}")
    if record.meta is not None:
        if record.meta.rationale:
            extra.append(f"rationale: {record.meta.rationale}")
        strategy = getattr(record.meta, "strategy", None)
        if strategy is not None and strategy.stance is not None:
            extra.append(f"stance: {strategy.stance}")

    return "\n".join([header, score_cost_line, verdict_line, quality_line, rates_line, *config_lines, *extra])


def _diagnoser_trial_line(record: TrialRecord) -> str:
    """One cost-free correctness line for the Diagnoser's trajectory view.

    Accuracy + per-span retrieval rates + acc-given-complete only: no config,
    no cost/tokens. Falls back to accuracy alone when metrics weren't recorded.
    """
    tm = record.trial_metrics
    if tm is None:
        return f"trial {record.trial_number}: acc={record.answer_accuracy:.3f}"
    return (
        f"trial {record.trial_number}: acc={record.answer_accuracy:.3f} | "
        f"retrieval complete={tm.retrieval_complete:.2f} "
        f"partial={tm.retrieval_partial:.2f} miss={tm.retrieval_miss:.2f} | "
        f"acc_given_complete={tm.answer_correct_given_complete_retrieval:.2f}"
    )


def _config_signature(c: TrialConfig) -> str:
    """Compact, complete one-line signature of every tunable lever.

    Backs the "configs already tried" index so the agent can avoid re-proposing
    a config even when its trial is collapsed out of the full-detail history.
    The programmatic no-repeat check (``record.config == config``) stays
    authoritative; this is only the agent-visible mirror of it, so model-name
    basenames are fine here even though they are not globally unique in theory.
    """
    index_value = getattr(c.index_type, "value", c.index_type)
    parts = [
        f"strategy={c.chunking_strategy}",
        f"chunk={c.chunk_token_size}/{c.chunk_token_overlap}",
        f"embed={c.embedding_model.split('/')[-1]}",
        f"index={index_value}",
        f"top_k={c.top_k}",
    ]
    if index_value == IndexType.HYBRID_BM25_VECTOR.value:
        parts.append(f"alpha={c.hybrid_alpha}/{c.bm25_vector_fusion}")
    if c.long_context_reorder:
        parts.append("reorder=on")
    reranker = c.reranker.split("/")[-1] if c.reranker and c.reranker != "none" else "none"
    parts.append(f"rerank={reranker}/{c.reranker_top_n}")
    parts.append(f"qexp={c.query_expansion}")
    parts.append(f"llm={_fmt_per_stage_llm(c)}")
    if c.reasoning:
        parts.append("reasoning=on")
    if index_value in _GRAPH_INDEX_VALUES:
        parts.append(f"graph={c.graph_query_mode}/{c.graph_top_k}")
    return "  ".join(parts)


def _full_detail_trials(
    records: list[TrialRecord],
    *,
    best_trial: int,
    recent_window: int,
) -> set[int]:
    """Trial numbers rendered in full detail: the most recent ``recent_window``,
    the best-accuracy trial, every Pareto-frontier member, and the
    ``_TOP_MOVERS_FULL`` trials whose accuracy moved most versus the running
    best before them. Everything else collapses to one index line, so this set —
    and hence prompt size — stays bounded regardless of trial count."""
    keep: set[int] = {best_trial}
    keep.update(r.trial_number for r in records[-recent_window:])
    keep.update(r.trial_number for r in records if r.is_pareto_optimal)

    movers: list[tuple[float, int]] = []
    running_best = float("-inf")
    for record in records:
        if running_best != float("-inf"):
            movers.append((abs(record.answer_accuracy - running_best), record.trial_number))
        running_best = max(running_best, record.answer_accuracy)
    movers.sort(reverse=True)
    keep.update(trial_number for _, trial_number in movers[:_TOP_MOVERS_FULL])
    return keep


def _configs_tried_index(records: list[TrialRecord]) -> str:
    """Complete, one-line-per-trial index of every config tried so the agent can
    avoid proposing a duplicate even though older trials are collapsed out of
    the full-detail view. The programmatic no-repeat check is authoritative;
    this is its agent-visible mirror."""
    lines = [
        f"- trial {record.trial_number} (acc={record.answer_accuracy:.3f}): {_config_signature(record.config)}"
        for record in records
    ]
    return "### Configs already tried (complete — do NOT propose any of these again)\n" + "\n".join(lines)
