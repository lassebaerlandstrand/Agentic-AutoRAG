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

    ``diagnosis`` is the structured output of the Diagnoser; ``meta`` is the
    structured output of the Proposer that produced the *next* trial's config.
    Both may be None for the final trial (no next-config proposal) or when an
    older record predates the structured hand-off.
    """

    trial_number: int
    config: TrialConfig
    score: float
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
            f"composite={self.score:.3f}{cost_tag} | "
            f"acc={self.answer_accuracy:.3f} ({verdict}), "
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
            "score": self.score,
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
            score=data["score"],
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
        """Return the record with the highest score, or None if empty."""
        if not self.records:
            return None
        return max(self.records, key=lambda r: r.score)

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

    def format_for_agent(self, last_n: int = 10, *, include_proposer_context: bool = True) -> str:
        """Format the last N trials as structured text for agent prompts.

        Each trial renders ALL ``TrialConfig`` fields and the full mechanical
        metric set (verdict breakdown, retrieval rates, retrieval/EM/F1 quality,
        cost) so the agent can do its own cross-trial aggregation from raw data
        without us pre-digesting "lever effects" or "hypothesis outcomes" — the
        kind of interpretive aggregation that introduces spurious confidence.

        When ``include_proposer_context`` is False (Diagnoser view), the
        Proposer-emitted fields (``rationale``, ``strategy``, the journal
        trailer) and any Diagnoser-emitted interpretive fields (failure
        attribution, regression flag) are suppressed so the Diagnoser cannot
        anchor on prior beliefs — only the mechanical cross-tab snapshot is
        retained as per-trial evidence.

        Pareto frontier annotations on the trial header come straight from
        ``record.is_pareto_optimal``; the orchestrator updates that flag on
        every new trial before this method is called.
        """
        if not self.records:
            return "No previous trials."

        knee_trial: int | None = _knee_trial_number(list(self.records))
        best_trial: int | None = max(self.records, key=lambda r: r.score).trial_number if self.records else None

        blocks: list[str] = []
        latest_journal: str = ""
        recent = self.records[-last_n:]
        for record in recent:
            blocks.append(
                _render_trial_block(
                    record,
                    is_knee=(record.trial_number == knee_trial),
                    is_best=(record.trial_number == best_trial),
                    include_proposer_context=include_proposer_context,
                )
            )
            strategy = getattr(record.meta, "strategy", None) if record.meta is not None else None
            if strategy is not None and strategy.journal:
                latest_journal = strategy.journal

        result = "\n\n".join(blocks)
        if include_proposer_context and latest_journal:
            result += f"\n\n### Latest agent journal (rewritten each trial)\n{latest_journal}"
        return result

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


def _knee_trial_number(records: list[TrialRecord]) -> int | None:
    """Trial number of the knee point (max score-per-cost) on the current frontier.

    Local import of ``pareto`` to avoid circulars at module load.
    """
    from agentic_autorag.optimizer import pareto

    if not records:
        return None
    frontier = pareto.compute_frontier(records)
    knee = pareto.find_knee(frontier)
    return knee.trial_number if knee is not None else None


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
    is_knee: bool = False,
    is_best: bool = False,
    include_proposer_context: bool = True,
) -> str:
    """Render every recorded field of a trial in a single block.

    The same renderer is used by ``HistoryLog.format_for_agent`` for every past
    trial. Fields that were not populated render with sensible zero defaults so
    the agent sees the schema even on early or partial records.

    When ``include_proposer_context`` is False (Diagnoser view), prior
    Proposer-emitted fields (rationale, strategy line) and prior
    Diagnoser-emitted interpretive fields (failure attribution, regression
    flag) are suppressed; only the mechanical cross-tab snapshot is retained.
    """
    tags: list[str] = []
    if record.is_pareto_optimal:
        tags.append("★on Pareto frontier")
    if is_knee:
        tags.append("(knee)")
    if is_best:
        tags.append("★best score")
    header = f"### Trial {record.trial_number}" + ("  " + "  ".join(tags) if tags else "")

    n_valid = record.trial_metrics.n_valid if record.trial_metrics is not None else 0
    n_em = record.n_em_correct
    n_yes = record.n_judge_correct
    n_no = record.n_judge_rejected
    n_no_ans = record.n_judge_no_answer
    n_failed = record.n_judge_failed
    n_calls = record.n_judge_calls
    score_cost_line = (
        f"score={record.score:.3f} (=accuracy)  "
        f"cost=${record.mean_llm_cost_per_query_usd:.4f}/q  "
        f"cost_total=${record.total_llm_cost_usd:.3f}"
    )
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
    if record.meta is not None and record.meta.changes:
        extra.append(f"changes from prev trial: {'; '.join(record.meta.changes)}")
    if include_proposer_context:
        if record.diagnosis is not None:
            fa = record.diagnosis.failure_attribution
            extra.append(
                f"failure_attribution: retrieval={fa.retrieval:.2f} ranking={fa.ranking:.2f} "
                f"generation={fa.generation:.2f} composition={fa.composition:.2f}"
            )
            if record.diagnosis.regression_detected:
                axes_str = ", ".join(record.diagnosis.regression_axes) or "<unspecified>"
                extra.append(f"regression_detected: true (axes: {axes_str})")
        if record.meta is not None:
            if record.meta.rationale:
                extra.append(f"rationale: {record.meta.rationale}")
            strategy = getattr(record.meta, "strategy", None)
            if strategy is not None:
                anchor_str = f" anchor=trial{strategy.anchor_trial}" if strategy.anchor_trial is not None else ""
                extra.append(
                    f"strategy: stance={strategy.stance}{anchor_str}"
                    f" revisions={strategy.revision_count} | intent: {strategy.intent}"
                )
    else:
        if record.cross_tab_snapshot:
            extra.append("cross_tab (this trial):")
            extra.extend(f"  {line}" for line in record.cross_tab_snapshot.splitlines() if line.strip())

    return "\n".join([header, score_cost_line, verdict_line, quality_line, rates_line, *config_lines, *extra])
