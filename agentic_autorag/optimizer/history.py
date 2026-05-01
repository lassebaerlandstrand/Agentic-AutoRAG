"""Trial history — JSONL persistence for optimization trial records."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from agentic_autorag.config.models import TrialConfig
from agentic_autorag.examiner.evaluator import QuestionResult
from agentic_autorag.optimizer.diagnosis import Diagnosis, ProposalMeta, StageMetrics

logger = logging.getLogger(__name__)


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
    n_judge_failed: int = 0
    n_no_answer: int = 0
    n_judge_calls: int = 0
    mean_em: float = 0.0
    mean_f1: float = 0.0
    stage_metrics: StageMetrics | None = None
    diagnosis: Diagnosis | None = None
    meta: ProposalMeta | None = None

    def summary(self) -> str:
        """One-line summary for agent context."""
        c = self.config
        reasoning_tag = " +reasoning" if c.reasoning else ""
        verdict = f"EM={self.n_em_correct}, judge=yes:{self.n_judge_correct}/no:{self.n_judge_rejected}"
        return (
            f"Trial {self.trial_number}: "
            f"composite={self.score:.3f} | "
            f"acc={self.answer_accuracy:.3f} ({verdict}), "
            f"rq={self.mean_retrieval_quality:.3f} | "
            f"chunk={c.chunk_token_size}, "
            f"embed={c.embedding_model}, "
            f"index={c.index_type.value}, "
            f"top_k={c.top_k}, "
            f"reranker={c.reranker}, "
            f"llm={c.llm_model}{reasoning_tag}"
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
            "n_judge_failed": self.n_judge_failed,
            "n_no_answer": self.n_no_answer,
            "n_judge_calls": self.n_judge_calls,
            "mean_em": self.mean_em,
            "mean_f1": self.mean_f1,
            "stage_metrics": self.stage_metrics.model_dump(mode="json") if self.stage_metrics else None,
            "diagnosis": self.diagnosis.model_dump(mode="json") if self.diagnosis else None,
            "meta": self.meta.model_dump(mode="json") if self.meta else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrialRecord:
        """Reconstruct a TrialRecord from a stored dict."""
        sm = data.get("stage_metrics")
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
            n_judge_failed=data.get("n_judge_failed", 0),
            n_no_answer=data.get("n_no_answer", 0),
            n_judge_calls=data.get("n_judge_calls", 0),
            mean_em=data.get("mean_em", 0.0),
            mean_f1=data.get("mean_f1", 0.0),
            stage_metrics=StageMetrics.model_validate(sm) if sm else None,
            diagnosis=Diagnosis.model_validate(diag) if diag else None,
            meta=ProposalMeta.model_validate(meta) if meta else None,
        )


class HistoryLog:
    """Persistent trial history stored as JSONL."""

    def __init__(self, path: str = "./experiments/history.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[TrialRecord] = []
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
                except (json.JSONDecodeError, KeyError, ValueError):
                    logger.warning("Skipping malformed record on line %d", line_num, exc_info=True)

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

    def get_best(self) -> TrialRecord | None:
        """Return the record with the highest score, or None if empty."""
        if not self.records:
            return None
        return max(self.records, key=lambda r: r.score)

    def format_for_agent(self, last_n: int = 10) -> str:
        """Format the last N trials as structured text for agent prompts.

        Emits per-trial stage metrics, the primary lever changed, the prior
        hypothesis outcome, and the latest working memo — not just config + score.
        This is the cross-trial memory both agents read.
        """
        if not self.records:
            return "No previous trials."
        blocks: list[str] = []
        latest_memo: list[str] = []
        for record in self.records[-last_n:]:
            c = record.config
            verdict = (
                f"EM={record.n_em_correct}, "
                f"judge=yes:{record.n_judge_correct}/no:{record.n_judge_rejected}"
                f"/failed:{record.n_judge_failed}/no_answer:{record.n_no_answer}"
            )
            lines = [
                f"### Trial {record.trial_number}",
                f"composite={record.score:.3f} | "
                f"accuracy={record.answer_accuracy:.3f} ({verdict}), "
                f"rq={record.mean_retrieval_quality:.3f}",
                f"config: index={c.index_type.value} embed={c.embedding_model} "
                f"chunk={c.chunk_token_size}/{c.chunk_token_overlap} "
                f"top_k={c.top_k} reranker={c.reranker} "
                f"llm={c.llm_model}{' +reasoning' if c.reasoning else ''}",
            ]
            if record.stage_metrics is not None:
                sm = record.stage_metrics
                lines.append(
                    f"stage_metrics: retrieval={sm.retrieval_success:.2f} "
                    f"ranking={sm.ranking_quality:.2f} "
                    f"gold_in_window={sm.gold_in_reranker_window:.2f} "
                    f"gen_given_context={sm.generation_given_context:.2f}"
                )
            if record.diagnosis is not None:
                d = record.diagnosis
                hc = d.hypothesis_check
                lines.append(f"bottleneck: {d.bottleneck.value} (confidence={d.confidence})")
                if hc.prior_hypothesis and hc.verdict != "n/a":
                    obs = f"{hc.observed_delta:+.3f}" if hc.observed_delta is not None else "n/a"
                    exp = f"{hc.expected_delta:+.3f}" if hc.expected_delta is not None else "n/a"
                    lines.append(
                        f"prior_hypothesis: {hc.prior_hypothesis} "
                        f"[target={hc.target_metric} expected={exp} observed={obs} → {hc.verdict}]"
                    )
            if record.meta is not None:
                m = record.meta
                lines.append(
                    f"move: {m.move_type.value} lever={m.primary_lever} "
                    f"hypothesis={m.hypothesis!r} "
                    f"target={m.target_metric} expected_delta={m.expected_delta:+.3f}"
                )
                if m.memo:
                    latest_memo = list(m.memo)
            blocks.append("\n".join(lines))
        result = "\n\n".join(blocks)
        if latest_memo:
            memo_block = "\n".join(f"- {bullet}" for bullet in latest_memo[:5])
            result += f"\n\n### Latest working memo\n{memo_block}"
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
