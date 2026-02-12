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

logger = logging.getLogger(__name__)


@dataclass
class TrialRecord:
    """A single optimization trial result with JSON serialization."""

    trial_number: int
    config: TrialConfig
    score: float
    error_trace: str
    question_results: list[QuestionResult]
    timestamp: datetime = field(default_factory=datetime.now)
    reasoning: str = ""

    def summary(self) -> str:
        """One-line summary for agent context."""
        c = self.config
        return (
            f"Trial {self.trial_number}: "
            f"score={self.score:.3f} | "
            f"chunk={c.structural.chunk_size}, "
            f"embed={c.structural.embedding_model}, "
            f"index={c.structural.index_type.value}, "
            f"top_k={c.runtime.top_k}, "
            f"reranker={c.runtime.reranker}, "
            f"llm={c.runtime.llm_model}"
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "trial_number": self.trial_number,
            "config": self.config.model_dump(mode="json"),
            "score": self.score,
            "error_trace": self.error_trace,
            "question_results": [qr.model_dump(mode="json") for qr in self.question_results],
            "timestamp": self.timestamp.isoformat(),
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrialRecord:
        """Reconstruct a TrialRecord from a stored dict."""
        return cls(
            trial_number=data["trial_number"],
            config=TrialConfig.model_validate(data["config"]),
            score=data["score"],
            error_trace=data["error_trace"],
            question_results=[QuestionResult.model_validate(qr) for qr in data["question_results"]],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            reasoning=data.get("reasoning", ""),
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

    def add(self, record: TrialRecord) -> None:
        """Append a record to in-memory list and persist to JSONL."""
        self.records.append(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def get_best(self) -> TrialRecord | None:
        """Return the record with the highest score, or None if empty."""
        if not self.records:
            return None
        return max(self.records, key=lambda r: r.score)

    def format_for_agent(self, last_n: int = 10) -> str:
        """Format the last N trials as readable text for the agent's prompt."""
        if not self.records:
            return "No previous trials."
        lines = []
        for record in self.records[-last_n:]:
            lines.append(record.summary())
        return "\n".join(lines)

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
