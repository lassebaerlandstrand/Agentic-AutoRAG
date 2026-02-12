"""Tests for the trial history module."""

from __future__ import annotations

import json

import numpy as np

from agentic_autorag.config.models import (
    IndexType,
    RuntimeConfig,
    StructuralConfig,
    TrialConfig,
)
from agentic_autorag.examiner.evaluator import QuestionResult
from agentic_autorag.optimizer.history import HistoryLog, TrialRecord


def _make_config(**overrides) -> TrialConfig:
    """Build a TrialConfig with sensible defaults, allowing overrides."""
    structural = overrides.pop("structural", None) or StructuralConfig(
        parser="pymupdf4llm",
        chunking_strategy="recursive",
        chunk_size=512,
        chunk_overlap=64,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_type=IndexType.VECTOR_ONLY,
    )
    runtime = overrides.pop("runtime", None) or RuntimeConfig(
        top_k=5,
        reranker="none",
        llm_model="ollama/llama3.2",
        temperature=0.0,
    )
    return TrialConfig(structural=structural, runtime=runtime, **overrides)


def _make_question_result(qid: str, *, correct: bool) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        correct=correct,
        selected_answer="A" if correct else "B",
        correct_answer="A",
        retrieved_context="some context",
        generated_response="A" if correct else "B",
    )


def _make_record(trial_number: int, score: float, question_ids: list[str] | None = None) -> TrialRecord:
    if question_ids is None:
        question_ids = ["q1", "q2", "q3"]
    return TrialRecord(
        trial_number=trial_number,
        config=_make_config(),
        score=score,
        error_trace="some error trace",
        question_results=[
            _make_question_result(qid, correct=(score > 0.5)) for qid in question_ids
        ],
    )


class TestTrialRecord:
    def test_summary_format(self) -> None:
        record = _make_record(3, 0.65)
        summary = record.summary()
        assert summary.startswith("Trial 3:")
        assert "score=0.650" in summary
        assert "chunk=512" in summary
        assert "embed=sentence-transformers/all-MiniLM-L6-v2" in summary
        assert "index=vector_only" in summary
        assert "top_k=5" in summary
        assert "reranker=none" in summary
        assert "llm=ollama/llama3.2" in summary

    def test_to_dict_roundtrip(self) -> None:
        record = _make_record(1, 0.8)
        data = record.to_dict()
        restored = TrialRecord.from_dict(data)
        assert restored.trial_number == record.trial_number
        assert restored.score == record.score
        assert restored.error_trace == record.error_trace
        assert restored.config.structural.chunk_size == record.config.structural.chunk_size
        assert len(restored.question_results) == len(record.question_results)

    def test_to_dict_is_json_serializable(self) -> None:
        record = _make_record(1, 0.5)
        # Should not raise
        json.dumps(record.to_dict())


class TestHistoryLog:
    def test_empty_log(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        assert log.records == []
        assert log.get_best() is None

    def test_add_and_get_best(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        log.add(_make_record(1, 0.5))
        log.add(_make_record(2, 0.8))
        log.add(_make_record(3, 0.6))
        assert len(log.records) == 3
        best = log.get_best()
        assert best is not None
        assert best.trial_number == 2
        assert best.score == 0.8

    def test_persistence(self, tmp_path) -> None:
        path = str(tmp_path / "history.jsonl")
        log1 = HistoryLog(path=path)
        log1.add(_make_record(1, 0.5))
        log1.add(_make_record(2, 0.9))

        # Reload from the same file
        log2 = HistoryLog(path=path)
        assert len(log2.records) == 2
        assert log2.records[0].trial_number == 1
        assert log2.records[1].score == 0.9

    def test_format_for_agent_empty(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        assert log.format_for_agent() == "No previous trials."

    def test_format_for_agent_last_n(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        for i in range(5):
            log.add(_make_record(i + 1, 0.1 * (i + 1)))

        text = log.format_for_agent(last_n=3)
        lines = text.strip().split("\n")
        assert len(lines) == 3
        assert "Trial 3:" in lines[0]
        assert "Trial 5:" in lines[2]

    def test_get_response_matrix_none_for_few_trials(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        assert log.get_response_matrix() is None

        log.add(_make_record(1, 0.5))
        assert log.get_response_matrix() is None

    def test_get_response_matrix_shape(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        log.add(_make_record(1, 0.8, question_ids=["q1", "q2", "q3"]))
        log.add(_make_record(2, 0.4, question_ids=["q1", "q2", "q3"]))

        matrix = log.get_response_matrix()
        assert matrix is not None
        assert matrix.shape == (2, 3)
        assert matrix.dtype == int

    def test_get_response_matrix_values(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))

        # Trial 1: all correct (score > 0.5 triggers correct=True in _make_record)
        log.add(_make_record(1, 0.8, question_ids=["q1", "q2"]))
        # Trial 2: all incorrect
        log.add(_make_record(2, 0.3, question_ids=["q1", "q2"]))

        matrix = log.get_response_matrix()
        assert matrix is not None
        np.testing.assert_array_equal(matrix[0], [1, 1])  # all correct
        np.testing.assert_array_equal(matrix[1], [0, 0])  # all incorrect

    def test_get_response_matrix_different_questions(self, tmp_path) -> None:
        """Trials with different question sets produce a padded matrix."""
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        log.add(_make_record(1, 0.8, question_ids=["q1", "q2"]))
        log.add(_make_record(2, 0.8, question_ids=["q2", "q3"]))

        matrix = log.get_response_matrix()
        assert matrix is not None
        # q1, q2, q3 → 3 columns
        assert matrix.shape == (2, 3)
        # Trial 1 didn't see q3 → defaults to 0
        assert matrix[0, 2] == 0
        # Trial 2 didn't see q1 → defaults to 0
        assert matrix[1, 0] == 0

    def test_get_response_matrix_for_exam_filters_columns(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        log.add(_make_record(1, 0.8, question_ids=["q1", "q2", "q3"]))
        log.add(_make_record(2, 0.3, question_ids=["q2", "q3", "q4"]))

        matrix = log.get_response_matrix_for_exam({"q2", "q4"})
        assert matrix is not None
        assert matrix.shape == (2, 2)
        # sorted columns => q2, q4
        np.testing.assert_array_equal(matrix[0], [1, 0])
        np.testing.assert_array_equal(matrix[1], [0, 0])
