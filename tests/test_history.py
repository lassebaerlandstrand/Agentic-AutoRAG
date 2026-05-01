"""Tests for the trial history module."""

from __future__ import annotations

import json

import numpy as np

from agentic_autorag.config.models import (
    IndexType,
    TrialConfig,
)
from agentic_autorag.examiner.evaluator import QuestionResult
from agentic_autorag.optimizer.diagnosis import (
    Diagnosis,
    HypothesisCheck,
    MoveType,
    ProposalMeta,
    Stage,
    StageMetrics,
)
from agentic_autorag.optimizer.history import HistoryLog, TrialRecord


def _make_config(**overrides) -> TrialConfig:
    defaults = dict(
        chunking_strategy="recursive",
        chunk_token_size=512,
        chunk_token_overlap=64,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_type=IndexType.VECTOR_ONLY,
        top_k=5,
        reranker="none",
        llm_model="ollama/llama3.2",
        temperature=0.0,
    )
    defaults.update(overrides)
    return TrialConfig(**defaults)


def _make_question_result(qid: str, *, correct: bool) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        correct=correct,
        selected_answer="A" if correct else "B",
        correct_answer="A",
        retrieved_context="some context",
        generated_response="A" if correct else "B",
    )


def _make_stage_metrics(retrieval: float = 0.7) -> StageMetrics:
    return StageMetrics(
        retrieval_success=retrieval,
        ranking_quality=0.5,
        gold_in_reranker_window=0.6,
        generation_given_context=0.8,
        n_eligible_for_generation=25,
    )


def _make_diagnosis() -> Diagnosis:
    return Diagnosis(
        stage_metrics=_make_stage_metrics(),
        bottleneck=Stage.RETRIEVAL,
        confidence="medium",
        hypothesis_check=HypothesisCheck(),
        applicable_levers=["embedding_model", "chunk_token_size"],
        narrative="retrieval looks weak",
    )


def _make_meta() -> ProposalMeta:
    return ProposalMeta(
        move_type=MoveType.PROBE,
        primary_lever="embedding_model",
        hypothesis="swap should raise retrieval_success by 0.08",
        target_metric="retrieval_success",
        expected_delta=0.08,
        rationale="diagnoser's top pick",
        memo=["bullet one"],
    )


def _make_record(
    trial_number: int,
    score: float,
    question_ids: list[str] | None = None,
    *,
    with_structured: bool = True,
) -> TrialRecord:
    if question_ids is None:
        question_ids = ["q1", "q2", "q3"]
    return TrialRecord(
        trial_number=trial_number,
        config=_make_config(),
        score=score,
        question_results=[_make_question_result(qid, correct=(score > 0.5)) for qid in question_ids],
        stage_metrics=_make_stage_metrics() if with_structured else None,
        diagnosis=_make_diagnosis() if with_structured else None,
        meta=_make_meta() if with_structured else None,
    )


class TestTrialRecord:
    def test_summary_format(self) -> None:
        record = _make_record(3, 0.65)

        summary = record.summary()

        assert summary.startswith("Trial 3:")
        assert "composite=0.650" in summary
        assert "chunk=512" in summary
        assert "llm=ollama/llama3.2" in summary

    def test_to_dict_roundtrip_with_structured(self) -> None:
        record = _make_record(1, 0.8)

        data = record.to_dict()
        restored = TrialRecord.from_dict(data)

        assert restored.trial_number == record.trial_number
        assert restored.score == record.score
        assert restored.stage_metrics is not None
        assert restored.stage_metrics.retrieval_success == record.stage_metrics.retrieval_success
        assert restored.diagnosis is not None
        assert restored.diagnosis.bottleneck == Stage.RETRIEVAL
        assert restored.meta is not None
        assert restored.meta.move_type == MoveType.PROBE

    def test_to_dict_roundtrip_without_structured(self) -> None:
        record = _make_record(1, 0.5, with_structured=False)

        data = record.to_dict()
        restored = TrialRecord.from_dict(data)

        assert restored.stage_metrics is None
        assert restored.diagnosis is None
        assert restored.meta is None

    def test_to_dict_is_json_serializable(self) -> None:
        record = _make_record(1, 0.5)

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
        best = log.get_best()

        assert len(log.records) == 3
        assert best is not None
        assert best.trial_number == 2
        assert best.score == 0.8

    def test_persistence_preserves_structured_fields(self, tmp_path) -> None:
        path = str(tmp_path / "history.jsonl")
        log1 = HistoryLog(path=path)
        log1.add(_make_record(1, 0.5))
        log1.add(_make_record(2, 0.9))

        log2 = HistoryLog(path=path)

        assert len(log2.records) == 2
        assert log2.records[0].trial_number == 1
        assert log2.records[1].score == 0.9
        assert log2.records[1].diagnosis is not None
        assert log2.records[1].meta is not None

    def test_add_strips_large_fields_from_memory(self, tmp_path) -> None:
        path = str(tmp_path / "history.jsonl")
        log = HistoryLog(path=path)
        log.add(_make_record(1, 0.8))

        qr = log.records[0].question_results[0]
        assert qr.retrieved_context == ""
        assert qr.generated_response == ""
        assert qr.question_id == "q1"
        assert qr.correct is True

        reloaded = HistoryLog(path=path)
        qr_disk = reloaded.records[0].question_results[0]
        assert qr_disk.retrieved_context == "some context"

    def test_format_for_agent_empty(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))

        result = log.format_for_agent()

        assert result == "No previous trials."

    def test_format_for_agent_includes_stage_metrics_and_memo(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        log.add(_make_record(1, 0.6))

        text = log.format_for_agent()

        assert "Trial 1" in text
        assert "retrieval=" in text
        assert "bottleneck: retrieval" in text
        assert "move: PROBE" in text
        assert "Latest working memo" in text
        assert "bullet one" in text

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

    def test_get_response_matrix_values(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        log.add(_make_record(1, 0.8, question_ids=["q1", "q2"]))
        log.add(_make_record(2, 0.3, question_ids=["q1", "q2"]))

        matrix = log.get_response_matrix()

        assert matrix is not None
        np.testing.assert_array_equal(matrix[0], [1, 1])
        np.testing.assert_array_equal(matrix[1], [0, 0])

    def test_get_response_matrix_for_exam_filters_columns(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        log.add(_make_record(1, 0.8, question_ids=["q1", "q2", "q3"]))
        log.add(_make_record(2, 0.3, question_ids=["q2", "q3", "q4"]))

        matrix = log.get_response_matrix_for_exam({"q2", "q4"})

        assert matrix is not None
        assert matrix.shape == (2, 2)
        np.testing.assert_array_equal(matrix[0], [1, 0])
        np.testing.assert_array_equal(matrix[1], [0, 0])
