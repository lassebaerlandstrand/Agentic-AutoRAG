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
    ProposalMeta,
    TrialMetrics,
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
        generator_llm="ollama/llama3.2",
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


def _make_trial_metrics(retrieval_complete: float = 0.7) -> TrialMetrics:
    return TrialMetrics(
        answer_accuracy=0.6,
        retrieval_complete=retrieval_complete,
        retrieval_partial=0.2,
        retrieval_miss=0.1,
        refusal_rate=0.05,
        answer_correct_given_complete_retrieval=0.85,
        n_valid=25,
        mean_llm_cost_per_query_usd=0.012,
    )


def _make_diagnosis() -> Diagnosis:
    return Diagnosis(
        trial_metrics=_make_trial_metrics(),
        narrative="retrieval looks weak",
        confirmed_findings=["12 of 20 failures are retrieval_miss"],
    )


def _make_meta() -> ProposalMeta:
    from agentic_autorag.optimizer.diagnosis import Strategy

    return ProposalMeta(
        rationale="diagnoser flagged retrieval primary; widening helps",
        strategy=Strategy(
            stance="explore",
            journal="bullet one — MiniLM misses span_B",
        ),
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
        trial_metrics=_make_trial_metrics() if with_structured else None,
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
        assert restored.trial_metrics is not None
        assert restored.trial_metrics.retrieval_complete == record.trial_metrics.retrieval_complete
        assert restored.diagnosis is not None
        assert restored.diagnosis.narrative == "retrieval looks weak"
        assert restored.diagnosis.confirmed_findings == ["12 of 20 failures are retrieval_miss"]
        assert restored.meta is not None
        assert restored.meta.rationale == "diagnoser flagged retrieval primary; widening helps"
        assert restored.meta.strategy is not None
        assert restored.meta.strategy.stance == "explore"

    def test_to_dict_roundtrip_without_structured(self) -> None:
        record = _make_record(1, 0.5, with_structured=False)

        data = record.to_dict()
        restored = TrialRecord.from_dict(data)

        assert restored.trial_metrics is None
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

    def test_format_for_agent_includes_full_trial_block_and_journal(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        # Two trials with different configs so the mechanical "changes vs prior"
        # line has something to render on trial 2.
        log.add(_make_record(1, 0.5))
        rec2 = _make_record(2, 0.6)
        rec2.config = rec2.config.model_copy(update={"embedding_model": "BAAI/bge-m3", "top_k": 10})
        log.add(rec2)

        text = log.format_for_agent()

        # Header + score/cost line
        assert "Trial 1" in text
        assert "Trial 2" in text
        assert "score=0.600" in text
        assert "cost=$" in text
        # Verdict breakdown
        assert "verdicts: EM=" in text
        assert "judge_yes=" in text
        assert "judge_no_answer=" in text
        # Quality + retrieval rates
        assert "quality:" in text
        assert "retrieval rates: complete=" in text
        # Full config rendering — every TrialConfig field name should appear
        for field_name in (
            "index_type",
            "embedding_model",
            "chunking_strategy",
            "chunk_token_size",
            "chunk_token_overlap",
            "top_k",
            "hybrid_alpha",
            "reranker",
            "reranker_top_n",
            "query_expansion",
            "generator_llm",
            "compressor_llm",
            "expander_llm",
            "temperature",
            "reasoning",
            "graph_query_mode",
            "graph_top_k",
        ):
            assert field_name in text, f"missing config field {field_name} in rendered block"
        # Mechanical diff between trial 1 and trial 2 configs.
        assert "changes vs prior:" in text
        assert "embedding_model:" in text and "BAAI/bge-m3" in text
        assert "top_k:" in text
        assert "rationale:" in text
        assert "stance: explore" in text
        assert "Latest agent journal" in text
        assert "MiniLM misses span_B" in text

    def test_format_for_agent_appends_current_trial_preview(self, tmp_path) -> None:
        # The orchestrator persists the just-completed trial to history AFTER
        # the Proposer runs, so during proposal the current trial is passed as
        # a preview record. format_for_agent must render it as the last block.
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        log.add(_make_record(1, 0.5))
        preview = _make_record(2, 0.82)
        preview.meta = None  # proposer hasn't emitted meta for the current trial yet

        text = log.format_for_agent(current_trial=preview)

        assert "Trial 1" in text
        assert "Trial 2" in text
        # The preview's score should appear and trial 2 should carry the best-score tag.
        assert "score=0.820" in text
        assert "★best score" in text
        # No persisted records were mutated.
        assert len(log.records) == 1

    def test_format_for_agent_empty_history_with_current_trial(self, tmp_path) -> None:
        # First trial of a run: no prior history, but the Proposer still needs
        # the current trial's block (and a "No previous trials" sentinel
        # would lie to it).
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        preview = _make_record(1, 0.6)
        preview.meta = None

        text = log.format_for_agent(current_trial=preview)

        assert text != "No previous trials."
        assert "Trial 1" in text
        assert "score=0.600" in text

    def test_format_for_agent_diagnoser_view_strips_proposer_fields(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        record = _make_record(1, 0.6)
        record.cross_tab_snapshot = "retrieval_miss × bridge(n=2): 4"
        log.add(record)

        text = log.format_for_agent(include_proposer_context=False)

        # Mechanical fields stay.
        assert "Trial 1" in text
        assert "score=0.600" in text
        assert "retrieval rates: complete=" in text
        # Proposer-side fields are gone.
        assert "rationale:" not in text
        assert "stance:" not in text
        assert "Latest agent journal" not in text
        # Cross-tab snapshot replaces them.
        assert "cross_tab (this trial):" in text
        assert "retrieval_miss × bridge(n=2): 4" in text

    def test_format_for_agent_marks_pareto_and_best(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        # Two trials: trial 1 cheap+ok, trial 2 expensive+best. After Pareto
        # recomputation both should be on the frontier; trial 2 is the best
        # score. The knee marker was dropped — the optimizer loop uses
        # best-score as the universal anchor instead.
        rec1 = _make_record(1, 0.6)
        rec1.mean_llm_cost_per_query_usd = 0.001
        rec2 = _make_record(2, 0.9)
        rec2.mean_llm_cost_per_query_usd = 0.05
        log.add(rec1)
        log.add(rec2)
        log.recompute_pareto_flags()

        text = log.format_for_agent()

        assert "★on Pareto frontier" in text
        assert "★best score" in text
        assert "(knee)" not in text

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
