"""Tests for agentic_autorag.optimizer.state — pure optimizer state functions."""

from __future__ import annotations

from agentic_autorag.config.models import IndexType, TrialConfig
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.optimizer.diagnosis import (
    Bottleneck,
    Diagnosis,
    ProposalMeta,
    TrialMetrics,
)
from agentic_autorag.optimizer.history import TrialRecord
from agentic_autorag.optimizer.state import build_state_card, compute_trial_metrics


def _make_config(**overrides) -> TrialConfig:
    defaults = dict(
        chunking_strategy="recursive",
        chunk_token_size=512,
        chunk_token_overlap=64,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_type=IndexType.VECTOR_ONLY,
        top_k=5,
        reranker="none",
        reranker_top_n=5,
        llm_model="ollama/llama3.2",
        temperature=0.0,
    )
    defaults.update(overrides)
    return TrialConfig(**defaults)


def _qr(
    qid: str,
    *,
    correct: bool,
    retrieved_spans: int = 0,
    n_spans: int = 2,
    refused: bool = False,
    generated_response: str = "A",
) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        correct=correct,
        selected_answer="A" if correct else "B",
        correct_answer="A",
        retrieved_context="",
        generated_response=generated_response,
        chunk_precision=0.2 if retrieved_spans > 0 else 0.0,
        source_fact_rank=1 if retrieved_spans > 0 else 0,
        retrieved_doc_ids=[],
        retrieved_spans=retrieved_spans,
        n_spans=n_spans,
        refused=refused,
    )


class TestComputeTrialMetrics:
    def test_empty_exam(self) -> None:
        result = ExamResult(score=0.0, n_correct=0, n_total=0, question_results=[])

        metrics = compute_trial_metrics(result)

        assert metrics.answer_accuracy == 0.0
        assert metrics.retrieval_complete == 0.0
        assert metrics.n_valid == 0

    def test_all_failure_modes(self) -> None:
        results = [
            _qr("q1", correct=True, retrieved_spans=2, n_spans=2),
            _qr("q2", correct=True, retrieved_spans=2, n_spans=2),
            _qr("q3", correct=False, retrieved_spans=2, n_spans=2),
            _qr("q4", correct=False, retrieved_spans=1, n_spans=2),
            _qr("q5", correct=False, retrieved_spans=1, n_spans=2),
            _qr("q6", correct=False, retrieved_spans=0, n_spans=2),
            _qr("q7", correct=False, retrieved_spans=0, n_spans=2, refused=True, generated_response="cannot answer"),
            _qr("q8", correct=False, retrieved_spans=1, n_spans=2, refused=True, generated_response="no information"),
        ]
        exam_result = ExamResult(score=0.25, n_correct=2, n_total=8, question_results=results)

        m = compute_trial_metrics(exam_result)

        assert m.n_valid == 8
        assert m.answer_accuracy == 0.25
        assert abs(m.retrieval_complete - 3 / 8) < 1e-6
        assert abs(m.retrieval_partial - 3 / 8) < 1e-6
        assert abs(m.retrieval_miss - 2 / 8) < 1e-6
        assert abs(m.refusal_rate - 2 / 8) < 1e-6
        # 2 correct out of 3 retrieval_complete
        assert abs(m.answer_correct_given_complete_retrieval - 2 / 3) < 1e-6

    def test_excludes_system_errors(self) -> None:
        results = [
            _qr("q1", correct=True, retrieved_spans=2, n_spans=2),
            _qr(
                "q2",
                correct=False,
                retrieved_spans=0,
                n_spans=2,
                generated_response="QUESTION_EVALUATION_ERROR",
            ),
        ]
        exam_result = ExamResult(score=1.0, n_correct=1, n_total=2, question_results=results)

        m = compute_trial_metrics(exam_result)

        assert m.n_valid == 1
        assert m.retrieval_complete == 1.0

    def test_acc_given_complete_zero_when_no_complete(self) -> None:
        results = [_qr("q1", correct=False, retrieved_spans=0, n_spans=2)]
        exam_result = ExamResult(score=0.0, n_correct=0, n_total=1, question_results=results)

        m = compute_trial_metrics(exam_result)

        assert m.answer_correct_given_complete_retrieval == 0.0


class TestBuildStateCard:
    def test_first_trial_no_history_in_search(self) -> None:
        card = build_state_card(
            trial_number=1,
            trials_remaining=9,
            current_score=0.55,
            history_records=[],
            max_trials=10,
            current_config=_make_config(),
        )

        assert card.trial_number == 1
        assert card.best_score_so_far == 0.55
        assert card.last_trial_delta == 0.0
        assert card.phase == "search"
        assert len(card.trial_summaries) == 1
        assert card.trial_summaries[0]["trial_number"] == 1

    def test_phase_search_in_first_half(self) -> None:
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(embedding_model="A"),
            score=0.55,
            question_results=[],
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_score=0.62,
            history_records=[prev],
            max_trials=10,
            current_config=_make_config(embedding_model="B"),
        )

        assert card.phase == "search"
        assert card.best_score_so_far == 0.62
        assert card.best_trial_number == 2
        assert abs(card.last_trial_delta - 0.07) < 1e-6

    def test_phase_polish_in_tail_with_decent_score(self) -> None:
        prev = TrialRecord(
            trial_number=7,
            config=_make_config(embedding_model="A"),
            score=0.70,
            question_results=[],
        )
        card = build_state_card(
            trial_number=8,
            trials_remaining=2,
            current_score=0.65,
            history_records=[prev],
            max_trials=10,
            current_config=_make_config(embedding_model="B"),
        )

        assert card.phase == "polish"

    def test_phase_search_when_score_below_floor_even_late(self) -> None:
        prev = TrialRecord(
            trial_number=7,
            config=_make_config(embedding_model="A"),
            score=0.30,
            question_results=[],
        )
        card = build_state_card(
            trial_number=8,
            trials_remaining=2,
            current_score=0.40,
            history_records=[prev],
            max_trials=10,
            current_config=_make_config(embedding_model="B"),
        )

        assert card.phase == "search"

    def test_trial_summaries_include_changes_and_failure_modes(self) -> None:
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(embedding_model="A", top_k=5),
            score=0.5,
            question_results=[],
            diagnosis=Diagnosis(
                trial_metrics=TrialMetrics(),
                bottlenecks=[
                    Bottleneck(stage="retrieval", severity="primary", evidence=""),
                    Bottleneck(stage="generation", severity="secondary", evidence=""),
                ],
            ),
            meta=ProposalMeta(changes=["embedding_model: A → B"], rationale="…"),
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_score=0.6,
            history_records=[prev],
            max_trials=10,
            current_config=_make_config(embedding_model="B", top_k=10),
            current_top_failure_modes=["ranking", "generation"],
        )

        # Two summaries: prev trial + current trial
        assert len(card.trial_summaries) == 2
        prev_summary = card.trial_summaries[0]
        assert prev_summary["trial_number"] == 1
        assert prev_summary["top_failure_modes"] == ["retrieval", "generation"]
        cur_summary = card.trial_summaries[1]
        assert cur_summary["trial_number"] == 2
        assert any("embedding_model" in c for c in cur_summary["what_changed_from_prev"])
        assert any("top_k" in c for c in cur_summary["what_changed_from_prev"])
        assert cur_summary["top_failure_modes"] == ["ranking", "generation"]
