"""Tests for agentic_autorag.optimizer.state — pure optimizer state functions."""

from __future__ import annotations

from agentic_autorag.config.models import IndexType, TrialConfig
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.optimizer.diagnosis import (
    Diagnosis,
    HypothesisCheck,
    MoveType,
    ProposalMeta,
    Stage,
    StageMetrics,
)
from agentic_autorag.optimizer.history import TrialRecord
from agentic_autorag.optimizer.state import (
    build_state_card,
    check_prior_hypothesis,
    compute_stage_metrics,
    suggest_move_type,
)


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
    context_sufficient: bool,
    source_fact_rank: int,
    doc_ids: list[str] | None = None,
    generated_response: str = "A",
) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        correct=correct,
        selected_answer="A" if correct else "B",
        correct_answer="A",
        retrieved_context="",
        generated_response=generated_response,
        chunk_precision=0.2 if context_sufficient else 0.0,
        source_fact_rank=source_fact_rank,
        retrieved_doc_ids=doc_ids or [],
    )


class TestComputeStageMetrics:
    def test_empty_exam(self) -> None:
        result = ExamResult(score=0.0, n_correct=0, n_total=0, question_results=[])

        metrics = compute_stage_metrics(result, reranker_top_n=5)

        assert metrics.retrieval_success == 0.0
        assert metrics.n_eligible_for_generation == 0

    def test_all_dimensions(self) -> None:
        results = [
            _qr("q1", correct=True, context_sufficient=True, source_fact_rank=1),
            _qr("q2", correct=True, context_sufficient=True, source_fact_rank=3),
            _qr("q3", correct=False, context_sufficient=True, source_fact_rank=8),
            _qr("q4", correct=False, context_sufficient=False, source_fact_rank=0),
        ]
        exam_result = ExamResult(score=0.5, n_correct=2, n_total=4, question_results=results)

        metrics = compute_stage_metrics(exam_result, reranker_top_n=5)

        assert metrics.retrieval_success == 0.75  # 3 of 4 sufficient
        assert metrics.n_eligible_for_generation == 3
        # generation_given_context = correct & sufficient / sufficient = 2/3
        assert abs(metrics.generation_given_context - 2 / 3) < 1e-6
        # ranking_quality = mean(1/1, 1/3, 1/8) over the retrieval-successful subset
        assert abs(metrics.ranking_quality - (1 + 1 / 3 + 1 / 8) / 3) < 1e-6
        # gold_in_reranker_window: source_fact_rank in [1..5] for q1 (1) and q2 (3) → 2/4
        assert metrics.gold_in_reranker_window == 0.5

    def test_excludes_system_errors(self) -> None:
        results = [
            _qr("q1", correct=True, context_sufficient=True, source_fact_rank=1),
            _qr(
                "q2",
                correct=False,
                context_sufficient=False,
                source_fact_rank=0,
                generated_response="QUESTION_EVALUATION_ERROR",
            ),
        ]
        exam_result = ExamResult(score=0.5, n_correct=1, n_total=2, question_results=results)

        metrics = compute_stage_metrics(exam_result, reranker_top_n=5)

        # Only q1 should count → retrieval_success = 1.0
        assert metrics.retrieval_success == 1.0


class TestBottleneck:
    def test_retrieval_is_bottleneck_when_below_threshold(self) -> None:
        sm = StageMetrics(retrieval_success=0.5, gold_in_reranker_window=0.9, generation_given_context=0.9)
        assert sm.bottleneck() == Stage.RETRIEVAL

    def test_ranking_is_bottleneck_when_retrieval_ok(self) -> None:
        sm = StageMetrics(retrieval_success=0.8, gold_in_reranker_window=0.4, generation_given_context=0.9)
        assert sm.bottleneck() == Stage.RANKING

    def test_generation_is_bottleneck_otherwise(self) -> None:
        sm = StageMetrics(retrieval_success=0.9, gold_in_reranker_window=0.9, generation_given_context=0.5)
        assert sm.bottleneck() == Stage.GENERATION


class TestCheckPriorHypothesis:
    def test_returns_na_without_prior(self) -> None:
        hc = check_prior_hypothesis(None, None, StageMetrics())
        assert hc.verdict == "n/a"

    def test_confirms_correct_prediction(self) -> None:
        prev_meta = ProposalMeta(
            move_type=MoveType.PROBE,
            primary_lever="embedding_model",
            hypothesis="swap",
            target_metric="retrieval_success",
            expected_delta=0.10,
        )
        prev_metrics = StageMetrics(retrieval_success=0.40)
        current = StageMetrics(retrieval_success=0.55)

        hc = check_prior_hypothesis(prev_meta, prev_metrics, current)

        assert hc.verdict == "confirmed"
        assert abs(hc.observed_delta - 0.15) < 1e-6

    def test_falsifies_wrong_direction(self) -> None:
        prev_meta = ProposalMeta(
            move_type=MoveType.PROBE,
            primary_lever="embedding_model",
            hypothesis="swap",
            target_metric="retrieval_success",
            expected_delta=0.10,
        )
        prev_metrics = StageMetrics(retrieval_success=0.60)
        current = StageMetrics(retrieval_success=0.55)

        hc = check_prior_hypothesis(prev_meta, prev_metrics, current)

        assert hc.verdict == "falsified"

    def test_falsifies_below_magnitude_tolerance(self) -> None:
        prev_meta = ProposalMeta(
            move_type=MoveType.PROBE,
            primary_lever="embedding_model",
            hypothesis="swap",
            target_metric="retrieval_success",
            expected_delta=0.10,
        )
        prev_metrics = StageMetrics(retrieval_success=0.40)
        # only +0.01 — same sign but magnitude well below 0.5 * 0.10
        current = StageMetrics(retrieval_success=0.41)

        hc = check_prior_hypothesis(prev_meta, prev_metrics, current)

        assert hc.verdict == "falsified"


class TestSuggestMoveType:
    def test_revert_on_regression(self) -> None:
        move = suggest_move_type(
            bottleneck=Stage.RETRIEVAL,
            bottleneck_stable=False,
            consecutive_non_improvements=1,
            last_trial_delta=-0.08,
            trials_remaining=5,
            interventions_tried=[],
        )
        assert move == MoveType.REVERT

    def test_pivot_on_sustained_stagnation(self) -> None:
        move = suggest_move_type(
            bottleneck=Stage.RETRIEVAL,
            bottleneck_stable=True,
            consecutive_non_improvements=3,
            last_trial_delta=0.00,
            trials_remaining=5,
            interventions_tried=[],
        )
        assert move == MoveType.PIVOT

    def test_compound_late_with_confirmed(self) -> None:
        interventions = [
            ("embedding_model", "old", "bge-m3", "confirmed"),
            ("reranker", "none", "bge-reranker-v2-m3", "confirmed"),
        ]
        move = suggest_move_type(
            bottleneck=Stage.GENERATION,
            bottleneck_stable=True,
            consecutive_non_improvements=0,
            last_trial_delta=0.01,
            trials_remaining=1,
            interventions_tried=interventions,
        )
        assert move == MoveType.COMPOUND

    def test_refine_when_stable_and_improving(self) -> None:
        move = suggest_move_type(
            bottleneck=Stage.RETRIEVAL,
            bottleneck_stable=True,
            consecutive_non_improvements=0,
            last_trial_delta=0.03,
            trials_remaining=5,
            interventions_tried=[],
        )
        assert move == MoveType.REFINE

    def test_probe_default(self) -> None:
        move = suggest_move_type(
            bottleneck=Stage.RANKING,
            bottleneck_stable=False,
            consecutive_non_improvements=0,
            last_trial_delta=0.00,
            trials_remaining=5,
            interventions_tried=[],
        )
        assert move == MoveType.PROBE


class TestBuildStateCard:
    def test_first_trial_no_history(self) -> None:
        metrics = StageMetrics(retrieval_success=0.6, gold_in_reranker_window=0.8)
        card = build_state_card(
            trial_number=1,
            trials_remaining=9,
            current_metrics=metrics,
            current_score=0.55,
            history_records=[],
        )

        assert card.trial_number == 1
        assert card.best_score_so_far == 0.55
        assert card.last_trial_delta == 0.0
        # first trial beats the initial -inf sentinel, so it's an "improvement" → 0
        assert card.consecutive_non_improvements == 0
        assert card.bottleneck_stable is False

    def test_with_history(self) -> None:
        # retrieval below bottleneck ceiling → bottleneck == RETRIEVAL
        metrics = StageMetrics(retrieval_success=0.5, gold_in_reranker_window=0.8)
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(),
            score=0.5,
            question_results=[],
            stage_metrics=StageMetrics(retrieval_success=0.4, gold_in_reranker_window=0.8),
            diagnosis=Diagnosis(
                stage_metrics=StageMetrics(),
                bottleneck=Stage.RETRIEVAL,
                hypothesis_check=HypothesisCheck(),
            ),
            meta=ProposalMeta(
                move_type=MoveType.PROBE,
                primary_lever="embedding_model",
                hypothesis="swap",
                target_metric="retrieval_success",
                expected_delta=0.1,
            ),
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_metrics=metrics,
            current_score=0.65,
            history_records=[prev],
        )

        assert card.best_score_so_far == 0.65
        assert card.best_trial_number == 2
        assert abs(card.last_trial_delta - 0.15) < 1e-6
        assert card.consecutive_non_improvements == 0
        assert card.current_bottleneck == Stage.RETRIEVAL
        assert card.bottleneck_stable is True
        # Trial 1's meta describes the intervention applied in Trial 2. Trial 2
        # hasn't been recorded in history yet, so the forward-pointing
        # intervention isn't reportable — interventions_tried is empty.
        assert card.interventions_tried == []

    def test_interventions_forward_pointing_attribution(self) -> None:
        # Three trials: Trial 1's meta says "swap embedding A→B" (realised in
        # Trial 2's config). Trial 2's meta says "swap B→C" (realised in
        # Trial 3's config). Trial 3's diagnosis records the hypothesis_check
        # for the B→C swap as falsified; Trial 2's diagnosis records A→B as
        # falsified.
        trial1 = TrialRecord(
            trial_number=1,
            config=_make_config(embedding_model="A"),
            score=0.5,
            question_results=[],
            diagnosis=Diagnosis(
                stage_metrics=StageMetrics(),
                bottleneck=Stage.RETRIEVAL,
                hypothesis_check=HypothesisCheck(),  # no prior hypothesis
            ),
            meta=ProposalMeta(
                move_type=MoveType.PROBE,
                primary_lever="embedding_model",
                hypothesis="swap A→B",
                target_metric="retrieval_success",
                expected_delta=0.1,
            ),
        )
        trial2 = TrialRecord(
            trial_number=2,
            config=_make_config(embedding_model="B"),
            score=0.5,
            question_results=[],
            diagnosis=Diagnosis(
                stage_metrics=StageMetrics(),
                bottleneck=Stage.RETRIEVAL,
                hypothesis_check=HypothesisCheck(verdict="falsified"),
            ),
            meta=ProposalMeta(
                move_type=MoveType.PROBE,
                primary_lever="embedding_model",
                hypothesis="swap B→C",
                target_metric="retrieval_success",
                expected_delta=0.1,
            ),
        )
        trial3 = TrialRecord(
            trial_number=3,
            config=_make_config(embedding_model="C"),
            score=0.5,
            question_results=[],
            diagnosis=Diagnosis(
                stage_metrics=StageMetrics(),
                bottleneck=Stage.RETRIEVAL,
                hypothesis_check=HypothesisCheck(verdict="falsified"),
            ),
            meta=None,  # no next-trial proposal recorded yet
        )
        card = build_state_card(
            trial_number=4,
            trials_remaining=6,
            current_metrics=StageMetrics(retrieval_success=0.5, gold_in_reranker_window=0.8),
            current_score=0.5,
            history_records=[trial1, trial2, trial3],
        )

        assert card.interventions_tried == [
            ("embedding_model", "A", "B", "falsified"),
            ("embedding_model", "B", "C", "falsified"),
        ]

    def test_pivot_triggered_when_all_bottleneck_interventions_falsified(self) -> None:
        # Two consecutive falsified interventions on the generation lever
        # (llm_model) should route suggest_move_type to PIVOT via
        # _all_bottleneck_interventions_failed, even when
        # consecutive_non_improvements < 2.
        move = suggest_move_type(
            bottleneck=Stage.GENERATION,
            bottleneck_stable=True,
            consecutive_non_improvements=1,
            last_trial_delta=0.00,
            trials_remaining=5,
            interventions_tried=[
                ("llm_model", "A", "B", "falsified"),
                ("llm_model", "B", "C", "falsified"),
            ],
        )
        assert move == MoveType.PIVOT
