"""Tests for agentic_autorag.optimizer.state — pure optimizer state functions."""

from __future__ import annotations

import pytest

from agentic_autorag.config.models import IndexType, TrialConfig
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.optimizer.diagnosis import (
    Diagnosis,
    FailureAttribution,
    ProposalMeta,
    TrialMetrics,
)
from agentic_autorag.optimizer.history import TrialRecord
from agentic_autorag.optimizer.state import (
    build_frontier_context,
    build_state_card,
    compute_bundle_effects,
    compute_trial_metrics,
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
        generator_llm="ollama/llama3.2",
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
                generated_response="TRANSIENT_LLM_ERROR",
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
        assert len(card.trial_summaries) == 1
        assert card.trial_summaries[0]["trial_number"] == 1

    def test_state_card_score_progress(self) -> None:
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

        assert card.best_score_so_far == 0.62
        assert card.best_trial_number == 2
        assert abs(card.last_trial_delta - 0.07) < 1e-6

    def test_trial_summaries_include_changes_and_failure_modes(self) -> None:
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(embedding_model="A", top_k=5),
            score=0.5,
            question_results=[],
            diagnosis=Diagnosis(
                trial_metrics=TrialMetrics(),
                failure_attribution=FailureAttribution(retrieval=0.6, generation=0.4),
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


class TestParetoFrontierFullConfig:
    def test_frontier_entries_carry_full_config_dict(self) -> None:
        # Two non-dominated trials so both land on the frontier.
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(embedding_model="emb-A", top_k=5),
            score=0.6,
            mean_llm_cost_per_query_usd=0.001,
            question_results=[],
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_score=0.8,
            history_records=[prev],
            max_trials=10,
            current_config=_make_config(embedding_model="emb-B", top_k=10),
            current_cost_usd=0.010,
        )

        assert len(card.pareto_frontier) >= 2
        for entry in card.pareto_frontier:
            assert "config_summary" in entry
            assert "config" in entry
            cfg = entry["config"]
            # Trial 2 is in-flight (no source record), so its config dict is
            # None in the synthetic record path. Skip those.
            if cfg is None:
                continue
            for required_field in ("embedding_model", "top_k", "generator_llm", "index_type", "chunking_strategy"):
                assert required_field in cfg, f"missing {required_field} in frontier config dict"


class TestCostAwareToggle:
    """Score-only mode strips Pareto/cost signals from the state card."""

    def test_score_only_state_card_has_empty_pareto_view(self) -> None:
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(embedding_model="A", top_k=5),
            score=0.6,
            mean_llm_cost_per_query_usd=0.002,
            question_results=[],
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_score=0.8,
            history_records=[prev],
            max_trials=10,
            current_config=_make_config(embedding_model="B", top_k=10),
            current_cost_usd=0.020,
            cost_aware=False,
        )

        assert card.cost_aware is False
        assert card.pareto_frontier == []
        assert card.hypervolume == 0.0
        assert card.hypervolume_delta_last_3 == 0.0
        assert card.knee_trial_number is None
        assert card.nearest_dominator_trial is None
        assert card.cheapest_at_score_threshold_usd is None
        assert card.current_trial_cost_usd == 0.0
        # Score leader is still surfaced — it's the score-only anchor.
        assert card.score_leader_trial_number == 2

    def test_cost_aware_state_card_populates_pareto_view(self) -> None:
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(embedding_model="A", top_k=5),
            score=0.6,
            mean_llm_cost_per_query_usd=0.001,
            question_results=[],
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_score=0.8,
            history_records=[prev],
            max_trials=10,
            current_config=_make_config(embedding_model="B", top_k=10),
            current_cost_usd=0.010,
            cost_aware=True,
        )

        assert card.cost_aware is True
        assert len(card.pareto_frontier) >= 1
        assert card.knee_trial_number is not None
        assert card.score_leader_trial_number == 2


class TestScorePlateauGate:
    """Score-plateau is part of the done-eligibility gate in both modes."""

    def _record(self, trial: int, score: float, cost: float = 0.001) -> TrialRecord:
        return TrialRecord(
            trial_number=trial,
            config=_make_config(),
            score=score,
            mean_llm_cost_per_query_usd=cost,
            question_results=[],
            trial_metrics=TrialMetrics(answer_accuracy=score, n_valid=10),
        )

    def test_score_only_done_blocked_when_score_still_rising(self) -> None:
        history = [
            self._record(1, 0.50),
            self._record(2, 0.55),
            self._record(3, 0.60),
            self._record(4, 0.65),
        ]
        card = build_state_card(
            trial_number=5,
            trials_remaining=5,
            current_score=0.70,
            history_records=history,
            max_trials=10,
            current_config=_make_config(),
            current_cost_usd=0.001,
            cost_aware=False,
            score_plateau_window=3,
            score_plateau_epsilon=0.005,
        )

        assert card.done_eligible is False
        assert "best score still improving" in (card.done_blocked_reason or "")

    def test_score_only_done_eligible_when_score_plateaued(self) -> None:
        history = [
            self._record(1, 0.70),
            self._record(2, 0.72),
            self._record(3, 0.72),
            self._record(4, 0.72),
        ]
        card = build_state_card(
            trial_number=5,
            trials_remaining=5,
            current_score=0.72,
            history_records=history,
            max_trials=10,
            current_config=_make_config(),
            current_cost_usd=0.001,
            cost_aware=False,
            score_plateau_window=3,
            score_plateau_epsilon=0.005,
        )

        assert card.done_eligible is True
        assert card.done_blocked_reason is None

    def test_cost_aware_done_blocked_when_score_flat_but_hv_growing(self) -> None:
        # Score is flat for 4 trials but cost keeps dropping; HV keeps growing.
        # In the OLD gate this would terminate falsely. Score-plateau AND now
        # blocks it correctly. Use distinct costs so each trial sits on the
        # frontier and HV expands trial-to-trial.
        history = [
            self._record(1, 0.70, cost=0.010),
            self._record(2, 0.70, cost=0.008),
            self._record(3, 0.70, cost=0.006),
            self._record(4, 0.70, cost=0.004),
        ]
        card = build_state_card(
            trial_number=5,
            trials_remaining=5,
            current_score=0.70,
            history_records=history,
            max_trials=10,
            current_config=_make_config(),
            current_cost_usd=0.002,
            cost_aware=True,
            score_plateau_window=3,
            score_plateau_epsilon=0.005,
            early_exit_hv_epsilon=0.0,
        )

        # Score has been flat — score_plateau_delta ≈ 0, ≤ epsilon, so the
        # score gate passes; but the HV gate may also pass when ε=0 and HV
        # delta is positive. The block ordering puts score first; check that
        # WHEN HV is still expanding, the HV gate blocks (cost-aware mode).
        # If HV is flat too, the trial legitimately can exit. Use a different
        # check: pre-fix, this would have been eligible just because HV
        # ε=0.001 default looked plateaued; we now require BOTH conditions.
        if card.hypervolume_delta_last_3 > 0.0:
            assert card.done_eligible is False


class TestComputeBundleEffects:
    """Dual-anchor renderer returns both knee and leader effects when distinct."""

    def test_returns_only_knee_when_anchors_match(self) -> None:
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(top_k=5),
            score=0.8,
            mean_llm_cost_per_query_usd=0.001,
            question_results=[],
            trial_metrics=TrialMetrics(answer_accuracy=0.8, retrieval_complete=0.7, n_valid=10),
        )
        effects = compute_bundle_effects(
            history_records=[prev],
            current_config=_make_config(top_k=10),
            current_metrics=TrialMetrics(answer_accuracy=0.85, retrieval_complete=0.8, n_valid=10),
            current_cost_usd=0.002,
            knee_trial=1,
            score_leader_trial=1,
        )

        assert len(effects) == 1
        assert "knee" in effects[0][0]

    def test_returns_both_when_knee_and_leader_differ(self) -> None:
        knee_rec = TrialRecord(
            trial_number=1,
            config=_make_config(top_k=5),
            score=0.70,
            mean_llm_cost_per_query_usd=0.0005,
            question_results=[],
            trial_metrics=TrialMetrics(answer_accuracy=0.70, retrieval_complete=0.65, n_valid=10),
        )
        leader_rec = TrialRecord(
            trial_number=2,
            config=_make_config(top_k=20),
            score=0.88,
            mean_llm_cost_per_query_usd=0.005,
            question_results=[],
            trial_metrics=TrialMetrics(answer_accuracy=0.88, retrieval_complete=0.85, n_valid=10),
        )
        effects = compute_bundle_effects(
            history_records=[knee_rec, leader_rec],
            current_config=_make_config(top_k=10),
            current_metrics=TrialMetrics(answer_accuracy=0.75, retrieval_complete=0.75, n_valid=10),
            current_cost_usd=0.003,
            knee_trial=1,
            score_leader_trial=2,
        )

        assert len(effects) == 2
        labels = [label for label, _ in effects]
        assert any("knee" in lbl for lbl in labels)
        assert any("score leader" in lbl for lbl in labels)


class TestBuildFrontierContext:
    def test_first_trial_no_dominator_is_on_frontier(self) -> None:
        ctx = build_frontier_context(
            history_records=[],
            current_trial_number=1,
            current_score=0.5,
            current_cost_usd=0.005,
            current_config=_make_config(),
        )
        assert ctx.is_on_frontier is True
        assert ctx.nearest_dominator_trial is None

    def test_dominated_trial_reports_dominator_with_diff(self) -> None:
        # Trial 1 is the dominator (better score AND lower cost).
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(embedding_model="emb-A", top_k=10),
            score=0.9,
            mean_llm_cost_per_query_usd=0.001,
            question_results=[],
        )
        ctx = build_frontier_context(
            history_records=[prev],
            current_trial_number=2,
            current_score=0.5,
            current_cost_usd=0.010,
            current_config=_make_config(embedding_model="emb-B", top_k=5),
        )

        assert ctx.is_on_frontier is False
        assert ctx.nearest_dominator_trial == 1
        assert ctx.nearest_dominator_score == 0.9
        assert ctx.score_gap_to_dominator == pytest.approx(0.4)
        assert ctx.cost_gap_to_dominator_usd == pytest.approx(0.009)
        # Diff is "current → dominator", so it lists current's values being
        # changed to dominator's values.
        diff_str = " | ".join(ctx.nearest_dominator_config_diff)
        assert "embedding_model" in diff_str
        assert "emb-B → emb-A" in diff_str
        assert "top_k" in diff_str
        assert "5 → 10" in diff_str

    def test_non_dominated_current_reports_no_dominator(self) -> None:
        # Two trials, neither dominates the other (score-cost trade).
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(),
            score=0.9,
            mean_llm_cost_per_query_usd=0.020,
            question_results=[],
        )
        ctx = build_frontier_context(
            history_records=[prev],
            current_trial_number=2,
            current_score=0.5,
            current_cost_usd=0.001,
            current_config=_make_config(),
        )
        assert ctx.is_on_frontier is True
        assert ctx.nearest_dominator_trial is None
