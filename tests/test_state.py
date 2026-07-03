"""Tests for agentic_autorag.optimizer.state — pure optimizer state functions."""

from __future__ import annotations

import pytest

from agentic_autorag.config.models import IndexType, TrialConfig
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.optimizer.diagnosis import (
    Diagnosis,
    ProposalMeta,
    TrialMetrics,
)
from agentic_autorag.optimizer.history import TrialRecord
from agentic_autorag.optimizer.state import (
    build_state_card,
    compute_bundle_effect,
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
        result = ExamResult(answer_accuracy=0.0, n_correct=0, n_total=0, question_results=[])

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
        exam_result = ExamResult(answer_accuracy=0.25, n_correct=2, n_total=8, question_results=results)

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
        exam_result = ExamResult(answer_accuracy=1.0, n_correct=1, n_total=2, question_results=results)

        m = compute_trial_metrics(exam_result)

        assert m.n_valid == 1
        assert m.retrieval_complete == 1.0

    def test_acc_given_complete_zero_when_no_complete(self) -> None:
        results = [_qr("q1", correct=False, retrieved_spans=0, n_spans=2)]
        exam_result = ExamResult(answer_accuracy=0.0, n_correct=0, n_total=1, question_results=results)

        m = compute_trial_metrics(exam_result)

        assert m.answer_correct_given_complete_retrieval == 0.0


class TestBuildStateCard:
    def test_first_trial_no_history_in_search(self) -> None:
        card = build_state_card(
            trial_number=1,
            trials_remaining=9,
            current_accuracy=0.55,
            history_records=[],
            current_config=_make_config(),
        )

        assert card.trial_number == 1
        assert card.best_accuracy_so_far == 0.55
        assert card.last_trial_delta == 0.0
        assert len(card.trial_summaries) == 1
        assert card.trial_summaries[0]["trial_number"] == 1

    def test_state_card_score_progress(self) -> None:
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(embedding_model="A"),
            answer_accuracy=0.55,
            question_results=[],
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_accuracy=0.62,
            history_records=[prev],
            current_config=_make_config(embedding_model="B"),
        )

        assert card.best_accuracy_so_far == 0.62
        assert card.best_trial_number == 2
        assert abs(card.last_trial_delta - 0.07) < 1e-6

    def test_hv_delta_window_controls_lookback(self) -> None:
        """``hv_delta_window`` parameterises the HV-Δ lookback surfaced in the
        cost-aware Pareto state card. With a larger window, the card compares
        against an earlier trial's HV — surfacing HV growth that a tighter
        window misses. Informational only (no termination gating)."""
        records: list[TrialRecord] = []
        # Trials 1..5 with monotonically improving (score, cost) frontier.
        cost_steps = [0.05, 0.04, 0.03, 0.02, 0.01]
        score_steps = [0.20, 0.30, 0.40, 0.50, 0.60]
        for i, (cost, score) in enumerate(zip(cost_steps, score_steps, strict=True), start=1):
            records.append(
                TrialRecord(
                    trial_number=i,
                    config=_make_config(),
                    answer_accuracy=score,
                    question_results=[],
                    mean_llm_cost_per_query_usd=cost,
                )
            )

        card_w1 = build_state_card(
            trial_number=6,
            trials_remaining=4,
            current_accuracy=0.65,
            history_records=records,
            current_config=_make_config(),
            cost_aware=True,
            current_cost_usd=0.005,
            hv_delta_window=1,
        )
        card_w4 = build_state_card(
            trial_number=6,
            trials_remaining=4,
            current_accuracy=0.65,
            history_records=records,
            current_config=_make_config(),
            cost_aware=True,
            current_cost_usd=0.005,
            hv_delta_window=4,
        )
        # A wider lookback captures more accumulated HV expansion.
        assert card_w4.hypervolume_delta_last_3 > card_w1.hypervolume_delta_last_3

    def test_trial_summaries_include_changes_and_failure_modes(self) -> None:
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(embedding_model="A", top_k=5),
            answer_accuracy=0.5,
            question_results=[
                QuestionResult(
                    question_id=f"q{i}",
                    correct=False,
                    selected_answer="",
                    correct_answer="",
                    retrieved_context="",
                    generated_response="wrong",
                    retrieved_spans=0,
                    n_spans=1,
                )
                for i in range(3)
            ],  # All failures → retrieval mode
            diagnosis=Diagnosis(
                trial_metrics=TrialMetrics(),
            ),
            meta=ProposalMeta(rationale="…"),
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_accuracy=0.6,
            history_records=[prev],
            current_config=_make_config(embedding_model="B", top_k=10),
            current_top_failure_modes=["ranking", "generation"],
        )

        # Two summaries: prev trial + current trial
        assert len(card.trial_summaries) == 2
        prev_summary = card.trial_summaries[0]
        assert prev_summary["trial_number"] == 1
        # All 3 failures are retrieval (no spans retrieved) → retrieval comes first.
        assert prev_summary["top_failure_modes"][0] == "retrieval"
        cur_summary = card.trial_summaries[1]
        assert cur_summary["trial_number"] == 2
        assert any("embedding_model" in c for c in cur_summary["what_changed_from_prev"])
        assert any("top_k" in c for c in cur_summary["what_changed_from_prev"])
        assert cur_summary["top_failure_modes"] == ["ranking", "generation"]

    def test_trial_summaries_include_retrieval_complete(self) -> None:
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(),
            answer_accuracy=0.5,
            question_results=[],
            trial_metrics=TrialMetrics(answer_accuracy=0.5, retrieval_complete=0.73, n_valid=10),
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_accuracy=0.6,
            history_records=[prev],
            current_config=_make_config(),
        )

        prev_summary = card.trial_summaries[0]
        assert prev_summary["retrieval_complete"] == pytest.approx(0.73)

    def test_trial_summaries_retrieval_complete_zero_when_metrics_missing(self) -> None:
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(),
            answer_accuracy=0.5,
            question_results=[],
            trial_metrics=None,
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_accuracy=0.6,
            history_records=[prev],
            current_config=_make_config(),
        )

        assert card.trial_summaries[0]["retrieval_complete"] == 0.0

    def test_current_trial_summary_includes_retrieval_complete(self) -> None:
        # The current trial isn't yet persisted to history when build_state_card
        # runs; it's appended as a synthetic dict. Verify retrieval_complete
        # flows through that path.
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(),
            answer_accuracy=0.5,
            question_results=[],
            trial_metrics=TrialMetrics(answer_accuracy=0.5, retrieval_complete=0.4, n_valid=10),
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_accuracy=0.8,
            history_records=[prev],
            current_config=_make_config(),
            current_retrieval_complete=0.93,
        )

        assert len(card.trial_summaries) == 2
        assert card.trial_summaries[-1]["trial_number"] == 2
        assert card.trial_summaries[-1]["retrieval_complete"] == pytest.approx(0.93)


class TestTrialsSinceBestScore:
    def test_zero_when_current_trial_is_best(self) -> None:
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(),
            answer_accuracy=0.4,
            question_results=[],
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_accuracy=0.7,
            history_records=[prev],
            current_config=_make_config(),
        )
        assert card.best_trial_number == 2
        assert card.trials_since_best_accuracy == 0

    def test_counts_elapsed_trials_since_best(self) -> None:
        records = [
            TrialRecord(trial_number=i, config=_make_config(), answer_accuracy=s, question_results=[])
            for i, s in enumerate([0.4, 0.9, 0.6, 0.55, 0.5], start=1)
        ]
        card = build_state_card(
            trial_number=6,
            trials_remaining=4,
            current_accuracy=0.45,
            history_records=records,
            current_config=_make_config(),
        )
        assert card.best_trial_number == 2
        assert card.trials_since_best_accuracy == 4

    def test_zero_when_no_history(self) -> None:
        card = build_state_card(
            trial_number=1,
            trials_remaining=9,
            current_accuracy=0.55,
            history_records=[],
            current_config=_make_config(),
        )
        assert card.trials_since_best_accuracy == 0


class TestSearchSpaceCoverage:
    def test_empty_when_sizes_not_supplied(self) -> None:
        card = build_state_card(
            trial_number=1,
            trials_remaining=9,
            current_accuracy=0.5,
            history_records=[],
            current_config=_make_config(),
        )
        assert card.coverage == []

    def test_counts_distinct_values_across_history_and_current(self) -> None:
        prev_a = TrialRecord(
            trial_number=1,
            config=_make_config(generator_llm="azure/gpt-5-mini", embedding_model="emb-A", reranker="none"),
            answer_accuracy=0.5,
            question_results=[],
        )
        prev_b = TrialRecord(
            trial_number=2,
            config=_make_config(generator_llm="azure/gpt-5-mini", embedding_model="emb-B", reranker="none"),
            answer_accuracy=0.6,
            question_results=[],
        )
        card = build_state_card(
            trial_number=3,
            trials_remaining=7,
            current_accuracy=0.7,
            history_records=[prev_a, prev_b],
            current_config=_make_config(
                generator_llm="azure/gpt-5.4-mini",
                embedding_model="emb-A",
                reranker="BAAI/bge-reranker-v2-m3",
            ),
            search_space_sizes={
                "generator_llm": 13,
                "embedding_model": 4,
                "reranker": 5,
            },
        )
        by_label = {entry["label"]: entry for entry in card.coverage}
        assert by_label["generators"] == {"label": "generators", "tried": 2, "total": 13}
        assert by_label["embeddings"] == {"label": "embeddings", "tried": 2, "total": 4}
        assert by_label["rerankers"] == {"label": "rerankers", "tried": 2, "total": 5}

    def test_skips_levers_with_zero_total(self) -> None:
        card = build_state_card(
            trial_number=1,
            trials_remaining=9,
            current_accuracy=0.5,
            history_records=[],
            current_config=_make_config(),
            search_space_sizes={
                "generator_llm": 3,
                "embedding_model": 0,
                "reranker": 1,
            },
        )
        labels = [entry["label"] for entry in card.coverage]
        assert "generators" in labels
        assert "rerankers" in labels
        assert "embeddings" not in labels


class TestParetoFrontierFullConfig:
    def test_frontier_entries_carry_full_config_dict(self) -> None:
        # Two non-dominated trials so both land on the frontier.
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(embedding_model="emb-A", top_k=5),
            answer_accuracy=0.6,
            mean_llm_cost_per_query_usd=0.001,
            question_results=[],
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_accuracy=0.8,
            history_records=[prev],
            current_config=_make_config(embedding_model="emb-B", top_k=10),
            current_cost_usd=0.010,
        )

        assert len(card.pareto_frontier) >= 2
        for entry in card.pareto_frontier:
            assert "config_summary" in entry
            assert "config" in entry
            assert "in_tok" in entry
            assert "out_tok" in entry
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
            answer_accuracy=0.6,
            mean_llm_cost_per_query_usd=0.002,
            question_results=[],
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_accuracy=0.8,
            history_records=[prev],
            current_config=_make_config(embedding_model="B", top_k=10),
            current_cost_usd=0.020,
            cost_aware=False,
        )

        assert card.cost_aware is False
        assert card.pareto_frontier == []
        assert card.hypervolume == 0.0
        assert card.hypervolume_delta_last_3 == 0.0
        assert card.current_trial_cost_usd == 0.0
        # Best-score trial is still surfaced — universal anchor in both modes.
        assert card.best_trial_number == 2

    def test_cost_aware_state_card_populates_pareto_view(self) -> None:
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(embedding_model="A", top_k=5),
            answer_accuracy=0.6,
            mean_llm_cost_per_query_usd=0.001,
            question_results=[],
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_accuracy=0.8,
            history_records=[prev],
            current_config=_make_config(embedding_model="B", top_k=10),
            current_cost_usd=0.010,
            cost_aware=True,
        )

        assert card.cost_aware is True
        assert len(card.pareto_frontier) >= 1
        assert card.best_trial_number == 2


class TestTrialsSinceFrontierImproved:
    """The frontier-stall counter: trailing trials whose config is dominated.

    The current (in-flight) trial is included as a synthetic record, so the
    counter reflects the position of the latest trial relative to the frontier.
    """

    def test_zero_when_current_trial_extends_frontier(self) -> None:
        # Higher score at higher cost → current trial is non-dominated.
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(),
            answer_accuracy=0.6,
            mean_llm_cost_per_query_usd=0.002,
            question_results=[],
        )
        card = build_state_card(
            trial_number=2,
            trials_remaining=8,
            current_accuracy=0.8,
            history_records=[prev],
            current_config=_make_config(),
            current_cost_usd=0.010,
            cost_aware=True,
        )
        assert card.trials_since_frontier_improved == 0

    def test_counts_trailing_dominated_trials(self) -> None:
        # Trial 1 dominates everything cheaper-and-lower-scoring; trials 2 and
        # the current trial both land dominated, so the counter is 2.
        records = [
            TrialRecord(
                trial_number=1,
                config=_make_config(),
                answer_accuracy=0.9,
                mean_llm_cost_per_query_usd=0.001,
                question_results=[],
            ),
            TrialRecord(
                trial_number=2,
                config=_make_config(),
                answer_accuracy=0.6,
                mean_llm_cost_per_query_usd=0.002,
                question_results=[],
            ),
        ]
        card = build_state_card(
            trial_number=3,
            trials_remaining=7,
            current_accuracy=0.7,
            history_records=records,
            current_config=_make_config(),
            current_cost_usd=0.004,
            cost_aware=True,
        )
        assert card.trials_since_frontier_improved == 2

    def test_resets_when_latest_trial_lands_new_point(self) -> None:
        # Trial 2 is dominated, but the current trial sets a new ceiling.
        records = [
            TrialRecord(
                trial_number=1,
                config=_make_config(),
                answer_accuracy=0.9,
                mean_llm_cost_per_query_usd=0.001,
                question_results=[],
            ),
            TrialRecord(
                trial_number=2,
                config=_make_config(),
                answer_accuracy=0.5,
                mean_llm_cost_per_query_usd=0.002,
                question_results=[],
            ),
        ]
        card = build_state_card(
            trial_number=3,
            trials_remaining=7,
            current_accuracy=0.95,
            history_records=records,
            current_config=_make_config(),
            current_cost_usd=0.003,
            cost_aware=True,
        )
        assert card.trials_since_frontier_improved == 0

    def test_zero_when_no_history(self) -> None:
        card = build_state_card(
            trial_number=1,
            trials_remaining=9,
            current_accuracy=0.55,
            history_records=[],
            current_config=_make_config(),
            current_cost_usd=0.001,
            cost_aware=True,
        )
        assert card.trials_since_frontier_improved == 0


class TestComputeBundleEffect:
    """Single-anchor bundle effect (best-score anchor; the dual-anchor variant was removed)."""

    def test_returns_delta_against_named_anchor(self) -> None:
        prev = TrialRecord(
            trial_number=1,
            config=_make_config(top_k=5),
            answer_accuracy=0.8,
            mean_llm_cost_per_query_usd=0.001,
            question_results=[],
            trial_metrics=TrialMetrics(answer_accuracy=0.8, retrieval_complete=0.7, n_valid=10),
        )
        effect = compute_bundle_effect(
            history_records=[prev],
            current_config=_make_config(top_k=10),
            current_metrics=TrialMetrics(answer_accuracy=0.85, retrieval_complete=0.8, n_valid=10),
            current_cost_usd=0.002,
            anchor_trial=1,
        )

        assert effect is not None
        assert any("top_k" in c for c in effect.changes)
        assert effect.accuracy_delta == pytest.approx(0.05, abs=1e-6)
