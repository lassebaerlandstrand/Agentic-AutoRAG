"""Tests for the mechanical evidence helpers used by the Diagnoser."""

from __future__ import annotations

import pytest

from agentic_autorag.config.models import IndexType, OpenEndedQuestion, TrialConfig
from agentic_autorag.examiner.evaluator import QuestionResult
from agentic_autorag.optimizer.diagnosis import (
    Diagnosis,
    ProposalMeta,
    Strategy,
    TrialMetrics,
)
from agentic_autorag.optimizer.history import TrialRecord
from agentic_autorag.optimizer.state import (
    build_failure_attribution,
    build_failure_cross_tab,
    compute_bundle_effect,
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
    generated_response: str = "B",
) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        correct=correct,
        selected_answer="A" if correct else "B",
        correct_answer="A",
        retrieved_context="",
        generated_response=generated_response,
        retrieved_spans=retrieved_spans,
        n_spans=n_spans,
        refused=refused,
    )


def _make_question(qid: str, reasoning_type: str = "bridge", n_hops: int = 2) -> OpenEndedQuestion:
    return OpenEndedQuestion(
        id=qid,
        question=f"question for {qid}",
        canonical_answer="alpha",
        reasoning_type=reasoning_type,  # type: ignore[arg-type]
        source_chunk_ids=[f"docA::{i}" for i in range(n_hops)],
        source_doc_ids=[f"docA{i}" for i in range(n_hops)],
        source_spans=[f"span {i}" for i in range(n_hops)],
    )


class TestBuildFailureAttribution:
    def test_no_failures_returns_zeros(self) -> None:
        attribution = build_failure_attribution(
            [_qr("q1", correct=True, retrieved_spans=2)],
        )
        assert attribution.retrieval == 0.0
        assert attribution.generation == 0.0

    def test_retrieval_miss_attributes_to_retrieval(self) -> None:
        attribution = build_failure_attribution(
            [
                _qr("q1", correct=False, retrieved_spans=0, n_spans=2),
                _qr("q2", correct=False, retrieved_spans=0, n_spans=2),
            ],
        )
        assert attribution.retrieval == 1.0
        assert attribution.generation == 0.0

    def test_retrieval_partial_attributes_to_retrieval(self) -> None:
        attribution = build_failure_attribution(
            [_qr("q1", correct=False, retrieved_spans=1, n_spans=2)],
        )
        assert attribution.retrieval == 1.0

    def test_generation_wrong_attributes_to_generation(self) -> None:
        attribution = build_failure_attribution(
            [_qr("q1", correct=False, retrieved_spans=2, n_spans=2)],
        )
        assert attribution.generation == 1.0
        assert attribution.retrieval == 0.0

    def test_refused_with_complete_retrieval_is_generation(self) -> None:
        attribution = build_failure_attribution(
            [_qr("q1", correct=False, retrieved_spans=2, n_spans=2, refused=True, generated_response="cannot answer")],
        )
        assert attribution.generation == 1.0
        assert attribution.retrieval == 0.0

    def test_refused_with_miss_is_retrieval(self) -> None:
        attribution = build_failure_attribution(
            [_qr("q1", correct=False, retrieved_spans=0, n_spans=2, refused=True, generated_response="cannot answer")],
        )
        assert attribution.retrieval == 1.0

    def test_mixed_fractions(self) -> None:
        attribution = build_failure_attribution(
            [
                _qr("q1", correct=False, retrieved_spans=0, n_spans=2),
                _qr("q2", correct=False, retrieved_spans=2, n_spans=2),
                _qr("q3", correct=False, retrieved_spans=1, n_spans=2),
                _qr("q4", correct=False, retrieved_spans=2, n_spans=2),
            ],
        )
        # 2/4 retrieval, 2/4 generation.
        assert attribution.retrieval == 0.5
        assert attribution.generation == 0.5

    def test_excludes_system_errors(self) -> None:
        from agentic_autorag.examiner._errors import TRANSIENT_ERROR_SENTINEL

        attribution = build_failure_attribution(
            [
                _qr("q1", correct=False, retrieved_spans=0, n_spans=2),
                _qr(
                    "q2",
                    correct=False,
                    retrieved_spans=0,
                    n_spans=2,
                    generated_response=TRANSIENT_ERROR_SENTINEL,
                ),
            ],
        )
        # Only q1 counts → all retrieval.
        assert attribution.retrieval == 1.0


class TestFailureCrossTab:
    def test_groups_by_mode_and_reasoning_type(self) -> None:
        results = [
            _qr("q1", correct=False, retrieved_spans=0, n_spans=2),
            _qr("q2", correct=False, retrieved_spans=0, n_spans=2),
            _qr("q3", correct=False, retrieved_spans=1, n_spans=2),
            _qr("q4", correct=False, retrieved_spans=2, n_spans=2),
        ]
        questions = [
            _make_question("q1", reasoning_type="bridge"),
            _make_question("q2", reasoning_type="bridge"),
            _make_question("q3", reasoning_type="comparison"),
            _make_question("q4", reasoning_type="numeric"),
        ]
        text = build_failure_cross_tab(results, questions)
        # Header counts total.
        assert "total failures: 4" in text
        # Each present cell renders.
        assert "retrieval_miss" in text
        assert "bridge" in text
        # The bridge-miss cell should report count=2.
        assert "retrieval_miss" in text
        # Counts go on the same line as the cell key.
        miss_line = next(line for line in text.splitlines() if "retrieval_miss" in line and "bridge" in line)
        assert miss_line.rstrip().endswith(": 2")

    def test_no_failures_returns_marker(self) -> None:
        text = build_failure_cross_tab([_qr("q1", correct=True, retrieved_spans=2)], [_make_question("q1")])
        assert "No failures" in text

    def test_unknown_question_falls_back_to_unknown_reasoning(self) -> None:
        results = [_qr("qX", correct=False, retrieved_spans=0, n_spans=2)]
        text = build_failure_cross_tab(results, [])
        # The cell rendering uses "unknown" when the question metadata is missing.
        assert "unknown" in text


def _make_record(
    trial_number: int,
    config: TrialConfig,
    metrics: TrialMetrics,
    cost: float,
) -> TrialRecord:
    return TrialRecord(
        trial_number=trial_number,
        config=config,
        answer_accuracy=metrics.answer_accuracy,
        question_results=[],
        trial_metrics=metrics,
        mean_llm_cost_per_query_usd=cost,
        meta=ProposalMeta(
            rationale="",
            strategy=Strategy(stance="explore"),
        ),
        diagnosis=Diagnosis(trial_metrics=metrics),
    )


class TestBundleEffect:
    def test_none_when_no_history(self) -> None:
        effect = compute_bundle_effect(
            history_records=[],
            current_config=_make_config(),
            current_metrics=TrialMetrics(answer_accuracy=0.5),
            current_cost_usd=0.01,
            anchor_trial=None,
        )
        assert effect is None

    def test_none_when_no_diff_against_anchor(self) -> None:
        config = _make_config()
        metrics = TrialMetrics(answer_accuracy=0.5)
        records = [_make_record(1, config, metrics, 0.01)]
        effect = compute_bundle_effect(
            history_records=records,
            current_config=config,
            current_metrics=TrialMetrics(answer_accuracy=0.6),
            current_cost_usd=0.012,
            anchor_trial=1,
        )
        assert effect is None

    def test_single_lever_difference(self) -> None:
        anchor_cfg = _make_config(top_k=5)
        cur_cfg = _make_config(top_k=10)
        anchor_metrics = TrialMetrics(
            answer_accuracy=0.5,
            answer_correct_given_complete_retrieval=0.8,
            retrieval_complete=0.6,
        )
        cur_metrics = TrialMetrics(
            answer_accuracy=0.6,
            answer_correct_given_complete_retrieval=0.75,
            retrieval_complete=0.7,
        )
        records = [_make_record(1, anchor_cfg, anchor_metrics, 0.005)]

        effect = compute_bundle_effect(
            history_records=records,
            current_config=cur_cfg,
            current_metrics=cur_metrics,
            current_cost_usd=0.010,
            anchor_trial=1,
        )

        assert effect is not None
        assert effect.changes == ["top_k: 5 → 10"]
        assert effect.accuracy_delta == pytest.approx(0.1)
        assert effect.acc_given_complete_delta == pytest.approx(-0.05)
        assert effect.retrieval_complete_delta == pytest.approx(0.1)
        assert effect.cost_delta_usd == pytest.approx(0.005)

    def test_falls_back_to_most_recent_when_anchor_missing(self) -> None:
        anchor_cfg = _make_config(top_k=5)
        cur_cfg = _make_config(top_k=10)
        anchor_metrics = TrialMetrics(answer_accuracy=0.5)
        records = [_make_record(1, anchor_cfg, anchor_metrics, 0.005)]
        effect = compute_bundle_effect(
            history_records=records,
            current_config=cur_cfg,
            current_metrics=TrialMetrics(answer_accuracy=0.6),
            current_cost_usd=0.010,
            anchor_trial=99,
        )
        assert effect is not None
        assert effect.changes == ["top_k: 5 → 10"]

    def test_multi_lever_bundle_collected_into_one_effect(self) -> None:
        anchor_cfg = _make_config(top_k=5, embedding_model="A")
        cur_cfg = _make_config(top_k=10, embedding_model="B")
        anchor_metrics = TrialMetrics(answer_accuracy=0.5)
        records = [_make_record(1, anchor_cfg, anchor_metrics, 0.005)]
        effect = compute_bundle_effect(
            history_records=records,
            current_config=cur_cfg,
            current_metrics=TrialMetrics(answer_accuracy=0.6),
            current_cost_usd=0.010,
            anchor_trial=1,
        )
        assert effect is not None
        assert set(effect.changes) == {"top_k: 5 → 10", "embedding_model: A → B"}
        # The bundle reports ONE set of deltas — not duplicated per-lever.
        assert effect.accuracy_delta == pytest.approx(0.1)
