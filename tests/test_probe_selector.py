"""Tests for the probe-based exam selection module (Phase 5)."""

from __future__ import annotations

import pytest

from agentic_autorag.config.models import MCQQuestion, ProjectConfig, TrialConfig
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.examiner.probe_selector import (
    score_questions_by_discrimination,
    select_exam,
    select_probe_configs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "meta": {
                "project_name": "test",
                "corpus_path": "/tmp",
                "output_dir": "/tmp/out",
            },
            "search_space": {
                "chunking": {
                    "strategies": ["recursive", "fixed"],
                    "chunk_size": {"min": 256, "max": 1024},
                    "chunk_overlap": {"min": 0, "max": 128},
                },
                "embedding_models": [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                ],
                "index_types": ["vector_only"],
                "top_k": {"min": 3, "max": 15},
                "hybrid_alpha": {"min": 0.0, "max": 1.0},
                "reranker": {"models": ["none"], "top_n": {"min": 3, "max": 5}},
                "query_expansion": ["none"],
                "llm_models": ["ollama/llama3.2", "ollama/mistral"],
                "temperature": {"min": 0.0, "max": 0.7},
            },
        }
    )


def _make_narrow_config() -> ProjectConfig:
    """Config with only one option for each parameter."""
    return ProjectConfig.model_validate(
        {
            "meta": {
                "project_name": "test",
                "corpus_path": "/tmp",
                "output_dir": "/tmp/out",
            },
            "search_space": {
                "chunking": {
                    "strategies": ["recursive"],
                    "chunk_size": {"min": 512, "max": 512},
                    "chunk_overlap": {"min": 0, "max": 0},
                },
                "embedding_models": ["sentence-transformers/all-MiniLM-L6-v2"],
                "index_types": ["vector_only"],
                "top_k": {"min": 5, "max": 5},
                "hybrid_alpha": {"min": 0.5, "max": 0.5},
                "reranker": {"models": ["none"], "top_n": {"min": 5, "max": 5}},
                "query_expansion": ["none"],
                "llm_models": ["ollama/llama3.2"],
                "temperature": {"min": 0.0, "max": 0.0},
            },
        }
    )


def _make_question(qid: str, cluster_id: int = 0) -> MCQQuestion:
    return MCQQuestion(
        id=qid,
        question=f"Question {qid}?",
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        correct_answer="A",
        source_doc_ids=["doc_0"],
        cluster_id=cluster_id,
    )


def _make_probe_result(question_ids: list[str], correct_ids: set[str]) -> ExamResult:
    results = [
        QuestionResult(
            question_id=qid,
            correct=qid in correct_ids,
            selected_answer="A" if qid in correct_ids else "B",
            correct_answer="A",
            retrieved_context="ctx",
            generated_response="A" if qid in correct_ids else "B",
        )
        for qid in question_ids
    ]
    n_correct = len(correct_ids)
    return ExamResult(
        score=n_correct / len(question_ids) if question_ids else 0.0,
        n_correct=n_correct,
        n_total=len(question_ids),
        question_results=results,
    )


# ---------------------------------------------------------------------------
# TestSelectProbeConfigs
# ---------------------------------------------------------------------------


class TestSelectProbeConfigs:
    def test_returns_list_of_trial_configs(self) -> None:
        config = _make_config()
        probes = select_probe_configs(config)
        assert len(probes) >= 1
        for p in probes:
            assert isinstance(p, TrialConfig)

    def test_probes_are_unique(self) -> None:
        config = _make_config()
        probes = select_probe_configs(config)
        # No two probes should be identical
        keys = [p.structural_fingerprint() + p.llm_model for p in probes]
        assert len(keys) == len(set(keys))

    def test_narrow_search_space_returns_at_least_one(self) -> None:
        config = _make_narrow_config()
        probes = select_probe_configs(config)
        assert len(probes) >= 1

    def test_probes_within_search_space(self) -> None:
        config = _make_config()
        ss = config.search_space
        probes = select_probe_configs(config)
        for p in probes:
            assert p.llm_model in ss.llm_models
            assert p.embedding_model in ss.embedding_models
            assert ss.chunking.chunk_size.min <= p.chunk_size <= ss.chunking.chunk_size.max

    def test_capped_at_n_probes(self) -> None:
        config = _make_config()
        probes = select_probe_configs(config)
        assert len(probes) <= 4


# ---------------------------------------------------------------------------
# TestScoreQuestionsByDiscrimination
# ---------------------------------------------------------------------------


class TestScoreQuestionsByDiscrimination:
    def test_all_correct_gives_zero_score(self) -> None:
        questions = [_make_question("q1"), _make_question("q2")]
        probe_result = _make_probe_result(["q1", "q2"], {"q1", "q2"})
        scores = score_questions_by_discrimination([probe_result], questions)
        assert scores["q1"] == pytest.approx(0.0)
        assert scores["q2"] == pytest.approx(0.0)

    def test_all_wrong_gives_zero_score(self) -> None:
        questions = [_make_question("q1"), _make_question("q2")]
        probe_result = _make_probe_result(["q1", "q2"], set())
        scores = score_questions_by_discrimination([probe_result], questions)
        assert scores["q1"] == pytest.approx(0.0)

    def test_mixed_results_give_positive_score(self) -> None:
        questions = [_make_question("q1"), _make_question("q2")]
        # Probe 1: q1 correct, q2 wrong
        probe1 = _make_probe_result(["q1", "q2"], {"q1"})
        # Probe 2: q1 wrong, q2 correct
        probe2 = _make_probe_result(["q1", "q2"], {"q2"})

        scores = score_questions_by_discrimination([probe1, probe2], questions)
        # Both q1 and q2 have mixed results → variance > 0
        assert scores["q1"] > 0.0
        assert scores["q2"] > 0.0

    def test_question_not_in_probe_treated_as_wrong(self) -> None:
        questions = [_make_question("q_missing")]
        probe_result = _make_probe_result(["q_other"], {"q_other"})
        scores = score_questions_by_discrimination([probe_result], questions)
        assert "q_missing" in scores
        assert scores["q_missing"] == pytest.approx(0.0)

    def test_empty_probe_results(self) -> None:
        questions = [_make_question("q1")]
        scores = score_questions_by_discrimination([], questions)
        assert scores == {"q1": 0.0}

    def test_empty_questions(self) -> None:
        scores = score_questions_by_discrimination([], [])
        assert scores == {}


# ---------------------------------------------------------------------------
# TestSelectExam
# ---------------------------------------------------------------------------


class TestSelectExam:
    def test_returns_up_to_exam_size(self) -> None:
        questions = [_make_question(f"q{i}") for i in range(20)]
        scores = {q.id: float(i) for i, q in enumerate(questions)}
        result = select_exam(questions, scores, exam_size=10)
        assert len(result) == 10

    def test_respects_cluster_diversity(self) -> None:
        # 10 questions: 5 from cluster 0, 5 from cluster 1
        qs_c0 = [_make_question(f"c0_q{i}", cluster_id=0) for i in range(5)]
        qs_c1 = [_make_question(f"c1_q{i}", cluster_id=1) for i in range(5)]
        all_qs = qs_c0 + qs_c1
        # All c0 have high scores, all c1 have low scores
        scores = {q.id: 1.0 for q in qs_c0}
        scores.update({q.id: 0.0 for q in qs_c1})

        result = select_exam(all_qs, scores, exam_size=6)
        cluster_ids = {q.cluster_id for q in result}
        # Both clusters should be represented
        assert 0 in cluster_ids
        assert 1 in cluster_ids

    def test_prefers_high_discrimination_within_cluster(self) -> None:
        questions = [_make_question(f"q{i}", cluster_id=0) for i in range(10)]
        # q9 has highest score
        scores = {f"q{i}": float(i) for i in range(10)}
        result = select_exam(questions, scores, exam_size=3)
        result_ids = {q.id for q in result}
        assert "q9" in result_ids
        assert "q8" in result_ids
        assert "q7" in result_ids

    def test_selects_all_when_fewer_than_exam_size(self) -> None:
        questions = [_make_question(f"q{i}") for i in range(3)]
        scores = {q.id: 1.0 for q in questions}
        result = select_exam(questions, scores, exam_size=10)
        assert len(result) == 3

    def test_empty_candidates(self) -> None:
        result = select_exam([], {}, exam_size=10)
        assert result == []

    def test_no_negative_scores_accepted(self) -> None:
        questions = [_make_question(f"q{i}") for i in range(5)]
        # Some questions have no score entry
        scores = {"q0": 0.5, "q1": 0.3}
        result = select_exam(questions, scores, exam_size=3)
        assert len(result) == 3
        assert all(isinstance(q, MCQQuestion) for q in result)

    def test_global_fill_used_when_cluster_quota_falls_short(self) -> None:
        """When cluster allocation < exam_size, global fill picks remaining."""
        # 3 questions in cluster 0, 2 in cluster 1, exam_size=6
        qs_c0 = [_make_question(f"c0_q{i}", cluster_id=0) for i in range(3)]
        qs_c1 = [_make_question(f"c1_q{i}", cluster_id=1) for i in range(2)]
        all_qs = qs_c0 + qs_c1
        scores = {q.id: 1.0 for q in all_qs}
        result = select_exam(all_qs, scores, exam_size=6)
        # All 5 available are returned (capped at available)
        assert len(result) == 5
