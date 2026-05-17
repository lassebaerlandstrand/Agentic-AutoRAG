"""Tests for the probe-based exam selection module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_autorag.config.models import OpenEndedQuestion, ProjectConfig, TrialConfig
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.examiner.probe_selector import (
    attach_probe_metadata,
    collect_probe_outcomes,
    rank_models_for_probes,
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
                    "chunk_token_size": {"min": 256, "max": 1024},
                    "chunk_token_overlap": {"min": 0, "max": 128},
                },
                "embedding_models": [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                ],
                "index_types": ["vector_only"],
                "top_k": {"min": 3, "max": 15},
                "hybrid_alpha": {"min": 0.0, "max": 1.0},
                "reranker": {
                    "models": ["none", "BAAI/bge-reranker-v2-m3"],
                    "top_n": {"min": 3, "max": 10},
                },
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
                    "chunk_token_size": {"min": 512, "max": 512},
                    "chunk_token_overlap": {"min": 0, "max": 0},
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


def _make_question(qid: str, cluster_id: int = 0) -> OpenEndedQuestion:
    return OpenEndedQuestion(
        id=qid,
        question=f"Question {qid}?",
        canonical_answer=f"answer_{qid}",
        reasoning_type="bridge",
        source_chunk_ids=[f"doc_a::chunk_0_{qid}", f"doc_b::chunk_0_{qid}"],
        source_doc_ids=["doc_a", "doc_b"],
        source_spans=["span A text", "span B text"],
        cluster_id=cluster_id,
    )


def _make_probe_result(question_ids: list[str], correct_ids: set[str]) -> ExamResult:
    results = [
        QuestionResult(
            question_id=qid,
            correct=qid in correct_ids,
            selected_answer=f"answer_{qid}" if qid in correct_ids else "wrong",
            correct_answer=f"answer_{qid}",
            retrieved_context="ctx",
            generated_response=f"answer_{qid}" if qid in correct_ids else "wrong",
            em=1.0 if qid in correct_ids else 0.0,
            f1=1.0 if qid in correct_ids else 0.0,
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
    def test_returns_labelled_trial_configs(self) -> None:
        config = _make_config()
        probes = select_probe_configs(config)
        assert len(probes) >= 1
        for label, tc in probes:
            assert isinstance(label, str)
            assert isinstance(tc, TrialConfig)

    def test_labels_contain_tier_name(self) -> None:
        config = _make_config()
        probes = select_probe_configs(config)
        labels = [label for label, _ in probes]
        tier_prefixes = {"Tier1-weak", "Tier2-lower-mid", "Tier3-upper-mid", "Tier4-strong"}
        for label in labels:
            assert any(label.startswith(t) for t in tier_prefixes), f"Label '{label}' missing tier prefix"

    def test_probes_are_unique(self) -> None:
        config = _make_config()
        probes = select_probe_configs(config)
        keys = [p.structural_fingerprint() + p.generator_llm + p.reranker for _, p in probes]
        assert len(keys) == len(set(keys))

    def test_narrow_search_space_returns_at_least_one(self) -> None:
        config = _make_narrow_config()
        probes = select_probe_configs(config)
        assert len(probes) >= 1

    def test_probes_within_search_space(self) -> None:
        config = _make_config()
        ss = config.search_space
        probes = select_probe_configs(config)
        for _, p in probes:
            assert p.generator_llm in ss.llm_models
            assert p.embedding_model in ss.embedding_models
            assert ss.chunking.chunk_token_size.min <= p.chunk_token_size <= ss.chunking.chunk_token_size.max

    def test_max_four_probes(self) -> None:
        config = _make_config()
        probes = select_probe_configs(config)
        assert len(probes) <= 4

    def test_strong_probe_has_reranker(self) -> None:
        """When search space has non-none rerankers, the strong probe uses one."""
        config = _make_config()
        probes = select_probe_configs(config)
        rerankers_used = {p.reranker for _, p in probes}
        assert "BAAI/bge-reranker-v2-m3" in rerankers_used

    def test_weak_probe_has_no_reranker(self) -> None:
        config = _make_config()
        probes = select_probe_configs(config)
        _, weak = probes[0]
        assert weak.reranker == "none"

    def test_no_reranker_when_only_none(self) -> None:
        config = _make_narrow_config()
        probes = select_probe_configs(config)
        for _, p in probes:
            assert p.reranker == "none"

    def test_uses_ranked_lists(self) -> None:
        """When ranked lists are provided, weak/strong picks follow them."""
        config = _make_config()
        ranked_llms = ["ollama/mistral", "ollama/llama3.2"]
        probes = select_probe_configs(config, ranked_llms=ranked_llms)
        _, weak = probes[0]
        assert weak.generator_llm == "ollama/mistral"

    def test_falls_back_to_search_space_without_ranked_lists(self) -> None:
        config = _make_config()
        probes = select_probe_configs(config)
        _, weak = probes[0]
        assert weak.generator_llm == config.search_space.llm_models[0]

    def test_four_distinct_llm_tiers_with_enough_models(self) -> None:
        config = _make_config()
        ranked_llms = ["weak/a", "mid_low/b", "mid_high/c", "strong/d"]
        probes = select_probe_configs(config, ranked_llms=ranked_llms)
        # Probes are emitted weakest-first; the rank-correlation discriminator
        # depends on this ordering, so the tier->llm mapping must be ordinal
        # and all four tier slots filled.
        tier_to_llm = {label.split(" ")[0]: tc.generator_llm for label, tc in probes}
        assert tier_to_llm["Tier1-weak"] == "weak/a"
        assert tier_to_llm["Tier2-lower-mid"] == "mid_low/b"
        assert tier_to_llm["Tier3-upper-mid"] == "strong/d"  # 3*4//4 == 3 = last index
        assert tier_to_llm["Tier4-strong"] == "strong/d"
        # Tiers are emitted in weakest→strongest order.
        ordered_llms = [tc.generator_llm for _, tc in probes]
        assert ordered_llms == ["weak/a", "mid_low/b", "strong/d", "strong/d"]

    def test_strong_tier_pairs_strong_llm_with_strong_retrieval(self) -> None:
        config = _make_config()
        ranked_llms = ["weak/a", "mid_low/b", "mid_high/c", "strong/d"]
        probes = select_probe_configs(config, ranked_llms=ranked_llms)
        strong = next((tc for label, tc in probes if label.startswith("Tier4-strong")), None)
        assert strong is not None
        assert strong.generator_llm == "strong/d"
        assert strong.reranker == "BAAI/bge-reranker-v2-m3"
        assert strong.embedding_model == config.search_space.embedding_models[-1]

    def test_chunk_token_size_capped_at_embedding_limit(self) -> None:
        """Probe chunk_token_size must not exceed the embedding model's max_tokens."""
        config = _make_config()
        config.embedding_token_limits = {
            "sentence-transformers/all-MiniLM-L6-v2": 256,
            "sentence-transformers/all-mpnet-base-v2": 512,
        }
        probes = select_probe_configs(config)
        for _, p in probes:
            limit = config.embedding_token_limits.get(p.embedding_model)
            if limit is not None:
                assert p.chunk_token_size <= limit, (
                    f"Probe chunk_token_size {p.chunk_token_size} exceeds {p.embedding_model} limit of {limit}"
                )


# ---------------------------------------------------------------------------
# TestRankModelsForProbes
# ---------------------------------------------------------------------------


class TestRankModelsForProbes:
    async def test_kb_sufficient_coverage(self) -> None:
        """When KB covers >= 3 models, uses KB ranking."""
        kb = MagicMock()
        kb.rank_llms.return_value = (["weak", "mid", "strong"], [])
        result = await rank_models_for_probes(["strong", "weak", "mid"], "llm", kb)
        assert result == ["weak", "mid", "strong"]

    async def test_kb_with_unknowns_interleaved(self) -> None:
        """Unknown models are placed at median of known ranking."""
        kb = MagicMock()
        kb.rank_llms.return_value = (["weak", "mid", "strong"], ["unknown1"])
        result = await rank_models_for_probes(["strong", "weak", "unknown1", "mid"], "llm", kb)
        # unknown1 should be at median position (index 1 of 3 known)
        assert result == ["weak", "unknown1", "mid", "strong"]

    async def test_kb_insufficient_llm_fallback(self) -> None:
        """When KB has < 3 known models, falls back to LLM ranking."""
        kb = MagicMock()
        kb.rank_llms.return_value = (["known1", "known2"], ["u1", "u2", "u3"])

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '["u1", "known1", "u2", "known2", "u3"]'

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            result = await rank_models_for_probes(
                ["known1", "u1", "u2", "known2", "u3"],
                "llm",
                kb,
                optimizer_model="test-model",
            )
        assert result == ["u1", "known1", "u2", "known2", "u3"]

    async def test_no_kb_no_optimizer(self) -> None:
        """Without KB or optimizer, returns original order."""
        result = await rank_models_for_probes(["a", "b", "c"], "llm", None)
        assert result == ["a", "b", "c"]

    async def test_single_model(self) -> None:
        result = await rank_models_for_probes(["only"], "llm", None)
        assert result == ["only"]

    async def test_embedding_ranking(self) -> None:
        kb = MagicMock()
        kb.rank_embeddings.return_value = (["weak_embed", "strong_embed"], [])
        result = await rank_models_for_probes(["strong_embed", "weak_embed"], "embedding", kb)
        assert result == ["weak_embed", "strong_embed"]

    async def test_reranker_ranking(self) -> None:
        kb = MagicMock()
        kb.rank_rerankers.return_value = (["none", "BAAI/bge-reranker-v2-m3"], [])
        result = await rank_models_for_probes(["BAAI/bge-reranker-v2-m3", "none"], "reranker", kb)
        assert result == ["none", "BAAI/bge-reranker-v2-m3"]

    async def test_llm_fallback_failure_uses_partial_kb(self) -> None:
        """When LLM fallback fails, uses partial KB data + warns."""
        kb = MagicMock()
        kb.rank_llms.return_value = (["known1", "known2"], ["u1", "u2", "u3"])

        with patch("litellm.acompletion", new=AsyncMock(side_effect=Exception("API error"))):
            result = await rank_models_for_probes(
                ["known1", "u1", "u2", "known2", "u3"],
                "llm",
                kb,
                optimizer_model="test-model",
            )
        # Should interleave unknowns at median of known ranking
        assert result[0] == "known1"
        assert result[-1] == "known2"


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

    def test_all_wrong_gets_positive_score(self) -> None:
        """All-wrong questions are genuinely hard — they get a small positive score."""
        questions = [_make_question("q1"), _make_question("q2")]
        probe_result = _make_probe_result(["q1", "q2"], set())
        scores = score_questions_by_discrimination([probe_result], questions)
        assert scores["q1"] > 0.0

    def test_mixed_aligned_with_strength_scores_positive(self) -> None:
        """Outcomes that align with probe-strength rank score positive;
        outcomes that anti-align (weak probe correct, strong probe wrong)
        clip to 0."""
        questions = [_make_question("q1"), _make_question("q2")]
        # Probes are passed weakest-first. q1 outcomes [1, 0] mean the WEAK
        # probe got it right and the STRONG one got it wrong (anti-aligned)
        # → tau = -1 → clipped to 0.
        # q2 outcomes [0, 1] mean WEAK got it wrong, STRONG got it right
        # (aligned) → tau = +1 → score = 1 * (1 - 0.5) = 0.5.
        probe1 = _make_probe_result(["q1", "q2"], {"q1"})
        probe2 = _make_probe_result(["q1", "q2"], {"q2"})

        scores = score_questions_by_discrimination([probe1, probe2], questions)
        assert scores["q1"] == 0.0
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
        assert all(isinstance(q, OpenEndedQuestion) for q in result)

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

    def test_all_wrong_cap_binds_when_pool_is_large(self) -> None:
        """When all-wrong count exceeds cap (30%), only cap-many survive."""
        aw_qs = [_make_question(f"aw_q{i}", cluster_id=0) for i in range(6)]
        mixed_qs = [_make_question(f"m_q{i}", cluster_id=0) for i in range(20)]
        all_qs = aw_qs + mixed_qs
        scores = {q.id: 1.0 for q in all_qs}
        all_wrong_ids = {q.id for q in aw_qs}
        result = select_exam(all_qs, scores, exam_size=10, all_wrong_ids=all_wrong_ids)
        n_all_wrong = sum(1 for q in result if q.id in all_wrong_ids)
        # Cap = 0.3 * 10 = 3
        assert n_all_wrong == 3
        assert len(result) == 10

    def test_all_wrong_cap_does_not_bind_when_pool_is_small(self) -> None:
        """When all-wrong count is below cap, all are admitted."""
        aw_qs = [_make_question(f"aw_q{i}", cluster_id=0) for i in range(2)]
        mixed_qs = [_make_question(f"m_q{i}", cluster_id=0) for i in range(10)]
        all_qs = aw_qs + mixed_qs
        scores = {q.id: 1.0 for q in all_qs}
        all_wrong_ids = {q.id for q in aw_qs}
        result = select_exam(all_qs, scores, exam_size=10, all_wrong_ids=all_wrong_ids)
        n_all_wrong = sum(1 for q in result if q.id in all_wrong_ids)
        assert n_all_wrong == 2

    def test_all_wrong_selection_under_cap_is_reproducible(self) -> None:
        """Same input → same survivors. Stable seed across processes."""
        aw_qs = [_make_question(f"aw_q{i}", cluster_id=0) for i in range(6)]
        mixed_qs = [_make_question(f"m_q{i}", cluster_id=0) for i in range(20)]
        all_qs = aw_qs + mixed_qs
        scores = {q.id: 1.0 for q in all_qs}
        all_wrong_ids = {q.id for q in aw_qs}
        r1 = select_exam(all_qs, scores, exam_size=10, all_wrong_ids=all_wrong_ids)
        r2 = select_exam(all_qs, scores, exam_size=10, all_wrong_ids=all_wrong_ids)
        aw_in_r1 = {q.id for q in r1 if q.id in all_wrong_ids}
        aw_in_r2 = {q.id for q in r2 if q.id in all_wrong_ids}
        assert aw_in_r1 == aw_in_r2


# ---------------------------------------------------------------------------
# TestCollectProbeOutcomes / TestAttachProbeMetadata
# ---------------------------------------------------------------------------


class TestCollectProbeOutcomes:
    def test_outcome_vectors_match_probe_results(self) -> None:
        questions = [_make_question("q1"), _make_question("q2")]
        # Probe 1: q1 wrong, q2 correct → (0, 1)
        # Probe 2: q1 correct, q2 wrong → (1, 0)
        probe1 = _make_probe_result(["q1", "q2"], {"q2"})
        probe2 = _make_probe_result(["q1", "q2"], {"q1"})
        outcomes = collect_probe_outcomes([probe1, probe2], questions)
        assert outcomes["q1"] == [0, 1]
        assert outcomes["q2"] == [1, 0]

    def test_missing_probe_evaluation_recorded_as_zero(self) -> None:
        questions = [_make_question("q_present"), _make_question("q_missing")]
        probe = _make_probe_result(["q_present"], {"q_present"})
        outcomes = collect_probe_outcomes([probe], questions)
        assert outcomes["q_present"] == [1]
        assert outcomes["q_missing"] == [0]

    def test_empty_probe_results_yields_empty_vectors(self) -> None:
        questions = [_make_question("q1"), _make_question("q2")]
        outcomes = collect_probe_outcomes([], questions)
        assert outcomes == {"q1": [], "q2": []}


class TestAttachProbeMetadata:
    def test_outcomes_and_entropy_persisted_on_questions(self) -> None:
        questions = [_make_question("q1"), _make_question("q2")]
        outcomes = {"q1": [0, 0, 1, 1], "q2": [1, 1, 1, 1]}
        scores = {"q1": 0.25, "q2": 0.0}
        updated = attach_probe_metadata(questions, outcomes, scores)
        assert updated[0].probe_outcomes == [0, 0, 1, 1]
        assert updated[0].discrimination_entropy == pytest.approx(0.25)
        assert updated[1].probe_outcomes == [1, 1, 1, 1]
        assert updated[1].discrimination_entropy == pytest.approx(0.0)

    def test_returns_copies_not_mutates_originals(self) -> None:
        original = _make_question("q1")
        updated = attach_probe_metadata([original], {"q1": [0, 1]}, {"q1": 0.25})
        assert original.probe_outcomes == []
        assert original.discrimination_entropy == 0.0
        assert updated[0].probe_outcomes == [0, 1]

    def test_question_with_no_probe_data_falls_back_to_defaults(self) -> None:
        q = _make_question("q1")
        updated = attach_probe_metadata([q], outcomes={}, scores={})
        assert updated[0].probe_outcomes == []
        assert updated[0].discrimination_entropy == 0.0
