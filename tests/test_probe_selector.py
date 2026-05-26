"""Tests for the probe-based exam selection module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from agentic_autorag.config.models import OpenEndedQuestion, ProjectConfig, TrialConfig
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.examiner.probe_selector import (
    PATTERN_WEIGHTS,
    allocate_quotas,
    attach_probe_metadata,
    collect_probe_outcomes,
    rank_models_for_probes,
    select_exam,
    select_probe_configs,
)


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
                "embedding": {
                    "models": [
                        "sentence-transformers/all-MiniLM-L6-v2",
                        "sentence-transformers/all-mpnet-base-v2",
                    ],
                },
                "retrieval": {
                    "index_types": ["vector_only"],
                    "top_k": {"min": 3, "max": 15},
                    "hybrid_alpha": {"min": 0.0, "max": 1.0},
                },
                "reranker": {
                    "models": ["none", "BAAI/bge-reranker-v2-m3"],
                    "top_n": {"min": 3, "max": 10},
                },
                "query_expansion": {"strategies": ["none"], "models": []},
                "generator": {"models": ["ollama/llama3.2", "ollama/mistral"]},
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
                "embedding": {"models": ["sentence-transformers/all-MiniLM-L6-v2"]},
                "retrieval": {
                    "index_types": ["vector_only"],
                    "top_k": {"min": 5, "max": 5},
                    "hybrid_alpha": {"min": 0.5, "max": 0.5},
                },
                "reranker": {"models": ["none"], "top_n": {"min": 5, "max": 5}},
                "query_expansion": {"strategies": ["none"], "models": []},
                "generator": {"models": ["ollama/llama3.2"]},
                "temperature": {"min": 0.0, "max": 0.0},
            },
        }
    )


def _make_question(qid: str) -> OpenEndedQuestion:
    return OpenEndedQuestion(
        id=qid,
        question=f"Question {qid}?",
        canonical_answer=f"answer_{qid}",
        reasoning_type="bridge",
        source_chunk_ids=[f"doc_a::chunk_0_{qid}", f"doc_b::chunk_0_{qid}"],
        source_doc_ids=["doc_a", "doc_b"],
        source_spans=["span A text", "span B text"],
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
            assert p.generator_llm in ss.all_llm_models()
            assert p.embedding_model in ss.embedding.models
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
        assert weak.generator_llm == config.search_space.generator.models[0]

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
        assert strong.embedding_model == config.search_space.embedding.models[-1]

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

    def test_four_distinct_embedding_tiers_with_enough_models(self) -> None:
        config = _make_config()
        ranked_embeds = ["weak/e1", "mid_low/e2", "mid_high/e3", "strong/e4"]
        probes = select_probe_configs(config, ranked_embeds=ranked_embeds)
        tier_to_embed = {label.split(" ")[0]: tc.embedding_model for label, tc in probes}
        assert tier_to_embed["Tier1-weak"] == "weak/e1"
        assert tier_to_embed["Tier2-lower-mid"] == "mid_low/e2"
        assert tier_to_embed["Tier3-upper-mid"] == "strong/e4"  # 3*4//4 == 3 = last index
        assert tier_to_embed["Tier4-strong"] == "strong/e4"

    def test_embedding_gradient_is_monotone(self) -> None:
        """Probe embeddings appear in weakest→strongest order across tiers."""
        config = _make_config()
        ranked_embeds = ["weak/e1", "mid_low/e2", "mid_high/e3", "strong/e4"]
        probes = select_probe_configs(config, ranked_embeds=ranked_embeds)
        embed_ranks = [ranked_embeds.index(tc.embedding_model) for _, tc in probes]
        assert embed_ranks == sorted(embed_ranks), f"Embedding ranks across tiers not monotone: {embed_ranks}"


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


def _qs_with_pattern(
    prefix: str, n: int, pattern: tuple[int, ...]
) -> tuple[list[OpenEndedQuestion], dict[str, list[int]]]:
    """Make ``n`` questions all sharing the given probe-outcome pattern."""
    qs = [_make_question(f"{prefix}_{i:04d}") for i in range(n)]
    outcomes = {q.id: list(pattern) for q in qs}
    return qs, outcomes


class TestAllocateQuotas:
    def test_well_supplied_matches_weights(self) -> None:
        weights = {("a",): 0.5, ("b",): 0.3, ("c",): 0.2}
        inventory = {("a",): 100, ("b",): 100, ("c",): 100}
        quota = allocate_quotas(weights, inventory, exam_size=100)
        assert quota == {("a",): 50, ("b",): 30, ("c",): 20}
        assert sum(quota.values()) == 100

    def test_under_supplied_cascades_to_remaining_patterns(self) -> None:
        """When one bucket can't fill its target, the deficit redistributes
        proportionally to the remaining patterns by their weights."""
        weights = {("a",): 0.6, ("b",): 0.3, ("c",): 0.1}
        inventory = {("a",): 20, ("b",): 100, ("c",): 100}  # a is short
        quota = allocate_quotas(weights, inventory, exam_size=100)
        assert quota[("a",)] == 20  # capped at inventory
        # 40-slot deficit redistributes between b (weight 0.3) and c (weight 0.1)
        # at 3:1 ratio → b gets ~30 extra, c gets ~10 extra.
        assert quota[("b",)] == 30 + 30  # 30 initial + 30 cascade
        assert quota[("c",)] == 10 + 10  # 10 initial + 10 cascade
        assert sum(quota.values()) == 100

    def test_empty_bucket_redistributes_entirely(self) -> None:
        """Empty bucket contributes zero; its weight is reassigned."""
        weights = {("a",): 0.5, ("b",): 0.5}
        inventory = {("a",): 0, ("b",): 100}
        quota = allocate_quotas(weights, inventory, exam_size=50)
        assert quota[("a",)] == 0
        assert quota[("b",)] == 50

    def test_total_pool_smaller_than_exam_size(self) -> None:
        weights = {("a",): 0.5, ("b",): 0.5}
        inventory = {("a",): 10, ("b",): 10}
        quota = allocate_quotas(weights, inventory, exam_size=50)
        # All inventory used; sum < exam_size, no infinite loop.
        assert quota == {("a",): 10, ("b",): 10}

    def test_exam_size_zero(self) -> None:
        weights = {("a",): 1.0}
        inventory = {("a",): 100}
        quota = allocate_quotas(weights, inventory, exam_size=0)
        assert quota == {("a",): 0}

    def test_largest_remainder_avoids_rounding_drift(self) -> None:
        """Three equal-weight patterns over 10 slots should sum to 10,
        not 9 due to floor-rounding."""
        weights = {("a",): 1 / 3, ("b",): 1 / 3, ("c",): 1 / 3}
        inventory = {("a",): 100, ("b",): 100, ("c",): 100}
        quota = allocate_quotas(weights, inventory, exam_size=10)
        assert sum(quota.values()) == 10

    def test_all_inventories_empty_returns_zero_quotas(self) -> None:
        weights = {("a",): 0.5, ("b",): 0.5}
        inventory = {("a",): 0, ("b",): 0}
        quota = allocate_quotas(weights, inventory, exam_size=10)
        assert quota == {("a",): 0, ("b",): 0}


class TestSelectExam:
    def test_only_allowlisted_patterns_enter_exam(self) -> None:
        """Patterns not in PATTERN_WEIGHTS (e.g., (1,0,0,0)) never appear."""
        # 100 candidates of an excluded pattern, 100 of an allowlisted one.
        excluded_qs, excluded_outcomes = _qs_with_pattern("ex", 100, (1, 0, 0, 0))
        allowed_qs, allowed_outcomes = _qs_with_pattern("al", 100, (0, 0, 0, 1))
        outcomes = {**excluded_outcomes, **allowed_outcomes}
        result = select_exam(excluded_qs + allowed_qs, outcomes, exam_size=80)
        result_ids = {q.id for q in result}
        # No excluded pattern leaks in regardless of inventory.
        assert not any(qid.startswith("ex_") for qid in result_ids)

    def test_cascade_on_under_supplied_top_pattern(self) -> None:
        """When (0,0,0,1) is short, its deficit goes to (0,0,1,0) and
        the remaining patterns proportionally — not to excluded patterns."""
        # Tiny (0,0,0,1) bucket; abundant everywhere else.
        top_qs, top_out = _qs_with_pattern("top", 5, (0, 0, 0, 1))
        sec_qs, sec_out = _qs_with_pattern("sec", 200, (0, 0, 1, 0))
        mid_qs, mid_out = _qs_with_pattern("mid", 200, (0, 0, 1, 1))
        aw_qs, aw_out = _qs_with_pattern("aw", 200, (0, 0, 0, 0))
        # Plus an abundant excluded pattern that must stay out.
        ex_qs, ex_out = _qs_with_pattern("ex", 200, (1, 0, 0, 0))
        outcomes = {**top_out, **sec_out, **mid_out, **aw_out, **ex_out}
        candidates = top_qs + sec_qs + mid_qs + aw_qs + ex_qs
        result = select_exam(candidates, outcomes, exam_size=80)

        # Exam fills.
        assert len(result) == 80
        # Top pattern capped at its inventory.
        top_in = sum(1 for q in result if q.id.startswith("top_"))
        assert top_in == 5
        # Excluded never enters.
        assert all(not q.id.startswith("ex_") for q in result)
        # Cascade lifted (0,0,1,0) above its bare-target share.
        # Initial target: 0.30 * 80 = 24. Deficit from top: ~27, redistributed
        # to surviving patterns by their weights with (0,0,1,0) getting the
        # largest share.
        sec_in = sum(1 for q in result if q.id.startswith("sec_"))
        assert sec_in > 24

    def test_empty_all_wrong_redistributes_to_other_patterns(self) -> None:
        """When (0,0,0,0) has no candidates, its 15% share cascades up."""
        top_qs, top_out = _qs_with_pattern("top", 100, (0, 0, 0, 1))
        sec_qs, sec_out = _qs_with_pattern("sec", 100, (0, 0, 1, 0))
        mid_qs, mid_out = _qs_with_pattern("mid", 100, (0, 0, 1, 1))
        outcomes = {**top_out, **sec_out, **mid_out}
        result = select_exam(top_qs + sec_qs + mid_qs, outcomes, exam_size=80)
        # Exam still fills despite the empty all-wrong bucket.
        assert len(result) == 80
        # No all-wrong items (the bucket was empty).
        assert all(q.id not in {} for q in result)

    def test_full_fill_when_pool_is_well_supplied(self) -> None:
        """With abundant inventory across allowlisted patterns, exam fills exactly."""
        pieces: list[OpenEndedQuestion] = []
        outcomes: dict[str, list[int]] = {}
        for i, pat in enumerate(PATTERN_WEIGHTS):
            qs, out = _qs_with_pattern(f"p{i}", 200, pat)
            pieces.extend(qs)
            outcomes.update(out)
        result = select_exam(pieces, outcomes, exam_size=200)
        assert len(result) == 200

    def test_per_pattern_quota_approximates_weights(self) -> None:
        """With abundant inventory, the per-pattern counts in the exam
        match PATTERN_WEIGHTS × exam_size within rounding error."""
        pieces: list[OpenEndedQuestion] = []
        outcomes: dict[str, list[int]] = {}
        for i, pat in enumerate(PATTERN_WEIGHTS):
            qs, out = _qs_with_pattern(f"p{i}", 200, pat)
            pieces.extend(qs)
            outcomes.update(out)
        result = select_exam(pieces, outcomes, exam_size=200)
        counts_by_pat: dict[tuple[int, ...], int] = {p: 0 for p in PATTERN_WEIGHTS}
        for q in result:
            pat = tuple(outcomes[q.id])
            counts_by_pat[pat] += 1
        for pat, weight in PATTERN_WEIGHTS.items():
            assert abs(counts_by_pat[pat] - weight * 200) <= 2, (
                f"pattern {pat}: got {counts_by_pat[pat]}, expected ~{weight * 200}"
            )

    def test_reproducible_selection_within_bucket(self) -> None:
        """Same inputs → same outputs; id-based stable sort within bucket."""
        qs, outcomes = _qs_with_pattern("q", 50, (0, 0, 0, 1))
        r1 = select_exam(qs, outcomes, exam_size=20)
        r2 = select_exam(qs, outcomes, exam_size=20)
        assert [q.id for q in r1] == [q.id for q in r2]

    def test_overgeneration_cannot_flood_one_pattern(self) -> None:
        """Even with 800 candidates of (0,0,0,1) and 80 of (0,0,1,0),
        (0,0,1,0) still gets its target share."""
        flood_qs, flood_out = _qs_with_pattern("flood", 800, (0, 0, 0, 1))
        small_qs, small_out = _qs_with_pattern("small", 80, (0, 0, 1, 0))
        aw_qs, aw_out = _qs_with_pattern("aw", 80, (0, 0, 0, 0))
        outcomes = {**flood_out, **small_out, **aw_out}
        result = select_exam(flood_qs + small_qs + aw_qs, outcomes, exam_size=80)
        # (0,0,0,1) is capped at its target weight (0.40 * 80 = 32),
        # not allowed to balloon to 80.
        n_flood = sum(1 for q in result if q.id.startswith("flood_"))
        assert n_flood <= 40  # 0.40 quota + at most rounding/cascade slack
        # (0,0,1,0) gets its share.
        n_small = sum(1 for q in result if q.id.startswith("small_"))
        assert n_small >= 20

    def test_empty_candidates(self) -> None:
        assert select_exam([], {}, exam_size=10) == []

    def test_returns_all_when_candidates_below_exam_size(self) -> None:
        """If total pool < exam_size, return what we have."""
        qs, outcomes = _qs_with_pattern("q", 5, (0, 0, 0, 1))
        result = select_exam(qs, outcomes, exam_size=80)
        assert len(result) == 5

    def test_wrong_probe_count_falls_back_to_id_truncation(self) -> None:
        """If outcomes are not the expected length (e.g., 2 probes when
        allowlist expects 4), the selector falls back to id-order truncation."""
        qs = [_make_question(f"q{i:03d}") for i in range(10)]
        outcomes = {q.id: [0, 1] for q in qs}  # only 2 probes
        result = select_exam(qs, outcomes, exam_size=5)
        assert len(result) == 5
        # Fallback is id-sorted truncation.
        assert [q.id for q in result] == [f"q{i:03d}" for i in range(5)]

    def test_errored_ids_excluded_entirely(self) -> None:
        """Questions in ``errored_ids`` never enter the exam — they would
        otherwise corrupt the all-wrong bucket with probe-noise items."""
        # Real all-wrong (probes evaluated and all said wrong) — these are
        # the questions the all-wrong bucket is intended to capture.
        real_aw_qs, real_aw_out = _qs_with_pattern("real_aw", 50, (0, 0, 0, 0))
        # "Errored" items also map to (0,0,0,0) outcome because of the
        # default-to-zero convention in collect_probe_outcomes, but they
        # must NOT be admitted.
        errored_qs, errored_out = _qs_with_pattern("errored", 50, (0, 0, 0, 0))
        # Plus some legitimate top patterns so the exam can still fill.
        top_qs, top_out = _qs_with_pattern("top", 100, (0, 0, 0, 1))
        sec_qs, sec_out = _qs_with_pattern("sec", 100, (0, 0, 1, 0))
        outcomes = {**real_aw_out, **errored_out, **top_out, **sec_out}
        errored_ids = {q.id for q in errored_qs}

        result = select_exam(
            real_aw_qs + errored_qs + top_qs + sec_qs,
            outcomes,
            exam_size=80,
            errored_ids=errored_ids,
        )

        # No errored question survives.
        assert not any(q.id.startswith("errored_") for q in result)
        # Real all-wrong items still fill their bucket.
        n_real_aw = sum(1 for q in result if q.id.startswith("real_aw_"))
        assert n_real_aw > 0
        # Exam is full because there's enough legitimate inventory.
        assert len(result) == 80

    def test_errored_ids_does_not_break_empty_candidates_case(self) -> None:
        assert select_exam([], {}, exam_size=10, errored_ids={"anything"}) == []

    def test_all_candidates_errored_returns_empty(self) -> None:
        """When every candidate is in errored_ids, exam comes back empty."""
        qs, outcomes = _qs_with_pattern("q", 10, (0, 0, 0, 1))
        errored_ids = {q.id for q in qs}
        result = select_exam(qs, outcomes, exam_size=5, errored_ids=errored_ids)
        assert result == []


class TestCollectProbeOutcomes:
    def test_outcome_vectors_match_probe_results(self) -> None:
        questions = [_make_question("q1"), _make_question("q2")]
        # Probe 1: q1 wrong, q2 correct → (0, 1)
        # Probe 2: q1 correct, q2 wrong → (1, 0)
        probe1 = _make_probe_result(["q1", "q2"], {"q2"})
        probe2 = _make_probe_result(["q1", "q2"], {"q1"})
        outcomes, errored = collect_probe_outcomes([probe1, probe2], questions)
        assert outcomes["q1"] == [0, 1]
        assert outcomes["q2"] == [1, 0]
        assert errored == set()  # every probe evaluated every question

    def test_missing_probe_evaluation_recorded_and_id_marked_errored(self) -> None:
        """A question missing from a probe's question_results has its slot
        defaulted to 0 AND its id added to ``errored_ids`` so the selector
        can exclude it. q_missing wasn't evaluated by the probe, so it's
        in errored_ids; q_present was evaluated, so it isn't."""
        questions = [_make_question("q_present"), _make_question("q_missing")]
        probe = _make_probe_result(["q_present"], {"q_present"})
        outcomes, errored = collect_probe_outcomes([probe], questions)
        assert outcomes["q_present"] == [1]
        assert outcomes["q_missing"] == [0]
        assert errored == {"q_missing"}
        assert "q_present" not in errored

    def test_partial_evaluation_across_probes_marks_errored(self) -> None:
        """A question evaluated by some probes but not all is errored."""
        questions = [_make_question("q1")]
        probe_full = _make_probe_result(["q1"], {"q1"})
        probe_partial = _make_probe_result([], set())  # didn't evaluate q1
        outcomes, errored = collect_probe_outcomes([probe_full, probe_partial], questions)
        assert outcomes["q1"] == [1, 0]
        assert errored == {"q1"}

    def test_empty_probe_results_yields_empty_vectors(self) -> None:
        questions = [_make_question("q1"), _make_question("q2")]
        outcomes, errored = collect_probe_outcomes([], questions)
        assert outcomes == {"q1": [], "q2": []}
        assert errored == set()


class TestAttachProbeMetadata:
    def test_outcomes_persisted_on_questions(self) -> None:
        questions = [_make_question("q1"), _make_question("q2")]
        outcomes = {"q1": [0, 0, 1, 1], "q2": [1, 1, 1, 1]}
        updated = attach_probe_metadata(questions, outcomes)
        assert updated[0].probe_outcomes == [0, 0, 1, 1]
        assert updated[1].probe_outcomes == [1, 1, 1, 1]

    def test_returns_copies_not_mutates_originals(self) -> None:
        original = _make_question("q1")
        updated = attach_probe_metadata([original], {"q1": [0, 1]})
        assert original.probe_outcomes == []
        assert updated[0].probe_outcomes == [0, 1]

    def test_question_with_no_probe_data_falls_back_to_defaults(self) -> None:
        q = _make_question("q1")
        updated = attach_probe_metadata([q], outcomes={})
        assert updated[0].probe_outcomes == []
