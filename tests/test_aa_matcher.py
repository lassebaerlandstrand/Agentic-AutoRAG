"""Tests for the AA↔LiteLLM matcher.

Each test pins a specific failure mode that the previous matcher missed.
The cases come from configs/hotpot_qa.yaml's search-space LLMs.
"""

from __future__ import annotations

from collections import Counter

import pytest

from agentic_autorag.config.aa_matcher import (
    build_aa_to_litellm_mapping,
    find_best_aa_slug,
    match_priority,
    normalize,
    tokens,
)


class TestNormalize:
    def test_strips_provider_prefix(self) -> None:
        assert normalize("vertex_ai/gemini-2.5-flash") == "gemini-2-5-flash"

    def test_strips_bedrock_region(self) -> None:
        assert normalize("bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0") == ("anthropic-claude-haiku-4-5")

    def test_strips_bedrock_region_keeps_maker(self) -> None:
        # 'deepseek' is part of the model identity — must be preserved
        assert normalize("bedrock/us.deepseek.r1-v1:0") == "deepseek-r-1"

    def test_doubled_maker_collapses(self) -> None:
        assert normalize("bedrock/qwen.qwen3-32b-v1:0") == "qwen-3-32b"

    def test_minimax_double_collapses(self) -> None:
        assert normalize("bedrock/minimax.minimax-m2.1") == "minimax-m-2-1"

    def test_family_version_split(self) -> None:
        # Numeric repeats from version digits stay (3.3 release vs 3.0 release)
        assert normalize("llama3-3-70b-instruct") == "llama-3-3-70b-instruct"
        assert normalize("qwen3-next-80b-a3b") == "qwen-3-next-80b-a-3b"

    def test_dedup_only_alpha_makers(self) -> None:
        # Doubled maker collapses; doubled version digit does not
        assert normalize("bedrock/mistral.mistral-large-3") == "mistral-large-3"
        assert normalize("llama-3-3-instruct-70b") == "llama-3-3-instruct-70b"

    def test_strips_nested_provider_segments(self) -> None:
        # Nested routes (`a/b/c/...`) collapse iteratively before the model name
        assert normalize("openrouter/openai/o3-mini-high") == "o-3-mini-high"
        assert normalize("together_ai/meta-llama/Llama-3-1-instruct-70b") == "llama-3-1-instruct-70b"

    def test_strips_dashed_aws_region(self) -> None:
        # bedrock/us-east-1/maker.model and bedrock/eu-central-1/maker.model
        assert normalize("bedrock/us-east-1/anthropic.claude-3-5-haiku-20241022-v1:0") == ("anthropic-claude-3-5-haiku")
        assert normalize("bedrock/eu-north-1/minimax.minimax-m2.1") == "minimax-m-2-1"

    def test_strips_date_suffix(self) -> None:
        assert normalize("anthropic/claude-3-5-haiku-20241022") == "claude-3-5-haiku"

    def test_strips_v_suffix(self) -> None:
        assert normalize("bedrock/openai.gpt-oss-20b-1:0") == "openai-gpt-oss-20b"
        assert normalize("bedrock/openai.gpt-oss-120b-1:0") == "openai-gpt-oss-120b"

    def test_does_not_eat_model_name_segments(self) -> None:
        # fal.ai image-gen ids embed `/` inside the model name. The stripper
        # must stop at `flux-pro/` — that's the model, not a provider.
        assert normalize("fal_ai/fal-ai/flux-pro/v1.1") == "flux-pro-v-1-1"
        assert normalize("aiml/flux-pro/v1.1-ultra") == "flux-pro-v-1-1-ultra"

    def test_strips_commitment_tier(self) -> None:
        # Bedrock occasionally includes a commitment tier and/or wildcard
        # region segment between the provider and the model name.
        assert (
            normalize("bedrock/us-east-1/1-month-commitment/anthropic.claude-3-5-haiku") == "anthropic-claude-3-5-haiku"
        )
        assert normalize("bedrock/*/anthropic.claude-3-5-haiku") == "anthropic-claude-3-5-haiku"


class TestTokensNoise:
    def test_drops_latest(self) -> None:
        # `-latest` is a pure decoration on xAI / OpenAI ids (e.g.
        # `xai/grok-4-1-fast-reasoning-latest`); it must not inflate the
        # token count and pull matches away from the right AA slug.
        assert tokens("grok-4-1-fast-reasoning-latest") == Counter(
            {"grok": 1, "4": 1, "1": 1, "fast": 1, "reasoning": 1}
        )


class TestTokens:
    def test_drops_instruct(self) -> None:
        assert tokens("qwen-3-32b-instruct") == Counter({"qwen": 1, "3": 1, "32b": 1})

    def test_drops_it(self) -> None:
        assert tokens("google-gemma-3-4b-it") == Counter({"google": 1, "gemma": 1, "3": 1, "4b": 1})

    def test_drops_zero(self) -> None:
        # `0` token is treated as noise (minor-version notation: nova-2-0-lite vs nova-2-lite)
        assert tokens("nova-2-0-lite") == Counter({"nova": 1, "2": 1, "lite": 1})

    def test_drops_terminal_chat(self) -> None:
        assert tokens("gpt-5-1-chat") == Counter({"gpt": 1, "5": 1, "1": 1})

    def test_keeps_interior_chat(self) -> None:
        # 'chat' is meaningful in legacy slugs like llama-2-chat-7b
        assert tokens("llama-2-chat-7b") == Counter({"llama": 1, "2": 1, "chat": 1, "7b": 1})

    def test_multiset_distinguishes_versions(self) -> None:
        # llama-3 vs llama-3-3: must NOT collide as multisets
        assert tokens("llama-3-instruct-70b") != tokens("llama-3-3-instruct-70b")


class TestMatchPriority:
    def test_exact_normalised_match_priority_3(self) -> None:
        assert match_priority("vertex_ai/gemini-2.5-flash", "gemini-2-5-flash") == 3

    def test_endswith_match_priority_3(self) -> None:
        assert match_priority("bedrock/openai.gpt-oss-120b-1:0", "gpt-oss-120b") == 3

    def test_endswith_requires_dash_boundary(self) -> None:
        # 'flash' must not match 'flash-lite' as a substring suffix
        assert match_priority("vertex_ai/gemini-2.5-flash-lite", "flash") is None

    def test_token_multiset_equality_priority_2(self) -> None:
        # bedrock/qwen.qwen3-32b-v1:0 (after dedup of doubled qwen + family-version split)
        # vs AA qwen3-32b-instruct (after dropping noise 'instruct')
        assert (
            match_priority(
                "bedrock/qwen.qwen3-32b-v1:0",
                "qwen3-32b-instruct",
            )
            == 2
        )

    def test_subset_priority_1(self) -> None:
        # AA tokens ⊊ LiteLLM tokens (AA is more general)
        assert (
            match_priority(
                "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0",
                "llama-4-maverick",
            )
            == 1
        )

    def test_subset_modality_safety(self) -> None:
        # nemotron-nano-12b-v2 must NOT subset-match the -vl variant
        assert (
            match_priority(
                "bedrock/nvidia.nemotron-nano-12b-v2",
                "nvidia-nemotron-nano-12b-v2-vl",
            )
            is None
        )

    def test_subset_modality_safety_symmetric(self) -> None:
        # The text-only AA slug `gemini-3-pro` must NOT claim the
        # image-generation LiteLLM id `vertex_ai/gemini-3-pro-image-preview`.
        # Previously the guard only fired when AA had the extra modality
        # token; here the modality token is on the LiteLLM side.
        assert match_priority("vertex_ai/gemini-3-pro-image-preview", "gemini-3-pro") is None
        assert match_priority("gemini/gemini-3.1-flash-live-preview", "gemini-3-flash") is None

    def test_no_match_for_unrelated(self) -> None:
        assert match_priority("bedrock/zai.glm-4.7", "claude-4-5-haiku") is None

    def test_subset_below_overlap_threshold(self) -> None:
        # `r1` alone is too short — must not subset-match `deepseek-r1` without context
        assert match_priority("ollama/r1", "deepseek-r1") is None


class TestFindBestAASlug:
    """End-to-end matching against the actual hotpot_qa search-space gaps."""

    @pytest.fixture
    def aa_slugs(self) -> list[str]:
        return [
            # bases the search-space models should resolve to
            "claude-4-5-haiku",
            "claude-4-5-sonnet",
            "nova-2-0-lite",
            "nova-micro",
            "gpt-oss-20b",
            "gpt-oss-120b",
            "qwen3-32b-instruct",
            "qwen3-next-80b-a3b-instruct",
            "minimax-m2-1",
            "minimax-m2-5",
            "magistral-small-2509",
            "mistral-7b-instruct",
            "mixtral-8x7b-instruct",
            "ministral-3-3b",
            "ministral-3-8b",
            "ministral-3-14b",
            "mistral-large-3",
            "mistral-large",  # decoy: less specific
            "gemma-3-4b",
            "gemma-3-12b",
            "gemma-3-27b",
            "gemma-3-1b",
            "gemma-3-270m",
            "kimi-k2-5",
            "glm-4-7",
            "glm-4-7-flash",
            "deepseek-r1",
            "gemini-2-5-flash",
            "gemini-2-5-flash-lite",
            "llama-3-1-instruct-8b",
            "llama-3-3-instruct-70b",
            "llama-3-instruct-70b",  # decoy: similar-looking older model
            "llama-4-maverick",
            "llama-4-scout",
            "nvidia-nemotron-3-nano-30b-a3b",
            "nvidia-nemotron-3-super-120b-a12b",
            "nvidia-nemotron-nano-9b-v2",  # decoy
            "nvidia-nemotron-nano-12b-v2-vl",  # modality decoy
            "gpt-4o",
            "gpt-4o-mini",
            "o4-mini",
            "gpt-5-1",
            "gpt-5-1-codex",  # decoy
            "gpt-5-4-mini",
            "gpt-5-4-nano",
            # AA-side variants that must NOT win against the base
            "gemini-2-5-flash-reasoning",
            "claude-4-5-haiku-reasoning",
            "gpt-5-1-non-reasoning",
            "minimax-m2-5-non-reasoning",
        ]

    @pytest.mark.parametrize(
        ("litellm_id", "expected_slug"),
        [
            ("bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0", "claude-4-5-haiku"),
            ("bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-4-5-sonnet"),
            ("bedrock/global.amazon.nova-2-lite-v1:0", "nova-2-0-lite"),
            ("bedrock/openai.gpt-oss-20b-1:0", "gpt-oss-20b"),
            ("bedrock/openai.gpt-oss-120b-1:0", "gpt-oss-120b"),
            ("bedrock/qwen.qwen3-32b-v1:0", "qwen3-32b-instruct"),
            ("bedrock/minimax.minimax-m2.1", "minimax-m2-1"),
            ("bedrock/minimax.minimax-m2.5", "minimax-m2-5"),
            ("bedrock/us.amazon.nova-micro-v1:0", "nova-micro"),
            ("bedrock/mistral.magistral-small-2509", "magistral-small-2509"),
            ("bedrock/mistral.mistral-7b-instruct-v0:2", "mistral-7b-instruct"),
            ("bedrock/mistral.mixtral-8x7b-instruct-v0:1", "mixtral-8x7b-instruct"),
            ("bedrock/moonshotai.kimi-k2.5", "kimi-k2-5"),
            ("bedrock/zai.glm-4.7-flash", "glm-4-7-flash"),
            ("bedrock/zai.glm-4.7", "glm-4-7"),
            ("vertex_ai/gemini-2.5-flash", "gemini-2-5-flash"),
            ("vertex_ai/gemini-2.5-flash-lite", "gemini-2-5-flash-lite"),
            ("azure/gpt-4o", "gpt-4o"),
            ("azure/gpt-4o-mini", "gpt-4o-mini"),
            ("azure/o4-mini", "o4-mini"),
            ("azure/gpt-5.4-mini", "gpt-5-4-mini"),
            ("azure/gpt-5.4-nano", "gpt-5-4-nano"),
            # The previously-unmatched cases:
            ("bedrock/nvidia.nemotron-nano-3-30b", "nvidia-nemotron-3-nano-30b-a3b"),
            ("bedrock/nvidia.nemotron-super-3-120b", "nvidia-nemotron-3-super-120b-a12b"),
            ("bedrock/us.meta.llama3-3-70b-instruct-v1:0", "llama-3-3-instruct-70b"),
            ("bedrock/us.meta.llama4-maverick-17b-instruct-v1:0", "llama-4-maverick"),
            ("bedrock/us.meta.llama3-1-8b-instruct-v1:0", "llama-3-1-instruct-8b"),
            ("bedrock/mistral.ministral-3-14b-instruct", "ministral-3-14b"),
            ("bedrock/mistral.ministral-3-8b-instruct", "ministral-3-8b"),
            ("bedrock/mistral.ministral-3-3b-instruct", "ministral-3-3b"),
            ("bedrock/mistral.mistral-large-3-675b-instruct", "mistral-large-3"),
            ("bedrock/qwen.qwen3-next-80b-a3b", "qwen3-next-80b-a3b-instruct"),
            ("bedrock/google.gemma-3-4b-it", "gemma-3-4b"),
            ("bedrock/google.gemma-3-12b-it", "gemma-3-12b"),
            ("bedrock/google.gemma-3-27b-it", "gemma-3-27b"),
            ("bedrock/us.deepseek.r1-v1:0", "deepseek-r1"),
            ("azure/gpt-5.1-chat", "gpt-5-1"),
        ],
    )
    def test_search_space_models_match(self, aa_slugs: list[str], litellm_id: str, expected_slug: str) -> None:
        result = find_best_aa_slug(litellm_id, aa_slugs)
        assert result == expected_slug, f"{litellm_id}: expected {expected_slug!r}, got {result!r}"

    def test_text_only_does_not_match_vl_variant(self, aa_slugs: list[str]) -> None:
        # No text-only nemotron-nano-12b-v2 in AA → should return None, not the -vl variant
        result = find_best_aa_slug("bedrock/nvidia.nemotron-nano-12b-v2", aa_slugs)
        assert result is None

    def test_variants_lose_to_base(self, aa_slugs: list[str]) -> None:
        assert find_best_aa_slug("vertex_ai/gemini-2.5-flash", aa_slugs) == "gemini-2-5-flash"
        assert (
            find_best_aa_slug("bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0", aa_slugs) == "claude-4-5-haiku"
        )

    def test_variant_marker_in_litellm_prefers_aa_variant(self, aa_slugs: list[str]) -> None:
        # When the LiteLLM id carries an explicit variant marker, the
        # corresponding AA variant slug should win over the AA base.
        assert (
            find_best_aa_slug(
                "xai/grok-4-1-fast-reasoning-latest", aa_slugs + ["grok-4-1-fast", "grok-4-1-fast-reasoning"]
            )
            == "grok-4-1-fast-reasoning"
        )
        assert find_best_aa_slug("openai/gpt-5.1-non-reasoning", aa_slugs) == "gpt-5-1-non-reasoning"

    def test_non_reasoning_litellm_prefers_base_over_reasoning_variant(self, aa_slugs: list[str]) -> None:
        # AA has `grok-4-1-fast` (base, semantically non-reasoning) and
        # `grok-4-1-fast-reasoning`, but NO `-non-reasoning` slug. A LiteLLM
        # `-non-reasoning` id must map to the base — opposite-polarity AA
        # variants are worse than a base-vs-suffix partial match.
        slugs = aa_slugs + ["grok-4-1-fast", "grok-4-1-fast-reasoning"]
        assert find_best_aa_slug("xai/grok-4-1-fast-non-reasoning", slugs) == "grok-4-1-fast"
        assert find_best_aa_slug("azure_ai/grok-4-1-fast-non-reasoning", slugs) == "grok-4-1-fast"

    def test_unknown_model_returns_none(self, aa_slugs: list[str]) -> None:
        assert find_best_aa_slug("ollama/totally-made-up-model", aa_slugs) is None

    def test_image_variant_does_not_match_text_base(self, aa_slugs: list[str]) -> None:
        # `gemini-3-pro` is text-only in AA. The image-preview LiteLLM ids
        # must not be assigned to it.
        slugs = aa_slugs + ["gemini-3-pro"]
        assert find_best_aa_slug("vertex_ai/gemini-3-pro-image-preview", slugs) is None
        assert find_best_aa_slug("gemini/gemini-3.1-flash-live-preview", slugs) is None

    def test_status_extended_litellm_prefers_base_over_versioned_variant(self) -> None:
        # `vertex_ai/gemini-3-pro-preview` and AA candidates `gemini-3-pro`
        # vs `gemini-3-1-pro-preview` both have one token of difference, but
        # `-preview` on LiteLLM is a status modifier while the `1` on the
        # other AA candidate is a version digit (different model). The match
        # must pick the AA base that LiteLLM strictly extends.
        slugs = ["gemini-3-pro", "gemini-3-1-pro-preview"]
        assert find_best_aa_slug("vertex_ai/gemini-3-pro-preview", slugs) == "gemini-3-pro"
        assert find_best_aa_slug("openrouter/google/gemini-3-pro-preview", slugs) == "gemini-3-pro"
        # And the explicit 3.1 LiteLLM id still routes to the 3.1 AA slug.
        assert find_best_aa_slug("vertex_ai/gemini-3.1-pro-preview", slugs) == "gemini-3-1-pro-preview"

    def test_short_litellm_prefers_smaller_aa_over_longer_unrelated(self) -> None:
        # `gemini-3-flash-preview` (LiteLLM) is the regular Flash, not the Lite.
        # AA has both `gemini-3-flash` (closer fit) and the longer but distinct
        # `gemini-3-1-flash-lite-preview`. The matcher must pick the closer
        # fit by token distance, not the slug with more tokens.
        slugs = ["gemini-3-flash", "gemini-3-1-flash-lite-preview", "gemini-3-flash-reasoning"]
        assert find_best_aa_slug("vertex_ai/gemini-3-flash-preview", slugs) == "gemini-3-flash"
        assert find_best_aa_slug("openrouter/google/gemini-3-flash-preview", slugs) == "gemini-3-flash"
        # Exact-match LiteLLM id for the lite variant still routes correctly.
        assert find_best_aa_slug("vertex_ai/gemini-3.1-flash-lite-preview", slugs) == "gemini-3-1-flash-lite-preview"

    def test_flux_image_gen_does_not_match_text_llm(self) -> None:
        # `fal_ai/fal-ai/flux-pro/v1.1` is an image-generation model with a
        # slash inside the model name. Earlier, normalization greedily ate
        # the `flux-pro/` segment leaving only `v-1-1`, which spuriously
        # subset-matched a Llama Nemotron AA slug via the `{v, 1, 1}` tokens.
        slugs = ["llama-3-1-nemotron-ultra-253b-v1-reasoning", "gpt-5-1"]
        assert find_best_aa_slug("fal_ai/fal-ai/flux-pro/v1.1", slugs) is None
        assert find_best_aa_slug("aiml/flux-pro/v1.1-ultra", slugs) is None

    def test_subset_match_requires_alpha_anchor(self) -> None:
        # `openai/gpt-5-pro` (3 tokens: gpt, 5, pro) is a strict subset of AA
        # `gpt-5-4-pro` (4 tokens), but the common tokens `{gpt, 5, pro}` have
        # no alphabetic anchor of length >= 4 — so the match must be rejected
        # rather than silently routing a generic LiteLLM id to a specific
        # variant.
        slugs = ["gpt-5-4-pro", "gpt-5-5-pro"]
        assert find_best_aa_slug("openai/gpt-5-pro", slugs) is None
        assert find_best_aa_slug("azure/gpt-5-pro", slugs) is None


class TestBuildMapping:
    def test_each_litellm_key_maps_to_one_aa_slug(self) -> None:
        aa_slugs = ["gemini-2-5-flash", "gemini-2-5-flash-lite"]
        litellm_keys = ["vertex_ai/gemini-2.5-flash", "vertex_ai/gemini-2.5-flash-lite"]

        mapping = build_aa_to_litellm_mapping(aa_slugs, litellm_keys)

        assert mapping["gemini-2-5-flash"] == ["vertex_ai/gemini-2.5-flash"]
        assert mapping["gemini-2-5-flash-lite"] == ["vertex_ai/gemini-2.5-flash-lite"]

    def test_unmatched_key_excluded(self) -> None:
        mapping = build_aa_to_litellm_mapping(["gpt-4o"], ["foo/bar-totally-unrelated"])
        assert mapping["gpt-4o"] == []
