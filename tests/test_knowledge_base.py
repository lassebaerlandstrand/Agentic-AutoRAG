"""Tests for the knowledge base module."""

from __future__ import annotations

from pathlib import Path

import yaml

from agentic_autorag.config.aa_matcher import normalize as _normalize
from agentic_autorag.config.knowledge_base import KnowledgeBase

# Minimal YAML fixtures used across tests

_LLM_YAML = {
    "_metadata": {"built_at": "2026-01-01T00:00:00", "matched_count": 2},
    "models": {
        "gemini-2-5-flash": {
            "name": "Gemini 2.5 Flash",
            "slug": "gemini-2-5-flash",
            "creator": "Google",
            "release_date": "2025-06-17",
            "litellm_ids": ["gemini/gemini-2.5-flash", "vertex_ai/gemini-2.5-flash"],
            "benchmarks": {
                "mmlu_pro": 0.75,
                "gpqa": 0.65,
                "ifbench": 0.60,
                "artificial_analysis_intelligence_index": 25.0,
            },
            "performance": {
                "median_output_tokens_per_second": 300.0,
                "median_time_to_first_token_seconds": 0.3,
            },
            "pricing": {
                "input_per_1m_tokens": 0.30,
                "output_per_1m_tokens": 2.50,
                "max_input_tokens": 1_000_000,
                "max_output_tokens": 65535,
            },
        },
        "claude-3-5-haiku": {
            "name": "Claude 3.5 Haiku",
            "slug": "claude-3-5-haiku",
            "creator": "Anthropic",
            "release_date": "2024-11-05",
            "litellm_ids": [
                "anthropic/claude-3-5-haiku-20241022",
                "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            ],
            "benchmarks": {
                "mmlu_pro": 0.64,
                "gpqa": 0.42,
                "ifbench": 0.55,
                "artificial_analysis_intelligence_index": 18.0,
            },
            "performance": {
                "median_output_tokens_per_second": 120.0,
                "median_time_to_first_token_seconds": 0.5,
            },
            "pricing": {
                "input_per_1m_tokens": 1.00,
                "output_per_1m_tokens": 5.00,
                "max_input_tokens": 200_000,
                "max_output_tokens": 8192,
            },
        },
    },
}

_EMBEDDING_YAML = {
    "_metadata": {"built_at": "2026-01-01T00:00:00", "benchmark": "MTEB(eng, v2)"},
    "models": {
        "BAAI/bge-m3": {
            "hf_id": "BAAI/bge-m3",
            "parameters_billions": 0.568,
            "memory_usage_mb": 2167,
            "embedding_dimensions": 1024,
            "max_tokens": 8194,
            "scores": {"retrieval": 0.5102, "sts": 0.7993, "reranking": 0.5200},
        },
        "sentence-transformers/all-MiniLM-L6-v2": {
            "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
            "parameters_billions": 0.023,
            "memory_usage_mb": 87,
            "embedding_dimensions": 384,
            "max_tokens": 256,
            "scores": {"retrieval": 0.4297, "sts": 0.7038, "reranking": 0.4446},
        },
    },
}

_RERANKER_YAML = {
    "models": {
        "BAAI/bge-reranker-v2-m3": {
            "name": "bge-reranker-v2-m3",
            "parameters": "0.6B",
            "scores": {"mteb_reranking": 57.03, "mmteb_reranking": 58.36, "followir": -0.01},
        },
        "cross-encoder/ms-marco-MiniLM-L-6-v2": {
            "name": "ms-marco-MiniLM-L-6-v2",
            "parameters": "22M",
            "scores": {"mteb_reranking": None, "mmteb_reranking": None, "followir": None},
        },
    },
}

_PARAMS_YAML = {
    "parameters": {
        "chunk_size": {
            "description": "Maximum characters per chunk.",
            "guidance": "Smaller for precision, larger for context.",
        },
        "llm_model": {
            "description": "Language model for answer generation.",
            "guidance": "Higher benchmark = better quality.",
        },
    },
}


def _write_kb(tmp_path: Path, llm: bool = True, embed: bool = True, reranker: bool = True, params: bool = True) -> None:
    if llm:
        (tmp_path / "llms.yaml").write_text(yaml.dump(_LLM_YAML), encoding="utf-8")
    if embed:
        (tmp_path / "embeddings.yaml").write_text(yaml.dump(_EMBEDDING_YAML), encoding="utf-8")
    if reranker:
        (tmp_path / "rerankers.yaml").write_text(yaml.dump(_RERANKER_YAML), encoding="utf-8")
    if params:
        (tmp_path / "parameter_descriptions.yaml").write_text(yaml.dump(_PARAMS_YAML), encoding="utf-8")


class TestNormalize:
    def test_strips_provider_prefix(self) -> None:
        assert _normalize("vertex_ai/gemini-2.5-flash") == "gemini-2-5-flash"

    def test_strips_gemini_prefix(self) -> None:
        assert _normalize("gemini/gemini-2.5-flash") == "gemini-2-5-flash"

    def test_strips_bedrock_region_and_provider(self) -> None:
        result = _normalize("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        assert "claude-haiku" in result
        assert "20251001" not in result
        assert "us" not in result

    def test_strips_date_suffix(self) -> None:
        result = _normalize("claude-3-5-haiku-20241022")
        assert "20241022" not in result
        assert "claude-3-5-haiku" in result

    def test_replaces_dots_with_dashes(self) -> None:
        assert _normalize("glm-4.7-flash") == "glm-4-7-flash"

    def test_lowercase(self) -> None:
        assert _normalize("GPT-4O") == "gpt-4o"


class TestKnowledgeBaseLoad:
    def test_loads_all_files(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)

        kb = KnowledgeBase(kb_dir=tmp_path)

        assert "gemini-2-5-flash" in kb._llms["models"]
        assert "BAAI/bge-m3" in kb._embeddings["models"]
        assert "BAAI/bge-reranker-v2-m3" in kb._rerankers["models"]
        assert "chunk_size" in kb._params["parameters"]

    def test_missing_files_warns_and_does_not_crash(self, tmp_path: Path, caplog) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            kb = KnowledgeBase(kb_dir=tmp_path)

        assert kb._llms == {}
        assert "not found" in caplog.text


class TestFindLlmEntry:
    def test_exact_litellm_id_match(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        entry = kb._find_llm_entry("vertex_ai/gemini-2.5-flash")

        assert entry is not None
        assert entry["slug"] == "gemini-2-5-flash"

    def test_bedrock_format_match(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        entry = kb._find_llm_entry("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")

        assert entry is not None
        assert entry["slug"] == "claude-3-5-haiku"

    def test_unknown_model_returns_none(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        assert kb._find_llm_entry("unknown/does-not-exist") is None

    def test_skips_variant_returns_base(self, tmp_path: Path) -> None:
        """_find_llm_entry must return the base, not a variant (variants have empty litellm_ids)."""
        llm_data = {
            "_metadata": {"built_at": "2026-01-01T00:00:00", "matched_count": 1},
            "models": {
                "model-x": {
                    "name": "Model X (Non-reasoning)",
                    "slug": "model-x",
                    "litellm_ids": ["provider/model-x"],
                    "benchmarks": {"mmlu_pro": 0.70},
                },
                "model-x-reasoning": {
                    "name": "Model X (Reasoning)",
                    "slug": "model-x-reasoning",
                    "litellm_ids": [],  # variants have empty IDs (unmatched in AA)
                    "base_slug": "model-x",
                    "variant_type": "reasoning",
                    "benchmarks": {"mmlu_pro": 0.80},
                },
            },
        }
        (tmp_path / "llms.yaml").write_text(yaml.dump(llm_data), encoding="utf-8")
        _write_kb(tmp_path, llm=False, embed=True, reranker=True, params=True)

        kb = KnowledgeBase(kb_dir=tmp_path)
        entry = kb._find_llm_entry("provider/model-x")

        assert entry is not None
        assert entry["slug"] == "model-x"  # base, not the reasoning variant
        assert entry.get("base_slug") is None  # confirmed it's not a variant


class TestFormatForPrompt:
    def test_filters_to_search_space_llms(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        result = kb.format_for_prompt(
            llm_models=["vertex_ai/gemini-2.5-flash"],
            embedding_models=[],
            reranker_models=[],
        )

        assert "gemini-2.5-flash" in result
        assert "claude" not in result.lower()

    def test_includes_embedding_data(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        result = kb.format_for_prompt(
            llm_models=[],
            embedding_models=["BAAI/bge-m3"],
            reranker_models=[],
        )

        assert "bge-m3" in result
        assert "all-MiniLM" not in result

    def test_reranker_none_excluded(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        result = kb.format_for_prompt(
            llm_models=[],
            embedding_models=[],
            reranker_models=["none", "BAAI/bge-reranker-v2-m3"],
        )

        assert "bge-reranker-v2-m3" in result
        # "none" should not appear as a table row
        lines = [ln for ln in result.splitlines() if "| `none`" in ln]
        assert not lines

    def test_parameter_guide_included(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        result = kb.format_for_prompt(
            llm_models=[],
            embedding_models=[],
            reranker_models=[],
        )

        assert "chunk_size" in result
        assert "llm_model" in result

    def test_empty_kb_returns_empty_string(self, tmp_path: Path) -> None:
        kb = KnowledgeBase(kb_dir=tmp_path)

        result = kb.format_for_prompt(
            llm_models=["any/model"],
            embedding_models=["some/embed"],
            reranker_models=["none"],
        )

        assert result == ""

    def test_unknown_models_rendered_with_em_dashes(self, tmp_path: Path) -> None:
        """Unknown models must still appear as rows so the Proposer sees they exist."""
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        result = kb.format_for_prompt(
            llm_models=["ollama/llama3.2"],
            embedding_models=["unknown/embed"],
            reranker_models=["none", "unknown/reranker"],
        )

        assert "chunk_size" in result
        # Every section renders with the unknown model as a row
        assert "LLM Models" in result
        assert "`ollama/llama3.2`" in result
        assert "Embedding Models" in result
        assert "`unknown/embed`" in result
        assert "Reranker Models" in result
        assert "`unknown/reranker`" in result
        # "none" is never rendered as a reranker row
        assert "| `none`" not in result

    def test_mixed_known_and_unknown_llms(self, tmp_path: Path) -> None:
        """Known models show benchmarks; unknown models appear in the same table with —."""
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        result = kb.format_for_prompt(
            llm_models=["vertex_ai/gemini-2.5-flash", "bedrock/unknown.model-v1:0"],
            embedding_models=[],
            reranker_models=[],
        )

        assert "`vertex_ai/gemini-2.5-flash`" in result
        assert "`bedrock/unknown.model-v1:0`" in result
        # Known model shows a real benchmark number
        assert "0.750" in result
        # Unknown row shows em-dashes for its data cells
        unknown_lines = [ln for ln in result.splitlines() if "bedrock/unknown.model-v1:0" in ln]
        assert unknown_lines, "unknown model row missing"
        assert unknown_lines[0].count("—") >= 8  # creator + 4 benchmarks + 2 prices + tokens/s + max input

    def test_knowledge_base_header(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        result = kb.format_for_prompt(
            llm_models=["vertex_ai/gemini-2.5-flash"],
            embedding_models=[],
            reranker_models=[],
        )

        assert result.startswith("## Knowledge Base")

    def test_supports_reasoning_column_shown_when_enabled(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        result = kb.format_for_prompt(
            llm_models=["vertex_ai/gemini-2.5-flash"],
            embedding_models=[],
            reranker_models=[],
            reasoning_allowed={"vertex_ai/gemini-2.5-flash": False},
            reasoning_enabled=True,
        )

        assert "Supports Reasoning" in result
        # Model is in the search space but is_reasoning_allowed=False → single row with ✗
        gemini_rows = [ln for ln in result.splitlines() if "`vertex_ai/gemini-2.5-flash`" in ln]
        assert len(gemini_rows) == 1
        assert gemini_rows[0].rstrip().endswith("✗ |")

    def test_supports_reasoning_column_hidden_when_disabled(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        result = kb.format_for_prompt(
            llm_models=["vertex_ai/gemini-2.5-flash"],
            embedding_models=[],
            reranker_models=[],
            reasoning_allowed={"vertex_ai/gemini-2.5-flash": False},
            reasoning_enabled=False,
        )

        assert "Supports Reasoning" not in result
        assert "✓" not in result and "✗" not in result
        # Reasoning parameter guide entry must also be suppressed so the proposer
        # doesn't waste tokens reasoning about a knob that isn't tunable. (The
        # synthetic _PARAMS_YAML has no ``reasoning`` key, so guard via the
        # description text we'd expect from a real KB.)
        assert "Enable extended reasoning" not in result

    def test_dual_rows_emitted_with_blanks_for_partial_reasoning_data(self, tmp_path: Path) -> None:
        """A reasoning-capable model with one missing variant still shows two rows.

        Per design: when ``reasoning_allowed=True`` for a model, the agent must
        see both ``(non-reasoning)`` and ``(reasoning)`` rows so it knows the
        choice exists — missing benchmarks render as blanks, not as a dropped row.
        """
        llm_data = {
            "_metadata": {"built_at": "2026-01-01T00:00:00", "matched_count": 1},
            "models": {
                "model-y": {
                    "name": "Model Y",
                    "slug": "model-y",
                    "creator": "Acme",
                    "litellm_ids": ["provider/model-y"],
                    "benchmarks": {"mmlu_pro": 0.50},
                    "pricing": {"input_per_1m_tokens": 1.0, "output_per_1m_tokens": 4.0},
                },
                # Sibling marks base as the OFF default; the ON sibling itself
                # has no benchmark data → the reasoning row renders blank.
                "model-y-reasoning": {
                    "name": "Model Y (Reasoning)",
                    "slug": "model-y-reasoning",
                    "litellm_ids": [],
                    "base_slug": "model-y",
                    "variant_type": "reasoning",
                },
            },
        }
        (tmp_path / "llms.yaml").write_text(yaml.dump(llm_data), encoding="utf-8")
        _write_kb(tmp_path, llm=False)
        kb = KnowledgeBase(kb_dir=tmp_path)

        result = kb.format_for_prompt(
            llm_models=["provider/model-y"],
            embedding_models=[],
            reranker_models=[],
            reasoning_allowed={"provider/model-y": True},
            reasoning_enabled=True,
        )

        non_reasoning = [ln for ln in result.splitlines() if "(non-reasoning)" in ln]
        reasoning = [ln for ln in result.splitlines() if "model-y (reasoning)" in ln]
        assert len(non_reasoning) == 1, result
        assert len(reasoning) == 1, result
        # OFF row has real data; ON row has blank cells but still shows ✓
        assert "0.500" in non_reasoning[0]
        assert non_reasoning[0].rstrip().endswith("✓ |")
        assert reasoning[0].count("—") >= 7
        assert reasoning[0].rstrip().endswith("✓ |")


class TestBuildNameMapping:
    """Tests for the name-mapping logic used in the build script."""

    def test_maps_gemini_litellm_to_aa_slug(self) -> None:
        from scripts.build_knowledge_base import _build_name_mapping

        slugs = ["gemini-2-5-flash"]
        litellm_keys = ["vertex_ai/gemini-2.5-flash", "gemini/gemini-2.5-flash"]

        mapping = _build_name_mapping(slugs, litellm_keys)

        assert set(mapping["gemini-2-5-flash"]) == {"vertex_ai/gemini-2.5-flash", "gemini/gemini-2.5-flash"}

    def test_unmatched_slug_returns_empty_list(self) -> None:
        from scripts.build_knowledge_base import _build_name_mapping

        mapping = _build_name_mapping(["obscure-model-xyz"], ["gpt-4o", "claude-3"])

        assert mapping["obscure-model-xyz"] == []

    def test_bedrock_slug_mapping(self) -> None:
        from scripts.build_knowledge_base import _build_name_mapping

        slugs = ["claude-3-5-haiku"]
        litellm_keys = [
            "anthropic/claude-3-5-haiku-20241022",
            "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        ]

        mapping = _build_name_mapping(slugs, litellm_keys)

        # anthropic key should match (normalises to claude-3-5-haiku)
        assert "anthropic/claude-3-5-haiku-20241022" in mapping["claude-3-5-haiku"]


class TestStripVariantSuffixes:
    def test_simple_reasoning_suffix(self) -> None:
        from scripts.build_knowledge_base import _strip_variant_suffixes

        base, vtype = _strip_variant_suffixes("gemini-2-5-flash-reasoning")
        assert base == "gemini-2-5-flash"
        assert vtype == "reasoning"

    def test_nested_variant(self) -> None:
        from scripts.build_knowledge_base import _strip_variant_suffixes

        base, vtype = _strip_variant_suffixes("nova-2-0-lite-reasoning-low")
        assert base == "nova-2-0-lite"
        assert vtype == "reasoning-low"

    def test_non_reasoning_suffix(self) -> None:
        from scripts.build_knowledge_base import _strip_variant_suffixes

        base, vtype = _strip_variant_suffixes("claude-sonnet-4-6-non-reasoning")
        assert base == "claude-sonnet-4-6"
        assert vtype == "non-reasoning"

    def test_no_known_suffix_returns_none(self) -> None:
        from scripts.build_knowledge_base import _strip_variant_suffixes

        base, vtype = _strip_variant_suffixes("gemini-2-5-flash")
        assert base is None
        assert vtype is None

    def test_non_reasoning_low_effort(self) -> None:
        from scripts.build_knowledge_base import _strip_variant_suffixes

        base, vtype = _strip_variant_suffixes("model-x-non-reasoning-low-effort")
        assert base == "model-x"
        assert vtype == "non-reasoning-low-effort"

    def test_thinking_suffix(self) -> None:
        from scripts.build_knowledge_base import _strip_variant_suffixes

        base, vtype = _strip_variant_suffixes("qwq-32b-thinking")
        assert base == "qwq-32b"
        assert vtype == "thinking"


class TestDetectVariants:
    def test_unmatched_slug_with_known_base(self) -> None:
        from scripts.build_knowledge_base import _detect_variants

        mapping = {
            "gemini-2-5-flash": ["vertex_ai/gemini-2.5-flash"],  # matched
            "gemini-2-5-flash-reasoning": [],  # unmatched variant
        }
        all_slugs = {"gemini-2-5-flash", "gemini-2-5-flash-reasoning"}

        variants = _detect_variants(mapping, all_slugs)

        assert "gemini-2-5-flash-reasoning" in variants
        assert variants["gemini-2-5-flash-reasoning"] == ("gemini-2-5-flash", "reasoning")

    def test_already_matched_slug_not_treated_as_variant(self) -> None:
        from scripts.build_knowledge_base import _detect_variants

        mapping = {
            "o3-mini-high": ["openai/o3-mini-high"],  # matched — not a variant
        }
        all_slugs = {"o3-mini-high"}

        variants = _detect_variants(mapping, all_slugs)

        assert "o3-mini-high" not in variants

    def test_base_not_in_aa_data_ignored(self) -> None:
        from scripts.build_knowledge_base import _detect_variants

        mapping = {
            "unknown-model-reasoning": [],
        }
        # "unknown-model" is NOT in all_slugs
        all_slugs = {"unknown-model-reasoning"}

        variants = _detect_variants(mapping, all_slugs)

        assert variants == {}


class TestVariantLitellmIds:
    def test_variant_keeps_empty_litellm_ids(self) -> None:
        """Variants are unmatched in AA — they should keep their empty litellm_ids."""
        models_out = {
            "gemini-2-5-flash": {
                "slug": "gemini-2-5-flash",
                "litellm_ids": ["gemini/gemini-2.5-flash", "deepinfra/google/gemini-2.5-flash"],
            },
            "gemini-2-5-flash-reasoning": {
                "slug": "gemini-2-5-flash-reasoning",
                "litellm_ids": [],
                "base_slug": "gemini-2-5-flash",
                "variant_type": "reasoning",
            },
        }

        # No inheritance loop — variant IDs stay empty
        variant = models_out["gemini-2-5-flash-reasoning"]
        assert variant["litellm_ids"] == []


class TestVariantIndex:
    def test_build_variant_index(self, tmp_path: Path) -> None:
        llm_data = {
            "_metadata": {"built_at": "2026-01-01T00:00:00", "matched_count": 1},
            "models": {
                "gemini-2-5-flash": {
                    "name": "Gemini 2.5 Flash",
                    "slug": "gemini-2-5-flash",
                    "creator": "Google",
                    "litellm_ids": ["vertex_ai/gemini-2.5-flash"],
                    "benchmarks": {"mmlu_pro": 0.72},
                },
                "gemini-2-5-flash-reasoning": {
                    "name": "Gemini 2.5 Flash (Reasoning)",
                    "slug": "gemini-2-5-flash-reasoning",
                    "creator": "Google",
                    "litellm_ids": [],
                    "base_slug": "gemini-2-5-flash",
                    "variant_type": "reasoning",
                    "benchmarks": {"mmlu_pro": 0.82},
                },
            },
        }
        (tmp_path / "llms.yaml").write_text(yaml.dump(llm_data), encoding="utf-8")
        _write_kb(tmp_path, llm=False, embed=True, reranker=True, params=True)

        kb = KnowledgeBase(kb_dir=tmp_path)

        assert "gemini-2-5-flash" in kb._base_to_variants
        assert len(kb._base_to_variants["gemini-2-5-flash"]) == 1
        assert kb._base_to_variants["gemini-2-5-flash"][0]["variant_type"] == "reasoning"

    def _write_base_plus_reasoning_variant(self, tmp_path: Path) -> None:
        """Write KB with a base (non-reasoning) + reasoning variant."""
        llm_data = {
            "_metadata": {"built_at": "2026-01-01T00:00:00", "matched_count": 1},
            "models": {
                "gemini-2-5-flash": {
                    "name": "Gemini 2.5 Flash (Non-reasoning)",
                    "slug": "gemini-2-5-flash",
                    "creator": "Google",
                    "litellm_ids": ["vertex_ai/gemini-2.5-flash"],
                    "benchmarks": {"mmlu_pro": 0.72},
                },
                "gemini-2-5-flash-reasoning": {
                    "name": "Gemini 2.5 Flash (Reasoning)",
                    "slug": "gemini-2-5-flash-reasoning",
                    "creator": "Google",
                    "litellm_ids": [],
                    "base_slug": "gemini-2-5-flash",
                    "variant_type": "reasoning",
                    "benchmarks": {"mmlu_pro": 0.82},
                },
            },
        }
        (tmp_path / "llms.yaml").write_text(yaml.dump(llm_data), encoding="utf-8")
        _write_kb(tmp_path, llm=False, embed=True, reranker=True, params=True)

    def test_format_for_prompt_shows_both_rows_with_labels_when_allowed(self, tmp_path: Path) -> None:
        """When reasoning is allowed: both rows shown with explicit labels, non-reasoning first."""
        self._write_base_plus_reasoning_variant(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        result = kb.format_for_prompt(
            llm_models=["vertex_ai/gemini-2.5-flash"],
            embedding_models=[],
            reranker_models=[],
            reasoning_allowed={"vertex_ai/gemini-2.5-flash": True},
        )

        assert "(non-reasoning)" in result
        assert "(reasoning)" in result
        # Non-reasoning row appears before reasoning row
        assert result.index("(non-reasoning)") < result.index("(reasoning)")
        # Base benchmarks (0.720) appear in non-reasoning row, not as the only row
        assert "0.720" in result
        assert "0.820" in result

    def test_format_for_prompt_single_plain_row_when_not_allowed(self, tmp_path: Path) -> None:
        """When reasoning is denied: single plain-name row with non-reasoning benchmarks."""
        self._write_base_plus_reasoning_variant(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        result = kb.format_for_prompt(
            llm_models=["vertex_ai/gemini-2.5-flash"],
            embedding_models=[],
            reranker_models=[],
            reasoning_allowed={"vertex_ai/gemini-2.5-flash": False},
        )

        # No labels — plain name only
        assert "(reasoning)" not in result
        assert "(non-reasoning)" not in result
        # Shows non-reasoning benchmarks (base entry: 0.72)
        assert "0.720" in result

    def test_format_for_prompt_base_is_reasoning_default(self, tmp_path: Path) -> None:
        """When base is reasoning-default and non-reasoning variant exists (GLM-style)."""
        llm_data = {
            "_metadata": {"built_at": "2026-01-01T00:00:00", "matched_count": 1},
            "models": {
                "glm-4-7-flash": {
                    "name": "GLM-4.7-Flash",
                    "slug": "glm-4-7-flash",
                    "creator": "Z AI",
                    "litellm_ids": ["bedrock/zai.glm-4.7-flash"],
                    "benchmarks": {"mmlu_pro": 0.75},
                },
                "glm-4-7-flash-non-reasoning": {
                    "name": "GLM-4.7-Flash (Non-reasoning)",
                    "slug": "glm-4-7-flash-non-reasoning",
                    "creator": "Z AI",
                    "litellm_ids": [],
                    "base_slug": "glm-4-7-flash",
                    "variant_type": "non-reasoning",
                    "benchmarks": {"mmlu_pro": 0.60},
                },
            },
        }
        (tmp_path / "llms.yaml").write_text(yaml.dump(llm_data), encoding="utf-8")
        _write_kb(tmp_path, llm=False, embed=True, reranker=True, params=True)

        kb = KnowledgeBase(kb_dir=tmp_path)
        result = kb.format_for_prompt(
            llm_models=["bedrock/zai.glm-4.7-flash"],
            embedding_models=[],
            reranker_models=[],
            reasoning_allowed={"bedrock/zai.glm-4.7-flash": True},
        )

        # Both rows shown
        assert "(non-reasoning)" in result
        assert "(reasoning)" in result
        # Non-reasoning first (lower benchmarks), then reasoning (base, higher)
        assert result.index("(non-reasoning)") < result.index("(reasoning)")
        assert "0.600" in result  # non-reasoning variant
        assert "0.750" in result  # base (reasoning default)


class TestRankLlms:
    def test_ranks_by_intelligence_index(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        ranked, unknowns = kb.rank_llms(["vertex_ai/gemini-2.5-flash", "anthropic/claude-3-5-haiku-20241022"])

        assert unknowns == []
        # Haiku (18.0) < Flash (25.0)
        assert ranked[0] == "anthropic/claude-3-5-haiku-20241022"
        assert ranked[1] == "vertex_ai/gemini-2.5-flash"

    def test_unknown_models_returned_separately(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        ranked, unknowns = kb.rank_llms(["vertex_ai/gemini-2.5-flash", "ollama/unknown-model"])

        assert ranked == ["vertex_ai/gemini-2.5-flash"]
        assert unknowns == ["ollama/unknown-model"]

    def test_fair_average_fallback(self, tmp_path: Path) -> None:
        """When Intel Index is missing, uses fair average of available benchmarks."""
        llm_data = {
            "_metadata": {"built_at": "2026-01-01T00:00:00", "matched_count": 2},
            "models": {
                "model-a": {
                    "slug": "model-a",
                    "litellm_ids": ["provider/model-a"],
                    "benchmarks": {"mmlu_pro": 0.80, "gpqa": 0.70},
                },
                "model-b": {
                    "slug": "model-b",
                    "litellm_ids": ["provider/model-b"],
                    "benchmarks": {"mmlu_pro": 0.60},
                },
            },
        }
        (tmp_path / "llms.yaml").write_text(yaml.dump(llm_data), encoding="utf-8")
        _write_kb(tmp_path, llm=False)

        kb = KnowledgeBase(kb_dir=tmp_path)
        ranked, _ = kb.rank_llms(["provider/model-a", "provider/model-b"])

        # model-b avg(0.60)*50 = 30.0, model-a avg(0.80, 0.70)*50 = 37.5
        assert ranked[0] == "provider/model-b"
        assert ranked[1] == "provider/model-a"

    def test_empty_list(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)
        ranked, unknowns = kb.rank_llms([])
        assert ranked == []
        assert unknowns == []


class TestRankEmbeddings:
    def test_ranks_by_retrieval_score(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        ranked, unknowns = kb.rank_embeddings(["BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2"])

        assert unknowns == []
        # MiniLM (0.4297) < BGE-M3 (0.5102)
        assert ranked[0] == "sentence-transformers/all-MiniLM-L6-v2"
        assert ranked[1] == "BAAI/bge-m3"

    def test_unknown_embeddings(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        ranked, unknowns = kb.rank_embeddings(["BAAI/bge-m3", "unknown/embed"])

        assert ranked == ["BAAI/bge-m3"]
        assert unknowns == ["unknown/embed"]


class TestRankRerankers:
    def test_none_always_first(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        ranked, unknowns = kb.rank_rerankers(["BAAI/bge-reranker-v2-m3", "none"])

        assert ranked[0] == "none"
        assert ranked[1] == "BAAI/bge-reranker-v2-m3"
        assert unknowns == []

    def test_reranker_with_null_scores_is_unknown(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        ranked, unknowns = kb.rank_rerankers(
            ["none", "cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-v2-m3"]
        )

        # ms-marco has None scores, so it's unknown
        assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in unknowns
        assert ranked[0] == "none"
        assert "BAAI/bge-reranker-v2-m3" in ranked
