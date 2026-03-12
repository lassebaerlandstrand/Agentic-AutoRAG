"""Tests for the knowledge base module."""

from __future__ import annotations

from pathlib import Path

import yaml

from agentic_autorag.config.knowledge_base import KnowledgeBase, _normalize

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

    def test_unknown_models_excluded_gracefully(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        result = kb.format_for_prompt(
            llm_models=["ollama/llama3.2"],
            embedding_models=["unknown/embed"],
            reranker_models=["none"],
        )

        # Parameter section still appears even when no models are matched
        assert "chunk_size" in result
        # No LLM or embedding table rows
        assert "LLM Models" not in result
        assert "Embedding Models" not in result

    def test_knowledge_base_header(self, tmp_path: Path) -> None:
        _write_kb(tmp_path)
        kb = KnowledgeBase(kb_dir=tmp_path)

        result = kb.format_for_prompt(
            llm_models=["vertex_ai/gemini-2.5-flash"],
            embedding_models=[],
            reranker_models=[],
        )

        assert result.startswith("## Knowledge Base")


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
