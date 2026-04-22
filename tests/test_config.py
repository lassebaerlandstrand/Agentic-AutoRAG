"""Tests for config models, validation, fingerprinting, and YAML loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import (
    AgentConfig,
    ExaminerConfig,
    GraphBuildConfig,
    GraphRetrievalSearchSpace,
    IndexType,
    MCQQuestion,
    NumericRange,
    ParsingConfig,
    ProjectConfig,
    RuntimeConfig,
    SearchSpace,
    StructuralConfig,
    TrialConfig,
    VLLMConfig,
)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


class TestNumericRange:
    def test_valid_range(self) -> None:
        r = NumericRange(min=0.0, max=1.0)
        assert r.min == 0.0
        assert r.max == 1.0

    def test_equal_min_max(self) -> None:
        r = NumericRange(min=5.0, max=5.0)
        assert r.contains(5.0)

    def test_invalid_range_max_lt_min(self) -> None:
        with pytest.raises(ValidationError, match="max must be >= min"):
            NumericRange(min=10.0, max=5.0)

    def test_contains_within(self) -> None:
        r = NumericRange(min=0.0, max=1.0)
        assert r.contains(0.5)

    def test_contains_at_boundaries(self) -> None:
        r = NumericRange(min=3.0, max=20.0)
        assert r.contains(3.0)
        assert r.contains(20.0)

    def test_contains_outside(self) -> None:
        r = NumericRange(min=3.0, max=20.0)
        assert not r.contains(2.9)
        assert not r.contains(20.1)


class TestStructuralConfig:
    """StructuralConfig is an internal engine type — basic smoke tests."""

    def test_overlap_gte_size_fails(self) -> None:
        with pytest.raises(ValidationError, match="chunk_token_overlap must be < chunk_token_size"):
            StructuralConfig(chunk_token_size=256, chunk_token_overlap=256)

    def test_index_type_from_string(self) -> None:
        cfg = StructuralConfig(index_type="hybrid_bm25_vector")
        assert cfg.index_type == IndexType.HYBRID_BM25_VECTOR


class TestRuntimeConfig:
    """RuntimeConfig is an internal engine type — basic smoke tests."""

    def test_defaults(self) -> None:
        cfg = RuntimeConfig(llm_model="test/model")
        assert cfg.top_k == 5
        assert cfg.reranker == "none"
        assert cfg.temperature == 0.0


class TestGraphBuildConfig:
    def test_required_extraction_model(self) -> None:
        cfg = GraphBuildConfig(extraction_model="gemini/gemini-2.5-flash-lite")
        assert cfg.extraction_model == "gemini/gemini-2.5-flash-lite"
        assert cfg.chunk_token_size is None
        assert cfg.entity_types is None
        assert cfg.max_parallel_insert == 2

    def test_optional_fields_set(self) -> None:
        cfg = GraphBuildConfig(
            extraction_model="gemini/test",
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            entity_types=["Person", "Concept"],
            max_parallel_insert=10,
        )
        assert cfg.chunk_token_size == 1200
        assert cfg.entity_types == ["Person", "Concept"]

    def test_max_parallel_insert_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            GraphBuildConfig(extraction_model="test/model", max_parallel_insert=0)


class TestGraphRetrievalSearchSpace:
    def test_defaults(self) -> None:
        gr = GraphRetrievalSearchSpace()
        assert "hybrid" in gr.graph_query_modes
        assert gr.graph_top_k.min == 20
        assert gr.graph_top_k.max == 100

    def test_custom_modes(self) -> None:
        gr = GraphRetrievalSearchSpace(graph_query_modes=["local", "global"])
        assert gr.graph_query_modes == ["local", "global"]


class TestVLLMConfig:
    def test_defaults(self) -> None:
        cfg = VLLMConfig()
        assert cfg.max_model_len is None
        assert cfg.gpu_memory_utilization == 0.90
        assert cfg.enforce_eager is True
        assert cfg.port == 8000
        assert cfg.startup_timeout == 180
        assert cfg.extra_args == []
        assert cfg.binary == "vllm"

    def test_explicit_max_model_len(self) -> None:
        cfg = VLLMConfig(max_model_len=4096)
        assert cfg.max_model_len == 4096

    def test_invalid_gpu_memory_utilization(self) -> None:
        with pytest.raises(ValidationError, match="gpu_memory_utilization"):
            VLLMConfig(gpu_memory_utilization=0.0)

    def test_project_config_with_vllm(self) -> None:
        cfg = ProjectConfig(
            search_space=SearchSpace(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                llm_models=["hosted_vllm/Qwen/Qwen3-8B"],
            ),
            vllm=VLLMConfig(max_model_len=4096),
        )
        assert cfg.vllm is not None
        assert cfg.vllm.max_model_len == 4096

    def test_project_config_without_vllm(self) -> None:
        cfg = ProjectConfig(
            search_space=SearchSpace(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                llm_models=["ollama/llama3.2"],
            ),
        )
        assert cfg.vllm is None

    def test_hosted_vllm_models_skip_probe(self) -> None:
        """hosted_vllm/ models should not be probed since vLLM isn't running at config time."""
        from unittest.mock import patch

        # Make _probe_model fail — hosted_vllm/ should never reach it
        with patch(
            "agentic_autorag.config.models._probe_model",
            return_value=(False, "connection refused"),
        ):
            cfg = ProjectConfig(
                search_space=SearchSpace(
                    embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                    llm_models=["hosted_vllm/Qwen/Qwen3-8B"],
                ),
            )
            assert "hosted_vllm/Qwen/Qwen3-8B" in cfg.search_space.llm_models


class TestTrialConfig:
    def _make_trial(self, index_type: IndexType = IndexType.VECTOR_ONLY, **kwargs) -> TrialConfig:
        return TrialConfig(llm_model="test/model", index_type=index_type, **kwargs)

    def test_valid_vector_only(self) -> None:
        trial = self._make_trial()
        assert trial.graph_query_mode == "hybrid"
        assert trial.graph_top_k == 60

    def test_overlap_gte_size_fails(self) -> None:
        with pytest.raises(ValidationError, match="chunk_token_overlap must be < chunk_token_size"):
            TrialConfig(llm_model="test/model", chunk_token_size=256, chunk_token_overlap=256)

    def test_graph_index_without_nested_config(self) -> None:
        """graph_only no longer requires a nested GraphConfig — params are flat."""
        trial = self._make_trial(index_type=IndexType.GRAPH_ONLY, graph_query_mode="local", graph_top_k=40)
        assert trial.index_type == IndexType.GRAPH_ONLY
        assert trial.graph_query_mode == "local"

    def test_hybrid_graph_vector_without_nested_config(self) -> None:
        trial = self._make_trial(index_type=IndexType.HYBRID_GRAPH_VECTOR)
        assert trial.index_type == IndexType.HYBRID_GRAPH_VECTOR

    def test_to_structural(self) -> None:
        trial = TrialConfig(llm_model="test/model", chunk_token_size=256, chunk_token_overlap=0)
        s = trial.to_structural()
        assert s.chunk_token_size == 256
        assert s.embedding_model == trial.embedding_model

    def test_to_runtime(self) -> None:
        trial = TrialConfig(
            llm_model="test/model",
            top_k=10,
            temperature=0.5,
            graph_query_mode="global",
            graph_top_k=50,
        )
        r = trial.to_runtime()
        assert r.top_k == 10
        assert r.llm_model == "test/model"
        assert r.graph_query_mode == "global"
        assert r.graph_top_k == 50

    def test_fingerprint_deterministic(self) -> None:
        trial = self._make_trial()
        fp1 = trial.structural_fingerprint()
        fp2 = trial.structural_fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 16

    def test_fingerprint_changes_with_index_building_param(self) -> None:
        trial_a = self._make_trial()
        trial_b = TrialConfig(llm_model="test/model", chunk_token_size=1024, chunk_token_overlap=128)
        assert trial_a.structural_fingerprint() != trial_b.structural_fingerprint()

    def test_fingerprint_unchanged_by_retrieval_params(self) -> None:
        trial_a = TrialConfig(llm_model="test/model", top_k=5)
        trial_b = TrialConfig(llm_model="test/model", top_k=15, temperature=0.9)
        assert trial_a.structural_fingerprint() == trial_b.structural_fingerprint()

    def test_fingerprint_unchanged_by_graph_retrieval_params(self) -> None:
        """Graph query mode/top_k are runtime params — they don't change the vector index."""
        trial_a = TrialConfig(llm_model="test/model", graph_query_mode="local", graph_top_k=20)
        trial_b = TrialConfig(llm_model="test/model", graph_query_mode="global", graph_top_k=80)
        assert trial_a.structural_fingerprint() == trial_b.structural_fingerprint()

    def test_fingerprint_unchanged_by_index_type(self) -> None:
        """index_type only routes queries — it does not change the cached index data."""
        from agentic_autorag.config.models import IndexType

        trial_a = TrialConfig(llm_model="test/model", index_type=IndexType.VECTOR_ONLY)
        trial_b = TrialConfig(llm_model="test/model", index_type=IndexType.HYBRID_BM25_VECTOR)
        trial_c = TrialConfig(llm_model="test/model", index_type=IndexType.HYBRID_GRAPH_VECTOR)
        assert trial_a.structural_fingerprint() == trial_b.structural_fingerprint()
        assert trial_a.structural_fingerprint() == trial_c.structural_fingerprint()

    def test_to_prompt_json_excludes_graph_when_disabled(self) -> None:
        trial = self._make_trial()
        result = trial.to_prompt_json(include_graph=False)
        assert "graph_query_mode" not in result
        assert "graph_top_k" not in result
        assert "llm_model" in result

    def test_to_prompt_json_includes_graph_when_enabled(self) -> None:
        trial = self._make_trial(graph_query_mode="local", graph_top_k=40)
        result = trial.to_prompt_json(include_graph=True)
        assert "graph_query_mode" in result
        assert "graph_top_k" in result

    def test_to_prompt_dump_excludes_graph_when_disabled(self) -> None:
        trial = self._make_trial()
        result = trial.to_prompt_dump(include_graph=False)
        assert "graph_query_mode" not in result
        assert "graph_top_k" not in result
        assert "llm_model" in result

    def test_to_prompt_dump_includes_graph_when_enabled(self) -> None:
        trial = self._make_trial()
        result = trial.to_prompt_dump(include_graph=True)
        assert "graph_query_mode" in result
        assert "graph_top_k" in result


def _make_project_config() -> ProjectConfig:
    """Create a representative search space for testing."""
    return ProjectConfig.model_validate(
        {
            "meta": {"project_name": "test"},
            "search_space": {
                "chunking": {
                    "strategies": ["recursive", "fixed"],
                    "chunk_token_size": {"min": 256, "max": 1024},
                    "chunk_token_overlap": {"min": 0, "max": 128},
                },
                "embedding_models": [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "BAAI/bge-m3",
                ],
                "index_types": ["vector_only", "hybrid_bm25_vector"],
                "top_k": {"min": 3, "max": 15},
                "hybrid_alpha": {"min": 0.0, "max": 1.0},
                "reranker": {
                    "models": ["none", "BAAI/bge-reranker-v2-m3"],
                    "top_n": {"min": 3, "max": 8},
                },
                "query_expansion": ["none", "hyde"],
                "llm_models": ["ollama/llama3.2", "ollama/mistral"],
                "temperature": {"min": 0.0, "max": 1.0},
            },
        }
    )


def _make_project_config_with_graph() -> ProjectConfig:
    """Create a search space that includes graph-based index types."""
    return ProjectConfig.model_validate(
        {
            "meta": {"project_name": "test"},
            "graph": {
                "extraction_model": "gemini/gemini-2.5-flash-lite",
            },
            "search_space": {
                "chunking": {
                    "strategies": ["recursive", "fixed"],
                    "chunk_token_size": {"min": 256, "max": 1024},
                    "chunk_token_overlap": {"min": 0, "max": 128},
                },
                "embedding_models": [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "BAAI/bge-m3",
                ],
                "index_types": ["vector_only", "graph_only", "hybrid_graph_vector"],
                "top_k": {"min": 3, "max": 15},
                "hybrid_alpha": {"min": 0.0, "max": 1.0},
                "reranker": {
                    "models": ["none", "BAAI/bge-reranker-v2-m3"],
                    "top_n": {"min": 3, "max": 8},
                },
                "query_expansion": ["none", "hyde"],
                "llm_models": ["ollama/llama3.2", "ollama/mistral"],
                "temperature": {"min": 0.0, "max": 1.0},
                "graph_retrieval": {
                    "graph_query_modes": ["local", "global", "hybrid"],
                    "graph_top_k": {"min": 20, "max": 100},
                },
            },
        }
    )


class TestExaminerConfig:
    def test_defaults(self) -> None:
        cfg = ExaminerConfig()
        assert cfg.exam_size == 60
        assert cfg.initial_candidate_multiplier == 2.5
        assert cfg.max_backfill_rounds == 3
        assert cfg.probe_selection is False
        assert cfg.detect_parametric_leaks is True
        assert cfg.parametric_leak_trials == 3
        assert cfg.parametric_leak_model is None
        assert cfg.source_fact_min_length == 150
        assert cfg.source_fact_verify_fuzzy_threshold == 0.9
        assert cfg.chunk_relevance_min_overlap_chars == 50
        assert cfg.chunk_relevance_ngram_size == 5
        assert cfg.chunk_relevance_overlap_threshold == 0.5
        assert cfg.chunk_relevance_min_run == 5
        assert cfg.doc_split_word_threshold == 24_000
        assert cfg.doc_section_word_size == 6_000
        assert cfg.min_doc_words == 200
        assert cfg.oracle_context_window_words == 300
        assert cfg.oracle_retry_with_full_doc is True
        assert cfg.max_questions_per_doc == 3
        assert cfg.max_generation_retries == 5
        assert cfg.dedup_similarity_threshold == 0.90
        assert cfg.discriminator_removal_pct == 0.05

    def test_initial_candidate_multiplier_below_one_invalid(self) -> None:
        with pytest.raises(ValidationError):
            ExaminerConfig(initial_candidate_multiplier=0.5)

    def test_probe_selection_enabled(self) -> None:
        cfg = ExaminerConfig(probe_selection=True)
        assert cfg.probe_selection is True

    def test_chunk_relevance_overlap_threshold_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ExaminerConfig(chunk_relevance_overlap_threshold=0.0)
        with pytest.raises(ValidationError):
            ExaminerConfig(chunk_relevance_overlap_threshold=1.1)
        cfg = ExaminerConfig(chunk_relevance_overlap_threshold=0.8)
        assert cfg.chunk_relevance_overlap_threshold == 0.8

    def test_parametric_leak_trials_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ExaminerConfig(parametric_leak_trials=0)
        with pytest.raises(ValidationError):
            ExaminerConfig(parametric_leak_trials=6)
        cfg = ExaminerConfig(parametric_leak_trials=5)
        assert cfg.parametric_leak_trials == 5

    def test_min_doc_words_non_negative(self) -> None:
        cfg = ExaminerConfig(min_doc_words=0)
        assert cfg.min_doc_words == 0

    def test_source_fact_min_length_positive(self) -> None:
        with pytest.raises(ValidationError):
            ExaminerConfig(source_fact_min_length=0)

    def test_doc_section_size_less_than_split_threshold(self) -> None:
        with pytest.raises(ValidationError, match="doc_section_word_size"):
            ExaminerConfig(doc_split_word_threshold=5000, doc_section_word_size=5000)


class TestAgentConfig:
    def test_defaults(self) -> None:
        cfg = AgentConfig()
        assert cfg.concurrency == 10

    def test_explicit_concurrency(self) -> None:
        cfg = AgentConfig(concurrency=3)
        assert cfg.concurrency == 3

    def test_concurrency_zero_is_invalid(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(concurrency=0)

    def test_concurrency_negative_is_invalid(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(concurrency=-1)


class TestParsingConfig:
    def test_defaults(self) -> None:
        cfg = ParsingConfig()
        assert cfg.parser == "docling"
        assert cfg.ocr is True
        assert cfg.table_structure is True

    def test_custom_values(self) -> None:
        cfg = ParsingConfig(parser="pymupdf4llm", ocr=False, table_structure=False)
        assert cfg.parser == "pymupdf4llm"
        assert cfg.ocr is False
        assert cfg.table_structure is False


class TestSearchSpaceValidation:
    def test_valid_trial_no_violations(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(
            chunking_strategy="recursive",
            chunk_token_size=512,
            chunk_token_overlap=64,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_type=IndexType.VECTOR_ONLY,
            top_k=5,
            hybrid_alpha=0.5,
            reranker="none",
            reranker_top_n=5,
            query_expansion="none",
            llm_model="ollama/llama3.2",
            temperature=0.0,
        )
        violations = cfg.validate_trial(trial)
        assert violations == []

    def test_chunking_strategy_violation(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(llm_model="ollama/llama3.2", chunking_strategy="semantic")
        violations = cfg.validate_trial(trial)
        assert any("chunking_strategy" in v for v in violations)

    def test_chunk_size_out_of_range(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(llm_model="ollama/llama3.2", chunk_token_size=2048, chunk_token_overlap=64)
        violations = cfg.validate_trial(trial)
        assert any("chunk_token_size" in v for v in violations)

    def test_chunk_overlap_out_of_range(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(llm_model="ollama/llama3.2", chunk_token_size=512, chunk_token_overlap=200)
        violations = cfg.validate_trial(trial)
        assert any("chunk_token_overlap" in v for v in violations)

    def test_embedding_model_violation(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(llm_model="ollama/llama3.2", embedding_model="unknown/model")
        violations = cfg.validate_trial(trial)
        assert any("embedding_model" in v for v in violations)

    def test_index_type_violation(self) -> None:
        cfg = _make_project_config()
        # graph_only is not in this search space (no graph config either)
        trial = TrialConfig(llm_model="ollama/llama3.2", index_type=IndexType.GRAPH_ONLY)
        violations = cfg.validate_trial(trial)
        assert any("index_type" in v for v in violations)

    def test_top_k_violation(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(llm_model="ollama/llama3.2", top_k=25)
        violations = cfg.validate_trial(trial)
        assert any("top_k" in v for v in violations)

    def test_reranker_violation(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(llm_model="ollama/llama3.2", reranker="cross-encoder/ms-marco-MiniLM-L-6-v2")
        violations = cfg.validate_trial(trial)
        assert any("reranker" in v for v in violations)

    def test_llm_model_violation(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(llm_model="openai/gpt-4o")
        violations = cfg.validate_trial(trial)
        assert any("llm_model" in v for v in violations)

    def test_temperature_violation(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(llm_model="ollama/llama3.2", temperature=1.5)
        violations = cfg.validate_trial(trial)
        assert any("temperature" in v for v in violations)

    def test_query_expansion_violation(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(llm_model="ollama/llama3.2", query_expansion="multi_query")
        violations = cfg.validate_trial(trial)
        assert any("query_expansion" in v for v in violations)

    def test_multiple_violations(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(llm_model="openai/gpt-4o", embedding_model="unknown/model", top_k=100)
        violations = cfg.validate_trial(trial)
        assert len(violations) >= 3

    def test_graph_query_mode_violation(self) -> None:
        cfg = _make_project_config_with_graph()
        trial = TrialConfig(
            llm_model="ollama/llama3.2",
            index_type=IndexType.GRAPH_ONLY,
            graph_query_mode="naive",  # not in allowed modes
            graph_top_k=50,
        )
        violations = cfg.validate_trial(trial)
        assert any("graph_query_mode" in v for v in violations)

    def test_graph_top_k_violation(self) -> None:
        cfg = _make_project_config_with_graph()
        trial = TrialConfig(
            llm_model="ollama/llama3.2",
            index_type=IndexType.GRAPH_ONLY,
            graph_query_mode="hybrid",
            graph_top_k=200,  # above max=100
        )
        violations = cfg.validate_trial(trial)
        assert any("graph_top_k" in v for v in violations)

    def test_graph_params_not_checked_for_vector_index(self) -> None:
        """Graph retrieval violations only apply when index_type is graph-based."""
        cfg = _make_project_config_with_graph()
        trial = TrialConfig(
            llm_model="ollama/llama3.2",
            index_type=IndexType.VECTOR_ONLY,
            graph_query_mode="naive",  # would be invalid for graph, but ignored for vector
            graph_top_k=200,  # same
        )
        violations = cfg.validate_trial(trial)
        assert not any("graph_query_mode" in v for v in violations)
        assert not any("graph_top_k" in v for v in violations)


class TestSearchSpaceAgentPrompt:
    def test_returns_string_with_search_space_heading(self) -> None:
        cfg = _make_project_config()
        prompt = cfg.to_agent_prompt()
        assert isinstance(prompt, str)
        assert "search space" in prompt.lower()

    def test_excludes_meta_examiner_agent(self) -> None:
        cfg = _make_project_config()
        prompt = cfg.to_agent_prompt()
        assert "project_name" not in prompt
        assert "exam_size" not in prompt
        assert "optimizer_model" not in prompt
        assert "examiner_model" not in prompt
        assert "max_history_trials" not in prompt
        assert "corpus_path" not in prompt

    def test_includes_all_optimizable_field_names(self) -> None:
        cfg = _make_project_config()
        prompt = cfg.to_agent_prompt()
        for field in ["chunking_strategy", "chunk_token_size", "chunk_token_overlap", "embedding_model", "index_type"]:
            assert field in prompt, f"Missing structural field: {field}"
        for field in [
            "top_k",
            "hybrid_alpha",
            "reranker",
            "reranker_top_n",
            "query_expansion",
            "llm_model",
            "temperature",
        ]:
            assert field in prompt, f"Missing runtime field: {field}"

    def test_contains_yaml_example_block(self) -> None:
        cfg = _make_project_config()
        prompt = cfg.to_agent_prompt()
        assert "```yaml" in prompt
        assert "```" in prompt
        assert "ollama/llama3.2" in prompt  # first llm_model

    def test_includes_graph_params_when_present(self) -> None:
        cfg = _make_project_config_with_graph()
        prompt = cfg.to_agent_prompt()
        assert "graph_query_mode" in prompt
        assert "graph_top_k" in prompt

    def test_excludes_graph_params_when_absent(self) -> None:
        cfg = ProjectConfig(
            search_space=SearchSpace(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                llm_models=["ollama/llama3.2"],
            ),
        )
        prompt = cfg.to_agent_prompt()
        assert "graph_query_mode" not in prompt
        assert "graph_top_k" not in prompt

    def test_shows_search_space_bounds(self) -> None:
        cfg = _make_project_config()
        prompt = cfg.to_agent_prompt()
        assert "[256, 1024]" in prompt  # chunk_token_size range (from test config)
        assert "[3, 15]" in prompt  # top_k range


class TestProjectConfigConsistency:
    """Tests for the graph/search-space consistency validator."""

    def test_graph_index_without_graph_config_raises(self) -> None:
        with pytest.raises(ValidationError, match="graph"):
            ProjectConfig.model_validate(
                {
                    "search_space": {
                        "embedding_models": ["sentence-transformers/all-MiniLM-L6-v2"],
                        "index_types": ["graph_only"],
                        "llm_models": ["ollama/llama3.2"],
                    },
                }
            )

    def test_graph_retrieval_without_graph_index_raises(self) -> None:
        with pytest.raises(ValidationError, match="graph_retrieval"):
            ProjectConfig.model_validate(
                {
                    "search_space": {
                        "embedding_models": ["sentence-transformers/all-MiniLM-L6-v2"],
                        "index_types": ["vector_only"],
                        "llm_models": ["ollama/llama3.2"],
                        "graph_retrieval": {
                            "graph_query_modes": ["hybrid"],
                            "graph_top_k": {"min": 20, "max": 100},
                        },
                    },
                }
            )

    def test_vector_only_without_graph_config_ok(self) -> None:
        cfg = ProjectConfig.model_validate(
            {
                "search_space": {
                    "embedding_models": ["sentence-transformers/all-MiniLM-L6-v2"],
                    "index_types": ["vector_only"],
                    "llm_models": ["ollama/llama3.2"],
                },
            }
        )
        assert cfg.graph is None
        assert not cfg.uses_graph()

    def test_graph_index_with_graph_config_ok(self) -> None:
        cfg = ProjectConfig.model_validate(
            {
                "graph": {"extraction_model": "gemini/test"},
                "search_space": {
                    "embedding_models": ["sentence-transformers/all-MiniLM-L6-v2"],
                    "index_types": ["vector_only", "graph_only"],
                    "llm_models": ["ollama/llama3.2"],
                    "graph_retrieval": {
                        "graph_query_modes": ["hybrid"],
                        "graph_top_k": {"min": 20, "max": 100},
                    },
                },
            }
        )
        assert cfg.graph is not None
        assert cfg.uses_graph()


class TestMCQQuestion:
    def test_valid_question(self) -> None:
        q = MCQQuestion(
            id="q1",
            question="What is RAG?",
            options={"A": "Retrieval", "B": "Random", "C": "Robust", "D": "Recursive"},
            correct_answer="A",
            source_doc_ids=["doc_0"],
            cluster_id=0,
        )
        assert q.correct_answer == "A"
        assert q.source_fact == []
        assert q.source_fact_offsets == []
        assert q.difficulty == 0.0
        assert q.discrimination == 1.0
        assert q.guessing == 0.25

    def test_source_fact_stored(self) -> None:
        q = MCQQuestion(
            id="q1",
            question="What is RAG?",
            options={"A": "Retrieval", "B": "Random", "C": "Robust", "D": "Recursive"},
            correct_answer="A",
            source_doc_ids=["doc_0"],
            source_fact=["RAG combines retrieval with generation."],
            source_fact_offsets=[(0, 40)],
            cluster_id=0,
        )
        assert q.source_fact == ["RAG combines retrieval with generation."]
        assert q.source_fact_offsets == [(0, 40)]

    def test_source_fact_string_coerced_to_list(self) -> None:
        q = MCQQuestion(
            id="q1",
            question="What is RAG?",
            options={"A": "Retrieval", "B": "Random", "C": "Robust", "D": "Recursive"},
            correct_answer="A",
            source_doc_ids=["doc_0"],
            source_fact="single string fact",
            cluster_id=0,
        )
        assert q.source_fact == ["single string fact"]

    def test_empty_source_doc_ids_invalid(self) -> None:
        with pytest.raises(ValidationError, match="source_doc_ids must not be empty"):
            MCQQuestion(
                id="q1",
                question="What is RAG?",
                options={"A": "Retrieval", "B": "Random", "C": "Robust", "D": "Recursive"},
                correct_answer="A",
                source_doc_ids=[],
                cluster_id=0,
            )

    def test_invalid_option_keys(self) -> None:
        with pytest.raises(ValidationError, match="options must have exactly keys"):
            MCQQuestion(
                id="q1",
                question="What is RAG?",
                options={"A": "Retrieval", "B": "Random", "C": "Robust"},
                correct_answer="A",
                source_doc_ids=["doc_0"],
                cluster_id=0,
            )

    def test_invalid_correct_answer(self) -> None:
        with pytest.raises(ValidationError, match="correct_answer"):
            MCQQuestion(
                id="q1",
                question="What is RAG?",
                options={"A": "Retrieval", "B": "Random", "C": "Robust", "D": "Recursive"},
                correct_answer="E",
                source_doc_ids=["doc_0"],
                cluster_id=0,
            )


MOCK_YAML_CONFIG = """
meta:
  project_name: "test-project"
  corpus_path: "./data/corpus/"
  corpus_description: "A small test corpus."
  output_dir: "./experiments/"
  max_trials: 10
parsing:
  parser: "pymupdf4llm"
  ocr: false
  table_structure: true
graph:
  extraction_model: "gemini/gemini-2.5-flash-lite"
  chunk_token_size: 1200
search_space:
  chunking:
    strategies: ["recursive"]
    chunk_token_size: { min: 256, max: 1024 }
    chunk_token_overlap: { min: 0, max: 128 }
  embedding_models: ["sentence-transformers/all-MiniLM-L6-v2"]
  index_types: ["vector_only", "graph_only"]
  top_k: { min: 3, max: 15 }
  hybrid_alpha: { min: 0.0, max: 1.0 }
  reranker:
    models: ["none"]
    top_n: { min: 3, max: 8 }
  query_expansion: ["none"]
  llm_models: ["ollama/llama3.2"]
  temperature: { min: 0.0, max: 1.0 }
  graph_retrieval:
    graph_query_modes: ["local", "global", "hybrid"]
    graph_top_k: { min: 20, max: 100 }
"""


class TestLoader:
    def _create_mock_config(self, tmp_path: Path, content: str) -> Path:
        """Helper to create a temporary config file."""
        config_file = tmp_path / "mock_config.yaml"
        config_file.write_text(content, encoding="utf-8")
        return config_file

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        config_file = self._create_mock_config(tmp_path, MOCK_YAML_CONFIG)

        cfg = load_config(config_file)

        assert cfg.meta.project_name == "test-project"
        assert cfg.meta.max_trials == 10
        assert cfg.parsing.parser == "pymupdf4llm"
        assert cfg.parsing.ocr is False
        assert "recursive" in cfg.search_space.chunking.strategies
        assert cfg.search_space.chunking.chunk_token_size.min == 256
        assert len(cfg.search_space.llm_models) == 1
        assert cfg.graph is not None
        assert cfg.graph.extraction_model == "gemini/gemini-2.5-flash-lite"
        assert cfg.graph.chunk_token_size == 1200
        assert cfg.search_space.graph_retrieval is not None
        assert "hybrid" in cfg.search_space.graph_retrieval.graph_query_modes

    def test_load_nonexistent_file(self) -> None:
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_loaded_search_space_validates_valid_trial(self, tmp_path: Path) -> None:
        config_file = self._create_mock_config(tmp_path, MOCK_YAML_CONFIG)
        cfg = load_config(config_file)
        trial = TrialConfig(
            chunking_strategy="recursive",
            chunk_token_size=512,
            chunk_token_overlap=64,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_type=IndexType.VECTOR_ONLY,
            top_k=5,
            hybrid_alpha=0.5,
            reranker="none",
            reranker_top_n=3,
            query_expansion="none",
            llm_model="ollama/llama3.2",
            temperature=0.3,
        )

        violations = cfg.validate_trial(trial)

        assert violations == []

    def test_loaded_search_space_catches_violation(self, tmp_path: Path) -> None:
        config_file = self._create_mock_config(tmp_path, MOCK_YAML_CONFIG)
        cfg = load_config(config_file)

        trial = TrialConfig(
            llm_model="ollama/llama3.2",
            chunk_token_size=2048,  # Out of range (max is 1024)
            chunk_token_overlap=64,
        )

        violations = cfg.validate_trial(trial)

        assert len(violations) > 0
        assert any("chunk_token_size" in v for v in violations)


class TestReasoningSearchSpace:
    """Tests for reasoning parameter support in the search space."""

    def test_mixed_llm_models_list_parsing(self) -> None:
        """Dict items in llm_models are normalized to plain strings."""
        ss = SearchSpace(
            embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
            llm_models=[
                "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                {"model": "vertex_ai/gemini-2.5-flash", "reasoning": True},
                "vertex_ai/gemini-2.5-flash-lite",
            ],
        )
        assert ss.llm_models == [
            "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "vertex_ai/gemini-2.5-flash",
            "vertex_ai/gemini-2.5-flash-lite",
        ]

    def test_reasoning_overrides_extracted_from_dicts(self) -> None:
        ss = SearchSpace(
            embedding_models=["e"],
            llm_models=[
                "model-a",
                {"model": "model-b", "reasoning": True},
                {"model": "model-c", "reasoning": False},
            ],
        )
        assert ss.reasoning_overrides == {"model-b": True, "model-c": False}

    def test_dict_without_reasoning_key_no_override(self) -> None:
        ss = SearchSpace(
            embedding_models=["e"],
            llm_models=[{"model": "model-x"}],
        )
        assert "model-x" not in ss.reasoning_overrides
        assert ss.llm_models == ["model-x"]

    def test_is_reasoning_allowed_global_default_true(self) -> None:
        # Use a real model where litellm.supports_reasoning returns True
        ss = SearchSpace(
            embedding_models=["e"],
            llm_models=["anthropic/claude-haiku-4-5-20251001"],
            reasoning=True,
        )
        assert ss.is_reasoning_allowed("anthropic/claude-haiku-4-5-20251001") is True

    def test_is_reasoning_allowed_global_default_false(self) -> None:
        ss = SearchSpace(
            embedding_models=["e"],
            llm_models=["anthropic/claude-haiku-4-5-20251001"],
            reasoning=False,
        )
        assert ss.is_reasoning_allowed("anthropic/claude-haiku-4-5-20251001") is False

    def test_is_reasoning_allowed_per_model_override_wins(self) -> None:
        ss = SearchSpace(
            embedding_models=["e"],
            llm_models=[
                {"model": "vertex_ai/gemini-2.5-flash", "reasoning": True},
            ],
            reasoning=False,  # global default is off
        )
        # Per-model override should override the global default
        assert ss.is_reasoning_allowed("vertex_ai/gemini-2.5-flash") is True

    def test_is_reasoning_allowed_ollama_auto_denied(self) -> None:
        ss = SearchSpace(
            embedding_models=["e"],
            llm_models=["ollama/llama3.2"],
            reasoning=True,  # global says yes, but ollama is auto-denied
        )
        assert ss.is_reasoning_allowed("ollama/llama3.2") is False

    def test_is_reasoning_allowed_ollama_override_honored(self) -> None:
        """A per-model override can force reasoning on even for ollama (user's choice)."""
        ss = SearchSpace(
            embedding_models=["e"],
            llm_models=[{"model": "ollama/qwen2.5:7b", "reasoning": True}],
            reasoning=False,
        )
        assert ss.is_reasoning_allowed("ollama/qwen2.5:7b") is True

    def test_is_reasoning_allowed_vllm_not_auto_denied(self) -> None:
        """hosted_vllm/ models are NOT in _REASONING_UNSUPPORTED_PREFIXES."""
        from unittest.mock import patch

        ss = SearchSpace(
            embedding_models=["e"],
            llm_models=["hosted_vllm/Qwen/Qwen3-8B"],
            reasoning=True,
        )
        with patch("litellm.supports_reasoning", return_value=True):
            assert ss.is_reasoning_allowed("hosted_vllm/Qwen/Qwen3-8B") is True

    def test_is_reasoning_allowed_vllm_with_override(self) -> None:
        """Per-model override enables reasoning for vLLM models."""
        ss = SearchSpace(
            embedding_models=["e"],
            llm_models=[{"model": "hosted_vllm/Qwen/Qwen3-8B", "reasoning": True}],
            reasoning=False,
        )
        assert ss.is_reasoning_allowed("hosted_vllm/Qwen/Qwen3-8B") is True

    def test_is_reasoning_allowed_litellm_unsupported_denied(self) -> None:
        """LiteLLM capability check: model marked as not supporting reasoning is denied."""
        from unittest.mock import patch

        ss = SearchSpace(
            embedding_models=["e"],
            llm_models=["anthropic/claude-haiku-4-5-20251001"],
            reasoning=True,
        )
        with patch("litellm.supports_reasoning", return_value=False):
            assert ss.is_reasoning_allowed("anthropic/claude-haiku-4-5-20251001") is False

    def test_is_reasoning_allowed_litellm_supported(self) -> None:
        """Model confirmed by litellm.supports_reasoning is allowed when global=True."""
        from unittest.mock import patch

        ss = SearchSpace(
            embedding_models=["e"],
            llm_models=["anthropic/claude-haiku-4-5-20251001"],
            reasoning=True,
        )
        with patch("litellm.supports_reasoning", return_value=True):
            assert ss.is_reasoning_allowed("anthropic/claude-haiku-4-5-20251001") is True

    def test_is_reasoning_allowed_per_model_override_wins_over_litellm(self) -> None:
        """Per-model override=True wins even when litellm says the model is unsupported."""
        from unittest.mock import patch

        ss = SearchSpace(
            embedding_models=["e"],
            llm_models=["anthropic/claude-haiku-4-5-20251001"],
            reasoning_overrides={"anthropic/claude-haiku-4-5-20251001": True},
            reasoning=False,
        )
        with patch("litellm.supports_reasoning", return_value=False):
            assert ss.is_reasoning_allowed("anthropic/claude-haiku-4-5-20251001") is True

    def test_validate_trial_reasoning_allowed(self) -> None:
        cfg = ProjectConfig(
            search_space=SearchSpace(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                llm_models=[{"model": "vertex_ai/gemini-2.5-flash", "reasoning": True}],
                reasoning=False,
            ),
        )
        trial = TrialConfig(llm_model="vertex_ai/gemini-2.5-flash", reasoning=True)
        violations = cfg.validate_trial(trial)
        assert not any("reasoning" in v for v in violations)

    def test_validate_trial_reasoning_denied_globally(self) -> None:
        cfg = ProjectConfig(
            search_space=SearchSpace(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                llm_models=["cloud/model-a"],
                reasoning=False,
            ),
        )
        trial = TrialConfig(llm_model="cloud/model-a", reasoning=True)
        violations = cfg.validate_trial(trial)
        assert any("reasoning" in v for v in violations)

    def test_validate_trial_reasoning_denied_for_ollama(self) -> None:
        cfg = ProjectConfig(
            search_space=SearchSpace(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                llm_models=["ollama/llama3.2"],
                reasoning=True,
            ),
        )
        trial = TrialConfig(llm_model="ollama/llama3.2", reasoning=True)
        violations = cfg.validate_trial(trial)
        assert any("reasoning" in v for v in violations)

    def test_validate_trial_reasoning_false_always_ok(self) -> None:
        cfg = ProjectConfig(
            search_space=SearchSpace(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                llm_models=["ollama/llama3.2"],
                reasoning=False,
            ),
        )
        trial = TrialConfig(llm_model="ollama/llama3.2", reasoning=False)
        violations = cfg.validate_trial(trial)
        assert not any("reasoning" in v for v in violations)


class TestValidateLlmModels:
    """Tests for ProjectConfig.validate_llm_models — static + live probe validation."""

    def _make_project_config_with_models(self, llm_models: list) -> ProjectConfig:
        return ProjectConfig.model_validate(
            {
                "search_space": {
                    "embedding_models": ["sentence-transformers/all-MiniLM-L6-v2"],
                    "llm_models": llm_models,
                }
            }
        )

    def test_known_provider_suffix_passes_static_check(self) -> None:
        """anthropic/claude-haiku-4-5-20251001 is in models_by_provider — no probe needed."""
        cfg = self._make_project_config_with_models(["anthropic/claude-haiku-4-5-20251001"])
        assert "anthropic/claude-haiku-4-5-20251001" in cfg.search_space.llm_models

    def test_unknown_provider_triggers_probe_and_fails(self) -> None:
        """cloud/fake-model has unknown provider → probe → failure → ValueError."""
        from unittest.mock import patch

        with (
            pytest.raises(ValidationError, match="could not be called"),
            patch("agentic_autorag.config.models._probe_model", return_value=(False, "Model not found")),
            patch.dict("litellm.models_by_provider", {}, clear=False),
        ):
            self._make_project_config_with_models(["cloud/fake-model"])

    def test_unknown_provider_triggers_probe_and_passes(self) -> None:
        """Model not in catalog but probe succeeds → config loads without error."""
        from unittest.mock import patch

        with patch("agentic_autorag.config.models._probe_model", return_value=(True, None)):
            cfg = self._make_project_config_with_models(["newprovider/some-model"])
        assert "newprovider/some-model" in cfg.search_space.llm_models

    def test_typo_in_known_provider_fails(self) -> None:
        """anthropic/cladue-haiku (typo) is not in anthropic's model list → probe → failure."""
        from unittest.mock import patch

        with (
            pytest.raises(ValidationError, match="could not be called"),
            patch("agentic_autorag.config.models._probe_model", return_value=(False, "Invalid model")),
        ):
            self._make_project_config_with_models(["anthropic/cladue-haiku-4.5"])

    def test_multiple_failures_all_reported(self) -> None:
        """All invalid models are reported together, not just the first."""
        from unittest.mock import patch

        with (
            pytest.raises(ValidationError) as exc_info,
            patch("agentic_autorag.config.models._probe_model", return_value=(False, "bad")),
        ):
            self._make_project_config_with_models(["cloud/fake-a", "cloud/fake-b"])

        err_str = str(exc_info.value)
        assert "cloud/fake-a" in err_str
        assert "cloud/fake-b" in err_str


class TestReasoningTrialConfig:
    def test_reasoning_defaults_to_false(self) -> None:
        trial = TrialConfig(llm_model="test/model")
        assert trial.reasoning is False

    def test_to_runtime_passes_reasoning(self) -> None:
        trial = TrialConfig(llm_model="test/model", reasoning=True)
        r = trial.to_runtime(reasoning_effort="high")
        assert r.reasoning is True
        assert r.reasoning_effort == "high"

    def test_to_runtime_default_effort(self) -> None:
        trial = TrialConfig(llm_model="test/model", reasoning=True)
        r = trial.to_runtime()
        assert r.reasoning_effort == "medium"

    def test_fingerprint_unchanged_by_reasoning(self) -> None:
        trial_a = TrialConfig(llm_model="test/model", reasoning=False)
        trial_b = TrialConfig(llm_model="test/model", reasoning=True)
        assert trial_a.structural_fingerprint() == trial_b.structural_fingerprint()


class TestReasoningAgentPrompt:
    def test_prompt_includes_reasoning_field(self) -> None:
        cfg = ProjectConfig(
            search_space=SearchSpace(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                llm_models=["cloud/model-a"],
            ),
        )
        prompt = cfg.to_agent_prompt()
        assert "reasoning" in prompt

    def test_prompt_shows_allowed_models(self) -> None:
        cfg = ProjectConfig(
            search_space=SearchSpace(
                embedding_models=["e"],
                llm_models=[
                    {"model": "vertex_ai/gemini-2.5-flash", "reasoning": True},
                    "vertex_ai/gemini-2.5-flash-lite",
                ],
                reasoning=False,
            ),
        )
        prompt = cfg.to_agent_prompt()
        assert "vertex_ai/gemini-2.5-flash" in prompt
        assert "allowed for" in prompt.lower() or "NOT allowed" in prompt

    def test_prompt_shows_denied_ollama(self) -> None:
        cfg = ProjectConfig(
            search_space=SearchSpace(
                embedding_models=["e"],
                llm_models=["ollama/llama3.2", "cloud/model-a"],
                reasoning=True,
            ),
        )
        prompt = cfg.to_agent_prompt()
        assert "ollama/llama3.2" in prompt
        assert "NOT allowed" in prompt

    def test_prompt_yaml_block_includes_reasoning(self) -> None:
        cfg = ProjectConfig(
            search_space=SearchSpace(
                embedding_models=["e"],
                llm_models=["cloud/model-a"],
            ),
        )
        prompt = cfg.to_agent_prompt()
        assert "reasoning: false" in prompt
