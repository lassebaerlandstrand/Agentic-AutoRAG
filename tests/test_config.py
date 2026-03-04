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
        with pytest.raises(ValidationError, match="chunk_overlap must be < chunk_size"):
            StructuralConfig(chunk_size=256, chunk_overlap=256)

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


class TestTrialConfig:
    def _make_trial(self, index_type: IndexType = IndexType.VECTOR_ONLY, **kwargs) -> TrialConfig:
        return TrialConfig(llm_model="test/model", index_type=index_type, **kwargs)

    def test_valid_vector_only(self) -> None:
        trial = self._make_trial()
        assert trial.graph_query_mode == "hybrid"
        assert trial.graph_top_k == 60

    def test_overlap_gte_size_fails(self) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap must be < chunk_size"):
            TrialConfig(llm_model="test/model", chunk_size=256, chunk_overlap=256)

    def test_graph_index_without_nested_config(self) -> None:
        """graph_only no longer requires a nested GraphConfig — params are flat."""
        trial = self._make_trial(index_type=IndexType.GRAPH_ONLY, graph_query_mode="local", graph_top_k=40)
        assert trial.index_type == IndexType.GRAPH_ONLY
        assert trial.graph_query_mode == "local"

    def test_hybrid_graph_vector_without_nested_config(self) -> None:
        trial = self._make_trial(index_type=IndexType.HYBRID_GRAPH_VECTOR)
        assert trial.index_type == IndexType.HYBRID_GRAPH_VECTOR

    def test_to_structural(self) -> None:
        trial = TrialConfig(llm_model="test/model", chunk_size=256, chunk_overlap=0)
        s = trial.to_structural()
        assert s.chunk_size == 256
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
        assert len(fp1) == 12

    def test_fingerprint_changes_with_index_building_param(self) -> None:
        trial_a = self._make_trial()
        trial_b = TrialConfig(llm_model="test/model", chunk_size=1024, chunk_overlap=128)
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


def _make_project_config() -> ProjectConfig:
    """Create a representative search space for testing."""
    return ProjectConfig.model_validate(
        {
            "meta": {"project_name": "test"},
            "search_space": {
                "chunking": {
                    "strategies": ["recursive", "fixed"],
                    "chunk_size": {"min": 256, "max": 1024},
                    "chunk_overlap": {"min": 0, "max": 128},
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
                    "chunk_size": {"min": 256, "max": 1024},
                    "chunk_overlap": {"min": 0, "max": 128},
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
        assert cfg.exam_size == 50
        assert cfg.diversity_clusters is None

    def test_explicit_clusters(self) -> None:
        cfg = ExaminerConfig(diversity_clusters=20)
        assert cfg.diversity_clusters == 20

    def test_auto_clusters_is_none(self) -> None:
        """None means 'compute automatically at runtime'."""
        cfg = ExaminerConfig(diversity_clusters=None)
        assert cfg.diversity_clusters is None


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
            chunk_size=512,
            chunk_overlap=64,
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
        trial = TrialConfig(llm_model="ollama/llama3.2", chunk_size=2048, chunk_overlap=64)
        violations = cfg.validate_trial(trial)
        assert any("chunk_size" in v for v in violations)

    def test_chunk_overlap_out_of_range(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(llm_model="ollama/llama3.2", chunk_size=512, chunk_overlap=200)
        violations = cfg.validate_trial(trial)
        assert any("chunk_overlap" in v for v in violations)

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
        for field in ["chunking_strategy", "chunk_size", "chunk_overlap", "embedding_model", "index_type"]:
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
        assert "[256, 1024]" in prompt  # chunk_size range
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
            source_chunk_id="chunk_0",
            cluster_id=0,
        )
        assert q.correct_answer == "A"
        assert q.difficulty == 0.0
        assert q.discrimination == 1.0
        assert q.guessing == 0.25

    def test_invalid_option_keys(self) -> None:
        with pytest.raises(ValidationError, match="options must have exactly keys"):
            MCQQuestion(
                id="q1",
                question="What is RAG?",
                options={"A": "Retrieval", "B": "Random", "C": "Robust"},
                correct_answer="A",
                source_chunk_id="chunk_0",
                cluster_id=0,
            )

    def test_invalid_correct_answer(self) -> None:
        with pytest.raises(ValidationError, match="correct_answer"):
            MCQQuestion(
                id="q1",
                question="What is RAG?",
                options={"A": "Retrieval", "B": "Random", "C": "Robust", "D": "Recursive"},
                correct_answer="E",
                source_chunk_id="chunk_0",
                cluster_id=0,
            )


MOCK_YAML_CONFIG = """
meta:
  project_name: "test-project"
  corpus_path: "./data/corpus/"
  corpus_description: "A small test corpus."
  output_dir: "./experiments/"
  max_trials: 10
  index_registry: true
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
    chunk_size: { min: 256, max: 1024 }
    chunk_overlap: { min: 0, max: 128 }
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
        assert cfg.search_space.chunking.chunk_size.min == 256
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
            chunk_size=512,
            chunk_overlap=64,
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
            chunk_size=2048,  # Out of range (max is 1024)
            chunk_overlap=64,
        )

        violations = cfg.validate_trial(trial)

        assert len(violations) > 0
        assert any("chunk_size" in v for v in violations)
