"""Tests for config models, validation, fingerprinting, and YAML loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import (
    ExaminerConfig,
    GraphConfig,
    IndexType,
    MCQQuestion,
    NumericRange,
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
    def test_valid_config(self) -> None:
        cfg = StructuralConfig(
            parser="pymupdf4llm",
            chunking_strategy="recursive",
            chunk_size=512,
            chunk_overlap=64,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_type=IndexType.VECTOR_ONLY,
        )
        assert cfg.chunk_size == 512
        assert cfg.index_type == IndexType.VECTOR_ONLY

    def test_defaults(self) -> None:
        cfg = StructuralConfig()
        assert cfg.parser == "pymupdf4llm"
        assert cfg.chunking_strategy == "recursive"
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 64

    def test_overlap_gte_size_fails(self) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap must be < chunk_size"):
            StructuralConfig(chunk_size=256, chunk_overlap=256)

    def test_overlap_gt_size_fails(self) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap must be < chunk_size"):
            StructuralConfig(chunk_size=256, chunk_overlap=300)

    def test_index_type_from_string(self) -> None:
        cfg = StructuralConfig(index_type="hybrid_bm25_vector")
        assert cfg.index_type == IndexType.HYBRID_BM25_VECTOR


class TestRuntimeConfig:
    def test_defaults(self) -> None:
        cfg = RuntimeConfig()
        assert cfg.top_k == 5
        assert cfg.reranker == "none"
        assert cfg.llm_model == "ollama/llama3.2"
        assert cfg.temperature == 0.0

    def test_custom_values(self) -> None:
        cfg = RuntimeConfig(
            top_k=10,
            hybrid_alpha=0.7,
            reranker="BAAI/bge-reranker-v2-m3",
            reranker_top_n=8,
            query_expansion="hyde",
            llm_model="ollama/mistral",
            temperature=0.3,
        )
        assert cfg.top_k == 10
        assert cfg.reranker == "BAAI/bge-reranker-v2-m3"


class TestGraphConfig:
    def test_defaults(self) -> None:
        cfg = GraphConfig()
        assert cfg.graph_backend == "networkx"
        assert cfg.traversal_depth == 2
        assert cfg.entity_types is None

    def test_with_entity_types(self) -> None:
        cfg = GraphConfig(entity_types=["Person", "Concept"])
        assert cfg.entity_types == ["Person", "Concept"]


class TestTrialConfig:
    def _make_trial(self, index_type: IndexType = IndexType.VECTOR_ONLY, **kwargs) -> TrialConfig:
        return TrialConfig(
            structural=StructuralConfig(index_type=index_type),
            runtime=RuntimeConfig(),
            **kwargs,
        )

    def test_valid_vector_only(self) -> None:
        trial = self._make_trial()
        assert trial.graph is None

    def test_graph_required_for_graph_index(self) -> None:
        with pytest.raises(ValidationError, match="graph config required"):
            self._make_trial(index_type=IndexType.GRAPH)

    def test_graph_required_for_hybrid_graph_vector(self) -> None:
        with pytest.raises(ValidationError, match="graph config required"):
            self._make_trial(index_type=IndexType.HYBRID_GRAPH_VECTOR)

    def test_graph_index_with_graph_config(self) -> None:
        trial = self._make_trial(
            index_type=IndexType.GRAPH,
            graph=GraphConfig(),
        )
        assert trial.graph is not None

    def test_fingerprint_deterministic(self) -> None:
        trial = self._make_trial()
        fp1 = trial.structural_fingerprint()
        fp2 = trial.structural_fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 12

    def test_fingerprint_changes_with_structural_param(self) -> None:
        trial_a = self._make_trial()
        trial_b = TrialConfig(
            structural=StructuralConfig(chunk_size=1024, chunk_overlap=128),
            runtime=RuntimeConfig(),
        )
        assert trial_a.structural_fingerprint() != trial_b.structural_fingerprint()

    def test_fingerprint_changes_with_parser(self) -> None:
        trial_a = self._make_trial()
        trial_b = TrialConfig(
            structural=StructuralConfig(parser="docling"),
            runtime=RuntimeConfig(),
        )
        assert trial_a.structural_fingerprint() != trial_b.structural_fingerprint()

    def test_fingerprint_unchanged_by_runtime(self) -> None:
        trial_a = TrialConfig(
            structural=StructuralConfig(),
            runtime=RuntimeConfig(top_k=5),
        )
        trial_b = TrialConfig(
            structural=StructuralConfig(),
            runtime=RuntimeConfig(top_k=15, temperature=0.9),
        )
        assert trial_a.structural_fingerprint() == trial_b.structural_fingerprint()

    def test_fingerprint_unchanged_by_traversal_depth(self) -> None:
        trial_a = TrialConfig(
            structural=StructuralConfig(index_type=IndexType.GRAPH),
            runtime=RuntimeConfig(),
            graph=GraphConfig(traversal_depth=1),
        )
        trial_b = TrialConfig(
            structural=StructuralConfig(index_type=IndexType.GRAPH),
            runtime=RuntimeConfig(),
            graph=GraphConfig(traversal_depth=3),
        )
        assert trial_a.structural_fingerprint() == trial_b.structural_fingerprint()

    def test_fingerprint_changes_with_graph_backend(self) -> None:
        trial_a = TrialConfig(
            structural=StructuralConfig(index_type=IndexType.GRAPH),
            runtime=RuntimeConfig(),
            graph=GraphConfig(graph_backend="networkx"),
        )
        trial_b = TrialConfig(
            structural=StructuralConfig(index_type=IndexType.GRAPH),
            runtime=RuntimeConfig(),
            graph=GraphConfig(graph_backend="neo4j"),
        )
        assert trial_a.structural_fingerprint() != trial_b.structural_fingerprint()

    def test_fingerprint_changes_with_entity_types(self) -> None:
        trial_a = TrialConfig(
            structural=StructuralConfig(index_type=IndexType.GRAPH),
            runtime=RuntimeConfig(),
            graph=GraphConfig(entity_types=None),
        )
        trial_b = TrialConfig(
            structural=StructuralConfig(index_type=IndexType.GRAPH),
            runtime=RuntimeConfig(),
            graph=GraphConfig(entity_types=["Person", "Concept"]),
        )
        assert trial_a.structural_fingerprint() != trial_b.structural_fingerprint()


def _make_search_space() -> SearchSpace:
    """Create a representative search space for testing."""
    return SearchSpace.model_validate(
        {
            "meta": {"project_name": "test"},
            "structural": {
                "parsers": ["pymupdf4llm", "docling"],
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
            },
            "runtime": {
                "retrieval": {
                    "top_k": {"min": 3, "max": 15},
                    "hybrid_alpha": {"min": 0.0, "max": 1.0},
                    "reranker": {
                        "models": ["none", "BAAI/bge-reranker-v2-m3"],
                        "top_n": {"min": 3, "max": 8},
                    },
                    "query_expansion": ["none", "hyde"],
                },
                "generation": {
                    "llm_models": ["ollama/llama3.2", "ollama/mistral"],
                    "temperature": {"min": 0.0, "max": 1.0},
                },
            },
            "graph": {
                "graph_backend": "networkx",
                "traversal_depth": {"min": 1, "max": 3},
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


class TestSearchSpaceValidation:
    def test_valid_trial_no_violations(self) -> None:
        ss = _make_search_space()
        trial = TrialConfig(
            structural=StructuralConfig(
                parser="pymupdf4llm",
                chunking_strategy="recursive",
                chunk_size=512,
                chunk_overlap=64,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                index_type=IndexType.VECTOR_ONLY,
            ),
            runtime=RuntimeConfig(
                top_k=5,
                hybrid_alpha=0.5,
                reranker="none",
                reranker_top_n=5,
                query_expansion="none",
                llm_model="ollama/llama3.2",
                temperature=0.0,
            ),
        )
        violations = ss.validate_trial(trial)
        assert violations == []

    def test_parser_violation(self) -> None:
        ss = _make_search_space()
        trial = TrialConfig(
            structural=StructuralConfig(parser="unstructured"),
            runtime=RuntimeConfig(llm_model="ollama/llama3.2"),
        )
        violations = ss.validate_trial(trial)
        assert any("parser" in v for v in violations)

    def test_chunking_strategy_violation(self) -> None:
        ss = _make_search_space()
        trial = TrialConfig(
            structural=StructuralConfig(chunking_strategy="semantic"),
            runtime=RuntimeConfig(llm_model="ollama/llama3.2"),
        )
        violations = ss.validate_trial(trial)
        assert any("chunking_strategy" in v for v in violations)

    def test_chunk_size_out_of_range(self) -> None:
        ss = _make_search_space()
        trial = TrialConfig(
            structural=StructuralConfig(chunk_size=2048, chunk_overlap=64),
            runtime=RuntimeConfig(llm_model="ollama/llama3.2"),
        )
        violations = ss.validate_trial(trial)
        assert any("chunk_size" in v for v in violations)

    def test_chunk_overlap_out_of_range(self) -> None:
        ss = _make_search_space()
        trial = TrialConfig(
            structural=StructuralConfig(chunk_size=512, chunk_overlap=200),
            runtime=RuntimeConfig(llm_model="ollama/llama3.2"),
        )
        violations = ss.validate_trial(trial)
        assert any("chunk_overlap" in v for v in violations)

    def test_embedding_model_violation(self) -> None:
        ss = _make_search_space()
        trial = TrialConfig(
            structural=StructuralConfig(embedding_model="unknown/model"),
            runtime=RuntimeConfig(llm_model="ollama/llama3.2"),
        )
        violations = ss.validate_trial(trial)
        assert any("embedding_model" in v for v in violations)

    def test_index_type_violation(self) -> None:
        ss = _make_search_space()
        trial = TrialConfig(
            structural=StructuralConfig(index_type=IndexType.GRAPH),
            runtime=RuntimeConfig(llm_model="ollama/llama3.2"),
            graph=GraphConfig(),
        )
        violations = ss.validate_trial(trial)
        assert any("index_type" in v for v in violations)

    def test_top_k_violation(self) -> None:
        ss = _make_search_space()
        trial = TrialConfig(
            structural=StructuralConfig(),
            runtime=RuntimeConfig(top_k=25, llm_model="ollama/llama3.2"),
        )
        violations = ss.validate_trial(trial)
        assert any("top_k" in v for v in violations)

    def test_reranker_violation(self) -> None:
        ss = _make_search_space()
        trial = TrialConfig(
            structural=StructuralConfig(),
            runtime=RuntimeConfig(
                reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
                llm_model="ollama/llama3.2",
            ),
        )
        violations = ss.validate_trial(trial)
        assert any("reranker" in v for v in violations)

    def test_llm_model_violation(self) -> None:
        ss = _make_search_space()
        trial = TrialConfig(
            structural=StructuralConfig(),
            runtime=RuntimeConfig(llm_model="openai/gpt-4o"),
        )
        violations = ss.validate_trial(trial)
        assert any("llm_model" in v for v in violations)

    def test_temperature_violation(self) -> None:
        ss = _make_search_space()
        trial = TrialConfig(
            structural=StructuralConfig(),
            runtime=RuntimeConfig(temperature=1.5, llm_model="ollama/llama3.2"),
        )
        violations = ss.validate_trial(trial)
        assert any("temperature" in v for v in violations)

    def test_query_expansion_violation(self) -> None:
        ss = _make_search_space()
        trial = TrialConfig(
            structural=StructuralConfig(),
            runtime=RuntimeConfig(query_expansion="multi_query", llm_model="ollama/llama3.2"),
        )
        violations = ss.validate_trial(trial)
        assert any("query_expansion" in v for v in violations)

    def test_multiple_violations(self) -> None:
        ss = _make_search_space()
        trial = TrialConfig(
            structural=StructuralConfig(
                parser="unstructured",
                embedding_model="unknown/model",
            ),
            runtime=RuntimeConfig(top_k=100, llm_model="openai/gpt-4o"),
        )
        violations = ss.validate_trial(trial)
        assert len(violations) >= 4

    def test_graph_traversal_depth_violation(self) -> None:
        ss = _make_search_space()
        # Manually build a trial that would pass Pydantic but violate search space
        # (graph index_type is not in this search space, but we can still check
        #  traversal_depth validation if graph config is present)
        trial = TrialConfig(
            structural=StructuralConfig(index_type=IndexType.VECTOR_ONLY),
            runtime=RuntimeConfig(llm_model="ollama/llama3.2"),
            graph=GraphConfig(traversal_depth=5),
        )
        violations = ss.validate_trial(trial)
        assert any("traversal_depth" in v for v in violations)


class TestSearchSpaceAgentPrompt:
    def test_returns_json_string(self) -> None:
        ss = _make_search_space()
        prompt = ss.to_agent_prompt()
        assert isinstance(prompt, str)
        assert "structural" in prompt
        assert "runtime" in prompt


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


class TestLoader:
    def test_load_starter_yaml(self) -> None:
        ss = load_config(CONFIGS_DIR / "starter.yaml")
        assert ss.meta.project_name == "my-rag-project"
        assert ss.meta.max_trials == 10
        assert "pymupdf4llm" in ss.structural.parsers
        assert "recursive" in ss.structural.chunking.strategies
        assert ss.structural.chunking.chunk_size.min == 256
        assert len(ss.runtime.generation.llm_models) == 1
        assert ss.graph is None

    def test_load_full_yaml(self) -> None:
        ss = load_config(CONFIGS_DIR / "full.yaml")
        assert ss.meta.max_trials == 30
        assert len(ss.structural.parsers) == 3
        assert len(ss.structural.embedding_models) == 3
        assert IndexType.GRAPH in ss.structural.index_types
        assert IndexType.HYBRID_GRAPH_VECTOR in ss.structural.index_types
        assert ss.graph is not None
        assert ss.graph.graph_backend == "networkx"
        assert len(ss.runtime.retrieval.reranker.models) == 3
        assert len(ss.runtime.generation.llm_models) == 3

    def test_load_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_loaded_search_space_validates_valid_trial(self) -> None:
        """End-to-end: load YAML → validate a trial against it."""
        ss = load_config(CONFIGS_DIR / "starter.yaml")
        trial = TrialConfig(
            structural=StructuralConfig(
                parser="pymupdf4llm",
                chunking_strategy="recursive",
                chunk_size=512,
                chunk_overlap=64,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                index_type=IndexType.VECTOR_ONLY,
            ),
            runtime=RuntimeConfig(
                top_k=5,
                hybrid_alpha=0.5,
                reranker="none",
                reranker_top_n=3,
                query_expansion="none",
                llm_model="ollama/llama3.2",
                temperature=0.3,
            ),
        )
        violations = ss.validate_trial(trial)
        assert violations == []

    def test_loaded_search_space_catches_violation(self) -> None:
        """End-to-end: load YAML → detect a trial that exceeds bounds."""
        ss = load_config(CONFIGS_DIR / "starter.yaml")
        trial = TrialConfig(
            structural=StructuralConfig(
                chunk_size=2048,
                chunk_overlap=64,
            ),
            runtime=RuntimeConfig(llm_model="ollama/llama3.2"),
        )
        violations = ss.validate_trial(trial)
        assert len(violations) > 0
        assert any("chunk_size" in v for v in violations)
