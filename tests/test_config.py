"""Tests for config models, validation, fingerprinting, and YAML loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import (
    AgentConfig,
    ChunkingSearchSpace,
    DiscreteValues,
    EmbeddingSearchSpace,
    ExaminerConfig,
    GeneratorSearchSpace,
    GraphBuildConfig,
    GraphRetrievalSearchSpace,
    IndexType,
    NumericDim,
    NumericRange,
    OpenEndedQuestion,
    ParsingConfig,
    PassageCompressorSearchSpace,
    ProjectConfig,
    QueryExpansionSearchSpace,
    RerankerSearchSpace,
    RetrievalSearchSpace,
    RuntimeConfig,
    SearchSpace,
    StructuralConfig,
    TrialConfig,
    VLLMConfig,
)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def _ss(
    *,
    embedding_models: list[str] = ("e1",),
    generator_models: list[str] = ("m1",),
    expander_models: list[str] | None = None,
    compressor_models: list[str] | None = None,
    reasoning: bool = True,
    reasoning_effort: str = "medium",
    index_types: list[IndexType] | None = None,
    top_k: NumericDim | None = None,
    hybrid_alpha: NumericDim | None = None,
    bm25_vector_fusion: list[str] | None = None,
    long_context_reorder: list[bool] | None = None,
    query_expansion_strategies: list[str] | None = None,
    passage_compressor_strategies: list[str] | None = None,
    chunking: ChunkingSearchSpace | None = None,
    reranker: RerankerSearchSpace | None = None,
    temperature: NumericRange | None = None,
    graph_retrieval: GraphRetrievalSearchSpace | None = None,
) -> SearchSpace:
    """Build a SearchSpace from flat kwargs.

    Test convenience over the v3 nested layout: pass per-stage pools and
    per-block knobs as flat keywords; the helper assembles the nested
    blocks. Mirrors the call style the test suite used pre-v3.
    """
    kwargs: dict = {
        "embedding": EmbeddingSearchSpace(models=list(embedding_models)),
        "generator": GeneratorSearchSpace(
            models=list(generator_models),
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
        ),
    }
    if chunking is not None:
        kwargs["chunking"] = chunking
    retrieval_kwargs: dict = {}
    if index_types is not None:
        retrieval_kwargs["index_types"] = index_types
    if top_k is not None:
        retrieval_kwargs["top_k"] = top_k
    if hybrid_alpha is not None:
        retrieval_kwargs["hybrid_alpha"] = hybrid_alpha
    if bm25_vector_fusion is not None:
        retrieval_kwargs["bm25_vector_fusion"] = bm25_vector_fusion
    if long_context_reorder is not None:
        retrieval_kwargs["long_context_reorder"] = long_context_reorder
    if retrieval_kwargs:
        kwargs["retrieval"] = RetrievalSearchSpace(**retrieval_kwargs)
    if query_expansion_strategies is not None or expander_models is not None:
        kwargs["query_expansion"] = QueryExpansionSearchSpace(
            strategies=query_expansion_strategies or ["none"],
            models=list(expander_models) if expander_models else [],
        )
    if passage_compressor_strategies is not None or compressor_models is not None:
        kwargs["passage_compressor"] = PassageCompressorSearchSpace(
            strategies=passage_compressor_strategies or ["none"],
            models=list(compressor_models) if compressor_models else [],
        )
    if reranker is not None:
        kwargs["reranker"] = reranker
    if temperature is not None:
        kwargs["temperature"] = temperature
    if graph_retrieval is not None:
        kwargs["graph_retrieval"] = graph_retrieval
    return SearchSpace(**kwargs)


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


class TestDiscreteValues:
    def test_valid_sorted(self) -> None:
        d = DiscreteValues(values=[10, 3, 5])
        assert d.values == [3, 5, 10]

    def test_contains(self) -> None:
        d = DiscreteValues(values=[3, 5, 10])
        assert d.contains(5)
        assert not d.contains(7)
        assert not d.contains(4.99)

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            DiscreteValues(values=[])

    def test_duplicates_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            DiscreteValues(values=[3, 5, 5, 10])

    def test_floats_allowed(self) -> None:
        d = DiscreteValues(values=[0.0, 0.5, 1.0])
        assert d.contains(0.5)
        assert not d.contains(0.25)


class TestNumericDimYAMLResolution:
    """The union NumericRange | DiscreteValues must resolve unambiguously."""

    def test_top_k_as_numeric_range(self) -> None:
        ss = SearchSpace.model_validate(
            {
                "embedding": {"models": ["e1"]},
                "generator": {"models": ["m1"]},
                "retrieval": {"top_k": {"min": 3, "max": 20}},
            }
        )
        assert isinstance(ss.retrieval.top_k, NumericRange)

    def test_top_k_as_discrete_values(self) -> None:
        ss = SearchSpace.model_validate(
            {
                "embedding": {"models": ["e1"]},
                "generator": {"models": ["m1"]},
                "retrieval": {"top_k": {"values": [3, 10, 20]}},
            }
        )
        assert isinstance(ss.retrieval.top_k, DiscreteValues)
        assert ss.retrieval.top_k.values == [3, 10, 20]

    def test_chunk_and_hybrid_alpha_discrete(self) -> None:
        ss = SearchSpace.model_validate(
            {
                "embedding": {"models": ["e1"]},
                "generator": {"models": ["m1"]},
                "chunking": {
                    "chunk_token_size": {"values": [256, 512]},
                    "chunk_token_overlap": {"values": [0]},
                },
                "retrieval": {"hybrid_alpha": {"values": [0.0, 0.5, 1.0]}},
                "reranker": {"top_n": {"values": [3, 5, 10]}},
            }
        )
        assert isinstance(ss.chunking.chunk_token_size, DiscreteValues)
        assert isinstance(ss.chunking.chunk_token_overlap, DiscreteValues)
        assert isinstance(ss.retrieval.hybrid_alpha, DiscreteValues)
        assert isinstance(ss.reranker.top_n, DiscreteValues)


class TestStagePoolValidation:
    """Per-stage pool classes reject empty/required-but-missing model pools."""

    def test_generator_models_required_non_empty(self) -> None:
        with pytest.raises(ValidationError, match="must be non-empty"):
            GeneratorSearchSpace(models=[])

    def test_embedding_models_required_non_empty(self) -> None:
        with pytest.raises(ValidationError, match="must be non-empty"):
            EmbeddingSearchSpace(models=[])

    def test_query_expansion_models_required_when_stage_runs(self) -> None:
        with pytest.raises(ValidationError, match="must be non-empty when"):
            QueryExpansionSearchSpace(strategies=["none", "hyde"], models=[])

    def test_passage_compressor_models_required_when_stage_runs(self) -> None:
        with pytest.raises(ValidationError, match="must be non-empty when"):
            PassageCompressorSearchSpace(strategies=["tree_summarize"], models=[])

    def test_query_expansion_models_optional_when_all_strategies_none(self) -> None:
        # Empty pool is fine when the stage never runs (strategies=["none"]).
        qe = QueryExpansionSearchSpace(strategies=["none"], models=[])
        assert qe.models == []

    def test_passage_compressor_models_optional_when_all_strategies_none(self) -> None:
        pc = PassageCompressorSearchSpace(strategies=["none"], models=[])
        assert pc.models == []

    def test_all_llm_models_dedup_preserves_order(self) -> None:
        ss = _ss(
            generator_models=["g1", "shared"],
            expander_models=["e1", "shared"],
            compressor_models=["c1", "shared"],
            query_expansion_strategies=["none", "hyde"],
            passage_compressor_strategies=["none", "tree_summarize"],
        )
        assert ss.all_llm_models() == ["g1", "shared", "e1", "c1"]


class TestSearchSpaceFeasibilityValidators:
    """SearchSpace catches structurally infeasible numeric grids at parse time.

    These validators replace silent sampler fallbacks: a misconfigured grid
    (e.g. every chunk_token_overlap >= every chunk_token_size) used to surface
    only at sample time with a snap-to-grid violation. Now it fails at load.
    """

    def _base(self, **overrides) -> dict:
        base = {
            "embedding": {"models": ["e1"]},
            "generator": {"models": ["m1"]},
        }
        base.update(overrides)
        return base

    def test_chunk_overlap_feasibility_passes(self) -> None:
        SearchSpace.model_validate(
            self._base(
                chunking={
                    "chunk_token_size": {"values": [256, 512]},
                    "chunk_token_overlap": {"values": [0, 64]},
                }
            )
        )

    def test_chunk_overlap_min_geq_size_max_rejected(self) -> None:
        with pytest.raises(ValidationError, match="chunk_token_overlap minimum"):
            SearchSpace.model_validate(
                self._base(
                    chunking={
                        "chunk_token_size": {"values": [50]},
                        "chunk_token_overlap": {"values": [100, 200]},
                    }
                )
            )

    def test_chunk_overlap_equal_to_size_rejected(self) -> None:
        with pytest.raises(ValidationError, match="chunk_token_overlap minimum"):
            SearchSpace.model_validate(
                self._base(
                    chunking={
                        "chunk_token_size": {"values": [128]},
                        "chunk_token_overlap": {"values": [128]},
                    }
                )
            )

    def test_reranker_top_n_feasibility_passes(self) -> None:
        SearchSpace.model_validate(
            self._base(
                retrieval={"top_k": {"values": [3, 10, 20]}},
                reranker={"models": ["none", "BAAI/bge-reranker-v2-m3"], "top_n": {"values": [3, 5, 10]}},
            )
        )

    def test_reranker_top_n_min_gt_top_k_max_rejected(self) -> None:
        with pytest.raises(ValidationError, match="reranker.top_n minimum"):
            SearchSpace.model_validate(
                self._base(
                    retrieval={"top_k": {"values": [3]}},
                    reranker={"models": ["BAAI/bge-reranker-v2-m3"], "top_n": {"values": [5, 10]}},
                )
            )

    def test_reranker_top_n_skipped_when_reranker_dead(self) -> None:
        SearchSpace.model_validate(
            self._base(
                retrieval={"top_k": {"values": [3]}},
                reranker={"models": ["none"], "top_n": {"values": [5, 10]}},
            )
        )


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
        cfg = RuntimeConfig(generator_llm="test/model")
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
            search_space=_ss(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                generator_models=["hosted_vllm/Qwen/Qwen3-8B"],
            ),
            agent=AgentConfig(
                optimizer_model="ollama/llama3.2",
                examiner_model="ollama/llama3.2",
                judge_model="ollama/llama3.2",
            ),
            vllm=VLLMConfig(max_model_len=4096),
        )
        assert cfg.vllm is not None
        assert cfg.vllm.max_model_len == 4096

    def test_project_config_without_vllm(self) -> None:
        cfg = ProjectConfig(
            search_space=_ss(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                generator_models=["ollama/llama3.2"],
            ),
            agent=AgentConfig(
                optimizer_model="ollama/llama3.2",
                examiner_model="ollama/llama3.2",
                judge_model="ollama/llama3.2",
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
                search_space=_ss(
                    embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                    generator_models=["hosted_vllm/Qwen/Qwen3-8B"],
                ),
                agent=AgentConfig(
                    optimizer_model="ollama/llama3.2",
                    examiner_model="ollama/llama3.2",
                    judge_model="ollama/llama3.2",
                ),
            )
            assert "hosted_vllm/Qwen/Qwen3-8B" in cfg.search_space.all_llm_models()


class TestTrialConfig:
    def _make_trial(self, index_type: IndexType = IndexType.VECTOR_ONLY, **kwargs) -> TrialConfig:
        return TrialConfig(generator_llm="test/model", index_type=index_type, **kwargs)

    def test_valid_vector_only(self) -> None:
        trial = self._make_trial()
        assert trial.graph_query_mode == "hybrid"
        assert trial.graph_top_k == 60

    def test_overlap_gte_size_fails(self) -> None:
        with pytest.raises(ValidationError, match="chunk_token_overlap must be < chunk_token_size"):
            TrialConfig(generator_llm="test/model", chunk_token_size=256, chunk_token_overlap=256)

    def test_graph_index_without_nested_config(self) -> None:
        """graph_only no longer requires a nested GraphConfig — params are flat."""
        trial = self._make_trial(index_type=IndexType.GRAPH_ONLY, graph_query_mode="local", graph_top_k=40)
        assert trial.index_type == IndexType.GRAPH_ONLY
        assert trial.graph_query_mode == "local"

    def test_hybrid_graph_vector_without_nested_config(self) -> None:
        trial = self._make_trial(index_type=IndexType.HYBRID_GRAPH_VECTOR)
        assert trial.index_type == IndexType.HYBRID_GRAPH_VECTOR

    def test_to_structural(self) -> None:
        trial = TrialConfig(generator_llm="test/model", chunk_token_size=256, chunk_token_overlap=0)
        s = trial.to_structural()
        assert s.chunk_token_size == 256
        assert s.embedding_model == trial.embedding_model

    def test_to_runtime(self) -> None:
        trial = TrialConfig(
            generator_llm="test/model",
            top_k=10,
            temperature=0.5,
            graph_query_mode="global",
            graph_top_k=50,
        )
        r = trial.to_runtime()
        assert r.top_k == 10
        assert r.generator_llm == "test/model"
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
        trial_b = TrialConfig(generator_llm="test/model", chunk_token_size=1024, chunk_token_overlap=128)
        assert trial_a.structural_fingerprint() != trial_b.structural_fingerprint()

    def test_fingerprint_unchanged_by_retrieval_params(self) -> None:
        trial_a = TrialConfig(generator_llm="test/model", top_k=5)
        trial_b = TrialConfig(generator_llm="test/model", top_k=15, temperature=0.9)
        assert trial_a.structural_fingerprint() == trial_b.structural_fingerprint()

    def test_fingerprint_unchanged_by_graph_retrieval_params(self) -> None:
        """Graph query mode/top_k are runtime params — they don't change the vector index."""
        trial_a = TrialConfig(generator_llm="test/model", graph_query_mode="local", graph_top_k=20)
        trial_b = TrialConfig(generator_llm="test/model", graph_query_mode="global", graph_top_k=80)
        assert trial_a.structural_fingerprint() == trial_b.structural_fingerprint()

    def test_fingerprint_unchanged_by_index_type(self) -> None:
        """index_type only routes queries — it does not change the cached index data."""
        from agentic_autorag.config.models import IndexType

        trial_a = TrialConfig(generator_llm="test/model", index_type=IndexType.VECTOR_ONLY)
        trial_b = TrialConfig(generator_llm="test/model", index_type=IndexType.HYBRID_BM25_VECTOR)
        trial_c = TrialConfig(generator_llm="test/model", index_type=IndexType.HYBRID_GRAPH_VECTOR)
        assert trial_a.structural_fingerprint() == trial_b.structural_fingerprint()
        assert trial_a.structural_fingerprint() == trial_c.structural_fingerprint()

    def test_to_prompt_json_excludes_graph_when_disabled(self) -> None:
        trial = self._make_trial()
        result = trial.to_prompt_json(include_graph=False)
        assert "graph_query_mode" not in result
        assert "graph_top_k" not in result
        assert "generator_llm" in result

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
        assert "generator_llm" in result

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
            "agent": {
                "optimizer_model": "ollama/llama3.2",
                "examiner_model": "ollama/llama3.2",
                "judge_model": "ollama/llama3.2",
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
                        "BAAI/bge-m3",
                    ],
                },
                "retrieval": {
                    "index_types": ["vector_only", "hybrid_bm25_vector"],
                    "top_k": {"min": 3, "max": 15},
                    "hybrid_alpha": {"min": 0.0, "max": 1.0},
                    "bm25_vector_fusion": ["alpha", "rrf"],
                    "long_context_reorder": [False, True],
                },
                "passage_compressor": {
                    "strategies": ["none", "tree_summarize"],
                    "models": ["ollama/llama3.2", "ollama/mistral"],
                },
                "reranker": {
                    "models": ["none", "BAAI/bge-reranker-v2-m3"],
                    "top_n": {"min": 3, "max": 8},
                },
                "query_expansion": {
                    "strategies": ["none", "hyde"],
                    "models": ["ollama/llama3.2", "ollama/mistral"],
                },
                "generator": {
                    "models": ["ollama/llama3.2", "ollama/mistral"],
                },
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
            "agent": {
                "optimizer_model": "ollama/llama3.2",
                "examiner_model": "ollama/llama3.2",
                "judge_model": "ollama/llama3.2",
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
                        "BAAI/bge-m3",
                    ],
                },
                "retrieval": {
                    "index_types": ["vector_only", "graph_only", "hybrid_graph_vector"],
                    "top_k": {"min": 3, "max": 15},
                    "hybrid_alpha": {"min": 0.0, "max": 1.0},
                },
                "reranker": {
                    "models": ["none", "BAAI/bge-reranker-v2-m3"],
                    "top_n": {"min": 3, "max": 8},
                },
                "query_expansion": {
                    "strategies": ["none", "hyde"],
                    "models": ["ollama/llama3.2", "ollama/mistral"],
                },
                "generator": {
                    "models": ["ollama/llama3.2", "ollama/mistral"],
                },
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
        assert cfg.exam_size == 80
        assert cfg.initial_question_multiplier == 1.5
        assert cfg.composition_temperature == 1.0
        assert cfg.probe_selection is True
        assert cfg.neighborhood_min_chunks == 12
        assert cfg.neighborhood_min_words == 5000
        assert cfg.neighborhood_same_doc_weight == 0.8
        assert cfg.neighborhood_cross_doc_weight == 0.2
        assert cfg.source_fact_verify_fuzzy_threshold == 0.9
        assert cfg.chunk_relevance_min_overlap_chars == 50
        assert cfg.chunk_relevance_ngram_size == 5
        assert cfg.chunk_relevance_overlap_threshold == 0.5
        assert cfg.chunk_relevance_min_run == 5
        assert cfg.max_chunk_words == 1_000
        assert cfg.min_doc_words == 200

    def test_composition_temperature_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ExaminerConfig(composition_temperature=-0.1)
        with pytest.raises(ValidationError):
            ExaminerConfig(composition_temperature=2.1)
        cfg = ExaminerConfig(composition_temperature=0.7)
        assert cfg.composition_temperature == 0.7

    def test_initial_question_multiplier_below_one_invalid(self) -> None:
        with pytest.raises(ValidationError):
            ExaminerConfig(initial_question_multiplier=0.5)

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

    def test_min_doc_words_non_negative(self) -> None:
        cfg = ExaminerConfig(min_doc_words=0)
        assert cfg.min_doc_words == 0

    def test_neighborhood_weights_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            ExaminerConfig(neighborhood_same_doc_weight=-0.1)
        with pytest.raises(ValidationError):
            ExaminerConfig(neighborhood_cross_doc_weight=-0.1)

    def test_neighborhood_weights_normalize_freely(self) -> None:
        cfg = ExaminerConfig(neighborhood_same_doc_weight=3.0, neighborhood_cross_doc_weight=7.0)
        assert cfg.neighborhood_same_doc_weight == 3.0
        assert cfg.neighborhood_cross_doc_weight == 7.0

    def test_neighborhood_weights_both_zero_invalid(self) -> None:
        with pytest.raises(ValidationError):
            ExaminerConfig(neighborhood_same_doc_weight=0.0, neighborhood_cross_doc_weight=0.0)

    def test_neighborhood_min_chunks_at_least_one(self) -> None:
        with pytest.raises(ValidationError):
            ExaminerConfig(neighborhood_min_chunks=0)

    def test_neighborhood_min_words_non_negative(self) -> None:
        cfg = ExaminerConfig(neighborhood_min_words=0)
        assert cfg.neighborhood_min_words == 0
        with pytest.raises(ValidationError):
            ExaminerConfig(neighborhood_min_words=-1)


class TestAgentConfig:
    @staticmethod
    def _agent(**overrides) -> AgentConfig:
        base = dict(
            optimizer_model="gemini/gemini-3-flash-preview",
            examiner_model="gemini/gemini-3-flash-preview",
            judge_model="gemini/gemini-3-flash-preview",
        )
        base.update(overrides)
        return AgentConfig(**base)

    def test_models_are_required(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig()
        with pytest.raises(ValidationError):
            AgentConfig(optimizer_model="x", examiner_model="y")  # judge_model missing

    def test_defaults(self) -> None:
        cfg = self._agent()
        assert cfg.concurrency == 10
        assert cfg.optimizer_reasoning_effort == "medium"
        assert cfg.examiner_reasoning_effort == "medium"

    def test_explicit_concurrency(self) -> None:
        cfg = self._agent(concurrency=3)
        assert cfg.concurrency == 3

    def test_concurrency_zero_is_invalid(self) -> None:
        with pytest.raises(ValidationError):
            self._agent(concurrency=0)

    def test_concurrency_negative_is_invalid(self) -> None:
        with pytest.raises(ValidationError):
            self._agent(concurrency=-1)

    def test_examiner_reasoning_effort_accepts_levels(self) -> None:
        for level in ("low", "medium", "high"):
            cfg = self._agent(examiner_reasoning_effort=level)
            assert cfg.examiner_reasoning_effort == level

    def test_examiner_reasoning_effort_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            self._agent(examiner_reasoning_effort="extreme")


class TestParsingConfig:
    def test_defaults(self) -> None:
        cfg = ParsingConfig()
        assert cfg.parser == "docling"
        assert cfg.ocr is True
        assert cfg.table_structure is True
        assert cfg.near_duplicate_threshold == 0.85
        assert cfg.near_duplicate_detection_enabled is True

    def test_custom_values(self) -> None:
        cfg = ParsingConfig(ocr=False, table_structure=False)
        assert cfg.ocr is False
        assert cfg.table_structure is False

    def test_parser_rejects_unsupported_value(self) -> None:
        with pytest.raises(ValidationError):
            ParsingConfig(parser="pymupdf4llm")

    def test_threshold_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ParsingConfig(near_duplicate_threshold=-0.1)
        with pytest.raises(ValidationError):
            ParsingConfig(near_duplicate_threshold=1.1)
        cfg = ParsingConfig(near_duplicate_threshold=1.0)
        assert cfg.near_duplicate_threshold == 1.0


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
            generator_llm="ollama/llama3.2",
            temperature=0.0,
        )
        violations = cfg.validate_trial(trial)
        assert violations == []

    def test_chunking_strategy_violation(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(generator_llm="ollama/llama3.2", chunking_strategy="semantic")
        violations = cfg.validate_trial(trial)
        assert any("chunking_strategy" in v for v in violations)

    def test_chunk_size_out_of_range(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(generator_llm="ollama/llama3.2", chunk_token_size=2048, chunk_token_overlap=64)
        violations = cfg.validate_trial(trial)
        assert any("chunk_token_size" in v for v in violations)

    def test_chunk_overlap_out_of_range(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(generator_llm="ollama/llama3.2", chunk_token_size=512, chunk_token_overlap=200)
        violations = cfg.validate_trial(trial)
        assert any("chunk_token_overlap" in v for v in violations)

    def test_embedding_model_violation(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(generator_llm="ollama/llama3.2", embedding_model="unknown/model")
        violations = cfg.validate_trial(trial)
        assert any("embedding_model" in v for v in violations)

    def test_index_type_violation(self) -> None:
        cfg = _make_project_config()
        # graph_only is not in this search space (no graph config either)
        trial = TrialConfig(generator_llm="ollama/llama3.2", index_type=IndexType.GRAPH_ONLY)
        violations = cfg.validate_trial(trial)
        assert any("index_type" in v for v in violations)

    def test_top_k_violation(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(generator_llm="ollama/llama3.2", top_k=25)
        violations = cfg.validate_trial(trial)
        assert any("top_k" in v for v in violations)

    def test_reranker_violation(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(generator_llm="ollama/llama3.2", reranker="cross-encoder/ms-marco-MiniLM-L-6-v2")
        violations = cfg.validate_trial(trial)
        assert any("reranker" in v for v in violations)

    def test_generator_llm_violation(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(generator_llm="openai/gpt-4o")
        violations = cfg.validate_trial(trial)
        assert any("generator_llm" in v for v in violations)

    def test_temperature_violation(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(generator_llm="ollama/llama3.2", temperature=1.5)
        violations = cfg.validate_trial(trial)
        assert any("temperature" in v for v in violations)

    def test_query_expansion_violation(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(
            generator_llm="ollama/llama3.2",
            query_expansion="multi_query",
            expander_llm="ollama/llama3.2",
        )
        violations = cfg.validate_trial(trial)
        assert any("query_expansion" in v for v in violations)

    def test_bm25_vector_fusion_violation(self) -> None:
        """A proposer that emits ``bm25_vector_fusion`` outside the enumerated
        list raises a violation — defends against an agent that hallucinates
        the dimension's allowed values."""
        cfg = _make_project_config()
        cfg.search_space.retrieval.bm25_vector_fusion = ["alpha"]
        trial = TrialConfig(generator_llm="ollama/llama3.2", bm25_vector_fusion="rrf")
        violations = cfg.validate_trial(trial)
        assert any("bm25_vector_fusion" in v for v in violations)

    def test_long_context_reorder_violation(self) -> None:
        cfg = _make_project_config()
        cfg.search_space.retrieval.long_context_reorder = [False]
        trial = TrialConfig(generator_llm="ollama/llama3.2", long_context_reorder=True)
        violations = cfg.validate_trial(trial)
        assert any("long_context_reorder" in v for v in violations)

    def test_passage_compressor_violation(self) -> None:
        cfg = _make_project_config()
        cfg.search_space.passage_compressor.strategies = ["none"]
        trial = TrialConfig(
            generator_llm="ollama/llama3.2",
            passage_compressor="tree_summarize",
            compressor_llm="ollama/llama3.2",
        )
        violations = cfg.validate_trial(trial)
        assert any("passage_compressor" in v for v in violations)

    def test_multiple_violations(self) -> None:
        cfg = _make_project_config()
        trial = TrialConfig(generator_llm="openai/gpt-4o", embedding_model="unknown/model", top_k=100)
        violations = cfg.validate_trial(trial)
        assert len(violations) >= 3

    def test_graph_query_mode_violation(self) -> None:
        cfg = _make_project_config_with_graph()
        trial = TrialConfig(
            generator_llm="ollama/llama3.2",
            index_type=IndexType.GRAPH_ONLY,
            graph_query_mode="naive",  # not in allowed modes
            graph_top_k=50,
        )
        violations = cfg.validate_trial(trial)
        assert any("graph_query_mode" in v for v in violations)

    def test_graph_top_k_violation(self) -> None:
        cfg = _make_project_config_with_graph()
        trial = TrialConfig(
            generator_llm="ollama/llama3.2",
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
            generator_llm="ollama/llama3.2",
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
        assert "Tunable parameters" in prompt

    def test_excludes_meta_examiner_agent(self) -> None:
        cfg = _make_project_config()
        prompt = cfg.to_agent_prompt()
        assert "project_name" not in prompt
        assert "exam_size" not in prompt
        assert "optimizer_model" not in prompt
        assert "examiner_model" not in prompt
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
            "generator_llm",
            "temperature",
        ]:
            assert field in prompt, f"Missing runtime field: {field}"

    def test_contains_yaml_example_block(self) -> None:
        cfg = _make_project_config()
        prompt = cfg.to_agent_prompt()
        assert "```yaml" in prompt
        assert "```" in prompt
        assert "ollama/llama3.2" in prompt  # first generator_llm

    def test_includes_graph_params_when_present(self) -> None:
        cfg = _make_project_config_with_graph()
        prompt = cfg.to_agent_prompt()
        assert "graph_query_mode" in prompt
        assert "graph_top_k" in prompt

    def test_excludes_graph_params_when_absent(self) -> None:
        cfg = ProjectConfig(
            search_space=_ss(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                generator_models=["ollama/llama3.2"],
            ),
            agent=AgentConfig(
                optimizer_model="ollama/llama3.2",
                examiner_model="ollama/llama3.2",
                judge_model="ollama/llama3.2",
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


class TestPinnedFieldValues:
    """``SearchSpace.pinned_field_values()`` covers every variety of pin."""

    def test_fully_tunable_returns_empty(self) -> None:
        cfg = _make_project_config()
        assert cfg.search_space.pinned_field_values() == {}

    def test_numeric_range_pin(self) -> None:
        ss = _ss(top_k=NumericRange(min=5, max=5))
        pinned = ss.pinned_field_values()
        assert pinned["top_k"] == 5

    def test_chunk_overlap_pinned_to_zero(self) -> None:
        """The exact case from the reported run: overlap pinned at 0."""
        ss = _ss(
            embedding_models=["e1", "e2"],
            chunking=ChunkingSearchSpace(
                strategies=["recursive"],
                chunk_token_size=NumericRange(min=256, max=256),
                chunk_token_overlap=NumericRange(min=0, max=0),
            ),
        )
        pinned = ss.pinned_field_values()
        assert pinned["chunk_token_size"] == 256
        assert pinned["chunk_token_overlap"] == 0
        assert pinned["chunking_strategy"] == "recursive"

    def test_single_choice_list_pin(self) -> None:
        ss = _ss(embedding_models=["only-one"], generator_models=["m1", "m2"])
        pinned = ss.pinned_field_values()
        assert pinned["embedding_model"] == "only-one"
        assert "generator_llm" not in pinned  # two choices — tunable

    def test_index_type_single_choice_pin(self) -> None:
        ss = _ss(index_types=[IndexType.VECTOR_ONLY])
        pinned = ss.pinned_field_values()
        assert pinned["index_type"] == "vector_only"

    def test_reranker_top_n_dead_when_only_none(self) -> None:
        ss = _ss(reranker=RerankerSearchSpace(models=["none"], top_n=NumericRange(min=3, max=10)))
        pinned = ss.pinned_field_values()
        assert "reranker_top_n" in pinned
        assert pinned["reranker_top_n"] == 3  # uses min as the dead default

    def test_reranker_top_n_tunable_when_real_reranker_in_space(self) -> None:
        ss = _ss(
            reranker=RerankerSearchSpace(
                models=["none", "BAAI/bge-reranker-v2-m3"],
                top_n=NumericRange(min=3, max=10),
            ),
        )
        pinned = ss.pinned_field_values()
        assert "reranker_top_n" not in pinned
        assert "reranker" not in pinned  # two reranker choices is tunable

    def test_hybrid_alpha_dead_when_no_hybrid_index(self) -> None:
        ss = _ss(index_types=[IndexType.VECTOR_ONLY])
        pinned = ss.pinned_field_values()
        assert "hybrid_alpha" in pinned

    def test_hybrid_alpha_tunable_when_hybrid_in_space(self) -> None:
        ss = _ss(
            index_types=[IndexType.VECTOR_ONLY, IndexType.HYBRID_BM25_VECTOR],
            bm25_vector_fusion=["alpha", "rrf"],
        )
        pinned = ss.pinned_field_values()
        assert "hybrid_alpha" not in pinned

    def test_hybrid_alpha_dead_when_only_rrf_fusion(self) -> None:
        """``hybrid_alpha`` is dead under pure RRF fusion (the value is
        ignored by ``_rrf_merge``). Pinned to its min so the YAML stays valid."""
        ss = _ss(
            index_types=[IndexType.HYBRID_BM25_VECTOR],
            bm25_vector_fusion=["rrf"],
        )
        pinned = ss.pinned_field_values()
        assert "hybrid_alpha" in pinned

    def test_bm25_vector_fusion_dead_when_no_hybrid_index(self) -> None:
        ss = _ss(
            index_types=[IndexType.VECTOR_ONLY],
            bm25_vector_fusion=["alpha", "rrf"],
        )
        pinned = ss.pinned_field_values()
        # Dead → pinned regardless of how many values are enumerated.
        assert "bm25_vector_fusion" in pinned
        assert pinned["bm25_vector_fusion"] == "alpha"

    def test_bm25_vector_fusion_tunable_when_hybrid_index(self) -> None:
        ss = _ss(
            index_types=[IndexType.HYBRID_BM25_VECTOR],
            bm25_vector_fusion=["alpha", "rrf"],
        )
        pinned = ss.pinned_field_values()
        assert "bm25_vector_fusion" not in pinned

    def test_long_context_reorder_dead_when_compressor_always_on(self) -> None:
        """When compressor always collapses retrieval to a single string,
        long_context_reorder is a no-op (len ≤ 1 ⇒ no duplication)."""
        ss = _ss(
            passage_compressor_strategies=["tree_summarize", "refine"],
            compressor_models=["m1"],
            long_context_reorder=[False, True],
        )
        pinned = ss.pinned_field_values()
        assert "long_context_reorder" in pinned

    def test_long_context_reorder_tunable_when_compressor_optional(self) -> None:
        """If ``"none"`` is enumerated for ``passage_compressor.strategies``,
        reorder is still potentially active."""
        ss = _ss(
            passage_compressor_strategies=["none", "tree_summarize"],
            compressor_models=["m1"],
            long_context_reorder=[False, True],
        )
        pinned = ss.pinned_field_values()
        assert "long_context_reorder" not in pinned

    def test_temperature_pinned_when_min_equals_max(self) -> None:
        ss = _ss(temperature=NumericRange(min=1.0, max=1.0))
        pinned = ss.pinned_field_values()
        assert pinned["temperature"] == 1.0

    def test_graph_pins_when_graph_retrieval_single_valued(self) -> None:
        ss = _ss(
            index_types=[IndexType.VECTOR_ONLY, IndexType.GRAPH_ONLY],
            graph_retrieval=GraphRetrievalSearchSpace(
                graph_query_modes=["hybrid"],
                graph_top_k=NumericRange(min=60, max=60),
            ),
        )
        pinned = ss.pinned_field_values()
        assert pinned["graph_query_mode"] == "hybrid"
        assert pinned["graph_top_k"] == 60

    def test_reasoning_pinned_when_generator_reasoning_false(self) -> None:
        """When the generator stage disables reasoning, ``reasoning`` is pinned
        to ``False`` so injection silently overrides any ``reasoning: true`` an
        agent emits — preventing a misleading per-model validation error from
        firing on a global gate."""
        ss = _ss(reasoning=False)
        pinned = ss.pinned_field_values()
        assert pinned["reasoning"] is False

    def test_reasoning_not_in_pinned_when_generator_reasoning_true(self) -> None:
        """When reasoning is tunable in the search space, it's not pinned —
        even if no model in the space happens to support reasoning_effort
        (that corner case is handled by the rendering path in to_agent_prompt)."""
        ss = _ss(reasoning=True)
        assert "reasoning" not in ss.pinned_field_values()

    def test_compressor_llm_pinned_to_none_when_stage_dead(self) -> None:
        """All-``none`` strategies → compressor_llm pinned to None statically."""
        ss = _ss(passage_compressor_strategies=["none"])
        pinned = ss.pinned_field_values()
        assert pinned["compressor_llm"] is None

    def test_compressor_llm_pinned_to_model_when_no_none_and_single_pool(self) -> None:
        """No ``none`` in strategies + single-model pool → compressor_llm
        statically pinned to that model."""
        ss = _ss(
            passage_compressor_strategies=["tree_summarize"],
            compressor_models=["azure/gpt-4o-mini"],
        )
        pinned = ss.pinned_field_values()
        assert pinned["compressor_llm"] == "azure/gpt-4o-mini"

    def test_compressor_llm_derived_when_mixed_strategies_and_single_pool(self) -> None:
        """The reported bug: mixed strategies + single-model pool → NOT
        statically pinned. Resolved at injection time by the proposer's
        ``passage_compressor`` choice."""
        ss = _ss(
            passage_compressor_strategies=["none", "tree_summarize", "refine"],
            compressor_models=["azure/gpt-4o-mini"],
        )
        pinned = ss.pinned_field_values()
        assert "compressor_llm" not in pinned
        assert ss.compressor_llm_is_derived() is True

    def test_compressor_llm_tunable_when_multi_model_pool(self) -> None:
        """Multi-LLM pool → compressor_llm is fully tunable, not pinned, not derived."""
        ss = _ss(
            passage_compressor_strategies=["none", "tree_summarize"],
            compressor_models=["azure/gpt-4o-mini", "azure/o4-mini"],
        )
        pinned = ss.pinned_field_values()
        assert "compressor_llm" not in pinned
        assert ss.compressor_llm_is_derived() is False

    def test_expander_llm_derived_when_mixed_strategies_and_single_pool(self) -> None:
        """Same conditional-pinning rule for the expander stage."""
        ss = _ss(
            query_expansion_strategies=["none", "hyde", "multi_query"],
            expander_models=["azure/gpt-4o-mini"],
        )
        pinned = ss.pinned_field_values()
        assert "expander_llm" not in pinned
        assert ss.expander_llm_is_derived() is True


class TestPinnedRenderingInAgentPrompt:
    """``to_agent_prompt`` partitions tunable vs pinned and keeps pinned out of the example."""

    def _hotpot_dev_like_config(self) -> ProjectConfig:
        """The shape that triggered the reported bug: chunking + reranker pinned,
        only embedding/top_k/llm tunable."""
        return ProjectConfig.model_validate(
            {
                "agent": {
                    "optimizer_model": "ollama/llama3.2",
                    "examiner_model": "ollama/llama3.2",
                    "judge_model": "ollama/llama3.2",
                },
                "search_space": {
                    "chunking": {
                        "strategies": ["recursive"],
                        "chunk_token_size": {"min": 256, "max": 256},
                        "chunk_token_overlap": {"min": 0, "max": 0},
                    },
                    "embedding": {"models": ["e1", "e2", "e3"]},
                    "retrieval": {
                        "index_types": ["vector_only"],
                        "top_k": {"min": 3, "max": 10},
                        "hybrid_alpha": {"min": 0.0, "max": 1.0},
                    },
                    "reranker": {"models": ["none"], "top_n": {"min": 3, "max": 5}},
                    "query_expansion": {"strategies": ["none"], "models": []},
                    "generator": {
                        "models": ["ollama/llama3.2", "ollama/mistral"],
                        "reasoning": False,
                    },
                    "temperature": {"min": 1.0, "max": 1.0},
                },
            }
        )

    def test_pinned_section_appears_with_fixed_fields(self) -> None:
        cfg = self._hotpot_dev_like_config()
        prompt = cfg.to_agent_prompt()
        assert "Fixed values for this run" in prompt
        # Every pinned field appears in the Fixed-values section.
        for field, value in [
            ("chunking_strategy", "recursive"),
            ("chunk_token_size", "256"),
            ("chunk_token_overlap", "0"),
            ("index_type", "vector_only"),
            ("reranker", "none"),
            ("temperature", "1.0"),
            ("query_expansion", "none"),
            ("reasoning", "false"),
        ]:
            assert f"{field}: {value}" in prompt, f"Pinned line missing for {field}"

    def test_example_yaml_omits_pinned_fields(self) -> None:
        cfg = self._hotpot_dev_like_config()
        prompt = cfg.to_agent_prompt()
        # Split out the example YAML block so we only inspect emission surface.
        start = prompt.index("```yaml")
        end = prompt.index("```", start + 7)
        example = prompt[start:end]
        for pinned_field in [
            "chunking_strategy:",
            "chunk_token_size:",
            "chunk_token_overlap:",
            "index_type:",
            "reranker:",
            "reranker_top_n:",
            "query_expansion:",
            "temperature:",
            "hybrid_alpha:",
            "reasoning:",
            "compressor_llm:",
            "expander_llm:",
        ]:
            assert pinned_field not in example, f"pinned field {pinned_field!r} leaked into example YAML"
        # The genuinely tunable fields must be present in the example.
        for tunable_field in ["embedding_model:", "top_k:", "generator_llm:"]:
            assert tunable_field in example, f"tunable field {tunable_field!r} missing from example YAML"

    def test_dead_knob_comments_render(self) -> None:
        cfg = self._hotpot_dev_like_config()
        prompt = cfg.to_agent_prompt()
        assert "# dead" in prompt

    def test_tunable_section_only_shows_tunable_fields(self) -> None:
        cfg = self._hotpot_dev_like_config()
        prompt = cfg.to_agent_prompt()
        tunable_block = prompt.split("### Fixed values")[0]
        assert "embedding_model:" in tunable_block
        assert "top_k:" in tunable_block
        assert "generator_llm:" in tunable_block
        # Pinned fields' declarations don't appear in the tunable block.
        # (They DO appear in the "Fixed values" block, which is excluded above.)
        assert "chunking_strategy:" not in tunable_block
        assert "chunk_token_size:" not in tunable_block
        assert "chunk_token_overlap:" not in tunable_block

    def test_partially_pinned_range_keeps_top_k_tunable(self) -> None:
        cfg = _make_project_config()  # top_k {min: 3, max: 15} — tunable
        prompt = cfg.to_agent_prompt()
        assert "top_k:" in prompt
        # The Fixed-values block is absent when everything is tunable.
        assert "Fixed values for this run" not in prompt


class TestProjectConfigConsistency:
    """Tests for the graph/search-space consistency validator."""

    def test_graph_index_without_graph_config_raises(self) -> None:
        with pytest.raises(ValidationError, match="graph"):
            ProjectConfig.model_validate(
                {
                    "agent": {
                        "optimizer_model": "ollama/llama3.2",
                        "examiner_model": "ollama/llama3.2",
                        "judge_model": "ollama/llama3.2",
                    },
                    "search_space": {
                        "embedding": {"models": ["sentence-transformers/all-MiniLM-L6-v2"]},
                        "retrieval": {"index_types": ["graph_only"]},
                        "generator": {"models": ["ollama/llama3.2"]},
                    },
                }
            )

    def test_graph_retrieval_without_graph_index_raises(self) -> None:
        with pytest.raises(ValidationError, match="graph_retrieval"):
            ProjectConfig.model_validate(
                {
                    "agent": {
                        "optimizer_model": "ollama/llama3.2",
                        "examiner_model": "ollama/llama3.2",
                        "judge_model": "ollama/llama3.2",
                    },
                    "search_space": {
                        "embedding": {"models": ["sentence-transformers/all-MiniLM-L6-v2"]},
                        "retrieval": {"index_types": ["vector_only"]},
                        "generator": {"models": ["ollama/llama3.2"]},
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
                "agent": {
                    "optimizer_model": "ollama/llama3.2",
                    "examiner_model": "ollama/llama3.2",
                    "judge_model": "ollama/llama3.2",
                },
                "search_space": {
                    "embedding": {"models": ["sentence-transformers/all-MiniLM-L6-v2"]},
                    "retrieval": {"index_types": ["vector_only"]},
                    "generator": {"models": ["ollama/llama3.2"]},
                },
            }
        )
        assert cfg.graph is None
        assert not cfg.uses_graph()

    def test_graph_index_with_graph_config_ok(self) -> None:
        cfg = ProjectConfig.model_validate(
            {
                "graph": {"extraction_model": "gemini/test"},
                "agent": {
                    "optimizer_model": "ollama/llama3.2",
                    "examiner_model": "ollama/llama3.2",
                    "judge_model": "ollama/llama3.2",
                },
                "search_space": {
                    "embedding": {"models": ["sentence-transformers/all-MiniLM-L6-v2"]},
                    "retrieval": {"index_types": ["vector_only", "graph_only"]},
                    "generator": {"models": ["ollama/llama3.2"]},
                    "graph_retrieval": {
                        "graph_query_modes": ["hybrid"],
                        "graph_top_k": {"min": 20, "max": 100},
                    },
                },
            }
        )
        assert cfg.graph is not None
        assert cfg.uses_graph()


class TestOpenEndedQuestion:
    def _make(self, **overrides) -> OpenEndedQuestion:
        defaults = dict(
            id="q1",
            question="Who founded the company that Acme acquired?",
            canonical_answer="Sarah Smith",
            answer_variants=["S. Smith"],
            reasoning_type="bridge",
            source_chunk_ids=["doc_a::chunk_0", "doc_b::chunk_0"],
            source_doc_ids=["doc_a", "doc_b"],
            source_spans=[
                "In 1998 Acme Corp acquired Beta Inc.",
                "Beta Inc was founded by Sarah Smith.",
            ],
        )
        defaults.update(overrides)
        return OpenEndedQuestion(**defaults)

    def test_valid_question(self) -> None:
        q = self._make()
        assert q.canonical_answer == "Sarah Smith"
        assert q.answer_variants == ["S. Smith"]
        assert q.reasoning_type == "bridge"
        assert q.num_hops == 2
        assert q.is_multi_doc is True
        assert q.probe_outcomes == []

    def test_gold_answers_includes_canonical_and_variants(self) -> None:
        q = self._make(answer_variants=["JFK", "Kennedy"])
        assert q.gold_answers == ["Sarah Smith", "JFK", "Kennedy"]

    def test_single_hop_question(self) -> None:
        q = self._make(
            source_chunk_ids=["only::chunk_0"],
            source_doc_ids=["only"],
            source_spans=["The single span text."],
            reasoning_type="bridge",
        )
        assert q.num_hops == 1
        assert q.is_multi_doc is False

    def test_misaligned_lists_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must align"):
            self._make(source_doc_ids=["doc_a"])

    def test_empty_source_chunk_ids_invalid(self) -> None:
        with pytest.raises(ValidationError, match="source_chunk_ids must not be empty"):
            self._make(source_chunk_ids=[], source_doc_ids=[], source_spans=[])

    def test_blank_canonical_answer_rejected(self) -> None:
        with pytest.raises(ValidationError, match="canonical_answer"):
            self._make(canonical_answer="   ")

    def test_blank_source_spans_rejected(self) -> None:
        with pytest.raises(ValidationError, match="source_spans"):
            self._make(source_spans=["   ", "valid span"])


MOCK_YAML_CONFIG = """
meta:
  project_name: "test-project"
  corpus_path: "./data/corpus/"
  corpus_description: "A small test corpus."
  output_dir: "./experiments/"
  max_trials: 10
parsing:
  ocr: false
  table_structure: true
agent:
  optimizer_model: "ollama/llama3.2"
  examiner_model: "ollama/llama3.2"
  judge_model: "ollama/llama3.2"
graph:
  extraction_model: "gemini/gemini-2.5-flash-lite"
  chunk_token_size: 1200
search_space:
  chunking:
    strategies: ["recursive"]
    chunk_token_size: { min: 256, max: 1024 }
    chunk_token_overlap: { min: 0, max: 128 }
  embedding:
    models: ["sentence-transformers/all-MiniLM-L6-v2"]
  retrieval:
    index_types: ["vector_only", "graph_only"]
    top_k: { min: 3, max: 15 }
    hybrid_alpha: { min: 0.0, max: 1.0 }
  reranker:
    models: ["none"]
    top_n: { min: 3, max: 8 }
  query_expansion:
    strategies: ["none"]
    models: []
  generator:
    models: ["ollama/llama3.2"]
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
        assert cfg.parsing.ocr is False
        assert "recursive" in cfg.search_space.chunking.strategies
        assert cfg.search_space.chunking.chunk_token_size.min == 256
        assert len(cfg.search_space.all_llm_models()) == 1
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
            generator_llm="ollama/llama3.2",
            temperature=0.3,
        )

        violations = cfg.validate_trial(trial)

        assert violations == []

    def test_loaded_search_space_catches_violation(self, tmp_path: Path) -> None:
        config_file = self._create_mock_config(tmp_path, MOCK_YAML_CONFIG)
        cfg = load_config(config_file)

        trial = TrialConfig(
            generator_llm="ollama/llama3.2",
            chunk_token_size=2048,  # Out of range (max is 1024)
            chunk_token_overlap=64,
        )

        violations = cfg.validate_trial(trial)

        assert len(violations) > 0
        assert any("chunk_token_size" in v for v in violations)


class TestReasoningSearchSpace:
    """Tests for reasoning parameter support in the search space.

    LiteLLM is ground truth — ``is_reasoning_allowed`` only consults the
    global ``reasoning`` flag, the ollama prefix gate, and
    ``litellm.supports_reasoning``. There is no per-model override.
    """

    def test_is_reasoning_allowed_global_default_true(self) -> None:
        # Use a real model where litellm.supports_reasoning returns True
        ss = _ss(
            embedding_models=["e"],
            generator_models=["anthropic/claude-haiku-4-5-20251001"],
            reasoning=True,
        )
        assert ss.is_reasoning_allowed("anthropic/claude-haiku-4-5-20251001") is True

    def test_is_reasoning_allowed_global_default_false(self) -> None:
        ss = _ss(
            embedding_models=["e"],
            generator_models=["anthropic/claude-haiku-4-5-20251001"],
            reasoning=False,
        )
        assert ss.is_reasoning_allowed("anthropic/claude-haiku-4-5-20251001") is False

    def test_is_reasoning_allowed_ollama_auto_denied(self) -> None:
        ss = _ss(
            embedding_models=["e"],
            generator_models=["ollama/llama3.2"],
            reasoning=True,  # global says yes, but ollama is auto-denied
        )
        assert ss.is_reasoning_allowed("ollama/llama3.2") is False

    def test_is_reasoning_allowed_vllm_not_auto_denied(self) -> None:
        """hosted_vllm/ models are NOT in _REASONING_UNSUPPORTED_PREFIXES."""
        from unittest.mock import patch

        ss = _ss(
            embedding_models=["e"],
            generator_models=["hosted_vllm/Qwen/Qwen3-8B"],
            reasoning=True,
        )
        with patch("litellm.supports_reasoning", return_value=True):
            assert ss.is_reasoning_allowed("hosted_vllm/Qwen/Qwen3-8B") is True

    def test_is_reasoning_allowed_litellm_unsupported_denied(self) -> None:
        """LiteLLM capability check: model marked as not supporting reasoning is denied."""
        from unittest.mock import patch

        ss = _ss(
            embedding_models=["e"],
            generator_models=["anthropic/claude-haiku-4-5-20251001"],
            reasoning=True,
        )
        with patch("litellm.supports_reasoning", return_value=False):
            assert ss.is_reasoning_allowed("anthropic/claude-haiku-4-5-20251001") is False

    def test_is_reasoning_allowed_litellm_supported(self) -> None:
        """Model confirmed by litellm.supports_reasoning is allowed when global=True."""
        from unittest.mock import patch

        ss = _ss(
            embedding_models=["e"],
            generator_models=["anthropic/claude-haiku-4-5-20251001"],
            reasoning=True,
        )
        with patch("litellm.supports_reasoning", return_value=True):
            assert ss.is_reasoning_allowed("anthropic/claude-haiku-4-5-20251001") is True

    def test_validate_trial_reasoning_allowed(self) -> None:
        from unittest.mock import patch

        cfg = ProjectConfig(
            search_space=_ss(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                generator_models=["vertex_ai/gemini-2.5-flash"],
                reasoning=True,
            ),
            agent=AgentConfig(
                optimizer_model="ollama/llama3.2",
                examiner_model="ollama/llama3.2",
                judge_model="ollama/llama3.2",
            ),
        )
        trial = TrialConfig(generator_llm="vertex_ai/gemini-2.5-flash", reasoning=True)
        with patch("litellm.supports_reasoning", return_value=True):
            violations = cfg.validate_trial(trial)
        assert not any("reasoning" in v for v in violations)

    def test_validate_trial_reasoning_denied_globally(self) -> None:
        cfg = ProjectConfig(
            search_space=_ss(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                generator_models=["cloud/model-a"],
                reasoning=False,
            ),
            agent=AgentConfig(
                optimizer_model="ollama/llama3.2",
                examiner_model="ollama/llama3.2",
                judge_model="ollama/llama3.2",
            ),
        )
        trial = TrialConfig(generator_llm="cloud/model-a", reasoning=True)
        violations = cfg.validate_trial(trial)
        assert any("reasoning" in v for v in violations)

    def test_validate_trial_reasoning_denied_for_ollama(self) -> None:
        cfg = ProjectConfig(
            search_space=_ss(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                generator_models=["ollama/llama3.2"],
                reasoning=True,
            ),
            agent=AgentConfig(
                optimizer_model="ollama/llama3.2",
                examiner_model="ollama/llama3.2",
                judge_model="ollama/llama3.2",
            ),
        )
        trial = TrialConfig(generator_llm="ollama/llama3.2", reasoning=True)
        violations = cfg.validate_trial(trial)
        assert any("reasoning" in v for v in violations)

    def test_validate_trial_reasoning_false_always_ok(self) -> None:
        cfg = ProjectConfig(
            search_space=_ss(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                generator_models=["ollama/llama3.2"],
                reasoning=False,
            ),
            agent=AgentConfig(
                optimizer_model="ollama/llama3.2",
                examiner_model="ollama/llama3.2",
                judge_model="ollama/llama3.2",
            ),
        )
        trial = TrialConfig(generator_llm="ollama/llama3.2", reasoning=False)
        violations = cfg.validate_trial(trial)
        assert not any("reasoning" in v for v in violations)


class TestValidateLlmModels:
    """Tests for ProjectConfig.validate_llm_models — static + live probe validation."""

    def _make_project_config_with_models(self, llm_models: list) -> ProjectConfig:
        return ProjectConfig.model_validate(
            {
                "agent": {
                    "optimizer_model": "ollama/llama3.2",
                    "examiner_model": "ollama/llama3.2",
                    "judge_model": "ollama/llama3.2",
                },
                "search_space": {
                    "embedding": {"models": ["sentence-transformers/all-MiniLM-L6-v2"]},
                    "generator": {"models": llm_models},
                },
            }
        )

    def test_known_provider_suffix_passes_static_check(self) -> None:
        """anthropic/claude-haiku-4-5-20251001 is in models_by_provider — no probe needed."""
        cfg = self._make_project_config_with_models(["anthropic/claude-haiku-4-5-20251001"])
        assert "anthropic/claude-haiku-4-5-20251001" in cfg.search_space.all_llm_models()

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
        assert "newprovider/some-model" in cfg.search_space.all_llm_models()

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
        trial = TrialConfig(generator_llm="test/model")
        assert trial.reasoning is False

    def test_to_runtime_passes_reasoning(self) -> None:
        trial = TrialConfig(generator_llm="test/model", reasoning=True)
        r = trial.to_runtime(reasoning_effort="high")
        assert r.reasoning is True
        assert r.reasoning_effort == "high"

    def test_to_runtime_default_effort(self) -> None:
        trial = TrialConfig(generator_llm="test/model", reasoning=True)
        r = trial.to_runtime()
        assert r.reasoning_effort == "medium"

    def test_fingerprint_unchanged_by_reasoning(self) -> None:
        trial_a = TrialConfig(generator_llm="test/model", reasoning=False)
        trial_b = TrialConfig(generator_llm="test/model", reasoning=True)
        assert trial_a.structural_fingerprint() == trial_b.structural_fingerprint()


class TestReasoningAgentPrompt:
    def test_prompt_includes_reasoning_field(self) -> None:
        cfg = ProjectConfig(
            search_space=_ss(
                embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
                generator_models=["cloud/model-a"],
            ),
            agent=AgentConfig(
                optimizer_model="ollama/llama3.2",
                examiner_model="ollama/llama3.2",
                judge_model="ollama/llama3.2",
            ),
        )
        prompt = cfg.to_agent_prompt()
        assert "reasoning" in prompt

    def test_prompt_shows_allowed_models(self) -> None:
        from unittest.mock import patch

        cfg = ProjectConfig(
            search_space=_ss(
                embedding_models=["e"],
                generator_models=[
                    "vertex_ai/gemini-2.5-flash",
                    "vertex_ai/gemini-2.5-flash-lite",
                ],
                reasoning=True,
            ),
            agent=AgentConfig(
                optimizer_model="ollama/llama3.2",
                examiner_model="ollama/llama3.2",
                judge_model="ollama/llama3.2",
            ),
        )

        # First model supports reasoning, second does not.
        def fake_supports(model: str) -> bool:
            return model == "vertex_ai/gemini-2.5-flash"

        with patch("litellm.supports_reasoning", side_effect=fake_supports):
            prompt = cfg.to_agent_prompt()
        assert "vertex_ai/gemini-2.5-flash" in prompt
        assert "allowed for" in prompt.lower() or "NOT allowed" in prompt

    def test_prompt_shows_denied_ollama(self) -> None:
        from unittest.mock import patch

        cfg = ProjectConfig(
            search_space=_ss(
                embedding_models=["e"],
                generator_models=["ollama/llama3.2", "cloud/model-a"],
                reasoning=True,
            ),
            agent=AgentConfig(
                optimizer_model="ollama/llama3.2",
                examiner_model="ollama/llama3.2",
                judge_model="ollama/llama3.2",
            ),
        )
        # cloud/model-a supports reasoning; ollama is always denied via prefix.
        # This forces the mixed-support branch of the prompt — both
        # ``allowed for`` and ``NOT allowed for`` lines render.
        with patch("litellm.supports_reasoning", return_value=True):
            prompt = cfg.to_agent_prompt()
        assert "ollama/llama3.2" in prompt
        assert "NOT allowed" in prompt

    def test_prompt_omits_reasoning_when_no_model_supports_it(self) -> None:
        cfg = ProjectConfig(
            search_space=_ss(
                embedding_models=["e"],
                generator_models=["ollama/llama3.2"],
                reasoning=True,
            ),
            agent=AgentConfig(
                optimizer_model="ollama/llama3.2",
                examiner_model="ollama/llama3.2",
                judge_model="ollama/llama3.2",
            ),
        )
        prompt = cfg.to_agent_prompt()
        # No per-model 'NOT allowed for: [...]' enumeration when nothing supports
        # reasoning.
        assert "NOT allowed for" not in prompt

    def test_prompt_yaml_block_includes_reasoning(self) -> None:
        cfg = ProjectConfig(
            search_space=_ss(
                embedding_models=["e"],
                generator_models=["cloud/model-a"],
            ),
            agent=AgentConfig(
                optimizer_model="ollama/llama3.2",
                examiner_model="ollama/llama3.2",
                judge_model="ollama/llama3.2",
            ),
        )
        prompt = cfg.to_agent_prompt()
        assert "reasoning: false" in prompt

    def test_prompt_pins_reasoning_when_generator_disables_it(self) -> None:
        """When ``generator.reasoning=False`` reasoning is pinned to ``false``
        — it appears in the Fixed-values block (so the proposer has context)
        but not in the Tunable-parameters block, not as an orphaned ``NOT
        allowed for:`` line, and not in the example YAML. The pinned-injection
        path then overrides any ``reasoning: true`` the agent attempts to emit."""
        cfg = ProjectConfig(
            search_space=_ss(
                embedding_models=["e"],
                generator_models=["cloud/model-a"],
                reasoning=False,
            ),
            agent=AgentConfig(
                optimizer_model="ollama/llama3.2",
                examiner_model="ollama/llama3.2",
                judge_model="ollama/llama3.2",
            ),
        )
        prompt = cfg.to_agent_prompt()
        # Tunable block must not list reasoning.
        tunable_block = prompt.split("### Fixed values")[0]
        assert "reasoning:" not in tunable_block
        assert "NOT allowed" not in prompt
        # Fixed-values block must list reasoning so the agent sees the lock.
        fixed_block = prompt.split("### Fixed values")[1].split("### Expected output format")[0]
        assert "reasoning: false" in fixed_block
        # Example YAML must omit reasoning entirely — it's pinned, not emitted.
        start = prompt.index("```yaml")
        end = prompt.index("```", start + 7)
        example = prompt[start:end]
        assert "reasoning:" not in example
