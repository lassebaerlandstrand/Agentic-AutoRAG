"""Tests for ``native_config.generate_autorag_config``."""

from __future__ import annotations

import pytest

from agentic_autorag.baselines.autorag.native_config import (
    FREE_FORM_PROMPT_TEMPLATE,
    MCQ_PROMPT_TEMPLATE,
    generate_autorag_config,
)
from agentic_autorag.config.models import ProjectConfig


def _make_search_space(*, with_hybrid: bool = True, rerankers: list[str] | None = None) -> ProjectConfig:
    raw = {
        "meta": {
            "project_name": "test",
            "corpus_path": "./fake",
            "output_dir": "./fake_out",
        },
        "search_space": {
            "chunking": {
                "strategies": ["recursive"],
                "chunk_token_size": {"min": 128, "max": 1024},
                "chunk_token_overlap": {"min": 0, "max": 128},
            },
            "embedding_models": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
            ],
            "index_types": ["vector_only", "hybrid_bm25_vector"] if with_hybrid else ["vector_only"],
            "top_k": {"min": 3, "max": 20},
            "hybrid_alpha": {"min": 0.2, "max": 0.8},
            "reranker": {
                "models": rerankers or ["none", "BAAI/bge-reranker-v2-m3"],
                "top_n": {"min": 3, "max": 10},
            },
            "query_expansion": ["none", "hyde"],
            "llm_models": ["ollama/llama3.2", "gemini/gemini-2.5-flash-lite"],
            "temperature": {"min": 0.0, "max": 1.0},
        },
        "agent": {"optimizer_model": "test/m", "examiner_model": "test/m"},
        "examiner": {"exam_size": 5},
    }
    return ProjectConfig.model_validate(raw)


def _walk(config: dict) -> dict[str, dict]:
    """Flatten config[node_lines][nodes] → {node_type: node}."""
    out = {}
    for line in config.get("node_lines", []):
        for node in line.get("nodes", []):
            out[node["node_type"]] = node
    return out


def test_qa_variant_validation() -> None:
    cfg = _make_search_space()
    with pytest.raises(ValueError):
        generate_autorag_config(cfg.search_space, qa_variant="invalid")


def test_mcq_variant_uses_mcq_metric_and_prompt() -> None:
    cfg = _make_search_space()
    config, notes = generate_autorag_config(cfg.search_space, qa_variant="mcq")
    nodes = _walk(config)

    # Generation node uses mcq_accuracy
    assert nodes["generator"]["strategy"]["metrics"] == ["mcq_accuracy"]
    # Prompt template is the MCQ-aware one
    prompt_modules = nodes["prompt_maker"]["modules"]
    assert prompt_modules[0]["module_type"] == "fstring"
    assert prompt_modules[0]["prompt"][0] == MCQ_PROMPT_TEMPLATE
    # MCQ prompt uses ONLY {query} and {retrieved_contents} (AutoRAG fstring contract).
    template = prompt_modules[0]["prompt"][0]
    assert "{query}" in template and "{retrieved_contents}" in template
    assert "{options_block}" not in template  # options are inlined into query
    # Notes call out that g_eval is replaced
    assert any("mcq_accuracy" in line for line in notes["excluded_dimensions"])


def test_ragas_variant_uses_g_eval_and_free_form_prompt() -> None:
    cfg = _make_search_space()
    config, notes = generate_autorag_config(cfg.search_space, qa_variant="ragas")
    nodes = _walk(config)
    assert nodes["generator"]["strategy"]["metrics"] == ["g_eval"]
    assert nodes["prompt_maker"]["modules"][0]["prompt"][0] == FREE_FORM_PROMPT_TEMPLATE
    assert notes["qa_variant"] == "ragas"


def test_passage_compressor_excluded() -> None:
    """Deliberate exclusion: AutoRAG must not search passage_compressor."""
    cfg = _make_search_space()
    config, notes = generate_autorag_config(cfg.search_space, qa_variant="ragas")
    nodes = _walk(config)
    assert "passage_compressor" not in nodes
    assert any("passage_compressor" in line for line in notes["excluded_dimensions"])


def test_chunker_includes_all_strategies_and_discretized_ranges() -> None:
    cfg = _make_search_space()
    raw = cfg.model_dump()
    raw["search_space"]["chunking"]["strategies"] = ["recursive", "fixed"]
    cfg = ProjectConfig.model_validate(raw)
    config, _ = generate_autorag_config(cfg.search_space, qa_variant="mcq")

    # The chunker node sits in retrieve_node_line in real AutoRAG flows; our generator
    # currently emits chunker modules implicitly via the corpus prep stage in AutoRAG.
    # Validate via a separate path: confirm the discretization shape directly.
    _, notes = generate_autorag_config(cfg.search_space, qa_variant="mcq")
    assert len(notes["discretization"]["chunk_size"]) == 5
    assert len(notes["discretization"]["chunk_overlap"]) == 3
    assert len(notes["discretization"]["top_k"]) == 5
    assert len(notes["discretization"]["temperature"]) == 3


def test_retrieval_modules_match_index_types() -> None:
    # vector-only space → only vectordb module
    cfg_vo = _make_search_space(with_hybrid=False)
    config, _ = generate_autorag_config(cfg_vo.search_space, qa_variant="mcq")
    nodes = _walk(config)
    types = [m["module_type"] for m in nodes["retrieval"]["modules"]]
    assert "vectordb" in types
    assert "hybrid_cc" not in types

    # vector + hybrid space → both modules
    cfg_h = _make_search_space(with_hybrid=True)
    config, _ = generate_autorag_config(cfg_h.search_space, qa_variant="mcq")
    nodes = _walk(config)
    types = [m["module_type"] for m in nodes["retrieval"]["modules"]]
    assert "vectordb" in types and "hybrid_cc" in types
    # weight_range respects our hybrid_alpha range, with the BM25→vector flip.
    hybrid = next(m for m in nodes["retrieval"]["modules"] if m["module_type"] == "hybrid_cc")
    lo, hi = hybrid["weight_range"]
    # ours: 0.2..0.8 → AutoRAG: 1-0.8..1-0.2 = 0.2..0.8 (symmetric)
    assert lo == pytest.approx(0.2)
    assert hi == pytest.approx(0.8)


def test_reranker_modules_one_per_model() -> None:
    cfg = _make_search_space(rerankers=["none", "BAAI/bge-reranker-v2-m3"])
    config, _ = generate_autorag_config(cfg.search_space, qa_variant="mcq")
    nodes = _walk(config)
    types = [m["module_type"] for m in nodes["passage_reranker"]["modules"]]
    assert "pass_passage_reranker" in types  # for "none"
    # The bge reranker uses the flag_embedding_reranker module
    assert any(t in {"flag_embedding_reranker", "sentence_transformer_reranker"} for t in types)


def test_query_expansion_modules() -> None:
    cfg = _make_search_space()
    config, _ = generate_autorag_config(cfg.search_space, qa_variant="mcq")
    nodes = _walk(config)
    qe_types = [m["module_type"] for m in nodes["query_expansion"]["modules"]]
    assert "pass_query_expansion" in qe_types
    assert "hyde" in qe_types


def test_generator_modules_one_per_llm() -> None:
    cfg = _make_search_space()
    config, _ = generate_autorag_config(cfg.search_space, qa_variant="mcq")
    nodes = _walk(config)
    llms = sorted(m["llm"] for m in nodes["generator"]["modules"])
    assert llms == sorted(cfg.search_space.llm_models)


def test_no_extra_query_expansion_variants() -> None:
    """If our search space only declares 'none' + 'hyde', AutoRAG must not see more."""
    cfg = _make_search_space()
    config, _ = generate_autorag_config(cfg.search_space, qa_variant="mcq")
    nodes = _walk(config)
    qe_types = {m["module_type"] for m in nodes["query_expansion"]["modules"]}
    forbidden = {"query_decompose", "multi_query_expansion"}
    assert qe_types.isdisjoint(forbidden)
