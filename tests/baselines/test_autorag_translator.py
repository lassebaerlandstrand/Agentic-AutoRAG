"""Tests for ``config_translator.translate_extracted_to_trial_config``."""

from __future__ import annotations

from pathlib import Path

import yaml

from agentic_autorag.baselines.autorag.config_translator import (
    translate_extracted_to_trial_config,
)
from agentic_autorag.config.models import IndexType, ProjectConfig


def _make_search_space() -> ProjectConfig:
    raw = {
        "meta": {"project_name": "t", "corpus_path": "./f", "output_dir": "./f_out"},
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
            "index_types": ["vector_only", "hybrid_bm25_vector"],
            "top_k": {"min": 3, "max": 20},
            "hybrid_alpha": {"min": 0.0, "max": 1.0},
            "reranker": {
                "models": ["none", "BAAI/bge-reranker-v2-m3"],
                "top_n": {"min": 3, "max": 10},
            },
            "query_expansion": ["none", "hyde"],
            "llm_models": ["ollama/llama3.2", "gemini/gemini-2.5-flash-lite"],
            "temperature": {"min": 0.0, "max": 1.0},
        },
        "agent": {"optimizer_model": "t/m", "examiner_model": "t/m"},
        "examiner": {"exam_size": 5},
    }
    return ProjectConfig.model_validate(raw)


def _write_extracted(tmp_path: Path, content: dict) -> Path:
    p = tmp_path / "extracted_sample.yaml"
    p.write_text(yaml.safe_dump(content), encoding="utf-8")
    return p


def test_vectordb_winner_translates_to_vector_only(tmp_path: Path) -> None:
    cfg = _make_search_space()
    extracted = _write_extracted(
        tmp_path,
        {
            "node_lines": [
                {
                    "nodes": [
                        {
                            "node_type": "chunker",
                            "modules": [
                                {
                                    "module_type": "llama_index_chunk",
                                    "chunk_method": "token",
                                    "chunk_size": 512,
                                    "chunk_overlap": 64,
                                }
                            ],
                        },
                        {
                            "node_type": "retrieval",
                            "modules": [
                                {
                                    "module_type": "vectordb",
                                    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                                    "top_k": 7,
                                }
                            ],
                        },
                        {
                            "node_type": "generator",
                            "modules": [
                                {
                                    "module_type": "llama_index_llm",
                                    "llm": "gemini/gemini-2.5-flash-lite",
                                    "temperature": 0.3,
                                }
                            ],
                        },
                    ]
                }
            ]
        },
    )
    config = translate_extracted_to_trial_config(extracted, cfg.search_space)
    assert config.index_type == IndexType.VECTOR_ONLY
    assert config.embedding_model == "sentence-transformers/all-mpnet-base-v2"
    assert config.top_k == 7
    assert config.chunk_token_size == 512
    assert config.chunk_token_overlap == 64
    assert config.llm_model == "gemini/gemini-2.5-flash-lite"
    assert config.temperature == 0.3
    # Final config is valid in our space.
    assert cfg.validate_trial(config) == []


def test_hybrid_cc_winner_translates_to_hybrid_with_complement(tmp_path: Path) -> None:
    """AutoRAG's BM25 weight = 0.3 → our vector-side hybrid_alpha = 0.7."""
    cfg = _make_search_space()
    extracted = _write_extracted(
        tmp_path,
        {
            "node_lines": [
                {
                    "nodes": [
                        {
                            "node_type": "retrieval",
                            "modules": [
                                {
                                    "module_type": "hybrid_cc",
                                    "weight": 0.3,
                                    "top_k": 10,
                                }
                            ],
                        },
                        {
                            "node_type": "generator",
                            "modules": [{"module_type": "llama_index_llm", "llm": "ollama/llama3.2"}],
                        },
                    ]
                }
            ]
        },
    )
    config = translate_extracted_to_trial_config(extracted, cfg.search_space)
    assert config.index_type == IndexType.HYBRID_BM25_VECTOR
    assert config.hybrid_alpha == 0.7
    assert config.top_k == 10


def test_passage_reranker_pass_through_means_none(tmp_path: Path) -> None:
    cfg = _make_search_space()
    extracted = _write_extracted(
        tmp_path,
        {
            "node_lines": [
                {
                    "nodes": [
                        {
                            "node_type": "passage_reranker",
                            "modules": [{"module_type": "pass_passage_reranker"}],
                        },
                        {
                            "node_type": "generator",
                            "modules": [{"module_type": "llama_index_llm", "llm": "ollama/llama3.2"}],
                        },
                    ]
                }
            ]
        },
    )
    config = translate_extracted_to_trial_config(extracted, cfg.search_space)
    assert config.reranker == "none"


def test_named_reranker_translates_by_model_name(tmp_path: Path) -> None:
    cfg = _make_search_space()
    extracted = _write_extracted(
        tmp_path,
        {
            "node_lines": [
                {
                    "nodes": [
                        {
                            "node_type": "passage_reranker",
                            "modules": [
                                {
                                    "module_type": "flag_embedding_reranker",
                                    "model_name": "BAAI/bge-reranker-v2-m3",
                                    "top_k": 5,
                                }
                            ],
                        },
                        {
                            "node_type": "retrieval",
                            "modules": [
                                {
                                    "module_type": "vectordb",
                                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                                    "top_k": 10,
                                }
                            ],
                        },
                        {
                            "node_type": "generator",
                            "modules": [{"module_type": "llama_index_llm", "llm": "ollama/llama3.2"}],
                        },
                    ]
                }
            ]
        },
    )
    config = translate_extracted_to_trial_config(extracted, cfg.search_space)
    assert config.reranker == "BAAI/bge-reranker-v2-m3"
    assert config.reranker_top_n == 5
    assert config.reranker_top_n <= config.top_k


def test_query_expansion_hyde_translates(tmp_path: Path) -> None:
    cfg = _make_search_space()
    extracted = _write_extracted(
        tmp_path,
        {
            "node_lines": [
                {
                    "nodes": [
                        {
                            "node_type": "query_expansion",
                            "modules": [{"module_type": "hyde"}],
                        },
                        {
                            "node_type": "generator",
                            "modules": [{"module_type": "llama_index_llm", "llm": "ollama/llama3.2"}],
                        },
                    ]
                }
            ]
        },
    )
    config = translate_extracted_to_trial_config(extracted, cfg.search_space)
    assert config.query_expansion == "hyde"


def test_empty_extracted_falls_back_to_search_space_minimums(tmp_path: Path) -> None:
    cfg = _make_search_space()
    extracted = _write_extracted(tmp_path, {"node_lines": []})
    config = translate_extracted_to_trial_config(extracted, cfg.search_space)
    # Falls back to the lower bound / first-listed defaults.
    assert config.chunk_token_size == 128
    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.index_type == IndexType.VECTOR_ONLY
    assert config.reranker == "none"
    assert config.query_expansion == "none"
    assert cfg.validate_trial(config) == []


def test_overlap_clamped_below_chunk_size(tmp_path: Path) -> None:
    """If AutoRAG produces overlap >= chunk_size (impossible if our config is mirrored),
    translator must clamp to satisfy TrialConfig invariants."""
    cfg = _make_search_space()
    extracted = _write_extracted(
        tmp_path,
        {
            "node_lines": [
                {
                    "nodes": [
                        {
                            "node_type": "chunker",
                            "modules": [
                                {
                                    "module_type": "llama_index_chunk",
                                    "chunk_method": "token",
                                    "chunk_size": 200,
                                    "chunk_overlap": 200,
                                }
                            ],
                        },
                        {
                            "node_type": "generator",
                            "modules": [{"module_type": "llama_index_llm", "llm": "ollama/llama3.2"}],
                        },
                    ]
                }
            ]
        },
    )
    config = translate_extracted_to_trial_config(extracted, cfg.search_space)
    assert config.chunk_token_overlap < config.chunk_token_size
