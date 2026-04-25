"""Generate an AutoRAG ``config.yaml`` strictly mirroring our ``SearchSpace``.

AutoRAG searches a richer native space (``passage_compressor``, ``prompt_maker``
template tuning, LongLLMLingua, additional ``query_expansion`` modules), but
exposing those would give AutoRAG search dimensions Random / Bayesian / our
agent don't have. This generator deliberately includes **only** dimensions
present in our ``SearchSpace`` and freezes everything else at sensible
defaults. The mirror is one-way, so the post-hoc translator
(``config_translator.py``) sees only fields it knows how to map.

Discretization: AutoRAG's chunker / generator / retriever modules accept
list-of-values for some parameters where we use ranges. We discretize each
range into a small fixed grid (5 values for chunk_size and top_k, 3 for
overlap and temperature) so AutoRAG can enumerate them in its greedy traversal.
"""

from __future__ import annotations

from agentic_autorag.config.models import IndexType, NumericRange, SearchSpace

# Fixed MCQ-aware prompt for the AutoRAG-MCQ variant. AutoRAG's fstring
# prompt_maker only supports ``{query}`` and ``{retrieved_contents}`` — the
# four options are baked into ``{query}`` itself by ``qa_mcq.py``. The
# registered ``mcq_accuracy`` metric scores the generation via normalized
# substring match against the gold option text.
MCQ_PROMPT_TEMPLATE = (
    "Answer the following multiple-choice question by giving the text of the correct option.\n"
    "\n"
    "Context:\n"
    "{retrieved_contents}\n"
    "\n"
    "{query}\n"
    "\n"
    "Answer with only the text of the correct option, nothing else."
)

# Free-form QA prompt for the AutoRAG-RAGAS variant.
FREE_FORM_PROMPT_TEMPLATE = (
    "Use the following context to answer the question.\n"
    "\n"
    "Context:\n"
    "{retrieved_contents}\n"
    "\n"
    "Question: {query}\n"
    "Answer with only the answer itself: no explanation, no quotes."
)


def _discretize_int(r: NumericRange, n: int) -> list[int]:
    """Return ``n`` evenly-spaced ints across ``[r.min, r.max]`` (deduped, sorted)."""
    lo, hi = int(r.min), int(r.max)
    if lo == hi:
        return [lo]
    if n <= 1:
        return [lo, hi]
    step = (hi - lo) / (n - 1)
    values = sorted({int(round(lo + i * step)) for i in range(n)})
    return values


def _discretize_float(r: NumericRange, n: int, *, precision: int = 2) -> list[float]:
    lo, hi = float(r.min), float(r.max)
    if lo == hi:
        return [round(lo, precision)]
    if n <= 1:
        return [round(lo, precision), round(hi, precision)]
    step = (hi - lo) / (n - 1)
    values = sorted({round(lo + i * step, precision) for i in range(n)})
    return values


def _guess_reranker_module(model_name: str) -> str:
    """Best-effort mapping from HF model name → AutoRAG reranker module_type."""
    name = model_name.lower()
    if "bge-reranker" in name or "flag" in name:
        return "flag_embedding_reranker"
    if "monot5" in name or "mt5" in name:
        return "monot5"
    if "colbert" in name:
        return "colbert_reranker"
    if "rankgpt" in name:
        return "rankgpt"
    if "jina" in name:
        return "jina_reranker"
    # sentence_transformer is the generic CrossEncoder wrapper — broadest fit
    return "sentence_transformer_reranker"


def generate_autorag_config(
    search_space: SearchSpace,
    *,
    qa_variant: str = "mcq",
) -> tuple[dict, dict]:
    """Build an AutoRAG ``config.yaml`` dict + a ``translation_notes.json`` dict.

    Parameters
    ----------
    search_space:
        The project's ``SearchSpace``. Every dimension of it appears as an
        AutoRAG node/module; nothing else does.
    qa_variant:
        ``"mcq"`` registers ``mcq_accuracy`` as the generation metric and uses
        ``MCQ_PROMPT_TEMPLATE``. ``"ragas"`` uses ``g_eval`` (LLM-as-judge) and
        the free-form prompt template.

    Returns
    -------
    (config, notes) — ``config`` is the AutoRAG yaml dict; ``notes`` records the
    deliberate exclusions for the paper's appendix.
    """
    if qa_variant not in ("mcq", "ragas"):
        raise ValueError(f"qa_variant must be 'mcq' or 'ragas', got {qa_variant!r}")
    ss = search_space

    chunk_sizes = _discretize_int(ss.chunking.chunk_token_size, n=5)
    chunk_overlaps = _discretize_int(ss.chunking.chunk_token_overlap, n=3)
    top_ks = _discretize_int(ss.top_k, n=5)
    temperatures = _discretize_float(ss.temperature, n=3)
    reranker_top_ks = _discretize_int(ss.reranker.top_n, n=3)

    # Chunker modules — one per strategy
    chunker_modules: list[dict] = []
    for strategy in ss.chunking.strategies:
        chunker_modules.append(
            {
                "module_type": "llama_index_chunk",
                "chunk_method": ["token" if strategy == "recursive" else strategy],
                "chunk_size": chunk_sizes,
                "chunk_overlap": chunk_overlaps,
            }
        )

    # Retrieval modules — gated by index_types
    retrieval_modules: list[dict] = []
    if IndexType.VECTOR_ONLY in ss.index_types:
        retrieval_modules.append(
            {
                "module_type": "vectordb",
                "embedding_model": list(ss.embedding_models),
            }
        )
    if IndexType.HYBRID_BM25_VECTOR in ss.index_types:
        # AutoRAG's hybrid_cc weight is BM25's; ours is vector's. Convert range.
        ha_lo = round(1.0 - ss.hybrid_alpha.max, 4)
        ha_hi = round(1.0 - ss.hybrid_alpha.min, 4)
        retrieval_modules.append(
            {
                "module_type": "hybrid_cc",
                "weight_range": [ha_lo, ha_hi],
                "normalize_method": ["mm", "tmm"],
            }
        )

    # Reranker modules
    reranker_modules: list[dict] = []
    for model in ss.reranker.models:
        if model == "none":
            reranker_modules.append({"module_type": "pass_passage_reranker"})
        else:
            reranker_modules.append(
                {
                    "module_type": _guess_reranker_module(model),
                    "model_name": model,
                }
            )

    # Query expansion modules
    query_expansion_modules: list[dict] = []
    for qe in ss.query_expansion:
        if qe == "none":
            query_expansion_modules.append({"module_type": "pass_query_expansion"})
        elif qe == "hyde":
            query_expansion_modules.append({"module_type": "hyde"})
        elif qe in {"multi_query", "multi_query_expansion"}:
            query_expansion_modules.append({"module_type": "multi_query_expansion"})
        else:
            query_expansion_modules.append({"module_type": qe})

    # Generator modules — one per LLM
    generator_modules: list[dict] = []
    for llm in ss.llm_models:
        generator_modules.append(
            {
                "module_type": "llama_index_llm",
                "llm": llm,
                "temperature": temperatures,
            }
        )

    # Strategy metrics
    if qa_variant == "mcq":
        gen_metrics = ["mcq_accuracy"]
        prompt_template = MCQ_PROMPT_TEMPLATE
    else:
        gen_metrics = ["g_eval"]
        prompt_template = FREE_FORM_PROMPT_TEMPLATE

    config = {
        "node_lines": [
            {
                "node_line_name": "pre_retrieve_node_line",
                "nodes": [
                    {
                        "node_type": "query_expansion",
                        "modules": query_expansion_modules,
                        "strategy": {
                            "metrics": ["retrieval_f1"],
                            "top_k": top_ks,
                        },
                    }
                ],
            },
            {
                "node_line_name": "retrieve_node_line",
                "nodes": [
                    {
                        "node_type": "retrieval",
                        "modules": retrieval_modules,
                        "strategy": {
                            "metrics": ["retrieval_f1", "retrieval_recall"],
                            "top_k": top_ks,
                        },
                    }
                ],
            },
            {
                "node_line_name": "post_retrieve_node_line",
                "nodes": [
                    {
                        "node_type": "passage_reranker",
                        "modules": reranker_modules,
                        "strategy": {
                            "metrics": ["retrieval_f1"],
                            "top_k": reranker_top_ks,
                        },
                    },
                    {
                        "node_type": "prompt_maker",
                        "modules": [
                            {
                                "module_type": "fstring",
                                "prompt": [prompt_template],
                            }
                        ],
                        "strategy": {"metrics": gen_metrics},
                    },
                    {
                        "node_type": "generator",
                        "modules": generator_modules,
                        "strategy": {"metrics": gen_metrics},
                    },
                ],
            },
        ],
    }

    excluded = [
        "passage_compressor (tree_summarize / refine / longllmlingua) — not in our search space",
        "prompt_maker template tuning beyond the single fstring — fixed in our pipeline",
        "query_expansion modules outside our list (e.g. query_decompose) — not in our search space",
    ]
    if qa_variant == "mcq":
        excluded.append("g_eval / sem_score / bleu / rouge generator metrics — replaced by registered mcq_accuracy")

    notes = {
        "qa_variant": qa_variant,
        "excluded_dimensions": excluded,
        "discretization": {
            "chunk_size": chunk_sizes,
            "chunk_overlap": chunk_overlaps,
            "top_k": top_ks,
            "reranker_top_k": reranker_top_ks,
            "temperature": temperatures,
        },
    }
    return config, notes
