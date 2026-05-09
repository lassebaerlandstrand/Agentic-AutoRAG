"""Conditional samplers for ``TrialConfig`` over a ``SearchSpace``.

Two entry points:

- ``sample_trial_config_random(rng, search_space, embedding_token_limits)`` — pure
  ``random.Random``-driven sampling for the Random-search baseline.
- ``sample_trial_config_optuna(trial, search_space, embedding_token_limits)`` —
  define-by-run sampling for the Optuna TPE baseline. Uses ``trial.suggest_*`` so
  the sampler observes only the dimensions active for the currently-sampled
  ``index_type`` / ``reranker`` choice (no irrelevant dimensions clutter TPE's
  surrogate model).

Both helpers honour the same conditional gates that the agent's proposer
must respect, and the result is guaranteed by construction to be inside the
``SearchSpace`` (callers should still call ``ProjectConfig.validate_trial`` as a
final guard — embedding-token-limit edge cases at the boundary can otherwise
slip through).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from agentic_autorag.config.models import (
    GRAPH_INDEX_TYPES,
    IndexType,
    NumericRange,
    SearchSpace,
    TrialConfig,
)

if TYPE_CHECKING:
    import optuna


def _midpoint(r: NumericRange) -> float:
    return (r.min + r.max) / 2.0


def _filter_compatible_embeddings(
    embedding_models: list[str],
    chunk_token_size: int,
    embedding_token_limits: dict[str, int],
) -> list[str]:
    """Return embeddings whose token cap accepts ``chunk_token_size``.

    Models without a recorded limit are kept (we only filter what we know).
    """
    compatible = []
    for model in embedding_models:
        limit = embedding_token_limits.get(model)
        if limit is None or chunk_token_size <= limit:
            compatible.append(model)
    return compatible


def sample_trial_config_random(
    rng: random.Random,
    search_space: SearchSpace,
    embedding_token_limits: dict[str, int] | None = None,
) -> TrialConfig:
    """Uniformly sample a valid ``TrialConfig`` from the search space."""
    embedding_token_limits = embedding_token_limits or {}
    ss = search_space

    chunking_strategy = rng.choice(ss.chunking.strategies)

    cs_lo = int(ss.chunking.chunk_token_size.min)
    cs_hi = int(ss.chunking.chunk_token_size.max)
    chunk_token_size = rng.randint(cs_lo, cs_hi)

    co_lo = int(ss.chunking.chunk_token_overlap.min)
    co_hi_user = int(ss.chunking.chunk_token_overlap.max)
    # overlap < chunk_token_size (TrialConfig validator)
    co_hi = max(co_lo, min(co_hi_user, chunk_token_size - 1))
    chunk_token_overlap = rng.randint(co_lo, co_hi)

    compatible = _filter_compatible_embeddings(ss.embedding_models, chunk_token_size, embedding_token_limits)
    if not compatible:
        # The currently-sampled chunk_token_size doesn't fit any embedding's cap.
        # Shrink it to the largest cap so we can still produce a valid config.
        max_supported = max(embedding_token_limits.values()) if embedding_token_limits else cs_hi
        chunk_token_size = max(cs_lo, min(chunk_token_size, max_supported))
        chunk_token_overlap = min(chunk_token_overlap, max(co_lo, chunk_token_size - 1))
        compatible = _filter_compatible_embeddings(
            ss.embedding_models, chunk_token_size, embedding_token_limits
        ) or list(ss.embedding_models)
    embedding_model = rng.choice(compatible)

    index_type = rng.choice(list(ss.index_types))

    top_k = rng.randint(int(ss.top_k.min), int(ss.top_k.max))

    if index_type == IndexType.HYBRID_BM25_VECTOR:
        hybrid_alpha = round(rng.uniform(ss.hybrid_alpha.min, ss.hybrid_alpha.max), 4)
    else:
        hybrid_alpha = round(_midpoint(ss.hybrid_alpha), 4)

    reranker = rng.choice(ss.reranker.models)
    if reranker != "none":
        rn_lo = int(ss.reranker.top_n.min)
        rn_hi = max(rn_lo, min(int(ss.reranker.top_n.max), top_k))
        reranker_top_n = rng.randint(rn_lo, rn_hi)
    else:
        # Range still validated; pick the lower bound (any valid value).
        reranker_top_n = int(ss.reranker.top_n.min)

    query_expansion = rng.choice(ss.query_expansion)

    llm_model = rng.choice(ss.llm_models)
    temperature = round(rng.uniform(ss.temperature.min, ss.temperature.max), 4)

    # 50/50 when allowed — the agent makes a richer choice; uniform is the right baseline.
    reasoning = rng.choice([False, True]) if ss.is_reasoning_allowed(llm_model) else False

    if index_type in GRAPH_INDEX_TYPES and ss.graph_retrieval is not None:
        gr = ss.graph_retrieval
        graph_query_mode = rng.choice(gr.graph_query_modes)
        graph_top_k = rng.randint(int(gr.graph_top_k.min), int(gr.graph_top_k.max))
    else:
        graph_query_mode = "hybrid"
        graph_top_k = 60

    return TrialConfig(
        chunking_strategy=chunking_strategy,
        chunk_token_size=chunk_token_size,
        chunk_token_overlap=chunk_token_overlap,
        embedding_model=embedding_model,
        index_type=index_type,
        top_k=top_k,
        hybrid_alpha=hybrid_alpha,
        reranker=reranker,
        reranker_top_n=reranker_top_n,
        query_expansion=query_expansion,
        llm_model=llm_model,
        temperature=temperature,
        reasoning=reasoning,
        graph_query_mode=graph_query_mode,
        graph_top_k=graph_top_k,
    )


def sample_trial_config_optuna(
    trial: optuna.Trial,
    search_space: SearchSpace,
    embedding_token_limits: dict[str, int] | None = None,
) -> TrialConfig:
    """Sample a ``TrialConfig`` via Optuna's define-by-run API.

    Conditionally calls ``trial.suggest_*`` only for active dimensions, so TPE
    doesn't observe ``hybrid_alpha`` for non-hybrid trials, ``reranker_top_n``
    when reranker is disabled, etc.

    May raise ``optuna.TrialPruned`` if no embedding model is compatible with
    the sampled ``chunk_token_size``.
    """
    import optuna

    embedding_token_limits = embedding_token_limits or {}
    ss = search_space

    chunking_strategy = trial.suggest_categorical("chunking_strategy", ss.chunking.strategies)

    cs_lo = int(ss.chunking.chunk_token_size.min)
    cs_hi = int(ss.chunking.chunk_token_size.max)
    chunk_token_size = trial.suggest_int("chunk_token_size", cs_lo, cs_hi)

    co_lo = int(ss.chunking.chunk_token_overlap.min)
    co_hi_user = int(ss.chunking.chunk_token_overlap.max)
    co_hi = max(co_lo, min(co_hi_user, chunk_token_size - 1))
    chunk_token_overlap = trial.suggest_int("chunk_token_overlap", co_lo, co_hi)

    compatible = _filter_compatible_embeddings(ss.embedding_models, chunk_token_size, embedding_token_limits)
    if not compatible:
        raise optuna.TrialPruned(f"No embedding model in search space supports chunk_token_size={chunk_token_size}")
    embedding_model = trial.suggest_categorical("embedding_model", compatible)

    index_type_str = trial.suggest_categorical("index_type", [it.value for it in ss.index_types])
    index_type = IndexType(index_type_str)

    top_k = trial.suggest_int("top_k", int(ss.top_k.min), int(ss.top_k.max))

    if index_type == IndexType.HYBRID_BM25_VECTOR:
        hybrid_alpha = trial.suggest_float("hybrid_alpha", ss.hybrid_alpha.min, ss.hybrid_alpha.max)
    else:
        hybrid_alpha = _midpoint(ss.hybrid_alpha)

    reranker = trial.suggest_categorical("reranker", ss.reranker.models)
    if reranker != "none":
        rn_lo = int(ss.reranker.top_n.min)
        rn_hi = max(rn_lo, min(int(ss.reranker.top_n.max), top_k))
        reranker_top_n = trial.suggest_int("reranker_top_n", rn_lo, rn_hi)
    else:
        reranker_top_n = int(ss.reranker.top_n.min)

    query_expansion = trial.suggest_categorical("query_expansion", ss.query_expansion)

    llm_model = trial.suggest_categorical("llm_model", ss.llm_models)
    temperature = trial.suggest_float("temperature", ss.temperature.min, ss.temperature.max)

    reasoning = trial.suggest_categorical("reasoning", [False, True]) if ss.is_reasoning_allowed(llm_model) else False

    if index_type in GRAPH_INDEX_TYPES and ss.graph_retrieval is not None:
        gr = ss.graph_retrieval
        graph_query_mode = trial.suggest_categorical("graph_query_mode", gr.graph_query_modes)
        graph_top_k = trial.suggest_int("graph_top_k", int(gr.graph_top_k.min), int(gr.graph_top_k.max))
    else:
        graph_query_mode = "hybrid"
        graph_top_k = 60

    return TrialConfig(
        chunking_strategy=chunking_strategy,
        chunk_token_size=chunk_token_size,
        chunk_token_overlap=chunk_token_overlap,
        embedding_model=embedding_model,
        index_type=index_type,
        top_k=top_k,
        hybrid_alpha=hybrid_alpha,
        reranker=reranker,
        reranker_top_n=reranker_top_n,
        query_expansion=query_expansion,
        llm_model=llm_model,
        temperature=temperature,
        reasoning=reasoning,
        graph_query_mode=graph_query_mode,
        graph_top_k=graph_top_k,
    )
