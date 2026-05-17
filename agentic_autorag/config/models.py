"""Pydantic models for Agentic AutoRAG configuration and data structures.

Concrete config models represent what the agent proposes (specific values).
Search space models represent what the YAML defines (ranges and option lists).
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

import litellm
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Allowed difficulty tags assigned by the two-gate validator.
DIFFICULTY_TAGS = ("easy", "medium")


class IndexType(StrEnum):
    VECTOR_ONLY = "vector_only"
    HYBRID_BM25_VECTOR = "hybrid_bm25_vector"
    GRAPH_ONLY = "graph_only"
    HYBRID_GRAPH_VECTOR = "hybrid_graph_vector"


GRAPH_INDEX_TYPES: frozenset[IndexType] = frozenset({IndexType.GRAPH_ONLY, IndexType.HYBRID_GRAPH_VECTOR})
_GRAPH_TRIAL_FIELDS = frozenset({"graph_query_mode", "graph_top_k"})


_QUERY_EXPANSION_MODE_DESCRIPTIONS: dict[str, str] = {
    "none": "pass through",
    "hyde": "hypothetical answer prepended",
    "multi_query": "3 rephrasings prepended",
    "query_decompose": (
        "N self-contained sub-queries REPLACE the original — useful for explicit multi-hop questions"
    ),
}

_PASSAGE_COMPRESSOR_MODE_DESCRIPTIONS: dict[str, str] = {
    "none": "verbatim",
    "tree_summarize": "recursive synthesis over batches of 16 passages",
    "refine": (
        "iterative running-answer threaded through each passage."
        " Compression collapses N passages to 1; costs extra LLM calls"
    ),
}


def _filter_mode_descriptions(values: list[str], descriptions: dict[str, str]) -> str:
    """Return '; '-joined "{value}={description}" pairs for values present in `values`."""
    parts = [f"{v}={descriptions[v]}" for v in values if v in descriptions]
    return "; ".join(parts)


def _validate_overlap_less_than_size(v: int, info) -> int:
    if "chunk_token_size" in info.data and v >= info.data["chunk_token_size"]:
        raise ValueError("chunk_token_overlap must be < chunk_token_size")
    return v


class NumericRange(BaseModel):
    """A min/max range for numeric parameters. The agent picks any value within."""

    model_config = ConfigDict(extra="forbid")

    min: float
    max: float

    @field_validator("max")
    @classmethod
    def max_gte_min(cls, v: float, info) -> float:
        if "min" in info.data and v < info.data["min"]:
            raise ValueError("max must be >= min")
        return v

    def contains(self, value: float) -> bool:
        return self.min <= value <= self.max


class DiscreteValues(BaseModel):
    """An explicit allowed-values set for a numeric parameter.

    Used in place of ``NumericRange`` when the search dimension must be a
    finite grid (e.g. for fair AutoRAG comparison — AutoRAG enumerates lists
    via ``itertools.product``, not continuous ranges).
    """

    model_config = ConfigDict(extra="forbid")

    values: list[float | int]

    @field_validator("values")
    @classmethod
    def non_empty_unique(cls, v: list[float | int]) -> list[float | int]:
        if not v:
            raise ValueError("DiscreteValues.values must be non-empty")
        if len(set(v)) != len(v):
            raise ValueError("DiscreteValues.values must be unique")
        return sorted(v)

    def contains(self, value: float) -> bool:
        return value in self.values


# Union type for numeric search dimensions. Pydantic resolves the YAML by
# tagged shape: ``{min, max}`` -> NumericRange, ``{values: [...]}`` ->
# DiscreteValues. ``extra="forbid"`` on both models keeps the union
# unambiguous.
NumericDim = NumericRange | DiscreteValues


def _dim_min_value(dim: NumericDim) -> int | float:
    """Return the lowest legal value for a numeric dim.

    Used to pin dead knobs (e.g. ``reranker_top_n`` when no real reranker
    is reachable) without scattering ``isinstance`` checks at the call sites.
    """
    return dim.values[0] if isinstance(dim, DiscreteValues) else dim.min


def _dim_max_value(dim: NumericDim) -> int | float:
    """Return the highest legal value for a numeric dim."""
    return dim.values[-1] if isinstance(dim, DiscreteValues) else dim.max


def _dim_is_fixed(dim: NumericDim) -> bool:
    """Whether the dim has exactly one legal value."""
    if isinstance(dim, DiscreteValues):
        return len(dim.values) == 1
    return dim.min == dim.max


def _describe_dim(dim: NumericDim) -> str:
    """Compact human description of a numeric dim, e.g. for violation messages."""
    if isinstance(dim, DiscreteValues):
        return f"one of {dim.values}"
    return f"[{dim.min}, {dim.max}]"


def _dim_midpoint(dim: NumericDim) -> float:
    """Median element (DiscreteValues) or arithmetic midpoint (NumericRange).

    Used by samplers and the AutoRAG translator to pick a "default" value
    when a dim is logically inactive for the current trial (e.g.
    ``hybrid_alpha`` under RRF fusion).
    """
    if isinstance(dim, DiscreteValues):
        return float(dim.values[len(dim.values) // 2])
    return (dim.min + dim.max) / 2.0


class StageLLMs(BaseModel):
    """Per-stage LLM option sets. Each stage picks one model per trial.

    Splitting LLMs by pipeline stage (generator / expander / compressor)
    lets the search space carry e.g. 10 generator LLMs (paper claim) while
    keeping the utility stages at 2 cheap LLMs each — the agent pays for
    generator capability where it matters and skips it where it doesn't.
    """

    model_config = ConfigDict(extra="forbid")

    generator: list[str]
    expander: list[str]
    compressor: list[str]

    @model_validator(mode="after")
    def all_non_empty(self) -> StageLLMs:
        for stage in ("generator", "expander", "compressor"):
            if not getattr(self, stage):
                raise ValueError(f"llm_models.{stage} must be non-empty")
        return self

    @classmethod
    def uniform(cls, models: list[str]) -> StageLLMs:
        """Construct a StageLLMs where every stage draws from ``models``."""
        return cls(generator=list(models), expander=list(models), compressor=list(models))

    def all_models(self) -> list[str]:
        """Deduplicated union across stages, preserving first-seen order."""
        seen: dict[str, None] = {}
        for stage_list in (self.generator, self.expander, self.compressor):
            for m in stage_list:
                seen.setdefault(m, None)
        return list(seen.keys())


class StructuralConfig(BaseModel):
    """Internal engine type: index-building parameters passed to IndexBuilder."""

    chunking_strategy: str = "recursive"
    chunk_token_size: int = 512
    chunk_token_overlap: int = 64
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_type: IndexType = IndexType.VECTOR_ONLY

    @field_validator("chunk_token_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        return _validate_overlap_less_than_size(v, info)

    def chunks_fingerprint(self, corpus_hash: str) -> str:
        """16-char hash of chunker params + corpus identity — keys the chunks cache."""
        data = {
            "chunking_strategy": self.chunking_strategy,
            "chunk_token_size": self.chunk_token_size,
            "chunk_token_overlap": self.chunk_token_overlap,
            "corpus_hash": corpus_hash,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def embeddings_fingerprint(self, corpus_hash: str) -> str:
        """16-char hash of chunks_fingerprint + embedding_model — keys the embeddings cache."""
        data = {
            "chunks_hash": self.chunks_fingerprint(corpus_hash),
            "embedding_model": self.embedding_model,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def fingerprint(self) -> str:
        """16-char hash over chunker params + embedding_model.

        In-memory key only — callers that group configs within a single session
        (probe selection, exam_index_cache, bench script). Excludes ``index_type``
        because the cached chunks + embeddings are identical across index types;
        only the query path in ``RAGPipeline`` differs. Disk caches must use
        ``chunks_fingerprint``/``embeddings_fingerprint`` with a ``corpus_hash``.
        """
        data = {
            "chunking_strategy": self.chunking_strategy,
            "chunk_token_size": self.chunk_token_size,
            "chunk_token_overlap": self.chunk_token_overlap,
            "embedding_model": self.embedding_model,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


class RuntimeConfig(BaseModel):
    """Internal engine type: retrieval/generation parameters passed to RAGPipeline."""

    top_k: int = 5
    hybrid_alpha: float = 0.5
    # Hybrid fusion strategy. "alpha" blends normalized scores via
    # ``hybrid_alpha``; "rrf" merges BM25 and vector by reciprocal rank.
    # Only consulted when ``index_type`` is hybrid_bm25_vector.
    bm25_vector_fusion: str = "alpha"
    # When True, duplicate the top-scored passage at the end of the joined
    # context (input order otherwise preserved).
    long_context_reorder: bool = False
    # Passage compression applied before reorder/join. "tree_summarize"
    # recursively synthesises passages in batches of
    # ``_PASSAGE_COMPRESSOR_BATCH_SIZE``; "refine" threads a running answer
    # through each passage.
    passage_compressor: str = "none"
    reranker: str = "none"
    reranker_top_n: int = 5
    query_expansion: str = "none"
    # Per-stage LLMs. ``compressor_llm`` is None when ``passage_compressor`` is
    # "none" (no LLM call), ``expander_llm`` is None when ``query_expansion`` is
    # "none". ``generator_llm`` is always set.
    compressor_llm: str | None = None
    expander_llm: str | None = None
    generator_llm: str
    temperature: float = 0.0
    reasoning: bool = False
    # ``reasoning_effort`` only applies to the generator call.
    reasoning_effort: str = "medium"
    # Timeouts
    llm_timeout_s: float = 100.0  # per-call timeout passed to litellm.acompletion
    # Graph retrieval parameters (only used when index_type is graph-based)
    graph_query_mode: str = "hybrid"
    graph_top_k: int = 60


class GraphBuildConfig(BaseModel):
    """Fixed graph build configuration — set once, outside the optimizer search space.

    These parameters control how LightRAG constructs the knowledge graph. Changing
    any *content-affecting* field invalidates the cached graph (see ``config_hash``);
    the build will refuse to reuse a graph built with different content-affecting
    config. Concurrency/throughput/timeout knobs do not invalidate the cache.
    """

    extraction_model: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_token_size: int | None = None
    chunk_overlap_token_size: int | None = None
    entity_types: list[str] | None = None
    # Concurrency: keep low to avoid exhausting API rate limits.
    max_parallel_insert: int = Field(default=2, ge=1)
    llm_model_max_async: int = Field(default=4, ge=1)
    embedding_func_max_async: int = Field(default=8, ge=1)
    # Retries with exponential back-off on transient errors (429, 503, etc.).
    # All retries happen in our own async loop (see graph_store._make_llm_func);
    # we never enable LiteLLM's internal retries because they would hold
    # LightRAG's semaphore across invisible waits and can deadlock the worker pool.
    llm_model_max_retries: int = Field(default=3, ge=0)
    # Timeouts. LightRAG kills workers at ``2 * default_*_timeout`` internally,
    # so our per-call timeout + retry budget must fit inside that window (enforced
    # by the validator below).
    default_llm_timeout: int = Field(default=180, ge=30)
    default_embedding_timeout: int = Field(default=30, ge=10)
    extraction_call_timeout_s: float = Field(default=45.0, gt=0.0)
    extraction_retry_backoff_base_s: float = Field(default=5.0, gt=0.0)
    extraction_retry_backoff_max_s: float = Field(default=30.0, gt=0.0)
    # Build is batched so partial progress survives crashes. After each batch of
    # ``build_batch_size`` documents the manifest is updated atomically — a restart
    # resumes from the last completed batch.
    build_batch_size: int = Field(default=20, ge=1)
    # Passed to SentenceTransformer.encode(). Higher = better GPU utilisation;
    # 32 is the library default and typically too small for modern GPUs.
    embedding_batch_size: int = Field(default=64, ge=1)

    @model_validator(mode="after")
    def retry_budget_fits_worker_cap(self) -> GraphBuildConfig:
        """Ensure worst-case retry budget is under LightRAG's worker kill window.

        LightRAG wraps our LLM func in a semaphore + worker timeout of
        ``2 * default_llm_timeout``. Our async retry loop holds that semaphore
        the whole time it's running. If the worst-case budget (all attempts
        time out + all sleeps hit the jitter ceiling) meets or exceeds the
        worker cap, the worker is killed mid-retry and we lose observability
        over which attempt failed. Fail at parse time instead.
        """
        attempts = self.llm_model_max_retries + 1
        base = self.extraction_retry_backoff_base_s
        cap = self.extraction_retry_backoff_max_s
        # Jitter multiplier is up to 1.5 (see _make_llm_func).
        total_sleep_worst = sum(min(base * 2**i, cap) for i in range(self.llm_model_max_retries)) * 1.5
        budget = self.extraction_call_timeout_s * attempts + total_sleep_worst
        worker_cap = self.default_llm_timeout * 2
        if budget >= worker_cap:
            raise ValueError(
                f"GraphBuildConfig retry budget {budget:.1f}s >= LightRAG worker cap "
                f"{worker_cap}s (2 * default_llm_timeout). Lower llm_model_max_retries "
                f"or extraction_call_timeout_s, cap extraction_retry_backoff_max_s, "
                f"or raise default_llm_timeout."
            )
        return self

    def config_hash(self) -> str:
        """16-char hash of content-affecting fields.

        Used to detect when a persisted graph was built with a different
        extraction model, embedding model, chunker, or entity_types — in which
        case the graph is not safe to reuse. Excludes all concurrency/throughput/
        timeout knobs (e.g. ``max_parallel_insert``, ``llm_model_max_async``,
        ``llm_model_max_retries``, ``default_llm_timeout``,
        ``extraction_call_timeout_s``, ``build_batch_size``, ``embedding_batch_size``)
        since those don't change the resulting graph.
        """
        data = {
            "extraction_model": self.extraction_model,
            "embedding_model": self.embedding_model,
            "chunk_token_size": self.chunk_token_size,
            "chunk_overlap_token_size": self.chunk_overlap_token_size,
            "entity_types": sorted(self.entity_types) if self.entity_types else None,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


class VLLMConfig(BaseModel):
    """vLLM server settings for framework-managed local model serving.

    When hosted_vllm/ models appear in the search space, the framework
    automatically starts and stops a vLLM server subprocess. This config
    is optional — sensible defaults are used when the section is omitted.
    """

    max_model_len: int | None = None  # None = vLLM auto-detects from model config
    gpu_memory_utilization: float = Field(default=0.90, gt=0.0, le=1.0)
    enforce_eager: bool = True  # Skip CUDA graphs for faster model swap (~30s vs ~80s)
    port: int = Field(default=8000, ge=1, le=65535)
    startup_timeout: int = Field(default=180, ge=10)
    extra_args: list[str] = Field(default_factory=list)
    binary: str = "vllm"


class TrialConfig(BaseModel):
    """Complete (flat) configuration for a single optimization trial.

    All tunable parameters live at the top level — no structural/runtime split.
    Use to_structural() and to_runtime() to get the internal engine types.
    """

    # Index-building parameters
    chunking_strategy: str = "recursive"
    chunk_token_size: int = 512
    chunk_token_overlap: int = 64
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_type: IndexType = IndexType.VECTOR_ONLY
    # Retrieval parameters
    top_k: int = 5
    hybrid_alpha: float = 0.5
    bm25_vector_fusion: str = "alpha"
    long_context_reorder: bool = False
    passage_compressor: str = "none"
    reranker: str = "none"
    reranker_top_n: int = 5
    query_expansion: str = "none"
    # Per-stage LLMs. ``compressor_llm`` is None when ``passage_compressor``
    # is "none"; ``expander_llm`` is None when ``query_expansion`` is "none".
    # ``generator_llm`` is always set. ``reasoning``/``reasoning_effort``
    # apply only to the generator call.
    compressor_llm: str | None = None
    expander_llm: str | None = None
    generator_llm: str
    temperature: float = 0.0
    reasoning: bool = False
    # Graph retrieval parameters (only active when index_type is graph-based)
    graph_query_mode: str = "hybrid"
    graph_top_k: int = 60

    @field_validator("chunk_token_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        return _validate_overlap_less_than_size(v, info)

    @model_validator(mode="after")
    def per_stage_llm_alive_iff_stage_active(self) -> TrialConfig:
        """``compressor_llm``/``expander_llm`` must be set iff the stage runs."""
        if self.passage_compressor == "none":
            if self.compressor_llm is not None:
                raise ValueError(
                    f"compressor_llm must be null when passage_compressor='none' "
                    f"(got {self.compressor_llm!r})"
                )
        else:
            if self.compressor_llm is None:
                raise ValueError(
                    f"compressor_llm is required when passage_compressor="
                    f"{self.passage_compressor!r}"
                )
        if self.query_expansion == "none":
            if self.expander_llm is not None:
                raise ValueError(
                    f"expander_llm must be null when query_expansion='none' "
                    f"(got {self.expander_llm!r})"
                )
        else:
            if self.expander_llm is None:
                raise ValueError(
                    f"expander_llm is required when query_expansion="
                    f"{self.query_expansion!r}"
                )
        return self

    def to_structural(self) -> StructuralConfig:
        """Extract index-building parameters as an internal StructuralConfig."""
        return StructuralConfig(
            chunking_strategy=self.chunking_strategy,
            chunk_token_size=self.chunk_token_size,
            chunk_token_overlap=self.chunk_token_overlap,
            embedding_model=self.embedding_model,
            index_type=self.index_type,
        )

    def to_runtime(self, reasoning_effort: str = "medium") -> RuntimeConfig:
        """Extract retrieval/generation parameters as an internal RuntimeConfig."""
        return RuntimeConfig(
            top_k=self.top_k,
            hybrid_alpha=self.hybrid_alpha,
            bm25_vector_fusion=self.bm25_vector_fusion,
            long_context_reorder=self.long_context_reorder,
            passage_compressor=self.passage_compressor,
            reranker=self.reranker,
            reranker_top_n=self.reranker_top_n,
            query_expansion=self.query_expansion,
            compressor_llm=self.compressor_llm,
            expander_llm=self.expander_llm,
            generator_llm=self.generator_llm,
            temperature=self.temperature,
            reasoning=self.reasoning,
            reasoning_effort=reasoning_effort,
            graph_query_mode=self.graph_query_mode,
            graph_top_k=self.graph_top_k,
        )

    def structural_fingerprint(self) -> str:
        """In-memory dedup key — delegates to StructuralConfig.fingerprint()."""
        return self.to_structural().fingerprint()

    def to_prompt_json(self, include_graph: bool) -> str:
        """Serialize to JSON for LLM prompts, optionally excluding graph fields."""
        exclude = _GRAPH_TRIAL_FIELDS if not include_graph else None
        return json.dumps(self.model_dump(mode="json", exclude=exclude), indent=2)

    def to_prompt_dump(self, include_graph: bool) -> dict:
        """Dump to dict, optionally excluding graph fields."""
        exclude = _GRAPH_TRIAL_FIELDS if not include_graph else None
        return self.model_dump(mode="json", exclude=exclude)


class ChunkingSearchSpace(BaseModel):
    """Allowed chunking strategies and parameter ranges."""

    strategies: list[str] = ["recursive"]
    chunk_token_size: NumericDim = NumericRange(min=64, max=512)
    chunk_token_overlap: NumericDim = NumericRange(min=0, max=128)


class RerankerSearchSpace(BaseModel):
    """Allowed reranker models and top_n range."""

    models: list[str] = ["none"]
    top_n: NumericDim = NumericRange(min=3, max=10)


class GraphRetrievalSearchSpace(BaseModel):
    """Graph retrieval parameters the optimizer can tune.

    Only relevant when index_types includes graph_only or hybrid_graph_vector.
    """

    graph_query_modes: list[str] = ["local", "global", "hybrid"]
    graph_top_k: NumericDim = NumericRange(min=20, max=100)


_REASONING_UNSUPPORTED_PREFIXES = ("ollama/",)


def _probe_model(model: str) -> tuple[bool, str | None]:
    """Live-test a model by making a minimal completion call.

    Called only for models that fail the static LiteLLM catalog check.
    """
    try:
        litellm.completion(model=model, messages=[{"role": "user", "content": "ping"}], max_tokens=1, timeout=10)
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _is_in_litellm_catalog(model: str) -> bool:
    if model in litellm.model_cost:
        return True
    if "/" in model:
        provider, suffix = model.split("/", 1)
        provider_models = litellm.models_by_provider.get(provider)
        if provider_models is not None and (
            suffix in provider_models or f"{provider}/{suffix}" in provider_models
        ):
            return True
    return False


class SearchSpace(BaseModel):
    """Flat search space: all parameters the optimizer can tune.

    Index-building params (chunking, embedding_models, index_types) trigger
    re-indexing when changed. All others are swappable without rebuilding.
    """

    # Index-building parameters
    chunking: ChunkingSearchSpace = ChunkingSearchSpace()
    embedding_models: list[str]
    index_types: list[IndexType] = [IndexType.VECTOR_ONLY]
    # Retrieval parameters
    top_k: NumericDim = NumericRange(min=3, max=20)
    hybrid_alpha: NumericDim = NumericRange(min=0.0, max=1.0)
    bm25_vector_fusion: list[str] = ["alpha"]
    long_context_reorder: list[bool] = [False]
    passage_compressor: list[str] = ["none"]
    reranker: RerankerSearchSpace = RerankerSearchSpace()
    query_expansion: list[str] = ["none"]
    # Generation parameters. ``llm_models`` is a required per-stage map —
    # ``generator`` (final answer LLMs), ``expander`` (query-expansion LLMs),
    # ``compressor`` (passage-compression LLMs). Use ``StageLLMs.uniform(...)``
    # in tests when all three stages share a pool.
    llm_models: StageLLMs
    temperature: NumericRange = NumericRange(min=0.0, max=1.0)
    reasoning: bool = True
    reasoning_effort: str = "medium"
    # Graph retrieval
    graph_retrieval: GraphRetrievalSearchSpace | None = None

    def is_reasoning_allowed(self, model: str) -> bool:
        """Whether ``reasoning_effort`` can be toggled for ``model``.

        LiteLLM is the ground truth — if it says the model doesn't support
        ``reasoning_effort``, the parameter is silently dropped and there's
        no point in the optimizer toggling it. The ollama prefix is gated
        explicitly because no ollama model surfaces reasoning through
        LiteLLM today. On lookup failure (provider not in catalog) defer
        to the global ``reasoning`` flag.
        """
        if not self.reasoning:
            return False
        if model.startswith(_REASONING_UNSUPPORTED_PREFIXES):
            return False
        try:
            return bool(litellm.supports_reasoning(model=model))
        except Exception:  # noqa: BLE001
            return self.reasoning

    def hybrid_alpha_is_dead(self) -> bool:
        """``hybrid_alpha`` only affects the pipeline when a hybrid_bm25_vector
        index is reachable in this run AND alpha-blend fusion is reachable."""
        if not any(it == IndexType.HYBRID_BM25_VECTOR for it in self.index_types):
            return True
        return "alpha" not in self.bm25_vector_fusion

    def bm25_vector_fusion_is_dead(self) -> bool:
        """``bm25_vector_fusion`` only affects the pipeline when a
        hybrid_bm25_vector index is reachable in this run."""
        return not any(it == IndexType.HYBRID_BM25_VECTOR for it in self.index_types)

    def long_context_reorder_is_dead(self) -> bool:
        """``long_context_reorder`` is a no-op when every passage-compressor
        choice collapses retrieval to a single string — there is nothing to
        reorder. Dead when ``"none"`` is not enumerated for
        ``passage_compressor``."""
        return "none" not in self.passage_compressor

    def reranker_top_n_is_dead(self) -> bool:
        """``reranker_top_n`` only affects the pipeline when some real reranker
        (i.e. anything other than ``"none"``) is reachable in this run."""
        return all(m == "none" for m in self.reranker.models)

    @model_validator(mode="after")
    def chunk_overlap_feasible(self) -> SearchSpace:
        """At least one (chunk_token_size, chunk_token_overlap) pair must
        satisfy ``overlap < size``.

        Catches misconfigured discrete grids at parse time. Without this,
        the sampler would fall back to a non-grid value at runtime, the
        ``validate_trial`` would flag the violation, and the trial would
        silently consume a budget slot.
        """
        size_max = _dim_max_value(self.chunking.chunk_token_size)
        overlap_min = _dim_min_value(self.chunking.chunk_token_overlap)
        if overlap_min >= size_max:
            raise ValueError(
                f"chunking.chunk_token_overlap minimum ({overlap_min}) must be "
                f"strictly less than chunking.chunk_token_size maximum ({size_max}); "
                "no legal (size, overlap) pair exists."
            )
        return self

    @model_validator(mode="after")
    def reranker_top_n_feasible(self) -> SearchSpace:
        """At least one (top_k, reranker_top_n) pair must satisfy ``top_n <= top_k``.

        Skipped when no real reranker is reachable (reranker_top_n is unused).
        """
        if self.reranker_top_n_is_dead():
            return self
        top_n_min = _dim_min_value(self.reranker.top_n)
        top_k_max = _dim_max_value(self.top_k)
        if top_n_min > top_k_max:
            raise ValueError(
                f"reranker.top_n minimum ({top_n_min}) must be <= top_k maximum "
                f"({top_k_max}); no legal (top_k, reranker_top_n) pair exists."
            )
        return self

    def compressor_llm_is_dead(self) -> bool:
        """``compressor_llm`` is unused when no compressor stage ever runs."""
        return all(c == "none" for c in self.passage_compressor)

    def expander_llm_is_dead(self) -> bool:
        """``expander_llm`` is unused when no query-expansion stage ever runs."""
        return all(qe == "none" for qe in self.query_expansion)

    def active_levers(self) -> set[str]:
        """Field names whose runtime behavior is non-trivial in this search space.

        A lever is *active* when at least one trial path exercises it — either
        because the agent can choose a non-trivial value, or because the
        single pinned value is non-trivial (e.g. ``passage_compressor=
        ["tree_summarize"]``). Pinning ≠ inactive. The agent benefits from
        guidance whenever a lever is active, even if the value is fixed.
        """
        active: set[str] = {
            "chunking_strategy",
            "chunk_token_size",
            "chunk_token_overlap",
            "embedding_model",
            "index_type",
            "top_k",
            "reranker",
            "generator_llm",
            "temperature",
            "reasoning",
        }
        if not self.expander_llm_is_dead():
            active.update({"query_expansion", "expander_llm"})
        if not self.compressor_llm_is_dead():
            active.update({"passage_compressor", "compressor_llm"})
        if not self.long_context_reorder_is_dead() and any(self.long_context_reorder):
            active.add("long_context_reorder")
        if not self.bm25_vector_fusion_is_dead():
            active.add("bm25_vector_fusion")
        if not self.hybrid_alpha_is_dead():
            active.add("hybrid_alpha")
        if not self.reranker_top_n_is_dead():
            active.add("reranker_top_n")
        if self.graph_retrieval is not None:
            active.update({"graph_query_mode", "graph_top_k"})
        return active

    def pinned_field_values(self) -> dict[str, object]:
        """Return the single legal TrialConfig value for each effectively-pinned field.

        A field is *pinned* when its search-space surface has exactly one legal
        value (numeric dim with one value or single-element choice list) OR
        when it is structurally dead given the rest of the search space
        (e.g. ``reranker_top_n`` when no real reranker is reachable). Pinned
        values get auto-injected into the proposer's YAML at parse time so the
        agent never has to emit them and never trips the validator for fields
        it could not have proposed differently.

        ``reasoning`` is pinned to ``False`` when the search space disables it
        globally (``ss.reasoning=False``). The corner case where ``ss.reasoning
        =True`` but no model in the space supports ``reasoning_effort`` is
        handled by the existing rendering path in ``to_agent_prompt`` (which
        emits an informational line and a literal ``false`` in the example).
        """
        pinned: dict[str, object] = {}

        if _dim_is_fixed(self.chunking.chunk_token_size):
            pinned["chunk_token_size"] = int(_dim_min_value(self.chunking.chunk_token_size))
        if _dim_is_fixed(self.chunking.chunk_token_overlap):
            pinned["chunk_token_overlap"] = int(_dim_min_value(self.chunking.chunk_token_overlap))
        if _dim_is_fixed(self.top_k):
            pinned["top_k"] = int(_dim_min_value(self.top_k))
        if _dim_is_fixed(self.hybrid_alpha):
            pinned["hybrid_alpha"] = float(_dim_min_value(self.hybrid_alpha))
        if _dim_is_fixed(self.reranker.top_n):
            pinned["reranker_top_n"] = int(_dim_min_value(self.reranker.top_n))
        if self.temperature.min == self.temperature.max:
            pinned["temperature"] = float(self.temperature.min)
        if not self.reasoning:
            pinned["reasoning"] = False

        if len(self.chunking.strategies) == 1:
            pinned["chunking_strategy"] = self.chunking.strategies[0]
        if len(self.embedding_models) == 1:
            pinned["embedding_model"] = self.embedding_models[0]
        if len(self.index_types) == 1:
            pinned["index_type"] = self.index_types[0].value
        if len(self.reranker.models) == 1:
            pinned["reranker"] = self.reranker.models[0]
        if len(self.query_expansion) == 1:
            pinned["query_expansion"] = self.query_expansion[0]
        if len(self.llm_models.generator) == 1:
            pinned["generator_llm"] = self.llm_models.generator[0]
        if not self.compressor_llm_is_dead() and len(self.llm_models.compressor) == 1:
            pinned["compressor_llm"] = self.llm_models.compressor[0]
        if not self.expander_llm_is_dead() and len(self.llm_models.expander) == 1:
            pinned["expander_llm"] = self.llm_models.expander[0]
        if self.compressor_llm_is_dead():
            pinned["compressor_llm"] = None
        if self.expander_llm_is_dead():
            pinned["expander_llm"] = None
        if len(self.bm25_vector_fusion) == 1:
            pinned["bm25_vector_fusion"] = self.bm25_vector_fusion[0]
        if len(self.long_context_reorder) == 1:
            pinned["long_context_reorder"] = self.long_context_reorder[0]
        if len(self.passage_compressor) == 1:
            pinned["passage_compressor"] = self.passage_compressor[0]

        if self.reranker_top_n_is_dead():
            pinned.setdefault("reranker_top_n", int(_dim_min_value(self.reranker.top_n)))
        if self.hybrid_alpha_is_dead():
            pinned.setdefault("hybrid_alpha", float(_dim_min_value(self.hybrid_alpha)))
        if self.bm25_vector_fusion_is_dead():
            pinned.setdefault("bm25_vector_fusion", self.bm25_vector_fusion[0])
        if self.long_context_reorder_is_dead():
            pinned.setdefault("long_context_reorder", self.long_context_reorder[0])

        if self.graph_retrieval is not None:
            if len(self.graph_retrieval.graph_query_modes) == 1:
                pinned["graph_query_mode"] = self.graph_retrieval.graph_query_modes[0]
            if _dim_is_fixed(self.graph_retrieval.graph_top_k):
                pinned["graph_top_k"] = int(_dim_min_value(self.graph_retrieval.graph_top_k))

        return pinned


class ParsingConfig(BaseModel):
    """Document parsing configuration.

    These settings control how raw files are converted to text before
    chunking. Not part of the optimizer search space — set once per project.

    The ``near_duplicate_*`` knobs feed the corpus-cleaner that runs once at
    setup; the cleaner only emits *metadata* (a canonical-doc-ids list and
    an alias-to-canonical map). The corpus the optimizer evaluates against
    is never modified — duplicates remain in the index for every trial so
    the framework recommends a configuration that wins on the user's real
    deployment.
    """

    parser: str = "docling"
    ocr: bool = True
    table_structure: bool = True
    # Containment cutoff for near-duplicate detection. The corpus cleaner
    # tokenises each document with a normalising regex (lowercase, word
    # characters only, drops single-char tokens) and clusters pairs whose
    # smaller token-shingle set is contained in the larger above this
    # fraction. 0.85 catches OCR-of-PDF page images (typically ~85-90%
    # containment due to character-substitution noise on dagger marks,
    # affiliation symbols, etc.); raise toward 1.0 for stricter clustering.
    # We use containment rather than Jaccard because containment subsumes
    # Jaccard at the same threshold and additionally catches asymmetric
    # subset relationships (a one-page image inside a multi-page PDF).
    near_duplicate_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    # Set to false to disable near-duplicate detection entirely (every doc is
    # its own canonical, alias map is identity). Useful for small synthetic
    # corpora and tests.
    near_duplicate_detection_enabled: bool = True


# Default eligible section labels for the chunk-pair indexer. The closed
# taxonomy lives in ``engine.section_classifier``; these are the string
# values that survive in YAML.
_DEFAULT_ELIGIBLE_SECTION_TYPES: tuple[str, ...] = (
    "body",
    "abstract",
    "methods",
    "results",
    "discussion",
    "other",
)
_VALID_SECTION_TYPES: frozenset[str] = frozenset(
    {
        "body",
        "abstract",
        "methods",
        "results",
        "discussion",
        "references",
        "acknowledgments",
        "author_info",
        "other",
    }
)


# Reasoning-type taxonomy for the typed composition prompt. Each generated
# question is tagged with the type the LLM produced (which may differ from
# the per-seed ``preferred_type`` if the chunks didn't naturally fit).
QUESTION_TYPES: tuple[str, ...] = (
    "extraction",
    "definitional",
    "bridge",
    "comparison",
    "numeric",
)
_VALID_QUESTION_TYPES: frozenset[str] = frozenset(QUESTION_TYPES)
_DEFAULT_QUESTION_TYPE_WEIGHTS: dict[str, float] = {
    "extraction": 0.25,
    "definitional": 0.10,
    "bridge": 0.25,
    "comparison": 0.20,
    "numeric": 0.20,
}


class ExaminerConfig(BaseModel):
    """Settings for the open-ended 2-hop exam generator.

    The generator embeds every eligible chunk once, pairs each chunk with its
    top-K cross-document nearest neighbours under cosine similarity, batches
    the resulting seeds into typed composition LLM calls, runs each candidate
    through an LLM single-hop probe and the oracle answerability gate, then
    selects the most discriminating subset via a 4-probe item-analysis pass
    over diverse RAG configurations. All LLM-billed work scales with
    ``exam_size`` rather than corpus size.
    """

    exam_size: int = 60
    # 3× over-generation absorbs Step B / C / gate rejections.
    pair_overgeneration_factor: float = Field(default=3.0, ge=1.0)
    # Probe-based discrimination filtering. When True, every candidate that
    # clears the oracle gate is run through 2-4 probes (search-space
    # extremes) and the exam is built from the most discriminating items.
    # Disable only for tests / debugging — the ordinary pipeline relies on
    # this for non-saturating exams.
    probe_selection: bool = True
    # Composition batching: K seeds per LLM call. K=4 is the documented sweet
    # spot — small enough that attention isn't diluted, large enough to amortise.
    composition_batch_size: int = Field(default=4, ge=1, le=10)

    # Sampling temperature for the composition LLM. Default 1.0 because
    # several frontier models require exactly that value; lower it on models
    # that allow flexibility for stricter rule-following.
    composition_temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    # Cooler temperature for batches whose seeds prefer the ``numeric`` type.
    # Math reliability matters more than diversity — a cooler temperature
    # reduces formula-mismatch rejections downstream. Set to None to fall
    # back to ``composition_temperature``.
    composition_temperature_numeric: float | None = Field(default=0.2, ge=0.0, le=2.0)

    # Per-seed PREFERRED question-type sampling weights. The composition LLM
    # is asked to generate the preferred type when the chunks support it, and
    # may fall back to any other type or refuse. Weights need not sum to 1.0
    # — they're normalised before sampling.
    question_type_weights: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_QUESTION_TYPE_WEIGHTS),
    )

    # Seed-origin mix: proportions of single-chunk, same-doc-pair, and
    # cross-doc-pair seeds in the final pool. Must sum to 1.0. Tune per
    # corpus: HotpotQA-like Wikipedia is cross-doc-rich; UniDoc-like medical
    # PDFs are cross-doc-sparse and benefit from single-chunk + same-doc.
    seed_mix: dict[str, float] = Field(
        default_factory=lambda: {
            "single_chunk": 0.4,
            "same_doc_pair": 0.3,
            "cross_doc_pair": 0.3,
        },
    )

    # Pair-embedding index for cross-doc 2-hop seed discovery. bge-m3 has an
    # 8192-token max — smaller models (max 256/512) silently truncate our
    # 1500-word chunks and embed only the intro paragraph.
    pair_embedding_model: str = "BAAI/bge-m3"
    # Per-chunk neighbour count. Higher values broaden the seed pool but may
    # include weaker pairs the LLM will refuse anyway.
    pair_top_k_per_chunk: int = Field(default=5, ge=1, le=20)

    # Same-doc pair generator: cosine band [min, max] for picking related-but-
    # non-paraphrase chunk pairs within one document. Section-disjoint filter
    # is applied on top.
    same_doc_pair_cosine_min: float = Field(default=0.4, ge=0.0, le=1.0)
    same_doc_pair_cosine_max: float = Field(default=0.85, ge=0.0, le=1.0)

    # Source fact verification (verbatim with fuzzy snap-to-source for minor LLM drift).
    source_fact_verify_fuzzy_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    # Chunk-relevance matcher thresholds (shared with retrieval evaluator).
    chunk_relevance_min_overlap_chars: int = Field(default=50, ge=1)
    chunk_relevance_ngram_size: int = Field(default=5, ge=1, le=20)
    chunk_relevance_overlap_threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    chunk_relevance_min_run: int = Field(default=5, ge=1)

    # Document handling — long PDFs get split before chunking by index_builder.
    doc_split_word_threshold: int = Field(default=24_000, ge=1_000)
    doc_section_word_size: int = Field(default=1_500, ge=200)
    min_doc_words: int = Field(default=200, ge=0)

    # Section classifier — chunk-pair indexer skips chunks whose heuristic
    # section label is NOT in this list. Default excludes references,
    # acknowledgments, and author_info. The taxonomy is defined in
    # ``engine.section_classifier.SectionLabel``.
    eligible_section_types: list[str] = Field(default_factory=lambda: list(_DEFAULT_ELIGIBLE_SECTION_TYPES))

    # Scoring.
    # composite = alpha * answer_accuracy + (1 - alpha) * mean_retrieval_quality
    retrieval_quality_alpha: float = Field(default=0.7, ge=0.0, le=1.0)

    # Embedding model fallback for any small-utility embedder paths.
    # Pairing uses a SEPARATE ``pair_embedding_model`` (above), since pairing
    # requires a long-context model that this fallback intentionally is not.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @model_validator(mode="after")
    def valid_doc_section_size(self) -> ExaminerConfig:
        if self.doc_section_word_size >= self.doc_split_word_threshold:
            raise ValueError("doc_section_word_size must be smaller than doc_split_word_threshold")
        if self.same_doc_pair_cosine_min >= self.same_doc_pair_cosine_max:
            raise ValueError("same_doc_pair_cosine_min must be < same_doc_pair_cosine_max")
        return self

    @field_validator("eligible_section_types")
    @classmethod
    def known_section_types(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - _VALID_SECTION_TYPES)
        if unknown:
            raise ValueError(
                f"eligible_section_types contains unknown labels: {unknown}. "
                f"Valid labels: {sorted(_VALID_SECTION_TYPES)}"
            )
        if not v:
            raise ValueError("eligible_section_types must not be empty (or every chunk would be skipped)")
        return v

    @field_validator("seed_mix")
    @classmethod
    def valid_seed_mix(cls, v: dict[str, float]) -> dict[str, float]:
        valid_origins = {"single_chunk", "same_doc_pair", "cross_doc_pair"}
        unknown = sorted(set(v) - valid_origins)
        if unknown:
            raise ValueError(f"seed_mix contains unknown origins: {unknown}. Valid origins: {sorted(valid_origins)}")
        missing = sorted(valid_origins - set(v))
        if missing:
            raise ValueError(f"seed_mix missing required origins: {missing}")
        if any(w < 0 for w in v.values()):
            raise ValueError("seed_mix weights must be non-negative")
        total = sum(v.values())
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"seed_mix weights must sum to 1.0 (got {total:.4f})")
        return v

    @field_validator("question_type_weights")
    @classmethod
    def known_question_types(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(v) - _VALID_QUESTION_TYPES)
        if unknown:
            raise ValueError(
                f"question_type_weights contains unknown types: {unknown}. Valid types: {sorted(_VALID_QUESTION_TYPES)}"
            )
        if any(w < 0 for w in v.values()):
            raise ValueError("question_type_weights must be non-negative")
        if sum(v.values()) <= 0:
            raise ValueError("question_type_weights must have at least one positive weight")
        return v


class AgentConfig(BaseModel):
    """Settings for the LLM agents."""

    optimizer_model: str = "gemini/gemini-3-flash-preview"
    examiner_model: str = "gemini/gemini-3-flash-preview"
    # Strong reference model for the oracle answerability gate (during exam
    # generation) and the trial-time judge (grades free-form predictions
    # against gold answers when EM=0). Acts as a ceiling check, so it must be
    # at least as strong as the strongest probe LLM. When None the framework
    # auto-picks the strongest LLM in the search space.
    judge_model: str | None = None
    # Reasoning effort for the optimizer (Diagnoser + Proposer) LLM calls. When
    # set and the model supports it, passes reasoning_effort through to
    # litellm.acompletion. Set to null in YAML to disable.
    optimizer_reasoning_effort: Literal["low", "medium", "high"] | None = "medium"
    # Reasoning effort for examiner LLM calls (composition + single-hop probe)
    # routed to ``examiner_model``. Same shape as ``optimizer_reasoning_effort``;
    # silently dropped on models that don't support reasoning. Defaults to None
    # so reasoning is opt-in for the examiner.
    examiner_reasoning_effort: Literal["low", "medium", "high"] | None = None
    max_history_trials: int = 10
    concurrency: int = Field(default=10, ge=1)


class MetaConfig(BaseModel):
    """Project-level settings."""

    project_name: str = "my-rag-project"
    corpus_path: str = "./data/corpus/"
    corpus_description: str = ""
    output_dir: str = "./experiments/"
    max_trials: int = 30
    cache_max_gb: float = Field(default=5.0, gt=0.0)
    # Score band around the current leader used to flag the cheapest-in-band
    # frontier member in the state card. The agent reads it as a soft target
    # for cost-cutting moves during ``polish`` stance.
    polish_score_tolerance: float = Field(default=0.05, ge=0.0, le=1.0)
    # Early-exit gate. The agent may emit ``strategy.stance="done"`` only when
    # ``allow_early_exit`` is True AND the state card's ``done_eligible``
    # flag is True. ``done_eligible`` is set by ``build_state_card`` from the
    # following knobs:
    #   - ``min_trials_before_done`` — minimum trial count before done is legal.
    #   - ``min_frontier_size_for_done`` — at least one observed cost/score
    #     trade-off (frontier ≥ 2) before terminating.
    #   - ``early_exit_hv_epsilon`` — recent HV expansion (last 3 trials) must
    #     be at or below this to count as "frontier not currently expanding".
    allow_early_exit: bool = True
    min_trials_before_done: int = Field(default=4, ge=1)
    min_frontier_size_for_done: int = Field(default=2, ge=1)
    early_exit_hv_epsilon: float = Field(default=0.001, ge=0.0)
    # Anti-flapping lock. Once the agent commits to a stance at trial K, it
    # must hold that stance for at least ``min_stance_lock_trials`` further
    # trials (legal transitions resume at K + min_stance_lock_trials + 1).
    # Default 1 gives one full trial of forced commitment.
    min_stance_lock_trials: int = Field(default=1, ge=0)
    # Minimum drop on any tracked axis (score, acc_given_complete,
    # retrieval_complete, cost-down) for a Diagnoser-claimed regression to be
    # accepted. The orchestrator validates ``Diagnosis.regression_detected``
    # against the just-computed ``LeverEffectDelta`` set; if no listed axis
    # drops by at least this much, the regression claim is rejected.
    regression_threshold: float = Field(default=0.03, ge=0.0)
    # Optional seed for stratified failure-sample selection. When None, the
    # sampler derives its seed from the trial number — deterministic per
    # trial, varying across trials so the deep blocks are not identical run
    # to run. Set to a fixed int for fully repeatable picks.
    failure_sample_seed: int | None = None


class ProjectConfig(BaseModel):
    """The full project configuration loaded from YAML.

    Contains the search space (tunable parameters) plus project-level settings
    for parsing, examiner, and agent that are fixed for the optimization run.
    """

    meta: MetaConfig = MetaConfig()
    parsing: ParsingConfig = ParsingConfig()
    search_space: SearchSpace
    graph: GraphBuildConfig | None = None
    vllm: VLLMConfig | None = None
    examiner: ExaminerConfig = ExaminerConfig()
    agent: AgentConfig = AgentConfig()

    # Maps short names used in the search space (and agent/graph model fields)
    # to the LiteLLM model identifier the framework actually calls. Simple
    # form: ``alias: "provider/deployment-name"``. Extended form:
    # ``alias: {model: ..., api_base: ..., api_key: ..., api_version: ...}``
    # for custom OpenAI-compatible endpoints. Omit entirely when every model
    # is reachable by its canonical LiteLLM name.
    model_aliases: dict[str, str | dict[str, Any]] = Field(default_factory=dict)

    # Populated at runtime from KnowledgeBase — not in YAML
    embedding_token_limits: dict[str, int] = Field(default_factory=dict, exclude=True)

    @field_validator("model_aliases")
    @classmethod
    def alias_values_well_formed(cls, v: dict[str, Any]) -> dict[str, Any]:
        for alias, target in v.items():
            if isinstance(target, str):
                if not target:
                    raise ValueError(f"model_aliases[{alias!r}] target is empty")
                continue
            if isinstance(target, dict):
                if "model" not in target or not isinstance(target["model"], str) or not target["model"]:
                    raise ValueError(
                        f"model_aliases[{alias!r}]: extended-form value must have a non-empty "
                        f"'model' string. Got: {target!r}"
                    )
                continue
            raise ValueError(
                f"model_aliases[{alias!r}]: must be a string or a dict with a 'model' key. Got: {type(target).__name__}"
            )
        return v

    def resolve_alias(self, name: str) -> str:
        """Return the LiteLLM target for ``name`` after alias lookup, or ``name`` if not aliased."""
        target = self.model_aliases.get(name)
        if target is None:
            return name
        return target if isinstance(target, str) else target["model"]

    @model_validator(mode="after")
    def validate_llm_models(self) -> ProjectConfig:
        """Validate that every llm_model is callable by LiteLLM.

        Step 1: static catalog check (free, covers most models). When the name
        is in ``model_aliases``, the alias key is what we check against the
        catalog — a known alias key means the canonical model is real and we
        trust the user-declared deployment behind it.

        Step 2: live probe via completion(max_tokens=1) for any name that
        fails the static check. For aliased names this probes the *resolved
        target* (the actual deployment), so a deployment-name typo surfaces
        here instead of during the first real run.

        Raises ValueError listing all models that fail both checks.
        """
        needs_probe: list[tuple[str, str]] = []  # (display_name, target_to_probe)
        for model in self.search_space.llm_models.all_models():
            if model.startswith("hosted_vllm/"):
                continue  # Framework-managed; vLLM server isn't running at config time
            if _is_in_litellm_catalog(model):
                continue
            target = self.resolve_alias(model)
            if target != model and _is_in_litellm_catalog(target):
                continue
            needs_probe.append((model, target))

        if not needs_probe:
            return self

        failed: list[str] = []
        errors: list[str] = []
        for display, target in needs_probe:
            ok, err = _probe_model(target)
            if not ok:
                label = display if display == target else f"{display} (→ {target})"
                failed.append(label)
                errors.append(f"  {label}: {err}")

        if failed:
            raise ValueError(
                "The following llm_models could not be called by LiteLLM:\n"
                + "\n".join(errors)
                + "\nCheck model names and ensure required API keys are set."
            )
        return self

    @model_validator(mode="after")
    def graph_consistency(self) -> ProjectConfig:
        """Enforce mutual consistency between the graph build config and search space."""
        ss = self.search_space
        uses_graph = any(it in GRAPH_INDEX_TYPES for it in ss.index_types)

        if uses_graph and self.graph is None:
            raise ValueError(
                "search_space.index_types includes graph-based types "
                f"({[it.value for it in ss.index_types if it in GRAPH_INDEX_TYPES]}) "
                "but no 'graph:' build config is defined. Add a 'graph:' section to your YAML."
            )

        if not uses_graph and ss.graph_retrieval is not None:
            raise ValueError(
                "search_space.graph_retrieval is defined but no graph-based index types are in "
                "search_space.index_types. Either add a graph index type or remove graph_retrieval."
            )

        return self

    def uses_graph(self) -> bool:
        """Return True if any graph-based index type is in the search space."""
        return any(it in GRAPH_INDEX_TYPES for it in self.search_space.index_types)

    def validate_trial(self, trial: TrialConfig) -> list[str]:
        """Check whether a proposed trial config falls within the search space.

        Returns a list of violation messages (empty = valid).
        """
        violations: list[str] = []
        ss = self.search_space

        # --- Index-building checks ---
        if trial.chunking_strategy not in ss.chunking.strategies:
            violations.append(f"chunking_strategy '{trial.chunking_strategy}' not in {ss.chunking.strategies}")
        if not ss.chunking.chunk_token_size.contains(trial.chunk_token_size):
            violations.append(
                f"chunk_token_size {trial.chunk_token_size} outside "
                f"{_describe_dim(ss.chunking.chunk_token_size)}"
            )
        if not ss.chunking.chunk_token_overlap.contains(trial.chunk_token_overlap):
            violations.append(
                f"chunk_token_overlap {trial.chunk_token_overlap} outside "
                f"{_describe_dim(ss.chunking.chunk_token_overlap)}"
            )
        if trial.embedding_model not in ss.embedding_models:
            violations.append(f"embedding_model '{trial.embedding_model}' not in {ss.embedding_models}")
        # Cross-field: chunk_token_size vs embedding model token capacity
        if trial.embedding_model in self.embedding_token_limits:
            max_tokens = self.embedding_token_limits[trial.embedding_model]
            if trial.chunk_token_size > max_tokens:
                violations.append(
                    f"chunk_token_size {trial.chunk_token_size} exceeds embedding model "
                    f"'{trial.embedding_model}' limit of {max_tokens} tokens. "
                    f"Reduce chunk_token_size to <={max_tokens} or choose a model with higher capacity."
                )
        if trial.index_type not in ss.index_types:
            violations.append(f"index_type '{trial.index_type.value}' not in {[t.value for t in ss.index_types]}")

        # --- Retrieval checks ---
        if not ss.top_k.contains(trial.top_k):
            violations.append(f"top_k {trial.top_k} outside {_describe_dim(ss.top_k)}")
        if not ss.hybrid_alpha.contains(trial.hybrid_alpha):
            violations.append(
                f"hybrid_alpha {trial.hybrid_alpha} outside {_describe_dim(ss.hybrid_alpha)}"
            )
        if trial.reranker not in ss.reranker.models:
            violations.append(f"reranker '{trial.reranker}' not in {ss.reranker.models}")
        if not ss.reranker.top_n.contains(trial.reranker_top_n):
            violations.append(
                f"reranker_top_n {trial.reranker_top_n} outside {_describe_dim(ss.reranker.top_n)}"
            )
        if trial.reranker != "none" and trial.reranker_top_n > trial.top_k:
            violations.append(f"reranker_top_n ({trial.reranker_top_n}) must be <= top_k ({trial.top_k})")
        if trial.query_expansion not in ss.query_expansion:
            violations.append(f"query_expansion '{trial.query_expansion}' not in {ss.query_expansion}")
        if trial.bm25_vector_fusion not in ss.bm25_vector_fusion:
            violations.append(
                f"bm25_vector_fusion '{trial.bm25_vector_fusion}' not in {ss.bm25_vector_fusion}"
            )
        if trial.long_context_reorder not in ss.long_context_reorder:
            violations.append(
                f"long_context_reorder {trial.long_context_reorder} not in {ss.long_context_reorder}"
            )
        if trial.passage_compressor not in ss.passage_compressor:
            violations.append(
                f"passage_compressor '{trial.passage_compressor}' not in {ss.passage_compressor}"
            )

        # --- Generation checks ---
        if trial.generator_llm not in ss.llm_models.generator:
            violations.append(f"generator_llm '{trial.generator_llm}' not in {ss.llm_models.generator}")
        if trial.compressor_llm is not None and trial.compressor_llm not in ss.llm_models.compressor:
            violations.append(f"compressor_llm '{trial.compressor_llm}' not in {ss.llm_models.compressor}")
        if trial.expander_llm is not None and trial.expander_llm not in ss.llm_models.expander:
            violations.append(f"expander_llm '{trial.expander_llm}' not in {ss.llm_models.expander}")
        if not ss.temperature.contains(trial.temperature):
            violations.append(f"temperature {trial.temperature} outside [{ss.temperature.min}, {ss.temperature.max}]")
        if trial.reasoning and not ss.is_reasoning_allowed(trial.generator_llm):
            violations.append(f"reasoning=true not allowed for generator_llm '{trial.generator_llm}'")

        # --- Graph retrieval checks ---
        if trial.index_type in GRAPH_INDEX_TYPES and ss.graph_retrieval is not None:
            gr = ss.graph_retrieval
            if trial.graph_query_mode not in gr.graph_query_modes:
                violations.append(f"graph_query_mode '{trial.graph_query_mode}' not in {gr.graph_query_modes}")
            if not gr.graph_top_k.contains(trial.graph_top_k):
                violations.append(
                    f"graph_top_k {trial.graph_top_k} outside {_describe_dim(gr.graph_top_k)}"
                )

        return violations

    @staticmethod
    def _fmt_range(r: NumericDim, label: str, dtype: str = "float", suffix: str = "") -> str:
        """Format a numeric dim, showing '(fixed)' when only one value is legal.

        DiscreteValues render as a literal enumeration ("one of [3, 5, 10]")
        so the agent's proposer sees the option set up front and emits a
        legal value on the first try rather than tripping the validator.
        """
        if _dim_is_fixed(r):
            v = _dim_min_value(r)
            val = int(v) if dtype == "integer" else v
            return f"  {label}{val} (fixed){suffix}"
        if isinstance(r, DiscreteValues):
            values = [int(v) for v in r.values] if dtype == "integer" else list(r.values)
            return f"  {label}one of {values}{suffix}"
        if dtype == "integer":
            return f"  {label}integer in [{int(r.min)}, {int(r.max)}]{suffix}"
        return f"  {label}float in [{r.min}, {r.max}]{suffix}"

    def to_agent_prompt(self) -> str:
        """Format the search space as a clear prompt for the agent.

        Renders two disjoint blocks — *tunable* parameters (what the proposer
        may move) and *fixed* parameters (auto-filled at parse time) — and an
        example YAML that enumerates ONLY the tunable fields. Pinned fields
        are removed from the emission surface entirely so the proposer cannot
        violate them; ``reasoning`` is suppressed by its own
        ``ss.reasoning`` gate (see below) rather than via the pinning path.
        """
        ss = self.search_space
        fmt = self._fmt_range
        pinned = ss.pinned_field_values()

        # Build per-section tunable entries: (field_name, rendered_line).
        index_entries: list[tuple[str, str]] = [
            ("chunking_strategy", f"  chunking_strategy: choose from {ss.chunking.strategies}"),
            (
                "chunk_token_size",
                fmt(ss.chunking.chunk_token_size, "chunk_token_size:  ", "integer", "  (in tokens, not characters)"),
            ),
            (
                "chunk_token_overlap",
                fmt(
                    ss.chunking.chunk_token_overlap,
                    "chunk_token_overlap: ",
                    "integer",
                    "  (must be < chunk_token_size)",
                ),
            ),
            ("embedding_model", f"  embedding_model:   choose from {ss.embedding_models}"),
            ("index_type", f"  index_type:        choose from {[t.value for t in ss.index_types]}"),
        ]
        retrieval_entries: list[tuple[str, str]] = [
            ("top_k", fmt(ss.top_k, "top_k:            ", "integer")),
            (
                "hybrid_alpha",
                fmt(
                    ss.hybrid_alpha,
                    "hybrid_alpha:     ",
                    "float",
                    "  (0=BM25 only, 1=vector only; only used for hybrid_bm25_vector with fusion='alpha')",
                ),
            ),
            (
                "bm25_vector_fusion",
                f"  bm25_vector_fusion: choose from {ss.bm25_vector_fusion}  "
                "(alpha=smooth score blend via hybrid_alpha; rrf=rank-based fusion robust "
                "to score-distribution mismatch between BM25 and vector; only used for "
                "hybrid_bm25_vector)",
            ),
            (
                "long_context_reorder",
                f"  long_context_reorder: choose from {ss.long_context_reorder}  "
                "(true=duplicate the top-scored passage at the end of the joined context "
                "(input order preserved) to mitigate the 'lost in the middle' attention "
                "degradation when top_k is large; no-op when passage_compressor != none)",
            ),
            (
                "passage_compressor",
                f"  passage_compressor: choose from {ss.passage_compressor}  "
                f"({_filter_mode_descriptions(ss.passage_compressor, _PASSAGE_COMPRESSOR_MODE_DESCRIPTIONS)})",
            ),
            ("reranker", f"  reranker:         choose from {ss.reranker.models}"),
            ("reranker_top_n", fmt(ss.reranker.top_n, "reranker_top_n:   ", "integer")),
            (
                "query_expansion",
                f"  query_expansion:  choose from {ss.query_expansion}  "
                f"({_filter_mode_descriptions(ss.query_expansion, _QUERY_EXPANSION_MODE_DESCRIPTIONS)})",
            ),
        ]
        active_compressor_modes = [v for v in ss.passage_compressor if v != "none"]
        active_expansion_modes = [v for v in ss.query_expansion if v != "none"]
        compressor_modes_str = "/".join(active_compressor_modes) or "tree_summarize/refine"
        expansion_modes_str = "/".join(active_expansion_modes) or "hyde/multi_query/query_decompose"
        generation_entries: list[tuple[str, str]] = [
            (
                "generator_llm",
                f"  generator_llm:    choose from {ss.llm_models.generator}  "
                "(LLM that produces the final answer; the one the user sees)",
            ),
            (
                "compressor_llm",
                f"  compressor_llm:   choose from {ss.llm_models.compressor} OR null when "
                f"passage_compressor='none' (LLM that runs {compressor_modes_str} — "
                "cheap-but-fluent picks fine; this stage rewards instruction-following)",
            ),
            (
                "expander_llm",
                f"  expander_llm:     choose from {ss.llm_models.expander} OR null when "
                f"query_expansion='none' (LLM that runs {expansion_modes_str} — "
                "cheap-but-fluent picks fine; this stage rewards diverse rewrites)",
            ),
            ("temperature", fmt(ss.temperature, "temperature:      ")),
        ]
        # ``reasoning`` is suppressed entirely when ``ss.reasoning`` is False;
        # otherwise it is rendered as tunable with per-model allowed/denied
        # caveats. The single ``ss.reasoning=True but no model supports it``
        # case still emits an informational line so the agent understands why
        # the field is absent from the example YAML.
        # Only generator-stage LLMs can use reasoning_effort (the reasoning
        # knob applies to the final-answer call), so the allowed/denied lists
        # are derived from ``ss.llm_models.generator`` only.
        reasoning_in_example = False
        if ss.reasoning:
            allowed = [m for m in ss.llm_models.generator if ss.is_reasoning_allowed(m)]
            denied = [m for m in ss.llm_models.generator if not ss.is_reasoning_allowed(m)]
            if allowed:
                reasoning_line = (
                    "  reasoning:        true or false "
                    f"(effort={ss.reasoning_effort} when enabled, applied to generator_llm only; "
                    f"allowed when generator_llm in: {allowed})"
                )
                if denied:
                    reasoning_line += f"\n                    NOT allowed when generator_llm in: {denied}"
                generation_entries.append(("reasoning", reasoning_line))
                reasoning_in_example = True
            else:
                generation_entries.append(
                    ("reasoning", "  reasoning:        false (no model in the search space supports reasoning_effort)")
                )
                reasoning_in_example = True

        graph_entries: list[tuple[str, str]] = []
        if ss.graph_retrieval is not None:
            gr = ss.graph_retrieval
            graph_entries = [
                ("graph_query_mode", f"  graph_query_mode:  choose from {gr.graph_query_modes}"),
                ("graph_top_k", fmt(gr.graph_top_k, "graph_top_k:       ", "integer")),
            ]

        def _tunable_only(entries: list[tuple[str, str]]) -> list[str]:
            return [line for field, line in entries if field not in pinned]

        index_lines = _tunable_only(index_entries)
        retrieval_lines = _tunable_only(retrieval_entries)
        generation_lines = _tunable_only(generation_entries)
        graph_lines = _tunable_only(graph_entries)

        any_tunable = bool(index_lines or retrieval_lines or generation_lines or graph_lines)

        lines: list[str] = []
        lines.append("### Tunable parameters (the search dimensions for this run)")
        lines.append("")
        if not any_tunable:
            lines.append("  (none — every parameter is fixed for this run; see 'Fixed values' below.)")
        if index_lines:
            lines.append("  # Index-building parameters:")
            lines.extend(index_lines)
            # The chunk-size/embedding-limit constraint is only relevant when
            # at least one of the two fields is tunable.
            if self.embedding_token_limits and ("embedding_model" not in pinned or "chunk_token_size" not in pinned):
                limits = ", ".join(f"{m}: {t}" for m, t in sorted(self.embedding_token_limits.items()))
                lines.append(
                    f"  # CONSTRAINT: chunk_token_size must not exceed the embedding model's token limit: {limits}"
                )
        if retrieval_lines:
            if index_lines:
                lines.append("")
            lines.append("  # Retrieval parameters:")
            lines.extend(retrieval_lines)
        if generation_lines:
            if index_lines or retrieval_lines:
                lines.append("")
            lines.append("  # Generation parameters:")
            lines.extend(generation_lines)
        if graph_lines:
            if index_lines or retrieval_lines or generation_lines:
                lines.append("")
            lines.append(
                "  # Graph retrieval parameters (only active when index_type is 'graph_only' or 'hybrid_graph_vector'):"
            )
            lines.extend(graph_lines)

        if pinned:
            lines.append("")
            lines.append("### Fixed values for this run (auto-filled at parse time — do NOT emit in your YAML)")
            field_order = [
                "chunking_strategy",
                "chunk_token_size",
                "chunk_token_overlap",
                "embedding_model",
                "index_type",
                "top_k",
                "hybrid_alpha",
                "bm25_vector_fusion",
                "long_context_reorder",
                "passage_compressor",
                "reranker",
                "reranker_top_n",
                "query_expansion",
                "generator_llm",
                "compressor_llm",
                "expander_llm",
                "temperature",
                "reasoning",
                "graph_query_mode",
                "graph_top_k",
            ]
            for field in field_order:
                if field not in pinned:
                    continue
                value = pinned[field]
                suffix = ""
                if field == "hybrid_alpha" and ss.hybrid_alpha_is_dead():
                    suffix = "  # dead — only used when index_type is hybrid_bm25_vector with fusion='alpha'"
                elif field == "bm25_vector_fusion" and ss.bm25_vector_fusion_is_dead():
                    suffix = "  # dead — only used when index_type is hybrid_bm25_vector"
                elif field == "long_context_reorder" and ss.long_context_reorder_is_dead():
                    suffix = "  # dead — no-op when passage_compressor != 'none' (compression collapses to one string)"
                elif field == "reranker_top_n" and ss.reranker_top_n_is_dead():
                    suffix = "  # dead — only used when reranker != 'none'"
                elif field == "compressor_llm" and ss.compressor_llm_is_dead():
                    suffix = "  # dead — only used when passage_compressor != 'none'"
                elif field == "expander_llm" and ss.expander_llm_is_dead():
                    suffix = "  # dead — only used when query_expansion != 'none'"
                rendered = "null" if value is None else "false" if value is False else "true" if value is True else value
                lines.append(f"  {field}: {rendered}{suffix}")

        # Example YAML — tunable fields only, TrialConfig field order.
        example_pairs: list[tuple[str, object]] = []
        if "chunking_strategy" not in pinned:
            example_pairs.append(("chunking_strategy", ss.chunking.strategies[0]))
        if "chunk_token_size" not in pinned:
            example_pairs.append(("chunk_token_size", int(_dim_min_value(ss.chunking.chunk_token_size))))
        if "chunk_token_overlap" not in pinned:
            example_pairs.append(("chunk_token_overlap", int(_dim_min_value(ss.chunking.chunk_token_overlap))))
        if "embedding_model" not in pinned:
            example_pairs.append(("embedding_model", ss.embedding_models[0]))
        if "index_type" not in pinned:
            example_pairs.append(("index_type", ss.index_types[0].value))
        if "top_k" not in pinned:
            example_pairs.append(("top_k", int(_dim_min_value(ss.top_k))))
        if "hybrid_alpha" not in pinned:
            # Lowest-legal value rather than a hard-coded 0.5 — for DiscreteValues
            # 0.5 may not be in the option set and would fail validation.
            example_pairs.append(("hybrid_alpha", float(_dim_min_value(ss.hybrid_alpha))))
        if "bm25_vector_fusion" not in pinned:
            example_pairs.append(("bm25_vector_fusion", ss.bm25_vector_fusion[0]))
        if "long_context_reorder" not in pinned:
            example_pairs.append(("long_context_reorder", ss.long_context_reorder[0]))
        if "passage_compressor" not in pinned:
            example_pairs.append(("passage_compressor", ss.passage_compressor[0]))
        if "reranker" not in pinned:
            example_pairs.append(("reranker", ss.reranker.models[0]))
        if "reranker_top_n" not in pinned:
            example_pairs.append(("reranker_top_n", int(_dim_min_value(ss.reranker.top_n))))
        if "query_expansion" not in pinned:
            example_pairs.append(("query_expansion", ss.query_expansion[0]))
        if "generator_llm" not in pinned:
            example_pairs.append(("generator_llm", ss.llm_models.generator[0]))
        # Per-stage LLMs must match the stage's example value: null when the
        # example picks "none" for the stage, else first LLM in that stage's pool.
        if "compressor_llm" not in pinned:
            example_compressor = None if ss.passage_compressor[0] == "none" else ss.llm_models.compressor[0]
            example_pairs.append(("compressor_llm", example_compressor))
        if "expander_llm" not in pinned:
            example_expander = None if ss.query_expansion[0] == "none" else ss.llm_models.expander[0]
            example_pairs.append(("expander_llm", example_expander))
        if "temperature" not in pinned:
            example_pairs.append(("temperature", ss.temperature.min))
        if reasoning_in_example:
            example_pairs.append(("reasoning", False))
        if ss.graph_retrieval is not None:
            gr = ss.graph_retrieval
            if "graph_query_mode" not in pinned:
                example_pairs.append(("graph_query_mode", gr.graph_query_modes[0]))
            if "graph_top_k" not in pinned:
                example_pairs.append(("graph_top_k", int(_dim_min_value(gr.graph_top_k))))

        lines.append("")
        lines.append("### Expected output format")
        if example_pairs:
            lines.append("Your YAML block MUST emit ONLY the tunable fields above:")
            lines.append("")
            lines.append("```yaml")
            for field, value in example_pairs:
                rendered = (
                    "null" if value is None
                    else "false" if value is False
                    else "true" if value is True
                    else value
                )
                lines.append(f"{field}: {rendered}")
            lines.append("```")
        else:
            lines.append(
                "Every TrialConfig field is fixed this run — emit no TrialConfig fields (only the `meta:` block):"
            )
            lines.append("")
            lines.append("```yaml")
            lines.append("# all TrialConfig fields are auto-filled; emit `meta:` only.")
            lines.append("```")

        return "\n".join(lines)


class OpenEndedQuestion(BaseModel):
    """A single open-ended short-answer question in the exam.

    The schema uses parallel lists (``source_chunk_ids``,
    ``source_doc_ids``, ``source_spans``, ``source_span_offsets``) so
    single-hop (one entry) and multi-hop (two or more entries) share the
    same type. ``reasoning_type`` records how the question reasons over
    its source chunks; ``QUESTION_TYPES`` defines the closed taxonomy.

    Scoring uses normalized EM against ``canonical_answer`` and
    ``answer_variants``, with an LLM judge fallback for synthesized
    answers (counts, comparatives, computed values).
    """

    id: str
    question: str
    canonical_answer: str
    answer_variants: list[str] = Field(default_factory=list)
    reasoning_type: Literal[
        "extraction",
        "definitional",
        "bridge",
        "comparison",
        "numeric",
    ]
    # Variable-length parallel lists. Length 1 for single-hop, 2+ for multi-hop.
    source_chunk_ids: list[str]
    source_doc_ids: list[str]
    source_spans: list[str]
    source_span_offsets: list[tuple[int, int] | None] = Field(default_factory=list)
    # Math verification — populated only for reasoning_type == "numeric".
    # ``formula`` is an arithmetic expression evaluated against
    # ``canonical_answer``.
    formula: str | None = None
    formula_kind: Literal["arithmetic"] | None = None
    cluster_id: int = 0
    # Correctness vector across the discrimination probes (ordered
    # weakest-first). Empty when the probe filter hasn't run.
    probe_outcomes: list[int] = Field(default_factory=list)
    # Variance of ``probe_outcomes``; 0.0 means uninformative.
    discrimination_entropy: float = 0.0

    @field_validator("answer_variants", mode="before")
    @classmethod
    def coerce_answer_variants(cls, v: str | list[str] | None) -> list[str]:
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [v]
        return [s.strip() for s in v if isinstance(s, str) and s.strip()]

    @model_validator(mode="after")
    def validate_parallel_lists(self) -> OpenEndedQuestion:
        if not self.source_chunk_ids:
            raise ValueError("source_chunk_ids must not be empty")
        if len(self.source_doc_ids) != len(self.source_chunk_ids):
            raise ValueError(
                f"source_doc_ids ({len(self.source_doc_ids)}) must align with "
                f"source_chunk_ids ({len(self.source_chunk_ids)})"
            )
        if len(self.source_spans) != len(self.source_chunk_ids):
            raise ValueError(
                f"source_spans ({len(self.source_spans)}) must align with "
                f"source_chunk_ids ({len(self.source_chunk_ids)})"
            )
        if not self.source_span_offsets:
            self.source_span_offsets = [None] * len(self.source_chunk_ids)
        elif len(self.source_span_offsets) != len(self.source_chunk_ids):
            raise ValueError(
                f"source_span_offsets ({len(self.source_span_offsets)}) must align with "
                f"source_chunk_ids ({len(self.source_chunk_ids)})"
            )
        if any(not s.strip() for s in self.source_spans):
            raise ValueError("all entries in source_spans must be non-empty")
        if not self.canonical_answer.strip():
            raise ValueError("canonical_answer must be non-empty")
        return self

    @property
    def num_hops(self) -> int:
        return len(self.source_spans)

    @property
    def is_multi_doc(self) -> bool:
        return len(set(self.source_doc_ids)) > 1

    @property
    def gold_answers(self) -> list[str]:
        """All acceptable answer surface forms (canonical first, then variants)."""
        return [self.canonical_answer, *self.answer_variants]
