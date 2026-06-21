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
    "query_decompose": ("N self-contained sub-queries REPLACE the original — useful for explicit multi-hop questions"),
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
    bm25_vector_fusion: str = "alpha"
    long_context_reorder: bool = False
    passage_compressor: str = "none"
    reranker: str = "none"
    reranker_top_n: int = 5
    query_expansion: str = "none"
    # ``compressor_llm`` / ``expander_llm`` are None when their corresponding
    # stage is "none"; ``generator_llm`` is always set.
    compressor_llm: str | None = None
    expander_llm: str | None = None
    generator_llm: str
    temperature: float = 0.0
    reasoning: bool = False
    reasoning_effort: str = "medium"  # generator only
    llm_timeout_s: float = 100.0
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
        """Ensure worst-case retry budget is under LightRAG's worker kill
        window (``2 * default_llm_timeout``). Our retry loop holds LightRAG's
        semaphore for the full duration; exceeding the cap kills the worker
        mid-retry and loses observability over which attempt failed."""
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
    """vLLM server settings for auto-managed local model serving.

    When hosted_vllm/ models appear in the search space, Agentic AutoRAG
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
                    f"compressor_llm must be null when passage_compressor='none' (got {self.compressor_llm!r})"
                )
        else:
            if self.compressor_llm is None:
                raise ValueError(f"compressor_llm is required when passage_compressor={self.passage_compressor!r}")
        if self.query_expansion == "none":
            if self.expander_llm is not None:
                raise ValueError(f"expander_llm must be null when query_expansion='none' (got {self.expander_llm!r})")
        else:
            if self.expander_llm is None:
                raise ValueError(f"expander_llm is required when query_expansion={self.query_expansion!r}")
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

    model_config = ConfigDict(extra="forbid")

    strategies: list[str] = ["recursive"]
    chunk_token_size: NumericDim = DiscreteValues(values=[128, 256, 384, 512])
    chunk_token_overlap: NumericDim = DiscreteValues(values=[0, 32, 48, 64, 128])


class EmbeddingSearchSpace(BaseModel):
    """Allowed embedding models for the retrieval index."""

    model_config = ConfigDict(extra="forbid")

    models: list[str]

    @model_validator(mode="after")
    def non_empty(self) -> EmbeddingSearchSpace:
        if not self.models:
            raise ValueError("embedding.models must be non-empty")
        return self


class RetrievalSearchSpace(BaseModel):
    """Retrieval-pipeline knobs that aren't owned by a specific stage."""

    model_config = ConfigDict(extra="forbid")

    index_types: list[IndexType] = [IndexType.VECTOR_ONLY]
    top_k: NumericDim = NumericRange(min=3, max=20)
    hybrid_alpha: NumericDim = NumericRange(min=0.0, max=1.0)
    bm25_vector_fusion: list[str] = ["alpha"]
    long_context_reorder: list[bool] = [False]


class QueryExpansionSearchSpace(BaseModel):
    """Allowed query-expansion strategies and the LLM pool that powers them.

    ``strategies`` gates whether the stage runs; ``models`` is the pool drawn
    from when the chosen strategy is non-"none". When every strategy is
    "none" the pool is unused and may be empty.
    """

    model_config = ConfigDict(extra="forbid")

    strategies: list[str] = ["none"]
    models: list[str] = []

    @model_validator(mode="after")
    def pool_required_when_stage_runs(self) -> QueryExpansionSearchSpace:
        if any(s != "none" for s in self.strategies) and not self.models:
            raise ValueError(
                "query_expansion.models must be non-empty when query_expansion.strategies includes a non-'none' value"
            )
        return self


class RerankerSearchSpace(BaseModel):
    """Allowed reranker models and top_n range."""

    model_config = ConfigDict(extra="forbid")

    models: list[str] = ["none"]
    top_n: NumericDim = NumericRange(min=3, max=10)


class PassageCompressorSearchSpace(BaseModel):
    """Allowed passage-compressor strategies and the LLM pool that powers them.

    ``strategies`` gates whether the stage runs; ``models`` is the pool drawn
    from when the chosen strategy is non-"none". When every strategy is
    "none" the pool is unused and may be empty.
    """

    model_config = ConfigDict(extra="forbid")

    strategies: list[str] = ["none"]
    models: list[str] = []

    @model_validator(mode="after")
    def pool_required_when_stage_runs(self) -> PassageCompressorSearchSpace:
        if any(s != "none" for s in self.strategies) and not self.models:
            raise ValueError(
                "passage_compressor.models must be non-empty when "
                "passage_compressor.strategies includes a non-'none' value"
            )
        return self


class GeneratorSearchSpace(BaseModel):
    """Generator-stage knobs: LLM pool + reasoning toggles.

    ``temperature`` lives at ``SearchSpace.temperature`` because the engine
    applies the same temperature to every LLM call (generator, compressor,
    expander) and a per-stage value would be misleading. ``reasoning`` /
    ``reasoning_effort`` are generator-only — the engine only forwards
    ``reasoning_effort`` for the final-answer call.

    ``reasoning`` defaults to True (matching the pre-v3 default): the
    optimizer may toggle reasoning_effort on for trials whose generator
    model supports it. Set to False to pin reasoning off across the run.
    """

    model_config = ConfigDict(extra="forbid")

    models: list[str]
    reasoning: bool = True
    reasoning_effort: str = "medium"

    @model_validator(mode="after")
    def non_empty(self) -> GeneratorSearchSpace:
        if not self.models:
            raise ValueError("generator.models must be non-empty")
        return self


class GraphRetrievalSearchSpace(BaseModel):
    """Graph retrieval parameters the optimizer can tune.

    Only relevant when index_types includes graph_only or hybrid_graph_vector.
    """

    model_config = ConfigDict(extra="forbid")

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
        if provider_models is not None and (suffix in provider_models or f"{provider}/{suffix}" in provider_models):
            return True
    return False


class SearchSpace(BaseModel):
    """Search space: all parameters the optimizer can tune, grouped by stage.

    Each pipeline stage owns its block — ``embedding`` / ``retrieval`` /
    ``query_expansion`` / ``reranker`` / ``passage_compressor`` / ``generator`` —
    plus ``chunking`` and (optional) ``graph_retrieval``. ``temperature``
    sits at the top level because the engine applies the same value to every
    LLM call regardless of stage.

    Index-building params (chunking, embedding, retrieval.index_types) trigger
    re-indexing when changed. All others are swappable without rebuilding.
    """

    model_config = ConfigDict(extra="forbid")

    chunking: ChunkingSearchSpace = ChunkingSearchSpace()
    embedding: EmbeddingSearchSpace
    retrieval: RetrievalSearchSpace = RetrievalSearchSpace()
    query_expansion: QueryExpansionSearchSpace = QueryExpansionSearchSpace()
    reranker: RerankerSearchSpace = RerankerSearchSpace()
    passage_compressor: PassageCompressorSearchSpace = PassageCompressorSearchSpace()
    generator: GeneratorSearchSpace
    # Applied to every LLM call (generator, compressor, expander) by the
    # engine — see pipeline.generate(). Single shared knob, not per-stage.
    # Defaults to a fixed 1.0 because several current frontier models reject
    # any temperature other than 1.0; widen the range for models that allow it.
    temperature: NumericRange = NumericRange(min=1.0, max=1.0)
    graph_retrieval: GraphRetrievalSearchSpace | None = None

    def is_reasoning_allowed(self, model: str) -> bool:
        """Whether ``reasoning_effort`` can be toggled for ``model``.

        LiteLLM is the ground truth — if it says the model doesn't support
        ``reasoning_effort``, the parameter is silently dropped and there's
        no point in the optimizer toggling it. The ollama prefix is gated
        explicitly because no ollama model surfaces reasoning through
        LiteLLM today. On lookup failure (provider not in catalog) defer
        to the generator-stage ``reasoning`` flag.
        """
        if not self.generator.reasoning:
            return False
        if model.startswith(_REASONING_UNSUPPORTED_PREFIXES):
            return False
        try:
            return bool(litellm.supports_reasoning(model=model))
        except Exception:  # noqa: BLE001
            return self.generator.reasoning

    def all_llm_models(self) -> list[str]:
        """Deduplicated union of every LLM that may run in this search space.

        Order: generator pool → expander pool → compressor pool, first seen wins.
        Used for cache verification and the project-level LiteLLM probe.
        """
        seen: dict[str, None] = {}
        for stage_list in (
            self.generator.models,
            self.query_expansion.models,
            self.passage_compressor.models,
        ):
            for m in stage_list:
                seen.setdefault(m, None)
        return list(seen.keys())

    def hybrid_alpha_is_dead(self) -> bool:
        """``hybrid_alpha`` only affects the pipeline when a hybrid_bm25_vector
        index is reachable in this run AND alpha-blend fusion is reachable."""
        if not any(it == IndexType.HYBRID_BM25_VECTOR for it in self.retrieval.index_types):
            return True
        return "alpha" not in self.retrieval.bm25_vector_fusion

    def bm25_vector_fusion_is_dead(self) -> bool:
        """``bm25_vector_fusion`` only affects the pipeline when a
        hybrid_bm25_vector index is reachable in this run."""
        return not any(it == IndexType.HYBRID_BM25_VECTOR for it in self.retrieval.index_types)

    def long_context_reorder_is_dead(self) -> bool:
        """``long_context_reorder`` is a no-op when every passage-compressor
        choice collapses retrieval to a single string — there is nothing to
        reorder. Dead when ``"none"`` is not enumerated for
        ``passage_compressor.strategies``."""
        return "none" not in self.passage_compressor.strategies

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
        top_k_max = _dim_max_value(self.retrieval.top_k)
        if top_n_min > top_k_max:
            raise ValueError(
                f"reranker.top_n minimum ({top_n_min}) must be <= retrieval.top_k maximum "
                f"({top_k_max}); no legal (top_k, reranker_top_n) pair exists."
            )
        return self

    def compressor_llm_is_dead(self) -> bool:
        """``compressor_llm`` is unused when no compressor stage ever runs."""
        return all(c == "none" for c in self.passage_compressor.strategies)

    def expander_llm_is_dead(self) -> bool:
        """``expander_llm`` is unused when no query-expansion stage ever runs."""
        return all(qe == "none" for qe in self.query_expansion.strategies)

    def compressor_llm_is_derived(self) -> bool:
        """Compressor LLM is *derived* (not statically pinned) when the stage
        list mixes ``"none"`` with non-``"none"`` values and the LLM pool is
        size 1. In that case its value at trial-assembly depends on which
        ``passage_compressor`` the proposer picks: ``None`` for ``"none"``,
        else the lone pool entry. Resolved by ``ReasoningAgent._inject_pinned``.
        """
        strategies = self.passage_compressor.strategies
        has_none = "none" in strategies
        has_active = any(s != "none" for s in strategies)
        return has_none and has_active and len(self.passage_compressor.models) == 1

    def expander_llm_is_derived(self) -> bool:
        """Mirror of ``compressor_llm_is_derived`` for the query-expansion stage."""
        strategies = self.query_expansion.strategies
        has_none = "none" in strategies
        has_active = any(s != "none" for s in strategies)
        return has_none and has_active and len(self.query_expansion.models) == 1

    def active_levers(self) -> set[str]:
        """Field names whose runtime behavior is non-trivial in this search space.

        A lever is *active* when at least one trial path exercises it — either
        because the agent can choose a non-trivial value, or because the
        single pinned value is non-trivial (e.g. ``passage_compressor.
        strategies=["tree_summarize"]``). Pinning ≠ inactive. The agent
        benefits from guidance whenever a lever is active, even if the value
        is fixed.
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
        if not self.long_context_reorder_is_dead() and any(self.retrieval.long_context_reorder):
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

        ``compressor_llm`` / ``expander_llm`` are statically pinned only when
        either (a) the stage is fully dead — pinned ``None`` — or (b) the
        stage always runs (no ``"none"`` in strategies) AND the LLM pool has
        size 1. The mixed case (some ``"none"`` strategies + single-model pool)
        is resolved at injection time by ``ReasoningAgent._inject_pinned``
        based on the proposer's strategy choice; see ``compressor_llm_is_derived``.

        ``reasoning`` is pinned to ``False`` when the generator stage disables
        it (``ss.generator.reasoning=False``). The corner case where
        ``reasoning=True`` but no generator model supports ``reasoning_effort``
        is handled by the rendering path in ``to_agent_prompt``.
        """
        pinned: dict[str, object] = {}

        if _dim_is_fixed(self.chunking.chunk_token_size):
            pinned["chunk_token_size"] = int(_dim_min_value(self.chunking.chunk_token_size))
        if _dim_is_fixed(self.chunking.chunk_token_overlap):
            pinned["chunk_token_overlap"] = int(_dim_min_value(self.chunking.chunk_token_overlap))
        if _dim_is_fixed(self.retrieval.top_k):
            pinned["top_k"] = int(_dim_min_value(self.retrieval.top_k))
        if _dim_is_fixed(self.retrieval.hybrid_alpha):
            pinned["hybrid_alpha"] = float(_dim_min_value(self.retrieval.hybrid_alpha))
        if _dim_is_fixed(self.reranker.top_n):
            pinned["reranker_top_n"] = int(_dim_min_value(self.reranker.top_n))
        if self.temperature.min == self.temperature.max:
            pinned["temperature"] = float(self.temperature.min)
        if not self.generator.reasoning:
            pinned["reasoning"] = False

        if len(self.chunking.strategies) == 1:
            pinned["chunking_strategy"] = self.chunking.strategies[0]
        if len(self.embedding.models) == 1:
            pinned["embedding_model"] = self.embedding.models[0]
        if len(self.retrieval.index_types) == 1:
            pinned["index_type"] = self.retrieval.index_types[0].value
        if len(self.reranker.models) == 1:
            pinned["reranker"] = self.reranker.models[0]
        if len(self.query_expansion.strategies) == 1:
            pinned["query_expansion"] = self.query_expansion.strategies[0]
        if len(self.generator.models) == 1:
            pinned["generator_llm"] = self.generator.models[0]

        if self.compressor_llm_is_dead():
            pinned["compressor_llm"] = None
        elif not self.compressor_llm_is_derived() and len(self.passage_compressor.models) == 1:
            pinned["compressor_llm"] = self.passage_compressor.models[0]
        if self.expander_llm_is_dead():
            pinned["expander_llm"] = None
        elif not self.expander_llm_is_derived() and len(self.query_expansion.models) == 1:
            pinned["expander_llm"] = self.query_expansion.models[0]

        if len(self.retrieval.bm25_vector_fusion) == 1:
            pinned["bm25_vector_fusion"] = self.retrieval.bm25_vector_fusion[0]
        if len(self.retrieval.long_context_reorder) == 1:
            pinned["long_context_reorder"] = self.retrieval.long_context_reorder[0]
        if len(self.passage_compressor.strategies) == 1:
            pinned["passage_compressor"] = self.passage_compressor.strategies[0]

        if self.reranker_top_n_is_dead():
            pinned.setdefault("reranker_top_n", int(_dim_min_value(self.reranker.top_n)))
        if self.hybrid_alpha_is_dead():
            pinned.setdefault("hybrid_alpha", float(_dim_min_value(self.retrieval.hybrid_alpha)))
        if self.bm25_vector_fusion_is_dead():
            pinned.setdefault("bm25_vector_fusion", self.retrieval.bm25_vector_fusion[0])
        if self.long_context_reorder_is_dead():
            pinned.setdefault("long_context_reorder", self.retrieval.long_context_reorder[0])

        if self.graph_retrieval is not None:
            if len(self.graph_retrieval.graph_query_modes) == 1:
                pinned["graph_query_mode"] = self.graph_retrieval.graph_query_modes[0]
            if _dim_is_fixed(self.graph_retrieval.graph_top_k):
                pinned["graph_top_k"] = int(_dim_min_value(self.graph_retrieval.graph_top_k))

        return pinned


class ParsingConfig(BaseModel):
    """Document parsing configuration. Not in the search space.

    Near-duplicate knobs feed the corpus-cleaner, which emits metadata only —
    duplicates stay in the index for every trial.
    """

    parser: Literal["docling"] = "docling"
    ocr: bool = True
    table_structure: bool = True
    # Containment cutoff for near-duplicate detection (see corpus_cleaner.py).
    # 0.85 catches OCR-of-PDF duplicates; raise toward 1.0 for stricter
    # clustering. Set ``near_duplicate_detection_enabled=False`` to skip.
    near_duplicate_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    near_duplicate_detection_enabled: bool = True


# Default boilerplate sections excluded from chunk-pair indexing. The closed
# taxonomy lives in ``engine.section_classifier``; these are the string
# values that survive in YAML.
_DEFAULT_EXCLUDED_SECTION_TYPES: tuple[str, ...] = (
    "references",
    "acknowledgments",
    "author_info",
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
    }
)


# Reasoning-type taxonomy for the composition prompt. Each generated
# question is tagged with the type the LLM produced. The composer chooses
# the type per question based on what the chunks support — there is no
# pre-seed type sampling.
QUESTION_TYPES: tuple[str, ...] = (
    "extraction",
    "definitional",
    "numeric_single",
    "inference",
    "bridge",
    "comparison",
    "numeric",
)


class ExaminerConfig(BaseModel):
    """Settings for the open-ended exam generator.

    The generator chunks the corpus, samples anchor chunks weighted by
    text length, builds an adaptive neighborhood around each anchor
    (same-document siblings + cosine-similar cross-document chunks), and
    issues one composition LLM call per neighborhood. Each call emits
    as many high-quality questions as the chunks support; the composer
    chooses hop count per question and cites which neighborhood
    chunks it used. Downstream validators (span verification, oracle
    answerability, decomposability) and the 4-probe discrimination
    selector then narrow the pool to ``exam_size`` discriminating
    questions. All LLM-billed work scales with ``exam_size`` rather
    than corpus size.
    """

    exam_size: int = 80
    # Per-neighborhood the composer can emit multiple questions, so the
    # number of anchors needed for the target candidate pool is
    # ``exam_size * initial_question_multiplier / avg_questions_per_nh``.
    # Over-generate to absorb validator/probe rejections; raise per corpus
    # if the exam under-fills.
    initial_question_multiplier: float = Field(default=1.5, ge=1.0)
    # Probe-based discrimination filtering. When True, every candidate that
    # clears the oracle gate is run through 2-4 probes (search-space
    # extremes) and the exam is built from the most discriminating items.
    probe_selection: bool = True

    # Persist exam-generation analysis artifacts to ``<output_dir>/details/debug/``:
    # the per-composition-call log, per-span verification outcomes, and
    # multi-hop rejection audit. On by default — these are how the exam
    # generator is inspected and tuned — and kept out of the headline output
    # so ``recommended.yaml`` and friends stay uncluttered. Set False to skip
    # writing them entirely.
    save_debug_artifacts: bool = True

    # Sampling temperature for the composition LLM. Default 1.0 because
    # several frontier models require exactly that value; lower it on models
    # that allow flexibility for stricter rule-following.
    composition_temperature: float = Field(default=1.0, ge=0.0, le=2.0)

    # Neighborhood-builder thresholds. The neighborhood grows until
    # ``len(chunks) >= neighborhood_min_chunks`` OR ``sum(words) >=
    # neighborhood_min_words``, whichever first. Small-chunk corpora
    # (Wikipedia paragraphs ≈ 100 words) hit the chunk floor at ~12;
    # large-chunk corpora (academic papers ≈ 1000 words/chunk) hit the
    # word floor at ~5 chunks. Both produce ~comparable context budgets
    # for the composer.
    neighborhood_min_chunks: int = Field(default=12, ge=1)
    neighborhood_min_words: int = Field(default=5000, ge=0)
    # Mix between same-document siblings and cross-doc cosine-similar
    # additions. Normalized internally — the absolute values do not
    # need to sum to 1. Tune per corpus: HotpotQA is cross-doc-rich
    # (paragraphs are tiny single-topic snippets) → cross-doc-heavy;
    # unidoc-style paper corpora have rich within-paper multi-hop and
    # weak cross-paper bridges → same-doc-heavy.
    neighborhood_same_doc_weight: float = Field(default=0.8, ge=0.0)
    neighborhood_cross_doc_weight: float = Field(default=0.2, ge=0.0)

    # Source fact verification (verbatim with fuzzy snap-to-source for minor LLM drift).
    source_fact_verify_fuzzy_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    # Chunk-relevance matcher thresholds (shared with retrieval evaluator).
    chunk_relevance_min_overlap_chars: int = Field(default=50, ge=1)
    chunk_relevance_ngram_size: int = Field(default=5, ge=1, le=20)
    chunk_relevance_overlap_threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    chunk_relevance_min_run: int = Field(default=5, ge=1)

    # Per-chunk word budget the examiner's HybridChunker uses for merge/split
    # decisions. 1200 fits the composition LLM context comfortably and exceeds
    # the max_seq_length of every embedder shipped in the default search space
    # except small 512-token models — those silently truncate, which is the
    # documented tradeoff (pair selection sees the chunk lead; LLM sees the
    # full chunk during composition).
    max_chunk_words: int = Field(default=1_000, ge=200)
    # Docs shorter than this are skipped during exam generation.
    min_doc_words: int = Field(default=200, ge=0)

    # Section classifier — chunk-pair indexer SKIPS chunks whose heuristic
    # section label appears in this deny-list. Default drops references,
    # acknowledgments, and author_info (the typical academic-paper
    # boilerplate). The taxonomy is defined in
    # ``engine.section_classifier.SectionLabel``.
    excluded_section_types: list[str] = Field(default_factory=lambda: list(_DEFAULT_EXCLUDED_SECTION_TYPES))

    @field_validator("excluded_section_types")
    @classmethod
    def known_section_types(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - _VALID_SECTION_TYPES)
        if unknown:
            raise ValueError(
                f"excluded_section_types contains unknown labels: {unknown}. "
                f"Valid labels: {sorted(_VALID_SECTION_TYPES)}"
            )
        return v

    @model_validator(mode="after")
    def neighborhood_weights_nonzero(self) -> ExaminerConfig:
        if self.neighborhood_same_doc_weight + self.neighborhood_cross_doc_weight <= 0:
            raise ValueError("neighborhood_same_doc_weight + neighborhood_cross_doc_weight must be > 0")
        return self


class AgentConfig(BaseModel):
    """Settings for the LLM agents."""

    # Required and explicit — no defaults, so an accidental/implicit model can
    # never silently drive the optimizer, examiner, or oracle/judge. The judge
    # grades trial answers and gates oracle answerability; pick one at least as
    # strong as the models in the search space.
    optimizer_model: str
    examiner_model: str
    judge_model: str
    optimizer_reasoning_effort: Literal["low", "medium", "high"] | None = "medium"
    examiner_reasoning_effort: Literal["low", "medium", "high"] | None = "medium"
    concurrency: int = Field(default=10, ge=1)


class MetaConfig(BaseModel):
    """Project-level settings."""

    project_name: str = "my-rag-project"
    corpus_path: str = "./data/corpus/"
    corpus_description: str = ""
    output_dir: str = "./experiments/"
    max_trials: int = 30
    cache_max_gb: float = Field(default=5.0, gt=0.0)
    # When True the optimizer is two-objective (score↑, cost↓) and the agent
    # declares an ``explore`` / ``refine`` stance. When False it's single-
    # objective (score↑ only); cost is still recorded for post-hoc analysis.
    cost_aware: bool = True
    # When None, the failure-sample seed is derived from the trial number —
    # deterministic per trial, varying across trials. Set to fix it.
    failure_sample_seed: int | None = None
    # Word-count cap for the corpus pre-sampler. Bounds per-trial wall-clock
    # on large corpora; None disables trimming.
    corpus_word_budget: int | None = 2_000_000
    corpus_sample_seed: int = 42
    # Lookback window (trials) for the hypervolume-Δ shown in the Pareto state
    # card. Informational only — does not gate termination.
    hv_delta_window: int = Field(default=3, ge=1)


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
    agent: AgentConfig

    # Maps short names in the search space to LiteLLM model IDs. Simple form:
    # ``alias: "provider/deployment"``. Extended form:
    # ``alias: {model: ..., api_base: ..., api_key: ..., api_version: ...}``
    # for custom OpenAI-compatible endpoints. Omit when every model is
    # reachable by its canonical LiteLLM name.
    model_aliases: dict[str, str | dict[str, Any]] = Field(default_factory=dict)

    # Populated at runtime from KnowledgeBase — informational only. Consumed
    # by probe_selector to keep examiner anchor chunks within the embedder's
    # cap (otherwise retrieval truncates the chunk the examiner used to write
    # the question, breaking grounding). Trial configs may freely exceed
    # these caps — the embedder truncates and the score reflects the loss.
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

        Step 1: static catalog check (free). Step 2: live probe via
        ``completion(max_tokens=1)`` for any name that fails the static check;
        aliased names probe the resolved target.
        """
        needs_probe: list[tuple[str, str]] = []  # (display_name, target_to_probe)
        for model in self.search_space.all_llm_models():
            if model.startswith("hosted_vllm/"):
                continue  # Auto-managed; vLLM server isn't running at config time
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
                "The following search_space LLM models could not be called by LiteLLM:\n"
                + "\n".join(errors)
                + "\nCheck model names and ensure required API keys are set."
            )
        return self

    @model_validator(mode="after")
    def graph_consistency(self) -> ProjectConfig:
        """Enforce mutual consistency between the graph build config and search space."""
        ss = self.search_space
        uses_graph = any(it in GRAPH_INDEX_TYPES for it in ss.retrieval.index_types)

        if uses_graph and self.graph is None:
            raise ValueError(
                "search_space.retrieval.index_types includes graph-based types "
                f"({[it.value for it in ss.retrieval.index_types if it in GRAPH_INDEX_TYPES]}) "
                "but no 'graph:' build config is defined. Add a 'graph:' section to your YAML."
            )

        if not uses_graph and ss.graph_retrieval is not None:
            raise ValueError(
                "search_space.graph_retrieval is defined but no graph-based index types are in "
                "search_space.retrieval.index_types. Either add a graph index type or remove graph_retrieval."
            )

        return self

    def uses_graph(self) -> bool:
        """Return True if any graph-based index type is in the search space."""
        return any(it in GRAPH_INDEX_TYPES for it in self.search_space.retrieval.index_types)

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
                f"chunk_token_size {trial.chunk_token_size} outside {_describe_dim(ss.chunking.chunk_token_size)}"
            )
        if not ss.chunking.chunk_token_overlap.contains(trial.chunk_token_overlap):
            violations.append(
                f"chunk_token_overlap {trial.chunk_token_overlap} outside "
                f"{_describe_dim(ss.chunking.chunk_token_overlap)}"
            )
        if trial.embedding_model not in ss.embedding.models:
            violations.append(f"embedding_model '{trial.embedding_model}' not in {ss.embedding.models}")
        if trial.index_type not in ss.retrieval.index_types:
            violations.append(
                f"index_type '{trial.index_type.value}' not in {[t.value for t in ss.retrieval.index_types]}"
            )

        # --- Retrieval checks ---
        if not ss.retrieval.top_k.contains(trial.top_k):
            violations.append(f"top_k {trial.top_k} outside {_describe_dim(ss.retrieval.top_k)}")
        if not ss.retrieval.hybrid_alpha.contains(trial.hybrid_alpha):
            violations.append(f"hybrid_alpha {trial.hybrid_alpha} outside {_describe_dim(ss.retrieval.hybrid_alpha)}")
        if trial.reranker not in ss.reranker.models:
            violations.append(f"reranker '{trial.reranker}' not in {ss.reranker.models}")
        if not ss.reranker.top_n.contains(trial.reranker_top_n):
            violations.append(f"reranker_top_n {trial.reranker_top_n} outside {_describe_dim(ss.reranker.top_n)}")
        if trial.reranker != "none" and trial.reranker_top_n > trial.top_k:
            violations.append(f"reranker_top_n ({trial.reranker_top_n}) must be <= top_k ({trial.top_k})")
        if trial.query_expansion not in ss.query_expansion.strategies:
            violations.append(f"query_expansion '{trial.query_expansion}' not in {ss.query_expansion.strategies}")
        if trial.bm25_vector_fusion not in ss.retrieval.bm25_vector_fusion:
            violations.append(
                f"bm25_vector_fusion '{trial.bm25_vector_fusion}' not in {ss.retrieval.bm25_vector_fusion}"
            )
        if trial.long_context_reorder not in ss.retrieval.long_context_reorder:
            violations.append(
                f"long_context_reorder {trial.long_context_reorder} not in {ss.retrieval.long_context_reorder}"
            )
        if trial.passage_compressor not in ss.passage_compressor.strategies:
            violations.append(
                f"passage_compressor '{trial.passage_compressor}' not in {ss.passage_compressor.strategies}"
            )

        # --- Generation checks ---
        if trial.generator_llm not in ss.generator.models:
            violations.append(f"generator_llm '{trial.generator_llm}' not in {ss.generator.models}")
        if trial.compressor_llm is not None and trial.compressor_llm not in ss.passage_compressor.models:
            violations.append(f"compressor_llm '{trial.compressor_llm}' not in {ss.passage_compressor.models}")
        if trial.expander_llm is not None and trial.expander_llm not in ss.query_expansion.models:
            violations.append(f"expander_llm '{trial.expander_llm}' not in {ss.query_expansion.models}")
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
                violations.append(f"graph_top_k {trial.graph_top_k} outside {_describe_dim(gr.graph_top_k)}")

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

        Renders three disjoint blocks — *tunable* parameters (what the
        proposer may move), *derived* parameters (auto-resolved at trial
        assembly from the proposer's strategy choice — currently
        ``compressor_llm`` / ``expander_llm`` when their pool is size 1 and
        their stage strategy list mixes ``"none"`` with non-``"none"``), and
        *fixed* parameters (statically pinned at parse time) — plus an
        example YAML that enumerates ONLY the tunable fields.
        """
        ss = self.search_space
        fmt = self._fmt_range
        pinned = ss.pinned_field_values()
        compressor_derived = ss.compressor_llm_is_derived()
        expander_derived = ss.expander_llm_is_derived()

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
            ("embedding_model", f"  embedding_model:   choose from {ss.embedding.models}"),
            ("index_type", f"  index_type:        choose from {[t.value for t in ss.retrieval.index_types]}"),
        ]
        retrieval_entries: list[tuple[str, str]] = [
            ("top_k", fmt(ss.retrieval.top_k, "top_k:            ", "integer")),
            (
                "hybrid_alpha",
                fmt(
                    ss.retrieval.hybrid_alpha,
                    "hybrid_alpha:     ",
                    "float",
                    "  (0=BM25 only, 1=vector only; only used for hybrid_bm25_vector with fusion='alpha')",
                ),
            ),
            (
                "bm25_vector_fusion",
                f"  bm25_vector_fusion: choose from {ss.retrieval.bm25_vector_fusion}  "
                "(alpha=smooth score blend via hybrid_alpha; rrf=rank-based fusion robust "
                "to score-distribution mismatch between BM25 and vector; only used for "
                "hybrid_bm25_vector)",
            ),
            (
                "long_context_reorder",
                f"  long_context_reorder: choose from {ss.retrieval.long_context_reorder}  "
                "(true=duplicate the top-scored passage at the end of the joined context "
                "(input order preserved) to mitigate the 'lost in the middle' attention "
                "degradation when top_k is large; no-op when passage_compressor != none)",
            ),
            (
                "passage_compressor",
                f"  passage_compressor: choose from {ss.passage_compressor.strategies}  "
                f"({
                    _filter_mode_descriptions(
                        ss.passage_compressor.strategies,
                        _PASSAGE_COMPRESSOR_MODE_DESCRIPTIONS,
                    )
                })",
            ),
            ("reranker", f"  reranker:         choose from {ss.reranker.models}"),
            ("reranker_top_n", fmt(ss.reranker.top_n, "reranker_top_n:   ", "integer")),
            (
                "query_expansion",
                f"  query_expansion:  choose from {ss.query_expansion.strategies}  "
                f"({_filter_mode_descriptions(ss.query_expansion.strategies, _QUERY_EXPANSION_MODE_DESCRIPTIONS)})",
            ),
        ]
        active_compressor_modes = [v for v in ss.passage_compressor.strategies if v != "none"]
        active_expansion_modes = [v for v in ss.query_expansion.strategies if v != "none"]
        compressor_modes_str = "/".join(active_compressor_modes) or "tree_summarize/refine"
        expansion_modes_str = "/".join(active_expansion_modes) or "hyde/multi_query/query_decompose"
        generation_entries: list[tuple[str, str]] = [
            (
                "generator_llm",
                f"  generator_llm:    choose from {ss.generator.models}  "
                "(LLM that produces the final answer; the one the user sees)",
            ),
            (
                "compressor_llm",
                f"  compressor_llm:   choose from {ss.passage_compressor.models} OR null when "
                f"passage_compressor='none' (LLM that runs {compressor_modes_str} — "
                "cheap-but-fluent picks fine; this stage rewards instruction-following)",
            ),
            (
                "expander_llm",
                f"  expander_llm:     choose from {ss.query_expansion.models} OR null when "
                f"query_expansion='none' (LLM that runs {expansion_modes_str} — "
                "cheap-but-fluent picks fine; this stage rewards diverse rewrites)",
            ),
            ("temperature", fmt(ss.temperature, "temperature:      ")),
        ]
        # ``reasoning`` is suppressed entirely when ``ss.generator.reasoning``
        # is False; otherwise it is rendered as tunable with per-model
        # allowed/denied caveats. The corner case where
        # ``generator.reasoning=True`` but no model supports ``reasoning_effort``
        # still emits an informational line so the agent understands why the
        # field is absent from the example YAML.
        reasoning_in_example = False
        if ss.generator.reasoning:
            allowed = [m for m in ss.generator.models if ss.is_reasoning_allowed(m)]
            denied = [m for m in ss.generator.models if not ss.is_reasoning_allowed(m)]
            if allowed:
                reasoning_line = (
                    "  reasoning:        true or false "
                    f"(effort={ss.generator.reasoning_effort} when enabled, applied to generator_llm only; "
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

        derived_fields = set()
        if compressor_derived:
            derived_fields.add("compressor_llm")
        if expander_derived:
            derived_fields.add("expander_llm")

        def _tunable_only(entries: list[tuple[str, str]]) -> list[str]:
            return [line for field, line in entries if field not in pinned and field not in derived_fields]

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
            # Surface embedder caps as informational context — chunks larger
            # than the cap are truncated by the embedder (allowed, but hurts
            # retrieval quality). Only relevant when either field is tunable.
            if self.embedding_token_limits and ("embedding_model" not in pinned or "chunk_token_size" not in pinned):
                limits = ", ".join(f"{m}: {t}" for m, t in sorted(self.embedding_token_limits.items()))
                lines.append(
                    "  # NOTE: embedding model max input tokens (chunks larger are truncated "
                    f"by the embedder, which typically hurts retrieval): {limits}"
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

        if derived_fields:
            lines.append("")
            lines.append("### Derived values (auto-resolved at trial assembly — do NOT emit in your YAML)")
            if compressor_derived:
                model = ss.passage_compressor.models[0]
                lines.append(f"  compressor_llm: null when passage_compressor='none', else {model}")
            if expander_derived:
                model = ss.query_expansion.models[0]
                lines.append(f"  expander_llm:   null when query_expansion='none', else {model}")

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
                rendered = (
                    "null" if value is None else "false" if value is False else "true" if value is True else value
                )
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
            example_pairs.append(("embedding_model", ss.embedding.models[0]))
        if "index_type" not in pinned:
            example_pairs.append(("index_type", ss.retrieval.index_types[0].value))
        if "top_k" not in pinned:
            example_pairs.append(("top_k", int(_dim_min_value(ss.retrieval.top_k))))
        if "hybrid_alpha" not in pinned:
            # Lowest-legal value rather than a hard-coded 0.5 — for DiscreteValues
            # 0.5 may not be in the option set and would fail validation.
            example_pairs.append(("hybrid_alpha", float(_dim_min_value(ss.retrieval.hybrid_alpha))))
        if "bm25_vector_fusion" not in pinned:
            example_pairs.append(("bm25_vector_fusion", ss.retrieval.bm25_vector_fusion[0]))
        if "long_context_reorder" not in pinned:
            example_pairs.append(("long_context_reorder", ss.retrieval.long_context_reorder[0]))
        if "passage_compressor" not in pinned:
            example_pairs.append(("passage_compressor", ss.passage_compressor.strategies[0]))
        if "reranker" not in pinned:
            example_pairs.append(("reranker", ss.reranker.models[0]))
        if "reranker_top_n" not in pinned:
            example_pairs.append(("reranker_top_n", int(_dim_min_value(ss.reranker.top_n))))
        if "query_expansion" not in pinned:
            example_pairs.append(("query_expansion", ss.query_expansion.strategies[0]))
        if "generator_llm" not in pinned:
            example_pairs.append(("generator_llm", ss.generator.models[0]))
        # Per-stage LLMs: omitted from the example when derived (resolved
        # post-emission by injection). When still tunable (multi-LLM pool),
        # match the stage's example: null when the stage example is "none",
        # else first LLM in that stage's pool.
        if "compressor_llm" not in pinned and "compressor_llm" not in derived_fields:
            example_compressor = (
                None if ss.passage_compressor.strategies[0] == "none" else ss.passage_compressor.models[0]
            )
            example_pairs.append(("compressor_llm", example_compressor))
        if "expander_llm" not in pinned and "expander_llm" not in derived_fields:
            example_expander = None if ss.query_expansion.strategies[0] == "none" else ss.query_expansion.models[0]
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
                    "null" if value is None else "false" if value is False else "true" if value is True else value
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
        "numeric_single",
        "inference",
        "bridge",
        "comparison",
        "numeric",
    ]
    # Variable-length parallel lists. Length 1 for single-hop, 2+ for multi-hop.
    source_chunk_ids: list[str]
    source_doc_ids: list[str]
    source_spans: list[str]
    source_span_offsets: list[tuple[int, int] | None] = Field(default_factory=list)
    # Math verification — populated for reasoning_type in {"numeric", "numeric_single"}.
    # ``formula`` is an arithmetic expression evaluated against
    # ``canonical_answer``.
    formula: str | None = None
    formula_kind: Literal["arithmetic"] | None = None
    # Correctness vector across the discrimination probes (ordered
    # weakest-first). Empty when the probe filter hasn't run.
    probe_outcomes: list[int] = Field(default_factory=list)

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
