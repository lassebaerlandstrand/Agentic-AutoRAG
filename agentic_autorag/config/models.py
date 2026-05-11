"""Pydantic models for Agentic AutoRAG configuration and data structures.

Concrete config models represent what the agent proposes (specific values).
Search space models represent what the YAML defines (ranges and option lists).
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

import litellm
from pydantic import BaseModel, Field, field_validator, model_validator

# Allowed difficulty tags assigned by the two-gate validator.
DIFFICULTY_TAGS = ("easy", "medium")


class IndexType(StrEnum):
    VECTOR_ONLY = "vector_only"
    HYBRID_BM25_VECTOR = "hybrid_bm25_vector"
    GRAPH_ONLY = "graph_only"
    HYBRID_GRAPH_VECTOR = "hybrid_graph_vector"


GRAPH_INDEX_TYPES: frozenset[IndexType] = frozenset({IndexType.GRAPH_ONLY, IndexType.HYBRID_GRAPH_VECTOR})
_GRAPH_TRIAL_FIELDS = frozenset({"graph_query_mode", "graph_top_k"})


def _validate_overlap_less_than_size(v: int, info) -> int:
    if "chunk_token_size" in info.data and v >= info.data["chunk_token_size"]:
        raise ValueError("chunk_token_overlap must be < chunk_token_size")
    return v


class NumericRange(BaseModel):
    """A min/max range for numeric parameters. The agent picks any value within."""

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
    reranker: str = "none"
    reranker_top_n: int = 5
    query_expansion: str = "none"
    llm_model: str
    temperature: float = 0.0
    reasoning: bool = False
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
    reranker: str = "none"
    reranker_top_n: int = 5
    query_expansion: str = "none"
    # Generation parameters
    llm_model: str
    temperature: float = 0.0
    reasoning: bool = False
    # Graph retrieval parameters (only active when index_type is graph-based)
    graph_query_mode: str = "hybrid"
    graph_top_k: int = 60

    @field_validator("chunk_token_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        return _validate_overlap_less_than_size(v, info)

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
            reranker=self.reranker,
            reranker_top_n=self.reranker_top_n,
            query_expansion=self.query_expansion,
            llm_model=self.llm_model,
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
    chunk_token_size: NumericRange = NumericRange(min=64, max=512)
    chunk_token_overlap: NumericRange = NumericRange(min=0, max=128)


class RerankerSearchSpace(BaseModel):
    """Allowed reranker models and top_n range."""

    models: list[str] = ["none"]
    top_n: NumericRange = NumericRange(min=3, max=10)


class GraphRetrievalSearchSpace(BaseModel):
    """Graph retrieval parameters the optimizer can tune.

    Only relevant when index_types includes graph_only or hybrid_graph_vector.
    """

    graph_query_modes: list[str] = ["local", "global", "hybrid"]
    graph_top_k: NumericRange = NumericRange(min=20, max=100)


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
    top_k: NumericRange = NumericRange(min=3, max=20)
    hybrid_alpha: NumericRange = NumericRange(min=0.0, max=1.0)
    reranker: RerankerSearchSpace = RerankerSearchSpace()
    query_expansion: list[str] = ["none"]
    # Generation parameters
    llm_models: list[str]
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
    # Pareto polish-phase budget knobs. The trial budget is split mechanically:
    # the last ``polish_fraction`` of trials become eligible for the cost-
    # reduction polish phase, gated on ``best_score >= polish_score_floor`` so
    # the agent never polishes a broken config. ``polish_score_tolerance`` is
    # the score band around the leader the agent is expected to hold during
    # polish moves. ``polish_fraction=0.0`` recovers pure score-only optimization.
    polish_fraction: float = Field(default=0.3, ge=0.0, le=1.0)
    polish_score_floor: float = Field(default=0.5, ge=0.0, le=1.0)
    polish_score_tolerance: float = Field(default=0.05, ge=0.0, le=1.0)


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

    # Populated at runtime from KnowledgeBase — not in YAML
    embedding_token_limits: dict[str, int] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def validate_llm_models(self) -> ProjectConfig:
        """Validate that every llm_model is callable by LiteLLM.

        Step 1: static catalog check (free, covers most models).
        Step 2: live probe via completion(max_tokens=1) for any model not in the catalog.
        Raises ValueError listing all models that fail both checks.
        """
        needs_probe: list[str] = []
        for model in self.search_space.llm_models:
            if model.startswith("hosted_vllm/"):
                continue  # Framework-managed; vLLM server isn't running at config time
            if model in litellm.model_cost:
                continue
            if "/" in model:
                provider, suffix = model.split("/", 1)
                provider_models = litellm.models_by_provider.get(provider)
                if provider_models is not None and (
                    suffix in provider_models or f"{provider}/{suffix}" in provider_models
                ):
                    continue
            needs_probe.append(model)

        if not needs_probe:
            return self

        failed: list[str] = []
        errors: list[str] = []
        for model in needs_probe:
            ok, err = _probe_model(model)
            if not ok:
                failed.append(model)
                errors.append(f"  {model}: {err}")

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
                f"[{ss.chunking.chunk_token_size.min}, {ss.chunking.chunk_token_size.max}]"
            )
        if not ss.chunking.chunk_token_overlap.contains(trial.chunk_token_overlap):
            violations.append(
                f"chunk_token_overlap {trial.chunk_token_overlap} outside "
                f"[{ss.chunking.chunk_token_overlap.min}, {ss.chunking.chunk_token_overlap.max}]"
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
            violations.append(f"top_k {trial.top_k} outside [{ss.top_k.min}, {ss.top_k.max}]")
        if not ss.hybrid_alpha.contains(trial.hybrid_alpha):
            violations.append(
                f"hybrid_alpha {trial.hybrid_alpha} outside [{ss.hybrid_alpha.min}, {ss.hybrid_alpha.max}]"
            )
        if trial.reranker not in ss.reranker.models:
            violations.append(f"reranker '{trial.reranker}' not in {ss.reranker.models}")
        if not ss.reranker.top_n.contains(trial.reranker_top_n):
            violations.append(
                f"reranker_top_n {trial.reranker_top_n} outside [{ss.reranker.top_n.min}, {ss.reranker.top_n.max}]"
            )
        if trial.reranker != "none" and trial.reranker_top_n > trial.top_k:
            violations.append(f"reranker_top_n ({trial.reranker_top_n}) must be <= top_k ({trial.top_k})")
        if trial.query_expansion not in ss.query_expansion:
            violations.append(f"query_expansion '{trial.query_expansion}' not in {ss.query_expansion}")

        # --- Generation checks ---
        if trial.llm_model not in ss.llm_models:
            violations.append(f"llm_model '{trial.llm_model}' not in {ss.llm_models}")
        if not ss.temperature.contains(trial.temperature):
            violations.append(f"temperature {trial.temperature} outside [{ss.temperature.min}, {ss.temperature.max}]")
        if trial.reasoning and not ss.is_reasoning_allowed(trial.llm_model):
            violations.append(f"reasoning=true not allowed for '{trial.llm_model}'")

        # --- Graph retrieval checks ---
        if trial.index_type in GRAPH_INDEX_TYPES and ss.graph_retrieval is not None:
            gr = ss.graph_retrieval
            if trial.graph_query_mode not in gr.graph_query_modes:
                violations.append(f"graph_query_mode '{trial.graph_query_mode}' not in {gr.graph_query_modes}")
            if not gr.graph_top_k.contains(trial.graph_top_k):
                violations.append(
                    f"graph_top_k {trial.graph_top_k} outside [{gr.graph_top_k.min}, {gr.graph_top_k.max}]"
                )

        return violations

    @staticmethod
    def _fmt_range(r: NumericRange, label: str, dtype: str = "float", suffix: str = "") -> str:
        """Format a numeric range, showing '(fixed)' when min == max."""
        if r.min == r.max:
            val = int(r.min) if dtype == "integer" else r.min
            return f"  {label}{val} (fixed){suffix}"
        if dtype == "integer":
            return f"  {label}integer in [{int(r.min)}, {int(r.max)}]{suffix}"
        return f"  {label}float in [{r.min}, {r.max}]{suffix}"

    def to_agent_prompt(self) -> str:
        """Format the search space as a clear prompt for the agent.

        Shows all tunable parameters and the exact flat TrialConfig YAML schema
        the agent must produce, so the LLM never has to guess field names.
        """
        lines: list[str] = []
        ss = self.search_space
        fmt = self._fmt_range

        lines.append("### Search space (parameters the optimizer can tune)")
        lines.append("")
        lines.append("  # Index-building parameters:")
        lines.append(f"  chunking_strategy: choose from {ss.chunking.strategies}")
        lines.append(
            fmt(ss.chunking.chunk_token_size, "chunk_token_size:  ", "integer", "  (in tokens, not characters)")
        )
        lines.append(
            fmt(
                ss.chunking.chunk_token_overlap,
                "chunk_token_overlap: ",
                "integer",
                "  (must be < chunk_token_size)",
            )
        )
        lines.append(f"  embedding_model:   choose from {ss.embedding_models}")
        if self.embedding_token_limits:
            limits = ", ".join(f"{m}: {t}" for m, t in sorted(self.embedding_token_limits.items()))
            lines.append(
                f"  # CONSTRAINT: chunk_token_size must not exceed the embedding model's token limit: {limits}"
            )
        lines.append(f"  index_type:        choose from {[t.value for t in ss.index_types]}")

        lines.append("")
        lines.append("  # Retrieval parameters:")
        lines.append(fmt(ss.top_k, "top_k:            ", "integer"))
        lines.append(
            fmt(
                ss.hybrid_alpha,
                "hybrid_alpha:     ",
                "float",
                "  (0=BM25 only, 1=vector only; only used for hybrid_bm25_vector)",
            )
        )
        lines.append(f"  reranker:         choose from {ss.reranker.models}")
        lines.append(fmt(ss.reranker.top_n, "reranker_top_n:   ", "integer"))
        lines.append(f"  query_expansion:  choose from {ss.query_expansion}")

        lines.append("")
        lines.append("  # Generation parameters:")
        lines.append(f"  llm_model:        choose from {ss.llm_models}")
        lines.append(fmt(ss.temperature, "temperature:      "))
        # ``reasoning`` is suppressed entirely when ``ss.reasoning`` is False —
        # the parameter is not tunable in this run, so listing per-model
        # allowed/denied splits only invites the proposer to try
        # ``reasoning: true`` configurations that the validator will reject.
        if ss.reasoning:
            allowed = [m for m in ss.llm_models if ss.is_reasoning_allowed(m)]
            denied = [m for m in ss.llm_models if not ss.is_reasoning_allowed(m)]
            if allowed:
                lines.append(
                    "  reasoning:        true or false "
                    f"(effort={ss.reasoning_effort} when enabled; allowed for: {allowed})"
                )
                if denied:
                    lines.append(f"                    NOT allowed for: {denied}")
            else:
                lines.append("  reasoning:        false (no model in the search space supports reasoning_effort)")

        # --- Graph parameters ---
        if ss.graph_retrieval is not None:
            gr = ss.graph_retrieval
            lines.append("")
            lines.append(
                "  # Graph retrieval parameters (only active when index_type is 'graph_only' or 'hybrid_graph_vector'):"
            )
            lines.append(f"  graph_query_mode:  choose from {gr.graph_query_modes}")
            lines.append(f"  graph_top_k:       integer in [{int(gr.graph_top_k.min)}, {int(gr.graph_top_k.max)}]")

        # --- Expected output format ---
        example_strategy = ss.chunking.strategies[0]
        example_chunk_size = int(ss.chunking.chunk_token_size.min)
        example_overlap = int(ss.chunking.chunk_token_overlap.min)
        example_embed = ss.embedding_models[0]
        example_index = ss.index_types[0].value
        example_topk = int(ss.top_k.min)
        example_reranker = ss.reranker.models[0]
        example_reranker_topn = int(ss.reranker.top_n.min)
        example_qe = ss.query_expansion[0]
        example_llm = ss.llm_models[0]
        example_temp = ss.temperature.min

        lines.append("")
        lines.append("### Expected output format")
        lines.append("Your YAML block MUST match this exact flat structure (all fields required):")
        lines.append("")
        lines.append("```yaml")
        lines.append(f"chunking_strategy: {example_strategy}")
        lines.append(f"chunk_token_size: {example_chunk_size}")
        lines.append(f"chunk_token_overlap: {example_overlap}")
        lines.append(f"embedding_model: {example_embed}")
        lines.append(f"index_type: {example_index}")
        lines.append(f"top_k: {example_topk}")
        lines.append("hybrid_alpha: 0.5")
        lines.append(f"reranker: {example_reranker}")
        lines.append(f"reranker_top_n: {example_reranker_topn}")
        lines.append(f"query_expansion: {example_qe}")
        lines.append(f"llm_model: {example_llm}")
        lines.append(f"temperature: {example_temp}")
        if ss.reasoning:
            lines.append("reasoning: false")

        if ss.graph_retrieval is not None:
            gr = ss.graph_retrieval
            lines.append(f"graph_query_mode: {gr.graph_query_modes[0]}")
            lines.append(f"graph_top_k: {int(gr.graph_top_k.min)}")

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
