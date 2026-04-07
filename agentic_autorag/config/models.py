"""Pydantic models for Agentic AutoRAG configuration and data structures.

Concrete config models represent what the agent proposes (specific values).
Search space models represent what the YAML defines (ranges and option lists).
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

import litellm
from pydantic import BaseModel, Field, field_validator, model_validator

MCQ_OPTIONS = 4
MCQ_OPTION_LABELS = ("A", "B", "C", "D")

_GRAPH_INDEX_TYPES = frozenset({"graph_only", "hybrid_graph_vector"})


class IndexType(StrEnum):
    VECTOR_ONLY = "vector_only"
    HYBRID_BM25_VECTOR = "hybrid_bm25_vector"
    GRAPH_ONLY = "graph_only"
    HYBRID_GRAPH_VECTOR = "hybrid_graph_vector"


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
        if "chunk_token_size" in info.data and v >= info.data["chunk_token_size"]:
            raise ValueError("chunk_token_overlap must be < chunk_token_size")
        return v

    def fingerprint(self) -> str:
        """Deterministic 12-char hash of structural parameters."""
        data = self.model_dump()
        data["index_type"] = data["index_type"].value if hasattr(data["index_type"], "value") else data["index_type"]
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


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
    llm_timeout_s: float = 80.0  # per-call timeout passed to litellm.acompletion
    # Graph retrieval parameters (only used when index_type is graph-based)
    graph_query_mode: str = "hybrid"
    graph_top_k: int = 60


class GraphBuildConfig(BaseModel):
    """Fixed graph build configuration — set once, outside the optimizer search space.

    These parameters control how LightRAG constructs the knowledge graph. Changing
    them requires deleting the persisted graph (working_dir) and rebuilding.
    """

    extraction_model: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_token_size: int | None = None
    chunk_overlap_token_size: int | None = None
    entity_types: list[str] | None = None
    # Concurrency: keep low to avoid exhausting API rate limits.
    max_parallel_insert: int = Field(default=2, ge=1)
    llm_model_max_async: int = Field(default=4, ge=1)
    # Retries with exponential back-off on transient errors (429, 503, etc.).
    llm_model_max_retries: int = Field(default=6, ge=0)


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
        if "chunk_token_size" in info.data and v >= info.data["chunk_token_size"]:
            raise ValueError("chunk_token_overlap must be < chunk_token_size")
        return v

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
        """Deterministic 12-char hash for vector index registry lookup.

        Only covers vector index parameters — the graph is stored separately
        in its own working_dir and is never keyed by this fingerprint.
        """
        return self.to_structural().fingerprint()


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

    ``llm_models`` accepts a mixed list of plain strings and dicts with
    per-model overrides::

        llm_models:
          - "anthropic/claude-sonnet-4-6"
          - model: "vertex_ai/gemini-2.5-flash"
            reasoning: true
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
    reasoning_overrides: dict[str, bool] = {}
    # Graph retrieval
    graph_retrieval: GraphRetrievalSearchSpace | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_llm_models(cls, data: dict) -> dict:
        """Extract per-model reasoning overrides from mixed string/dict list."""
        raw = data.get("llm_models")
        if not raw or not isinstance(raw, list):
            return data
        models: list[str] = []
        overrides: dict[str, bool] = dict(data.get("reasoning_overrides") or {})
        for item in raw:
            if isinstance(item, str):
                models.append(item)
            elif isinstance(item, dict):
                model_name = item["model"]
                models.append(model_name)
                if "reasoning" in item:
                    overrides[model_name] = item["reasoning"]
        data["llm_models"] = models
        data["reasoning_overrides"] = overrides
        return data

    def is_reasoning_allowed(self, model: str) -> bool:
        """Check whether reasoning can be enabled for a given model.

        Priority order:
        1. Per-model override (always honored, even overrides LiteLLM capability check)
        2. Known unsupported prefixes (ollama/)
        3. LiteLLM capability check — if LiteLLM says the model doesn't support
           reasoning_effort, auto-deny (prevents wasted API calls or errors)
        4. Global ``reasoning`` default
        """
        if model in self.reasoning_overrides:
            return self.reasoning_overrides[model]
        if model.startswith(_REASONING_UNSUPPORTED_PREFIXES):
            return False
        if self.reasoning:
            try:
                if not litellm.supports_reasoning(model=model):
                    return False
            except Exception:  # noqa: BLE001
                pass
        return self.reasoning


class ParsingConfig(BaseModel):
    """Document parsing configuration.

    These settings control how raw files are converted to text before
    chunking. Not part of the optimizer search space — set once per project.
    """

    parser: str = "docling"
    ocr: bool = True
    table_structure: bool = True


class ExaminerConfig(BaseModel):
    """Settings for the exam generator."""

    exam_size: int = 60
    initial_candidate_multiplier: float = Field(default=2.5, ge=1.0)
    max_backfill_rounds: int = Field(default=3, ge=0)
    probe_selection: bool = False
    detect_parametric_leaks: bool = True
    parametric_leak_trials: int = Field(default=3, ge=1, le=5)
    parametric_leak_model: str | None = None

    # Source fact verification
    source_fact_threshold: float = 0.65
    source_fact_substring_fallback: bool = True
    source_fact_min_length: int = Field(default=60, ge=1)
    source_fact_window_chunk_size: int = Field(default=300, ge=50)
    source_fact_window_chunk_overlap: int = Field(default=150, ge=0)

    # Document handling
    doc_split_word_threshold: int = Field(default=24_000, ge=1_000)
    doc_section_word_size: int = Field(default=6_000, ge=500)
    min_doc_words: int = Field(default=200, ge=0)

    # Oracle
    oracle_context_window_words: int = Field(default=300, ge=50)
    oracle_retry_with_full_doc: bool = True

    # Multi-question per document
    max_questions_per_doc: int = Field(default=3, ge=1)

    # Quality filters
    max_generation_retries: int = Field(default=5, ge=1, le=10)
    dedup_similarity_threshold: float = Field(default=0.90, ge=0.5, le=1.0)
    discriminator_removal_pct: float = Field(default=0.05, ge=0.0, le=0.5)
    retrieval_difficulty_top_k: int = Field(default=1, ge=1, le=5)

    # Difficulty-aware allocation
    difficulty_weighted_allocation: bool = True
    min_questions_per_cluster: int = Field(default=1, ge=0, le=5)

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @field_validator("source_fact_threshold")
    @classmethod
    def valid_threshold(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"source_fact_threshold must be in (0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def valid_source_fact_windows(self) -> ExaminerConfig:
        if self.source_fact_window_chunk_overlap >= self.source_fact_window_chunk_size:
            raise ValueError("source_fact_window_chunk_overlap must be smaller than source_fact_window_chunk_size")
        return self

    @model_validator(mode="after")
    def valid_doc_section_size(self) -> ExaminerConfig:
        if self.doc_section_word_size >= self.doc_split_word_threshold:
            raise ValueError("doc_section_word_size must be smaller than doc_split_word_threshold")
        return self


class AgentConfig(BaseModel):
    """Settings for the LLM agents."""

    optimizer_model: str = "gemini/gemini-3-flash-preview"
    examiner_model: str = "gemini/gemini-3-flash-preview"
    max_history_trials: int = 10
    concurrency: int = Field(default=10, ge=1)


class MetaConfig(BaseModel):
    """Project-level settings."""

    project_name: str = "my-rag-project"
    corpus_path: str = "./data/corpus/"
    corpus_description: str = ""
    output_dir: str = "./experiments/"
    max_trials: int = 30
    index_registry: bool = True


class ProjectConfig(BaseModel):
    """The full project configuration loaded from YAML.

    Contains the search space (tunable parameters) plus project-level settings
    for parsing, examiner, and agent that are fixed for the optimization run.
    """

    meta: MetaConfig = MetaConfig()
    parsing: ParsingConfig = ParsingConfig()
    search_space: SearchSpace
    graph: GraphBuildConfig | None = None
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
        uses_graph = any(it.value in _GRAPH_INDEX_TYPES for it in ss.index_types)

        if uses_graph and self.graph is None:
            raise ValueError(
                "search_space.index_types includes graph-based types "
                f"({[it.value for it in ss.index_types if it.value in _GRAPH_INDEX_TYPES]}) "
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
        return any(it.value in _GRAPH_INDEX_TYPES for it in self.search_space.index_types)

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
        if trial.index_type.value in _GRAPH_INDEX_TYPES and ss.graph_retrieval is not None:
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
        allowed = [m for m in ss.llm_models if ss.is_reasoning_allowed(m)]
        denied = [m for m in ss.llm_models if not ss.is_reasoning_allowed(m)]
        if allowed:
            lines.append(
                f"  reasoning:        true or false (effort={ss.reasoning_effort} when enabled; allowed for: {allowed})"
            )
        if denied:
            lines.append(f"                    NOT allowed for: {denied}")

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
        lines.append("reasoning: false")

        if ss.graph_retrieval is not None:
            gr = ss.graph_retrieval
            lines.append(f"graph_query_mode: {gr.graph_query_modes[0]}")
            lines.append(f"graph_top_k: {int(gr.graph_top_k.min)}")

        lines.append("```")

        return "\n".join(lines)


class MCQQuestion(BaseModel):
    """A single multiple-choice question in the exam.

    IRT parameters match Guinet et al. (ICML 2024, Appendix B.1):
    - discrimination: 1.0 (init), bounds [0.1, 1.5]
    - difficulty: 0.0 (init, solver clips to 0.01), bounds [0.01, 1.0]
    - guessing: 0.25 (= 1/4 for 4-option MCQ), bounds [0.2, 0.4]
    """

    id: str
    question: str
    options: dict[str, str]  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_answer: str  # "A", "B", "C", or "D"
    source_doc_ids: list[str]  # document(s) the question was generated from
    source_fact: str = ""  # exact passage from the document that answers the question
    bloom_level: str = ""  # Bloom's taxonomy level (Remember, Understand, Apply, Analyze, Evaluate)
    cluster_id: int
    difficulty: float = 0.0  # updated by post-hoc IRT (b_j)
    discrimination: float = 1.0  # updated by post-hoc IRT (a_j)
    guessing: float = 0.25  # updated by post-hoc IRT (g_j), initialized to 1/4

    @field_validator("source_doc_ids")
    @classmethod
    def non_empty_doc_ids(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("source_doc_ids must not be empty")
        return v

    @field_validator("options")
    @classmethod
    def exactly_four_options(cls, v: dict[str, str]) -> dict[str, str]:
        expected = set(MCQ_OPTION_LABELS)
        if set(v.keys()) != expected:
            raise ValueError(f"options must have exactly keys {expected}, got {set(v.keys())}")
        return v

    @field_validator("correct_answer")
    @classmethod
    def valid_answer_key(cls, v: str) -> str:
        if v not in MCQ_OPTION_LABELS:
            raise ValueError(f"correct_answer must be one of {MCQ_OPTION_LABELS}, got '{v}'")
        return v
