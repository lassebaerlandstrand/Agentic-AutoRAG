"""Pydantic models for Agentic AutoRAG configuration and data structures.

Concrete config models represent what the agent proposes (specific values).
Search space models represent what the YAML defines (ranges and option lists).
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

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
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_type: IndexType = IndexType.VECTOR_ONLY

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be < chunk_size")
        return v


class RuntimeConfig(BaseModel):
    """Internal engine type: retrieval/generation parameters passed to RAGPipeline."""

    top_k: int = 5
    hybrid_alpha: float = 0.5
    reranker: str = "none"
    reranker_top_n: int = 5
    query_expansion: str = "none"
    llm_model: str
    temperature: float = 0.0
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
    chunk_size: int = 512
    chunk_overlap: int = 64
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
    # Graph retrieval parameters (only active when index_type is graph-based)
    graph_query_mode: str = "hybrid"
    graph_top_k: int = 60

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be < chunk_size")
        return v

    def to_structural(self) -> StructuralConfig:
        """Extract index-building parameters as an internal StructuralConfig."""
        return StructuralConfig(
            chunking_strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedding_model=self.embedding_model,
            index_type=self.index_type,
        )

    def to_runtime(self) -> RuntimeConfig:
        """Extract retrieval/generation parameters as an internal RuntimeConfig."""
        return RuntimeConfig(
            top_k=self.top_k,
            hybrid_alpha=self.hybrid_alpha,
            reranker=self.reranker,
            reranker_top_n=self.reranker_top_n,
            query_expansion=self.query_expansion,
            llm_model=self.llm_model,
            temperature=self.temperature,
            graph_query_mode=self.graph_query_mode,
            graph_top_k=self.graph_top_k,
        )

    def structural_fingerprint(self) -> str:
        """Deterministic 12-char hash for vector index registry lookup.

        Only covers vector index parameters — the graph is stored separately
        in its own working_dir and is never keyed by this fingerprint.
        """
        data = self.to_structural().model_dump()
        data["index_type"] = data["index_type"].value if hasattr(data["index_type"], "value") else data["index_type"]
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


class ChunkingSearchSpace(BaseModel):
    """Allowed chunking strategies and parameter ranges."""

    strategies: list[str] = ["recursive"]
    chunk_size: NumericRange = NumericRange(min=256, max=2048)
    chunk_overlap: NumericRange = NumericRange(min=0, max=256)


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
    # Graph retrieval
    graph_retrieval: GraphRetrievalSearchSpace | None = None


class ParsingConfig(BaseModel):
    """Document parsing configuration.

    These settings control how raw files are converted to text before
    chunking. Not part of the optimizer search space — set once per project.
    """

    parser: str = "docling"
    ocr: bool = True
    table_structure: bool = True


class ExaminerConfig(BaseModel):
    """Settings for the adaptive examiner."""

    exam_size: int = 50
    diversity_clusters: int | None = None  # None = auto (sqrt of chunk count, capped at exam_size)
    irt_discrimination_threshold: float = 0.3
    refresh_interval_trials: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


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
        if not ss.chunking.chunk_size.contains(trial.chunk_size):
            violations.append(
                f"chunk_size {trial.chunk_size} outside [{ss.chunking.chunk_size.min}, {ss.chunking.chunk_size.max}]"
            )
        if not ss.chunking.chunk_overlap.contains(trial.chunk_overlap):
            violations.append(
                f"chunk_overlap {trial.chunk_overlap} outside "
                f"[{ss.chunking.chunk_overlap.min}, {ss.chunking.chunk_overlap.max}]"
            )
        if trial.embedding_model not in ss.embedding_models:
            violations.append(f"embedding_model '{trial.embedding_model}' not in {ss.embedding_models}")
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
        if trial.query_expansion not in ss.query_expansion:
            violations.append(f"query_expansion '{trial.query_expansion}' not in {ss.query_expansion}")

        # --- Generation checks ---
        if trial.llm_model not in ss.llm_models:
            violations.append(f"llm_model '{trial.llm_model}' not in {ss.llm_models}")
        if not ss.temperature.contains(trial.temperature):
            violations.append(f"temperature {trial.temperature} outside [{ss.temperature.min}, {ss.temperature.max}]")

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

    def to_agent_prompt(self) -> str:
        """Format the search space as a clear prompt for the agent.

        Shows all tunable parameters and the exact flat TrialConfig YAML schema
        the agent must produce, so the LLM never has to guess field names.
        """
        lines: list[str] = []
        ss = self.search_space

        lines.append("### Search space (parameters the optimizer can tune)")
        lines.append("")
        lines.append("  # Index-building parameters:")
        lines.append(f"  chunking_strategy: choose from {ss.chunking.strategies}")
        lines.append(
            f"  chunk_size:        integer in [{int(ss.chunking.chunk_size.min)}, {int(ss.chunking.chunk_size.max)}]"
        )
        overlap_min = int(ss.chunking.chunk_overlap.min)
        overlap_max = int(ss.chunking.chunk_overlap.max)
        lines.append(f"  chunk_overlap:     integer in [{overlap_min}, {overlap_max}]  (must be < chunk_size)")
        lines.append(f"  embedding_model:   choose from {ss.embedding_models}")
        lines.append(f"  index_type:        choose from {[t.value for t in ss.index_types]}")

        lines.append("")
        lines.append("  # Retrieval parameters:")
        lines.append(f"  top_k:            integer in [{int(ss.top_k.min)}, {int(ss.top_k.max)}]")
        lines.append(
            f"  hybrid_alpha:     float in [{ss.hybrid_alpha.min}, {ss.hybrid_alpha.max}]"
            "  (0=BM25 only, 1=vector only; only used for hybrid_bm25_vector)"
        )
        lines.append(f"  reranker:         choose from {ss.reranker.models}")
        lines.append(f"  reranker_top_n:   integer in [{int(ss.reranker.top_n.min)}, {int(ss.reranker.top_n.max)}]")
        lines.append(f"  query_expansion:  choose from {ss.query_expansion}")

        lines.append("")
        lines.append("  # Generation parameters:")
        lines.append(f"  llm_model:        choose from {ss.llm_models}")
        lines.append(f"  temperature:      float in [{ss.temperature.min}, {ss.temperature.max}]")

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
        example_chunk_size = int(ss.chunking.chunk_size.min)
        example_overlap = int(ss.chunking.chunk_overlap.min)
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
        lines.append(f"chunk_size: {example_chunk_size}")
        lines.append(f"chunk_overlap: {example_overlap}")
        lines.append(f"embedding_model: {example_embed}")
        lines.append(f"index_type: {example_index}")
        lines.append(f"top_k: {example_topk}")
        lines.append("hybrid_alpha: 0.5")
        lines.append(f"reranker: {example_reranker}")
        lines.append(f"reranker_top_n: {example_reranker_topn}")
        lines.append(f"query_expansion: {example_qe}")
        lines.append(f"llm_model: {example_llm}")
        lines.append(f"temperature: {example_temp}")

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
    source_chunk_id: str
    cluster_id: int
    difficulty: float = 0.0  # updated by IRT (b_j)
    discrimination: float = 1.0  # updated by IRT (a_j)
    guessing: float = 0.25  # updated by IRT (g_j), initialized to 1/4

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
