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


class IndexType(StrEnum):
    VECTOR_ONLY = "vector_only"
    HYBRID_BM25_VECTOR = "hybrid_bm25_vector"
    GRAPH = "graph"
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


class GraphConfig(BaseModel):
    """Graph-specific parameters, only relevant for graph index types."""

    graph_backend: str = "networkx"
    traversal_depth: int = 2
    entity_types: list[str] | None = None


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
    # Graph (optional)
    graph: GraphConfig | None = None

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be < chunk_size")
        return v

    @model_validator(mode="after")
    def graph_required_for_graph_index(self) -> TrialConfig:
        if self.index_type in (IndexType.GRAPH, IndexType.HYBRID_GRAPH_VECTOR) and self.graph is None:
            raise ValueError("graph config required when index_type uses graph")
        return self

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
        )

    def structural_fingerprint(self) -> str:
        """Deterministic hash for index registry lookup.

        Includes index-building params + graph_backend/entity_types.
        """
        data = self.to_structural().model_dump()
        # Serialize index_type enum as its value
        data["index_type"] = data["index_type"].value if hasattr(data["index_type"], "value") else data["index_type"]
        if self.graph:
            data["graph_backend"] = self.graph.graph_backend
            if self.graph.entity_types is not None:
                data["graph_entity_types"] = sorted(self.graph.entity_types)
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


class GraphSearchSpace(BaseModel):
    """Graph-specific search space parameters."""

    graph_backend: str = "networkx"
    traversal_depth: NumericRange = NumericRange(min=1, max=3)
    entity_types: list[str] | None = None


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
    graph: GraphSearchSpace | None = None
    examiner: ExaminerConfig = ExaminerConfig()
    agent: AgentConfig = AgentConfig()

    def validate_trial(self, trial: TrialConfig) -> list[str]:
        """Check whether a proposed trial config falls within the search space.

        Returns a list of violation messages (empty = valid).
        """
        violations: list[str] = []
        ss = self.search_space

        # --- Index-building checks ---
        if trial.chunking_strategy not in ss.chunking.strategies:
            violations.append(
                f"chunking_strategy '{trial.chunking_strategy}' not in {ss.chunking.strategies}"
            )
        if not ss.chunking.chunk_size.contains(trial.chunk_size):
            violations.append(
                f"chunk_size {trial.chunk_size} outside "
                f"[{ss.chunking.chunk_size.min}, {ss.chunking.chunk_size.max}]"
            )
        if not ss.chunking.chunk_overlap.contains(trial.chunk_overlap):
            violations.append(
                f"chunk_overlap {trial.chunk_overlap} outside "
                f"[{ss.chunking.chunk_overlap.min}, {ss.chunking.chunk_overlap.max}]"
            )
        if trial.embedding_model not in ss.embedding_models:
            violations.append(f"embedding_model '{trial.embedding_model}' not in {ss.embedding_models}")
        if trial.index_type not in ss.index_types:
            violations.append(
                f"index_type '{trial.index_type.value}' not in {[t.value for t in ss.index_types]}"
            )

        # --- Retrieval checks ---
        if not ss.top_k.contains(trial.top_k):
            violations.append(f"top_k {trial.top_k} outside [{ss.top_k.min}, {ss.top_k.max}]")
        if not ss.hybrid_alpha.contains(trial.hybrid_alpha):
            violations.append(
                f"hybrid_alpha {trial.hybrid_alpha} outside "
                f"[{ss.hybrid_alpha.min}, {ss.hybrid_alpha.max}]"
            )
        if trial.reranker not in ss.reranker.models:
            violations.append(f"reranker '{trial.reranker}' not in {ss.reranker.models}")
        if not ss.reranker.top_n.contains(trial.reranker_top_n):
            violations.append(
                f"reranker_top_n {trial.reranker_top_n} outside "
                f"[{ss.reranker.top_n.min}, {ss.reranker.top_n.max}]"
            )
        if trial.query_expansion not in ss.query_expansion:
            violations.append(f"query_expansion '{trial.query_expansion}' not in {ss.query_expansion}")

        # --- Generation checks ---
        if trial.llm_model not in ss.llm_models:
            violations.append(f"llm_model '{trial.llm_model}' not in {ss.llm_models}")
        if not ss.temperature.contains(trial.temperature):
            violations.append(
                f"temperature {trial.temperature} outside "
                f"[{ss.temperature.min}, {ss.temperature.max}]"
            )

        # --- Graph checks ---
        if trial.graph and self.graph and not self.graph.traversal_depth.contains(trial.graph.traversal_depth):
            violations.append(
                f"traversal_depth {trial.graph.traversal_depth} outside "
                f"[{self.graph.traversal_depth.min}, {self.graph.traversal_depth.max}]"
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
        lines.append(
            f"  chunk_overlap:     integer in [{int(ss.chunking.chunk_overlap.min)}, {int(ss.chunking.chunk_overlap.max)}]"
            "  (must be < chunk_size)"
        )
        lines.append(f"  embedding_model:   choose from {ss.embedding_models}")
        lines.append(f"  index_type:        choose from {[t.value for t in ss.index_types]}")

        lines.append("")
        lines.append("  # Retrieval parameters:")
        lines.append(f"  top_k:            integer in [{int(ss.top_k.min)}, {int(ss.top_k.max)}]")
        lines.append(
            f"  hybrid_alpha:     float in [{ss.hybrid_alpha.min}, {ss.hybrid_alpha.max}]"
            "  (0=BM25 only, 1=vector only)"
        )
        lines.append(f"  reranker:         choose from {ss.reranker.models}")
        lines.append(
            f"  reranker_top_n:   integer in [{int(ss.reranker.top_n.min)}, {int(ss.reranker.top_n.max)}]"
        )
        lines.append(f"  query_expansion:  choose from {ss.query_expansion}")

        lines.append("")
        lines.append("  # Generation parameters:")
        lines.append(f"  llm_model:        choose from {ss.llm_models}")
        lines.append(
            f"  temperature:      float in [{ss.temperature.min}, {ss.temperature.max}]"
        )

        # --- Graph parameters (only if graph is in the search space) ---
        if self.graph is not None:
            g = self.graph
            lines.append("")
            lines.append("  # Graph parameters (only when index_type includes 'graph'):")
            lines.append(f"  graph_backend:     {g.graph_backend}")
            lines.append(
                f"  traversal_depth:   integer in [{int(g.traversal_depth.min)}, {int(g.traversal_depth.max)}]"
            )
            if g.entity_types:
                lines.append(f"  entity_types:      {g.entity_types}")

        # --- Expected output format ---
        # Build a concrete example using the first/default value for each param
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
        lines.append(f"hybrid_alpha: 0.5")
        lines.append(f"reranker: {example_reranker}")
        lines.append(f"reranker_top_n: {example_reranker_topn}")
        lines.append(f"query_expansion: {example_qe}")
        lines.append(f"llm_model: {example_llm}")
        lines.append(f"temperature: {example_temp}")

        if self.graph is not None:
            lines.append("graph:")
            lines.append(f"  graph_backend: {self.graph.graph_backend}")
            lines.append(f"  traversal_depth: 2")
            if self.graph.entity_types:
                lines.append(f"  entity_types: {self.graph.entity_types}")

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
