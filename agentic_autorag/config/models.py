"""Pydantic models for Agentic AutoRAG configuration and data structures.

Concrete config models represent what the agent proposes (specific values).
Search space models represent what the YAML defines (ranges and option lists).
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, field_validator, model_validator


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
    """Parameters that require re-indexing when changed."""

    parser: str = "pymupdf4llm"
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
    """Parameters that can be changed without re-indexing."""

    top_k: int = 5
    hybrid_alpha: float = 0.5
    reranker: str = "none"
    reranker_top_n: int = 5
    query_expansion: str = "none"
    llm_model: str = "ollama/llama3.2"
    temperature: float = 0.0


class GraphConfig(BaseModel):
    """Graph-specific parameters, only relevant for graph index types."""

    graph_backend: str = "networkx"
    traversal_depth: int = 2
    entity_types: list[str] | None = None


class TrialConfig(BaseModel):
    """Complete configuration for a single optimization trial."""

    structural: StructuralConfig
    runtime: RuntimeConfig
    graph: GraphConfig | None = None

    @model_validator(mode="after")
    def graph_required_for_graph_index(self) -> TrialConfig:
        if self.structural.index_type in (IndexType.GRAPH, IndexType.HYBRID_GRAPH_VECTOR) and self.graph is None:
            raise ValueError("graph config required when index_type uses graph")
        return self

    def structural_fingerprint(self) -> str:
        """Deterministic hash for index registry lookup.

        Includes structural params + graph_backend/entity_types.
        """
        data = self.structural.model_dump()
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


class RetrievalSearchSpace(BaseModel):
    """Allowed retrieval parameters."""

    top_k: NumericRange = NumericRange(min=3, max=20)
    hybrid_alpha: NumericRange = NumericRange(min=0.0, max=1.0)
    reranker: RerankerSearchSpace = RerankerSearchSpace()
    query_expansion: list[str] = ["none"]


class GenerationSearchSpace(BaseModel):
    """Allowed generation parameters."""

    llm_models: list[str]
    temperature: NumericRange = NumericRange(min=0.0, max=1.0)


class StructuralSearchSpace(BaseModel):
    """Structural search space: parameters that trigger re-indexing."""

    parsers: list[str] = ["pymupdf4llm"]
    chunking: ChunkingSearchSpace = ChunkingSearchSpace()
    embedding_models: list[str]
    index_types: list[IndexType] = [IndexType.VECTOR_ONLY]


class RuntimeSearchSpace(BaseModel):
    """Runtime search space: parameters swappable without re-indexing."""

    retrieval: RetrievalSearchSpace = RetrievalSearchSpace()
    generation: GenerationSearchSpace


class GraphSearchSpace(BaseModel):
    """Graph-specific search space parameters."""

    graph_backend: str = "networkx"
    traversal_depth: NumericRange = NumericRange(min=1, max=3)
    entity_types: list[str] | None = None


class ExaminerConfig(BaseModel):
    """Settings for the adaptive examiner."""

    exam_size: int = 50
    diversity_clusters: int | None = None  # None = auto (sqrt of chunk count, capped at exam_size)
    irt_discrimination_threshold: float = 0.3
    refresh_interval_trials: int = 5
    mcq_options_count: int = 4


class AgentConfig(BaseModel):
    """Settings for the LLM agents."""

    optimizer_model: str = "gemini/gemini-3-flash-preview"
    examiner_model: str = "gemini/gemini-3-flash-preview"
    max_history_trials: int = 10


class MetaConfig(BaseModel):
    """Project-level settings."""

    project_name: str = "my-rag-project"
    corpus_path: str = "./data/corpus/"
    corpus_description: str = ""
    output_dir: str = "./experiments/"
    max_trials: int = 30
    index_registry: bool = True


class SearchSpace(BaseModel):
    """The full search space loaded from YAML.

    Numeric parameters are stored as NumericRange (min/max).
    Categorical parameters are stored as lists.
    The agent is told these constraints and proposes values within them.
    """

    meta: MetaConfig = MetaConfig()
    structural: StructuralSearchSpace
    runtime: RuntimeSearchSpace
    graph: GraphSearchSpace | None = None
    examiner: ExaminerConfig = ExaminerConfig()
    agent: AgentConfig = AgentConfig()

    def validate_trial(self, trial: TrialConfig) -> list[str]:
        """Check whether a proposed trial config falls within the search space.

        Returns a list of violation messages (empty = valid).
        """
        violations: list[str] = []
        s = self.structural
        r = self.runtime

        # --- Structural checks ---
        if trial.structural.parser not in s.parsers:
            violations.append(f"parser '{trial.structural.parser}' not in {s.parsers}")
        if trial.structural.chunking_strategy not in s.chunking.strategies:
            violations.append(
                f"chunking_strategy '{trial.structural.chunking_strategy}' not in {s.chunking.strategies}"
            )
        if not s.chunking.chunk_size.contains(trial.structural.chunk_size):
            violations.append(
                f"chunk_size {trial.structural.chunk_size} outside "
                f"[{s.chunking.chunk_size.min}, {s.chunking.chunk_size.max}]"
            )
        if not s.chunking.chunk_overlap.contains(trial.structural.chunk_overlap):
            violations.append(
                f"chunk_overlap {trial.structural.chunk_overlap} outside "
                f"[{s.chunking.chunk_overlap.min}, {s.chunking.chunk_overlap.max}]"
            )
        if trial.structural.embedding_model not in s.embedding_models:
            violations.append(f"embedding_model '{trial.structural.embedding_model}' not in {s.embedding_models}")
        if trial.structural.index_type not in s.index_types:
            violations.append(
                f"index_type '{trial.structural.index_type.value}' not in {[t.value for t in s.index_types]}"
            )

        # --- Runtime retrieval checks ---
        if not r.retrieval.top_k.contains(trial.runtime.top_k):
            violations.append(f"top_k {trial.runtime.top_k} outside [{r.retrieval.top_k.min}, {r.retrieval.top_k.max}]")
        if not r.retrieval.hybrid_alpha.contains(trial.runtime.hybrid_alpha):
            violations.append(
                f"hybrid_alpha {trial.runtime.hybrid_alpha} outside "
                f"[{r.retrieval.hybrid_alpha.min}, {r.retrieval.hybrid_alpha.max}]"
            )
        if trial.runtime.reranker not in r.retrieval.reranker.models:
            violations.append(f"reranker '{trial.runtime.reranker}' not in {r.retrieval.reranker.models}")
        if not r.retrieval.reranker.top_n.contains(trial.runtime.reranker_top_n):
            violations.append(
                f"reranker_top_n {trial.runtime.reranker_top_n} outside "
                f"[{r.retrieval.reranker.top_n.min}, {r.retrieval.reranker.top_n.max}]"
            )
        if trial.runtime.query_expansion not in r.retrieval.query_expansion:
            violations.append(f"query_expansion '{trial.runtime.query_expansion}' not in {r.retrieval.query_expansion}")

        # --- Runtime generation checks ---
        if trial.runtime.llm_model not in r.generation.llm_models:
            violations.append(f"llm_model '{trial.runtime.llm_model}' not in {r.generation.llm_models}")
        if not r.generation.temperature.contains(trial.runtime.temperature):
            violations.append(
                f"temperature {trial.runtime.temperature} outside "
                f"[{r.generation.temperature.min}, {r.generation.temperature.max}]"
            )

        # --- Graph checks ---
        if trial.graph and self.graph and not self.graph.traversal_depth.contains(trial.graph.traversal_depth):
            violations.append(
                f"traversal_depth {trial.graph.traversal_depth} outside "
                f"[{self.graph.traversal_depth.min}, {self.graph.traversal_depth.max}]"
            )

        return violations

    def to_agent_prompt(self) -> str:
        """Format the search space as readable text for the agent's LLM context."""
        return json.dumps(self.model_dump(), indent=2, default=str)


class MCQQuestion(BaseModel):
    """A single multiple-choice question in the exam."""

    id: str
    question: str
    options: dict[str, str]  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_answer: str  # "A", "B", "C", or "D"
    source_chunk_id: str
    cluster_id: int
    difficulty: float = 0.5  # updated by IRT (b_j)
    discrimination: float = 1.0  # updated by IRT (a_j)
    guessing: float = 0.25  # updated by IRT (g_j), initialized to 1/mcq_options_count

    @field_validator("correct_answer")
    @classmethod
    def valid_answer_key(cls, v: str, info) -> str:
        options: dict[str, str] | None = info.data.get("options")
        if options and v not in options:
            raise ValueError(f"correct_answer '{v}' must be one of the option keys: {list(options.keys())}")
        return v
