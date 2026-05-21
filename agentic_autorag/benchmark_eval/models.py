"""Result models written out by ``benchmark-evaluate``."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_autorag.benchmarks.schema import BenchmarkManifest


class QAResult(BaseModel):
    """Per-question evaluation record."""

    id: str
    question: str
    gold_answers: list[str]
    pred: str
    em: float
    f1: float
    judge: int | None = None  # 1=correct, 0=incorrect, None=disabled or parse-fail
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    supporting_doc_ids: list[str] = Field(default_factory=list)
    retrieval_rank_of_first_gold: int | None = None
    # 1-indexed rank at which the retrieved list first contains every gold doc.
    # Equals ``retrieval_rank_of_first_gold`` on single-gold questions; on
    # multi-hop questions it is the rank of the *last-needed* gold doc.
    retrieval_complete_rank: int | None = None
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    # LLM cost for this question (query expansion + generation). Excludes
    # embedder, reranker (local), and judge. 0.0 when LiteLLM has no pricing.
    llm_cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None  # sentinel when the question failed permanently


class BenchmarkResult(BaseModel):
    """Aggregate + per-question results, enough for paper traceability."""

    benchmark: str
    n_total: int
    n_valid: int
    n_judge_invalid: int = 0
    em: float
    f1: float
    llm_judge_accuracy: float | None = None
    recall_at_1: float | None = None
    recall_at_2: float | None = None
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    joint_recall_at_1: float | None = None
    joint_recall_at_2: float | None = None
    joint_recall_at_5: float | None = None
    joint_recall_at_10: float | None = None
    mrr_first: float | None = None
    mrr_complete: float | None = None
    avg_retrieval_s: float = 0.0
    avg_generation_s: float = 0.0
    total_cost_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    per_question: list[QAResult]
    judge_model: str | None = None
    trial_config_hash: str
    project_config_hash: str
    corpus_hash: str
    benchmark_manifest: BenchmarkManifest
