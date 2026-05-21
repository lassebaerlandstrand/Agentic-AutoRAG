"""Pydantic models shared by benchmark adapters."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkQAPair(BaseModel):
    """A single held-out QA pair from a public benchmark.

    Produced by a `BenchmarkAdapter` alongside a prepared corpus directory.
    The corpus is consumed by the normal `optimize` run; these QA pairs are
    kept aside and scored against the best config by `benchmark-evaluate`.

    ``supporting_doc_ids`` references filenames (without extension) in the
    prepared corpus dir and is used by the free-form evaluator to compute
    Recall@k and MRR. Empty list is allowed for benchmarks without
    passage-level gold.
    """

    id: str
    question: str
    gold_answers: list[str]
    supporting_doc_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class BenchmarkManifest(BaseModel):
    """Reproducibility metadata written to ``metadata.json`` by each adapter."""

    name: str
    split: str
    sample_size: int
    seed: int
    adapter_version: str
    hf_revision: str | None = None
    corpus_doc_count: int = 0
    corpus_total_chars: int = 0
    corpus_total_words: int = 0
    corpus_avg_words_per_doc: float = 0.0
