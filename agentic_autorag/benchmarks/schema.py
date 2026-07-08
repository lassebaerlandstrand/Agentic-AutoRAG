"""Pydantic models shared by benchmark adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agentic_autorag.config.models import OpenEndedQuestion


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

    def to_open_ended(self) -> OpenEndedQuestion:
        """Convert to a tier-A/B ``OpenEndedQuestion`` for the exam evaluator.

        Doc-level gold (``supporting_doc_ids``) is carried through as the tier-B
        lane; no spans are attached (``reasoning_type=None``). This lets the
        custom-exam loader feed raw benchmark QA through the same evaluator/
        diagnoser path as a self-generated exam without forcing the heavier
        held-out free-form evaluator. The first gold answer is canonical, the
        rest become variants.
        """
        # Local import avoids a benchmarks <-> config.models import cycle.
        from agentic_autorag.config.models import OpenEndedQuestion

        gold = [g for g in self.gold_answers if g and g.strip()]
        if not gold:
            raise ValueError(f"BenchmarkQAPair {self.id!r} has no non-empty gold answer")
        return OpenEndedQuestion(
            id=self.id,
            question=self.question,
            canonical_answer=gold[0],
            answer_variants=gold[1:],
            supporting_doc_ids=list(self.supporting_doc_ids),
        )


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
