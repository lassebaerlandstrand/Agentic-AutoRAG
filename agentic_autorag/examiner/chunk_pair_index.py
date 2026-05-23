"""Chunk records and seed pairs for the open-ended exam pipeline.

Pairing itself lives in ``embedding_pair_index``. This module owns only the
two data classes the pairing step produces and the composition stage
consumes — keeping the dependency graph one-way (composition imports here;
nothing here imports composition).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentic_autorag.engine.section_classifier import SectionLabel

SeedOrigin = Literal["single_chunk", "same_doc_pair", "cross_doc_pair"]


@dataclass
class ChunkRecord:
    """A single chunk fed to the pairing step.

    ``chunk_id`` must be globally unique (e.g. ``f"{doc_id}::chunk_{i}"``).
    ``section`` is the heuristic section label (``body``, ``references``, …)
    populated by the section classifier; ``None`` means "unknown" and is
    treated as eligible.
    """

    chunk_id: str
    doc_id: str
    text: str
    section: SectionLabel | None = None


@dataclass
class Seed:
    """One candidate chunk (or chunk pair) ready for LLM composition.

    ``chunk_b`` is None for single-chunk seeds. ``origin`` tells the
    composition layer which user-prompt branch to use. ``score`` carries
    the cosine similarity for paired seeds (used only for diagnostic
    logging).
    """

    chunk_a: ChunkRecord
    chunk_b: ChunkRecord | None = None
    score: float = 0.0
    origin: SeedOrigin = "cross_doc_pair"
