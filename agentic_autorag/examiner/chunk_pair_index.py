"""Chunk records and seed pairs for the open-ended exam pipeline.

Pairing itself lives in ``embedding_pair_index``. This module owns only the
two data classes the pairing step produces and the composition stage
consumes — keeping the dependency graph one-way (composition imports here;
nothing here imports composition).
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_autorag.engine.section_classifier import SectionLabel


@dataclass
class ChunkRecord:
    """A single chunk fed to the pairing step.

    ``chunk_id`` must be globally unique (e.g. ``f"{doc_id}::chunk_{i}"``).
    ``cluster_id`` is optional; pairing currently ignores it. ``section`` is
    the heuristic section label (``body``, ``references``, …) populated by
    the section classifier; ``None`` means "unknown" and is treated as
    eligible.
    """

    chunk_id: str
    doc_id: str
    text: str
    cluster_id: int = 0
    section: SectionLabel | None = None


@dataclass
class Seed:
    """One candidate 2-hop chunk pair ready for LLM composition.

    ``score`` carries the cosine similarity between the two chunks under the
    pair-embedding model. The composition LLM never sees the score — it's
    used only for diagnostic logging.
    """

    chunk_a: ChunkRecord
    chunk_b: ChunkRecord
    score: float = 0.0
