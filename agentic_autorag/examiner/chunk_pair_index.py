"""Chunk records, anchors, and neighborhoods for the open-ended exam pipeline.

Each composition call processes one ``Neighborhood`` — an anchor chunk
plus its K-1 related neighbors, ordered with the anchor at position 0.
The composer cites chunks by their position in ``Neighborhood.chunks``
via ``selected_chunk_ids``. This file owns only the data classes the
seeders produce and the composition stage consumes; pairing / neighborhood
construction live in their own modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_autorag.engine.section_classifier import SectionLabel


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
class Anchor:
    """One chunk picked by the seeder as a neighborhood anchor.

    The neighborhood builder expands each anchor into a ``Neighborhood``;
    the composer never sees a bare ``Anchor``.
    """

    chunk: ChunkRecord


@dataclass
class Neighborhood:
    """An anchor chunk plus its related neighbors, ordered.

    ``chunks[0]`` is always the anchor; positions 1.. are the neighbors
    returned by the neighborhood builder (a mix of same-document siblings
    and cross-document cosine-similar chunks, sized adaptively to the
    corpus's chunk-word distribution). The composer cites chunks by
    position via ``CompositionResult.selected_chunk_ids``.
    """

    chunks: list[ChunkRecord]

    @property
    def anchor(self) -> ChunkRecord:
        return self.chunks[0]

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("Neighborhood must have at least one chunk (the anchor)")
