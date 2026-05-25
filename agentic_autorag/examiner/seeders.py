"""Anchor seeder for the exam composition pipeline.

Each anchor is a single chunk; the orchestrator expands every anchor into
a ``Neighborhood`` (via ``neighborhoods.build_neighborhood``) before
handing it to the composer. The composer then picks which subset of the
neighborhood to use per question and emits as many questions as the
chunks support.

Anchor sampling is text-length-weighted and deterministic given a fixed
``rng_seed`` — longer chunks have more factoid material so they get
preferential selection, but every eligible chunk has positive
probability of being picked so different runs explore different corners
of the corpus when the seed changes.
"""

from __future__ import annotations

import logging
import random

from agentic_autorag.examiner.chunk_pair_index import Anchor, ChunkRecord

logger = logging.getLogger(__name__)

# Text-length floor for anchor eligibility — a chunk shorter than this
# has too little substance to host a factoid lookup. On short-doc corpora
# (Wikipedia paragraphs, HotpotQA distractors) chunk lengths cluster
# around 300-700 chars after greedy-merge, so the floor is set tight
# enough to discard headers/captions/affiliations but loose enough to
# keep typical paragraph chunks.
_ANCHOR_MIN_TEXT_CHARS = 100

# Re-export under the old name so external imports keep working until the
# transition is complete. (exam_agent.py still references this constant
# in its prepare_corpus diagnostic logging.)
_SINGLE_CHUNK_MIN_TEXT_CHARS = _ANCHOR_MIN_TEXT_CHARS


def emit_anchor_seeds(
    chunks: list[ChunkRecord],
    *,
    target_count: int,
    min_text_chars: int = _ANCHOR_MIN_TEXT_CHARS,
    rng_seed: int | str | None = None,
) -> list[Anchor]:
    """Sample ``target_count`` anchor chunks, weighted by text length.

    Eligible chunks are those with ``len(text) >= min_text_chars``. Each
    eligible chunk is weighted by its word count so chunks with more
    factoid material are more likely picked — but every eligible chunk
    has positive weight, so on small corpora the sampler doesn't
    deterministically lock onto the same few chunks every run.

    Sampling is without replacement. If ``target_count`` exceeds the
    eligible pool, every eligible chunk is returned.
    """
    if target_count < 1:
        return []
    eligible = [c for c in chunks if len(c.text) >= min_text_chars]
    if not eligible:
        logger.info(
            "Emitted 0 anchor seeds (no chunks >= %d chars among %d input chunks)",
            min_text_chars,
            len(chunks),
        )
        return []

    rng = random.Random(rng_seed)
    n_pick = min(target_count, len(eligible))

    weights = [max(1, len(c.text.split())) for c in eligible]
    # Sample without replacement via Efraimidis-Spirakis: assign each
    # eligible chunk a key u**(1/w) where u ~ U(0,1), keep top n_pick.
    # This is the standard weighted-without-replacement reservoir
    # algorithm and is deterministic given the seeded RNG.
    keys = [(rng.random() ** (1.0 / w), i) for i, w in enumerate(weights)]
    keys.sort(reverse=True)
    picked_indices = sorted(i for _, i in keys[:n_pick])
    anchors = [Anchor(chunk=eligible[i]) for i in picked_indices]

    logger.info(
        "Emitted %d anchor seeds from %d eligible chunks (target=%d, min_chars=%d)",
        len(anchors),
        len(eligible),
        target_count,
        min_text_chars,
    )
    return anchors
