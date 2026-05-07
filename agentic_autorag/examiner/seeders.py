"""Seed generators for the exam composition pipeline.

Three flavours of seed feed the composition LLM, each with its own user-
prompt branch:

  ``cross_doc_pair`` — two chunks from DIFFERENT documents, paired by
                        cosine similarity. Wikipedia-style 2-hop.

  ``same_doc_pair``  — two chunks from the SAME document, with mid-band
                        cosine and section-disjoint, so the chunks share
                        a topic without being paraphrases. Single-doc
                        multi-hop (e.g. methods + results in one paper).

  ``single_chunk``   — one chunk on its own. Single-hop questions
                        (extraction / definitional).

The cross-doc generator preserves the round-robin top-K logic from
``embedding_pair_index.emit_embedding_pairs``; this module adds the same-
doc and single-chunk paths and is the new entry point.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import numpy as np

from agentic_autorag.engine.section_classifier import SectionLabel
from agentic_autorag.examiner.chunk_pair_index import ChunkRecord, Seed
from agentic_autorag.examiner.embedding_pair_index import emit_embedding_pairs

logger = logging.getLogger(__name__)


def emit_single_chunk_seeds(
    chunks: list[ChunkRecord],
    *,
    target_count: int,
    eligible_sections: Iterable[SectionLabel] | None = None,
    min_text_chars: int = 200,
) -> list[Seed]:
    """Emit single-chunk seeds for single-hop question composition.

    Selects chunks long enough to contain a substantive factoid, in
    deterministic chunk-id order.
    """
    if target_count < 1:
        return []
    eligible_set = frozenset(eligible_sections) if eligible_sections is not None else None
    eligible = [
        c
        for c in chunks
        if (eligible_set is None or c.section is None or c.section in eligible_set) and len(c.text) >= min_text_chars
    ]
    eligible.sort(key=lambda c: c.chunk_id)
    seeds = [Seed(chunk_a=c, chunk_b=None, score=0.0, origin="single_chunk") for c in eligible[:target_count]]
    logger.info(
        "Emitted %d single-chunk seeds from %d eligible chunks (target=%d)",
        len(seeds),
        len(eligible),
        target_count,
    )
    return seeds


def emit_same_doc_pair_seeds(
    chunks: list[ChunkRecord],
    embed_callable: Callable[[list[str]], np.ndarray],
    *,
    target_count: int,
    cos_min: float,
    cos_max: float,
    eligible_sections: Iterable[SectionLabel] | None = None,
    model_name: str | None = None,
) -> list[Seed]:
    """Emit chunk pairs FROM THE SAME DOCUMENT for single-doc multi-hop.

    Pair scoring keeps cosines in ``[cos_min, cos_max]`` (mid-band):
    chunks should share a topic without being paraphrases. Pairs whose
    chunks share the same ``SectionLabel`` are dropped — UNLESS the corpus
    has fewer than 2 distinct section labels, in which case section
    information is uninformative (e.g. Wikipedia paragraphs that all
    classify as ``body``) and the section-disjoint filter is skipped.
    """
    if target_count < 1:
        return []
    eligible_set = frozenset(eligible_sections) if eligible_sections is not None else None
    eligible_chunks = [c for c in chunks if eligible_set is None or c.section is None or c.section in eligible_set]
    if len(eligible_chunks) < 2:
        return []

    embeddings = embed_callable([c.text for c in eligible_chunks])
    if embeddings.shape[0] != len(eligible_chunks):
        raise ValueError(f"embed_callable returned {embeddings.shape[0]} vectors for {len(eligible_chunks)} chunks")
    if model_name:
        logger.info("Same-doc pair embedding via %s on %d chunks", model_name, len(eligible_chunks))

    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -np.inf)

    doc_ids = np.array([c.doc_id for c in eligible_chunks])
    same_doc_mask = doc_ids[:, None] == doc_ids[None, :]
    sim = np.where(same_doc_mask, sim, -np.inf)

    sections = [c.section for c in eligible_chunks]
    sections_arr = np.array([s.value if s is not None else "" for s in sections])
    distinct_sections = {s for s in sections_arr.tolist() if s}
    apply_section_disjoint = len(distinct_sections) >= 2
    if apply_section_disjoint:
        same_section_mask = (sections_arr[:, None] == sections_arr[None, :]) & (sections_arr[:, None] != "")
        sim = np.where(same_section_mask, -np.inf, sim)
    else:
        logger.info(
            "DIAG section-disjoint filter skipped: only %d distinct section label(s) "
            "across %d chunks (cosine band carries the load)",
            len(distinct_sections),
            len(eligible_chunks),
        )

    finite_sim_mask = np.isfinite(sim)
    intra_doc_pairs = int(finite_sim_mask.sum() // 2)
    in_band = (sim >= cos_min) & (sim <= cos_max) & finite_sim_mask

    n = len(eligible_chunks)
    candidate_pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if in_band[i, j]:
                candidate_pairs.append((i, j, float(sim[i, j])))
    candidate_pairs.sort(key=lambda t: (-t[2], eligible_chunks[t[0]].chunk_id, eligible_chunks[t[1]].chunk_id))
    logger.info(
        "DIAG Same-doc band utilisation: %d intra-doc candidate pairs, %d in band [%.2f, %.2f] (section-disjoint=%s)",
        intra_doc_pairs,
        len(candidate_pairs),
        cos_min,
        cos_max,
        apply_section_disjoint,
    )

    seeds: list[Seed] = []
    seen_pairs: set[tuple[str, str]] = set()
    for i, j, cosine in candidate_pairs:
        if len(seeds) >= target_count:
            break
        a, b = eligible_chunks[i], eligible_chunks[j]
        cid_a, cid_b = sorted((a.chunk_id, b.chunk_id))
        if (cid_a, cid_b) in seen_pairs:
            continue
        seen_pairs.add((cid_a, cid_b))
        if cid_a == a.chunk_id:
            seeds.append(Seed(chunk_a=a, chunk_b=b, score=cosine, origin="same_doc_pair"))
        else:
            seeds.append(Seed(chunk_a=b, chunk_b=a, score=cosine, origin="same_doc_pair"))

    logger.info(
        "Emitted %d same-doc seeds from %d eligible chunks (target=%d, cos band=[%.2f,%.2f])",
        len(seeds),
        len(eligible_chunks),
        target_count,
        cos_min,
        cos_max,
    )
    return seeds


def emit_cross_doc_pair_seeds(
    chunks: list[ChunkRecord],
    embed_callable: Callable[[list[str]], np.ndarray],
    *,
    top_k_per_chunk: int,
    target_count: int,
    eligible_sections: Iterable[SectionLabel] | None = None,
    model_name: str | None = None,
) -> list[Seed]:
    """Cross-doc round-robin top-K pairing — wraps the existing implementation."""
    seeds = emit_embedding_pairs(
        chunks,
        embed_callable,
        top_k_per_chunk=top_k_per_chunk,
        target_count=target_count,
        eligible_sections=eligible_sections,
        model_name=model_name,
    )
    for s in seeds:
        s.origin = "cross_doc_pair"
    return seeds
