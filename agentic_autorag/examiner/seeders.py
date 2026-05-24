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
                        (extraction / definitional / numeric_single /
                        inference).

All embedding-using seeders take a pre-computed (n_chunks, d) matrix; the
orchestrator computes the embeddings once in ``prepare_corpus`` and reuses
across same-doc + cross-doc. Section/eligibility filtering is the caller's
responsibility — pass already-filtered chunks (and embeddings aligned to
those chunks).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from agentic_autorag.examiner.chunk_pair_index import ChunkRecord, Seed
from agentic_autorag.examiner.embedding_pair_index import emit_embedding_pairs
from agentic_autorag.examiner.idf_pair_index import emit_idf_pairs

logger = logging.getLogger(__name__)

# RRF (reciprocal rank fusion, Cormack et al. 2009) smoothing constant.
# k=60 is the standard IR default; it caps the influence of rank-1 hits so
# a pair only one ranker found doesn't trivially outrank a pair both rankers
# liked. Any k in 20-100 produces similar orderings.
_RRF_K = 60

# Overgeneration factor for each ranking before fusion. The factor is small
# (2x) — pairs that don't make either ranker's top 2*target_count aren't
# bridges we want.
_FUSION_OVERGEN_FACTOR = 2

# Backstop text-length floor for single-chunk seeds. ExamAgent.prepare_corpus
# greedy-packs surviving chunks up to ``max_chunk_words``, so in practice
# eligible chunks are far above this floor; it only fires for pathological
# inputs (e.g. a single-chunk tiny doc that still cleared ``min_doc_words``).
_SINGLE_CHUNK_MIN_TEXT_CHARS = 500


def emit_single_chunk_seeds(
    chunks: list[ChunkRecord],
    *,
    target_count: int,
    min_text_chars: int = _SINGLE_CHUNK_MIN_TEXT_CHARS,
) -> list[Seed]:
    """Emit single-chunk seeds for single-hop question composition.

    Selects chunks long enough to contain a substantive factoid, in
    deterministic chunk-id order. The caller is responsible for any section
    filtering before calling.
    """
    if target_count < 1:
        return []
    eligible = [c for c in chunks if len(c.text) >= min_text_chars]
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
    embeddings: np.ndarray,
    *,
    target_count: int,
    cos_min: float,
    cos_max: float,
) -> list[Seed]:
    """Emit chunk pairs FROM THE SAME DOCUMENT for single-doc multi-hop.

    Pair scoring keeps cosines in ``[cos_min, cos_max]`` (mid-band): chunks
    should share a topic without being paraphrases. Pairs whose chunks share
    the same ``SectionLabel`` are dropped — UNLESS the corpus has fewer than
    2 distinct section labels, in which case section information is
    uninformative (e.g. Wikipedia paragraphs that all classify as ``body``)
    and the section-disjoint filter is skipped.

    ``chunks`` and ``embeddings`` must be aligned.
    """
    if target_count < 1:
        return []
    if len(chunks) < 2:
        return []
    if embeddings.shape[0] != len(chunks):
        raise ValueError(f"embeddings ({embeddings.shape[0]}) and chunks ({len(chunks)}) must align")

    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -np.inf)
    doc_ids = np.array([c.doc_id for c in chunks])
    same_doc_mask = doc_ids[:, None] == doc_ids[None, :]
    sim = np.where(same_doc_mask, sim, -np.inf)

    sections_arr = np.array([c.section.value if c.section is not None else "" for c in chunks])
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
            len(chunks),
        )

    finite_sim_mask = np.isfinite(sim)
    intra_doc_pairs = int(finite_sim_mask.sum() // 2)
    in_band = (sim >= cos_min) & (sim <= cos_max) & finite_sim_mask

    n = len(chunks)
    candidate_pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if in_band[i, j]:
                candidate_pairs.append((i, j, float(sim[i, j])))
    candidate_pairs.sort(key=lambda t: (-t[2], chunks[t[0]].chunk_id, chunks[t[1]].chunk_id))
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
        a, b = chunks[i], chunks[j]
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
        len(chunks),
        target_count,
        cos_min,
        cos_max,
    )
    return seeds


def emit_cross_doc_pair_seeds(
    chunks: list[ChunkRecord],
    embeddings: np.ndarray,
    *,
    top_k_per_chunk: int,
    target_count: int,
) -> list[Seed]:
    """Cross-doc pairing by reciprocal-rank-fusion of two signals.

    The embedding pairer captures semantic similarity (good for "two articles
    about the same kind of thing"); the IDF pairer captures rare-token
    overlap (good for "two articles sharing a specific entity"). Fusion
    surfaces bridges that either ranker alone misses.

    ``chunks`` and ``embeddings`` must be aligned.
    """
    overgen_count = max(target_count * _FUSION_OVERGEN_FACTOR, target_count)
    embedding_seeds = emit_embedding_pairs(
        chunks,
        embeddings,
        top_k_per_chunk=top_k_per_chunk,
        target_count=overgen_count,
    )
    idf_seeds = emit_idf_pairs(chunks, target_count=overgen_count)
    return _rrf_fuse(embedding_seeds, idf_seeds, target_count=target_count)


def _rrf_fuse(
    embedding_seeds: list[Seed],
    idf_seeds: list[Seed],
    *,
    target_count: int,
) -> list[Seed]:
    """Reciprocal rank fusion of two ranked Seed lists.

    Pairs are keyed by canonical ``tuple(sorted((chunk_id_a, chunk_id_b)))``.
    RRF score is ``sum_r 1/(k + rank_r)`` across rankers that ranked the pair.
    The output ``Seed.score`` carries the RRF value so downstream logging
    stays interpretable.
    """
    fused_scores: dict[tuple[str, str], float] = defaultdict(float)
    seed_by_key: dict[tuple[str, str], Seed] = {}

    for source in (embedding_seeds, idf_seeds):
        for rank, seed in enumerate(source, 1):
            if seed.chunk_b is None:
                continue
            key = tuple(sorted((seed.chunk_a.chunk_id, seed.chunk_b.chunk_id)))
            fused_scores[key] += 1.0 / (_RRF_K + rank)
            if key not in seed_by_key:
                seed_by_key[key] = seed

    ranked = sorted(
        fused_scores.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
    )

    out: list[Seed] = []
    for key, fused_score in ranked[:target_count]:
        proto = seed_by_key[key]
        out.append(
            Seed(
                chunk_a=proto.chunk_a,
                chunk_b=proto.chunk_b,
                score=fused_score,
                origin="cross_doc_pair",
            )
        )
    logger.info(
        "RRF-fused %d embedding + %d IDF seeds → %d unique pairs (top %d returned)",
        len(embedding_seeds),
        len(idf_seeds),
        len(fused_scores),
        len(out),
    )
    return out
