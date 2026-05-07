"""Embedding-similarity chunk pairing for cross-document 2-hop seed discovery.

Replaces the entity-cooccurrence indexer for the seeding step. Each chunk gets
its top-K most-similar cross-document neighbours under cosine similarity. Pairs
are selected via per-chunk round-robin over rank: round 1 takes every chunk's
rank-1 neighbour, round 2 takes every chunk's rank-2 neighbour, and so on. No
absolute cosine thresholds — top-K adapts to the corpus naturally (a tight
corpus produces tight cosines, a diverse corpus produces wide cosines, but
each chunk still gets its K best).

Near-duplicate filtering is intentionally not done here — that's the
``corpus_cleaner.detect_near_duplicates`` step's job at the document level. If
a chunk-level near-duplicate slips through, the LLM correctly identifies it
as same-content and refuses; one wasted seed per such pair is acceptable.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable

import numpy as np

from agentic_autorag.engine.section_classifier import SectionLabel
from agentic_autorag.examiner.chunk_pair_index import ChunkRecord, Seed

logger = logging.getLogger(__name__)


def emit_embedding_pairs(
    chunks: list[ChunkRecord],
    embed_callable: Callable[[list[str]], np.ndarray],
    *,
    top_k_per_chunk: int,
    target_count: int,
    eligible_sections: Iterable[SectionLabel] | None = None,
    model_name: str | None = None,
) -> list[Seed]:
    """Emit seed pairs by embedding similarity.

    Args:
        chunks: every ChunkRecord in the corpus (eligible + ineligible).
        embed_callable: produces an L2-normalised (N, D) float32 matrix from a
            list of N texts. Production wraps a SentenceTransformer; tests
            inject a deterministic stub.
        top_k_per_chunk: per-chunk neighbour count.
        target_count: maximum number of seeds to emit.
        eligible_sections: chunks whose ``section`` is not in this set are
            excluded from pairing (kept consistent with the section-classifier
            filter the entity pipeline used). ``None`` keeps every chunk
            eligible.

    Returns:
        A list of Seeds, deduped by canonical (chunk_a_id, chunk_b_id) tuple,
        truncated to ``target_count`` via per-rank round-robin.
    """
    if top_k_per_chunk < 1:
        raise ValueError(f"top_k_per_chunk must be >= 1, got {top_k_per_chunk}")
    if target_count < 1:
        return []

    eligible_set = frozenset(eligible_sections) if eligible_sections is not None else None
    eligible_chunks = [c for c in chunks if eligible_set is None or c.section is None or c.section in eligible_set]
    if eligible_set is not None and len(eligible_chunks) < len(chunks):
        logger.info(
            "Section filter: pairing over %d/%d chunks (skipped %d in non-eligible sections)",
            len(eligible_chunks),
            len(chunks),
            len(chunks) - len(eligible_chunks),
        )
    if len(eligible_chunks) < 2:
        logger.warning(
            "Embedding pairing: only %d eligible chunks — no cross-doc pairs possible",
            len(eligible_chunks),
        )
        return []

    t0 = time.perf_counter()
    embeddings = embed_callable([c.text for c in eligible_chunks])
    elapsed = time.perf_counter() - t0
    if model_name:
        logger.info("Pair-embedded %d chunks via %s in %.1fs", len(eligible_chunks), model_name, elapsed)
    else:
        logger.info("Pair-embedded %d chunks in %.1fs", len(eligible_chunks), elapsed)

    if embeddings.shape[0] != len(eligible_chunks):
        raise ValueError(f"embed_callable returned {embeddings.shape[0]} vectors for {len(eligible_chunks)} chunks")

    # Cosine matrix (assumes L2-normalised embeddings).
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -np.inf)  # mask self
    # Mask same-doc neighbours.
    doc_ids = np.array([c.doc_id for c in eligible_chunks])
    same_doc_mask = doc_ids[:, None] == doc_ids[None, :]
    sim = np.where(same_doc_mask, -np.inf, sim)

    _log_cosine_histogram(sim)

    # Per-chunk top-K (descending).
    n = len(eligible_chunks)
    k = min(top_k_per_chunk, n - 1)
    # argpartition is faster than full argsort for top-K; then sort just the top-K.
    top_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    top_sim = sim[rows, top_idx]
    order = np.argsort(-top_sim, axis=1)
    top_idx = top_idx[rows, order]
    top_sim = top_sim[rows, order]

    # Round-robin selection by rank, in deterministic chunk order. Chunks are
    # ordered by chunk_id so the same input always yields the same seed list.
    chunk_order = sorted(range(n), key=lambda i: eligible_chunks[i].chunk_id)
    seeds_by_pair: dict[tuple[str, str], Seed] = {}
    for rank in range(k):
        if len(seeds_by_pair) >= target_count:
            break
        for i in chunk_order:
            j = int(top_idx[i, rank])
            cosine = float(top_sim[i, rank])
            if not np.isfinite(cosine):
                continue  # all candidates exhausted (e.g. only one cross-doc partner)
            chunk_a = eligible_chunks[i]
            chunk_b = eligible_chunks[j]
            cid_a, cid_b = sorted((chunk_a.chunk_id, chunk_b.chunk_id))
            if (cid_a, cid_b) in seeds_by_pair:
                continue
            if cid_a == chunk_a.chunk_id:
                seeds_by_pair[(cid_a, cid_b)] = Seed(chunk_a=chunk_a, chunk_b=chunk_b, score=cosine)
            else:
                seeds_by_pair[(cid_a, cid_b)] = Seed(chunk_a=chunk_b, chunk_b=chunk_a, score=cosine)
            if len(seeds_by_pair) >= target_count:
                break

    seeds = list(seeds_by_pair.values())
    logger.info(
        "Emitted %d seeds via round-robin top-%d from %d chunks (target=%d)",
        len(seeds),
        k,
        n,
        target_count,
    )
    return seeds


def _log_cosine_histogram(sim: np.ndarray) -> None:
    """Log the distribution of cross-doc cosines.

    Purely diagnostic — the algorithm doesn't use absolute thresholds. The
    histogram lets the operator see whether the corpus is tightly clustered
    (everything in [0.7, 0.95]) or spread out. Percentile readout makes it
    easy to spot a corpus where even the strongest cross-doc pairs are weak
    (e.g. p99=0.42 → mostly topically disjoint).
    """
    finite = sim[np.isfinite(sim)]
    if finite.size == 0:
        return
    bands = np.arange(0.0, 1.05, 0.1)
    counts, _ = np.histogram(finite, bins=bands)
    total = counts.sum()
    if total == 0:
        return
    parts = [
        f"{bands[i]:.1f}-{bands[i + 1]:.1f}: {counts[i]} ({100 * counts[i] / total:.1f}%)" for i in range(len(counts))
    ]
    logger.info("Pair cosine distribution (cross-doc, N=%d): %s", int(total), ", ".join(parts))
    # DIAG cosine percentiles — quick read of the upper tail.
    p50, p90, p95, p99 = np.percentile(finite, [50, 90, 95, 99])
    logger.info(
        "DIAG Cross-doc cosine percentiles (n=%d): p50=%.3f p90=%.3f p95=%.3f p99=%.3f",
        int(finite.size),
        p50,
        p90,
        p95,
        p99,
    )


def make_pair_embedder(model_name: str, *, batch_size: int = 32) -> Callable[[list[str]], np.ndarray]:
    """Return an ``embed_callable`` backed by a SentenceTransformer.

    L2-normalises outputs so cosine similarity reduces to a dot product. Loads
    the model lazily on first call so importing this module is cheap and tests
    that inject their own embedder never trigger a model download.
    """

    def _encode(texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        return np.asarray(
            model.encode(
                texts,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    return _encode
