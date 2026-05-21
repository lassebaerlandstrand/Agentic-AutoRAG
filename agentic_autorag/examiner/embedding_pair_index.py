"""Embedding-similarity chunk pairing for cross-document 2-hop seed discovery.

Each chunk gets its top-K most-similar cross-document neighbours under cosine
similarity. Pairs are selected via per-chunk round-robin over rank: round 1
takes every chunk's rank-1 neighbour, round 2 takes every chunk's rank-2
neighbour, and so on. No absolute cosine thresholds — top-K adapts to the
corpus naturally (a tight corpus produces tight cosines, a diverse corpus
produces wide cosines, but each chunk still gets its K best).

Near-duplicate filtering is intentionally not done here — that's the
``corpus_cleaner.detect_near_duplicates`` step's job at the document level. If
a chunk-level near-duplicate slips through, the LLM correctly identifies it
as same-content and refuses; one wasted seed per such pair is acceptable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from agentic_autorag.examiner.chunk_pair_index import ChunkRecord, Seed

logger = logging.getLogger(__name__)


def emit_embedding_pairs(
    chunks: list[ChunkRecord],
    embeddings: np.ndarray,
    *,
    top_k_per_chunk: int,
    target_count: int,
) -> list[Seed]:
    """Emit cross-doc seed pairs by embedding similarity.

    ``chunks`` and ``embeddings`` must be aligned — row ``i`` of the matrix is
    the L2-normalised embedding of ``chunks[i]``. The caller is responsible
    for any section filtering before calling.

    Returns a list of Seeds deduped by canonical (chunk_a_id, chunk_b_id)
    tuple, truncated to ``target_count`` via per-rank round-robin.
    """
    if top_k_per_chunk < 1:
        raise ValueError(f"top_k_per_chunk must be >= 1, got {top_k_per_chunk}")
    if target_count < 1:
        return []
    if len(chunks) < 2:
        logger.warning("Embedding pairing: only %d chunks — no cross-doc pairs possible", len(chunks))
        return []
    if embeddings.shape[0] != len(chunks):
        raise ValueError(f"embeddings ({embeddings.shape[0]}) and chunks ({len(chunks)}) must align")

    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -np.inf)
    doc_ids = np.array([c.doc_id for c in chunks])
    same_doc_mask = doc_ids[:, None] == doc_ids[None, :]
    sim = np.where(same_doc_mask, -np.inf, sim)

    _log_cosine_histogram(sim)

    n = len(chunks)
    k = min(top_k_per_chunk, n - 1)
    top_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    top_sim = sim[rows, top_idx]
    order = np.argsort(-top_sim, axis=1)
    top_idx = top_idx[rows, order]
    top_sim = top_sim[rows, order]

    # Round-robin selection by rank, in deterministic chunk order.
    chunk_order = sorted(range(n), key=lambda i: chunks[i].chunk_id)
    seeds_by_pair: dict[tuple[str, str], Seed] = {}
    for rank in range(k):
        if len(seeds_by_pair) >= target_count:
            break
        for i in chunk_order:
            j = int(top_idx[i, rank])
            cosine = float(top_sim[i, rank])
            if not np.isfinite(cosine):
                continue
            chunk_a = chunks[i]
            chunk_b = chunks[j]
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
    """Log cross-doc cosine distribution + upper-tail percentiles."""
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

    L2-normalises outputs so cosine similarity reduces to a dot product. The
    model loads lazily on first call and is cached on the closure, so
    re-invoking the returned callable does not reload the model. Importing
    this module remains cheap; tests inject their own embedder.

    On GPU the model is loaded in fp16 (matching the trial-time embedder in
    ``engine.index_builder.get_embedder``) — ~30-50% faster than fp32 with
    no measurable effect on cosine ranking. Output is still cast to float32
    for downstream consumers that expect that dtype.
    """
    model = None

    def _encode(texts: list[str]) -> np.ndarray:
        nonlocal model
        if model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            model_kwargs = {"dtype": torch.float16} if torch.cuda.is_available() else {}
            model = SentenceTransformer(model_name, model_kwargs=model_kwargs)
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
