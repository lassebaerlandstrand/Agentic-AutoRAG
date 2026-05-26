"""Neighborhood builder: expand each anchor into a related-chunk cluster.

A neighborhood is the composer's design palette: the anchor + same-doc
siblings + cross-doc chunks ranked by word n-gram (1-3) TF-IDF cosine to
the palette centroid. Lexical n-grams (not dense embeddings) are used on
the cross-doc side: dense cosine would collapse the construction signal
into the retrieval signal and make the exam non-discriminative across
retrieval configurations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from statistics import median

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from agentic_autorag.examiner.chunk_pair_index import ChunkRecord, Neighborhood

logger = logging.getLogger(__name__)


@dataclass
class NeighborhoodDiagnostic:
    """Per-neighborhood diagnostic data carried alongside the palette.

    ``centroid`` is the L2-normalized 1D TF-IDF vector of (anchor +
    same-doc picks) — the vector cross-doc candidates were ranked
    against. ``position_kinds`` labels each entry of
    ``Neighborhood.chunks`` as ``"anchor"`` / ``"same_doc"`` /
    ``"cross_doc"``, useful for downstream logging.
    """

    centroid: np.ndarray
    position_kinds: list[str]


def build_tfidf_matrix(chunks: list[ChunkRecord]) -> tuple[csr_matrix, TfidfVectorizer]:
    """Build an L2-normalised TF-IDF matrix over chunk texts.

    Returns ``(tfidf, vectorizer)``. Each row of ``tfidf`` is the
    L2-normalised TF-IDF vector for the corresponding chunk over a
    word n-gram (1-3) feature space, so the inner product between two
    rows is their cosine similarity.

    ``ngram_range=(1, 3)`` mixes unigrams with bi/tri-grams: a shared
    trigram like ``dual antiplatelet therapy`` is far rarer than its
    unigrams and dominates cosine ranking when present. ``stop_words``
    drops English function words (incl. ``you/your/will/can/...``) so
    consumer-style chunks don't bridge to each other through stylistic
    pronoun overlap. ``max_df=0.5`` drops corpus-stopwords (terms
    appearing in more than half of chunks; mostly inert at trigram
    level). ``min_df=2`` drops hapaxes (typos, OCR artefacts) and
    keeps the trigram vocabulary bounded.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9\-]{2,}\b",
        max_df=0.5,
        min_df=2,
        sublinear_tf=True,
        norm="l2",
        ngram_range=(1, 3),
        stop_words=list(ENGLISH_STOP_WORDS),
    )
    tfidf = vectorizer.fit_transform(c.text for c in chunks)
    return tfidf, vectorizer


def _target_size(
    anchor_words: int,
    other_word_counts: list[int],
    min_chunks: int,
    min_words: int,
) -> int:
    """Smallest neighborhood size satisfying ``min_chunks OR min_words``.

    Density is estimated from the median chunk size in the pool — not
    the largest — because the fill takes chunks by TF-IDF cosine
    ranking, not by size. Using the descending max would underestimate
    how many chunks are needed to reach ``min_words`` on corpora with a
    small median and a long right tail (e.g. HotpotQA paragraphs, where
    median≈81 but max≈999). Anchor counts as one chunk and contributes
    ``anchor_words`` toward the word total.
    """
    if not other_word_counts:
        return 1
    extras_for_chunks = max(0, min_chunks - 1)
    median_w = median(other_word_counts)
    if median_w > 0:
        remaining = max(0, min_words - anchor_words)
        # ``statistics.median`` returns float for even-length pools (avg of
        # two middle ints), so ceil over float-divide to keep this int.
        extras_for_words = math.ceil(remaining / median_w)
    else:
        extras_for_words = extras_for_chunks
    target_extras = min(extras_for_chunks, extras_for_words)
    return min(1 + target_extras, 1 + len(other_word_counts))


def build_neighborhood(
    anchor_idx: int,
    chunks: list[ChunkRecord],
    tfidf: csr_matrix,
    *,
    min_chunks: int = 12,
    min_words: int = 5000,
    same_doc_weight: float = 0.8,
    cross_doc_weight: float = 0.2,
) -> tuple[Neighborhood, NeighborhoodDiagnostic]:
    """Grow a neighborhood around ``chunks[anchor_idx]``.

    Algorithm:

      1. Pre-compute target size ``N`` from ``min_chunks`` / ``min_words``
         (the smaller of the two floors, estimating the word floor with
         the pool's median chunk size).
      2. Normalize ``(same_doc_weight, cross_doc_weight)`` and split the
         non-anchor slots ``N - 1`` into ``target_same`` /
         ``target_cross``.
      3. Take up to ``target_same`` same-doc siblings in
         document-natural order (the chunker's emission order, which
         ``enumerate(chunks)`` already preserves). If the pool is
         smaller, redirect the deficit to the cross-doc target.
      4. Build the palette centroid = L2-normalize(sum of TF-IDF rows of
         anchor + same-doc picks).
      5. Rank cross-doc candidates by cosine to that centroid,
         break ties by chunk_id, take up to ``target_cross +
         redirected_deficit``.

    No interleaving; the final size never exceeds ``N``. If both pools
    exhaust before ``N`` is reached, return the smaller palette
    (anchor-only minimum). Also returns a ``NeighborhoodDiagnostic``
    with the centroid and per-position kind labels, used by downstream
    logging to attribute per-chunk shared-term contributions.
    """
    if not chunks:
        raise ValueError("chunks must be non-empty")
    if not (0 <= anchor_idx < len(chunks)):
        raise IndexError(f"anchor_idx {anchor_idx} out of range for {len(chunks)} chunks")
    if tfidf.shape[0] != len(chunks):
        raise ValueError(f"tfidf ({tfidf.shape[0]}) and chunks ({len(chunks)}) must align")
    if min_chunks < 1:
        raise ValueError(f"min_chunks must be >= 1, got {min_chunks}")
    if min_words < 0:
        raise ValueError(f"min_words must be >= 0, got {min_words}")
    if same_doc_weight < 0 or cross_doc_weight < 0:
        raise ValueError(f"weights must be >= 0, got same={same_doc_weight}, cross={cross_doc_weight}")
    total_weight = same_doc_weight + cross_doc_weight
    if total_weight <= 0:
        raise ValueError("same_doc_weight + cross_doc_weight must be > 0")

    anchor = chunks[anchor_idx]
    anchor_words = len(anchor.text.split())

    same_doc_pool: list[int] = [i for i, c in enumerate(chunks) if c.doc_id == anchor.doc_id and i != anchor_idx]
    cross_doc_pool_indices: list[int] = [i for i, c in enumerate(chunks) if c.doc_id != anchor.doc_id]

    other_word_counts = [len(chunks[i].text.split()) for i in (*same_doc_pool, *cross_doc_pool_indices)]
    target_n = _target_size(anchor_words, other_word_counts, min_chunks, min_words)
    extras = target_n - 1
    same_ratio = same_doc_weight / total_weight
    target_same = round(extras * same_ratio)
    target_cross = extras - target_same

    same_picked = same_doc_pool[:target_same]
    deficit = target_same - len(same_picked)
    cross_budget = target_cross + deficit

    selected_with_anchor = [anchor_idx, *same_picked]
    row_sum = tfidf[selected_with_anchor].sum(axis=0)
    centroid = np.asarray(row_sum).ravel()
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    cross_picked: list[int] = []
    if cross_budget > 0 and cross_doc_pool_indices:
        centroid_sparse = csr_matrix(centroid)
        sims = (tfidf @ centroid_sparse.T).toarray().ravel()
        cross_ranked = sorted(
            cross_doc_pool_indices,
            key=lambda i: (-float(sims[i]), chunks[i].chunk_id),
        )
        cross_picked = cross_ranked[:cross_budget]

    ordered_indices = [anchor_idx, *same_picked, *cross_picked]
    position_kinds = ["anchor"] + ["same_doc"] * len(same_picked) + ["cross_doc"] * len(cross_picked)
    nh = Neighborhood(chunks=[chunks[i] for i in ordered_indices])
    diag = NeighborhoodDiagnostic(centroid=centroid, position_kinds=position_kinds)
    return nh, diag
