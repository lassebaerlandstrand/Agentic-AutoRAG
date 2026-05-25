"""Neighborhood builder: expand each anchor into a related-chunk cluster.

A neighborhood is the design palette the composer sees. It contains the
anchor chunk plus a configurable mix of (a) same-document siblings —
useful for paper-like corpora where multi-hop reasoning happens
within-document — and (b) cross-document chunks ranked by TF-IDF cosine
similarity to the anchor — useful for Wikipedia-like corpora where
bridges live across documents via shared distinctive vocabulary.

The size criterion is adaptive: the neighborhood grows until it contains
at least ``min_chunks`` chunks OR at least ``min_words`` total words,
whichever is satisfied first. This automatically calibrates to chunk
granularity — small-chunk corpora hit the chunk floor (12 chunks ≈ a
few hundred words), large-chunk corpora hit the word floor (5 chunks ≈
5000 words for typical academic papers).

Cross-doc ranking uses TF-IDF cosine over the chunk text rather than
dense embedding cosine. The motivation: dense embeddings cluster chunks
by overall semantic similarity, so any embedding-based retriever
trivially co-locates them — which collapses the construction signal
into the retrieval signal and makes the resulting exam non-discriminative
across retrieval configurations. TF-IDF cosine surfaces chunks that
share *rare distinctive vocabulary* with the anchor, often spanning
different topical contexts (e.g., two chunks both mentioning a specific
lab but discussing different aspects of it) — exactly the multi-hop
bridge material embedding-NN misses.
"""

from __future__ import annotations

import logging

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from agentic_autorag.examiner.chunk_pair_index import ChunkRecord, Neighborhood

logger = logging.getLogger(__name__)


def build_tfidf_matrix(chunks: list[ChunkRecord]) -> tuple[csr_matrix, TfidfVectorizer]:
    """Build an L2-normalised TF-IDF matrix over chunk texts.

    Returns ``(tfidf, vectorizer)``. Each row of ``tfidf`` is the
    L2-normalised TF-IDF vector for the corresponding chunk, so the inner
    product between two rows is their cosine similarity.

    ``max_df=0.5`` drops corpus-stopwords (terms appearing in more than
    half of chunks). ``min_df=2`` drops hapaxes (typos, OCR artefacts).
    These are corpus-relative and don't need per-corpus tuning.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9\-]{2,}\b",
        max_df=0.5,
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    tfidf = vectorizer.fit_transform(c.text for c in chunks)
    return tfidf, vectorizer


def build_neighborhood(
    anchor_idx: int,
    chunks: list[ChunkRecord],
    tfidf: csr_matrix,
    *,
    min_chunks: int = 12,
    min_words: int = 5000,
    same_doc_fraction: float = 0.4,
) -> Neighborhood:
    """Grow a neighborhood around ``chunks[anchor_idx]``.

    Algorithm:

      1. Start with the anchor.
      2. Build two candidate pools:
         - same-doc: other chunks in the anchor's document, ordered by
           the document's natural chunk order.
         - cross-doc: chunks in OTHER documents, ordered by TF-IDF
           cosine similarity to the anchor (descending).
      3. Interleave the two pools using ``same_doc_fraction`` as the
         target mix: after each addition, pick from the pool whose
         current share is below its target. When one pool is exhausted,
         draw exclusively from the other.
      4. Stop as soon as the neighborhood has ``>= min_chunks`` chunks
         OR ``>= min_words`` total words.

    ``tfidf`` is the precomputed sparse TF-IDF matrix from
    ``build_tfidf_matrix`` (n_chunks, n_vocab); rows are L2-normalised so
    the inner product between rows is a cosine similarity.

    Returns a ``Neighborhood`` with the anchor at position 0.
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
    if not (0.0 <= same_doc_fraction <= 1.0):
        raise ValueError(f"same_doc_fraction must be in [0, 1], got {same_doc_fraction}")

    anchor = chunks[anchor_idx]

    same_doc_pool: list[int] = [i for i, c in enumerate(chunks) if c.doc_id == anchor.doc_id and i != anchor_idx]

    sims = (tfidf @ tfidf[anchor_idx].T).toarray().ravel()
    cross_doc_candidates = [(float(sims[i]), i) for i, c in enumerate(chunks) if c.doc_id != anchor.doc_id]
    cross_doc_candidates.sort(key=lambda t: (-t[0], chunks[t[1]].chunk_id))
    cross_doc_pool: list[int] = [i for _, i in cross_doc_candidates]

    selected_indices: list[int] = [anchor_idx]
    selected_set: set[int] = {anchor_idx}
    total_words = len(anchor.text.split())

    same_doc_cursor = 0
    cross_doc_cursor = 0
    n_same_added = 0
    n_cross_added = 0

    while len(selected_indices) < min_chunks and total_words < min_words:
        same_exhausted = same_doc_cursor >= len(same_doc_pool)
        cross_exhausted = cross_doc_cursor >= len(cross_doc_pool)
        if same_exhausted and cross_exhausted:
            break

        n_added = n_same_added + n_cross_added
        current_same_share = 0.0 if n_added == 0 else n_same_added / n_added

        pick_same = (not same_exhausted) and (cross_exhausted or current_same_share < same_doc_fraction)

        if pick_same:
            idx = same_doc_pool[same_doc_cursor]
            same_doc_cursor += 1
            if idx in selected_set:
                continue
            selected_indices.append(idx)
            selected_set.add(idx)
            total_words += len(chunks[idx].text.split())
            n_same_added += 1
        else:
            idx = cross_doc_pool[cross_doc_cursor]
            cross_doc_cursor += 1
            if idx in selected_set:
                continue
            selected_indices.append(idx)
            selected_set.add(idx)
            total_words += len(chunks[idx].text.split())
            n_cross_added += 1

    return Neighborhood(chunks=[chunks[i] for i in selected_indices])
