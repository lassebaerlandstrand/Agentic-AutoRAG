"""Near-duplicate document detection for exam-generation prep.

Returns *metadata only* — never modifies the corpus the optimizer sees.
The optimization loop must score trial configurations against the same
documents the user will deploy against, duplicates included; otherwise a
configuration that wins under a deduplicated view will under-perform on
the user's real data.

Pipeline:
  1. Tokenize each document with a normalising regex (lowercase, word
     characters only) so OCR noise and Unicode quirks don't destroy
     n-gram matches.
  2. Compute a 5-token-n-gram hash set per document (stable 64-bit hashes).
  3. Pairwise containment via sparse-matmul: an (n_docs × n_ngrams)
     0/1 presence matrix gives all intersection counts as X @ X.T;
     containment = intersection / min(|set_i|, |set_j|).
  4. Cluster documents whose containment is at or above the threshold via
     union-find.
  5. For each cluster, pick the longest document as the canonical and
     emit ``(canonical_doc_ids, duplicate_clusters)``.

We use containment rather than Jaccard because Jaccard is symmetric and
penalises size mismatch — a one-page image whose tokens are a subset of
a multi-page PDF gets Jaccard ≈ 1/N. Containment normalises by the
smaller document, so a true subset reaches 1.0. Anything Jaccard catches
at threshold T, containment catches at threshold T as well (containment
≥ Jaccard always), so containment subsumes Jaccard at the same threshold.

Detection is **purely content-based**. Filenames are deliberately not
consulted: file-naming conventions are corpus-specific and this code
is meant to generalise. A consequence is that parser-induced word
reordering on heavily templated content (e.g. publisher front-page
adverts, where the PDF parser and PNG OCR emit identical words in
different reading orders) can defeat n-gram-based matching. Such cases
fall through to the LLM-side composition refusals downstream.

The downstream consumers (``ExamAgent.prepare_corpus`` and the validator
BM25 builder) read ``canonical_doc_ids`` and skip the rest. The
``OpenEndedEvaluator`` reads ``duplicate_clusters`` to canonicalize
retrieved doc_ids when scoring chunk relevance.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Token-level n-gram size for the document fingerprint. 5 is the standard
# choice for near-duplicate detection: long enough that random sentence-
# level overlap doesn't dominate, short enough that minor edits leave most
# n-grams intact.
_NGRAM_SIZE = 5
_HASH_DIGEST_BYTES = 8

# Word-only normalisation. Lowercases, drops Unicode marks/punctuation,
# and discards single-character tokens (typical OCR garbage). Two docs
# that differ only in capitalisation, dagger / footnote markers, or
# stray punctuation produce the same n-gram set after this pass.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MIN_TOKEN_LEN = 2


@dataclass
class DuplicateClusters:
    """Output of ``detect_near_duplicates``.

    Both fields are pure metadata — the caller does not modify the corpus.
    ``canonical_doc_ids`` is a subset of the input ``doc_ids``;
    ``alias_to_canonical`` maps every input ``doc_id`` to its cluster
    representative (canonical docs map to themselves).
    """

    canonical_doc_ids: list[str]
    alias_to_canonical: dict[str, str]

    @property
    def n_clusters(self) -> int:
        return len(self.canonical_doc_ids)

    @property
    def n_duplicates(self) -> int:
        return sum(1 for alias, canon in self.alias_to_canonical.items() if alias != canon)

    def canonicalize(self, doc_id: str) -> str:
        """Return the canonical id for ``doc_id``, or the input unchanged."""
        return self.alias_to_canonical.get(doc_id, doc_id)


def _stable_hash(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=_HASH_DIGEST_BYTES).digest(), "little")


def _tokenize_normalized(text: str) -> list[str]:
    """Lowercase + word-only token extraction.

    Robust to OCR character-substitution noise: everything outside the
    [a-z0-9] alphabet (Unicode marks, punctuation, dagger characters)
    is dropped, and tokens shorter than ``_MIN_TOKEN_LEN`` are discarded
    so single-character OCR garbage doesn't pollute the n-gram set.
    """
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if len(tok) >= _MIN_TOKEN_LEN]


def _ngram_set(text: str, ngram_size: int = _NGRAM_SIZE) -> frozenset[int]:
    """Return the set of stable 64-bit hashes of token n-grams in ``text``.

    Documents with fewer normalised tokens than ``ngram_size`` produce
    an empty set; they're treated as singletons by the clusterer.
    """
    tokens = _tokenize_normalized(text)
    if len(tokens) < ngram_size:
        return frozenset()
    return frozenset(
        _stable_hash(" ".join(tokens[i : i + ngram_size]).encode("utf-8")) for i in range(len(tokens) - ngram_size + 1)
    )


def _union_find_clusters(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Group indices in [0, n) by transitive closure of ``edges``."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            # Deterministic: smaller index becomes the root, so cluster
            # composition is independent of edge insertion order.
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry

    for x, y in edges:
        union(x, y)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def detect_near_duplicates(
    documents: list[str],
    doc_ids: list[str],
    *,
    threshold: float = 0.85,
    ngram_size: int = _NGRAM_SIZE,
) -> DuplicateClusters:
    """Cluster documents whose containment crosses ``threshold``.

    Containment = |A ∩ B| / min(|A|, |B|). Two documents land in the
    same cluster when the smaller one's tokens are mostly contained in
    the larger. This catches both:
      - symmetric near-duplicates (full-text vs OCR-noisy full-text),
      - asymmetric subsets (a single-page image extracted from a PDF).

    Detection is content-only; filenames are not consulted because file-
    naming conventions are corpus-specific and this code is meant
    to generalise.

    Returns the cluster representatives plus an alias→canonical map that
    covers every input ``doc_id``. The longest document in each cluster
    is chosen as canonical; ties break on lexicographically smaller
    doc_id so the result is deterministic.

    Args:
        documents: parallel to ``doc_ids``; each entry is the full text
            of one document.
        doc_ids: parallel to ``documents``.
        threshold: containment cutoff. 0.85 is a permissive default
            tuned to catch OCR-of-PDF page images even when ~12-15% of
            n-grams diverge; tighten toward 1.0 for stricter clustering.
        ngram_size: token n-gram width used for fingerprinting.
    """
    if len(documents) != len(doc_ids):
        raise ValueError(f"documents ({len(documents)}) and doc_ids ({len(doc_ids)}) must align")
    if not documents:
        return DuplicateClusters(canonical_doc_ids=[], alias_to_canonical={})
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

    n = len(documents)
    logger.info("Detecting near-duplicates across %d documents at containment ≥ %.2f", n, threshold)

    # Stage 1: fingerprint each document as a set of token n-gram hashes.
    ngrams: list[frozenset[int]] = []
    for text in tqdm(documents, desc="Fingerprinting documents", unit="doc"):
        ngrams.append(_ngram_set(text, ngram_size))

    # Stage 2: sparse-matmul pair intersections. Assign each unique n-gram
    # hash to a column index, then build a CSR presence matrix. X @ X.T's
    # nonzeros are exactly the pairs sharing ≥1 n-gram, with values equal
    # to the intersection cardinality.
    ngram_to_col: dict[int, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for i, ng_set in enumerate(ngrams):
        for ng in ng_set:
            col = ngram_to_col.setdefault(ng, len(ngram_to_col))
            rows.append(i)
            cols.append(col)

    edges: list[tuple[int, int]] = []
    if ngram_to_col:
        data = np.ones(len(rows), dtype=np.float32)
        X = sp.csr_matrix(
            (data, (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
            shape=(n, len(ngram_to_col)),
            dtype=np.float32,
        )
        sizes = np.asarray(X.sum(axis=1)).ravel()
        inter = (X @ X.T).tocoo()
        upper = inter.row < inter.col
        rows_u = inter.row[upper]
        cols_u = inter.col[upper]
        data_u = inter.data[upper]
        min_sizes = np.minimum(sizes[rows_u], sizes[cols_u])
        containment = data_u / np.maximum(min_sizes, 1.0)
        above = containment >= threshold
        edges = list(zip(rows_u[above].tolist(), cols_u[above].tolist(), strict=True))

    clusters = _union_find_clusters(n, edges)

    canonical_doc_ids: list[str] = []
    alias_to_canonical: dict[str, str] = {}
    for cluster in clusters:
        # Canonical = longest text; tie-break on smaller doc_id for stability.
        canonical_idx = min(
            cluster,
            key=lambda idx: (-len(documents[idx]), doc_ids[idx]),
        )
        canonical_id = doc_ids[canonical_idx]
        canonical_doc_ids.append(canonical_id)
        for idx in cluster:
            alias_to_canonical[doc_ids[idx]] = canonical_id

    return DuplicateClusters(
        canonical_doc_ids=canonical_doc_ids,
        alias_to_canonical=alias_to_canonical,
    )


__all__ = ["DuplicateClusters", "detect_near_duplicates"]
