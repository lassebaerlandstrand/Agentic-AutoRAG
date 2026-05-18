"""Near-duplicate document detection for exam-generation prep.

Returns *metadata only* — never modifies the corpus the optimizer sees.
The optimization loop must score trial configurations against the same
documents the user will deploy against, duplicates included; otherwise a
configuration that wins under a deduplicated view will under-perform on
the user's real data.

Pipeline:
  1. Tokenize each document with a normalising regex (lowercase, word
     characters only) so OCR noise and Unicode quirks don't destroy
     shingle matches.
  2. Compute a 5-shingle hash set per document (stable 64-bit hashes).
  3. Compute pairwise containment over those sets:
        |A ∩ B| / min(|A|, |B|)
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
different reading orders) can defeat shingle-based matching. Such cases
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

logger = logging.getLogger(__name__)

# Token-level n-gram size for shingles. 5 is the standard choice for
# near-duplicate detection: long enough that random sentence-level overlap
# doesn't dominate, short enough that minor edits leave most shingles intact.
_SHINGLE_SIZE = 5
_HASH_DIGEST_BYTES = 8

# Word-only normalisation. Lowercases, drops Unicode marks/punctuation,
# and discards single-character tokens (typical OCR garbage). Two docs
# that differ only in capitalisation, dagger / footnote markers, or
# stray punctuation produce the same shingle set after this pass.
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
    so single-character OCR garbage doesn't pollute the shingle set.
    """
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if len(tok) >= _MIN_TOKEN_LEN]


def _shingle_set(text: str, shingle_size: int = _SHINGLE_SIZE) -> frozenset[int]:
    """Return the set of stable 64-bit hashes of token n-grams in ``text``.

    Documents with fewer normalised tokens than ``shingle_size`` produce
    an empty set; they're treated as singletons by the clusterer.
    """
    tokens = _tokenize_normalized(text)
    if len(tokens) < shingle_size:
        return frozenset()
    return frozenset(
        _stable_hash(" ".join(tokens[i : i + shingle_size]).encode("utf-8"))
        for i in range(len(tokens) - shingle_size + 1)
    )


def _containment(a: frozenset[int], b: frozenset[int]) -> float:
    """Asymmetric containment: |A ∩ B| / min(|A|, |B|).

    Catches "small fully inside large" — e.g. a single-page image's
    shingles are (mostly) a subset of the multi-page PDF's shingles.
    Symmetric Jaccard underweights this case because it normalises by
    the union (which the larger doc dominates).
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / min(len(a), len(b))


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
    shingle_size: int = _SHINGLE_SIZE,
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
            shingles diverge; tighten toward 1.0 for stricter clustering.
        shingle_size: token n-gram width used for fingerprinting.
    """
    if len(documents) != len(doc_ids):
        raise ValueError(f"documents ({len(documents)}) and doc_ids ({len(doc_ids)}) must align")
    if not documents:
        return DuplicateClusters(canonical_doc_ids=[], alias_to_canonical={})
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

    n = len(documents)
    shingles = [_shingle_set(text, shingle_size) for text in documents]

    # All-pairs comparison. For corpus sizes typical of this tool
    # (hundreds of documents) this is fine; if the corpus grows past a
    # few thousand documents, swap in MinHash + LSH.
    edges: list[tuple[int, int]] = []
    for i in range(n):
        if not shingles[i]:
            continue
        for j in range(i + 1, n):
            if not shingles[j]:
                continue
            if _containment(shingles[i], shingles[j]) >= threshold:
                edges.append((i, j))

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

    n_dups = len(documents) - len(canonical_doc_ids)
    if n_dups:
        logger.info(
            "Near-duplicate detection: %d documents → %d clusters (%d duplicates merged at containment>=%.2f)",
            len(documents),
            len(canonical_doc_ids),
            n_dups,
            threshold,
        )
    else:
        logger.info(
            "Near-duplicate detection: %d documents, no duplicates above containment>=%.2f",
            len(documents),
            threshold,
        )

    return DuplicateClusters(
        canonical_doc_ids=canonical_doc_ids,
        alias_to_canonical=alias_to_canonical,
    )


__all__ = ["DuplicateClusters", "detect_near_duplicates"]
