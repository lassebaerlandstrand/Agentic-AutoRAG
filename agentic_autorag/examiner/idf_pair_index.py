"""IDF-overlap chunk pairing for cross-document 2-hop seed discovery.

Complements ``embedding_pair_index``. The embedding pairer captures semantic
similarity (good for "two articles about the same kind of thing"); this
pairer captures rare-token overlap (good for "two articles that share a
specific entity"). Fusion across the two rankings — handled by
``seeders.emit_cross_doc_pair_seeds`` — finds bridges neither method finds
alone.

Pair score is the sum of the IDF of the **top-K shared tokens** between two
chunks. Summing only the top-K (rather than every shared token) penalises
near-duplicate articles that share dozens of generic-but-rare tokens —
those collapse to roughly the same top-K total as a true bridge pair that
shares one or two distinctive entities.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from agentic_autorag.examiner.chunk_pair_index import ChunkRecord, Seed

logger = logging.getLogger(__name__)


# Number of highest-IDF shared tokens summed into the pair score. Using a
# small fixed top-K (rather than full sum) collapses series-article pairs
# that share many low-content tokens to the same score floor as genuine
# bridge pairs, letting the rare-token signal drive the ranking.
_BRIDGE_TOPK = 3

# Document-frequency gates for tokens that are allowed into the inverted
# index. Tokens with df<2 cannot bridge any pair (only one chunk has them);
# tokens with df above the corpus-relative ceiling are either stop-word-like
# noise (very low IDF anyway) or would blow up the inverted-index walk's
# O(df²) cost. The ceiling is computed as ``max(int(N*fraction), df_floor+1)``;
# on small corpora (N≤10 with fraction=0.3) the floor-based clamp dominates,
# meaning every token appearing in ≥4 chunks is dropped. Production corpora
# have N≫10 so the fraction governs.
_DF_FLOOR = 2
_DF_CEILING_FRACTION = 0.3


# Smooth-IDF formula matches the textbook variant used by sklearn's
# TfidfVectorizer: log((N+1)/(df+1)) + 1. The "+1" floor keeps every
# admitted token contributing strictly positive weight.
def _idf(n_chunks: int, df: int) -> float:
    return math.log((n_chunks + 1) / (df + 1)) + 1.0


# Minimal English stop-list. IDF down-weights frequent words to near zero
# already, but explicit dropping is faster than carrying them through the
# inverted index. Kept short and conservative — anything mid-frequency
# falls through to IDF.
_STOP_TOKENS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "for",
        "to",
        "from",
        "in",
        "on",
        "at",
        "by",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "their",
        "there",
        "here",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        "him",
        "her",
        "them",
        "us",
        "me",
        "my",
        "your",
        "our",
        "his",
        "hers",
        "theirs",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "some",
        "no",
        "not",
        "nor",
        "only",
        "also",
        "too",
        "very",
        "just",
        "so",
        "than",
        "such",
        "own",
        "same",
        "each",
        "every",
        "other",
        "while",
        "during",
        "between",
        "among",
        "into",
        "onto",
        "off",
        "out",
        "up",
        "down",
        "over",
        "under",
        "again",
        "still",
        "yet",
        "always",
        "never",
        "often",
        "sometimes",
        "once",
        "two",
        "three",
        "many",
        "much",
        "more",
        "most",
        "less",
        "fewer",
        "new",
        "old",
        "high",
        "low",
        "big",
        "small",
        "good",
        "bad",
        "first",
        "last",
        "next",
        "early",
        "late",
        "different",
    }
)


# Inline citation patterns stripped before tokenization. They contribute
# no bridgeable content but rare author last names would otherwise rank as
# strong shared "entities" between two chunks that merely cite the same
# paper.
_CITATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "(Smith et al., 2020; Müller and Jones, 2019)" — multi-author parenthetical
    re.compile(
        r"\([A-ZÀ-Ÿ][\w'’\-]+(?:\s+(?:et\s+al\.?|and\s+[A-ZÀ-Ÿ][\w'’\-]+))?(?:,?\s*\d{4}[a-z]?)?(?:;\s*[^)]+)*\)"
    ),
    re.compile(r"\b[A-ZÀ-Ÿ][\w'’\-]+\s+et\s+al\.?(?:\s*\(\d{4}[a-z]?\))?"),
    re.compile(r"\b[A-ZÀ-Ÿ][\w'’\-]+\s+and\s+[A-ZÀ-Ÿ][\w'’\-]+\s+\(\d{4}[a-z]?\)"),
    re.compile(r"\b[A-ZÀ-Ÿ][\w'’\-]+\s+\(\d{4}[a-z]?\)"),
    re.compile(r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]"),
    re.compile(r"\bdoi:?\s*\S+", re.IGNORECASE),
    re.compile(r"\barXiv:\S+", re.IGNORECASE),
    re.compile(r"\bpp?\.\s*\d+(?:[-–]\d+)?"),
)


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9][\w'’\-]*", re.UNICODE)

# Line-break hyphenation in PDFs: "informa-\ntion" → "information".
# Matches LF and CRLF — Windows-extracted PDFs use \r\n line endings.
_LINE_HYPHEN_RE = re.compile(r"(\w)-\r?\n(\w)")
# Soft-hyphen character (U+00AD) used by some PDF extractors.
_SOFT_HYPHEN = "­"


def _normalise(text: str) -> str:
    """Strip PDF artifacts and inline citations before tokenization."""
    text = text.replace(_SOFT_HYPHEN, "")
    text = _LINE_HYPHEN_RE.sub(r"\1\2", text)
    for pattern in _CITATION_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def tokenize(text: str) -> list[str]:
    """Lowercase content tokens; drop stop-words, very short tokens, short numerics."""
    normalised = _normalise(text)
    out: list[str] = []
    for match in _TOKEN_RE.finditer(normalised.lower()):
        tok = match.group(0)
        if tok in _STOP_TOKENS:
            continue
        if tok.isdigit():
            # Keep 4-digit years (potential bridge entities); drop counts/indexes.
            if len(tok) != 4:
                continue
        elif len(tok) < 3:
            continue
        out.append(tok)
    return out


# Multiplier on ``target_count`` used to size the sparse-matmul candidate
# pool: the matmul ranks pairs by the *sum* of all shared IDFs, which is an
# upper bound on the top-K-IDF sum. Taking 5x the target leaves comfortable
# headroom for the refinement step's stable re-ranking to bubble true top-K
# pairs upward without exhaustively scoring every pair.
_CANDIDATE_OVERSAMPLE = 5


def emit_idf_pairs(
    chunks: list[ChunkRecord],
    *,
    target_count: int,
) -> list[Seed]:
    """Emit cross-doc seed pairs ranked by sum of top-K shared-token IDFs.

    Caller is responsible for any section/eligibility filtering — pass the
    already-filtered chunk list.

    Two stages:
      1) Sparse (n_chunks × n_tokens) IDF-weighted presence matrix; compute
         ``S = X @ X.T``. ``S[i,j]`` is the sum of IDFs of all tokens shared
         between chunks ``i`` and ``j``. Same-doc and self pairs are masked.
         Take the top ``5 * target_count`` candidate pairs by this score.
      2) For each candidate, compute the *true* top-K-IDF sum (the algorithm
         we want; previously O(Σ df²) in pure Python). Re-rank candidates by
         this score and return the top ``target_count`` seeds.

    Returns: pairs in descending score order, deduped by canonical
    (chunk_a_id, chunk_b_id), truncated to ``target_count``. Cross-doc only.
    """
    if target_count < 1:
        return []
    if len(chunks) < 2:
        logger.warning("IDF pairing: only %d chunks — no cross-doc pairs possible", len(chunks))
        return []

    t0 = time.perf_counter()
    n = len(chunks)

    df: dict[str, int] = defaultdict(int)
    token_sets: list[frozenset[str]] = []
    for chunk in chunks:
        token_set = frozenset(tokenize(chunk.text))
        token_sets.append(token_set)
        for t in token_set:
            df[t] += 1

    df_ceiling = max(int(n * _DF_CEILING_FRACTION), _DF_FLOOR + 1)
    # Eligible vocabulary: token → contiguous column id + idf weight.
    vocab: dict[str, int] = {}
    idfs: list[float] = []
    for tok, df_t in df.items():
        if df_t < _DF_FLOOR or df_t > df_ceiling:
            continue
        vocab[tok] = len(vocab)
        idfs.append(_idf(n, df_t))

    if not vocab:
        logger.info("Emitted 0 IDF-overlap seeds from %d chunks (no eligible tokens)", n)
        return []

    n_vocab = len(vocab)
    idf_arr = np.asarray(idfs, dtype=np.float32)

    # Build sparse (n_chunks × n_vocab) presence matrix in COO form.
    rows: list[int] = []
    cols: list[int] = []
    for i, token_set in enumerate(token_sets):
        for tok in token_set:
            col = vocab.get(tok)
            if col is None:
                continue
            rows.append(i)
            cols.append(col)
    data = np.ones(len(rows), dtype=np.float32)
    # Weight each presence by sqrt(IDF) so that X @ X.T gives Σ_shared IDF.
    sqrt_idf = np.sqrt(idf_arr)
    weights = sqrt_idf[np.asarray(cols, dtype=np.int64)]
    X = sp.csr_matrix(
        (data * weights, (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n, n_vocab),
        dtype=np.float32,
    )

    # Stage 1: candidate pre-ranking by Σ-shared-IDF via sparse matmul.
    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, 0.0)
    doc_ids = np.array([c.doc_id for c in chunks])
    same_doc_mask = doc_ids[:, None] == doc_ids[None, :]
    sim = np.where(same_doc_mask, 0.0, sim)
    # Upper triangle only — each pair appears once.
    iu = np.triu_indices(n, k=1)
    pair_scores = sim[iu]
    nonzero = pair_scores > 0.0
    if not np.any(nonzero):
        logger.info(
            "Emitted 0 IDF-overlap seeds from %d chunks (no cross-doc token overlap, %d tokens, %.1fs)",
            n,
            n_vocab,
            time.perf_counter() - t0,
        )
        return []
    cand_pair_idx = np.where(nonzero)[0]
    cand_scores = pair_scores[cand_pair_idx]
    candidate_budget = min(len(cand_pair_idx), target_count * _CANDIDATE_OVERSAMPLE)
    if candidate_budget < len(cand_pair_idx):
        top_local = np.argpartition(-cand_scores, kth=candidate_budget - 1)[:candidate_budget]
        cand_pair_idx = cand_pair_idx[top_local]

    # Stage 2: exact top-K-IDF re-ranking on candidate pairs only.
    refined: list[tuple[float, int, int]] = []
    for pair_idx in cand_pair_idx:
        i = int(iu[0][pair_idx])
        j = int(iu[1][pair_idx])
        shared = token_sets[i] & token_sets[j]
        shared_idfs = sorted(
            (idf_arr[vocab[tok]] for tok in shared if tok in vocab),
            reverse=True,
        )
        if not shared_idfs:
            continue
        score = float(sum(shared_idfs[:_BRIDGE_TOPK]))
        refined.append((score, i, j))

    refined.sort(
        key=lambda t: (
            -t[0],
            chunks[t[1]].chunk_id,
            chunks[t[2]].chunk_id,
        )
    )

    seeds: list[Seed] = []
    for score, i, j in refined:
        if len(seeds) >= target_count:
            break
        a, b = chunks[i], chunks[j]
        cid_a, cid_b = sorted((a.chunk_id, b.chunk_id))
        if cid_a == a.chunk_id:
            seeds.append(Seed(chunk_a=a, chunk_b=b, score=score))
        else:
            seeds.append(Seed(chunk_a=b, chunk_b=a, score=score))

    logger.info(
        "Emitted %d IDF-overlap seeds from %d chunks (target=%d, %d eligible tokens, %d candidates refined, %.1fs)",
        len(seeds),
        n,
        target_count,
        n_vocab,
        len(cand_pair_idx),
        time.perf_counter() - t0,
    )
    return seeds
