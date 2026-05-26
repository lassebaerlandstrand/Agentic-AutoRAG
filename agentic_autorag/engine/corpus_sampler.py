"""Deterministic word-budget sampler over a corpus directory.

Re-embedding per trial dominates wall-clock for large corpora. The sampler
walks a shuffled file list and stops when the running word estimate hits
``word_budget``; the selection is cached so a re-run with unchanged inputs
returns the same subset.

Word estimation: PyMuPDF for PDFs, byte-count × 0.18 for text formats,
0 for everything else (still included for proportional representation).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path

import fitz  # type: ignore[import-untyped]
from tqdm import tqdm

from agentic_autorag.engine._io import SKIP_FILENAMES

logger = logging.getLogger(__name__)

SELECTION_CACHE_FILENAME = "corpus_selection.json"
SELECTION_CACHE_SCHEMA_VERSION = 1

# Bytes-per-word for cheap text-format word estimation. Calibrated on
# HotpotQA wiki-paragraph .md files (~5.5 chars/word incl. whitespace
# + markup overhead) and UTF-8 plain text.
_TEXT_BYTES_PER_WORD_RATIO = 0.18

TEXT_LIKE_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt", ".html", ".adoc", ".csv"})
PDF_EXTENSION = ".pdf"

# TQDM smoothing biases the rate estimate toward the whole-run average.
# PDF parse latency is bursty (5ms-50ms per file depending on page count),
# so the default smoothing=0.3 produces wildly fluctuating ETAs.
_TQDM_SMOOTHING = 0.05


def _estimate_pdf_words(path: Path) -> int:
    """Open with PyMuPDF, concat page text, return whitespace-split word count."""
    try:
        with fitz.open(path) as doc:
            text_parts = [page.get_text("text") for page in doc]
    except Exception as exc:  # noqa: BLE001
        logger.debug("PyMuPDF failed on %s: %s", path, exc)
        return 0
    full = " ".join(text_parts)
    return len(full.split())


def _estimate_text_words(path: Path) -> int:
    """Byte-count heuristic for text-like formats."""
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    return int(size * _TEXT_BYTES_PER_WORD_RATIO)


def _estimate_words(path: Path) -> int:
    """Per-extension dispatch. Unmeasurable formats contribute 0."""
    suffix = path.suffix.lower()
    if suffix == PDF_EXTENSION:
        return _estimate_pdf_words(path)
    if suffix in TEXT_LIKE_EXTENSIONS:
        return _estimate_text_words(path)
    return 0


def _enumerate_corpus_files(corpus_path: Path, parser_extensions: frozenset[str]) -> list[Path]:
    """Walk corpus_path, return parser-supported files in sorted order.

    Mirrors ``orchestrator._load_and_parse_corpus`` enumeration rules: skip
    hidden files (leading ``.``), skip ``SKIP_FILENAMES``, accept text-like
    formats unconditionally, accept anything else only if the parser
    advertises it. Sorted for deterministic hashing.
    """
    eligible: list[Path] = []
    for file_path in sorted(corpus_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        if file_path.name in SKIP_FILENAMES:
            continue
        suffix = file_path.suffix.lower()
        if suffix in TEXT_LIKE_EXTENSIONS or suffix in parser_extensions:
            eligible.append(file_path)
    return eligible


def _hash_listing(files: list[Path]) -> str:
    """SHA-256 over sorted ``relpath:size`` pairs. Detects adds, deletes, and resizes."""
    h = hashlib.sha256()
    for path in files:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        h.update(f"{path}:{size}\n".encode())
    return h.hexdigest()


def _load_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("corpus selection cache at %s is unreadable; regenerating", cache_path)
        return None
    if not isinstance(data, dict) or data.get("schema_version") != SELECTION_CACHE_SCHEMA_VERSION:
        return None
    return data


def _write_cache(
    cache_path: Path,
    *,
    corpus_path: Path,
    listing_hash: str,
    word_budget: int | None,
    sample_seed: int,
    selected_files: list[Path],
    stats: dict,
) -> None:
    payload = {
        "schema_version": SELECTION_CACHE_SCHEMA_VERSION,
        "corpus_path": str(corpus_path),
        "corpus_listing_hash": listing_hash,
        "word_budget": word_budget,
        "sample_seed": sample_seed,
        "selected_files": [str(p) for p in selected_files],
        "stats": stats,
    }
    try:
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("Failed to write corpus selection cache to %s", cache_path, exc_info=True)


def sample_corpus(
    *,
    corpus_path: Path,
    parser_extensions: frozenset[str],
    word_budget: int | None,
    sample_seed: int,
    cache_dir: Path,
) -> list[Path]:
    """Pre-sample the corpus down to ``word_budget`` words.

    Returns a list of file paths to parse. Idempotent: same inputs produce
    the same output. When ``word_budget`` is None the full corpus is
    returned without measurement.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {corpus_path}")

    eligible = _enumerate_corpus_files(corpus_path, parser_extensions)
    if not eligible:
        logger.info("Corpus pre-sampling: 0 file(s) discovered under %s", corpus_path)
        return []

    if word_budget is None:
        logger.info("Corpus pre-sampling disabled (corpus_word_budget=None); using all %d file(s)", len(eligible))
        return eligible

    listing_hash = _hash_listing(eligible)
    cache_path = cache_dir / SELECTION_CACHE_FILENAME
    cached = _load_cache(cache_path)
    if (
        cached is not None
        and cached.get("corpus_path") == str(corpus_path)
        and cached.get("corpus_listing_hash") == listing_hash
        and cached.get("word_budget") == word_budget
        and cached.get("sample_seed") == sample_seed
    ):
        stats = cached.get("stats", {})
        selected_files = [Path(p) for p in cached.get("selected_files", [])]
        # listing_hash matches by file path + size, but a file that was
        # truncated to zero or deleted post-write would still slip past.
        # Cheap existence check — fall through to a fresh walk on any miss.
        missing = [p for p in selected_files if not p.exists()]
        if missing:
            logger.warning(
                "Corpus selection cache invalidated: %d cached file(s) no longer exist (e.g. %s); re-sampling",
                len(missing),
                missing[0],
            )
        else:
            logger.info(
                "Corpus selection cache hit: %d file(s), ~%s words (budget %s, seed %d)",
                len(selected_files),
                _fmt_count(stats.get("estimated_words", 0)),
                _fmt_count(word_budget),
                sample_seed,
            )
            return selected_files

    logger.info(
        "Corpus pre-sampling: word_budget=%s, seed=%d",
        _fmt_count(word_budget),
        sample_seed,
    )
    logger.info("  Discovered %s file(s) (corpus_listing_hash=%s)", _fmt_count(len(eligible)), listing_hash[:8])

    rng = random.Random(sample_seed)
    shuffled = list(eligible)
    rng.shuffle(shuffled)

    selected: list[Path] = []
    extension_counts: Counter[str] = Counter()
    accumulated_words = 0
    limiter_kicked_in = False
    with tqdm(total=len(shuffled), desc="   Measuring corpus", unit="file", smoothing=_TQDM_SMOOTHING) as pbar:
        for path in shuffled:
            # Stop BEFORE the file that would push us over budget so a single
            # giant PDF can't blow past a tight cap by orders of magnitude.
            # When ``selected`` is still empty we always take the first file —
            # an empty selection is worse than a single-file overshoot.
            words = _estimate_words(path)
            if selected and accumulated_words + words > word_budget:
                limiter_kicked_in = True
                pbar.total = len(selected)
                pbar.refresh()
                break
            selected.append(path)
            accumulated_words += words
            extension_counts[path.suffix.lower()] += 1
            pbar.update(1)
            if accumulated_words >= word_budget:
                limiter_kicked_in = True
                pbar.total = len(selected)
                pbar.refresh()
                break

    # Sort the picked subset so downstream parsing / doc_ids are stable across
    # runs and easier to reason about. The shuffle is what randomises which
    # files get picked; the final ordering need not match it.
    selected.sort()

    stats = {
        "n_files_in": len(eligible),
        "n_files_out": len(selected),
        "estimated_words": accumulated_words,
        "limiter_kicked_in": limiter_kicked_in,
        "extension_counts": dict(extension_counts),
    }

    if limiter_kicked_in:
        logger.info(
            "  Selected %s / %s file(s); estimated %s words",
            _fmt_count(len(selected)),
            _fmt_count(len(eligible)),
            _fmt_count(accumulated_words),
        )
        logger.info("  Limiter active: skipping %s file(s)", _fmt_count(len(eligible) - len(selected)))
    else:
        logger.info(
            "  All %s file(s) fit within budget (estimated %s words). No trimming.",
            _fmt_count(len(selected)),
            _fmt_count(accumulated_words),
        )

    _write_cache(
        cache_path,
        corpus_path=corpus_path,
        listing_hash=listing_hash,
        word_budget=word_budget,
        sample_seed=sample_seed,
        selected_files=selected,
        stats=stats,
    )
    logger.info("  Wrote selection cache to %s", cache_path.name)
    return selected


def _fmt_count(n: int) -> str:
    """Render a count with thousands underscores (e.g. ``2_000_000``)."""
    return f"{n:_}"
