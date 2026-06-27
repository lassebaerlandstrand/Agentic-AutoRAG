"""Filesystem helpers and shared corpus-iteration constants."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

# Filenames skipped when iterating a corpus directory (manifest / sidecar files
# that aren't part of the document set).
SKIP_FILENAMES: frozenset[str] = frozenset({"metadata.json"})

# Extensions read directly as text, bypassing the parser pipeline.
DIRECT_READ_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt"})


def iter_corpus_files(corpus_path: Path) -> Iterator[Path]:
    """Yield corpus files in deterministic sorted order, skipping hidden files
    and ``SKIP_FILENAMES``. Shared by the optimizer's corpus loaders and the
    held-out runner so both see the identical doc-id universe, in the identical
    order."""
    for f in sorted(corpus_path.rglob("*")):
        if not f.is_file() or f.name.startswith(".") or f.name in SKIP_FILENAMES:
            continue
        yield f


def load_direct_read_corpus(corpus_path: Path) -> tuple[list[str], list[str]]:
    """Read every ``.md``/``.txt`` file under *corpus_path* into ``(stems, texts)``.

    The single loader the optimizer (for the retrieval-scoring index) and the
    held-out runner share for direct-read corpora, so both index the identical
    doc-id universe — stems, raw text with headings inline — in the identical
    order. Raises on unsupported extensions and on an empty corpus; empty files
    are skipped (their stem never enters the doc-id universe).
    """
    supported: list[Path] = []
    unsupported: list[Path] = []
    for f in iter_corpus_files(corpus_path):
        if f.suffix.lower() in DIRECT_READ_EXTENSIONS:
            supported.append(f)
        else:
            unsupported.append(f)
    if unsupported:
        sample = ", ".join(p.name for p in unsupported[:3])
        raise RuntimeError(
            f"direct-read corpus loader only supports .md/.txt; found {len(unsupported)} "
            f"unsupported file(s) under {corpus_path} (e.g. {sample})."
        )
    if not supported:
        raise RuntimeError(f"No .md/.txt files found under {corpus_path}")

    stems: list[str] = []
    texts: list[str] = []
    for f in supported:
        text = f.read_text(encoding="utf-8").strip()
        if not text:
            continue
        stems.append(f.stem)
        texts.append(text)
    return stems, texts


def atomic_write_text(path: Path, data: str) -> None:
    """Write *data* to *path* via a sibling tempfile + ``os.replace``."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)
