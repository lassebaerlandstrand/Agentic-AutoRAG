"""Filesystem helpers and shared corpus-iteration constants."""

from __future__ import annotations

import os
from pathlib import Path

# Filenames skipped when iterating a corpus directory (manifest / sidecar files
# that aren't part of the document set).
SKIP_FILENAMES: frozenset[str] = frozenset({"metadata.json"})

# Extensions read directly as text, bypassing the parser pipeline.
DIRECT_READ_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt"})


def atomic_write_text(path: Path, data: str) -> None:
    """Write *data* to *path* via a sibling tempfile + ``os.replace``."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)
