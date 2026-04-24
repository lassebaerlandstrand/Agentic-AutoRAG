"""Adapter protocol shared by all benchmark loaders."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from agentic_autorag.benchmarks.schema import BenchmarkManifest

_SLUG_STRIP_RE = re.compile(r"[^a-zA-Z0-9]+")


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Each benchmark implements this to materialise a prepared directory."""

    name: str
    adapter_version: str

    def prepare(
        self,
        output_dir: Path,
        split: str,
        sample_size: int | None,
        seed: int,
        hf_revision: str | None,
    ) -> BenchmarkManifest: ...


def slugify(text: str, *, used: set[str] | None = None) -> str:
    """Lowercase, strip non-alnum, join with underscores.

    When ``used`` is provided, disambiguate collisions by appending a
    short sha1 suffix derived from the original ``text``. Mutates
    ``used`` to record the returned slug so subsequent calls see it.
    """
    base = _SLUG_STRIP_RE.sub("_", text).strip("_").lower() or "doc"
    if used is None:
        return base
    if base not in used:
        used.add(base)
        return base
    suffix = hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]
    candidate = f"{base}__{suffix}"
    # Extreme edge case: even the sha1-suffixed slug clashes. Extend with
    # a counter rather than silently overwriting.
    counter = 1
    while candidate in used:
        candidate = f"{base}__{suffix}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate
