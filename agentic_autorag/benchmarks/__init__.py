"""Benchmark adapters: convert public QA datasets into prepared corpora + qa.json."""

from __future__ import annotations

from pathlib import Path

from agentic_autorag.benchmarks.base import BenchmarkAdapter
from agentic_autorag.benchmarks.hotpot_qa import HotpotQAAdapter
from agentic_autorag.benchmarks.multihop_rag import MultiHopRAGAdapter
from agentic_autorag.benchmarks.musique import MuSiQueAdapter
from agentic_autorag.benchmarks.schema import BenchmarkManifest, BenchmarkQAPair

ADAPTERS: dict[str, type[BenchmarkAdapter]] = {
    HotpotQAAdapter.name: HotpotQAAdapter,
    MuSiQueAdapter.name: MuSiQueAdapter,
    MultiHopRAGAdapter.name: MultiHopRAGAdapter,
}


def prepare(
    name: str,
    output_dir: Path,
    split: str = "validation",
    sample_size: int | None = 500,
    seed: int = 42,
    hf_revision: str | None = None,
) -> BenchmarkManifest:
    """Dispatch to the registered adapter by ``name``."""
    if name not in ADAPTERS:
        raise ValueError(f"Unknown benchmark {name!r}. Known: {sorted(ADAPTERS)}")
    adapter = ADAPTERS[name]()
    return adapter.prepare(
        output_dir=Path(output_dir),
        split=split,
        sample_size=sample_size,
        seed=seed,
        hf_revision=hf_revision,
    )


def load_qa(qa_path: Path) -> list[BenchmarkQAPair]:
    """Load a prepared qa.json file into a list of ``BenchmarkQAPair``."""
    import json

    raw = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    return [BenchmarkQAPair(**row) for row in raw]


__all__ = [
    "ADAPTERS",
    "BenchmarkAdapter",
    "BenchmarkManifest",
    "BenchmarkQAPair",
    "HotpotQAAdapter",
    "MuSiQueAdapter",
    "MultiHopRAGAdapter",
    "load_qa",
    "prepare",
]
