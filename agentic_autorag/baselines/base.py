"""Shared helpers for baseline drivers.

All baselines (Random, Bayesian, AutoRAG-RAGAS, AutoRAG-MCQ) write the same
shape of output: ``best_config.yaml`` (TrialConfig schema, identical to what
``optimize`` produces), ``history.jsonl`` (JSONL of TrialRecord), and
``optimizer_meta.json`` (algorithm + seed + budget + accounting). This module
keeps that shape uniform.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from agentic_autorag.optimizer.history import TrialRecord


@dataclass
class BaselineMeta:
    """Per-run accounting written to ``optimizer_meta.json``."""

    algorithm: str
    seed: int
    max_trials: int
    n_trials_completed: int = 0
    n_validation_rejects: int = 0
    n_pruned: int = 0
    wall_clock_s: float = 0.0
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def write_optimizer_meta(output_dir: Path, meta: BaselineMeta) -> None:
    """Persist ``optimizer_meta.json`` next to history.jsonl + best_config.yaml."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "optimizer_meta.json").write_text(
        json.dumps(meta.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def save_best_config(output_dir: Path, best: TrialRecord | None, *, include_graph: bool) -> None:
    """Persist the winning ``TrialConfig`` as ``best_config.yaml`` (same shape as ``optimize``)."""
    if best is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = best.config.to_prompt_dump(include_graph=include_graph)
    (output_dir / "best_config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
