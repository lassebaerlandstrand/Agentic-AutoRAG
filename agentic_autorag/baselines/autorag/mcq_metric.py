"""Custom AutoRAG generation metric: MCQ accuracy.

Registered via AutoRAG's metric registry so AutoRAG-MCQ optimizes against the
same MCQ signal our agent + Random + Bayesian baselines use. Mirrors the
``OpenEndedEvaluator`` answer-checking logic: case-insensitive normalized
substring match against any element of the gold-answer list.

This module is intentionally dependency-free (no ``agentic_autorag`` imports)
so it can be loaded by AutoRAG running in a separate venv. Path is passed to
AutoRAG via ``--custom_metric`` or by registering through AutoRAG's metric
registration API in a small wrapper script.
"""

from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    return " ".join(_PUNCT_RE.sub(" ", text.lower()).split())


def mcq_accuracy(pred: str, gt: list[str]) -> float:
    """Return 1.0 if any gold answer is a normalized substring of the prediction.

    Mirrors the substring-tolerant matching in OpenEndedEvaluator so models that
    emit "The correct answer is **Paris**." score correctly when gt == ["Paris"].
    """
    pred_norm = _normalize(pred)
    if not pred_norm:
        return 0.0
    for gold in gt or []:
        gold_norm = _normalize(gold)
        if gold_norm and gold_norm in pred_norm:
            return 1.0
    return 0.0
