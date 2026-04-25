"""Test ``mcq_metric.mcq_accuracy``."""

from __future__ import annotations

import pytest

from agentic_autorag.baselines.autorag.mcq_metric import mcq_accuracy


@pytest.mark.parametrize(
    "pred, gt, expected",
    [
        ("Paris", ["Paris"], 1.0),
        ("paris", ["Paris"], 1.0),  # case-insensitive
        ("Paris.", ["Paris"], 1.0),  # punctuation tolerant
        ("The answer is Paris.", ["Paris"], 1.0),  # substring tolerant
        ("**Paris**", ["Paris"], 1.0),  # markdown bold tolerant
        ("London", ["Paris"], 0.0),
        ("", ["Paris"], 0.0),  # empty pred
        ("Paris", [], 0.0),  # empty gt
        ("Paris", ["Paris", "City of Light"], 1.0),  # multi-gold
        ("city of light", ["Paris", "City of Light"], 1.0),  # second gold
        ("New York City", ["new york"], 1.0),  # spelling/whitespace
    ],
)
def test_mcq_accuracy(pred: str, gt: list[str], expected: float) -> None:
    assert mcq_accuracy(pred, gt) == expected
