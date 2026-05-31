"""Tests for the cost-aware recommendation parser in ``final_report``.

The report model emits ``recommended_trial: <n>`` ahead of the markdown body;
``_parse_recommendation`` splits that into ``(trial, body)`` and the caller
validates the trial against the Pareto frontier (with a retry + max-score
fallback). These tests cover the pure parsing layer.
"""

from __future__ import annotations

from agentic_autorag.optimizer.final_report import _parse_recommendation


def test_parses_leading_recommendation_line() -> None:
    raw = "recommended_trial: 7\n\n## Summary\nTrial 7 is the pick.\n"
    trial, body = _parse_recommendation(raw)
    assert trial == 7
    assert body.startswith("## Summary")
    assert "recommended_trial" not in body


def test_parse_is_case_and_space_insensitive() -> None:
    raw = "RECOMMENDED_TRIAL:   12  \n\nbody text"
    trial, body = _parse_recommendation(raw)
    assert trial == 12
    assert body == "body text"


def test_missing_line_returns_none_and_full_body() -> None:
    raw = "## Summary\nNo machine-readable pick here.\n"
    trial, body = _parse_recommendation(raw)
    assert trial is None
    assert body.startswith("## Summary")


def test_strips_surrounding_code_fence() -> None:
    raw = "```\nrecommended_trial: 3\n\n## Summary\nx\n```"
    trial, body = _parse_recommendation(raw)
    assert trial == 3
    assert body.startswith("## Summary")
    assert "```" not in body
