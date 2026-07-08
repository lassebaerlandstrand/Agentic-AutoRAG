"""Tests for the low-coverage trial guard (``_check_trial_coverage``).

A trial whose evaluable fraction (n_valid / n_total) is too low must be routed
to failure-recovery rather than scored: excluded (errored) questions inflate the
surviving subset's accuracy toward 1.0 and would otherwise lure the optimizer
into selecting a rate-limited or unavailable generator.
"""

from __future__ import annotations

import pytest

from agentic_autorag.examiner._errors import (
    AllQuestionsErrored,
    InsufficientTrialCoverage,
)
from agentic_autorag.examiner.evaluator import ExamResult
from agentic_autorag.orchestrator import (
    MIN_TRIAL_COVERAGE_FRACTION,
    _check_trial_coverage,
)


def _result(
    n_total: int,
    n_valid: int,
    *,
    all_errored: bool = False,
    error_sentinel: str | None = None,
) -> ExamResult:
    return ExamResult(
        n_correct=n_valid,
        n_total=n_total,
        n_valid=n_valid,
        question_results=[],
        answer_accuracy=1.0 if n_valid else 0.0,
        all_errored=all_errored,
        error_sentinel=error_sentinel,
    )


def test_full_coverage_passes():
    # Arrange: every question evaluable. Act/Assert: no raise.
    _check_trial_coverage(_result(100, 100))


def test_at_threshold_passes():
    # Exactly MIN_TRIAL_COVERAGE_FRACTION of n_total is allowed (strict <).
    n = 100
    _check_trial_coverage(_result(n, int(MIN_TRIAL_COVERAGE_FRACTION * n)))


def test_just_below_threshold_raises():
    with pytest.raises(InsufficientTrialCoverage):
        _check_trial_coverage(_result(100, 49))


def test_catastrophic_coverage_raises_with_counts():
    # The exact failure mode observed: ~2/100 evaluable, accuracy inflated to 1.0.
    with pytest.raises(InsufficientTrialCoverage) as exc:
        _check_trial_coverage(_result(100, 2))
    assert exc.value.n_valid == 2
    assert exc.value.n_total == 100


def test_zero_valid_raises_all_questions_errored():
    # n_valid == 0 is the all-errored case and takes precedence over coverage.
    with pytest.raises(AllQuestionsErrored):
        _check_trial_coverage(_result(100, 0, all_errored=True, error_sentinel="TRANSIENT_LLM_ERROR"))
