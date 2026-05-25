"""Tests for the stance-mode pairing validator and the Strategy pydantic.

The previous stance-lattice (search → polish → done with lock-in,
regression-gated retreat, and done-eligibility gate) was removed in favour
of a self-declared stance: ``explore`` (score-chasing) or ``refine``
(cost-chasing), only declared in cost-aware mode. These tests cover the
``_validate_stance_for_mode`` pairing rule and the slimmed pydantic.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_autorag.optimizer.diagnosis import Strategy
from agentic_autorag.optimizer.reasoning_agent import _validate_stance_for_mode


class TestValidateStanceForMode:
    def test_cost_aware_accepts_explore(self) -> None:
        _validate_stance_for_mode(stance="explore", cost_aware=True)

    def test_cost_aware_accepts_refine(self) -> None:
        _validate_stance_for_mode(stance="refine", cost_aware=True)

    def test_cost_aware_rejects_none(self) -> None:
        with pytest.raises(ValueError, match="required in cost-aware mode"):
            _validate_stance_for_mode(stance=None, cost_aware=True)

    def test_cost_aware_rejects_invalid_label(self) -> None:
        with pytest.raises(ValueError, match="required in cost-aware mode"):
            _validate_stance_for_mode(stance="search", cost_aware=True)

    def test_score_only_accepts_none(self) -> None:
        _validate_stance_for_mode(stance=None, cost_aware=False)

    def test_score_only_rejects_explore(self) -> None:
        with pytest.raises(ValueError, match="must be omitted in score-only mode"):
            _validate_stance_for_mode(stance="explore", cost_aware=False)

    def test_score_only_rejects_refine(self) -> None:
        with pytest.raises(ValueError, match="must be omitted in score-only mode"):
            _validate_stance_for_mode(stance="refine", cost_aware=False)


class TestStrategyPydantic:
    def test_default_stance_is_none(self) -> None:
        s = Strategy()
        assert s.stance is None
        assert s.journal == ""

    def test_stance_accepts_explore_and_refine(self) -> None:
        assert Strategy(stance="explore").stance == "explore"
        assert Strategy(stance="refine").stance == "refine"

    def test_stance_rejects_legacy_labels(self) -> None:
        for legacy in ("search", "polish", "done"):
            with pytest.raises(ValidationError):
                Strategy(stance=legacy)

    def test_journal_max_length_is_6000_chars(self) -> None:
        """The journal cap was bumped from 4000 to 6000 chars (~1500 tokens)
        to give the LLM more working memory while still bounding context."""
        Strategy(journal="x" * 6000)
        with pytest.raises(ValidationError):
            Strategy(journal="x" * 6001)
