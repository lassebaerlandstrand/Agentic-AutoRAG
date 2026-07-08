"""Tests for the agent-owned campaign-plan pydantic (``Strategy``).

The earlier stance/phase machinery (an ``explore``/``refine`` self-label, then a
named ``ceiling``/``frontier`` campaign phase) was removed: the agent now carries
a persistent campaign plan it re-authors each trial as ``plan`` and ``notes``.
There is no per-mode validator — both modes carry the same plan object.
Over-length fields are truncated, not rejected.
"""

from __future__ import annotations

from agentic_autorag.optimizer.diagnosis import Strategy


class TestStrategyPydantic:
    def test_defaults_are_empty(self) -> None:
        s = Strategy()
        assert s.plan == ""
        assert s.notes == ""

    def test_round_trips_fields(self) -> None:
        s = Strategy(
            plan="hold the second half for the cheap end of the frontier",
            notes="glm-4.7 scored low — likely a fit fluke, not a dead tier",
        )
        assert s.plan.startswith("hold the second half")
        assert "fluke" in s.notes

    def test_field_length_caps_truncate_not_reject(self) -> None:
        # Over-length fields are clipped to the cap rather than raising, so a
        # verbose plan never wastes a Proposer retry.
        assert len(Strategy(plan="x" * 5000).plan) == 4000
        assert len(Strategy(notes="x" * 5000).notes) == 3000
        # At-or-under the cap is left untouched.
        assert Strategy(plan="x" * 4000).plan == "x" * 4000

    def test_non_string_fields_coerce_to_str(self) -> None:
        # A non-string value (e.g. the LLM emits a bare number) is coerced
        # rather than rejected.
        assert Strategy(plan=123).plan == "123"
