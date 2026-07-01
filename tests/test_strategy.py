"""Tests for the agent-owned campaign-plan pydantic (``Strategy``).

The earlier stance machinery (an ``explore``/``refine`` self-label plus the
``_validate_stance_for_mode`` pairing rule) was replaced by a persistent
campaign plan the agent re-authors each trial: ``phase``, ``plan``, and
``notes``. There is no per-mode validator — both modes carry the same plan
object; only the phase vocabulary differs (it is set in the prompt, not
validated here). Over-length fields are truncated, not rejected.
"""

from __future__ import annotations

from agentic_autorag.optimizer.diagnosis import INITIAL_PHASE, Strategy


class TestStrategyPydantic:
    def test_defaults_are_empty(self) -> None:
        s = Strategy()
        assert s.phase == ""
        assert s.plan == ""
        assert s.notes == ""

    def test_round_trips_all_fields(self) -> None:
        s = Strategy(
            phase="ceiling",
            plan="ceiling ~0.84; hold the second half for the frontier",
            notes="glm-4.7 scored low — likely a fit fluke, not a dead tier",
        )
        assert s.phase == "ceiling"
        assert s.plan.startswith("ceiling ~0.84")
        assert "fluke" in s.notes

    def test_initial_phase_is_accepted(self) -> None:
        # The orchestrator seeds the first plan with INITIAL_PHASE; it must be a
        # legal phase value the model accepts.
        assert Strategy(phase=INITIAL_PHASE).phase == INITIAL_PHASE

    def test_phase_is_free_text_not_an_enum(self) -> None:
        # Phase vocabulary differs by mode and is prompt-guided, not validated,
        # so any short label is accepted.
        for label in ("ceiling", "frontier", "refine", "anything"):
            assert Strategy(phase=label).phase == label

    def test_field_length_caps_truncate_not_reject(self) -> None:
        # Over-length fields are clipped to the cap rather than raising, so a
        # verbose plan never wastes a Proposer retry.
        assert len(Strategy(phase="x" * 100).phase) == 40
        assert len(Strategy(plan="x" * 5000).plan) == 4000
        assert len(Strategy(notes="x" * 5000).notes) == 3000
        # At-or-under the cap is left untouched.
        assert Strategy(plan="x" * 4000).plan == "x" * 4000

    def test_non_string_fields_coerce_to_str(self) -> None:
        # A non-string value (e.g. the LLM emits a bare number) is coerced
        # rather than rejected.
        assert Strategy(plan=123).plan == "123"
