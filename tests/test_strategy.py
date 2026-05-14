"""Tests for the agent-owned Strategy: ratchet, lock-in, and early-exit gate."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_autorag.optimizer.diagnosis import (
    Diagnosis,
    StateCard,
    Strategy,
    TrialMetrics,
)
from agentic_autorag.optimizer.reasoning_agent import (
    _finalize_strategy,
    _validate_strategy_transition,
)


def _state_card(*, done_eligible: bool = True, done_blocked_reason: str | None = None) -> StateCard:
    """Minimal state card with an explicit done-eligibility setting."""
    return StateCard(
        trial_number=4,
        trials_remaining=4,
        best_score_so_far=0.7,
        best_trial_number=2,
        last_trial_delta=0.0,
        done_eligible=done_eligible,
        done_blocked_reason=done_blocked_reason,
    )


def _diagnosis(*, regression_detected: bool = False, regression_axes: list[str] | None = None) -> Diagnosis:
    """Minimal Diagnosis for the strategy-transition validator."""
    return Diagnosis(
        trial_metrics=TrialMetrics(),
        regression_detected=regression_detected,
        regression_axes=regression_axes or [],  # type: ignore[arg-type]
    )


def _strategy(
    stance: str,
    *,
    intent: str = "",
    committed_at_trial: int = 1,
    revision_count: int = 0,
    done_reason: str | None = None,
    regression_reason: str | None = None,
) -> Strategy:
    return Strategy(
        stance=stance,  # type: ignore[arg-type]
        intent=intent,
        committed_at_trial=committed_at_trial,
        revision_count=revision_count,
        done_reason=done_reason,  # type: ignore[arg-type]
        regression_reason=regression_reason,
    )


class TestRatchetForward:
    def test_search_to_polish_legal(self) -> None:
        prev = _strategy("search", committed_at_trial=1)
        proposed = _strategy("polish")
        _validate_strategy_transition(
            previous=prev,
            proposed=proposed,
            intended_trial=3,
            last_diagnosis=_diagnosis(),
            state_card=_state_card(),
            min_stance_lock_trials=1,
        )

    def test_polish_to_done_legal(self) -> None:
        prev = _strategy("polish", committed_at_trial=3)
        proposed = _strategy("done", done_reason="score_plateau_at_target")
        _validate_strategy_transition(
            previous=prev,
            proposed=proposed,
            intended_trial=5,
            last_diagnosis=_diagnosis(),
            state_card=_state_card(done_eligible=True),
            min_stance_lock_trials=1,
        )

    def test_search_directly_to_done_legal_when_eligible(self) -> None:
        prev = _strategy("search", committed_at_trial=1)
        proposed = _strategy("done", done_reason="converged_no_new_information")
        _validate_strategy_transition(
            previous=prev,
            proposed=proposed,
            intended_trial=3,
            last_diagnosis=_diagnosis(),
            state_card=_state_card(done_eligible=True),
            min_stance_lock_trials=1,
        )

    def test_same_stance_always_legal(self) -> None:
        prev = _strategy("search", committed_at_trial=1)
        proposed = _strategy("search", intent="now narrowed to retrieval")
        _validate_strategy_transition(
            previous=prev,
            proposed=proposed,
            intended_trial=2,
            last_diagnosis=_diagnosis(),
            state_card=_state_card(),
            min_stance_lock_trials=1,
        )


class TestRatchetTerminal:
    def test_done_is_terminal_no_transition_out(self) -> None:
        prev = _strategy("done", done_reason="score_plateau_at_target", committed_at_trial=4)
        proposed = _strategy("search")
        with pytest.raises(ValueError, match="terminal"):
            _validate_strategy_transition(
                previous=prev,
                proposed=proposed,
                intended_trial=6,
                last_diagnosis=_diagnosis(),
                state_card=_state_card(),
                min_stance_lock_trials=1,
            )


class TestRetreat:
    def test_polish_to_search_requires_regression_flag(self) -> None:
        prev = _strategy("polish", committed_at_trial=3)
        proposed = _strategy("search", regression_reason="cost-cut tanked score")
        with pytest.raises(ValueError, match="regression_detected=true"):
            _validate_strategy_transition(
                previous=prev,
                proposed=proposed,
                intended_trial=5,
                last_diagnosis=_diagnosis(),
                state_card=_state_card(),
                min_stance_lock_trials=1,
            )

    def test_polish_to_search_requires_primary_axis(self) -> None:
        """A regression on cost or retrieval_complete alone doesn't unlock retreat."""
        prev = _strategy("polish", committed_at_trial=3)
        proposed = _strategy("search", regression_reason="cost spiked")
        with pytest.raises(ValueError, match="regression_axes"):
            _validate_strategy_transition(
                previous=prev,
                proposed=proposed,
                intended_trial=5,
                last_diagnosis=_diagnosis(regression_detected=True, regression_axes=["cost"]),
                state_card=_state_card(),
                min_stance_lock_trials=1,
            )

    def test_polish_to_search_requires_regression_reason(self) -> None:
        prev = _strategy("polish", committed_at_trial=3)
        proposed = _strategy("search")
        with pytest.raises(ValueError, match="regression_reason"):
            _validate_strategy_transition(
                previous=prev,
                proposed=proposed,
                intended_trial=5,
                last_diagnosis=_diagnosis(regression_detected=True, regression_axes=["score"]),
                state_card=_state_card(),
                min_stance_lock_trials=1,
            )

    def test_polish_to_search_legal_with_signal_and_reason(self) -> None:
        prev = _strategy("polish", committed_at_trial=3)
        proposed = _strategy("search", regression_reason="cost-cut tanked score")
        _validate_strategy_transition(
            previous=prev,
            proposed=proposed,
            intended_trial=5,
            last_diagnosis=_diagnosis(regression_detected=True, regression_axes=["score"]),
            state_card=_state_card(),
            min_stance_lock_trials=1,
        )

    def test_polish_to_search_legal_with_acc_given_complete_axis(self) -> None:
        prev = _strategy("polish", committed_at_trial=3)
        proposed = _strategy("search", regression_reason="hybrid switch broke generation")
        _validate_strategy_transition(
            previous=prev,
            proposed=proposed,
            intended_trial=5,
            last_diagnosis=_diagnosis(
                regression_detected=True,
                regression_axes=["acc_given_complete"],
            ),
            state_card=_state_card(),
            min_stance_lock_trials=1,
        )


class TestDoneGate:
    def test_done_blocked_when_state_card_says_so(self) -> None:
        prev = _strategy("search", committed_at_trial=1)
        proposed = _strategy("done", done_reason="budget_efficient_stop")
        with pytest.raises(ValueError, match="not currently allowed"):
            _validate_strategy_transition(
                previous=prev,
                proposed=proposed,
                intended_trial=3,
                last_diagnosis=_diagnosis(),
                state_card=_state_card(done_eligible=False, done_blocked_reason="trial 2 below minimum floor"),
                min_stance_lock_trials=1,
            )

    def test_done_requires_done_reason_on_model(self) -> None:
        with pytest.raises(ValidationError):
            Strategy(stance="done")  # type: ignore[call-arg]

    def test_done_reason_only_valid_when_stance_done(self) -> None:
        with pytest.raises(ValidationError):
            Strategy(stance="search", done_reason="budget_efficient_stop")  # type: ignore[arg-type]


class TestLockIn:
    def test_lock_blocks_transition_one_trial_after_commit(self) -> None:
        prev = _strategy("search", committed_at_trial=2)
        proposed = _strategy("polish")
        # intended_trial=3, committed_at=2, lock=1 → earliest legal = 2+1+1=4
        with pytest.raises(ValueError, match="stance lock"):
            _validate_strategy_transition(
                previous=prev,
                proposed=proposed,
                intended_trial=3,
                last_diagnosis=_diagnosis(),
                state_card=_state_card(),
                min_stance_lock_trials=1,
            )

    def test_lock_allows_transition_after_lock_period(self) -> None:
        prev = _strategy("search", committed_at_trial=2)
        proposed = _strategy("polish")
        # intended_trial=4, committed_at=2, lock=1 → earliest legal = 4 → just barely allowed
        _validate_strategy_transition(
            previous=prev,
            proposed=proposed,
            intended_trial=4,
            last_diagnosis=_diagnosis(),
            state_card=_state_card(),
            min_stance_lock_trials=1,
        )

    def test_lock_zero_allows_transitions_immediately(self) -> None:
        prev = _strategy("search", committed_at_trial=1)
        proposed = _strategy("polish")
        _validate_strategy_transition(
            previous=prev,
            proposed=proposed,
            intended_trial=2,
            last_diagnosis=_diagnosis(),
            state_card=_state_card(),
            min_stance_lock_trials=0,
        )


class TestFinalizeStrategy:
    def test_first_strategy_resets_counters(self) -> None:
        proposed = _strategy("search", intent="initial broad sweep")
        final = _finalize_strategy(proposed=proposed, previous=None, intended_trial=1)
        assert final.committed_at_trial == 1
        assert final.revision_count == 0

    def test_stance_transition_bumps_count_and_resets_commit(self) -> None:
        prev = _strategy("search", committed_at_trial=1, revision_count=2)
        proposed = _strategy("polish", intent="cut cost on leader")
        final = _finalize_strategy(proposed=proposed, previous=prev, intended_trial=4)
        assert final.committed_at_trial == 4
        assert final.revision_count == 3

    def test_intent_change_within_same_stance_bumps_count(self) -> None:
        prev = _strategy("search", intent="broad", committed_at_trial=1, revision_count=1)
        proposed = _strategy("search", intent="narrowed to retrieval")
        final = _finalize_strategy(proposed=proposed, previous=prev, intended_trial=2)
        assert final.committed_at_trial == 1
        assert final.revision_count == 2

    def test_identical_continuation_preserves_count(self) -> None:
        prev = _strategy("search", intent="broad sweep", committed_at_trial=1, revision_count=1)
        proposed = _strategy("search", intent="broad sweep")
        final = _finalize_strategy(proposed=proposed, previous=prev, intended_trial=2)
        assert final.committed_at_trial == 1
        assert final.revision_count == 1


class TestProposalMetaSerialization:
    def test_strategy_roundtrips_through_jsonl(self, tmp_path) -> None:
        from agentic_autorag.optimizer.diagnosis import ProposalMeta
        from agentic_autorag.optimizer.history import HistoryLog
        from tests.test_history import _make_diagnosis, _make_record

        meta = ProposalMeta(
            changes=["top_k: 5 → 7"],
            rationale="widening retrieval",
            strategy=Strategy(
                stance="polish",
                intent="cut LLM cost on leader",
                anchor_trial=2,
                committed_at_trial=3,
                revision_count=4,
                journal="leader at trial 2; swapping to haiku for cost",
            ),
        )
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        rec = _make_record(1, 0.6)
        rec.diagnosis = _make_diagnosis()
        rec.meta = meta
        log.add(rec)

        # Reload from disk and check the strategy round-trip.
        log2 = HistoryLog(path=str(tmp_path / "history.jsonl"))
        loaded = log2.records[0]
        assert loaded.meta is not None
        assert loaded.meta.strategy is not None
        s = loaded.meta.strategy
        assert s.stance == "polish"
        assert s.intent == "cut LLM cost on leader"
        assert s.anchor_trial == 2
        assert s.committed_at_trial == 3
        assert s.revision_count == 4
        assert "haiku" in s.journal
