"""Tests for ``_build_diagnosis`` validation (regression evidence, qid membership)."""

from __future__ import annotations

import pytest

from agentic_autorag.config.models import (
    IndexType,
    ProjectConfig,
    SearchSpace,
    TrialConfig,
)
from agentic_autorag.optimizer.diagnosis import (
    LeverEffectDelta,
    ProposalMeta,
    Strategy,
    TrialMetrics,
)
from agentic_autorag.optimizer.history import HistoryLog, TrialRecord
from agentic_autorag.optimizer.reasoning_agent import ReasoningAgent


def _make_agent(tmp_path, *, regression_threshold: float = 0.03) -> ReasoningAgent:
    cfg = ProjectConfig(
        search_space=SearchSpace(
            embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
            index_types=[IndexType.VECTOR_ONLY],
            llm_models=["ollama/llama3.2"],
        ),
    )
    cfg.meta.regression_threshold = regression_threshold
    history = HistoryLog(path=str(tmp_path / "history.jsonl"))
    return ReasoningAgent(agent_model="test-model", config=cfg, history=history)


def _delta(score: float = 0.0, acc: float = 0.0, rcomp: float = 0.0, cost: float = 0.0) -> LeverEffectDelta:
    return LeverEffectDelta(
        change="top_k: 5 → 10",
        score_delta=score,
        acc_given_complete_delta=acc,
        retrieval_complete_delta=rcomp,
        cost_delta_usd=cost,
    )


REGRESSION_YAML = """\
```yaml
narrative: "score dropped after the chunk-size revert"
failure_attribution: {retrieval: 0.5, generation: 0.5}
regression_detected: true
regression_axes: [score]
illustrative_qids: []
```
"""

REGRESSION_MULTIAXIS_YAML = """\
```yaml
narrative: "everything got worse"
failure_attribution: {retrieval: 0.5, generation: 0.5}
regression_detected: true
regression_axes: [score, acc_given_complete]
illustrative_qids: []
```
"""

REGRESSION_COST_YAML = """\
```yaml
narrative: "cost spiked"
failure_attribution: {retrieval: 0.5, generation: 0.5}
regression_detected: true
regression_axes: [cost]
illustrative_qids: []
```
"""


class TestRegressionValidation:
    def test_accepts_regression_with_axis_evidence(self, tmp_path) -> None:
        agent = _make_agent(tmp_path)
        diagnosis = agent._build_diagnosis(
            raw=REGRESSION_YAML,
            trial_metrics=TrialMetrics(),
            mechanical_attribution=None,
            lever_deltas=[_delta(score=-0.05)],
            exam_qids=set(),
        )
        assert diagnosis.regression_detected is True
        assert diagnosis.regression_axes == ["score"]

    def test_rejects_regression_without_numeric_evidence(self, tmp_path) -> None:
        agent = _make_agent(tmp_path)
        with pytest.raises(ValueError, match="regression_threshold"):
            agent._build_diagnosis(
                raw=REGRESSION_YAML,
                trial_metrics=TrialMetrics(),
                mechanical_attribution=None,
                lever_deltas=[_delta(score=-0.01)],  # below default threshold
                exam_qids=set(),
            )

    def test_rejects_one_unsupported_axis_in_multi(self, tmp_path) -> None:
        agent = _make_agent(tmp_path)
        # Score drops enough but acc_given_complete does not.
        with pytest.raises(ValueError, match="acc_given_complete"):
            agent._build_diagnosis(
                raw=REGRESSION_MULTIAXIS_YAML,
                trial_metrics=TrialMetrics(),
                mechanical_attribution=None,
                lever_deltas=[_delta(score=-0.05, acc=-0.01)],
                exam_qids=set(),
            )

    def test_cost_axis_uses_upward_threshold(self, tmp_path) -> None:
        agent = _make_agent(tmp_path)
        # Cost UP by more than threshold is a regression.
        diagnosis = agent._build_diagnosis(
            raw=REGRESSION_COST_YAML,
            trial_metrics=TrialMetrics(),
            mechanical_attribution=None,
            lever_deltas=[_delta(cost=0.05)],
            exam_qids=set(),
        )
        assert diagnosis.regression_detected is True

    def test_cost_axis_rejects_downward_delta(self, tmp_path) -> None:
        agent = _make_agent(tmp_path)
        with pytest.raises(ValueError, match="cost"):
            agent._build_diagnosis(
                raw=REGRESSION_COST_YAML,
                trial_metrics=TrialMetrics(),
                mechanical_attribution=None,
                lever_deltas=[_delta(cost=-0.05)],  # cost went DOWN; not a regression
                exam_qids=set(),
            )

    def test_threshold_is_configurable(self, tmp_path) -> None:
        # Below default 0.03 but above a relaxed 0.005.
        agent = _make_agent(tmp_path, regression_threshold=0.005)
        diagnosis = agent._build_diagnosis(
            raw=REGRESSION_YAML,
            trial_metrics=TrialMetrics(),
            mechanical_attribution=None,
            lever_deltas=[_delta(score=-0.01)],
            exam_qids=set(),
        )
        assert diagnosis.regression_detected is True

    def test_regression_axes_empty_is_rejected(self, tmp_path) -> None:
        agent = _make_agent(tmp_path)
        yaml = """
```yaml
narrative: "regressed but no axis named"
failure_attribution: {retrieval: 0.5, generation: 0.5}
regression_detected: true
regression_axes: []
illustrative_qids: []
```
"""
        with pytest.raises(ValueError, match="non-empty regression_axes"):
            agent._build_diagnosis(
                raw=yaml,
                trial_metrics=TrialMetrics(),
                mechanical_attribution=None,
                lever_deltas=[_delta(score=-0.05)],
                exam_qids=set(),
            )


class TestIllustrativeQidsValidation:
    def test_qids_must_belong_to_exam(self, tmp_path) -> None:
        agent = _make_agent(tmp_path)
        yaml = """
```yaml
narrative: "x"
failure_attribution: {retrieval: 1.0}
regression_detected: false
illustrative_qids: [q1, q2, qXX]
```
"""
        with pytest.raises(ValueError, match="qXX"):
            agent._build_diagnosis(
                raw=yaml,
                trial_metrics=TrialMetrics(),
                mechanical_attribution=None,
                lever_deltas=[],
                exam_qids={"q1", "q2"},
            )


class TestMechanicalAttributionMerge:
    def test_mechanical_attribution_used_when_agent_emits_zeros(self, tmp_path) -> None:
        from agentic_autorag.optimizer.diagnosis import FailureAttribution

        agent = _make_agent(tmp_path)
        yaml = "```yaml\nnarrative: 'x'\n```\n"  # no failure_attribution
        diagnosis = agent._build_diagnosis(
            raw=yaml,
            trial_metrics=TrialMetrics(),
            mechanical_attribution=FailureAttribution(retrieval=0.7, generation=0.3),
            lever_deltas=[],
            exam_qids=set(),
        )
        assert diagnosis.failure_attribution.retrieval == pytest.approx(0.7)
        assert diagnosis.failure_attribution.generation == pytest.approx(0.3)

    def test_agent_attribution_takes_precedence_when_provided(self, tmp_path) -> None:
        from agentic_autorag.optimizer.diagnosis import FailureAttribution

        agent = _make_agent(tmp_path)
        yaml = """
```yaml
narrative: x
failure_attribution: {retrieval: 0.2, generation: 0.8}
```
"""
        diagnosis = agent._build_diagnosis(
            raw=yaml,
            trial_metrics=TrialMetrics(),
            mechanical_attribution=FailureAttribution(retrieval=1.0),
            lever_deltas=[],
            exam_qids=set(),
        )
        # Agent's non-zero attribution wins.
        assert diagnosis.failure_attribution.retrieval == pytest.approx(0.2)
        assert diagnosis.failure_attribution.generation == pytest.approx(0.8)


# Trial-2 mirror: an integration test against the canonical bug the redesign
# targets. In the original run log, trial 2 switched vector_only → hybrid and
# saw score↑ (+0.05), retrieval_complete↑ (+0.12), but acc_given_complete↓
# (-0.072). The old Diagnoser categorised this as ``progress_signal:
# new_information`` and missed the regression entirely. The new evidence
# pipeline must surface the regression to the Diagnoser so it can flag it.


def _trial2_anchor_record() -> TrialRecord:
    """Trial-1-shaped record (anchor for the trial-2 mirror)."""
    config = TrialConfig(
        chunking_strategy="recursive",
        chunk_token_size=256,
        chunk_token_overlap=0,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_type=IndexType.VECTOR_ONLY,
        top_k=15,
        reranker="none",
        reranker_top_n=5,
        llm_model="ollama/llama3.2",
        temperature=0.0,
    )
    metrics = TrialMetrics(
        answer_accuracy=0.59,
        retrieval_complete=0.64,
        answer_correct_given_complete_retrieval=0.875,
    )
    return TrialRecord(
        trial_number=1,
        config=config,
        score=0.59,
        question_results=[],
        trial_metrics=metrics,
        mean_llm_cost_per_query_usd=0.0040,
        meta=ProposalMeta(strategy=Strategy(stance="search", anchor_trial=None)),
    )


def _trial2_current_state() -> tuple[TrialConfig, TrialMetrics, float]:
    """Trial-2 config + metrics: hybrid switch lifts retrieval but hurts gen."""
    config = TrialConfig(
        chunking_strategy="recursive",
        chunk_token_size=256,
        chunk_token_overlap=0,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_type=IndexType.HYBRID_BM25_VECTOR,
        top_k=15,
        reranker="none",
        reranker_top_n=10,
        llm_model="ollama/llama3.2",
        temperature=0.0,
    )
    metrics = TrialMetrics(
        answer_accuracy=0.64,
        retrieval_complete=0.76,
        answer_correct_given_complete_retrieval=0.803,
    )
    return config, metrics, 0.0050


class TestTrial2RegressionMirror:
    """The canonical bug: surface the acc_given_complete regression in trial 2.

    The old Diagnoser called this ``progress_signal: new_information`` because
    score went UP. The new Diagnoser must see the regression in the lever-
    effect delta table and be able to emit ``regression_detected: true`` with
    ``regression_axes: [acc_given_complete]`` without the validator rejecting it.
    """

    def test_lever_effect_deltas_capture_acc_drop(self) -> None:
        from agentic_autorag.optimizer.state import compute_lever_effect_deltas

        anchor = _trial2_anchor_record()
        cur_config, cur_metrics, cur_cost = _trial2_current_state()

        deltas = compute_lever_effect_deltas(
            history_records=[anchor],
            current_config=cur_config,
            current_metrics=cur_metrics,
            current_cost_usd=cur_cost,
            anchor_trial=1,
        )

        # Two changed levers: index_type and reranker_top_n.
        assert len(deltas) == 2
        change_names = {d.change for d in deltas}
        assert any("index_type:" in c for c in change_names)
        assert any("reranker_top_n:" in c for c in change_names)
        # Each delta carries the SAME aggregate signal across all four axes.
        for d in deltas:
            assert d.score_delta == pytest.approx(0.05, abs=1e-6)
            assert d.retrieval_complete_delta == pytest.approx(0.12, abs=1e-6)
            # Trial-2's acc_given_complete drop — the canonical regression signal.
            assert d.acc_given_complete_delta == pytest.approx(-0.072, abs=1e-6)
            assert d.cost_delta_usd == pytest.approx(0.001, abs=1e-6)

    def test_diagnoser_can_emit_regression_on_acc_axis(self, tmp_path) -> None:
        """The validator must ACCEPT regression_detected=true on acc_given_complete."""
        agent = _make_agent(tmp_path)
        from agentic_autorag.optimizer.state import compute_lever_effect_deltas

        anchor = _trial2_anchor_record()
        cur_config, cur_metrics, cur_cost = _trial2_current_state()
        deltas = compute_lever_effect_deltas(
            history_records=[anchor],
            current_config=cur_config,
            current_metrics=cur_metrics,
            current_cost_usd=cur_cost,
            anchor_trial=1,
        )
        yaml = """
```yaml
narrative: "hybrid switch lifted retrieval_complete +0.12 but tanked acc_given_complete -0.07."
failure_attribution: {retrieval: 0.4, generation: 0.6}
confirmed_findings:
  - "hybrid switch raised retrieval_complete by 0.12 (good)"
  - "but acc_given_complete dropped by 0.07 (bad) — more context, more distractor pressure"
regression_detected: true
regression_axes: [acc_given_complete]
notable_deltas:
  - "+12 pts retrieval_complete vs -7 pts acc_given_complete from the same hybrid switch"
illustrative_qids: []
```
"""
        diagnosis = agent._build_diagnosis(
            raw=yaml,
            trial_metrics=cur_metrics,
            mechanical_attribution=None,
            lever_deltas=deltas,
            exam_qids=set(),
        )
        # Regression accepted — this is the load-bearing assertion.
        assert diagnosis.regression_detected is True
        assert diagnosis.regression_axes == ["acc_given_complete"]
        # And the Diagnoser's narrative + findings carry the trend signal.
        assert any("acc_given_complete" in f for f in diagnosis.confirmed_findings)

    def test_retreat_unlocks_on_acc_axis_regression(self, tmp_path) -> None:
        """Strategy validator: regression on acc_given_complete unlocks polish→search."""
        from agentic_autorag.optimizer.diagnosis import Diagnosis, StateCard
        from agentic_autorag.optimizer.reasoning_agent import _validate_strategy_transition

        diagnosis = Diagnosis(
            trial_metrics=TrialMetrics(),
            regression_detected=True,
            regression_axes=["acc_given_complete"],
        )
        state_card = StateCard(
            trial_number=4,
            trials_remaining=4,
            best_score_so_far=0.64,
            best_trial_number=2,
            last_trial_delta=-0.07,
            done_eligible=True,
        )
        # Retreat polish → search must succeed.
        _validate_strategy_transition(
            previous=Strategy(stance="polish", committed_at_trial=3),
            proposed=Strategy(stance="search", regression_reason="hybrid switch hurt acc_given_complete"),
            intended_trial=5,
            last_diagnosis=diagnosis,
            state_card=state_card,
            min_stance_lock_trials=1,
        )

    def test_retreat_rejected_when_only_retrieval_complete_regresses(self, tmp_path) -> None:
        """retrieval_complete dropping alone is NOT a primary-axis regression."""
        from agentic_autorag.optimizer.diagnosis import Diagnosis, StateCard
        from agentic_autorag.optimizer.reasoning_agent import _validate_strategy_transition

        diagnosis = Diagnosis(
            trial_metrics=TrialMetrics(),
            regression_detected=True,
            regression_axes=["retrieval_complete"],
        )
        state_card = StateCard(
            trial_number=4,
            trials_remaining=4,
            best_score_so_far=0.64,
            best_trial_number=2,
            last_trial_delta=0.0,
            done_eligible=True,
        )
        with pytest.raises(ValueError, match="regression_axes"):
            _validate_strategy_transition(
                previous=Strategy(stance="polish", committed_at_trial=3),
                proposed=Strategy(stance="search", regression_reason="retrieval drift"),
                intended_trial=5,
                last_diagnosis=diagnosis,
                state_card=state_card,
                min_stance_lock_trials=1,
            )
