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


def _seed_history(
    agent: ReasoningAgent,
    *,
    score: float = 0.5,
    acc: float = 0.5,
    rcomp: float = 0.5,
    cost: float = 0.001,
    n_valid: int = 1000,
) -> None:
    """Append one prior trial with the given baseline metrics.

    ``n_valid`` defaults to 1000 so the variance-derived noise floor stays
    small (≈0.03) and tests that assert specific threshold behavior aren't
    inadvertently gated by the CI half-width. Tests exercising variance
    behavior should override n_valid to a smaller value.
    """
    cfg = TrialConfig(
        chunking_strategy="recursive",
        chunk_token_size=512,
        chunk_token_overlap=64,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_type=IndexType.VECTOR_ONLY,
        top_k=5,
        reranker="none",
        reranker_top_n=5,
        generator_llm="ollama/llama3.2",
        temperature=0.0,
        reasoning=False,
    )
    agent.history.records.append(
        TrialRecord(
            trial_number=1,
            config=cfg,
            score=score,
            question_results=[],
            mean_llm_cost_per_query_usd=cost,
            trial_metrics=TrialMetrics(
                answer_accuracy=score,
                answer_correct_given_complete_retrieval=acc,
                retrieval_complete=rcomp,
                mean_llm_cost_per_query_usd=cost,
                n_valid=n_valid,
            ),
        )
    )


class TestRegressionValidation:
    def test_accepts_regression_with_axis_evidence(self, tmp_path) -> None:
        agent = _make_agent(tmp_path)
        _seed_history(agent, score=0.50)
        # Current trial scores 0.40, baseline best is 0.50 → drop of 0.10 >= 0.03.
        diagnosis = agent._build_diagnosis(
            raw=REGRESSION_YAML,
            trial_metrics=TrialMetrics(answer_accuracy=0.40),
            mechanical_attribution=None,
            exam_qids=set(),
        )
        assert diagnosis.regression_detected is True
        assert diagnosis.regression_axes == ["score"]

    def test_rejects_regression_below_threshold(self, tmp_path) -> None:
        agent = _make_agent(tmp_path)
        _seed_history(agent, score=0.50)
        # Drop of 0.01 is below the 0.03 threshold → must be rejected.
        with pytest.raises(ValueError, match="regressed by"):
            agent._build_diagnosis(
                raw=REGRESSION_YAML,
                trial_metrics=TrialMetrics(answer_accuracy=0.49),
                mechanical_attribution=None,
                exam_qids=set(),
            )

    def test_rejects_one_unsupported_axis_in_multi(self, tmp_path) -> None:
        agent = _make_agent(tmp_path)
        _seed_history(agent, score=0.50, acc=0.80)
        # Score drops enough (0.50 → 0.40) but acc_given_complete does not
        # (0.80 → 0.79 = 0.01 drop, below 0.03 threshold).
        with pytest.raises(ValueError, match="acc_given_complete"):
            agent._build_diagnosis(
                raw=REGRESSION_MULTIAXIS_YAML,
                trial_metrics=TrialMetrics(
                    answer_accuracy=0.40,
                    answer_correct_given_complete_retrieval=0.79,
                ),
                mechanical_attribution=None,
                exam_qids=set(),
            )

    def test_cost_axis_uses_upward_threshold(self, tmp_path) -> None:
        agent = _make_agent(tmp_path)
        _seed_history(agent, cost=0.001)
        # Cost rose by 0.05 (>> 0.03 threshold) → cost regression.
        diagnosis = agent._build_diagnosis(
            raw=REGRESSION_COST_YAML,
            trial_metrics=TrialMetrics(mean_llm_cost_per_query_usd=0.051),
            mechanical_attribution=None,
            exam_qids=set(),
        )
        assert diagnosis.regression_detected is True

    def test_cost_axis_rejects_downward_delta(self, tmp_path) -> None:
        agent = _make_agent(tmp_path)
        _seed_history(agent, cost=0.05)
        # Cost dropped (0.05 → 0.001) — not a regression.
        with pytest.raises(ValueError, match="cost"):
            agent._build_diagnosis(
                raw=REGRESSION_COST_YAML,
                trial_metrics=TrialMetrics(mean_llm_cost_per_query_usd=0.001),
                mechanical_attribution=None,
                exam_qids=set(),
            )

    def test_threshold_is_configurable_above_variance_floor(self, tmp_path) -> None:
        """The static threshold can be relaxed when the exam is large enough
        that variance is below the relaxed threshold. With n=100000 and
        p=0.50 the CI half-width is ≈0.003, so a threshold of 0.005 dominates
        and a 0.01 drop registers as a regression."""
        agent = _make_agent(tmp_path, regression_threshold=0.005)
        _seed_history(agent, score=0.50, n_valid=100000)
        diagnosis = agent._build_diagnosis(
            raw=REGRESSION_YAML,
            trial_metrics=TrialMetrics(answer_accuracy=0.49, n_valid=100000),
            mechanical_attribution=None,
            exam_qids=set(),
        )
        assert diagnosis.regression_detected is True

    def test_static_threshold_cannot_go_below_variance_floor(self, tmp_path) -> None:
        """The variance floor is a hard floor: a user can't claim regressions
        inside the 95% CI of the best, regardless of how lax the static
        threshold is."""
        agent = _make_agent(tmp_path, regression_threshold=0.001)
        _seed_history(agent, score=0.50, n_valid=100)
        # CI half-width at n=100 is ≈0.098 — a 0.02 drop is well inside it.
        with pytest.raises(ValueError, match="regressed by"):
            agent._build_diagnosis(
                raw=REGRESSION_YAML,
                trial_metrics=TrialMetrics(answer_accuracy=0.48, n_valid=100),
                mechanical_attribution=None,
                exam_qids=set(),
            )

    def test_regression_axes_empty_is_rejected(self, tmp_path) -> None:
        agent = _make_agent(tmp_path)
        _seed_history(agent, score=0.50)
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
                trial_metrics=TrialMetrics(answer_accuracy=0.40),
                mechanical_attribution=None,
                exam_qids=set(),
            )

    def test_regression_uses_best_not_anchor(self, tmp_path) -> None:
        """Trial K may regress vs the run's best even when lever_effect_deltas
        compare to a much earlier anchor and look positive overall. The
        validation must catch this case via history-best comparison."""
        agent = _make_agent(tmp_path)
        # trial 1 baseline = 0.28; trial 2 became the new best at 0.45.
        _seed_history(agent, score=0.28)
        agent.history.records.append(
            TrialRecord(
                trial_number=2,
                config=agent.history.records[0].config,
                score=0.45,
                question_results=[],
                trial_metrics=TrialMetrics(answer_accuracy=0.45, n_valid=1000),
            )
        )
        # Current trial scores 0.30 — still above trial-1-anchor (0.28) but
        # regressed vs trial-2-best (0.45) by 0.15, well outside noise at n=1000.
        diagnosis = agent._build_diagnosis(
            raw=REGRESSION_YAML,
            trial_metrics=TrialMetrics(answer_accuracy=0.30),
            mechanical_attribution=None,
            exam_qids=set(),
        )
        assert diagnosis.regression_detected is True
        assert diagnosis.regression_axes == ["score"]

    def test_regression_rejected_when_within_variance_noise_floor(self, tmp_path) -> None:
        """At small exam sizes, drops within the 95% CI half-width are noise,
        not signal. The variance-aware threshold must reject them even when
        the static threshold alone would accept."""
        agent = _make_agent(tmp_path)
        # Small exam (n=100) with baseline 0.50 → CI half-width ≈ 0.098.
        # A 0.05 drop is below noise floor and should be rejected.
        _seed_history(agent, score=0.50, n_valid=100)
        with pytest.raises(ValueError, match="regressed by"):
            agent._build_diagnosis(
                raw=REGRESSION_YAML,
                trial_metrics=TrialMetrics(answer_accuracy=0.45, n_valid=100),
                mechanical_attribution=None,
                exam_qids=set(),
            )

    def test_missing_n_valid_falls_back_to_static_threshold(self, tmp_path) -> None:
        """When n_valid is missing/0 on the baseline, the variance term is
        unknown — the check must fall back to the static threshold alone
        rather than guess an exam size."""
        agent = _make_agent(tmp_path)
        # Seed with n_valid=0 explicitly (simulates a malformed historical record).
        _seed_history(agent, score=0.50, n_valid=0)
        # A 0.04 drop is above the 0.03 static threshold; without the variance
        # term widening it, the regression must register.
        diagnosis = agent._build_diagnosis(
            raw=REGRESSION_YAML,
            trial_metrics=TrialMetrics(answer_accuracy=0.46),
            mechanical_attribution=None,
            exam_qids=set(),
        )
        assert diagnosis.regression_detected is True

    def test_regression_accepted_when_outside_variance_noise_floor(self, tmp_path) -> None:
        """Drops larger than the variance noise floor still flag at small n."""
        agent = _make_agent(tmp_path)
        # Same n=100 baseline at 0.50; CI half-width ≈ 0.098.
        # A 0.15 drop is clearly outside noise.
        _seed_history(agent, score=0.50, n_valid=100)
        diagnosis = agent._build_diagnosis(
            raw=REGRESSION_YAML,
            trial_metrics=TrialMetrics(answer_accuracy=0.35, n_valid=100),
            mechanical_attribution=None,
            exam_qids=set(),
        )
        assert diagnosis.regression_detected is True


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
        generator_llm="ollama/llama3.2",
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
        generator_llm="ollama/llama3.2",
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

    def test_bundle_effect_captures_acc_drop(self) -> None:
        from agentic_autorag.optimizer.state import compute_bundle_effect

        anchor = _trial2_anchor_record()
        cur_config, cur_metrics, cur_cost = _trial2_current_state()

        effect = compute_bundle_effect(
            history_records=[anchor],
            current_config=cur_config,
            current_metrics=cur_metrics,
            current_cost_usd=cur_cost,
            anchor_trial=1,
        )

        # Bundle captures BOTH changed levers in a single effect entry.
        assert effect is not None
        assert any("index_type:" in c for c in effect.changes)
        assert any("reranker_top_n:" in c for c in effect.changes)
        # The deltas reflect the bundled effect of the trial-2 hybrid switch.
        assert effect.score_delta == pytest.approx(0.05, abs=1e-6)
        assert effect.retrieval_complete_delta == pytest.approx(0.12, abs=1e-6)
        # Trial-2's acc_given_complete drop — the canonical regression signal.
        assert effect.acc_given_complete_delta == pytest.approx(-0.072, abs=1e-6)
        assert effect.cost_delta_usd == pytest.approx(0.001, abs=1e-6)

    def test_diagnoser_can_emit_regression_on_acc_axis(self, tmp_path) -> None:
        """The validator must ACCEPT regression_detected=true on acc_given_complete."""
        agent = _make_agent(tmp_path)
        # Populate history with the trial-1 record so the regression check has
        # a baseline to compare against. Boost n_valid so the variance floor
        # doesn't swallow the trial-2 acc drop (-0.072 is real signal here).
        anchor = _trial2_anchor_record()
        anchor.trial_metrics = anchor.trial_metrics.model_copy(update={"n_valid": 1000})
        agent.history.records.append(anchor)
        _cur_config, cur_metrics, _cur_cost = _trial2_current_state()
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
