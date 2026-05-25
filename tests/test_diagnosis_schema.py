"""Tests for ``_build_diagnosis`` validation (qid membership)."""

from __future__ import annotations

import pytest

from agentic_autorag.config.models import (
    EmbeddingSearchSpace,
    GeneratorSearchSpace,
    IndexType,
    ProjectConfig,
    RetrievalSearchSpace,
    SearchSpace,
)
from agentic_autorag.optimizer.diagnosis import TrialMetrics
from agentic_autorag.optimizer.history import HistoryLog
from agentic_autorag.optimizer.reasoning_agent import ReasoningAgent


def _make_agent(tmp_path) -> ReasoningAgent:
    cfg = ProjectConfig(
        search_space=SearchSpace(
            embedding=EmbeddingSearchSpace(models=["sentence-transformers/all-MiniLM-L6-v2"]),
            retrieval=RetrievalSearchSpace(index_types=[IndexType.VECTOR_ONLY]),
            generator=GeneratorSearchSpace(models=["ollama/llama3.2"]),
        ),
    )
    history = HistoryLog(path=str(tmp_path / "history.jsonl"))
    return ReasoningAgent(agent_model="test-model", config=cfg, history=history)


class TestIllustrativeQidsValidation:
    def test_qids_must_belong_to_exam(self, tmp_path) -> None:
        agent = _make_agent(tmp_path)
        yaml = """
```yaml
narrative: "x"
illustrative_qids: [q1, q2, qXX]
```
"""
        with pytest.raises(ValueError, match="qXX"):
            agent._build_diagnosis(
                raw=yaml,
                trial_metrics=TrialMetrics(),
                exam_qids={"q1", "q2"},
            )

    def test_diagnosis_parses_minimal_yaml(self, tmp_path) -> None:
        """The slimmed Diagnoser only requires narrative; lists are optional."""
        agent = _make_agent(tmp_path)
        yaml = """
```yaml
narrative: "the run looks fine."
```
"""
        diagnosis = agent._build_diagnosis(
            raw=yaml,
            trial_metrics=TrialMetrics(answer_accuracy=0.5),
            exam_qids=set(),
        )
        assert diagnosis.narrative == "the run looks fine."
        assert diagnosis.illustrative_qids == []
        assert diagnosis.confirmed_findings == []

    def test_diagnosis_caps_lists(self, tmp_path) -> None:
        agent = _make_agent(tmp_path)
        yaml = """
```yaml
narrative: "long lists test"
confirmed_findings: [a, b, c, d, e, f, g]
notable_deltas: [w, x, y, z, q, r]
illustrative_qids: [q1, q2, q3, q4, q5, q6]
```
"""
        diagnosis = agent._build_diagnosis(
            raw=yaml,
            trial_metrics=TrialMetrics(),
            exam_qids={"q1", "q2", "q3", "q4", "q5", "q6"},
        )
        assert len(diagnosis.confirmed_findings) == 5
        assert len(diagnosis.notable_deltas) == 4
        assert len(diagnosis.illustrative_qids) == 5
