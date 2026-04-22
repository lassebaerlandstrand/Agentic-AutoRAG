"""Tests for the reasoning agent module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_autorag.config.models import (
    IndexType,
    MCQQuestion,
    ProjectConfig,
    SearchSpace,
    TrialConfig,
)
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.optimizer.diagnosis import (
    Diagnosis,
    HypothesisCheck,
    MoveType,
    ProposalMeta,
    Stage,
    StageMetrics,
    StateCard,
)
from agentic_autorag.optimizer.history import HistoryLog, TrialRecord
from agentic_autorag.optimizer.reasoning_agent import ReasoningAgent


def _make_project_config(llm_models: list[str] | None = None) -> ProjectConfig:
    return ProjectConfig(
        search_space=SearchSpace(
            embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
            index_types=[IndexType.VECTOR_ONLY],
            llm_models=llm_models or ["ollama/llama3.2"],
        ),
    )


def _make_config(**overrides) -> TrialConfig:
    defaults = dict(
        chunking_strategy="recursive",
        chunk_token_size=512,
        chunk_token_overlap=64,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_type=IndexType.VECTOR_ONLY,
        top_k=5,
        reranker="none",
        reranker_top_n=5,
        llm_model="ollama/llama3.2",
        temperature=0.0,
        reasoning=False,
    )
    defaults.update(overrides)
    return TrialConfig(**defaults)


def _make_exam_question(qid: str = "q1") -> MCQQuestion:
    return MCQQuestion(
        id=qid,
        question=f"What does {qid} ask?",
        options={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        correct_answer="A",
        source_doc_ids=["doc1"],
        cluster_id=0,
    )


VALID_INITIAL_YAML = """\
Reasoning text here...

```yaml
chunking_strategy: recursive
chunk_token_size: 512
chunk_token_overlap: 64
embedding_model: sentence-transformers/all-MiniLM-L6-v2
index_type: vector_only
top_k: 5
hybrid_alpha: 0.5
reranker: none
reranker_top_n: 5
query_expansion: none
llm_model: ollama/llama3.2
temperature: 0.0
reasoning: false
```
"""

VALID_DIAGNOSIS_YAML = """\
Narrative: retrieval is underperforming.

```yaml
bottleneck: retrieval
confidence: medium
narrative: "retrieval_success is 0.3; retrieval-stage levers should help."
applicable_levers:
  - embedding_model
  - chunk_token_size
```
"""

VALID_PROPOSER_YAML = """\
Changing embedding model per diagnosis.

```yaml
chunking_strategy: recursive
chunk_token_size: 512
chunk_token_overlap: 64
embedding_model: BAAI/bge-m3
index_type: vector_only
top_k: 5
hybrid_alpha: 0.5
reranker: none
reranker_top_n: 5
query_expansion: none
llm_model: ollama/llama3.2
temperature: 0.0
reasoning: false
meta:
  move_type: PROBE
  primary_lever: embedding_model
  hypothesis: "swap embedding_model to bge-m3 should raise retrieval_success by +0.10"
  target_metric: retrieval_success
  expected_delta: 0.10
  rationale: "applicable_levers named embedding_model; retrieval is the clear bottleneck."
  memo:
    - "MiniLM underperforms on this corpus"
```
"""


def _mock_completion(content: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


class TestExtractYaml:
    def test_yaml_block(self) -> None:
        text = "some text\n```yaml\nfoo: bar\n```\nmore text"
        result = ReasoningAgent._extract_yaml(text)
        assert result == {"foo": "bar"}

    def test_no_block_raises(self) -> None:
        with pytest.raises(ValueError, match="No YAML block found"):
            ReasoningAgent._extract_yaml("no yaml here")


class TestFormatFailures:
    def test_includes_question_and_options(self) -> None:
        failures = [
            QuestionResult(
                question_id="q1",
                correct=False,
                selected_answer="B",
                correct_answer="A",
                retrieved_context="context 1",
                generated_response="B",
                retrieved_doc_ids=["docA", "docB", "docC"],
            ),
        ]
        questions_by_id = {"q1": _make_exam_question("q1")}

        text = ReasoningAgent._format_failures(failures, questions_by_id)

        assert "Failure 1" in text
        assert "q1" in text
        assert "What does q1 ask?" in text
        assert "alpha" in text  # option A text
        assert "Correct answer: A" in text
        assert "Selected answer: B" in text
        assert "docA" in text
        assert "docB" in text

    def test_handles_missing_question(self) -> None:
        failures = [
            QuestionResult(
                question_id="qX",
                correct=False,
                selected_answer="A",
                correct_answer="B",
                retrieved_context="ctx",
                generated_response="A",
            ),
        ]
        text = ReasoningAgent._format_failures(failures, {})
        assert "<question text unavailable>" in text
        assert "<unavailable>" in text

    def test_tag_renders_in_header(self) -> None:
        failures = [
            QuestionResult(
                question_id="q1",
                correct=True,
                selected_answer="A",
                correct_answer="A",
                retrieved_context="ctx1",
                generated_response="A",
                chunk_precision=0.0,
                source_fact_rank=0,
                retrieved_doc_ids=["docZ"],
            ),
            QuestionResult(
                question_id="q2",
                correct=False,
                selected_answer="B",
                correct_answer="A",
                retrieved_context="ctx2",
                generated_response="B",
                chunk_precision=0.2,
                source_fact_rank=1,
            ),
        ]
        questions_by_id = {q: _make_exam_question(q) for q in ("q1", "q2")}
        tags = {"q1": "Retrieval-miss (correct by guess)", "q2": "Failure"}

        text = ReasoningAgent._format_failures(failures, questions_by_id, tags=tags)

        assert "### Retrieval-miss (correct by guess) 1" in text
        assert "### Failure 2" in text
        # Retrieval-miss block should still carry full diagnostic content.
        assert "What does q1 ask?" in text
        assert "alpha" in text  # option A
        assert "docZ" in text
        # Retrieval-miss should precede Failure in output order (caller-controlled).
        assert text.index("Retrieval-miss") < text.index("### Failure 2")


class TestDiagnoseClassification:
    """The diagnose method classifies question_results into real failures and
    retrieval-miss guesses, rendering both with full detail and prioritising
    real failures when the sample cap is reached."""

    def _make_result(
        self,
        *,
        qid: str,
        correct: bool,
        context_sufficient: bool,
        generated_response: str = "A",
    ) -> QuestionResult:
        return QuestionResult(
            question_id=qid,
            correct=correct,
            selected_answer="A" if correct else "B",
            correct_answer="A",
            retrieved_context=f"retrieved {qid}",
            generated_response=generated_response,
            chunk_precision=0.2 if context_sufficient else 0.0,
            source_fact_rank=1 if context_sufficient else 0,
            retrieved_doc_ids=[f"doc_{qid}"],
        )

    def _build_agent(self, tmp_path):
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        return ReasoningAgent(agent_model="test-model", config=cfg, history=history)

    async def test_both_buckets_rendered_with_tagged_headers(self, tmp_path) -> None:
        real_failures = [self._make_result(qid=f"f{i}", correct=False, context_sufficient=True) for i in range(3)]
        misses = [self._make_result(qid=f"m{i}", correct=True, context_sufficient=False) for i in range(4)]
        exam_result = ExamResult(
            score=0.0,
            n_correct=4,
            n_total=7,
            question_results=real_failures + misses,
        )
        exam_questions = [_make_exam_question(qr.question_id) for qr in real_failures + misses]

        agent = self._build_agent(tmp_path)

        captured: dict[str, str] = {}

        async def _capture_prompt(messages):
            captured["prompt"] = messages[-1]["content"]
            return VALID_DIAGNOSIS_YAML

        with patch.object(agent, "_llm_complete_messages", side_effect=_capture_prompt):
            await agent._diagnose(
                exam_result=exam_result,
                exam_questions=exam_questions,
                current_config=_make_config(),
                stage_metrics=StageMetrics(retrieval_success=0.43, gold_in_reranker_window=0.43),
                hypothesis_check=HypothesisCheck(),
                state_card=MagicMock(
                    trial_number=1,
                    trials_remaining=9,
                    best_score_so_far=0.0,
                    best_trial_number=1,
                    last_trial_delta=0.0,
                    consecutive_non_improvements=0,
                    current_bottleneck=Stage.RETRIEVAL,
                    bottleneck_stable=False,
                    suggested_move_type=MoveType.PROBE,
                    interventions_tried=[],
                    top_trials=[],
                ),
            )

        prompt = captured.get("prompt", "")
        # Real failures appear first with the "Failure" tag (blocks 1..3).
        for i in range(1, 4):
            assert f"### Failure {i}" in prompt, f"missing '### Failure {i}'"
        # Retrieval-miss blocks get the new tag and follow the failures (blocks 4..7).
        for i in range(4, 8):
            assert f"### Retrieval-miss (correct by guess) {i}" in prompt, (
                f"missing '### Retrieval-miss (correct by guess) {i}'"
            )
        # The old Lucky-questions summary must NOT be emitted anymore.
        assert "### Lucky questions" not in prompt
        # Every retrieval-miss block carries full question + options text.
        for miss in misses:
            assert f"What does {miss.question_id} ask?" in prompt
            assert f"Question ID: {miss.question_id}" in prompt

    async def test_real_failures_prioritised_over_misses_at_sample_cap(self, tmp_path) -> None:
        from agentic_autorag.optimizer.reasoning_agent import _MAX_FAILURE_SAMPLE

        real_failures = [
            self._make_result(qid=f"f{i}", correct=False, context_sufficient=True) for i in range(_MAX_FAILURE_SAMPLE)
        ]
        misses = [self._make_result(qid=f"m{i}", correct=True, context_sufficient=False) for i in range(5)]
        exam_result = ExamResult(
            score=0.0,
            n_correct=5,
            n_total=_MAX_FAILURE_SAMPLE + 5,
            question_results=real_failures + misses,
        )
        exam_questions = [_make_exam_question(qr.question_id) for qr in real_failures + misses]

        agent = self._build_agent(tmp_path)

        captured: dict[str, str] = {}

        async def _capture_prompt(messages):
            captured["prompt"] = messages[-1]["content"]
            return VALID_DIAGNOSIS_YAML

        with patch.object(agent, "_llm_complete_messages", side_effect=_capture_prompt):
            await agent._diagnose(
                exam_result=exam_result,
                exam_questions=exam_questions,
                current_config=_make_config(),
                stage_metrics=StageMetrics(retrieval_success=0.75, gold_in_reranker_window=0.75),
                hypothesis_check=HypothesisCheck(),
                state_card=MagicMock(
                    trial_number=1,
                    trials_remaining=9,
                    best_score_so_far=0.0,
                    best_trial_number=1,
                    last_trial_delta=0.0,
                    consecutive_non_improvements=0,
                    current_bottleneck=Stage.GENERATION,
                    bottleneck_stable=False,
                    suggested_move_type=MoveType.PROBE,
                    interventions_tried=[],
                    top_trials=[],
                ),
            )

        prompt = captured.get("prompt", "")
        # All real failures appear; no miss blocks because the cap is full.
        for i in range(_MAX_FAILURE_SAMPLE):
            assert f"Question ID: f{i}" in prompt
        for i in range(5):
            assert f"Question ID: m{i}" not in prompt


class TestProposeInitial:
    @patch("agentic_autorag.optimizer.reasoning_agent.litellm")
    async def test_returns_valid_config(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_completion(VALID_INITIAL_YAML))
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        config = await agent.propose_initial("A test corpus.")

        assert isinstance(config, TrialConfig)
        assert config.chunk_token_size == 512
        mock_litellm.acompletion.assert_called_once()

    @patch("agentic_autorag.optimizer.reasoning_agent.litellm")
    async def test_retry_on_invalid_yaml(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _mock_completion("no yaml here"),
                _mock_completion(VALID_INITIAL_YAML),
            ]
        )
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        config = await agent.propose_initial("A test corpus.")

        assert isinstance(config, TrialConfig)
        assert mock_litellm.acompletion.call_count == 2

    @patch("agentic_autorag.optimizer.reasoning_agent.litellm")
    async def test_raises_after_max_retries(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_completion("no yaml at all"))
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        with pytest.raises(RuntimeError, match="Failed to get valid config"):
            await agent.propose_initial("A test corpus.")


class TestAnalyzeAndPropose:
    @patch("agentic_autorag.optimizer.reasoning_agent.litellm")
    async def test_returns_full_tuple(self, mock_litellm, tmp_path) -> None:
        # First call is Diagnoser, second is Proposer.
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _mock_completion(VALID_DIAGNOSIS_YAML),
                _mock_completion(VALID_PROPOSER_YAML),
            ]
        )
        cfg = _make_project_config(llm_models=["ollama/llama3.2"])
        # Allow the bge-m3 embedding model swap
        cfg.search_space.embedding_models = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-m3"]
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        exam = [_make_exam_question("q1"), _make_exam_question("q2")]
        exam_result = ExamResult(
            score=0.5,
            n_correct=1,
            n_total=2,
            question_results=[
                QuestionResult(
                    question_id="q1",
                    correct=True,
                    selected_answer="A",
                    correct_answer="A",
                    retrieved_context="ctx",
                    generated_response="A",
                    context_sufficient=True,
                ),
                QuestionResult(
                    question_id="q2",
                    correct=False,
                    selected_answer="B",
                    correct_answer="A",
                    retrieved_context="ctx",
                    generated_response="B",
                ),
            ],
        )

        stage_metrics, diagnosis, next_config, meta = await agent.analyze_and_propose(
            exam_result,
            exam,
            _make_config(),
            trial_number=1,
            trials_remaining=9,
        )

        assert isinstance(stage_metrics, StageMetrics)
        assert isinstance(diagnosis, Diagnosis)
        assert isinstance(next_config, TrialConfig)
        assert isinstance(meta, ProposalMeta)
        assert meta.move_type == MoveType.PROBE
        assert meta.primary_lever == "embedding_model"
        assert next_config.embedding_model == "BAAI/bge-m3"
        assert diagnosis.applicable_levers == ["embedding_model", "chunk_token_size"]
        assert diagnosis.bottleneck == Stage.RETRIEVAL
        assert mock_litellm.acompletion.call_count == 2


class TestReconcileStateCard:
    """``_reconcile_state_card`` must recompute ``bottleneck_stable`` against the
    Diagnoser's chosen bottleneck, not the mechanical one. Otherwise the
    recomputed ``suggested_move_type`` reads a stale stability flag."""

    def _state_card(self, *, current: Stage, stable: bool) -> StateCard:
        return StateCard(
            trial_number=2,
            trials_remaining=5,
            best_score_so_far=0.6,
            best_trial_number=1,
            last_trial_delta=0.02,
            consecutive_non_improvements=0,
            current_bottleneck=current,
            bottleneck_stable=stable,
            interventions_tried=[],
            top_trials=[],
            suggested_move_type=MoveType.REFINE,
        )

    def _diagnosis(self, bottleneck: Stage) -> Diagnosis:
        return Diagnosis(
            stage_metrics=StageMetrics(),
            bottleneck=bottleneck,
            confidence="medium",
            hypothesis_check=HypothesisCheck(),
            applicable_levers=[],
            narrative="",
        )

    def _history_with_prior_bottleneck(self, tmp_path, bottleneck: Stage) -> list:
        diag = self._diagnosis(bottleneck)
        return [
            TrialRecord(
                trial_number=1,
                config=_make_config(),
                score=0.55,
                mcq_accuracy=0.6,
                question_results=[],
                diagnosis=diag,
            )
        ]

    def test_unchanged_bottleneck_returns_card_unchanged(self, tmp_path) -> None:
        card = self._state_card(current=Stage.RETRIEVAL, stable=True)
        diag = self._diagnosis(Stage.RETRIEVAL)

        out = ReasoningAgent._reconcile_state_card(card, diag, history_records=[])

        assert out is card  # early return short-circuits model_copy

    def test_override_recomputes_stable_flag_to_true(self, tmp_path) -> None:
        # Mechanical said RETRIEVAL; history's prior diagnosis was GENERATION; Diagnoser now says GENERATION.
        # New bottleneck_stable should be True (matches prior).
        card = self._state_card(current=Stage.RETRIEVAL, stable=False)
        diag = self._diagnosis(Stage.GENERATION)
        history = self._history_with_prior_bottleneck(tmp_path, Stage.GENERATION)

        out = ReasoningAgent._reconcile_state_card(card, diag, history_records=history)

        assert out.current_bottleneck == Stage.GENERATION
        assert out.bottleneck_stable is True

    def test_override_recomputes_stable_flag_to_false(self, tmp_path) -> None:
        # Mechanical said RETRIEVAL (stable=True against prior RETRIEVAL); Diagnoser overrides to GENERATION.
        # New bottleneck_stable must flip to False.
        card = self._state_card(current=Stage.RETRIEVAL, stable=True)
        diag = self._diagnosis(Stage.GENERATION)
        history = self._history_with_prior_bottleneck(tmp_path, Stage.RETRIEVAL)

        out = ReasoningAgent._reconcile_state_card(card, diag, history_records=history)

        assert out.current_bottleneck == Stage.GENERATION
        assert out.bottleneck_stable is False

    def test_override_with_no_prior_history_sets_stable_false(self, tmp_path) -> None:
        card = self._state_card(current=Stage.RETRIEVAL, stable=True)
        diag = self._diagnosis(Stage.RANKING)

        out = ReasoningAgent._reconcile_state_card(card, diag, history_records=[])

        assert out.current_bottleneck == Stage.RANKING
        assert out.bottleneck_stable is False


class TestMoveValidator:
    """Unit tests for the move-type-aware validator on _validate_move."""

    def _agent(self, tmp_path, extra_llms: list[str] | None = None) -> ReasoningAgent:
        cfg = _make_project_config(llm_models=extra_llms or ["ollama/llama3.2", "ollama/llama3.1"])
        cfg.search_space.embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-m3",
        ]
        cfg.search_space.reranker.models = ["none", "BAAI/bge-reranker-v2-m3"]
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        return ReasoningAgent(agent_model="test-model", config=cfg, history=history)

    def _state_card(self, confirmed: list[tuple[str, str]] | None = None):
        """Build a state card for validator tests.

        ``confirmed`` is a list of (lever, value_to) pairs. COMPOUND matches on
        concrete ``value_to``, not just lever name.
        """
        from agentic_autorag.optimizer.diagnosis import StateCard

        entries = [(lever, "", value_to, "confirmed") for lever, value_to in (confirmed or [])]
        return StateCard(
            trial_number=2,
            trials_remaining=5,
            best_score_so_far=0.5,
            best_trial_number=1,
            last_trial_delta=0.02,
            consecutive_non_improvements=0,
            current_bottleneck=Stage.RETRIEVAL,
            bottleneck_stable=True,
            interventions_tried=entries,
            top_trials=[],
            suggested_move_type=MoveType.PROBE,
        )

    def test_probe_requires_single_lever(self, tmp_path) -> None:
        agent = self._agent(tmp_path)
        current = _make_config()
        proposed = _make_config(embedding_model="BAAI/bge-m3", llm_model="ollama/llama3.1")
        meta = ProposalMeta(
            move_type=MoveType.PROBE,
            primary_lever="embedding_model",
            hypothesis="x",
            target_metric="retrieval_success",
            expected_delta=0.05,
        )
        with pytest.raises(ValueError, match="PROBE requires exactly 1"):
            agent._validate_move(current, proposed, meta, self._state_card())

    def test_probe_requires_nontrivial_delta(self, tmp_path) -> None:
        agent = self._agent(tmp_path)
        current = _make_config()
        proposed = _make_config(embedding_model="BAAI/bge-m3")
        meta = ProposalMeta(
            move_type=MoveType.PROBE,
            primary_lever="embedding_model",
            hypothesis="x",
            target_metric="retrieval_success",
            expected_delta=0.001,  # below threshold
        )
        with pytest.raises(ValueError, match="expected_delta"):
            agent._validate_move(current, proposed, meta, self._state_card())

    def test_probe_accepts_valid_single_lever(self, tmp_path) -> None:
        agent = self._agent(tmp_path)
        current = _make_config()
        proposed = _make_config(embedding_model="BAAI/bge-m3")
        meta = ProposalMeta(
            move_type=MoveType.PROBE,
            primary_lever="embedding_model",
            hypothesis="x",
            target_metric="retrieval_success",
            expected_delta=0.05,
        )
        agent._validate_move(current, proposed, meta, self._state_card())

    def test_rejects_wrong_polarity_expected_delta(self, tmp_path) -> None:
        """All tracked metrics are higher-is-better; a -delta predicts a regression."""
        agent = self._agent(tmp_path)
        current = _make_config()
        proposed = _make_config(embedding_model="BAAI/bge-m3")
        meta = ProposalMeta(
            move_type=MoveType.PROBE,
            primary_lever="embedding_model",
            hypothesis="lowering retrieval_success (wrong direction)",
            target_metric="retrieval_success",
            expected_delta=-0.05,  # wrong sign for a higher-is-better metric
        )
        with pytest.raises(ValueError, match="predicts a regression"):
            agent._validate_move(current, proposed, meta, self._state_card())

    def test_refine_rejects_discrete_lever_change(self, tmp_path) -> None:
        agent = self._agent(tmp_path)
        current = _make_config()
        proposed = _make_config(embedding_model="BAAI/bge-m3")
        meta = ProposalMeta(
            move_type=MoveType.REFINE,
            primary_lever="embedding_model",
            hypothesis="x",
            target_metric="retrieval_success",
            expected_delta=0.05,
        )
        with pytest.raises(ValueError, match="REFINE cannot change discrete"):
            agent._validate_move(current, proposed, meta, self._state_card())

    def test_refine_accepts_top_k_small_step(self, tmp_path) -> None:
        agent = self._agent(tmp_path)
        current = _make_config(top_k=5)
        proposed = _make_config(top_k=7)
        meta = ProposalMeta(
            move_type=MoveType.REFINE,
            primary_lever="top_k",
            hypothesis="tweak top_k",
            target_metric="ranking_quality",
            expected_delta=0.02,
        )
        agent._validate_move(current, proposed, meta, self._state_card())

    def test_refine_primary_lever_must_match_change(self, tmp_path) -> None:
        """REFINE is no longer exempt from the primary_lever-matches-changed-lever check."""
        agent = self._agent(tmp_path)
        current = _make_config(chunk_token_size=512, top_k=5)
        proposed = _make_config(chunk_token_size=512, top_k=7)  # only top_k changed
        meta = ProposalMeta(
            move_type=MoveType.REFINE,
            primary_lever="chunk_token_size",  # lies about which lever changed
            hypothesis="lying about the lever",
            target_metric="retrieval_success",
            expected_delta=0.02,
        )
        with pytest.raises(ValueError, match="did not change"):
            agent._validate_move(current, proposed, meta, self._state_card())

    def test_refine_rejects_top_k_large_step(self, tmp_path) -> None:
        agent = self._agent(tmp_path)
        current = _make_config(top_k=5)
        proposed = _make_config(top_k=15)  # delta 10 > 3
        meta = ProposalMeta(
            move_type=MoveType.REFINE,
            primary_lever="top_k",
            hypothesis="tweak top_k",
            target_metric="ranking_quality",
            expected_delta=0.02,
        )
        with pytest.raises(ValueError, match="small-step"):
            agent._validate_move(current, proposed, meta, self._state_card())

    def test_pivot_requires_structural_change(self, tmp_path) -> None:
        agent = self._agent(tmp_path)
        current = _make_config(top_k=5)
        proposed = _make_config(top_k=12)  # not structural
        meta = ProposalMeta(
            move_type=MoveType.PIVOT,
            primary_lever="top_k",
            hypothesis="bigger k",
            target_metric="ranking_quality",
            expected_delta=0.1,
        )
        with pytest.raises(ValueError, match="PIVOT must change at least one structural"):
            agent._validate_move(current, proposed, meta, self._state_card())

    def test_compound_requires_evidence(self, tmp_path) -> None:
        agent = self._agent(tmp_path)
        current = _make_config()
        proposed = _make_config(embedding_model="BAAI/bge-m3", reranker="BAAI/bge-reranker-v2-m3")
        meta = ProposalMeta(
            move_type=MoveType.COMPOUND,
            primary_lever="embedding_model",
            hypothesis="combine known good",
            target_metric="retrieval_success",
            expected_delta=0.05,
        )
        # only embedding_model confirmed — reranker value is unsupported
        with pytest.raises(ValueError, match="Unsupported"):
            agent._validate_move(
                current,
                proposed,
                meta,
                self._state_card(confirmed=[("embedding_model", "BAAI/bge-m3")]),
            )

    def test_compound_rejects_fresh_value_for_confirmed_lever(self, tmp_path) -> None:
        """Confirmation is specific to a concrete value, not the lever name."""
        agent = self._agent(tmp_path)
        current = _make_config()
        proposed = _make_config(embedding_model="BAAI/bge-m3", reranker="BAAI/bge-reranker-v2-m3")
        meta = ProposalMeta(
            move_type=MoveType.COMPOUND,
            primary_lever="embedding_model",
            hypothesis="combine known good",
            target_metric="retrieval_success",
            expected_delta=0.05,
        )
        with pytest.raises(ValueError, match="Unsupported"):
            agent._validate_move(
                current,
                proposed,
                meta,
                # Both levers previously confirmed, but with DIFFERENT values
                self._state_card(
                    confirmed=[
                        ("embedding_model", "different-embed"),
                        ("reranker", "different-reranker"),
                    ]
                ),
            )

    def test_compound_accepts_when_all_values_confirmed(self, tmp_path) -> None:
        agent = self._agent(tmp_path)
        current = _make_config()
        proposed = _make_config(embedding_model="BAAI/bge-m3", reranker="BAAI/bge-reranker-v2-m3")
        meta = ProposalMeta(
            move_type=MoveType.COMPOUND,
            primary_lever="embedding_model",
            hypothesis="combine known good",
            target_metric="retrieval_success",
            expected_delta=0.05,
        )
        agent._validate_move(
            current,
            proposed,
            meta,
            self._state_card(
                confirmed=[
                    ("embedding_model", "BAAI/bge-m3"),
                    ("reranker", "BAAI/bge-reranker-v2-m3"),
                ]
            ),
        )

    def test_revert_requires_reference_trial(self, tmp_path) -> None:
        agent = self._agent(tmp_path)
        current = _make_config()
        proposed = _make_config(embedding_model="BAAI/bge-m3")
        meta = ProposalMeta(
            move_type=MoveType.REVERT,
            primary_lever="embedding_model",
            hypothesis="undo and retry",
            target_metric="retrieval_success",
            expected_delta=0.05,
        )
        with pytest.raises(ValueError, match="meta.revert_to_trial"):
            agent._validate_move(current, proposed, meta, self._state_card())

    def test_revert_validates_against_history(self, tmp_path) -> None:
        agent = self._agent(tmp_path)
        # Seed history with trial 1 using a certain config
        baseline_cfg = _make_config(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        agent.history.records.append(
            TrialRecord(
                trial_number=1,
                config=baseline_cfg,
                score=0.6,
                question_results=[],
            )
        )
        # proposed config differs from baseline by one primary lever
        proposed = _make_config(embedding_model="BAAI/bge-m3")
        meta = ProposalMeta(
            move_type=MoveType.REVERT,
            primary_lever="embedding_model",
            hypothesis="retry with different embedding",
            target_metric="retrieval_success",
            expected_delta=0.05,
            revert_to_trial=1,
        )
        agent._validate_move(_make_config(top_k=12), proposed, meta, self._state_card())
