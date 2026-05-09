"""Tests for the reasoning agent module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_autorag.config.models import (
    IndexType,
    OpenEndedQuestion,
    ProjectConfig,
    SearchSpace,
    TrialConfig,
)
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.optimizer.diagnosis import (
    Bottleneck,
    Diagnosis,
    FrontierContext,
    ProposalMeta,
    TrialMetrics,
)
from agentic_autorag.optimizer.history import HistoryLog
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


def _make_exam_question(qid: str = "q1") -> OpenEndedQuestion:
    return OpenEndedQuestion(
        id=qid,
        question=f"What does {qid} ask?",
        canonical_answer="alpha",
        answer_variants=["alpha-2"],
        reasoning_type="bridge",
        source_chunk_ids=["docA::chunk_0", "docB::chunk_0"],
        source_doc_ids=["docA", "docB"],
        source_spans=[f"chunk A span for {qid}", f"chunk B span for {qid}"],
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
narrative: "retrieval_miss=0.50; the retriever is missing both spans on most failures."
bottlenecks:
  - stage: retrieval
    severity: primary
    evidence: "12 of 20 are retrieval_miss; q07 retrieved 0 chunks from source docs."
  - stage: generation
    severity: secondary
    evidence: "3 of 20 are generation_wrong on arithmetic 2-hop questions."
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
  changes:
    - "embedding_model: sentence-transformers/all-MiniLM-L6-v2 → BAAI/bge-m3"
  rationale: "Diagnoser flagged retrieval primary; bge-m3 has higher MTEB."
  memo:
    - "MiniLM consistently misses span_B on this corpus."
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
    def test_includes_question_and_gold_answer(self) -> None:
        failures = [
            QuestionResult(
                question_id="q1",
                correct=False,
                selected_answer="wrong text",
                correct_answer="alpha",
                retrieved_context="context 1",
                generated_response="wrong text",
                retrieved_doc_ids=["docA", "docB", "docC"],
                em=0.0,
                f1=0.0,
                retrieved_spans=2,
                n_spans=2,
            ),
        ]
        questions_by_id = {"q1": _make_exam_question("q1")}
        tags = {"q1": "generation_wrong"}

        text = ReasoningAgent._format_failures(failures, questions_by_id, tags=tags)

        assert "### generation_wrong 1" in text
        assert "q1" in text
        assert "What does q1 ask?" in text
        assert "alpha" in text
        assert "Predicted answer: wrong text" in text
        assert "failure_mode: generation_wrong" in text
        assert "Retrieval status: complete" in text

    def test_handles_missing_question(self) -> None:
        failures = [
            QuestionResult(
                question_id="qX",
                correct=False,
                selected_answer="some prediction",
                correct_answer="some gold",
                retrieved_context="ctx",
                generated_response="some prediction",
            ),
        ]
        text = ReasoningAgent._format_failures(failures, {})
        assert "<question text unavailable>" in text
        assert "<unavailable>" in text


class TestDiagnoseClassification:
    """Failure classification surfaces per-question failure_mode tags."""

    def _make_result(
        self,
        *,
        qid: str,
        correct: bool,
        retrieved_spans: int,
        n_spans: int = 2,
        refused: bool = False,
        generated_response: str = "B",
    ) -> QuestionResult:
        return QuestionResult(
            question_id=qid,
            correct=correct,
            selected_answer="A" if correct else "B",
            correct_answer="A",
            retrieved_context=f"retrieved {qid}",
            generated_response=generated_response,
            chunk_precision=0.2 if retrieved_spans > 0 else 0.0,
            source_fact_rank=1 if retrieved_spans > 0 else 0,
            retrieved_doc_ids=[f"doc_{qid}"],
            retrieved_spans=retrieved_spans,
            n_spans=n_spans,
            refused=refused,
        )

    def _build_agent(self, tmp_path):
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        return ReasoningAgent(agent_model="test-model", config=cfg, history=history)

    async def test_failure_mode_tags_render_in_prompt(self, tmp_path) -> None:
        results = [
            self._make_result(qid="q1", correct=False, retrieved_spans=0),
            self._make_result(qid="q2", correct=False, retrieved_spans=1),
            self._make_result(qid="q3", correct=False, retrieved_spans=1),
            self._make_result(
                qid="q4",
                correct=False,
                retrieved_spans=0,
                refused=True,
                generated_response="cannot answer based on provided context",
            ),
            self._make_result(qid="q5", correct=False, retrieved_spans=2),
        ]
        exam_result = ExamResult(score=0.0, n_correct=0, n_total=5, question_results=results)
        exam_questions = [_make_exam_question(qr.question_id) for qr in results]

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
                trial_metrics=TrialMetrics(answer_accuracy=0.0, retrieval_complete=0.2),
                trial_number=1,
                trials_remaining=9,
                frontier_context=FrontierContext(),
            )

        prompt = captured["prompt"]
        assert "### retrieval_miss" in prompt
        assert "### retrieval_partial" in prompt
        assert "### refused" in prompt
        assert "### generation_wrong" in prompt
        # Each failure block carries question text.
        for qid in ("q1", "q2", "q3", "q4", "q5"):
            assert f"What does {qid} ask?" in prompt


class TestProposeInitial:
    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_returns_valid_config(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_completion(VALID_INITIAL_YAML))
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        config = await agent.propose_initial("A test corpus.")

        assert isinstance(config, TrialConfig)
        assert config.chunk_token_size == 512
        mock_litellm.acompletion.assert_called_once()

    @patch("agentic_autorag.litellm_runtime.litellm")
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

    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_raises_after_max_retries(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_completion("no yaml at all"))
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        with pytest.raises(RuntimeError, match="Failed to get valid config"):
            await agent.propose_initial("A test corpus.")


class TestAnalyzeAndPropose:
    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_returns_full_tuple(self, mock_litellm, tmp_path) -> None:
        # First call is Diagnoser, second is Proposer.
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _mock_completion(VALID_DIAGNOSIS_YAML),
                _mock_completion(VALID_PROPOSER_YAML),
            ]
        )
        cfg = _make_project_config(llm_models=["ollama/llama3.2"])
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
                    retrieved_spans=2,
                    n_spans=2,
                ),
                QuestionResult(
                    question_id="q2",
                    correct=False,
                    selected_answer="B",
                    correct_answer="A",
                    retrieved_context="ctx",
                    generated_response="B",
                    retrieved_spans=0,
                    n_spans=2,
                ),
            ],
        )

        trial_metrics, diagnosis, next_config, meta = await agent.analyze_and_propose(
            exam_result,
            exam,
            _make_config(),
            trial_number=1,
            trials_remaining=9,
        )

        assert isinstance(trial_metrics, TrialMetrics)
        assert isinstance(diagnosis, Diagnosis)
        assert isinstance(next_config, TrialConfig)
        assert isinstance(meta, ProposalMeta)
        assert next_config.embedding_model == "BAAI/bge-m3"
        # Bottlenecks parsed from YAML, primary first.
        assert len(diagnosis.bottlenecks) == 2
        assert diagnosis.bottlenecks[0].stage == "retrieval"
        assert diagnosis.bottlenecks[0].severity == "primary"
        assert diagnosis.bottlenecks[1].stage == "generation"
        assert meta.changes == ["embedding_model: sentence-transformers/all-MiniLM-L6-v2 → BAAI/bge-m3"]


class TestBuildDiagnosis:
    def test_parses_bottlenecks_and_narrative(self, tmp_path) -> None:
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        diagnosis = agent._build_diagnosis(
            raw=VALID_DIAGNOSIS_YAML,
            trial_metrics=TrialMetrics(answer_accuracy=0.4, retrieval_complete=0.5),
        )

        assert isinstance(diagnosis, Diagnosis)
        assert len(diagnosis.bottlenecks) == 2
        assert diagnosis.bottlenecks[0].stage == "retrieval"
        assert "retriever is missing both spans" in diagnosis.narrative
        # Trial metrics merged in mechanically, not from YAML.
        assert diagnosis.trial_metrics.retrieval_complete == 0.5

    def test_falls_back_when_yaml_missing(self, tmp_path) -> None:
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        # No fenced block — _extract_yaml will raise; _build_diagnosis is only
        # called after a successful parse, so simulate an empty bottleneck list.
        empty_yaml = "narrative-only.\n\n```yaml\nnarrative: empty\n```\n"
        diagnosis = agent._build_diagnosis(
            raw=empty_yaml,
            trial_metrics=TrialMetrics(),
        )

        assert diagnosis.bottlenecks == []
        assert diagnosis.narrative == "empty"


VALID_RECOVERY_YAML = """\
The reranker failed to load — swapping it.

```yaml
chunking_strategy: recursive
chunk_token_size: 512
chunk_token_overlap: 64
embedding_model: sentence-transformers/all-MiniLM-L6-v2
index_type: vector_only
top_k: 5
hybrid_alpha: 0.5
reranker: BAAI/bge-reranker-v2-m3
reranker_top_n: 5
query_expansion: none
llm_model: ollama/llama3.2
temperature: 0.0
reasoning: false
meta:
  changes:
    - "reranker: jinaai/jina-reranker-v2-base-multilingual → BAAI/bge-reranker-v2-m3"
  rationale: "Jina reranker requires trust_remote_code which is not enabled."
  memo:
    - "jinaai reranker is incompatible — drop from candidates."
```
"""


class TestProposeAfterFailure:
    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_returns_alternative_config(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_completion(VALID_RECOVERY_YAML))
        cfg = _make_project_config()
        cfg.search_space.reranker.models = [
            "jinaai/jina-reranker-v2-base-multilingual",
            "BAAI/bge-reranker-v2-m3",
            "none",
        ]
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        failed = _make_config(reranker="jinaai/jina-reranker-v2-base-multilingual")
        config, meta = await agent.propose_after_failure(
            failed_config=failed,
            error_summary="ValueError: requires trust_remote_code=True",
            failure_history=[(failed, "ValueError: requires trust_remote_code=True")],
        )

        assert isinstance(config, TrialConfig)
        assert config.reranker == "BAAI/bge-reranker-v2-m3"
        assert isinstance(meta, ProposalMeta)
        assert meta.changes
        assert "jinaai" in meta.memo[0]


class TestModelDataIntegrity:
    """Pydantic types must reject invalid inputs and accept valid ones."""

    def test_bottleneck_rejects_invalid_stage(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Bottleneck(stage="garbage", severity="primary")  # type: ignore[arg-type]

    def test_bottleneck_rejects_invalid_severity(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Bottleneck(stage="retrieval", severity="critical")  # type: ignore[arg-type]
