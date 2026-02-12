"""Tests for the reasoning agent module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_autorag.config.models import (
    GenerationSearchSpace,
    IndexType,
    RuntimeConfig,
    RuntimeSearchSpace,
    SearchSpace,
    StructuralConfig,
    StructuralSearchSpace,
    TrialConfig,
)
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.optimizer.history import HistoryLog
from agentic_autorag.optimizer.reasoning_agent import ReasoningAgent


def _make_search_space() -> SearchSpace:
    return SearchSpace(
        structural=StructuralSearchSpace(
            parsers=["pymupdf4llm"],
            embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
            index_types=[IndexType.VECTOR_ONLY],
        ),
        runtime=RuntimeSearchSpace(
            generation=GenerationSearchSpace(
                llm_models=["ollama/llama3.2"],
            ),
        ),
    )


def _make_config() -> TrialConfig:
    return TrialConfig(
        structural=StructuralConfig(
            parser="pymupdf4llm",
            chunking_strategy="recursive",
            chunk_size=512,
            chunk_overlap=64,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_type=IndexType.VECTOR_ONLY,
        ),
        runtime=RuntimeConfig(
            top_k=5,
            reranker="none",
            llm_model="ollama/llama3.2",
            temperature=0.0,
        ),
    )


VALID_YAML_RESPONSE = """\
Here is my reasoning...

```yaml
structural:
  parser: pymupdf4llm
  chunking_strategy: recursive
  chunk_size: 512
  chunk_overlap: 64
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  index_type: vector_only
runtime:
  top_k: 5
  hybrid_alpha: 0.5
  reranker: none
  reranker_top_n: 5
  query_expansion: none
  llm_model: ollama/llama3.2
  temperature: 0.0
```
"""


def _mock_completion(content: str) -> MagicMock:
    """Build a mock litellm response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


class TestExtractYaml:
    def test_yaml_block(self) -> None:
        text = "some text\n```yaml\nfoo: bar\n```\nmore text"
        result = ReasoningAgent._extract_yaml(text)
        assert result == {"foo": "bar"}

    def test_yml_block(self) -> None:
        text = "some text\n```yml\nfoo: bar\n```\nmore text"
        result = ReasoningAgent._extract_yaml(text)
        assert result == {"foo": "bar"}

    def test_bare_block(self) -> None:
        text = "some text\n```\nfoo: bar\n```\nmore text"
        result = ReasoningAgent._extract_yaml(text)
        assert result == {"foo": "bar"}

    def test_no_block_raises(self) -> None:
        with pytest.raises(ValueError, match="No YAML block found"):
            ReasoningAgent._extract_yaml("no yaml here")


class TestFormatFailures:
    def test_format(self) -> None:
        failures = [
            QuestionResult(
                question_id="q1",
                correct=False,
                selected_answer="B",
                correct_answer="A",
                retrieved_context="context 1",
                generated_response="B",
            ),
        ]
        text = ReasoningAgent._format_failures(failures)
        assert "Failure 1" in text
        assert "q1" in text
        assert "Correct answer: A" in text
        assert "Selected answer: B" in text


class TestProposeInitial:
    @patch("agentic_autorag.optimizer.reasoning_agent.litellm")
    async def test_returns_valid_config(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_completion(VALID_YAML_RESPONSE))
        space = _make_search_space()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", search_space=space, history=history)

        config = await agent.propose_initial("A test corpus.")
        assert isinstance(config, TrialConfig)
        assert config.structural.chunk_size == 512
        mock_litellm.acompletion.assert_called_once()

    @patch("agentic_autorag.optimizer.reasoning_agent.litellm")
    async def test_retry_on_invalid_yaml(self, mock_litellm, tmp_path) -> None:
        """First attempt returns bad YAML, second returns valid."""
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _mock_completion("no yaml here"),
                _mock_completion(VALID_YAML_RESPONSE),
            ]
        )
        space = _make_search_space()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", search_space=space, history=history)

        config = await agent.propose_initial("A test corpus.")
        assert isinstance(config, TrialConfig)
        assert mock_litellm.acompletion.call_count == 2

    @patch("agentic_autorag.optimizer.reasoning_agent.litellm")
    async def test_raises_after_max_retries(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_completion("no yaml at all")
        )
        space = _make_search_space()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", search_space=space, history=history)

        with pytest.raises(RuntimeError, match="Failed to get valid config"):
            await agent.propose_initial("A test corpus.")


class TestAnalyzeAndPropose:
    @patch("agentic_autorag.optimizer.reasoning_agent.litellm")
    async def test_returns_error_trace_and_config(self, mock_litellm, tmp_path) -> None:
        # First call is _diagnose, second call is _propose
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _mock_completion("Error trace: retrieval failures detected"),
                _mock_completion(VALID_YAML_RESPONSE),
            ]
        )
        space = _make_search_space()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", search_space=space, history=history)

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

        error_trace, next_config = await agent.analyze_and_propose(exam_result, _make_config())
        assert "retrieval failures" in error_trace
        assert isinstance(next_config, TrialConfig)
        assert mock_litellm.acompletion.call_count == 2
