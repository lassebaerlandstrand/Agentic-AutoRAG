"""Tests for the ExamAgent MCQ generation pipeline.

All LLM calls are mocked
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from agentic_autorag.config.models import ExaminerConfig, MCQQuestion
from agentic_autorag.examiner.exam_agent import ExamAgent

VALID_MCQ_JSON = json.dumps(
    {
        "question": "What is the primary purpose of RAG?",
        "options": {
            "A": "Retrieval-Augmented Generation for grounding LLM answers",
            "B": "Random Access Generation for faster inference",
            "C": "Recursive Algorithm for Graph traversal",
            "D": "Real-time Aggregation of Gradients",
        },
        "correct_answer": "A",
    }
)

VALID_MCQ_MARKDOWN_WRAPPED = f"```json\n{VALID_MCQ_JSON}\n```"


def _make_litellm_response(content: str) -> SimpleNamespace:
    """Build a mock litellm response object."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class DummyEmbeddingModel:
    """Simple deterministic embedding stub for exam quality checks."""

    def encode(self, texts: list[str]):
        vectors = []
        for text in texts:
            tokens = text.lower().split()
            vectors.append(
                [
                    float(len(tokens)),
                    float(sum(ord(ch) for ch in text) % 997),
                    float(text.lower().count("rag")),
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


def _make_agent(exam_size: int = 10, diversity_clusters: int = 3) -> ExamAgent:
    config = ExaminerConfig(exam_size=exam_size, diversity_clusters=diversity_clusters)
    agent = ExamAgent(
        config=config,
        examiner_model="gemini/gemini-3-flash-preview",
        embedding_model=DummyEmbeddingModel(),
    )
    agent._check_discriminator_quality = lambda *args, **kwargs: True
    return agent


def _make_clustered_embeddings(
    n_per_cluster: int = 20, n_clusters: int = 3, dim: int = 4
) -> tuple[list[str], list[str], np.ndarray]:
    """Create well-separated synthetic chunks/embeddings for testing."""
    rng = np.random.default_rng(0)
    chunks: list[str] = []
    chunk_ids: list[str] = []
    embeds: list[np.ndarray] = []
    for c in range(n_clusters):
        center = np.zeros(dim)
        center[c % dim] = 100.0  # push clusters apart
        for i in range(n_per_cluster):
            chunks.append(f"Chunk text for cluster {c} item {i}")
            chunk_ids.append(f"chunk_{c}_{i}")
            embeds.append(center + rng.standard_normal(dim) * 0.1)
    return chunks, chunk_ids, np.vstack(embeds)


class TestParseMcqResponse:
    def test_valid_json(self) -> None:
        agent = _make_agent()
        result = agent._parse_mcq_response(VALID_MCQ_JSON, "chunk_0", 0)
        assert result is not None
        assert isinstance(result, MCQQuestion)
        assert result.correct_answer == "A"
        assert result.source_chunk_id == "chunk_0"
        assert result.cluster_id == 0

    def test_markdown_wrapped_json(self) -> None:
        agent = _make_agent()
        result = agent._parse_mcq_response(VALID_MCQ_MARKDOWN_WRAPPED, "chunk_1", 2)
        assert result is not None
        assert result.correct_answer == "A"
        assert result.source_chunk_id == "chunk_1"
        assert result.cluster_id == 2

    def test_invalid_json_returns_none(self) -> None:
        agent = _make_agent()
        result = agent._parse_mcq_response("this is not json", "chunk_0", 0)
        assert result is None

    def test_missing_key_returns_none(self) -> None:
        incomplete = json.dumps({"question": "What?", "options": {"A": "a", "B": "b"}})
        agent = _make_agent()
        result = agent._parse_mcq_response(incomplete, "chunk_0", 0)
        assert result is None

    def test_invalid_correct_answer_returns_none(self) -> None:
        bad_answer = json.dumps(
            {
                "question": "What?",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "correct_answer": "Z",
            }
        )
        agent = _make_agent()
        result = agent._parse_mcq_response(bad_answer, "chunk_0", 0)
        assert result is None

class TestGenerateMcqWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_attempt(self) -> None:
        agent = _make_agent()
        mock_resp = _make_litellm_response(VALID_MCQ_JSON)

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = await agent._generate_mcq_with_retry("Some chunk text", "chunk_0", 0)
        assert result is not None
        assert result.correct_answer in {"A", "B", "C", "D"}

    @pytest.mark.asyncio
    async def test_success_after_retry(self) -> None:
        agent = _make_agent()
        bad_resp = _make_litellm_response("not json")
        good_resp = _make_litellm_response(VALID_MCQ_JSON)

        mock = AsyncMock(side_effect=[bad_resp, good_resp])
        with patch("litellm.acompletion", mock):
            result = await agent._generate_mcq_with_retry("Some chunk", "chunk_0", 0)
        assert result is not None
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_fail(self) -> None:
        agent = _make_agent()
        bad_resp = _make_litellm_response("garbage")

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=bad_resp):
            result = await agent._generate_mcq_with_retry("Some chunk", "chunk_0", 0)
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_is_caught_and_retried(self) -> None:
        agent = _make_agent()
        good_resp = _make_litellm_response(VALID_MCQ_JSON)

        mock = AsyncMock(side_effect=[RuntimeError("timeout"), good_resp])
        with patch("litellm.acompletion", mock):
            result = await agent._generate_mcq_with_retry("Some chunk", "chunk_0", 0)
        assert result is not None
        assert mock.call_count == 2


class TestGenerateExam:
    @pytest.mark.asyncio
    async def test_generates_expected_number_of_questions(self) -> None:
        chunks, chunk_ids, embeddings = _make_clustered_embeddings(n_per_cluster=20, n_clusters=3)
        agent = _make_agent(exam_size=9, diversity_clusters=3)

        mock_resp = _make_litellm_response(VALID_MCQ_JSON)
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            questions = await agent.generate_exam(chunks, chunk_ids, embeddings)

        assert len(questions) == 9

    @pytest.mark.asyncio
    async def test_questions_have_populated_fields(self) -> None:
        chunks, chunk_ids, embeddings = _make_clustered_embeddings(n_per_cluster=10, n_clusters=2)
        agent = _make_agent(exam_size=4, diversity_clusters=2)

        mock_resp = _make_litellm_response(VALID_MCQ_JSON)
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            questions = await agent.generate_exam(chunks, chunk_ids, embeddings)

        for q in questions:
            assert q.id  # UUID populated
            assert q.question
            assert len(q.options) == 4
            assert q.source_chunk_id
            assert q.correct_answer in q.options

    @pytest.mark.asyncio
    async def test_cluster_diversity(self) -> None:
        """Questions should come from multiple clusters, not just one."""
        chunks, chunk_ids, embeddings = _make_clustered_embeddings(n_per_cluster=20, n_clusters=3)
        agent = _make_agent(exam_size=9, diversity_clusters=3)

        mock_resp = _make_litellm_response(VALID_MCQ_JSON)
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            questions = await agent.generate_exam(chunks, chunk_ids, embeddings)

        cluster_ids_seen = {q.cluster_id for q in questions}
        # All 3 clusters should contribute at least one question
        assert len(cluster_ids_seen) == 3

    @pytest.mark.asyncio
    async def test_skips_failed_chunks_and_tries_next(self) -> None:
        """If some chunks fail MCQ generation, the agent moves to the next chunk."""
        chunks, chunk_ids, embeddings = _make_clustered_embeddings(n_per_cluster=20, n_clusters=2)
        agent = _make_agent(exam_size=4, diversity_clusters=2)

        # Alternate between failure and success
        bad = _make_litellm_response("not json")
        good = _make_litellm_response(VALID_MCQ_JSON)
        call_count = 0

        async def _alternating(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Fail on odd calls, succeed on even
            if call_count % 2 == 1:
                return bad
            return good

        with patch("litellm.acompletion", side_effect=_alternating):
            questions = await agent.generate_exam(chunks, chunk_ids, embeddings)

        # Should still produce 4 questions (retries + fallback chunks)
        assert len(questions) == 4
