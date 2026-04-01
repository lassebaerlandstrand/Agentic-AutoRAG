"""Tests for the ExamAgent MCQ generation pipeline (document-level).

All LLM calls are mocked.
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
        "source_fact": (
            "RAG combines retrieval with generation to ground LLM answers in factual context. "
            "The retriever fetches relevant passages before the model composes an answer. "
            "This design reduces hallucinations by anchoring output to external evidence. "
            "In practice, the pipeline first retrieves and then conditions the response on those passages."
        ),
    }
)

VALID_MCQ_MARKDOWN_WRAPPED = f"```json\n{VALID_MCQ_JSON}\n```"


def _make_litellm_response(content: str) -> SimpleNamespace:
    """Build a mock litellm response object."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class DummyEmbeddingModel:
    """Simple deterministic embedding stub for similarity checks."""

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


def _make_agent(exam_size: int = 10) -> ExamAgent:
    config = ExaminerConfig(exam_size=exam_size, min_doc_words=0)
    agent = ExamAgent(
        config=config,
        examiner_model="gemini/gemini-3-flash-preview",
        embedding_model=DummyEmbeddingModel(),
    )
    agent._deduplicate_exam = lambda questions: questions
    agent._filter_discriminator_quality = lambda questions, *a, **kw: questions
    return agent


def _make_documents(
    n_per_cluster: int = 5, n_clusters: int = 3, dim: int = 4
) -> tuple[list[str], list[str], np.ndarray]:
    """Create well-separated synthetic documents and embeddings for testing."""
    rng = np.random.default_rng(0)
    documents: list[str] = []
    doc_ids: list[str] = []
    embeds: list[np.ndarray] = []
    for c in range(n_clusters):
        center = np.zeros(dim)
        center[c % dim] = 100.0  # push clusters apart
        for i in range(n_per_cluster):
            documents.append(f"Document text for cluster {c} item {i}. " * 10)
            doc_ids.append(f"doc_{c}_{i}")
            embeds.append(center + rng.standard_normal(dim) * 0.1)
    return documents, doc_ids, np.vstack(embeds)


class TestParseMcqResponse:
    def test_valid_json(self) -> None:
        agent = _make_agent()

        result = agent._parse_mcq_response(VALID_MCQ_JSON, "doc_0", 0)

        assert result is not None
        assert isinstance(result, MCQQuestion)
        assert result.correct_answer == "A"
        assert result.source_doc_ids == ["doc_0"]
        assert "RAG combines retrieval with generation" in result.source_fact
        assert result.cluster_id == 0

    def test_markdown_wrapped_json(self) -> None:
        agent = _make_agent()

        result = agent._parse_mcq_response(VALID_MCQ_MARKDOWN_WRAPPED, "doc_1", 2)

        assert result is not None
        assert result.correct_answer == "A"
        assert result.source_doc_ids == ["doc_1"]
        assert result.cluster_id == 2

    def test_invalid_json_returns_none(self) -> None:
        agent = _make_agent()

        result = agent._parse_mcq_response("this is not json", "doc_0", 0)

        assert result is None

    def test_missing_key_returns_none(self) -> None:
        incomplete = json.dumps({"question": "What?", "options": {"A": "a", "B": "b"}})
        agent = _make_agent()

        result = agent._parse_mcq_response(incomplete, "doc_0", 0)

        assert result is None

    def test_invalid_correct_answer_returns_none(self) -> None:
        bad_answer = json.dumps(
            {
                "question": "What?",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "correct_answer": "Z",
                "source_fact": "some fact",
            }
        )
        agent = _make_agent()

        result = agent._parse_mcq_response(bad_answer, "doc_0", 0)

        assert result is None

    def test_missing_source_fact_defaults_to_empty(self) -> None:
        no_fact = json.dumps(
            {
                "question": "What is RAG?",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "correct_answer": "A",
            }
        )
        agent = _make_agent()

        result = agent._parse_mcq_response(no_fact, "doc_0", 0)

        assert result is not None
        assert result.source_fact == ""


class TestGenerateMcqForDocument:
    @pytest.mark.asyncio
    async def test_success_first_attempt(self) -> None:
        agent = _make_agent()
        mock_resp = _make_litellm_response(VALID_MCQ_JSON)

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = await agent._generate_mcq_for_document("Some document text " * 20, "doc_0", 0, [])

        assert result is not None
        assert result.correct_answer in {"A", "B", "C", "D"}

    @pytest.mark.asyncio
    async def test_success_after_retry(self) -> None:
        agent = _make_agent()
        bad_resp = _make_litellm_response("not json")
        good_resp = _make_litellm_response(VALID_MCQ_JSON)
        mock = AsyncMock(side_effect=[bad_resp, good_resp])

        with patch("litellm.acompletion", mock):
            result = await agent._generate_mcq_for_document("Some document text " * 20, "doc_0", 0, [])

        assert result is not None
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_fail(self) -> None:
        agent = _make_agent()
        bad_resp = _make_litellm_response("garbage")

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=bad_resp):
            result = await agent._generate_mcq_for_document("Some document text " * 20, "doc_0", 0, [])

        assert result is None

    @pytest.mark.asyncio
    async def test_avoid_questions_passed_to_prompt(self) -> None:
        """Prompt should include existing questions to avoid duplicates."""
        agent = _make_agent()
        mock_resp = _make_litellm_response(VALID_MCQ_JSON)
        captured_prompts: list[str] = []

        async def _capture(*args, **kwargs):
            msgs = kwargs.get("messages", [])
            for m in msgs:
                captured_prompts.append(m.get("content", ""))
            return mock_resp

        existing = ["What is the capital of France?"]
        with patch("litellm.acompletion", side_effect=_capture):
            await agent._generate_mcq_for_document("doc text " * 30, "doc_0", 0, existing)

        user_prompt = "\n".join(captured_prompts)
        assert "What is the capital of France?" in user_prompt

    @pytest.mark.asyncio
    async def test_rejects_too_short_source_fact(self) -> None:
        agent = _make_agent()
        short_fact_payload = json.dumps(
            {
                "question": "What is the ID number assigned to Partner #1, John Doe, in this report?",
                "options": {"A": "P1-1058", "B": "1058B", "C": "1058", "D": "JD1058"},
                "correct_answer": "C",
                "source_fact": "Partner #1",
            }
        )
        mock_resp = _make_litellm_response(short_fact_payload)

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = await agent._generate_mcq_for_document("Some document text " * 20, "doc_0", 0, [])

        assert result is None

    @pytest.mark.asyncio
    async def test_rejects_answer_on_first_line_source_fact(self) -> None:
        agent = _make_agent()
        artifact_payload = json.dumps(
            {
                "question": "What ID is assigned to the partner?",
                "options": {"A": "P1-1058", "B": "1058B", "C": "1058", "D": "JD1058"},
                "correct_answer": "C",
                "source_fact": (
                    "1058\n"
                    "The onboarding record confirms the identifier in a broader review context. "
                    "Additional policy text is included for nearby context in the document. "
                    "A compliance sentence follows with unrelated procedural language. "
                    "The section closes with a generic note about document processing."
                ),
            }
        )
        mock_resp = _make_litellm_response(artifact_payload)

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = await agent._generate_mcq_for_document("Some document text " * 20, "doc_0", 0, [])

        assert result is None


class TestGenerateExam:
    @pytest.mark.asyncio
    async def test_generates_candidate_questions(self) -> None:
        documents, doc_ids, _ = _make_documents(n_per_cluster=5, n_clusters=3)
        agent = _make_agent(exam_size=9)
        mock_resp = _make_litellm_response(VALID_MCQ_JSON)

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            questions = await agent.generate_exam(documents, doc_ids)

        # Should generate up to exam_size * candidate_multiplier candidates
        assert len(questions) > 0

    @pytest.mark.asyncio
    async def test_questions_have_populated_fields(self) -> None:
        documents, doc_ids, _ = _make_documents(n_per_cluster=5, n_clusters=2)
        agent = _make_agent(exam_size=4)
        mock_resp = _make_litellm_response(VALID_MCQ_JSON)

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            questions = await agent.generate_exam(documents, doc_ids)

        for q in questions:
            assert q.id
            assert q.question
            assert len(q.options) == 4
            assert q.source_doc_ids
            assert q.correct_answer in q.options

    @pytest.mark.asyncio
    async def test_cluster_diversity(self) -> None:
        """Questions should come from multiple clusters, not just one."""
        documents, doc_ids, _ = _make_documents(n_per_cluster=5, n_clusters=3)
        agent = _make_agent(exam_size=9)
        mock_resp = _make_litellm_response(VALID_MCQ_JSON)

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            questions = await agent.generate_exam(documents, doc_ids)

        cluster_ids_seen = {q.cluster_id for q in questions}
        assert len(cluster_ids_seen) > 1

    @pytest.mark.asyncio
    async def test_skips_failed_docs_and_tries_next(self) -> None:
        """If some documents fail MCQ generation, the agent moves on."""
        documents, doc_ids, _ = _make_documents(n_per_cluster=5, n_clusters=2)
        agent = _make_agent(exam_size=4)

        bad = _make_litellm_response("not json")
        good = _make_litellm_response(VALID_MCQ_JSON)
        call_count = 0

        async def _alternating(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 1:
                return bad
            return good

        with patch("litellm.acompletion", side_effect=_alternating):
            questions = await agent.generate_exam(documents, doc_ids)

        assert len(questions) > 0

    @pytest.mark.asyncio
    async def test_allows_multiple_questions_per_doc_when_corpus_small(self) -> None:
        """When corpus has fewer docs than target candidates, docs are reused."""
        # 2 docs, exam_size=6, candidate_multiplier=1.5 → target=9 candidates
        documents = ["Document about RAG systems. " * 30, "Document about embeddings. " * 30]
        doc_ids = ["doc_0", "doc_1"]
        agent = _make_agent(exam_size=6)
        mock_resp = _make_litellm_response(VALID_MCQ_JSON)

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            questions = await agent.generate_exam(documents, doc_ids)

        # Should have attempted more than 2 questions total
        assert len(questions) > 2
