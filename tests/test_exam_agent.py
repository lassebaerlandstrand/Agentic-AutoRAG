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
from agentic_autorag.examiner.exam_agent import BLOOM_LEVELS, ExamAgent

_VALID_MCQ_DICT = {
    "question": "What is the primary purpose of RAG?",
    "options": {
        "A": "Retrieval-Augmented Generation for grounding LLM answers",
        "B": "Random Access Generation for faster inference",
        "C": "Recursive Algorithm for Graph traversal",
        "D": "Real-time Aggregation of Gradients",
    },
    "correct_answer": "A",
    "source_fact": [
        "RAG combines retrieval with generation to ground LLM answers in factual context. "
        "The retriever fetches relevant passages before the model composes an answer. "
        "This design reduces hallucinations by anchoring output to external evidence. "
        "In practice, the pipeline first retrieves and then conditions the response on those passages."
    ],
}

_VALID_MCQ_DICT_2 = {
    "question": "What embedding dimension does the model use?",
    "options": {
        "A": "384",
        "B": "768",
        "C": "1024",
        "D": "512",
    },
    "correct_answer": "B",
    "source_fact": [
        "The all-MiniLM-L6-v2 model produces embeddings of dimension 768. "
        "This is a compact representation that balances quality and speed. "
        "The model was trained on a large corpus of sentence pairs for similarity."
    ],
}

VALID_MCQ_JSON = json.dumps(_VALID_MCQ_DICT)
VALID_MCQ_BATCH_JSON = json.dumps([_VALID_MCQ_DICT])
VALID_MCQ_BATCH_2_JSON = json.dumps([_VALID_MCQ_DICT, _VALID_MCQ_DICT_2])

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
    """Create well-separated synthetic documents and embeddings for testing.

    Each doc includes the VALID_MCQ_DICT source_fact text verbatim so the
    examiner's _is_source_fact_valid check passes when these docs are paired
    with VALID_MCQ_BATCH responses in tests.
    """
    rng = np.random.default_rng(0)
    documents: list[str] = []
    doc_ids: list[str] = []
    embeds: list[np.ndarray] = []
    fact_block = _VALID_MCQ_DICT["source_fact"][0] + "\n\n" + _VALID_MCQ_DICT_2["source_fact"][0]
    for c in range(n_clusters):
        center = np.zeros(dim)
        center[c % dim] = 100.0  # push clusters apart
        for i in range(n_per_cluster):
            documents.append(f"Document text for cluster {c} item {i}. " * 10 + "\n\n" + fact_block)
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
        assert result.source_fact and "RAG combines retrieval with generation" in result.source_fact[0]
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
        assert result.source_fact == []


class TestParseBatchResponse:
    def test_valid_json_array(self) -> None:
        agent = _make_agent()

        results = agent._parse_batch_response(VALID_MCQ_BATCH_2_JSON, "doc_0", 0, ["Remember", "Understand"])

        assert len(results) == 2
        assert results[0] is not None
        assert results[0].bloom_level == "Remember"
        assert results[1] is not None
        assert results[1].bloom_level == "Understand"

    def test_markdown_wrapped_array(self) -> None:
        agent = _make_agent()
        wrapped = f"```json\n{VALID_MCQ_BATCH_2_JSON}\n```"

        results = agent._parse_batch_response(wrapped, "doc_0", 0, ["Apply", "Analyze"])

        assert len(results) == 2
        assert all(r is not None for r in results)

    def test_partial_array_with_invalid_element(self) -> None:
        agent = _make_agent()
        items = [
            _VALID_MCQ_DICT,
            {"question": "Missing fields"},  # invalid — no options/correct_answer
            _VALID_MCQ_DICT_2,
        ]
        raw = json.dumps(items)

        results = agent._parse_batch_response(raw, "doc_0", 0, ["R", "U", "Ap"])

        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is None  # invalid element
        assert results[2] is not None

    def test_single_object_fallback(self) -> None:
        """When LLM returns a single object instead of an array, wrap it."""
        agent = _make_agent()

        results = agent._parse_batch_response(VALID_MCQ_JSON, "doc_0", 0, ["Remember"])

        assert len(results) == 1
        assert results[0] is not None

    def test_garbage_returns_empty(self) -> None:
        agent = _make_agent()

        results = agent._parse_batch_response("this is not json at all", "doc_0", 0, ["R"])

        assert results == []

    def test_trailing_commas_handled(self) -> None:
        agent = _make_agent()
        raw = f"[{VALID_MCQ_JSON},]"

        results = agent._parse_batch_response(raw, "doc_0", 0, ["Remember"])

        assert len(results) == 1
        assert results[0] is not None

    def test_bloom_level_assigned_by_position(self) -> None:
        agent = _make_agent()

        results = agent._parse_batch_response(VALID_MCQ_BATCH_2_JSON, "doc_0", 0, ["Evaluate", "Apply"])

        assert results[0] is not None
        assert results[0].bloom_level == "Evaluate"
        assert results[1] is not None
        assert results[1].bloom_level == "Apply"


class TestExtractJsonArray:
    def test_basic_extraction(self) -> None:
        agent = _make_agent()
        text = f"Here are the questions: {VALID_MCQ_BATCH_2_JSON} done."

        result = agent._extract_json_array(text)

        assert result is not None
        assert len(result) == 2

    def test_no_array_returns_none(self) -> None:
        agent = _make_agent()

        result = agent._extract_json_array("no brackets here")

        assert result is None

    def test_nested_objects_in_array(self) -> None:
        agent = _make_agent()

        result = agent._extract_json_array(VALID_MCQ_BATCH_JSON)

        assert result is not None
        assert len(result) == 1
        assert "question" in result[0]


class TestGenerateBatchForDocument:
    @pytest.mark.asyncio
    async def test_batch_success(self) -> None:
        agent = _make_agent()
        mock_resp = _make_litellm_response(VALID_MCQ_BATCH_2_JSON)
        bloom_levels = [BLOOM_LEVELS[0], BLOOM_LEVELS[1]]
        # Doc must contain the verbatim source_fact spans for _is_source_fact_valid to pass.
        doc_text = _VALID_MCQ_DICT["source_fact"][0] + "\n\n" + _VALID_MCQ_DICT_2["source_fact"][0]

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = await agent._generate_batch_for_document(
                doc_text,
                "doc_0",
                0,
                bloom_levels,
                [],
                [],
                global_failures={},
            )

        assert len(result) == 2
        assert all(q.correct_answer in {"A", "B", "C", "D"} for q in result)

    @pytest.mark.asyncio
    async def test_batch_partial_success(self) -> None:
        """One question in the batch has a bad source_fact, others pass."""
        agent = _make_agent()
        items = [
            _VALID_MCQ_DICT,
            {**_VALID_MCQ_DICT_2, "source_fact": ["too short"]},  # fails source_fact check
            _VALID_MCQ_DICT,
        ]
        mock_resp = _make_litellm_response(json.dumps(items))
        bloom_levels = [BLOOM_LEVELS[0], BLOOM_LEVELS[1], BLOOM_LEVELS[2]]
        # Doc must contain the verbatim source_fact of _VALID_MCQ_DICT (but not "too short").
        doc_text = _VALID_MCQ_DICT["source_fact"][0]

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            failures: dict[str, int] = {}
            result = await agent._generate_batch_for_document(
                doc_text,
                "doc_0",
                0,
                bloom_levels,
                [],
                [],
                global_failures=failures,
            )

        assert len(result) == 2
        assert failures.get("source_fact", 0) == 1

    @pytest.mark.asyncio
    async def test_batch_total_parse_failure(self) -> None:
        agent = _make_agent()
        mock_resp = _make_litellm_response("garbage response")
        bloom_levels = [BLOOM_LEVELS[0]]

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            failures: dict[str, int] = {}
            result = await agent._generate_batch_for_document(
                "Some document text " * 20,
                "doc_0",
                0,
                bloom_levels,
                [],
                [],
                global_failures=failures,
            )

        assert result == []
        assert failures.get("parse", 0) >= 1

    @pytest.mark.asyncio
    async def test_single_object_fallback(self) -> None:
        """LLM returns a single JSON object instead of array — still works."""
        agent = _make_agent()
        mock_resp = _make_litellm_response(VALID_MCQ_JSON)
        bloom_levels = [BLOOM_LEVELS[0]]
        doc_text = _VALID_MCQ_DICT["source_fact"][0]

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = await agent._generate_batch_for_document(
                doc_text,
                "doc_0",
                0,
                bloom_levels,
                [],
                [],
                global_failures={},
            )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_exclude_section_in_prompt(self) -> None:
        """Prompt should include exclude questions for backfill rounds."""
        agent = _make_agent()
        mock_resp = _make_litellm_response(VALID_MCQ_BATCH_JSON)
        captured_prompts: list[str] = []

        async def _capture(*args, **kwargs):
            msgs = kwargs.get("messages", [])
            for m in msgs:
                captured_prompts.append(m.get("content", ""))
            return mock_resp

        exclude_q = ["What is the capital of France?"]
        exclude_f = ["Paris is the capital of France and the largest city in the country by population."]
        with patch("litellm.acompletion", side_effect=_capture):
            await agent._generate_batch_for_document(
                "doc text " * 30,
                "doc_0",
                0,
                [BLOOM_LEVELS[0]],
                exclude_q,
                exclude_f,
                global_failures={},
            )

        user_prompt = "\n".join(captured_prompts)
        assert "What is the capital of France?" in user_prompt

    @pytest.mark.asyncio
    async def test_rejects_too_short_source_fact(self) -> None:
        agent = _make_agent()
        short_fact_item = {
            "question": "What is the ID number assigned to Partner #1?",
            "options": {"A": "P1-1058", "B": "1058B", "C": "1058", "D": "JD1058"},
            "correct_answer": "C",
            "source_fact": "Partner #1",
        }
        mock_resp = _make_litellm_response(json.dumps([short_fact_item]))

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = await agent._generate_batch_for_document(
                "Some document text " * 20,
                "doc_0",
                0,
                [BLOOM_LEVELS[0]],
                [],
                [],
                global_failures={},
            )

        assert result == []


class TestGenerateExam:
    @pytest.mark.asyncio
    async def test_generates_candidate_questions(self) -> None:
        documents, doc_ids, _ = _make_documents(n_per_cluster=5, n_clusters=3)
        agent = _make_agent(exam_size=9)
        mock_resp = _make_litellm_response(VALID_MCQ_BATCH_JSON)

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            questions = await agent.generate_exam(documents, doc_ids)

        # Should generate candidates (at least some from the batch responses)
        assert len(questions) > 0

    @pytest.mark.asyncio
    async def test_questions_have_populated_fields(self) -> None:
        documents, doc_ids, _ = _make_documents(n_per_cluster=5, n_clusters=2)
        agent = _make_agent(exam_size=4)
        mock_resp = _make_litellm_response(VALID_MCQ_BATCH_JSON)

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
        mock_resp = _make_litellm_response(VALID_MCQ_BATCH_JSON)

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
        good = _make_litellm_response(VALID_MCQ_BATCH_JSON)
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
        """When corpus has fewer docs than target candidates, docs get batch K > 1."""
        # 2 docs, each ~3000 words → 3000 // 500 = 6, clamped to max_questions_per_doc=3.
        # Embed the source_facts in each doc so verbatim validation passes.
        fact_block = _VALID_MCQ_DICT["source_fact"][0] + "\n\n" + _VALID_MCQ_DICT_2["source_fact"][0]
        documents = [
            "Document about RAG systems and retrieval. " * 500 + "\n\n" + fact_block,
            "Document about embeddings and models. " * 500 + "\n\n" + fact_block,
        ]
        doc_ids = ["doc_0", "doc_1"]
        agent = _make_agent(exam_size=6)
        # Return 2 questions per batch call so multi-slot docs produce results
        mock_resp = _make_litellm_response(VALID_MCQ_BATCH_2_JSON)

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            questions = await agent.generate_exam(documents, doc_ids)

        assert len(questions) > 2

    @pytest.mark.asyncio
    async def test_warns_when_corpus_capacity_below_wave_size(self, caplog: pytest.LogCaptureFixture) -> None:
        """When sum(_cluster_capacities) < wave_size, a user-visible warning fires
        and the attempt total is bounded by real capacity rather than silently dropped."""
        # Three short docs, each ~60 words → real capacity 1 each → total 3.
        # exam_size=6 × initial_candidate_multiplier=2.5 (default) = wave_size 15.
        fact_block = _VALID_MCQ_DICT["source_fact"][0]
        short_body = "Short document body. " * 20  # ~60 words
        documents = [short_body + "\n\n" + fact_block for _ in range(3)]
        doc_ids = [f"doc_{i}" for i in range(3)]
        agent = _make_agent(exam_size=6)
        mock_resp = _make_litellm_response(VALID_MCQ_BATCH_JSON)

        with (
            caplog.at_level("WARNING", logger="agentic_autorag.run"),
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp),
        ):
            questions = await agent.generate_exam(documents, doc_ids)

        warning_records = [
            r
            for r in caplog.records
            if r.name == "agentic_autorag.run" and "Corpus capacity supports only" in r.getMessage()
        ]
        assert warning_records, "Expected a capacity-shortfall warning on run_logger"
        # Capacity (3) bounds the attempt total; questions returned cannot exceed it.
        assert len(questions) <= 3
