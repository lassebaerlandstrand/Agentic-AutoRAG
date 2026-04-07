"""Tests for the exam quality validation pipeline (Layers 2-4)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from agentic_autorag.config.models import MCQQuestion
from agentic_autorag.examiner.exam_validator import (
    check_oracle,
    check_parametric_leaks,
    filter_easy_retrieval,
    run_validation_pipeline,
    verify_source_facts,
)


def _make_question(
    qid: str = "q1",
    correct: str = "A",
    source_fact: str = (
        "RAG combines retrieval with generation to improve factual grounding in responses. "
        "The retriever supplies relevant external passages before generation."
    ),
    doc_id: str = "doc_0",
) -> MCQQuestion:
    return MCQQuestion(
        id=qid,
        question="What is the primary mechanism of RAG systems?",
        options={
            "A": "Combining retrieval with neural generation",
            "B": "Using only parametric knowledge",
            "C": "Random sampling from a corpus",
            "D": "Training a classifier on labeled data",
        },
        correct_answer=correct,
        source_doc_ids=[doc_id],
        source_fact=source_fact,
        cluster_id=0,
    )


def _make_litellm_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class DummyEmbedder:
    """Deterministic embedder for testing. Returns controlled similarity values."""

    def __init__(self, similarity_value: float = 0.9):
        self.similarity_value = similarity_value
        self._responses: dict[str, np.ndarray] = {}

    def add_response(self, text: str, vector: np.ndarray) -> None:
        self._responses[text] = vector

    def encode(self, texts: list[str]) -> np.ndarray:
        results = []
        for text in texts:
            if text in self._responses:
                results.append(self._responses[text])
            else:
                # Default: orthogonal basis vector based on hash
                idx = hash(text) % 128
                v = np.zeros(128, dtype=np.float32)
                v[idx] = 1.0
                results.append(v)
        return np.array(results, dtype=np.float32)


class TestVerifySourceFacts:
    def test_passes_high_similarity_question(self) -> None:
        """Questions whose source_fact closely matches the document pass."""
        source_fact = "RAG combines retrieval with generation."
        doc_text = "RAG combines retrieval with generation. It is widely used in NLP."

        # Use a real-ish embedder that will return high similarity for matching text
        embedder = DummyEmbedder()
        # Make source_fact and a window have the same vector (perfect similarity)
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        embedder.add_response(source_fact, v)
        embedder.add_response(doc_text, v)  # the whole doc is one "window" when short

        q = _make_question(source_fact=source_fact)
        documents = {"doc_0": doc_text}

        result = verify_source_facts([q], documents, embedder, threshold=0.75, min_source_fact_length=1)

        assert len(result) == 1

    def test_removes_low_similarity_question(self) -> None:
        """Questions with hallucinated source_facts are removed."""
        source_fact = "The moon is made of cheese."
        doc_text = "RAG combines retrieval with generation. It is widely used in NLP."

        embedder = DummyEmbedder()
        # source_fact gets a different vector than document windows
        embedder.add_response(source_fact, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        embedder.add_response(doc_text, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

        q = _make_question(source_fact=source_fact)
        documents = {"doc_0": doc_text}

        result = verify_source_facts([q], documents, embedder, threshold=0.75, min_source_fact_length=1)

        assert len(result) == 0

    def test_removes_question_with_empty_source_fact(self) -> None:
        """Questions with no source_fact are removed by minimum-length guard."""
        q = _make_question(source_fact="")
        documents = {"doc_0": "Some document text."}

        result = verify_source_facts([q], documents, DummyEmbedder(), threshold=0.75)

        assert len(result) == 0

    def test_skips_question_with_missing_document(self) -> None:
        """Questions whose doc_id is not in documents dict skip Layer 2."""
        q = _make_question(source_fact="some fact", doc_id="missing_doc")
        documents = {}  # doc not present

        result = verify_source_facts([q], documents, DummyEmbedder(), threshold=0.75, min_source_fact_length=1)

        assert len(result) == 1

    def test_empty_input(self) -> None:
        result = verify_source_facts([], {}, DummyEmbedder(), threshold=0.75)
        assert result == []

    def test_threshold_boundary(self) -> None:
        """Question at exactly the threshold should pass."""
        source_fact = "Exactly at threshold with enough text for validation logic."
        # Use perfect match (sim=1.0) to verify >= threshold passes
        embedder = DummyEmbedder()
        perfect = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        embedder.add_response(source_fact, perfect)
        doc_text = "some doc text"
        embedder.add_response(doc_text, perfect)

        q = _make_question(source_fact=source_fact)
        documents = {"doc_0": doc_text}

        result = verify_source_facts([q], documents, embedder, threshold=0.75, min_source_fact_length=1)
        assert len(result) == 1

    def test_removes_short_source_fact(self) -> None:
        q = _make_question(source_fact="Partner #1")
        documents = {"doc_0": "Partner #1 has ID 1058 and signed the agreement."}

        result = verify_source_facts([q], documents, DummyEmbedder(), threshold=0.75)

        assert len(result) == 0

    def test_normalized_substring_fallback_handles_newlines(self) -> None:
        source_fact = "Effective: 1 July 2021"
        doc_text = "The policy update became Effective:\n1 July 2021 and applies immediately."
        q = _make_question(source_fact=source_fact)
        documents = {"doc_0": doc_text}

        result = verify_source_facts(
            [q],
            documents,
            DummyEmbedder(),
            threshold=0.99,
            substring_fallback=True,
            min_source_fact_length=10,
        )

        assert len(result) == 1


class TestCheckParametricLeaks:
    @pytest.mark.asyncio
    async def test_removes_questions_llm_answers_correctly(self) -> None:
        """Questions are removed only when all trials are correct."""
        q = _make_question(correct="A")
        mock_resp = _make_litellm_response("A")  # LLM picks correct answer every trial

        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await check_parametric_leaks([q], model="test-model", n_trials=3)

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_keeps_questions_llm_answers_wrongly(self) -> None:
        """Questions with non-unanimous correctness are kept."""
        q = _make_question(correct="A")
        mock_resp = _make_litellm_response("B")  # LLM picks wrong answer every trial

        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await check_parametric_leaks([q], model="test-model", n_trials=3)

        assert len(result) == 1
        assert result[0].id == "q1"

    @pytest.mark.asyncio
    async def test_single_trial_mode(self) -> None:
        """With n_trials=1, unanimous means one correct response removes."""
        q = _make_question(correct="A")
        mock_resp = _make_litellm_response("A")

        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await check_parametric_leaks([q], model="test-model", n_trials=1)

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_majority_vote_keeps_minority_correct(self) -> None:
        """If LLM gets only 1/3 trials correct, question is kept (not a leak)."""
        q = _make_question(correct="A")
        # 3 trials: correct once, wrong twice → 1/3 < majority(2) → keep
        responses = [
            _make_litellm_response("A"),  # trial 1: correct
            _make_litellm_response("B"),  # trial 2: wrong
            _make_litellm_response("C"),  # trial 3: wrong
        ]

        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=responses,
        ):
            result = await check_parametric_leaks(
                [q],
                model="test-model",
                concurrency=1,
                n_trials=3,
            )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_two_of_three_correct_is_removed_under_majority_rule(self) -> None:
        """With majority voting (threshold=2 for n_trials=3), 2/3 correct is a leak."""
        q = _make_question(correct="A")
        responses = [
            _make_litellm_response("A"),
            _make_litellm_response("A"),
            _make_litellm_response("B"),
        ]

        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=responses,
        ):
            result = await check_parametric_leaks([q], model="test-model", concurrency=1, n_trials=3)

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_dont_know_option_keeps_question(self) -> None:
        q = _make_question(correct="A")
        responses = [
            _make_litellm_response("E"),
            _make_litellm_response("E"),
            _make_litellm_response("E"),
        ]

        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=responses,
        ):
            result = await check_parametric_leaks([q], model="test-model", concurrency=1, n_trials=3)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_mixed_batch(self) -> None:
        """Correctly filters a mixed batch with multi-trial."""
        q_leak = _make_question(qid="q_leak", correct="A")
        q_ok = _make_question(qid="q_ok", correct="A")
        # With concurrency=1 and n_trials=3: 3 calls for q_leak, 3 calls for q_ok
        responses = [
            _make_litellm_response("A"),  # q_leak trial 1: correct
            _make_litellm_response("A"),  # q_leak trial 2: correct
            _make_litellm_response("A"),  # q_leak trial 3: correct → leak (3/3)
            _make_litellm_response("C"),  # q_ok trial 1: wrong
            _make_litellm_response("B"),  # q_ok trial 2: wrong
            _make_litellm_response("D"),  # q_ok trial 3: wrong → keep (0/3)
        ]

        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=responses,
        ):
            result = await check_parametric_leaks(
                [q_leak, q_ok],
                model="test-model",
                concurrency=1,
                n_trials=3,
            )

        kept_ids = {q.id for q in result}
        assert "q_ok" in kept_ids
        assert "q_leak" not in kept_ids

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        result = await check_parametric_leaks([], model="test-model")
        assert result == []

    @pytest.mark.asyncio
    async def test_transient_error_then_recovery(self) -> None:
        """A transient error on first attempt is retried; question processed on retry."""
        q = _make_question(correct="A")

        class RateLimitError(Exception):
            """Simulated transient rate limit error."""

        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:  # first 3 trials all raise transient
                raise RateLimitError("rate limit hit")
            return _make_litellm_response("B")  # retry trials → wrong → keep

        with (
            patch(
                "agentic_autorag.examiner.exam_validator.litellm.acompletion",
                side_effect=_side_effect,
            ),
            patch("agentic_autorag.examiner._errors.is_transient_llm_error", return_value=True),
            patch("agentic_autorag.examiner.exam_validator.is_transient_llm_error", return_value=True),
            patch("asyncio.sleep"),
        ):
            result = await check_parametric_leaks([q], model="test-model", n_trials=3)

        # After retry, LLM returned wrong answers → question is kept (not a leak)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_permanent_transient_error_removes_question_conservatively(self) -> None:
        """If all retries fail with transient errors, question is removed (potential leak)."""
        q = _make_question(correct="A")

        with (
            patch(
                "agentic_autorag.examiner.exam_validator.litellm.acompletion",
                side_effect=Exception("transient"),
            ),
            patch("agentic_autorag.examiner.exam_validator.is_transient_llm_error", return_value=True),
            patch("asyncio.sleep"),
        ):
            result = await check_parametric_leaks([q], model="test-model", n_trials=1)

        # Permanently failed → removed conservatively (treated as potential leak)
        assert len(result) == 0


class TestCheckOracle:
    @pytest.mark.asyncio
    async def test_keeps_questions_llm_answers_correctly_with_context(self) -> None:
        """Questions the LLM answers correctly with source_fact are kept."""
        q = _make_question(correct="A")
        mock_resp = _make_litellm_response("A")

        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await check_oracle([q], model="test-model")

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_removes_questions_llm_answers_wrongly_with_context(self) -> None:
        """Questions the LLM cannot answer even with source_fact are removed."""
        q = _make_question(correct="A")
        mock_resp = _make_litellm_response("B")  # LLM picks wrong despite context

        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await check_oracle([q], model="test-model")

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_removes_when_llm_selects_e_insufficient_context(self) -> None:
        """When the LLM selects E (insufficient context), question is removed."""
        q = _make_question(correct="A")
        mock_resp = _make_litellm_response("E")

        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await check_oracle([q], model="test-model")

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        result = await check_oracle([], model="test-model")
        assert result == []

    @pytest.mark.asyncio
    async def test_transient_error_then_recovery(self) -> None:
        """A transient error on first attempt is retried; question evaluated on retry."""
        q = _make_question(correct="A")
        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("transient")
            return _make_litellm_response("A")  # correct on retry → keep

        with (
            patch(
                "agentic_autorag.examiner.exam_validator.litellm.acompletion",
                side_effect=_side_effect,
            ),
            patch("agentic_autorag.examiner.exam_validator.is_transient_llm_error", return_value=True),
            patch("asyncio.sleep"),
        ):
            result = await check_oracle([q], model="test-model")

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_permanent_transient_error_removes_question_conservatively(self) -> None:
        """If all retries fail with transient errors, question is removed (conservative)."""
        q = _make_question(correct="A")

        with (
            patch(
                "agentic_autorag.examiner.exam_validator.litellm.acompletion",
                side_effect=Exception("transient"),
            ),
            patch("agentic_autorag.examiner.exam_validator.is_transient_llm_error", return_value=True),
            patch("asyncio.sleep"),
        ):
            result = await check_oracle([q], model="test-model")

        # Permanently failed → removed conservatively (INVALID treatment)
        assert len(result) == 0


class TestRunValidationPipeline:
    @pytest.mark.asyncio
    async def test_all_layers_applied_sequentially(self) -> None:
        """Full pipeline removes questions that fail any layer."""
        # q1: passes all layers (LLM picks wrong without context, right with context)
        q1 = _make_question(
            qid="q1",
            correct="A",
            source_fact=(
                "RAG fact here with sufficient context for similarity checks. "
                "More text about retrieval and generation in this document."
            ),
        )
        # q2: fails layer 3 (LLM picks correct without context = parametric leak)
        q2 = _make_question(
            qid="q2",
            correct="A",
            source_fact=(
                "Another fact with sufficient context for source verification. "
                "It includes surrounding details for robust matching."
            ),
        )
        # q3: fails layer 4 (LLM picks wrong even with source_fact)
        q3 = _make_question(
            qid="q3",
            correct="A",
            source_fact=(
                "Third fact with enough contextual details to pass minimum length checks. "
                "This sentence provides additional nearby context."
            ),
        )

        documents = {
            "doc_0": "RAG fact here. More text about RAG systems." * 10,
        }

        # Perfect embedder (all source facts match perfectly)
        embedder = DummyEmbedder()
        perfect_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for q in [q1, q2, q3]:
            embedder.add_response(q.source_fact, perfect_vec)
        embedder.add_response(documents["doc_0"], perfect_vec)

        # LLM responses (3 trials per question for parametric check):
        # Layer 3 (parametric check, no context, n_trials=3, concurrency=1):
        #   q1: 3 trials all wrong → keep
        #   q2: 3 trials all correct → remove (leak, majority 3/3)
        #   q3: 3 trials all wrong → keep
        # Layer 4 (oracle check, with source_fact):
        #   q1: picks "A" (correct) → keep
        #   q3: picks "E" (insufficient) → remove
        layer3_responses = [
            # q1: 3 trials, all wrong
            _make_litellm_response("B"),
            _make_litellm_response("C"),
            _make_litellm_response("D"),
            # q2: 3 trials, all correct → leak
            _make_litellm_response("A"),
            _make_litellm_response("A"),
            _make_litellm_response("A"),
            # q3: 3 trials, all wrong
            _make_litellm_response("C"),
            _make_litellm_response("B"),
            _make_litellm_response("D"),
        ]
        layer4_responses = [
            _make_litellm_response("A"),  # q1: correct → keep
            _make_litellm_response("E"),  # q3: E → remove
        ]

        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=layer3_responses + layer4_responses,
        ):
            result = await run_validation_pipeline(
                [q1, q2, q3],
                documents=documents,
                embedder=embedder,
                model="test-model",
                concurrency=1,
                source_fact_threshold=0.75,
                detect_parametric_leaks=True,
                source_fact_min_length=1,
                parametric_leak_trials=3,
            )

        assert len(result) == 1
        assert result[0].id == "q1"

    @pytest.mark.asyncio
    async def test_skips_layer3_when_disabled(self) -> None:
        """When detect_parametric_leaks=False, Layer 3 is skipped."""
        q = _make_question(
            qid="q1",
            correct="A",
            source_fact=(
                "RAG fact with surrounding context so the source passage is self-contained enough. "
                "Additional sentence for robustness."
            ),
        )
        documents = {"doc_0": "RAG fact. More text." * 20}

        embedder = DummyEmbedder()
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        embedder.add_response(q.source_fact, v)
        embedder.add_response(documents["doc_0"], v)

        # Only Layer 4 call (no Layer 3)
        oracle_response = _make_litellm_response("A")

        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=oracle_response,
        ) as mock_llm:
            result = await run_validation_pipeline(
                [q],
                documents=documents,
                embedder=embedder,
                model="test-model",
                concurrency=1,
                source_fact_threshold=0.75,
                detect_parametric_leaks=False,
                source_fact_min_length=1,
            )

        # Only 1 LLM call (oracle only, no parametric check)
        assert mock_llm.call_count == 1
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Retrieval difficulty filter
# ---------------------------------------------------------------------------


class TestFilterEasyRetrieval:
    def test_removes_trivially_retrievable(self) -> None:
        """Question whose source_fact overlaps with the top-1 chunk is removed."""
        q = _make_question(
            qid="easy",
            source_fact="RAG combines retrieval with generation to improve factual grounding in responses.",
        )
        # Chunk that contains the source_fact content
        chunks = ["RAG combines retrieval with generation to improve factual grounding in responses."]
        chunk_embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        mock_embedder = SimpleNamespace(encode=lambda texts: np.array([[1.0, 0.0, 0.0]] * len(texts), dtype=np.float32))
        result = filter_easy_retrieval(
            [q],
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            embedder=mock_embedder,
        )
        assert len(result) == 0

    def test_keeps_hard_question(self) -> None:
        """Question whose source_fact does NOT overlap with top-1 chunk is kept."""
        q = _make_question(
            qid="hard",
            source_fact="The specific mutation rate in exon 7 was 0.003 per nucleotide per generation.",
        )
        # Top-1 chunk is about a completely different topic
        chunks = ["Solar panel efficiency has improved by 25% over the last decade."]
        chunk_embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        mock_embedder = SimpleNamespace(encode=lambda texts: np.array([[1.0, 0.0, 0.0]] * len(texts), dtype=np.float32))
        result = filter_easy_retrieval(
            [q],
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            embedder=mock_embedder,
        )
        assert len(result) == 1
        assert result[0].id == "hard"

    def test_empty_input(self) -> None:
        result = filter_easy_retrieval(
            [],
            chunks=["some chunk"],
            chunk_embeddings=np.array([[1.0]], dtype=np.float32),
            embedder=SimpleNamespace(encode=lambda texts: np.array([[1.0]] * len(texts))),
        )
        assert result == []

    def test_question_without_source_fact_kept(self) -> None:
        """Questions missing source_fact are kept (can't evaluate difficulty)."""
        q = _make_question(qid="no_fact", source_fact="")
        chunks = ["anything"]
        chunk_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        mock_embedder = SimpleNamespace(encode=lambda texts: np.array([[1.0, 0.0]] * len(texts), dtype=np.float32))

        result = filter_easy_retrieval([q], chunks=chunks, chunk_embeddings=chunk_embeddings, embedder=mock_embedder)
        assert len(result) == 1

    def test_max_easy_rank_higher(self) -> None:
        """With max_easy_rank=2, questions found in top-2 chunks are removed."""
        q = _make_question(
            qid="medium",
            source_fact="RAG combines retrieval with generation to improve factual grounding in responses.",
        )
        # Two chunks: first is irrelevant, second contains the fact
        chunks = [
            "Solar panel efficiency has improved by 25% over the last decade.",
            "RAG combines retrieval with generation to improve factual grounding in responses.",
        ]
        chunk_embeddings = np.array([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32)

        # Embedder returns vector closest to first chunk (irrelevant), but top-2 includes second
        mock_embedder = SimpleNamespace(encode=lambda texts: np.array([[0.9, 0.1]] * len(texts), dtype=np.float32))

        # With max_easy_rank=1, kept (top-1 is the irrelevant chunk)
        result_1 = filter_easy_retrieval(
            [q], chunks=chunks, chunk_embeddings=chunk_embeddings, embedder=mock_embedder, max_easy_rank=1
        )
        assert len(result_1) == 1

        # With max_easy_rank=2, removed (top-2 includes the matching chunk)
        result_2 = filter_easy_retrieval(
            [q], chunks=chunks, chunk_embeddings=chunk_embeddings, embedder=mock_embedder, max_easy_rank=2
        )
        assert len(result_2) == 0
