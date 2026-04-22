"""Tests for the exam quality validation pipeline (Layers 2-4)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from agentic_autorag.config.models import MCQQuestion
from agentic_autorag.engine.pipeline import RetrievedDocument
from agentic_autorag.examiner.exam_validator import (
    _intervals_overlap,
    _locate_span_in_doc,
    check_oracle,
    check_parametric_leaks,
    chunk_contains_source_fact,
    filter_easy_retrieval,
    ngram_relevance,
    normalized_contains,
    run_validation_pipeline,
    verify_source_facts,
)


def _make_question(
    qid: str = "q1",
    correct: str = "A",
    source_fact: list[str] | None = None,
    source_fact_offsets: list[tuple[int, int]] | None = None,
    doc_id: str = "doc_0",
) -> MCQQuestion:
    if source_fact is None:
        source_fact = [
            "RAG combines retrieval with generation to improve factual grounding in responses."
            " The retriever supplies relevant external passages before generation."
        ]
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
        source_fact_offsets=source_fact_offsets or [],
        cluster_id=0,
    )


def _make_litellm_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class TestIntervalsOverlap:
    def test_exact_kiss_not_overlapping(self) -> None:
        assert _intervals_overlap((0, 100), (100, 200), min_chars=1) is False

    def test_single_char_overlap_below_floor(self) -> None:
        assert _intervals_overlap((0, 101), (100, 200), min_chars=10) is False

    def test_overlap_at_floor_passes(self) -> None:
        assert _intervals_overlap((0, 60), (10, 100), min_chars=50) is True

    def test_full_containment_both_directions(self) -> None:
        assert _intervals_overlap((0, 1000), (100, 200), min_chars=50) is True
        assert _intervals_overlap((100, 200), (0, 1000), min_chars=50) is True

    def test_disjoint(self) -> None:
        assert _intervals_overlap((0, 50), (100, 200), min_chars=1) is False


class TestNgramRelevance:
    def test_chunk_contains_span_verbatim(self) -> None:
        span = "the quick brown fox jumps over the lazy dog runs through the deep forest"
        chunk = f"Intro. {span}. Outro."
        assert ngram_relevance([span], chunk) is True

    def test_disjoint_content_not_relevant(self) -> None:
        span = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        chunk = "completely unrelated text about weather patterns and meteorology"
        assert ngram_relevance([span], chunk) is False

    def test_consecutive_run_triggers_relevance(self) -> None:
        """A 9+ word consecutive match (5 consecutive 5-grams) is enough."""
        span = "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth"
        # Chunk has a 10-word run verbatim, plus unrelated padding.
        chunk = "padding padding first second third fourth fifth sixth seventh eighth ninth tenth more padding"
        assert ngram_relevance([span], chunk, coverage_threshold=0.99, min_run=5) is True

    def test_multi_span_any_match(self) -> None:
        span_1 = "topic one about neural networks and deep learning transformer architectures with attention mechanisms"
        span_2 = "topic two about database query optimization and cost models with histograms and join selectivity"
        chunk = (
            "topic two about database query optimization and cost models with histograms and join selectivity indexes"
        )
        assert ngram_relevance([span_1, span_2], chunk) is True

    def test_short_span_substring_fallback(self) -> None:
        """Spans too short for 5-grams fall back to normalized substring."""
        span = "foo bar"
        chunk = "noise foo bar noise"
        assert ngram_relevance([span], chunk) is True
        assert ngram_relevance([span], "unrelated content") is False


class TestNormalizedContains:
    def test_pipes_stripped(self) -> None:
        assert normalized_contains("Alpha | Beta", "Alpha Beta in context")

    def test_unicode_folded(self) -> None:
        assert normalized_contains("he said “hello”", 'he said "hello"')


class TestLocateSpanInDoc:
    def test_exact_match(self) -> None:
        doc = "First paragraph. The answer is here. Next section."
        span = "The answer is here."
        result = _locate_span_in_doc(span, doc, fuzzy_threshold=0.9)
        assert result is not None
        start, end, text = result
        assert doc[start:end] == text == span

    def test_whitespace_tolerant(self) -> None:
        doc = "Intro.\nLine 1: value = 42\nLine 2: extra"
        # LLM drifted the whitespace
        span = "Line 1: value = 42 Line 2: extra"
        result = _locate_span_in_doc(span, doc, fuzzy_threshold=0.9)
        assert result is not None
        _, _, actual = result
        # The returned text is the actual doc substring (with original whitespace).
        assert "Line 1: value = 42" in actual

    def test_fuzzy_snap_on_punctuation_drift(self) -> None:
        doc = (
            "The treatment achieved a response rate of 45.2% in the severe cohort "
            "with statistical significance. Further analysis is needed."
        )
        # Minor drift: swapped punctuation, extra spaces.
        span = "The treatment achieved a response rate of 45.2%  in the severe cohort with statistical significance"
        result = _locate_span_in_doc(span, doc, fuzzy_threshold=0.5)
        assert result is not None

    def test_missing_returns_none(self) -> None:
        doc = "Some unrelated text about weather."
        span = "Completely different content about space exploration and rockets"
        assert _locate_span_in_doc(span, doc, fuzzy_threshold=0.9) is None


class TestVerifySourceFacts:
    def test_verbatim_span_passes_and_records_offset(self) -> None:
        span = "The primary finding is a 45% reduction in adverse events."
        doc = f"Introduction. {span} This confirms prior work."
        q = _make_question(source_fact=[span])

        result = verify_source_facts([q], {"doc_0": doc}, min_source_fact_length=10)

        assert len(result) == 1
        updated = result[0]
        assert updated.source_fact_offsets
        start, end = updated.source_fact_offsets[0]
        assert doc[start:end] == updated.source_fact[0]

    def test_empty_span_list_rejected(self) -> None:
        q = _make_question(source_fact=[])
        result = verify_source_facts([q], {"doc_0": "any text"}, min_source_fact_length=10)
        assert result == []

    def test_total_length_below_min(self) -> None:
        q = _make_question(source_fact=["short"])
        result = verify_source_facts([q], {"doc_0": "short"}, min_source_fact_length=100)
        assert result == []

    def test_missing_doc_id_passes_through(self) -> None:
        q = _make_question(source_fact=["anything goes here"])
        # Unknown doc_id → skip verification, keep question
        result = verify_source_facts([q], {"other_doc": "text"}, min_source_fact_length=5)
        assert len(result) == 1

    def test_non_verbatim_span_rejected(self) -> None:
        span = "this exact wording is not in the document anywhere"
        q = _make_question(source_fact=[span])
        # Doc has different content entirely
        doc = "A completely unrelated document about weather patterns in the tropics."
        result = verify_source_facts([q], {"doc_0": doc}, min_source_fact_length=10)
        assert result == []

    def test_multi_span_all_must_locate(self) -> None:
        span_1 = "First part of the answer about neural networks."
        span_2 = "Missing from the document entirely — fabricated content."
        doc = f"Intro. {span_1} More text."
        q = _make_question(source_fact=[span_1, span_2])

        result = verify_source_facts([q], {"doc_0": doc}, min_source_fact_length=10)
        assert result == []


class TestChunkContainsSourceFact:
    def test_vector_chunk_interval_match(self) -> None:
        q = _make_question(
            source_fact=["x" * 200],
            source_fact_offsets=[(100, 300)],
        )
        chunk = RetrievedDocument(
            id="chunk_0",
            text="x" * 100,
            score=1.0,
            metadata={"doc_id": "doc_0"},
            char_range=(150, 250),  # 100 chars of overlap
        )
        assert chunk_contains_source_fact(q, chunk) is True

    def test_vector_chunk_different_doc_not_matched(self) -> None:
        q = _make_question(
            source_fact=["x" * 200],
            source_fact_offsets=[(0, 200)],
            doc_id="doc_0",
        )
        chunk = RetrievedDocument(
            id="chunk_0",
            text="x" * 100,
            score=1.0,
            metadata={"doc_id": "doc_1"},  # different doc
            char_range=(0, 100),
        )
        # With no ngram fallback match, should return False.
        assert chunk_contains_source_fact(q, chunk) is False

    def test_graph_verbatim_chunk_located_via_find(self) -> None:
        span = "The mitochondrion is the powerhouse of the cell in eukaryotic biology."
        doc = f"Intro paragraph. {span} Continuation text with more content."
        q = _make_question(
            source_fact=[span],
            source_fact_offsets=[(len("Intro paragraph. "), len("Intro paragraph. ") + len(span))],
        )
        chunk = RetrievedDocument(
            id="lgchunk_xyz",
            text=span,
            score=1.0,
            metadata={"file_path": "doc_0"},
            char_range=None,
        )
        cache: dict = {}
        assert chunk_contains_source_fact(q, chunk, docs={"doc_0": doc}, offset_cache=cache) is True
        # Second call should hit cache.
        assert chunk.id in cache

    def test_synthesized_entity_uses_ngram_fallback(self) -> None:
        shared = "factor alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
        q = _make_question(
            source_fact=[shared + " padding " * 20],
            source_fact_offsets=[(0, 100)],
        )
        chunk = RetrievedDocument(
            id="lgentity_factor_alpha",
            text=f"[Entity: factor_alpha] {shared}",
            score=0.5,
            metadata={},
            char_range=None,
        )
        assert chunk_contains_source_fact(q, chunk, docs={}, offset_cache={}) is True


class TestCheckParametricLeaks:
    """Mock LLM answers; verify majority-vote removal."""

    async def test_question_answerable_without_context_removed(self) -> None:
        q = _make_question()
        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=_make_litellm_response("A"),  # always correct
        ):
            result = await check_parametric_leaks([q], model="m", concurrency=1, n_trials=3)

        assert result == []  # leaked → removed

    async def test_unleaked_question_kept(self) -> None:
        q = _make_question()
        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=_make_litellm_response("E"),  # insufficient context
        ):
            result = await check_parametric_leaks([q], model="m", concurrency=1, n_trials=3)

        assert len(result) == 1


class TestCheckOracle:
    async def test_oracle_accepts_when_correct(self) -> None:
        q = _make_question()
        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=_make_litellm_response("A"),
        ):
            result = await check_oracle([q], model="m", concurrency=1)

        assert len(result) == 1

    async def test_oracle_rejects_when_wrong(self) -> None:
        q = _make_question()
        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=_make_litellm_response("D"),
        ):
            result = await check_oracle([q], model="m", concurrency=1)

        assert result == []

    async def test_oracle_retries_with_full_doc_on_e(self) -> None:
        span = "The treatment achieved a 45% response rate."
        doc = f"Long context. {span} More text."
        q = _make_question(source_fact=[span])
        calls = [_make_litellm_response("E"), _make_litellm_response("A")]

        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=calls,
        ) as mock:
            result = await check_oracle(
                [q],
                model="m",
                concurrency=1,
                documents={"doc_0": doc},
                oracle_retry_with_full_doc=True,
            )

        assert len(result) == 1
        assert mock.await_count == 2


class TestFilterEasyRetrieval:
    def test_easy_question_removed_via_ngram(self) -> None:
        span = "the mitochondrion is the powerhouse of the cell in eukaryotic organisms"
        q = _make_question(source_fact=[span + " extra padding content"])
        chunks = [span + " nearby text", "unrelated alpha beta gamma"]
        chunk_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        class DummyEmbedder:
            def encode(self, texts: list[str]) -> np.ndarray:
                return np.array([[1.0, 0.0]], dtype=np.float32)

        result = filter_easy_retrieval(
            [q],
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            embedder=DummyEmbedder(),
            max_easy_rank=1,
        )
        assert result == []

    def test_hard_question_kept(self) -> None:
        span = "the very specific fact that appears nowhere in the candidate chunks"
        q = _make_question(source_fact=[span + " with lots of unique surrounding padding text"])
        chunks = ["completely unrelated content about weather", "more weather discussion"]
        chunk_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        class DummyEmbedder:
            def encode(self, texts: list[str]) -> np.ndarray:
                return np.array([[1.0, 0.0]], dtype=np.float32)

        result = filter_easy_retrieval(
            [q],
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            embedder=DummyEmbedder(),
            max_easy_rank=1,
        )
        assert len(result) == 1

    def test_offset_based_match_when_ranges_provided(self) -> None:
        q = _make_question(
            source_fact=["x" * 200],
            source_fact_offsets=[(100, 300)],
        )
        chunks = ["some chunk text"]
        chunk_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        chunk_ranges = [(150, 250)]  # overlaps source_fact range
        chunk_doc_ids = ["doc_0"]

        class DummyEmbedder:
            def encode(self, texts: list[str]) -> np.ndarray:
                return np.array([[1.0, 0.0]], dtype=np.float32)

        result = filter_easy_retrieval(
            [q],
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            embedder=DummyEmbedder(),
            max_easy_rank=1,
            chunk_ranges=chunk_ranges,
            chunk_doc_ids=chunk_doc_ids,
        )
        assert result == []  # matched, removed as trivially retrievable


class TestRunValidationPipeline:
    async def test_rejects_non_verbatim_source_fact(self) -> None:
        q = _make_question(source_fact=["this span does not appear anywhere in the doc"])
        docs = {"doc_0": "Entirely different content about astronomy and stars."}

        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=_make_litellm_response("A"),
        ):
            result = await run_validation_pipeline(
                [q],
                documents=docs,
                model="m",
                concurrency=1,
                detect_parametric_leaks=False,
                source_fact_min_length=10,
            )

        assert result == []

    async def test_accepts_verbatim_question_that_passes_oracle(self) -> None:
        span = "The primary mechanism of RAG combines retrieval with neural generation."
        doc = f"Intro. {span} More context."
        q = _make_question(source_fact=[span])

        with patch(
            "agentic_autorag.examiner.exam_validator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=_make_litellm_response("A"),
        ):
            result = await run_validation_pipeline(
                [q],
                documents={"doc_0": doc},
                model="m",
                concurrency=1,
                detect_parametric_leaks=False,
                source_fact_min_length=10,
            )

        assert len(result) == 1
        assert result[0].source_fact_offsets
