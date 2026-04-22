"""Tests for the MCQ evaluator module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_autorag.config.models import MCQQuestion
from agentic_autorag.engine.pipeline import RetrievalResult, RetrievedDocument
from agentic_autorag.examiner.evaluator import ExamResult, MCQEvaluator, QuestionResult

FOUR_KEYS = {"A", "B", "C", "D"}
THREE_KEYS = {"A", "B", "C"}


class TestParseAnswer:
    """Test the regex-based answer extraction."""

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("B", "B"),
            ("b", "B"),
            ("B)", "B"),
            ("B.", "B"),
            ("B: something", "B"),
            ("The answer is B", "B"),
            ("The answer is: C", "C"),
            ("answer: D", "D"),
            ("A) First option", "A"),
            ("  D  ", "D"),
        ],
    )
    def test_common_formats(self, response: str, expected: str) -> None:
        result = MCQEvaluator._parse_answer(response, FOUR_KEYS)

        assert result == expected

    def test_invalid_response(self) -> None:
        result1 = MCQEvaluator._parse_answer("I don't know", FOUR_KEYS)
        result2 = MCQEvaluator._parse_answer("", FOUR_KEYS)

        assert result1 == "INVALID"
        assert result2 == "INVALID"

    def test_respects_valid_keys(self) -> None:
        """With a 3-option MCQ, 'D' must not be extracted even if present."""
        result = MCQEvaluator._parse_answer("The answer is D", THREE_KEYS)

        assert result == "INVALID"


class TestExamResult:
    def test_failed_questions(self) -> None:
        results = [
            QuestionResult(
                question_id="q1",
                correct=True,
                selected_answer="A",
                correct_answer="A",
                retrieved_context="",
                generated_response="A",
            ),
            QuestionResult(
                question_id="q2",
                correct=False,
                selected_answer="B",
                correct_answer="A",
                retrieved_context="",
                generated_response="B",
            ),
        ]
        exam = ExamResult(
            score=0.5,
            n_correct=1,
            n_total=2,
            question_results=results,
            mcq_accuracy=0.5,
            mean_retrieval_quality=0.0,
        )

        failed = exam.failed_questions()

        assert len(failed) == 1
        assert failed[0].question_id == "q2"

    def test_context_sufficient_derived_from_precision(self) -> None:
        qr = QuestionResult(
            question_id="q",
            correct=True,
            selected_answer="A",
            correct_answer="A",
            retrieved_context="",
            generated_response="A",
            chunk_precision=0.5,
        )
        assert qr.context_sufficient is True

        qr2 = qr.model_copy(update={"chunk_precision": 0.0})
        assert qr2.context_sufficient is False


def _make_question(
    qid: str,
    correct: str,
    *,
    source_fact: list[str] | None = None,
    source_fact_offsets: list[tuple[int, int]] | None = None,
    source_doc_id: str = "doc_0",
) -> MCQQuestion:
    return MCQQuestion(
        id=qid,
        question="What is X?",
        options={"A": "opt a", "B": "opt b", "C": "opt c", "D": "opt d"},
        correct_answer=correct,
        source_doc_ids=[source_doc_id],
        source_fact=source_fact or [],
        source_fact_offsets=source_fact_offsets or [],
        cluster_id=0,
    )


def _mock_pipeline(answer_text: str, docs: list[RetrievedDocument] | None = None) -> MagicMock:
    """Mock RAGPipeline: one default doc + deterministic generation."""
    pipeline = MagicMock()
    pipeline.config = MagicMock(llm_timeout_s=None)
    retrieval_docs = docs if docs is not None else [RetrievedDocument(id="d0", text="some context", score=1.0)]
    pipeline.retrieve = AsyncMock(return_value=RetrievalResult(documents=retrieval_docs))
    pipeline.generate = AsyncMock(return_value=answer_text)
    return pipeline


class TestEvaluate:
    async def test_all_correct(self) -> None:
        exam = [_make_question("q1", "B"), _make_question("q2", "B")]
        pipeline = _mock_pipeline("B")

        result = await MCQEvaluator().evaluate(pipeline, exam)

        assert result.mcq_accuracy == 1.0
        assert result.n_correct == 2
        assert result.n_total == 2
        assert result.failed_questions() == []

    async def test_mixed_results(self) -> None:
        exam = [_make_question("q1", "A"), _make_question("q2", "C")]
        pipeline = _mock_pipeline("A")  # always answers A

        result = await MCQEvaluator().evaluate(pipeline, exam)

        assert result.n_correct == 1
        assert result.n_total == 2
        assert result.mcq_accuracy == pytest.approx(0.5)
        assert len(result.failed_questions()) == 1
        assert result.failed_questions()[0].question_id == "q2"

    async def test_invalid_answer(self) -> None:
        exam = [_make_question("q1", "A")]
        pipeline = _mock_pipeline("I have no idea")

        result = await MCQEvaluator().evaluate(pipeline, exam)

        assert result.n_correct == 0
        assert result.question_results[0].selected_answer == "INVALID"

    async def test_empty_exam(self) -> None:
        pipeline = _mock_pipeline("A")

        result = await MCQEvaluator().evaluate(pipeline, [])

        assert result.mcq_accuracy == 0.0
        assert result.n_total == 0


class TestErrorSentinelExclusion:
    """mcq_accuracy and mean_retrieval_quality must exclude error-sentinel questions."""

    async def test_mcq_accuracy_denominator_excludes_sentinels(self) -> None:
        exam = [_make_question("q1", "A"), _make_question("q2", "A"), _make_question("q3", "A")]
        pipeline = MagicMock()
        pipeline.config = MagicMock(llm_timeout_s=None)
        pipeline.retrieve = AsyncMock(
            return_value=RetrievalResult(
                documents=[RetrievedDocument(id="d0", text="ctx", score=1.0)],
            ),
        )

        call_idx = [0]

        async def _tagged_generate(prompt: str) -> str:
            idx = call_idx[0]
            call_idx[0] += 1
            if idx == 2:
                raise RuntimeError("ContentPolicyViolation: blocked by policy")
            return "A"

        pipeline.generate = AsyncMock(side_effect=_tagged_generate)

        result = await MCQEvaluator().evaluate(pipeline, exam)

        assert result.n_total == 3
        assert result.n_valid == 2
        assert result.n_correct == 2
        assert result.mcq_accuracy == pytest.approx(1.0)


class TestRetryOnTransientFailure:
    """Transient errors should trigger retries; permanent errors should not."""

    async def test_timeout_surfaces_as_sentinel(self) -> None:
        exam = [_make_question("q1", "A")]
        pipeline = MagicMock()
        pipeline.config = MagicMock(llm_timeout_s=1)
        pipeline.retrieve = AsyncMock(
            return_value=RetrievalResult(
                documents=[RetrievedDocument(id="d0", text="ctx", score=1.0)],
            ),
        )
        pipeline.generate = AsyncMock(side_effect=TimeoutError("timed out"))

        with patch("agentic_autorag.examiner.evaluator.asyncio.sleep", new_callable=AsyncMock):
            result = await MCQEvaluator().evaluate(pipeline, exam)

        assert result.n_correct == 0
        assert result.question_results[0].selected_answer == "INVALID"


class TestDeterministicChunkPrecision:
    """Chunk precision is computed via offset interval overlap or n-gram fallback."""

    async def test_vector_chunk_with_offset_overlap_marks_relevant(self) -> None:
        span_text = "a" * 100  # 100-char verbatim span
        doc_text = span_text + " other content " * 50
        q = _make_question(
            "q1",
            "A",
            source_fact=[span_text],
            source_fact_offsets=[(0, 100)],
        )
        retrieved = [
            RetrievedDocument(
                id="chunk_0",
                text=span_text[:80],  # partial — 80 chars overlap
                score=1.0,
                metadata={"doc_id": "doc_0"},
                char_range=(0, 80),
            ),
            RetrievedDocument(
                id="chunk_1",
                text="unrelated",
                score=0.5,
                metadata={"doc_id": "doc_0"},
                char_range=(500, 600),  # no overlap with [0, 100]
            ),
        ]
        pipeline = _mock_pipeline("A", docs=retrieved)

        evaluator = MCQEvaluator(documents={"doc_0": doc_text})
        result = await evaluator.evaluate(pipeline, [q])

        qr = result.question_results[0]
        assert qr.chunk_precision == pytest.approx(0.5)
        assert qr.source_fact_rank == 1
        assert qr.context_sufficient is True

    async def test_below_min_overlap_chars_not_relevant(self) -> None:
        # Overlap is only 10 chars, below the default 50-char floor.
        q = _make_question(
            "q1",
            "A",
            source_fact=["x" * 200],
            source_fact_offsets=[(100, 300)],
        )
        retrieved = [
            RetrievedDocument(
                id="chunk_0",
                text="boundary",
                score=1.0,
                metadata={"doc_id": "doc_0"},
                char_range=(290, 310),  # 10-char overlap with [100, 300]
            ),
        ]
        pipeline = _mock_pipeline("A", docs=retrieved)
        result = await MCQEvaluator(documents={"doc_0": "x" * 500}).evaluate(pipeline, [q])

        qr = result.question_results[0]
        assert qr.chunk_precision == 0.0
        assert qr.source_fact_rank == 0

    async def test_multi_span_matches_either_span(self) -> None:
        q = _make_question(
            "q1",
            "A",
            source_fact=["alpha" * 20, "beta" * 20],
            source_fact_offsets=[(0, 100), (500, 580)],
        )
        retrieved = [
            RetrievedDocument(
                id="chunk_0",
                text="overlaps second span",
                score=1.0,
                metadata={"doc_id": "doc_0"},
                char_range=(510, 570),
            ),
        ]
        pipeline = _mock_pipeline("A", docs=retrieved)
        result = await MCQEvaluator(documents={"doc_0": "x" * 1000}).evaluate(pipeline, [q])

        assert result.question_results[0].chunk_precision == 1.0

    async def test_graph_chunk_offset_lookup_via_str_find(self) -> None:
        """lgchunk_* chunks have no char_range; evaluator should locate via str.find."""
        span_text = "the treatment was effective in severe cases"
        doc_text = f"Intro paragraph. {span_text}. Outro."
        q = _make_question(
            "q1",
            "A",
            source_fact=[span_text + " " + "filler " * 30],
            source_fact_offsets=[(17, 17 + len(span_text) + len(" filler" * 30))],
        )
        # Mimic graph_store._normalise_result: lgchunk_ prefix, file_path in metadata.
        retrieved = [
            RetrievedDocument(
                id="lgchunk_abc123",
                text=span_text,
                score=1.0,
                metadata={"file_path": "doc_0"},
                char_range=None,
            ),
        ]
        pipeline = _mock_pipeline("A", docs=retrieved)
        # Make source_fact_offsets realistic: exactly the span location in doc_text.
        q = q.model_copy(update={"source_fact_offsets": [(17, 17 + len(span_text))]})

        result = await MCQEvaluator(documents={"doc_0": doc_text}).evaluate(pipeline, [q])

        assert result.question_results[0].chunk_precision == pytest.approx(1.0)

    async def test_synthesized_graph_content_uses_ngram_fallback(self) -> None:
        """lgentity_* has no verbatim offset; match via n-gram coverage."""
        # Span contains the same 5-grams as the entity description text.
        shared_text = "the quick brown fox jumps over the lazy dog and runs into the deep forest"
        span = shared_text + " " + "padding " * 30
        q = _make_question(
            "q1",
            "A",
            source_fact=[span],
            source_fact_offsets=[(0, len(span))],
        )
        retrieved = [
            RetrievedDocument(
                id="lgentity_fox",
                text=f"[Entity: fox] {shared_text}",
                score=0.5,
                metadata={},
                char_range=None,
            ),
        ]
        pipeline = _mock_pipeline("A", docs=retrieved)
        result = await MCQEvaluator(documents={"doc_0": span}).evaluate(pipeline, [q])

        assert result.question_results[0].chunk_precision == pytest.approx(1.0)

    async def test_empty_source_fact_yields_zero_precision(self) -> None:
        q = _make_question("q1", "A")
        pipeline = _mock_pipeline("A")
        result = await MCQEvaluator().evaluate(pipeline, [q])

        assert result.question_results[0].chunk_precision == 0.0
        assert result.question_results[0].source_fact_rank == 0
