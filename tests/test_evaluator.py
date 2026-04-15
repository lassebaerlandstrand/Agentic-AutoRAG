"""Tests for the MCQ evaluator module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_autorag.config.models import MCQQuestion
from agentic_autorag.engine.pipeline import RetrievalResult, RetrievedDocument
from agentic_autorag.examiner.evaluator import (
    ExamResult,
    MCQEvaluator,
    QuestionResult,
    _judge_chunk_relevance,
    _parse_chunk_verdicts,
)

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

    def test_three_options_rejects_d(self) -> None:
        result = MCQEvaluator._parse_answer("D", THREE_KEYS)

        assert result == "INVALID"

    def test_three_options_accepts_valid(self) -> None:
        result = MCQEvaluator._parse_answer("C", THREE_KEYS)

        assert result == "C"


class TestExamResult:
    def test_failed_questions_returns_incorrect_only(self) -> None:
        results = [
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
                correct_answer="C",
                retrieved_context="ctx",
                generated_response="B",
            ),
        ]
        exam_result = ExamResult(
            score=0.5,
            n_correct=1,
            n_total=2,
            question_results=results,
        )

        failed = exam_result.failed_questions()

        assert len(failed) == 1
        assert failed[0].question_id == "q2"

    def test_all_correct(self) -> None:
        result = ExamResult(
            score=1.0,
            n_correct=1,
            n_total=1,
            question_results=[
                QuestionResult(
                    question_id="q1",
                    correct=True,
                    selected_answer="A",
                    correct_answer="A",
                    retrieved_context="",
                    generated_response="A",
                ),
            ],
        )

        failed = result.failed_questions()

        assert failed == []


def _make_question(qid: str, correct: str) -> MCQQuestion:
    return MCQQuestion(
        id=qid,
        question="What is X?",
        options={"A": "opt a", "B": "opt b", "C": "opt c", "D": "opt d"},
        correct_answer=correct,
        source_doc_ids=["doc_0"],
        cluster_id=0,
    )


def _mock_pipeline(answer_text: str) -> MagicMock:
    """Return a mock RAGPipeline that always retrieves one doc and generates *answer_text*."""
    pipeline = MagicMock()
    pipeline.config = MagicMock(llm_timeout_s=None)
    pipeline.retrieve = AsyncMock(
        return_value=RetrievalResult(
            documents=[RetrievedDocument(id="d0", text="some context", score=1.0)],
        ),
    )
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


class TestRetryOnTransientFailure:
    async def test_retries_after_transient_error(self) -> None:
        """A question that fails on the first pass should succeed after retry."""
        exam = [_make_question("q1", "B"), _make_question("q2", "B")]
        pipeline = MagicMock()
        pipeline.config = MagicMock(llm_timeout_s=None)
        pipeline.retrieve = AsyncMock(
            return_value=RetrievalResult(
                documents=[RetrievedDocument(id="d0", text="ctx", score=1.0)],
            ),
        )
        # q2 fails once then succeeds; q1 always succeeds
        call_count = 0

        async def _generate_side_effect(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ConnectionError("503 Service Unavailable")
            return "B"

        pipeline.generate = AsyncMock(side_effect=_generate_side_effect)

        with patch("agentic_autorag.examiner.evaluator.asyncio.sleep", new_callable=AsyncMock):
            result = await MCQEvaluator().evaluate(pipeline, exam)

        assert result.n_correct == 2
        assert result.n_total == 2
        assert result.mcq_accuracy == 1.0
        assert all(qr.selected_answer == "B" for qr in result.question_results)

    async def test_permanent_failure_after_all_retries(self) -> None:
        """A question that always fails stays as INVALID after all retry rounds."""
        exam = [_make_question("q1", "A")]
        pipeline = MagicMock()
        pipeline.config = MagicMock(llm_timeout_s=None)
        pipeline.retrieve = AsyncMock(
            return_value=RetrievalResult(
                documents=[RetrievedDocument(id="d0", text="ctx", score=1.0)],
            ),
        )
        pipeline.generate = AsyncMock(side_effect=ConnectionError("always fails"))

        with patch("agentic_autorag.examiner.evaluator.asyncio.sleep", new_callable=AsyncMock):
            result = await MCQEvaluator().evaluate(pipeline, exam)

        assert result.n_correct == 0
        assert result.question_results[0].selected_answer == "INVALID"

    async def test_no_retry_when_all_pass(self) -> None:
        """When all questions succeed, no retry rounds should happen."""
        exam = [_make_question("q1", "A")]
        pipeline = _mock_pipeline("A")

        with patch("agentic_autorag.examiner.evaluator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await MCQEvaluator().evaluate(pipeline, exam)

        mock_sleep.assert_not_called()
        assert result.n_correct == 1

    async def test_timeout_treated_as_transient(self) -> None:
        """A TimeoutError should be treated as transient and retried."""
        exam = [_make_question("q1", "A")]
        pipeline = MagicMock()
        pipeline.config = MagicMock(llm_timeout_s=None)
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


class TestParseChunkVerdicts:
    """Test parsing of the judge's per-chunk YES/NO response."""

    def test_all_yes(self) -> None:
        result = _parse_chunk_verdicts("1: YES\n2: YES\n3: YES", n_chunks=3)

        assert result == [True, True, True]

    def test_all_no(self) -> None:
        result = _parse_chunk_verdicts("1: NO\n2: NO\n3: NO", n_chunks=3)

        assert result == [False, False, False]

    def test_mixed(self) -> None:
        result = _parse_chunk_verdicts("1: YES\n2: NO\n3: YES", n_chunks=3)

        assert result == [True, False, True]

    def test_case_insensitive(self) -> None:
        result = _parse_chunk_verdicts("1: yes\n2: No\n3: YES", n_chunks=3)

        assert result == [True, False, True]

    def test_alternative_separators(self) -> None:
        result = _parse_chunk_verdicts("1. YES\n2) NO\n3: YES", n_chunks=3)

        assert result == [True, False, True]

    def test_missing_entry_defaults_to_false(self) -> None:
        result = _parse_chunk_verdicts("1: YES\n3: YES", n_chunks=3)

        assert result == [True, False, True]

    def test_out_of_range_ignored(self) -> None:
        result = _parse_chunk_verdicts("1: YES\n5: YES", n_chunks=3)

        assert result == [True, False, False]

    def test_malformed_response_defaults_all_false(self) -> None:
        result = _parse_chunk_verdicts("I'm not sure about these chunks.", n_chunks=3)

        assert result == [False, False, False]

    def test_empty_response(self) -> None:
        result = _parse_chunk_verdicts("", n_chunks=3)

        assert result == [False, False, False]


class TestJudgeChunkRelevance:
    """Test the LLM-as-judge chunk relevance function with mocked LLM calls."""

    @staticmethod
    def _docs(*texts: str) -> list[RetrievedDocument]:
        return [RetrievedDocument(id=f"d{i}", text=t, score=1.0) for i, t in enumerate(texts)]

    @staticmethod
    def _mock_completion(content: str) -> MagicMock:
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=content))]
        return response

    async def test_all_chunks_relevant(self) -> None:
        q = _make_question("q1", "A")
        docs = self._docs("c0", "c1", "c2")

        with patch(
            "agentic_autorag.examiner.evaluator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=self._mock_completion("1: YES\n2: YES\n3: YES"),
        ):
            sufficient, precision, rank, status = await _judge_chunk_relevance(q, docs, "model/x")

        assert sufficient is True
        assert precision == pytest.approx(1.0)
        assert rank == 1
        assert status == "ok"

    async def test_no_chunks_relevant(self) -> None:
        q = _make_question("q1", "A")
        docs = self._docs("c0", "c1", "c2")

        with patch(
            "agentic_autorag.examiner.evaluator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=self._mock_completion("1: NO\n2: NO\n3: NO"),
        ):
            sufficient, precision, rank, status = await _judge_chunk_relevance(q, docs, "model/x")

        assert sufficient is False
        assert precision == pytest.approx(0.0)
        assert rank == 0
        assert status == "ok"

    async def test_mixed_relevance(self) -> None:
        q = _make_question("q1", "A")
        docs = self._docs("c0", "c1", "c2", "c3")

        with patch(
            "agentic_autorag.examiner.evaluator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=self._mock_completion("1: NO\n2: YES\n3: NO\n4: YES"),
        ):
            sufficient, precision, rank, status = await _judge_chunk_relevance(q, docs, "model/x")

        assert sufficient is True
        assert precision == pytest.approx(0.5)
        assert rank == 2
        assert status == "ok"

    async def test_malformed_response_logs_warning_and_returns_malformed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        q = _make_question("q1", "A")
        docs = self._docs("c0", "c1")

        with (
            caplog.at_level("WARNING", logger="agentic_autorag.run"),
            patch(
                "agentic_autorag.examiner.evaluator.litellm.acompletion",
                new_callable=AsyncMock,
                return_value=self._mock_completion("I cannot determine relevance."),
            ),
        ):
            sufficient, precision, rank, status = await _judge_chunk_relevance(q, docs, "model/x")

        assert (sufficient, precision, rank, status) == (False, 0.0, 0, "malformed")
        assert any("malformed" in rec.message.lower() for rec in caplog.records)

    async def test_non_transient_error_returns_error_status(self, caplog: pytest.LogCaptureFixture) -> None:
        q = _make_question("q1", "A")
        docs = self._docs("c0", "c1")

        with (
            caplog.at_level("WARNING", logger="agentic_autorag.run"),
            patch(
                "agentic_autorag.examiner.evaluator.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=ValueError("bad request"),
            ),
        ):
            sufficient, precision, rank, status = await _judge_chunk_relevance(q, docs, "model/x")

        assert (sufficient, precision, rank, status) == (False, 0.0, 0, "error")
        assert any("judge llm error" in rec.message.lower() for rec in caplog.records)

    async def test_transient_error_retries_then_succeeds(self) -> None:
        """A transient error should trigger a retry — subsequent success returns 'ok'."""
        q = _make_question("q1", "A")
        docs = self._docs("c0")

        class RateLimitError(Exception):
            pass

        side_effects = [RateLimitError("429"), self._mock_completion("1: YES")]

        with (
            patch("agentic_autorag.examiner.evaluator.asyncio.sleep", new_callable=AsyncMock),
            patch(
                "agentic_autorag.examiner.evaluator.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=side_effects,
            ),
        ):
            sufficient, precision, rank, status = await _judge_chunk_relevance(q, docs, "model/x")

        assert (sufficient, precision, rank, status) == (True, 1.0, 1, "ok")

    async def test_empty_docs_skips_llm_call(self) -> None:
        q = _make_question("q1", "A")

        with patch(
            "agentic_autorag.examiner.evaluator.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_call:
            sufficient, precision, rank, status = await _judge_chunk_relevance(q, [], "model/x")

        mock_call.assert_not_called()
        assert (sufficient, precision, rank, status) == (False, 0.0, 0, "skipped")

    async def test_no_model_skips_llm_call(self) -> None:
        q = _make_question("q1", "A")
        docs = self._docs("c0")

        with patch(
            "agentic_autorag.examiner.evaluator.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_call:
            sufficient, precision, rank, status = await _judge_chunk_relevance(q, docs, None)

        mock_call.assert_not_called()
        assert (sufficient, precision, rank, status) == (False, 0.0, 0, "skipped")


class TestRetrievalQualityScore:
    """Test that the composite score is driven by chunk_precision, not retrieval_mrr."""

    async def test_score_uses_chunk_precision(self) -> None:
        exam = [_make_question("q1", "A"), _make_question("q2", "A")]
        pipeline = _mock_pipeline("A")

        with patch(
            "agentic_autorag.examiner.evaluator.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="1: YES"))],
            ),
        ):
            result = await MCQEvaluator(examiner_model="model/x").evaluate(pipeline, exam)

        # 2/2 correct, precision=1.0 on all retrieved docs (1 relevant / 1 retrieved per q)
        assert result.mcq_accuracy == pytest.approx(1.0)
        assert result.mean_retrieval_quality == pytest.approx(1.0)
        # composite = 0.3 * 1.0 + 0.7 * 1.0 = 1.0
        assert result.score == pytest.approx(1.0)
        assert all(qr.context_sufficient for qr in result.question_results)
        assert all(qr.chunk_precision == 1.0 for qr in result.question_results)
        assert all(qr.first_relevant_rank == 1 for qr in result.question_results)

    async def test_score_without_examiner_model_ignores_retrieval(self) -> None:
        """Without an examiner model, chunk_precision stays 0 — score reflects MCQ only."""
        exam = [_make_question("q1", "A"), _make_question("q2", "A")]
        pipeline = _mock_pipeline("A")

        result = await MCQEvaluator().evaluate(pipeline, exam)

        assert result.mcq_accuracy == pytest.approx(1.0)
        assert result.mean_retrieval_quality == pytest.approx(0.0)
        # composite = 0.3 * 1.0 + 0.7 * 0.0 = 0.3
        assert result.score == pytest.approx(0.3)


class TestSourceFactAnomalyWarning:
    """source_fact at rank >= 1 but judge says 0 relevant → WARN (the Q25 pattern)."""

    async def test_anomaly_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        fact = "the mitochondrion is the powerhouse of the cell"
        q = MCQQuestion(
            id="q1",
            question="What is X?",
            options={"A": "opt a", "B": "opt b", "C": "opt c", "D": "opt d"},
            correct_answer="A",
            source_doc_ids=["doc_0"],
            source_fact=fact,
            cluster_id=0,
        )
        pipeline = MagicMock()
        pipeline.config = MagicMock(llm_timeout_s=None)
        pipeline.retrieve = AsyncMock(
            return_value=RetrievalResult(
                documents=[RetrievedDocument(id="d0", text=f"Intro. {fact}. Outro.", score=1.0)],
            ),
        )
        pipeline.generate = AsyncMock(return_value="A")

        with (
            caplog.at_level("WARNING", logger="agentic_autorag.run"),
            patch(
                "agentic_autorag.examiner.evaluator.litellm.acompletion",
                new_callable=AsyncMock,
                return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="1: NO"))]),
            ),
        ):
            result = await MCQEvaluator(examiner_model="model/x").evaluate(pipeline, [q])

        qr = result.question_results[0]
        assert qr.source_fact_rank == 1
        assert qr.first_relevant_rank == 0
        assert qr.judge_status == "ok"
        assert any("judge anomaly" in rec.message.lower() for rec in caplog.records)
