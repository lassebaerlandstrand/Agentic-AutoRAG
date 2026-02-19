"""MCQ evaluator — runs an exam against a RAG pipeline and scores the results."""

from __future__ import annotations

import asyncio
import logging
import re
import time

from pydantic import BaseModel
from tqdm import tqdm

from agentic_autorag.config.models import MCQQuestion
from agentic_autorag.engine.pipeline import RAGPipeline
from agentic_autorag.examiner._errors import format_llm_error

logger = logging.getLogger(__name__)

_ERROR_SENTINEL = "QUESTION_EVALUATION_ERROR"
_RETRY_COOLDOWNS = (10, 30, 60)


class QuestionResult(BaseModel):
    """Result of evaluating a single MCQ question."""

    question_id: str
    correct: bool
    selected_answer: str  # option letter or "INVALID"
    correct_answer: str
    retrieved_context: str
    generated_response: str


class ExamResult(BaseModel):
    """Aggregated result of evaluating a full MCQ exam."""

    score: float
    n_correct: int
    n_total: int
    question_results: list[QuestionResult]

    def failed_questions(self) -> list[QuestionResult]:
        """Return only the incorrect question results."""
        return [qr for qr in self.question_results if not qr.correct]


class MCQEvaluator:
    """Evaluates a RAG pipeline against an MCQ exam."""

    DEFAULT_BATCH_SIZE = 10

    MCQ_ANSWER_PROMPT = """\
Answer the following multiple-choice question based ONLY on the provided context. \
Reply with just the letter (A, B, C, or D).

Context:
{context}

Question: {question}
{options}

Answer:"""

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.batch_size = batch_size

    async def evaluate(
        self,
        pipeline: RAGPipeline,
        exam: list[MCQQuestion],
    ) -> ExamResult:
        """Run every question through the pipeline and aggregate scores.

        Questions are processed concurrently in fixed-size batches. Questions that
        fail with transient errors (503, 429, etc.) are retried in batch after
        escalating cooldowns.
        """
        if not exam:
            return ExamResult(score=0.0, n_correct=0, n_total=0, question_results=[])

        results_by_id: dict[str, QuestionResult] = {}

        await self._run_pass(results_by_id, pipeline, exam, desc="Evaluating MCQs")

        for retry_round, cooldown in enumerate(_RETRY_COOLDOWNS, start=1):
            failed_questions = [q for q in exam if results_by_id[q.id].generated_response == _ERROR_SENTINEL]
            if not failed_questions:
                break

            tqdm.write(
                f"\n  {len(failed_questions)} question(s) failed"
                f" — retrying after {cooldown}s cooldown"
                f" (round {retry_round}/{len(_RETRY_COOLDOWNS)})"
            )
            await asyncio.sleep(cooldown)

            await self._run_pass(
                results_by_id,
                pipeline,
                failed_questions,
                desc=f"Retry round {retry_round}",
            )

        still_failed = sum(1 for q in exam if results_by_id[q.id].generated_response == _ERROR_SENTINEL)
        if still_failed:
            tqdm.write(f"\n  {still_failed} question(s) still failed after {len(_RETRY_COOLDOWNS)} retry rounds")

        results = [results_by_id[q.id] for q in exam]
        n_correct = sum(1 for r in results if r.correct)
        n_total = len(results)
        return ExamResult(
            score=n_correct / n_total if n_total else 0.0,
            n_correct=n_correct,
            n_total=n_total,
            question_results=results,
        )

    async def _run_pass(
        self,
        results_by_id: dict[str, QuestionResult],
        pipeline: RAGPipeline,
        questions: list[MCQQuestion],
        desc: str,
    ) -> None:
        """Run a concurrent pass over *questions* in fixed-size batches."""
        with tqdm(total=len(questions), desc=desc, unit="q") as pbar:
            for batch_start in range(0, len(questions), self.batch_size):
                batch = questions[batch_start : batch_start + self.batch_size]
                batch_t0 = time.monotonic()

                batch_results = await asyncio.gather(*[self._evaluate_single(pipeline, q) for q in batch])
                batch_elapsed = time.monotonic() - batch_t0

                for q, qr in zip(batch, batch_results, strict=True):
                    if not qr.correct and qr.generated_response != _ERROR_SENTINEL:
                        tqdm.write(
                            f"  MISS {q.id}"
                            f" | selected={qr.selected_answer} correct={qr.correct_answer}"
                            f" | {batch_elapsed:.1f}s"
                        )
                    results_by_id[q.id] = qr
                pbar.update(len(batch))

    async def _evaluate_single(
        self,
        pipeline: RAGPipeline,
        q: MCQQuestion,
    ) -> QuestionResult:
        """Evaluate a single MCQ question against the pipeline."""
        try:
            retrieval_result = await pipeline.retrieve(q.question)
            context = "\n".join(doc.text for doc in retrieval_result.documents)

            options_text = "\n".join(f"{k}) {v}" for k, v in q.options.items())
            prompt = self.MCQ_ANSWER_PROMPT.format(
                context=context,
                question=q.question,
                options=options_text,
            )

            answer = await pipeline.generate(prompt)
            selected = self._parse_answer(answer, valid_keys=set(q.options.keys()))

            return QuestionResult(
                question_id=q.id,
                correct=selected == q.correct_answer,
                selected_answer=selected,
                correct_answer=q.correct_answer,
                retrieved_context=context,
                generated_response=answer,
            )
        except Exception as exc:
            error_summary = format_llm_error(exc)
            tqdm.write(f"  ERROR {q.id} | {error_summary}")
            logger.debug("Question evaluation failed for %s", q.id, exc_info=True)
            return QuestionResult(
                question_id=q.id,
                correct=False,
                selected_answer="INVALID",
                correct_answer=q.correct_answer,
                retrieved_context="",
                generated_response=_ERROR_SENTINEL,
            )

    @staticmethod
    def _parse_answer(response: str, valid_keys: set[str]) -> str:
        """Extract the first valid option letter from a free-form LLM response.

        Handles formats like "B", "b", "The answer is B", "B)", "B.", etc.
        Returns "INVALID" if no valid letter is found.
        """
        text = response.strip()

        keys_upper = sorted(valid_keys)
        keys_pattern = "|".join(keys_upper)

        patterns = [
            rf"(?:the\s+)?answer\s*(?:is|:)\s*({keys_pattern})\b",
            rf"\b({keys_pattern})\s*[).:]\s",
            rf"^({keys_pattern})\b",
            rf"\b({keys_pattern})$",
            rf"\b({keys_pattern})\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return "INVALID"
