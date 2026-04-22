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
from agentic_autorag.examiner._errors import format_llm_error, is_permanent_llm_error

logger = logging.getLogger(__name__)
run_logger = logging.getLogger("agentic_autorag.run")

_ERROR_SENTINEL = "QUESTION_EVALUATION_ERROR"
_PERMANENT_ERROR_SENTINEL = "QUESTION_PERMANENT_ERROR"
_RETRY_COOLDOWNS = (10, 30, 60)
_SLOW_THRESHOLD_S = 40.0


class QuestionResult(BaseModel):
    """Result of evaluating a single MCQ question."""

    question_id: str
    correct: bool
    selected_answer: str  # option letter or "INVALID"
    correct_answer: str
    retrieved_context: str
    generated_response: str
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    model_s: float = 0.0  # actual retrieval model compute (excludes queue wait)
    source_fact_rank: int = 0  # 1-indexed rank of first chunk overlapping any source_fact span, 0 = not found
    chunk_precision: float = 0.0  # fraction of retrieved chunks that overlap source_fact (deterministic)
    retrieved_doc_ids: list[str] = []  # source document id per retrieved chunk, in rank order

    @property
    def context_sufficient(self) -> bool:
        """True when at least one retrieved chunk overlaps the source_fact."""
        return self.chunk_precision > 0


class ExamResult(BaseModel):
    """Aggregated result of evaluating a full MCQ exam.

    ``n_total`` is the exam size. ``n_valid`` is the subset whose responses
    resolved without system-error sentinels (timeouts, API failures).
    ``mcq_accuracy`` and ``mean_retrieval_quality`` are computed over ``n_valid``
    so external flakiness doesn't penalise a configuration. ``compute_stage_metrics``
    already applies the same exclusion; keeping it consistent here prevents
    cross-trial comparisons from drifting when a few questions fail to resolve.
    """

    score: float  # composite: alpha * mcq_accuracy + (1 - alpha) * mean_retrieval_quality
    n_correct: int
    n_total: int
    n_valid: int = 0  # questions that produced a non-sentinel response
    question_results: list[QuestionResult]
    mcq_accuracy: float = 0.0
    mean_retrieval_quality: float = 0.0

    def failed_questions(self) -> list[QuestionResult]:
        """Return only the incorrect question results."""
        return [qr for qr in self.question_results if not qr.correct]


class MCQEvaluator:
    """Evaluates a RAG pipeline against an MCQ exam."""

    MCQ_ANSWER_PROMPT = """\
Answer the following multiple-choice question based ONLY on the provided context. \
Reply with just the letter (A, B, C, or D).

Context:
{context}

Question: {question}
{options}

Answer:"""

    def __init__(
        self,
        concurrency: int = 10,
        retrieval_quality_alpha: float = 0.3,
        *,
        documents: dict[str, str] | None = None,
        chunk_relevance_min_overlap_chars: int = 50,
        chunk_relevance_ngram_size: int = 5,
        chunk_relevance_overlap_threshold: float = 0.5,
        chunk_relevance_min_run: int = 5,
    ) -> None:
        self.concurrency = concurrency
        self.alpha = retrieval_quality_alpha
        # Doc texts for eval-time offset lookup of verbatim graph chunks.
        self.documents: dict[str, str] = documents or {}
        # LRU-ish cache: chunk_id -> (doc_id, start, end) or None (not locatable).
        self._graph_offset_cache: dict[str, tuple[str, int, int] | None] = {}
        self.min_overlap_chars = chunk_relevance_min_overlap_chars
        self.ngram_size = chunk_relevance_ngram_size
        self.coverage_threshold = chunk_relevance_overlap_threshold
        self.min_run = chunk_relevance_min_run

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
        qnum_map = {q.id: i for i, q in enumerate(exam, start=1)}

        await self._run_pass(results_by_id, pipeline, exam, qnum_map, desc="Evaluating MCQs")

        n_permanent = sum(1 for q in exam if results_by_id[q.id].generated_response == _PERMANENT_ERROR_SENTINEL)
        if n_permanent:
            tqdm.write(f"\n  {n_permanent} question(s) hit permanent errors (content policy, etc.) — skipping retries")

        for retry_round, cooldown in enumerate(_RETRY_COOLDOWNS, start=1):
            retryable = [q for q in exam if results_by_id[q.id].generated_response == _ERROR_SENTINEL]
            if not retryable:
                break

            tqdm.write(
                f"\n  {len(retryable)} question(s) failed (transient)"
                f" — retrying after {cooldown}s cooldown"
                f" (round {retry_round}/{len(_RETRY_COOLDOWNS)})"
            )
            await asyncio.sleep(cooldown)

            await self._run_pass(
                results_by_id,
                pipeline,
                retryable,
                qnum_map,
                desc=f"Retry round {retry_round}",
            )

        still_failed = sum(
            1 for q in exam if results_by_id[q.id].generated_response in (_ERROR_SENTINEL, _PERMANENT_ERROR_SENTINEL)
        )
        if still_failed:
            tqdm.write(f"\n  {still_failed} question(s) still failed after retries")

        results = [results_by_id[q.id] for q in exam]
        n_total = len(results)
        # Error-sentinel responses reflect external flakiness, not pipeline quality.
        # Exclude them from both numerator and denominator so a trial isn't penalised
        # by timeouts it didn't cause. Matches compute_stage_metrics' convention.
        valid_results = [r for r in results if r.generated_response not in (_ERROR_SENTINEL, _PERMANENT_ERROR_SENTINEL)]
        n_valid = len(valid_results)
        n_correct = sum(1 for r in valid_results if r.correct)
        mcq_accuracy = n_correct / n_valid if n_valid else 0.0
        mean_retrieval_quality = sum(r.chunk_precision for r in valid_results) / n_valid if n_valid else 0.0
        score = self.alpha * mcq_accuracy + (1 - self.alpha) * mean_retrieval_quality

        run_logger.info(
            "Chunk precision: mean=%.3f over %d valid questions",
            mean_retrieval_quality,
            n_valid,
        )
        if n_valid < n_total:
            run_logger.info(
                "Excluded %d error-sentinel question(s) from mcq_accuracy/rq (n_valid=%d of %d)",
                n_total - n_valid,
                n_valid,
                n_total,
            )
        return ExamResult(
            score=score,
            n_correct=n_correct,
            n_total=n_total,
            n_valid=n_valid,
            question_results=results,
            mcq_accuracy=mcq_accuracy,
            mean_retrieval_quality=mean_retrieval_quality,
        )

    async def _run_pass(
        self,
        results_by_id: dict[str, QuestionResult],
        pipeline: RAGPipeline,
        questions: list[MCQQuestion],
        qnum_map: dict[str, int],
        desc: str,
    ) -> None:
        """Run a semaphore-bounded concurrent pass over *questions*.

        Up to *concurrency* questions run simultaneously. Each question acquires
        the semaphore individually, so a slow question never holds back others.
        """
        sem = asyncio.Semaphore(self.concurrency)

        with tqdm(total=len(questions), desc=desc, unit="q") as pbar:

            async def _bounded(q: MCQQuestion) -> None:
                async with sem:
                    t0 = time.monotonic()
                    qr = await self._evaluate_single(pipeline, q)
                    elapsed = time.monotonic() - t0

                qnum = qnum_map.get(q.id, 0)
                label = f"Q{qnum:02d}"
                queue_s = max(qr.retrieval_s - qr.model_s, 0.0)
                timing_detail = f"(retr={qr.model_s:.1f}s llm={qr.generation_s:.1f}s queue={queue_s:.1f}s)"

                if qr.generated_response in (_ERROR_SENTINEL, _PERMANENT_ERROR_SENTINEL):
                    pass  # already printed in _evaluate_single
                elif not qr.correct:
                    tqdm.write(
                        f"  MISS {label}"
                        f" | selected={qr.selected_answer} correct={qr.correct_answer}"
                        f" | {elapsed:.1f}s {timing_detail}"
                    )
                elif elapsed >= _SLOW_THRESHOLD_S:
                    tqdm.write(f"  SLOW {label} | {elapsed:.1f}s {timing_detail}")

                results_by_id[q.id] = qr
                pbar.update(1)

            await asyncio.gather(*[_bounded(q) for q in questions])

    async def _evaluate_single(
        self,
        pipeline: RAGPipeline,
        q: MCQQuestion,
    ) -> QuestionResult:
        """Evaluate a single MCQ question against the pipeline."""
        llm_timeout = pipeline.config.llm_timeout_s
        question_timeout = llm_timeout + 30 if llm_timeout is not None else None
        try:
            async with asyncio.timeout(question_timeout):
                t0 = time.monotonic()
                retrieval_result = await pipeline.retrieve(q.question)
                retrieval_s = time.monotonic() - t0

                # Deterministic chunk-relevance: tier 1 offset interval overlap,
                # tier 2 str.find for verbatim graph chunks, tier 3 n-gram for
                # synthesized graph content. Lazy import to avoid circular dep.
                from agentic_autorag.examiner.exam_validator import chunk_contains_source_fact

                source_fact_rank = 0
                n_relevant = 0
                for rank, doc in enumerate(retrieval_result.documents, start=1):
                    if chunk_contains_source_fact(
                        q,
                        doc,
                        docs=self.documents,
                        offset_cache=self._graph_offset_cache,
                        min_overlap_chars=self.min_overlap_chars,
                        ngram_size=self.ngram_size,
                        coverage_threshold=self.coverage_threshold,
                        min_run=self.min_run,
                    ):
                        n_relevant += 1
                        if source_fact_rank == 0:
                            source_fact_rank = rank
                chunk_precision = n_relevant / len(retrieval_result.documents) if retrieval_result.documents else 0.0

                context = "\n".join(doc.text for doc in retrieval_result.documents)

                options_text = "\n".join(f"{k}) {v}" for k, v in q.options.items())
                prompt = self.MCQ_ANSWER_PROMPT.format(
                    context=context,
                    question=q.question,
                    options=options_text,
                )

                t0 = time.monotonic()
                answer = await pipeline.generate(prompt)
                generation_s = time.monotonic() - t0

            selected = self._parse_answer(answer, valid_keys=set(q.options.keys()))

            retrieved_doc_ids = [str(doc.metadata.get("doc_id", "")) for doc in retrieval_result.documents]

            return QuestionResult(
                question_id=q.id,
                correct=selected == q.correct_answer,
                selected_answer=selected,
                correct_answer=q.correct_answer,
                retrieved_context=context,
                generated_response=answer,
                retrieval_s=retrieval_s,
                model_s=retrieval_result.timing.model_s,
                generation_s=generation_s,
                source_fact_rank=source_fact_rank,
                chunk_precision=chunk_precision,
                retrieved_doc_ids=retrieved_doc_ids,
            )
        except TimeoutError:
            timeout_msg = f"exceeded {question_timeout:.0f}s" if question_timeout is not None else "timed out"
            tqdm.write(f"  TIMEOUT {q.id} | {timeout_msg}")
            return QuestionResult(
                question_id=q.id,
                correct=False,
                selected_answer="INVALID",
                correct_answer=q.correct_answer,
                retrieved_context="",
                generated_response=_ERROR_SENTINEL,
            )
        except Exception as exc:
            error_summary = format_llm_error(exc)
            permanent = is_permanent_llm_error(exc)
            sentinel = _PERMANENT_ERROR_SENTINEL if permanent else _ERROR_SENTINEL
            tqdm.write(f"  ERROR {q.id} | {error_summary}")
            logger.debug("Question evaluation failed for %s (permanent=%s)", q.id, permanent, exc_info=True)
            return QuestionResult(
                question_id=q.id,
                correct=False,
                selected_answer="INVALID",
                correct_answer=q.correct_answer,
                retrieved_context="",
                generated_response=sentinel,
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
