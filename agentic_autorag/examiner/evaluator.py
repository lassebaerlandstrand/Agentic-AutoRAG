"""MCQ evaluator — runs an exam against a RAG pipeline and scores the results."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Literal

import litellm
from pydantic import BaseModel
from tqdm import tqdm

from agentic_autorag.config.models import MCQQuestion
from agentic_autorag.engine.pipeline import RAGPipeline, RetrievedDocument
from agentic_autorag.examiner._errors import format_llm_error, is_permanent_llm_error, is_transient_llm_error

logger = logging.getLogger(__name__)
run_logger = logging.getLogger("agentic_autorag.run")

_ERROR_SENTINEL = "QUESTION_EVALUATION_ERROR"
_PERMANENT_ERROR_SENTINEL = "QUESTION_PERMANENT_ERROR"
_RETRY_COOLDOWNS = (10, 30, 60)
_SLOW_THRESHOLD_S = 40.0

_JUDGE_RETRY_COOLDOWNS = (5, 15)
_JUDGE_SAMPLE_LIMIT = 2  # DEBUG-log this many raw judge exchanges per evaluate() pass
_judge_sample_counter = 0

JudgeStatus = Literal["ok", "malformed", "error", "skipped"]

_CHUNK_RELEVANCE_PROMPT = """\
You are evaluating retrieval quality. For each retrieved chunk below, decide \
whether it contains information relevant to answering this question (fully or partially).

Question: {question}

Retrieved chunks (in retrieval order):
{chunks}

Reply with one line per chunk, one verdict per line, in the exact format \
"N: YES" or "N: NO" — no other text, no explanations.

Example response for 3 chunks where the first and third are relevant:
1: YES
2: NO
3: YES

Your response:"""

_CHUNK_VERDICT_PATTERN = re.compile(r"^\s*(\d+)\s*[:.)]\s*(YES|NO)\b", re.IGNORECASE | re.MULTILINE)


# TODO: When multi-hop questions are implemented, extend the prompt to instruct
# the judge to mark chunks containing ANY part of the needed information as YES
# (vs requiring the full answer in a single chunk). Consider adding an explicit
# "overall sufficient?" question in the same call to catch multi-hop cases
# where no single chunk is sufficient but the combination is.
async def _judge_chunk_relevance(
    q: MCQQuestion,
    retrieved_docs: list[RetrievedDocument],
    model: str | None,
) -> tuple[bool, float, int, JudgeStatus]:
    """Ask a judge LLM to rate each retrieved chunk for relevance.

    Returns (context_sufficient, chunk_precision, first_relevant_rank, judge_status).
    judge_status distinguishes normal operation from silent failures so the evaluator
    can report aggregate counts and flag anomalies.
    """
    if not model or not retrieved_docs:
        return False, 0.0, 0, "skipped"

    chunks_text = "\n".join(f"[{i}] {doc.text}" for i, doc in enumerate(retrieved_docs, start=1))
    prompt = _CHUNK_RELEVANCE_PROMPT.format(question=q.question, chunks=chunks_text)

    raw = ""
    last_exc: Exception | None = None
    for cooldown in (0, *_JUDGE_RETRY_COOLDOWNS):
        if cooldown:
            await asyncio.sleep(cooldown)
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                num_retries=0,
            )
            raw = response.choices[0].message.content or ""
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            if not is_transient_llm_error(exc):
                break

    if last_exc is not None:
        run_logger.warning("Judge LLM error for %s: %s", q.id, format_llm_error(last_exc))
        return False, 0.0, 0, "error"

    global _judge_sample_counter
    if _judge_sample_counter < _JUDGE_SAMPLE_LIMIT:
        _judge_sample_counter += 1
        run_logger.debug(
            "Judge sample %d for %s:\nPROMPT:\n%s\n\nRESPONSE:\n%s",
            _judge_sample_counter,
            q.id,
            prompt,
            raw,
        )

    if _CHUNK_VERDICT_PATTERN.search(raw) is None:
        run_logger.warning(
            "Judge malformed response for %s (chunks=%d): %s",
            q.id,
            len(retrieved_docs),
            raw[:200].replace("\n", " "),
        )
        return False, 0.0, 0, "malformed"

    verdicts = _parse_chunk_verdicts(raw, n_chunks=len(retrieved_docs))
    n_relevant = sum(1 for v in verdicts if v)
    context_sufficient = n_relevant > 0
    chunk_precision = n_relevant / len(retrieved_docs)
    first_relevant_rank = next((i for i, v in enumerate(verdicts, start=1) if v), 0)
    return context_sufficient, chunk_precision, first_relevant_rank, "ok"


def _parse_chunk_verdicts(text: str, n_chunks: int) -> list[bool]:
    """Parse lines like '1: YES' / '2: NO' into a list of booleans of length n_chunks.

    Missing or malformed entries default to False.
    """
    verdicts = [False] * n_chunks
    for match in _CHUNK_VERDICT_PATTERN.finditer(text):
        idx = int(match.group(1)) - 1
        if 0 <= idx < n_chunks:
            verdicts[idx] = match.group(2).upper() == "YES"
    return verdicts


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
    source_fact_rank: int = 0  # 1-indexed rank of first chunk containing source_fact, 0 = not found
    retrieval_mrr: float = 0.0  # 1/source_fact_rank, 0.0 if not found
    context_sufficient: bool = False  # LLM judge: any retrieved chunk was rated relevant
    chunk_precision: float = 0.0  # fraction of retrieved chunks judged relevant
    first_relevant_rank: int = 0  # 1-indexed rank of first chunk judged relevant, 0 = none
    judge_status: JudgeStatus = "skipped"  # diagnostic: how the judge call resolved


class ExamResult(BaseModel):
    """Aggregated result of evaluating a full MCQ exam."""

    score: float  # composite: alpha * mcq_accuracy + (1 - alpha) * mean_retrieval_quality
    n_correct: int
    n_total: int
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
        examiner_model: str | None = None,
    ) -> None:
        self.concurrency = concurrency
        self.alpha = retrieval_quality_alpha
        self.examiner_model = examiner_model

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

        global _judge_sample_counter
        _judge_sample_counter = 0  # reset so each evaluate() samples fresh

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
        n_correct = sum(1 for r in results if r.correct)
        n_total = len(results)
        mcq_accuracy = n_correct / n_total if n_total else 0.0
        mean_retrieval_quality = sum(r.chunk_precision for r in results) / n_total if n_total else 0.0
        score = self.alpha * mcq_accuracy + (1 - self.alpha) * mean_retrieval_quality

        judge_counts = {"ok": 0, "malformed": 0, "error": 0, "skipped": 0}
        for r in results:
            judge_counts[r.judge_status] += 1
        run_logger.info(
            "Judge: ok=%d malformed=%d errors=%d skipped=%d (of %d)",
            judge_counts["ok"],
            judge_counts["malformed"],
            judge_counts["error"],
            judge_counts["skipped"],
            n_total,
        )
        return ExamResult(
            score=score,
            n_correct=n_correct,
            n_total=n_total,
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

                # Compute source_fact rank for retrieval quality scoring.
                # Lazy import to avoid circular dependency (exam_validator → evaluator).
                from agentic_autorag.examiner.exam_validator import source_fact_matches

                source_fact_rank = 0
                if q.source_fact:
                    for rank, doc in enumerate(retrieval_result.documents, start=1):
                        if source_fact_matches(q.source_fact, doc.text):
                            source_fact_rank = rank
                            break
                retrieval_mrr = 1.0 / source_fact_rank if source_fact_rank > 0 else 0.0

                context = "\n".join(doc.text for doc in retrieval_result.documents)

                options_text = "\n".join(f"{k}) {v}" for k, v in q.options.items())
                prompt = self.MCQ_ANSWER_PROMPT.format(
                    context=context,
                    question=q.question,
                    options=options_text,
                )

                # Run the trial LLM's MCQ answer and the judge's relevance check
                # concurrently — the judge uses a cheap model and finishes before
                # the trial LLM in most cases, so wall-clock impact is near-zero.
                t0 = time.monotonic()
                answer, judge_result = await asyncio.gather(
                    pipeline.generate(prompt),
                    _judge_chunk_relevance(q, retrieval_result.documents, self.examiner_model),
                )
                context_sufficient, chunk_precision, first_relevant_rank, judge_status = judge_result
                generation_s = time.monotonic() - t0

            if source_fact_rank >= 1 and first_relevant_rank == 0 and judge_status == "ok":
                run_logger.warning(
                    "Judge anomaly for %s: source_fact at rank %d but judge rated no chunks relevant",
                    q.id,
                    source_fact_rank,
                )

            selected = self._parse_answer(answer, valid_keys=set(q.options.keys()))

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
                retrieval_mrr=retrieval_mrr,
                context_sufficient=context_sufficient,
                chunk_precision=chunk_precision,
                first_relevant_rank=first_relevant_rank,
                judge_status=judge_status,
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
