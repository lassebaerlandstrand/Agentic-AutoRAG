"""Open-ended evaluator — runs an exam against a RAG pipeline and scores answers.

Free-text scoring stack (cheapest first):
  1. Normalised EM against canonical_answer + variants  (free, deterministic)
  2. Token F1 against canonical_answer + variants        (free, deterministic)
  3. LLM judge fallback (CRAG 3-way)                    (only when EM=0)

A question is **correct** iff EM>0 or judge=1. The ``ExamResult`` reports
verdict-path counts and per-trial open-ended failure-mode counters
(retrieval_complete / partial_a / partial_b / miss / refused / correct-given-complete)
so the diagnoser can see *why* a given accuracy materialised.

Retrieval diagnostics:
  - retrieval_status: per-question {both, only_A, only_B, neither} —
    load-bearing 2-hop signal; a question is retrieval-complete only when
    both gold spans land in the retrieved chunks.
  - chunk_precision: fraction of retrieved chunks overlapping either span
    (kept as a continuous diagnostic; no longer drives the composite).
  - source_fact_rank: 1-indexed rank of the first overlapping chunk.

Composite score = answer_accuracy.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from typing import Literal

from pydantic import BaseModel
from tqdm import tqdm

from agentic_autorag.benchmark_eval.scoring import best_em, best_f1, llm_judge
from agentic_autorag.config.models import OpenEndedQuestion
from agentic_autorag.engine.pipeline import RAGPipeline
from agentic_autorag.examiner._errors import format_llm_error, is_permanent_llm_error
from agentic_autorag.examiner.prompts import NAIVE_RAG_PROMPT

logger = logging.getLogger(__name__)
run_logger = logging.getLogger("agentic_autorag.run")

_ERROR_SENTINEL = "QUESTION_EVALUATION_ERROR"
_PERMANENT_ERROR_SENTINEL = "QUESTION_PERMANENT_ERROR"
_RETRY_COOLDOWNS = (10, 30, 60)
_SLOW_THRESHOLD_S = 40.0


class QuestionResult(BaseModel):
    """Result of evaluating a single open-ended question."""

    question_id: str
    correct: bool
    selected_answer: str  # the model's free-text answer (or "INVALID")
    correct_answer: str  # canonical gold answer for human-readable diagnostics
    retrieved_context: str
    generated_response: str
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    model_s: float = 0.0  # actual retrieval model compute (excludes queue wait)
    source_fact_rank: int = 0
    chunk_precision: float = 0.0
    retrieved_doc_ids: list[str] = []
    em: float = 0.0
    f1: float = 0.0
    judge: int | None = None  # 1, 0, or None when judge wasn't called / failed
    # Per-span retrieval diagnostic. ``both`` = both gold spans were
    # retrieved (sufficient context for a 2-hop answer); ``only_A`` /
    # ``only_B`` = partial; ``neither`` = retrieval miss.
    retrieval_status: Literal["both", "only_A", "only_B", "neither"] = "neither"
    # The model produced a refusal phrase rather than an attempted answer
    # (e.g. "Cannot answer based on provided context"). Detected via
    # ``_detect_refusal`` on the raw generated response.
    refused: bool = False

    @property
    def context_sufficient(self) -> bool:
        return self.retrieval_status == "both"


class ExamResult(BaseModel):
    """Aggregated result of evaluating a full open-ended exam.

    ``score`` equals ``answer_accuracy``. Retrieval signals are diagnostic
    only and feed the optimizer's diagnoser, not the composite objective.

    Verdict-path breakdown (sums to n_valid):
      - n_em_correct + n_judge_correct = n_correct
      - n_judge_rejected (judge said NO)
      - n_judge_failed (judge errored)
      - n_no_answer (RAG produced empty pred)
      = n_valid

    Open-ended failure-mode counters (sum to n_valid for retrieval; refused
    and correct-given-complete are independent slices):
      - n_retrieval_complete + n_retrieval_partial_a_only
        + n_retrieval_partial_b_only + n_retrieval_miss = n_valid
      - n_refused: questions whose generated response was a refusal phrase
      - n_correct_given_complete_retrieval: correct AND retrieval_status=both
    """

    score: float
    n_correct: int
    n_total: int
    n_valid: int = 0
    question_results: list[QuestionResult]
    answer_accuracy: float = 0.0
    mean_retrieval_quality: float = 0.0
    n_em_correct: int = 0
    n_judge_correct: int = 0
    n_judge_rejected: int = 0
    n_judge_failed: int = 0
    n_no_answer: int = 0
    n_judge_calls: int = 0
    mean_em: float = 0.0
    mean_f1: float = 0.0
    n_retrieval_complete: int = 0
    n_retrieval_partial_a_only: int = 0
    n_retrieval_partial_b_only: int = 0
    n_retrieval_miss: int = 0
    n_refused: int = 0
    n_correct_given_complete_retrieval: int = 0

    def failed_questions(self) -> list[QuestionResult]:
        return [qr for qr in self.question_results if not qr.correct]


class OpenEndedEvaluator:
    """Evaluates a RAG pipeline against an open-ended 2-hop exam."""

    def __init__(
        self,
        concurrency: int = 10,
        retrieval_quality_alpha: float = 0.7,
        judge_model: str | None = None,
        *,
        documents: dict[str, str] | None = None,
        chunk_relevance_min_overlap_chars: int = 50,
        chunk_relevance_ngram_size: int = 5,
        chunk_relevance_overlap_threshold: float = 0.5,
        chunk_relevance_min_run: int = 5,
        duplicate_alias_map: dict[str, str] | None = None,
        debug_eval_samples: int = 0,
    ) -> None:
        self.concurrency = concurrency
        self.alpha = retrieval_quality_alpha
        self.judge_model = judge_model
        self.documents: dict[str, str] = documents or {}
        self._graph_offset_cache: dict[str, tuple[str, int, int] | None] = {}
        self.min_overlap_chars = chunk_relevance_min_overlap_chars
        self.ngram_size = chunk_relevance_ngram_size
        self.coverage_threshold = chunk_relevance_overlap_threshold
        self.min_run = chunk_relevance_min_run
        # alias_doc_id → canonical_doc_id. Used to canonicalize retrieved
        # doc_ids when scoring chunk relevance so duplicates count as their
        # canonical. Set by the orchestrator after near-duplicate detection.
        self.duplicate_alias_map: dict[str, str] = duplicate_alias_map or {}
        self.debug_eval_samples = debug_eval_samples

    async def evaluate(
        self,
        pipeline: RAGPipeline,
        exam: list[OpenEndedQuestion],
    ) -> ExamResult:
        if not exam:
            return ExamResult(score=0.0, n_correct=0, n_total=0, question_results=[])

        results_by_id: dict[str, QuestionResult] = {}
        qnum_map = {q.id: i for i, q in enumerate(exam, start=1)}

        await self._run_pass(results_by_id, pipeline, exam, qnum_map, desc="Evaluating questions")

        n_permanent = sum(1 for q in exam if results_by_id[q.id].generated_response == _PERMANENT_ERROR_SENTINEL)
        if n_permanent:
            tqdm.write(f"\n  {n_permanent} question(s) hit permanent errors — skipping retries")

        for retry_round, cooldown in enumerate(_RETRY_COOLDOWNS, start=1):
            retryable = [q for q in exam if results_by_id[q.id].generated_response == _ERROR_SENTINEL]
            if not retryable:
                break
            tqdm.write(
                f"\n  {len(retryable)} question(s) failed (transient) — "
                f"retrying after {cooldown}s cooldown (round {retry_round}/{len(_RETRY_COOLDOWNS)})"
            )
            await asyncio.sleep(cooldown)
            await self._run_pass(results_by_id, pipeline, retryable, qnum_map, desc=f"Retry {retry_round}")

        results = [results_by_id[q.id] for q in exam]
        n_total = len(results)
        valid_results = [r for r in results if r.generated_response not in (_ERROR_SENTINEL, _PERMANENT_ERROR_SENTINEL)]
        n_valid = len(valid_results)

        # Verdict-path breakdown. Sums equal n_valid:
        #   n_em_correct + n_judge_correct + n_judge_rejected
        #   + n_judge_failed + n_no_answer = n_valid
        n_em_correct = 0
        n_judge_correct = 0
        n_judge_rejected = 0
        n_judge_failed = 0
        n_no_answer = 0
        for r in valid_results:
            if r.em > 0:
                n_em_correct += 1
                continue
            if not r.selected_answer or r.selected_answer == "INVALID":
                n_no_answer += 1
            elif r.judge == 1:
                n_judge_correct += 1
            elif r.judge == 0:
                n_judge_rejected += 1
            else:
                # Judge attempted but returned None (parse error / API failure).
                n_judge_failed += 1
        n_judge_calls = n_judge_correct + n_judge_rejected + n_judge_failed
        n_correct = n_em_correct + n_judge_correct
        accuracy = n_correct / n_valid if n_valid else 0.0
        mean_em = sum(r.em for r in valid_results) / n_valid if n_valid else 0.0
        mean_f1 = sum(r.f1 for r in valid_results) / n_valid if n_valid else 0.0
        mean_rq = sum(r.chunk_precision for r in valid_results) / n_valid if n_valid else 0.0
        score = accuracy

        n_retrieval_complete = sum(1 for r in valid_results if r.retrieval_status == "both")
        n_retrieval_partial_a_only = sum(1 for r in valid_results if r.retrieval_status == "only_A")
        n_retrieval_partial_b_only = sum(1 for r in valid_results if r.retrieval_status == "only_B")
        n_retrieval_miss = sum(1 for r in valid_results if r.retrieval_status == "neither")
        n_refused = sum(1 for r in valid_results if r.refused)
        n_correct_given_complete_retrieval = sum(1 for r in valid_results if r.correct and r.retrieval_status == "both")

        run_logger.info(
            "Eval: score=%.3f (=accuracy) | accuracy=%.3f (%d/%d) | "
            "retrieval: complete=%.2f only_A=%.2f only_B=%.2f miss=%.2f | "
            "refusal_rate=%.2f | mean_chunk_precision=%.3f (diagnostic)",
            score,
            accuracy,
            n_correct,
            n_valid,
            n_retrieval_complete / n_valid if n_valid else 0.0,
            n_retrieval_partial_a_only / n_valid if n_valid else 0.0,
            n_retrieval_partial_b_only / n_valid if n_valid else 0.0,
            n_retrieval_miss / n_valid if n_valid else 0.0,
            n_refused / n_valid if n_valid else 0.0,
            mean_rq,
        )
        run_logger.info(
            "  Verdicts: %d EM, %d judge=yes, %d judge=no, %d judge_failed, %d no_answer "
            "(judge calls: %d) | mean F1=%.3f (diagnostic only)",
            n_em_correct,
            n_judge_correct,
            n_judge_rejected,
            n_judge_failed,
            n_no_answer,
            n_judge_calls,
            mean_f1,
        )
        if n_judge_failed > 0:
            run_logger.warning(
                "  %d judge call(s) failed — accuracy may be UNDER-reported. "
                "Check WARNING logs from agentic_autorag.benchmark_eval.scoring for the cause.",
                n_judge_failed,
            )
        if n_valid < n_total:
            for r in results:
                cls = _sentinel_class(r.generated_response)
                if cls is not None:
                    run_logger.info(
                        "  Excluded sentinel: q_id=%s class=%s",
                        r.question_id,
                        cls,
                    )
            run_logger.info(
                "  Excluded %d error-sentinel question(s) (n_valid=%d of %d)",
                n_total - n_valid,
                n_valid,
                n_total,
            )
        if self.debug_eval_samples > 0 and valid_results:
            self._log_eval_samples(exam, valid_results)

        return ExamResult(
            score=score,
            n_correct=n_correct,
            n_total=n_total,
            n_valid=n_valid,
            question_results=results,
            answer_accuracy=accuracy,
            mean_retrieval_quality=mean_rq,
            n_em_correct=n_em_correct,
            n_judge_correct=n_judge_correct,
            n_judge_rejected=n_judge_rejected,
            n_judge_failed=n_judge_failed,
            n_no_answer=n_no_answer,
            n_judge_calls=n_judge_calls,
            mean_em=mean_em,
            mean_f1=mean_f1,
            n_retrieval_complete=n_retrieval_complete,
            n_retrieval_partial_a_only=n_retrieval_partial_a_only,
            n_retrieval_partial_b_only=n_retrieval_partial_b_only,
            n_retrieval_miss=n_retrieval_miss,
            n_refused=n_refused,
            n_correct_given_complete_retrieval=n_correct_given_complete_retrieval,
        )

    def _log_eval_samples(
        self,
        exam: list[OpenEndedQuestion],
        valid_results: list[QuestionResult],
    ) -> None:
        """Log a deterministic sample of (question, retrieved context, RAG answer) triples.

        Sampling: deterministic — sorted by question_id and stride-picked, so
        the same exam yields the same samples across trials, making A/B
        comparison across configs straightforward. Mixes correct and
        incorrect when both are available so the diagnostic isn't biased
        toward only one outcome.
        """
        n = min(self.debug_eval_samples, len(valid_results))
        question_by_id = {q.id: q for q in exam}
        # Half from incorrect (most informative), half from correct.
        incorrect = sorted([r for r in valid_results if not r.correct], key=lambda r: r.question_id)
        correct = sorted([r for r in valid_results if r.correct], key=lambda r: r.question_id)
        n_wrong = min(n // 2, len(incorrect))
        n_right = min(n - n_wrong, len(correct))
        # Top up if one bucket is empty.
        if n_wrong < n // 2 and len(correct) > n_right:
            n_right = min(n - n_wrong, len(correct))
        if n_right < n - n_wrong and len(incorrect) > n_wrong:
            n_wrong = min(n - n_right, len(incorrect))
        sampled = incorrect[:n_wrong] + correct[:n_right]
        # Deterministic shuffle for visual variety without randomness.
        rng = random.Random(0)
        rng.shuffle(sampled)
        run_logger.debug("=== Eval sample (%d question(s)) ===", len(sampled))
        for r in sampled:
            q = question_by_id.get(r.question_id)
            if q is None:
                continue
            ctx = r.retrieved_context or ""
            ctx_preview = ctx if len(ctx) <= 800 else ctx[:800] + " […truncated]"
            verdict = _verdict_label(r)
            extras = [f"em={r.em:.0f}", f"f1={r.f1:.2f}", f"rq={r.chunk_precision:.2f}"]
            if r.judge is not None:
                extras.append(f"judge={r.judge}")
            if not r.correct:
                extras.append(f"retr={r.retrieval_status}")
            run_logger.debug(
                "[%s] %s\n  gold:    %s\n  pred:    %s\n  verdict: %s | %s\n  context: %s",
                r.question_id,
                q.question,
                r.correct_answer,
                r.selected_answer,
                verdict,
                " ".join(extras),
                ctx_preview.replace("\n", " "),
            )
        run_logger.debug("=== End eval sample ===")

    async def _run_pass(
        self,
        results_by_id: dict[str, QuestionResult],
        pipeline: RAGPipeline,
        questions: list[OpenEndedQuestion],
        qnum_map: dict[str, int],
        desc: str,
    ) -> None:
        sem = asyncio.Semaphore(self.concurrency)
        with tqdm(total=len(questions), desc=desc, unit="q") as pbar:

            async def _bounded(q: OpenEndedQuestion) -> None:
                async with sem:
                    t0 = time.monotonic()
                    qr = await self._evaluate_single(pipeline, q)
                    elapsed = time.monotonic() - t0
                qnum = qnum_map.get(q.id, 0)
                label = f"Q{qnum:02d}"
                queue_s = max(qr.retrieval_s - qr.model_s, 0.0)
                timing_detail = f"(retr={qr.model_s:.1f}s llm={qr.generation_s:.1f}s queue={queue_s:.1f}s)"
                if qr.generated_response in (_ERROR_SENTINEL, _PERMANENT_ERROR_SENTINEL):
                    pass
                elif not qr.correct:
                    tqdm.write(
                        f"  MISS {label}"
                        f" | pred={_truncate(qr.selected_answer, 40)!r}"
                        f" gold={_truncate(qr.correct_answer, 40)!r}"
                        f" em={qr.em:.0f} f1={qr.f1:.2f}"
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
        q: OpenEndedQuestion,
    ) -> QuestionResult:
        llm_timeout = pipeline.config.llm_timeout_s
        question_timeout = llm_timeout + 30 if llm_timeout is not None else None
        try:
            async with asyncio.timeout(question_timeout):
                t0 = time.monotonic()
                retrieval_result = await pipeline.retrieve(q.question)
                retrieval_s = time.monotonic() - t0

                from agentic_autorag.examiner.exam_validator import chunk_contains_source_fact

                source_fact_rank = 0
                n_relevant = 0
                found_span_a = False
                found_span_b = False

                def _check_span(doc, span_idx: int) -> bool:
                    return chunk_contains_source_fact(
                        q,
                        doc,
                        docs=self.documents,
                        offset_cache=self._graph_offset_cache,
                        min_overlap_chars=self.min_overlap_chars,
                        ngram_size=self.ngram_size,
                        coverage_threshold=self.coverage_threshold,
                        min_run=self.min_run,
                        duplicate_alias_map=self.duplicate_alias_map,
                        span_indices=(span_idx,),
                    )

                for rank, doc in enumerate(retrieval_result.documents, start=1):
                    hit_a = _check_span(doc, 0)
                    hit_b = _check_span(doc, 1)
                    found_span_a = found_span_a or hit_a
                    found_span_b = found_span_b or hit_b
                    if hit_a or hit_b:
                        n_relevant += 1
                        if source_fact_rank == 0:
                            source_fact_rank = rank
                chunk_precision = n_relevant / len(retrieval_result.documents) if retrieval_result.documents else 0.0
                retrieval_status = _retrieval_status_label(found_span_a, found_span_b)

                context = "\n".join(doc.text for doc in retrieval_result.documents)
                prompt = NAIVE_RAG_PROMPT.format(context=context, question=q.question)

                t0 = time.monotonic()
                raw_answer = await pipeline.generate(prompt)
                generation_s = time.monotonic() - t0

            pred = (raw_answer or "").strip()
            em = best_em(pred, q.gold_answers)
            f1 = best_f1(pred, q.gold_answers)  # diagnostic only — not used for correctness
            judge_score: int | None = None
            # Correctness stack: EM is free and exact; otherwise the judge
            # is the canonical arbiter. F1 is no longer a correctness gate
            # (its 0.5 threshold was a magic number with no semantic meaning);
            # it stays on QuestionResult as a diagnostic.
            if em > 0:
                correct = True
            elif self.judge_model is not None and pred:
                judge_score = await llm_judge(self.judge_model, q.question, pred, q.gold_answers)
                correct = judge_score == 1
            else:
                correct = False

            retrieved_doc_ids = [str(doc.metadata.get("doc_id", "")) for doc in retrieval_result.documents]

            return QuestionResult(
                question_id=q.id,
                correct=correct,
                selected_answer=pred or "INVALID",
                correct_answer=q.canonical_answer,
                retrieved_context=context,
                generated_response=raw_answer or "",
                retrieval_s=retrieval_s,
                model_s=retrieval_result.timing.model_s,
                generation_s=generation_s,
                source_fact_rank=source_fact_rank,
                chunk_precision=chunk_precision,
                retrieved_doc_ids=retrieved_doc_ids,
                em=em,
                f1=f1,
                judge=judge_score,
                retrieval_status=retrieval_status,
                refused=_detect_refusal(raw_answer),
            )
        except TimeoutError:
            timeout_msg = f"exceeded {question_timeout:.0f}s" if question_timeout is not None else "timed out"
            tqdm.write(f"  TIMEOUT {q.id} | {timeout_msg}")
            return QuestionResult(
                question_id=q.id,
                correct=False,
                selected_answer="INVALID",
                correct_answer=q.canonical_answer,
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
                correct_answer=q.canonical_answer,
                retrieved_context="",
                generated_response=sentinel,
            )


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _verdict_label(qr: QuestionResult) -> str:
    """Identify which scoring path declared this question correct/incorrect."""
    if qr.em > 0:
        return "CORRECT (EM)"
    if qr.judge == 1:
        return "CORRECT (judge=YES)"
    if qr.judge == 0:
        return "INCORRECT (judge=NO)"
    if not qr.selected_answer or qr.selected_answer == "INVALID":
        return "INCORRECT (no answer)"
    return "INCORRECT (judge failed)"


_REFUSAL_PATTERNS = re.compile(
    r"(cannot answer|can't answer|don't have (?:enough|sufficient)|"
    r"do not have (?:enough|sufficient)|no (?:information|context|relevant)|"
    r"context (?:does not|doesn't)|insufficient (?:information|context)|"
    r"unable to determine|not (?:enough|sufficient) (?:information|context)|"
    r"i don't know|there is no (?:information|context))",
    re.IGNORECASE,
)


def _detect_refusal(text: str | None) -> bool:
    """True when the model declined to answer rather than attempted one."""
    if not text:
        return False
    return bool(_REFUSAL_PATTERNS.search(text))


def _retrieval_status_label(found_a: bool, found_b: bool) -> Literal["both", "only_A", "only_B", "neither"]:
    if found_a and found_b:
        return "both"
    if found_a:
        return "only_A"
    if found_b:
        return "only_B"
    return "neither"


def _sentinel_class(generated_response: str) -> Literal["transient", "permanent"] | None:
    """Classify a sentinel-marked QuestionResult by error class for diagnostics."""
    if generated_response == _PERMANENT_ERROR_SENTINEL:
        return "permanent"
    if generated_response == _ERROR_SENTINEL:
        return "transient"
    return None


# Legacy alias kept so orchestrator and tests that imported MCQEvaluator
# continue to work without a global rename. The class is the open-ended
# evaluator; MCQ is just historical naming.
MCQEvaluator = OpenEndedEvaluator
