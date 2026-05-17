"""Open-ended evaluator — runs an exam against a RAG pipeline and scores answers.

Free-text scoring stack (cheapest first):
  1. Normalised EM against canonical_answer + variants  (free, deterministic)
  2. Token F1 against canonical_answer + variants        (free, deterministic)
  3. LLM judge fallback (CRAG 3-way)                    (only when EM=0)

A question is **correct** iff EM>0 or judge=1. The ``ExamResult`` reports
verdict-path counts and per-trial open-ended failure-mode counters
(retrieval_complete / partial / miss / refused / correct-given-complete)
so the diagnoser can see *why* a given accuracy materialised.

Retrieval diagnostics:
  - retrieved_spans / n_spans: per-question integer counts. A question is
    retrieval-complete when ``retrieved_spans == n_spans``; partial when
    ``0 < retrieved_spans < n_spans``; miss when ``retrieved_spans == 0``.
    Generalises across single-hop (n_spans=1) and multi-hop questions.
  - chunk_precision: fraction of retrieved chunks overlapping any span
    (kept as a continuous diagnostic; no longer drives the composite).
  - source_fact_rank: 1-indexed rank of the first overlapping chunk.

Composite score = answer_accuracy.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Literal

from pydantic import BaseModel
from tqdm import tqdm

from agentic_autorag.benchmark_eval.scoring import best_em, best_f1, llm_judge
from agentic_autorag.config.models import OpenEndedQuestion
from agentic_autorag.engine.pipeline import RAGPipeline
from agentic_autorag.examiner._errors import (
    ERROR_SENTINELS,
    PERMANENT_ERROR_SENTINEL,
    RETRY_COOLDOWNS_S,
    TRANSIENT_ERROR_SENTINEL,
    format_llm_error,
    is_permanent_llm_error,
)
from agentic_autorag.examiner.prompts import NAIVE_RAG_PROMPT, answer_format_hint

logger = logging.getLogger(__name__)
run_logger = logging.getLogger("agentic_autorag.run")

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
    # Per-chunk text aligned with ``retrieved_doc_ids``. The Diagnoser uses
    # this for failure-mode-adaptive chunk rendering (windows around gold
    # spans). The LLM-facing ``retrieved_context`` is the same chunks joined
    # by a single newline.
    retrieved_chunks: list[str] = []
    em: float = 0.0
    f1: float = 0.0
    # LLM cost in USD for this question's generation calls (synthesis + any
    # query-expansion call). Excludes embedder, reranker (local), and judge.
    # 0.0 when LiteLLM has no pricing for the model (self-hosted/local).
    llm_cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Judge verdict: 1=YES, 0=NO, -1=NO_ANSWER, None=not called / failed.
    judge: int | None = None
    # Per-span retrieval diagnostic, generalised across hop counts:
    # ``retrieved_spans`` is how many gold spans landed in the retrieved
    # chunks; ``n_spans`` is the total number of gold spans the question
    # has. A question is retrieval-complete when ``retrieved_spans ==
    # n_spans``; partial when ``0 < retrieved_spans < n_spans``; miss when
    # ``retrieved_spans == 0``.
    retrieved_spans: int = 0
    n_spans: int = 0
    # Per-chunk × per-span match table. ``chunk_satisfies_spans[i]`` is the
    # sorted list of gold-span indices that the evaluator's matcher considered
    # satisfied by retrieved chunk ``i`` (rank ``i+1``). Aligned with
    # ``retrieved_doc_ids`` and ``retrieved_chunks``. The matcher uses
    # char-range overlap, unicode-folded substring search, AND n-gram coverage
    # — so an entry here means "this chunk has the information" even when the
    # text doesn't contain the gold span verbatim. The Diagnoser's renderer
    # uses this to surface ALL chunks the evaluator credited with each span,
    # not just the ones where the span is verbatim-locatable.
    chunk_satisfies_spans: list[list[int]] = []
    # The model didn't attempt the answer — either it produced an empty
    # response or the judge classified its output as NO_ANSWER. Replaces
    # the previous regex-based refusal detector so phrasing differences
    # across LLMs don't change the count.
    refused: bool = False

    @property
    def context_sufficient(self) -> bool:
        return self.n_spans > 0 and self.retrieved_spans == self.n_spans

    @property
    def retrieval_status(self) -> str:
        """Human-readable summary: ``complete`` / ``partial M/N`` / ``none``."""
        if self.n_spans == 0:
            return "none"
        if self.retrieved_spans == 0:
            return "none"
        if self.retrieved_spans == self.n_spans:
            return "complete"
        return f"partial {self.retrieved_spans}/{self.n_spans}"


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
      - n_retrieval_complete + n_retrieval_partial + n_retrieval_miss = n_valid
      - n_refused: questions whose generated response was a refusal phrase
      - n_correct_given_complete_retrieval: correct AND all gold spans retrieved
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
    n_judge_no_answer: int = 0
    n_judge_failed: int = 0
    n_no_answer: int = 0
    n_judge_calls: int = 0
    mean_em: float = 0.0
    mean_f1: float = 0.0
    n_retrieval_complete: int = 0
    n_retrieval_partial: int = 0
    n_retrieval_miss: int = 0
    n_refused: int = 0
    n_correct_given_complete_retrieval: int = 0
    # LLM cost roll-ups, deployment-side only (synthesis + query expansion).
    # ``mean_llm_cost_per_query_usd`` averages over ``n_valid``; ``total_llm_cost_usd``
    # sums all valid questions. Excludes judge LLM cost (eval-only).
    mean_llm_cost_per_query_usd: float = 0.0
    total_llm_cost_usd: float = 0.0
    mean_prompt_tokens: float = 0.0
    mean_completion_tokens: float = 0.0

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
        quiet_per_question: bool = False,
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
        # Suppresses the per-question MISS/SLOW status lines. Error notices
        # (ERROR, TIMEOUT, permanent-error summaries) are unaffected.
        self.quiet_per_question = quiet_per_question

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

        n_permanent = sum(1 for q in exam if results_by_id[q.id].generated_response == PERMANENT_ERROR_SENTINEL)
        if n_permanent:
            tqdm.write(f"\n  {n_permanent} question(s) hit permanent errors — skipping retries")

        for retry_round, cooldown in enumerate(RETRY_COOLDOWNS_S, start=1):
            retryable = [q for q in exam if results_by_id[q.id].generated_response == TRANSIENT_ERROR_SENTINEL]
            if not retryable:
                break
            tqdm.write(
                f"\n  {len(retryable)} question(s) failed (transient) — "
                f"retrying after {cooldown}s cooldown (round {retry_round}/{len(RETRY_COOLDOWNS_S)})"
            )
            await asyncio.sleep(cooldown)
            await self._run_pass(results_by_id, pipeline, retryable, qnum_map, desc=f"Retry {retry_round}")

        results = [results_by_id[q.id] for q in exam]
        n_total = len(results)
        valid_results = [r for r in results if r.generated_response not in ERROR_SENTINELS]
        n_valid = len(valid_results)

        # Verdict-path breakdown. Sums equal n_valid:
        #   n_em_correct + n_judge_correct + n_judge_rejected
        #   + n_judge_no_answer + n_judge_failed + n_no_answer = n_valid
        n_em_correct = 0
        n_judge_correct = 0
        n_judge_rejected = 0
        n_judge_no_answer = 0
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
            elif r.judge == -1:
                n_judge_no_answer += 1
            else:
                # Judge attempted but returned None (parse error / API failure).
                n_judge_failed += 1
        n_judge_calls = n_judge_correct + n_judge_rejected + n_judge_no_answer + n_judge_failed
        n_correct = n_em_correct + n_judge_correct
        accuracy = n_correct / n_valid if n_valid else 0.0
        mean_em = sum(r.em for r in valid_results) / n_valid if n_valid else 0.0
        mean_f1 = sum(r.f1 for r in valid_results) / n_valid if n_valid else 0.0
        mean_rq = sum(r.chunk_precision for r in valid_results) / n_valid if n_valid else 0.0
        score = accuracy

        n_retrieval_complete = sum(1 for r in valid_results if r.context_sufficient)
        n_retrieval_miss = sum(1 for r in valid_results if r.retrieved_spans == 0)
        n_retrieval_partial = sum(1 for r in valid_results if 0 < r.retrieved_spans < r.n_spans)
        n_refused = sum(1 for r in valid_results if r.refused)
        n_correct_given_complete_retrieval = sum(1 for r in valid_results if r.correct and r.context_sufficient)

        total_llm_cost_usd = sum(r.llm_cost_usd for r in valid_results)
        mean_llm_cost_per_query_usd = total_llm_cost_usd / n_valid if n_valid else 0.0
        mean_prompt_tokens = sum(r.prompt_tokens for r in valid_results) / n_valid if n_valid else 0.0
        mean_completion_tokens = sum(r.completion_tokens for r in valid_results) / n_valid if n_valid else 0.0

        run_logger.info("")
        run_logger.info(
            "Eval: score=%.3f (=accuracy) | accuracy=%.3f (%d/%d) | "
            "retrieval: complete=%.2f partial=%.2f miss=%.2f | "
            "refusal_rate=%.2f | mean_chunk_precision=%.3f (diagnostic) | "
            "cost: $%.4f/q (total $%.3f, prompt~%.0f tok, completion~%.0f tok)",
            score,
            accuracy,
            n_correct,
            n_valid,
            n_retrieval_complete / n_valid if n_valid else 0.0,
            n_retrieval_partial / n_valid if n_valid else 0.0,
            n_retrieval_miss / n_valid if n_valid else 0.0,
            n_refused / n_valid if n_valid else 0.0,
            mean_rq,
            mean_llm_cost_per_query_usd,
            total_llm_cost_usd,
            mean_prompt_tokens,
            mean_completion_tokens,
        )
        run_logger.info(
            "  Verdicts: %d EM, %d judge=yes, %d judge=no, %d judge=no_answer, "
            "%d judge_failed, %d empty_pred (judge calls: %d) | mean F1=%.3f (diagnostic only)",
            n_em_correct,
            n_judge_correct,
            n_judge_rejected,
            n_judge_no_answer,
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
            n_judge_no_answer=n_judge_no_answer,
            n_judge_failed=n_judge_failed,
            n_no_answer=n_no_answer,
            n_judge_calls=n_judge_calls,
            mean_em=mean_em,
            mean_f1=mean_f1,
            n_retrieval_complete=n_retrieval_complete,
            n_retrieval_partial=n_retrieval_partial,
            n_retrieval_miss=n_retrieval_miss,
            n_refused=n_refused,
            n_correct_given_complete_retrieval=n_correct_given_complete_retrieval,
            mean_llm_cost_per_query_usd=mean_llm_cost_per_query_usd,
            total_llm_cost_usd=total_llm_cost_usd,
            mean_prompt_tokens=mean_prompt_tokens,
            mean_completion_tokens=mean_completion_tokens,
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
                if qr.generated_response in ERROR_SENTINELS:
                    pass
                elif self.quiet_per_question:
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
                n_spans_total = q.num_hops
                span_found = [False] * n_spans_total
                chunk_satisfies_spans: list[list[int]] = []

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
                    matched_spans: list[int] = []
                    for span_idx in range(n_spans_total):
                        if _check_span(doc, span_idx):
                            span_found[span_idx] = True
                            matched_spans.append(span_idx)
                    chunk_satisfies_spans.append(matched_spans)
                    if matched_spans:
                        n_relevant += 1
                        if source_fact_rank == 0:
                            source_fact_rank = rank
                chunk_precision = n_relevant / len(retrieval_result.documents) if retrieval_result.documents else 0.0
                retrieved_spans_count = sum(1 for f in span_found if f)

                context, prep_cost = await pipeline.prepare_context(q.question, retrieval_result)
                prompt = NAIVE_RAG_PROMPT.format(
                    context=context,
                    question=q.question,
                    answer_format_hint=answer_format_hint(q.reasoning_type, q.formula_kind),
                )

                t0 = time.monotonic()
                raw_answer, gen_cost = await pipeline.generate(prompt)
                generation_s = time.monotonic() - t0

            expansion_cost = retrieval_result.expansion_cost
            llm_cost_usd = (
                float(expansion_cost.get("usd", 0.0))
                + float(prep_cost.get("usd", 0.0))
                + float(gen_cost.get("usd", 0.0))
            )
            prompt_tokens_total = (
                int(expansion_cost.get("prompt_tokens", 0))
                + int(prep_cost.get("prompt_tokens", 0))
                + int(gen_cost.get("prompt_tokens", 0))
            )
            completion_tokens_total = (
                int(expansion_cost.get("completion_tokens", 0))
                + int(prep_cost.get("completion_tokens", 0))
                + int(gen_cost.get("completion_tokens", 0))
            )

            pred = (raw_answer or "").strip()
            em = best_em(pred, q.gold_answers)
            f1 = best_f1(pred, q.gold_answers)  # diagnostic only — not used for correctness
            judge_score: int | None = None
            # Correctness stack: EM is free and exact; otherwise the judge
            # is the canonical arbiter and also tells us whether the model
            # actually attempted an answer (NO_ANSWER) vs got it wrong (NO).
            # Empty predictions skip the judge — the answer is "no answer"
            # by definition and there's nothing for the judge to grade.
            refused = False
            if em > 0:
                correct = True
            elif not pred:
                correct = False
                refused = True
            elif self.judge_model is not None:
                judge_score = await llm_judge(self.judge_model, q.question, pred, q.gold_answers)
                correct = judge_score == 1
                refused = judge_score == -1
            else:
                correct = False

            retrieved_doc_ids = [str(doc.metadata.get("doc_id", "")) for doc in retrieval_result.documents]
            retrieved_chunks = [doc.text for doc in retrieval_result.documents]

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
                retrieved_chunks=retrieved_chunks,
                chunk_satisfies_spans=chunk_satisfies_spans,
                em=em,
                f1=f1,
                llm_cost_usd=llm_cost_usd,
                prompt_tokens=prompt_tokens_total,
                completion_tokens=completion_tokens_total,
                judge=judge_score,
                retrieved_spans=retrieved_spans_count,
                n_spans=n_spans_total,
                refused=refused,
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
                generated_response=TRANSIENT_ERROR_SENTINEL,
            )
        except Exception as exc:
            error_summary = format_llm_error(exc)
            permanent = is_permanent_llm_error(exc)
            sentinel = PERMANENT_ERROR_SENTINEL if permanent else TRANSIENT_ERROR_SENTINEL
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
    if qr.judge == -1:
        return "INCORRECT (judge=NO_ANSWER)"
    if not qr.selected_answer or qr.selected_answer == "INVALID":
        return "INCORRECT (empty pred)"
    return "INCORRECT (judge failed)"


def _sentinel_class(generated_response: str) -> Literal["transient", "permanent"] | None:
    """Classify a sentinel-marked QuestionResult by error class for diagnostics."""
    if generated_response == PERMANENT_ERROR_SENTINEL:
        return "permanent"
    if generated_response == TRANSIENT_ERROR_SENTINEL:
        return "transient"
    return None
