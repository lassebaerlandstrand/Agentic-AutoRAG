"""Runs a built ``RAGPipeline`` over held-out free-form QA and scores it.

Mirrors the concurrency + retry semantics of
``examiner.evaluator.OpenEndedEvaluator``: a semaphore-bounded
``asyncio.gather`` pass with escalating-cooldown retries for transient errors,
permanent errors classified via ``examiner._errors`` and excluded from all
denominators.
"""

from __future__ import annotations

import asyncio
import logging
import time

from tqdm import tqdm

from agentic_autorag.benchmark_eval.models import QAResult
from agentic_autorag.benchmark_eval.prompts import FREE_FORM_ANSWER_PROMPT
from agentic_autorag.benchmark_eval.scoring import (
    best_em,
    best_f1,
    llm_judge,
    retrieval_metrics,
)
from agentic_autorag.benchmarks.schema import BenchmarkQAPair
from agentic_autorag.engine.pipeline import RAGPipeline
from agentic_autorag.examiner._errors import (
    ERROR_SENTINELS,
    PERMANENT_ERROR_SENTINEL,
    RETRY_COOLDOWNS_S,
    TRANSIENT_ERROR_SENTINEL,
    format_llm_error,
    is_permanent_llm_error,
)

logger = logging.getLogger(__name__)


class FreeFormEvaluator:
    """Score a RAG pipeline against free-form benchmark QA pairs."""

    def __init__(
        self,
        concurrency: int = 10,
        judge_model: str | None = None,
        judge_timeout_s: float = 30.0,
    ) -> None:
        self.concurrency = concurrency
        self.judge_model = judge_model
        self.judge_timeout_s = judge_timeout_s

    async def evaluate(
        self,
        pipeline: RAGPipeline,
        qa_pairs: list[BenchmarkQAPair],
    ) -> list[QAResult]:
        """Run every QA pair through ``pipeline`` and return per-question results."""
        if not qa_pairs:
            return []

        results: dict[str, QAResult] = {}
        await self._run_pass(results, pipeline, qa_pairs, desc="Evaluating QA")

        n_permanent = sum(1 for qa in qa_pairs if results[qa.id].error == PERMANENT_ERROR_SENTINEL)
        if n_permanent:
            tqdm.write(f"\n  {n_permanent} question(s) hit permanent errors — skipping retries")

        for retry_round, cooldown in enumerate(RETRY_COOLDOWNS_S, start=1):
            retryable = [qa for qa in qa_pairs if results[qa.id].error == TRANSIENT_ERROR_SENTINEL]
            if not retryable:
                break
            tqdm.write(
                f"\n  {len(retryable)} question(s) failed (transient) — "
                f"retrying after {cooldown}s cooldown (round {retry_round}/{len(RETRY_COOLDOWNS_S)})"
            )
            await asyncio.sleep(cooldown)
            await self._run_pass(results, pipeline, retryable, desc=f"Retry round {retry_round}")

        return [results[qa.id] for qa in qa_pairs]

    async def _run_pass(
        self,
        results: dict[str, QAResult],
        pipeline: RAGPipeline,
        qa_pairs: list[BenchmarkQAPair],
        desc: str,
    ) -> None:
        sem = asyncio.Semaphore(self.concurrency)
        # smoothing=0.05 biases the rate estimate toward the whole-run average
        # instead of the last few items — per-question latency here is very
        # bursty (fast factoid vs slow multi-paragraph LLM calls) so tqdm's
        # default smoothing=0.3 produces wildly fluctuating ETAs.
        with tqdm(total=len(qa_pairs), desc=desc, unit="q", smoothing=0.05) as pbar:

            async def _bounded(qa: BenchmarkQAPair) -> None:
                async with sem:
                    qr = await self._evaluate_single(pipeline, qa)
                results[qa.id] = qr
                pbar.update(1)

            await asyncio.gather(*[_bounded(qa) for qa in qa_pairs])

    async def _evaluate_single(
        self,
        pipeline: RAGPipeline,
        qa: BenchmarkQAPair,
    ) -> QAResult:
        llm_timeout = pipeline.config.llm_timeout_s
        question_timeout = llm_timeout + 30 if llm_timeout is not None else None
        try:
            async with asyncio.timeout(question_timeout):
                t0 = time.monotonic()
                retrieval = await pipeline.retrieve(qa.question)
                retrieval_s = time.monotonic() - t0

                context = "\n".join(doc.text for doc in retrieval.documents)
                prompt = FREE_FORM_ANSWER_PROMPT.format(context=context, question=qa.question)

                t0 = time.monotonic()
                raw_answer, gen_cost = await pipeline.generate(prompt)
                generation_s = time.monotonic() - t0

            expansion_cost = retrieval.expansion_cost
            llm_cost_usd = float(expansion_cost.get("usd", 0.0)) + float(gen_cost.get("usd", 0.0))
            prompt_tokens = int(expansion_cost.get("prompt_tokens", 0)) + int(gen_cost.get("prompt_tokens", 0))
            completion_tokens = int(expansion_cost.get("completion_tokens", 0)) + int(
                gen_cost.get("completion_tokens", 0)
            )

            pred = (raw_answer or "").strip()
            retrieved_doc_ids = [str(doc.metadata.get("doc_id", "")) for doc in retrieval.documents]

            em = best_em(pred, qa.gold_answers)
            f1 = best_f1(pred, qa.gold_answers)
            _, first_gold_rank = retrieval_metrics(retrieved_doc_ids, qa.supporting_doc_ids)

            judge_score: int | None = None
            if self.judge_model is not None:
                judge_score = await llm_judge(
                    self.judge_model, qa.question, pred, qa.gold_answers, timeout_s=self.judge_timeout_s
                )

            return QAResult(
                id=qa.id,
                question=qa.question,
                gold_answers=qa.gold_answers,
                pred=pred,
                em=em,
                f1=f1,
                judge=judge_score,
                retrieved_doc_ids=retrieved_doc_ids,
                supporting_doc_ids=qa.supporting_doc_ids,
                retrieval_rank_of_first_gold=first_gold_rank,
                retrieval_s=retrieval_s,
                generation_s=generation_s,
                llm_cost_usd=llm_cost_usd,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except TimeoutError:
            tqdm.write(f"  TIMEOUT {qa.id}")
            return _error_result(qa, TRANSIENT_ERROR_SENTINEL)
        except Exception as exc:
            error_summary = format_llm_error(exc)
            sentinel = PERMANENT_ERROR_SENTINEL if is_permanent_llm_error(exc) else TRANSIENT_ERROR_SENTINEL
            tqdm.write(f"  ERROR {qa.id} | {error_summary}")
            logger.debug("QA evaluation failed for %s", qa.id, exc_info=True)
            return _error_result(qa, sentinel)


def _error_result(qa: BenchmarkQAPair, sentinel: str) -> QAResult:
    return QAResult(
        id=qa.id,
        question=qa.question,
        gold_answers=qa.gold_answers,
        pred="",
        em=0.0,
        f1=0.0,
        supporting_doc_ids=qa.supporting_doc_ids,
        error=sentinel,
    )


def is_error_sentinel(result: QAResult) -> bool:
    return result.error in ERROR_SENTINELS
