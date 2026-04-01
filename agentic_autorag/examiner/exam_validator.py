"""Quality validation pipeline for generated MCQ questions.

Runs candidate questions through four layers:
  Layer 1: Structural checks (handled by ExamAgent before this module)
  Layer 2: Source fact verification (embedding similarity, no LLM)
  Layer 3: Parametric leak check (LLM answers without context → remove)
  Layer 4: Oracle check (LLM can't answer WITH source_fact → remove)
"""

from __future__ import annotations

import asyncio
import logging

import litellm
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from agentic_autorag.config.models import MCQQuestion
from agentic_autorag.examiner._errors import format_llm_error, is_transient_llm_error
from agentic_autorag.examiner.evaluator import MCQEvaluator

logger = logging.getLogger(__name__)

_WINDOW_CHUNK_SIZE = 300
_WINDOW_CHUNK_OVERLAP = 150
_RETRY_COOLDOWNS = (10, 30, 60)

_ORACLE_ANSWER_PROMPT = """\
Answer the following multiple-choice question using ONLY the information provided \
in the context below. If the context does not contain enough information to determine \
the correct answer, select E.

Context:
{context}

Question: {question}
{options}
E) The provided context does not contain enough information to answer this question.

Reply with just the letter (A, B, C, D, or E).

Answer:"""

_PARAMETRIC_LEAK_ANSWER_PROMPT = """\
Answer the following multiple-choice question using NO external context.

Context:
{context}

Question: {question}
{options}
E) I don't know / insufficient information without context.

Reply with just the letter (A, B, C, D, or E).

Answer:"""


def _log_rejection(logger_: logging.Logger, reason: str, q: MCQQuestion, extra: str = "") -> None:
    """Emit a structured multi-line rejection log for a candidate question."""
    logger_.info("--- REMOVED: %s ---", reason)
    logger_.info("  Q: %s", q.question)
    for option_key in sorted(q.options.keys()):
        logger_.info("  %s: %s", option_key, q.options[option_key])
    logger_.info("  Correct: %s", q.correct_answer)
    logger_.info("  Source fact: %s", q.source_fact or "(none)")
    if extra:
        logger_.info("  %s", extra)
    logger_.info("")


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace so multiline snippets can be matched robustly."""
    return " ".join(text.split())


def _normalized_contains(needle: str, haystack: str) -> bool:
    """Return True when normalized needle is a substring of normalized haystack."""
    needle_norm = _normalize_whitespace(needle)
    haystack_norm = _normalize_whitespace(haystack)
    if not needle_norm:
        return False
    return needle_norm in haystack_norm


def verify_source_facts(
    questions: list[MCQQuestion],
    documents: dict[str, str],
    embedder,
    threshold: float = 0.65,
    substring_fallback: bool = True,
    min_source_fact_length: int = 60,
    window_chunk_size: int = _WINDOW_CHUNK_SIZE,
    window_chunk_overlap: int = _WINDOW_CHUNK_OVERLAP,
) -> list[MCQQuestion]:
    """Layer 2: Verify source facts are grounded in the source document.

    Splits the source document into overlapping windows, computes embeddings,
    and checks that the source_fact has high cosine similarity to at least one window.
    Questions with source_facts below the threshold are removed.

    When ``substring_fallback`` is True, source facts that appear verbatim in the
    document text pass automatically without the embedding check.

    Args:
        questions: Candidate questions with source_fact and source_doc_ids.
        documents: Mapping of doc_id to full document text.
        embedder: SentenceTransformer-compatible embedder.
        threshold: Minimum cosine similarity required (default 0.65).
        substring_fallback: If True, verbatim substrings pass without embedding check.

    Returns:
        Questions that passed the source fact verification.
    """
    if not questions:
        return []

    effective_overlap = min(window_chunk_overlap, window_chunk_size - 1)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=window_chunk_size * 5,  # ~5 chars per token estimate
        chunk_overlap=effective_overlap * 5,
    )

    passed: list[MCQQuestion] = []
    skipped_no_doc = 0
    skipped_too_short = 0
    similarities: list[float] = []

    for q in questions:
        source_fact = _normalize_whitespace(q.source_fact)
        if len(source_fact) < min_source_fact_length:
            skipped_too_short += 1
            _log_rejection(
                logger,
                reason=(f"source_fact_too_short (len={len(source_fact)} < min={min_source_fact_length})"),
                q=q,
            )
            continue

        doc_id = q.source_doc_ids[0]
        if doc_id not in documents:
            # Source document not available — let it through
            passed.append(q)
            skipped_no_doc += 1
            continue

        doc_text = documents[doc_id]

        # Substring fallback: if the source_fact appears verbatim in the document,
        # it is definitionally grounded — skip the embedding check.
        if substring_fallback and _normalized_contains(source_fact, doc_text):
            passed.append(q)
            continue

        windows = splitter.split_text(doc_text)
        if not windows:
            passed.append(q)
            continue

        all_texts = [source_fact] + windows
        all_embeddings = np.asarray(embedder.encode(all_texts), dtype=np.float32)
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        all_embeddings = all_embeddings / norms
        fact_embedding = all_embeddings[0:1]
        window_embeddings = all_embeddings[1:]

        sims = cosine_similarity(fact_embedding, window_embeddings)[0]
        max_sim = float(sims.max())
        similarities.append(max_sim)

        logger.debug("Source fact similarity for %s: %.3f", q.id, max_sim)

        if max_sim >= threshold:
            passed.append(q)
        else:
            _log_rejection(
                logger,
                reason=f"source_fact (sim={max_sim:.3f} < {threshold:.2f})",
                q=q,
            )

    n_removed = len(questions) - len(passed)
    n_checked = len(questions) - skipped_no_doc
    logger.info(
        "Source fact verification: %d/%d passed (threshold=%.2f, %d skipped: too_short=%d, no_doc=%d)",
        len(passed),
        len(questions),
        threshold,
        skipped_too_short + skipped_no_doc,
        skipped_too_short,
        skipped_no_doc,
    )
    if similarities:
        logger.debug(
            "Similarity stats: mean=%.3f, min=%.3f, max=%.3f",
            np.mean(similarities),
            np.min(similarities),
            np.max(similarities),
        )

    high_failure_threshold = 0.5
    if n_checked > 0 and (n_removed / n_checked) > high_failure_threshold:
        logger.warning(
            "WARNING: %d questions removed — examiner may be hallucinating facts. "
            "Consider a more capable examiner model.",
            n_removed,
        )

    return passed


async def check_parametric_leaks(
    questions: list[MCQQuestion],
    model: str,
    concurrency: int = 10,
    n_trials: int = 3,
) -> list[MCQQuestion]:
    """Layer 3: Remove questions answerable without any context (parametric leaks).

    Sends each question to the LLM with no context. When ``n_trials`` > 1, each
    question is checked multiple times with temperature > 0. A question is flagged
    as a leak only if ALL trials answer correctly. This substantially reduces
    false positives from occasional guessing.

    Transient LLM errors (rate limits, server errors) are retried after escalating
    cooldowns. Questions that permanently fail are kept conservatively (treated as
    if the LLM answered incorrectly).

    Args:
        questions: Candidate questions to check.
        model: LLM model string (passed to litellm).
        concurrency: Maximum concurrent LLM calls.
        n_trials: Number of independent trials per question (default 3).

    Returns:
        Questions that are NOT answerable from parametric knowledge alone.
    """
    if not questions:
        return []

    temperature = 0.7 if n_trials > 1 else 0.0
    _TRANSIENT = object()

    # results[i] = correct_count (int) after success, or _TRANSIENT sentinel on error
    results: dict[int, int | object] = {}
    sem = asyncio.Semaphore(concurrency)

    async def _run_pass(indices: list[int], pbar: tqdm) -> None:  # type: ignore[type-arg]
        async def _check_one(idx: int) -> None:
            q = questions[idx]
            try:
                correct_count = 0
                for _trial in range(n_trials):
                    async with sem:
                        selected = await _call_mcq(
                            q,
                            context="No context available.",
                            model=model,
                            prompt_template=_PARAMETRIC_LEAK_ANSWER_PROMPT,
                            valid_keys=set(q.options.keys()) | {"E"},
                            temperature=temperature,
                        )
                    if selected == q.correct_answer:
                        correct_count += 1
                results[idx] = correct_count
                pbar.update(1)
            except Exception as exc:
                if is_transient_llm_error(exc):
                    error_summary = format_llm_error(exc)
                    tqdm.write(f"  TRANSIENT ERROR leak check q[{idx}] | {error_summary}")
                    results[idx] = _TRANSIENT
                else:
                    logger.debug("Leak check failed for question %d: %s", idx, exc, exc_info=True)
                    results[idx] = 0  # treat as wrong → keep question
                    pbar.update(1)

        await asyncio.gather(*[_check_one(i) for i in indices])

    with tqdm(total=len(questions), desc="Checking parametric leaks", unit="q") as pbar:
        await _run_pass(list(range(len(questions))), pbar)

        for retry_round, cooldown in enumerate(_RETRY_COOLDOWNS, start=1):
            error_indices = [i for i, r in results.items() if r is _TRANSIENT]
            if not error_indices:
                break
            tqdm.write(
                f"\n  {len(error_indices)} leak check(s) failed"
                f" — retrying after {cooldown}s cooldown"
                f" (round {retry_round}/{len(_RETRY_COOLDOWNS)})"
            )
            await asyncio.sleep(cooldown)
            await _run_pass(error_indices, pbar)

    # Permanently failed → keep conservatively (treat as answered incorrectly)
    n_permanent = sum(1 for r in results.values() if r is _TRANSIENT)
    if n_permanent:
        logger.warning(
            "%d question(s) could not be leak-checked after all retries; keeping them conservatively",
            n_permanent,
        )

    passed: list[MCQQuestion] = []
    for idx, q in enumerate(questions):
        correct_count = results.get(idx, 0)
        if correct_count is _TRANSIENT:
            correct_count = 0
        if int(correct_count) < n_trials:
            passed.append(q)
        else:
            _log_rejection(
                logger,
                reason=f"parametric_leak_unanimous ({correct_count}/{n_trials} correct without context)",
                q=q,
            )

    n_removed = len(questions) - len(passed)
    logger.info(
        "Parametric leak check: %d questions removed (LLM answered without context)",
        n_removed,
    )

    leak_rate = n_removed / len(questions) if questions else 0.0
    if leak_rate > 0.30:
        logger.warning(
            "WARNING: %.0f%% parametric leak rate — corpus may contain commonly known information.",
            leak_rate * 100,
        )

    return passed


async def check_oracle(
    questions: list[MCQQuestion],
    model: str,
    concurrency: int = 10,
) -> list[MCQQuestion]:
    """Layer 4: Remove questions that are broken even when given the source_fact.

    Sends each question with source_fact as context (plus an 'E: insufficient context'
    escape option). If the LLM selects E or the wrong answer, the question is removed.

    Transient LLM errors (rate limits, server errors) are retried after escalating
    cooldowns. Questions that permanently fail are removed conservatively (treated as
    INVALID — better to discard a potentially broken question than to keep it).

    Args:
        questions: Candidate questions to check.
        model: LLM model string (passed to litellm).
        concurrency: Maximum concurrent LLM calls.

    Returns:
        Questions that are answerable when given the correct source_fact.
    """
    if not questions:
        return []

    _TRANSIENT = object()
    results: dict[int, str | object] = {}
    sem = asyncio.Semaphore(concurrency)

    async def _run_pass(indices: list[int], pbar: tqdm) -> None:  # type: ignore[type-arg]
        async def _check_one(idx: int) -> None:
            q = questions[idx]
            context = q.source_fact if q.source_fact else "No source fact available."
            try:
                async with sem:
                    selected = await _call_mcq(
                        q,
                        context=context,
                        model=model,
                        prompt_template=_ORACLE_ANSWER_PROMPT,
                        valid_keys=set(q.options.keys()) | {"E"},
                    )
                results[idx] = selected
                pbar.update(1)
            except Exception as exc:
                if is_transient_llm_error(exc):
                    error_summary = format_llm_error(exc)
                    tqdm.write(f"  TRANSIENT ERROR oracle check q[{idx}] | {error_summary}")
                    results[idx] = _TRANSIENT
                else:
                    logger.debug("Oracle check failed for question %d: %s", idx, exc, exc_info=True)
                    results[idx] = "INVALID"
                    pbar.update(1)

        await asyncio.gather(*[_check_one(i) for i in indices])

    with tqdm(total=len(questions), desc="Running oracle verification", unit="q") as pbar:
        await _run_pass(list(range(len(questions))), pbar)

        for retry_round, cooldown in enumerate(_RETRY_COOLDOWNS, start=1):
            error_indices = [i for i, r in results.items() if r is _TRANSIENT]
            if not error_indices:
                break
            tqdm.write(
                f"\n  {len(error_indices)} oracle check(s) failed"
                f" — retrying after {cooldown}s cooldown"
                f" (round {retry_round}/{len(_RETRY_COOLDOWNS)})"
            )
            await asyncio.sleep(cooldown)
            await _run_pass(error_indices, pbar)

    # Permanently failed → remove conservatively (unknown = potentially broken)
    n_permanent = sum(1 for r in results.values() if r is _TRANSIENT)
    if n_permanent:
        logger.warning(
            "%d question(s) could not be oracle-checked after all retries; removing them conservatively",
            n_permanent,
        )

    passed: list[MCQQuestion] = []
    for idx, q in enumerate(questions):
        selected = results.get(idx, "INVALID")
        if selected is _TRANSIENT:
            selected = "INVALID"
        if selected == q.correct_answer:
            passed.append(q)
        else:
            _log_rejection(
                logger,
                reason=f"oracle_fail (selected={selected}, correct={q.correct_answer})",
                q=q,
            )

    n_removed = len(questions) - len(passed)
    logger.info(
        "Oracle check: %d questions removed (unanswerable even with source fact)",
        n_removed,
    )
    return passed


async def run_validation_pipeline(
    questions: list[MCQQuestion],
    documents: dict[str, str],
    embedder,
    model: str,
    concurrency: int = 10,
    source_fact_threshold: float = 0.65,
    detect_parametric_leaks: bool = True,
    source_fact_substring_fallback: bool = True,
    source_fact_min_length: int = 60,
    source_fact_window_chunk_size: int = _WINDOW_CHUNK_SIZE,
    source_fact_window_chunk_overlap: int = _WINDOW_CHUNK_OVERLAP,
    parametric_leak_trials: int = 3,
) -> list[MCQQuestion]:
    """Run the full quality validation pipeline (Layers 2-4) on candidate questions.

    Layers are applied sequentially:
      Layer 2: Source fact verification (embedding similarity + optional substring fallback)
      Layer 3: Parametric leak check (multi-trial LLM, optional)
      Layer 4: Oracle check (LLM)

    Args:
        questions: Candidate questions (already passed Layer 1 structural checks).
        documents: Mapping of doc_id to document text for Layer 2.
        embedder: SentenceTransformer-compatible embedder for Layer 2.
        model: LLM model for Layers 3-4.
        concurrency: Max concurrent LLM calls for Layers 3-4.
        source_fact_threshold: Minimum source fact similarity (Layer 2).
        detect_parametric_leaks: Whether to run Layer 3.
        source_fact_substring_fallback: Pass verbatim source facts without embedding check.
        parametric_leak_trials: Number of independent trials for Layer 3.

    Returns:
        Questions that passed all enabled layers.
    """
    n_candidates = len(questions)
    logger.info("Starting validation pipeline with %d candidates", n_candidates)

    # Layer 2: Source fact verification
    questions = verify_source_facts(
        questions,
        documents,
        embedder,
        threshold=source_fact_threshold,
        substring_fallback=source_fact_substring_fallback,
        min_source_fact_length=source_fact_min_length,
        window_chunk_size=source_fact_window_chunk_size,
        window_chunk_overlap=source_fact_window_chunk_overlap,
    )
    n_after_source = len(questions)
    logger.info("After Layer 2 (source fact): %d remaining", n_after_source)

    if not questions:
        logger.warning("No questions survived source fact verification")
        return []

    # Layer 3: Parametric leak check
    n_after_leak = n_after_source
    if detect_parametric_leaks:
        questions = await check_parametric_leaks(
            questions,
            model=model,
            concurrency=concurrency,
            n_trials=parametric_leak_trials,
        )
        n_after_leak = len(questions)
        logger.info("After Layer 3 (parametric check): %d remaining", n_after_leak)

    if not questions:
        logger.warning("No questions survived parametric leak check")
        return []

    # Layer 4: Oracle check
    questions = await check_oracle(questions, model=model, concurrency=concurrency)
    n_after_oracle = len(questions)
    logger.info("After Layer 4 (oracle check): %d remaining", n_after_oracle)

    # Funnel summary
    funnel_parts = [f"{n_candidates} candidates"]
    funnel_parts.append(f"{n_after_source} source_fact")
    if detect_parametric_leaks:
        funnel_parts.append(f"{n_after_leak} parametric")
    funnel_parts.append(f"{n_after_oracle} oracle (final)")
    logger.info("Validation funnel: %s", " -> ".join(funnel_parts))

    return questions


async def _call_mcq(
    q: MCQQuestion,
    context: str,
    model: str,
    prompt_template: str,
    valid_keys: set[str],
    temperature: float = 0.0,
) -> str:
    """Make a single MCQ LLM call and parse the answer.

    Returns the selected option letter or "INVALID" on failure.
    """
    options_text = "\n".join(f"{k}) {v}" for k, v in q.options.items())
    prompt = prompt_template.format(
        context=context,
        question=q.question,
        options=options_text,
    )
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            num_retries=0,
        )
        raw = response.choices[0].message.content
        return MCQEvaluator._parse_answer(raw, valid_keys=valid_keys)
    except Exception as exc:
        if is_transient_llm_error(exc):
            raise
        logger.debug("MCQ call failed for question %s: %s", q.id, exc)
        return "INVALID"
