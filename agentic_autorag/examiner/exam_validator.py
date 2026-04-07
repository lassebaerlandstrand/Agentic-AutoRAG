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
import re

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
Answer the following multiple-choice question. The context below contains \
the information needed to determine the correct answer. Use the context \
as your primary source. If the context is clearly insufficient, select E.

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


_SYNTHESIS_PREFIX = "From the document's data:"

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "in",
        "of",
        "to",
        "and",
        "or",
        "for",
        "with",
        "that",
        "this",
        "from",
        "by",
        "on",
        "at",
        "be",
        "as",
        "it",
        "its",
        "has",
        "have",
        "had",
        "not",
        "but",
        "no",
        "so",
        "if",
        "than",
        "into",
        "also",
        "been",
        "which",
        "when",
        "where",
        "who",
        "will",
        "would",
        "can",
        "could",
        "may",
        "each",
        "all",
        "both",
        "their",
        "there",
        "then",
        "these",
        "those",
        "such",
        "other",
        "more",
        "about",
        "between",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "over",
        "under",
        "only",
        "very",
    }
)


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace so multiline snippets can be matched robustly."""
    return " ".join(text.split())


def _normalize_for_matching(text: str) -> str:
    """Normalize text for matching, stripping formatting artifacts from tables."""
    text = re.sub(r"[|+\-]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _normalized_contains(needle: str, haystack: str) -> bool:
    """Return True when normalized needle is a substring of normalized haystack."""
    needle_norm = _normalize_for_matching(needle)
    haystack_norm = _normalize_for_matching(haystack)
    if not needle_norm:
        return False
    return needle_norm in haystack_norm


def _token_overlap_ratio(source_fact: str, doc_text: str) -> float:
    """Fraction of source_fact content tokens found in the document.

    Removes stop words to avoid inflated overlap from common words.
    Handles table-derived source_facts where the LLM reformulates table data —
    the key terms (numbers, names, technical terms) will still be present in
    the document even if the formatting is different.
    """
    fact_tokens = set(_normalize_for_matching(source_fact).split()) - _STOP_WORDS
    doc_tokens = set(_normalize_for_matching(doc_text).split()) - _STOP_WORDS
    if not fact_tokens:
        return 0.0
    return len(fact_tokens & doc_tokens) / len(fact_tokens)


_TOKEN_OVERLAP_THRESHOLD = 0.7
_SYNTHESIS_THRESHOLD_REDUCTION = 0.10


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

    Uses a three-strategy cascade (tried in order, first pass wins):
      1. Normalized substring match (fast, handles exact/near-exact copies)
      2. Token overlap match (fast, handles table-derived synthesized facts)
      3. Embedding similarity (expensive, handles paraphrased facts)

    Source facts prefixed with "From the document's data:" are recognized as
    LLM-synthesized summaries of table/list content. For these, substring match
    is skipped and embedding similarity uses a relaxed threshold.

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
        chunk_size=window_chunk_size * 5,
        chunk_overlap=effective_overlap * 5,
    )

    passed: list[MCQQuestion] = []
    skipped_no_doc = 0
    skipped_too_short = 0
    n_substring_pass = 0
    n_token_overlap_pass = 0
    similarities: list[float] = []

    for q in questions:
        source_fact = _normalize_whitespace(q.source_fact)
        if len(source_fact) < min_source_fact_length:
            skipped_too_short += 1
            _log_rejection(
                logger,
                reason=f"source_fact_too_short (len={len(source_fact)} < min={min_source_fact_length})",
                q=q,
            )
            continue

        doc_id = q.source_doc_ids[0]
        if doc_id not in documents:
            passed.append(q)
            skipped_no_doc += 1
            continue

        doc_text = documents[doc_id]
        is_synthesis = source_fact.startswith(_SYNTHESIS_PREFIX)

        # Strategy 1: Normalized substring match (skip for synthesized facts)
        if not is_synthesis and substring_fallback and _normalized_contains(source_fact, doc_text):
            passed.append(q)
            n_substring_pass += 1
            continue

        # Strategy 2: Token overlap match (handles table-derived content)
        overlap = _token_overlap_ratio(source_fact, doc_text)
        if overlap >= _TOKEN_OVERLAP_THRESHOLD:
            passed.append(q)
            n_token_overlap_pass += 1
            continue

        # Strategy 3: Embedding similarity (most expensive, last resort)
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

        effective_threshold = threshold - _SYNTHESIS_THRESHOLD_REDUCTION if is_synthesis else threshold
        logger.debug("Source fact similarity for %s: %.3f (threshold=%.2f)", q.id, max_sim, effective_threshold)

        if max_sim >= effective_threshold:
            passed.append(q)
        else:
            _log_rejection(
                logger,
                reason=f"source_fact (sim={max_sim:.3f} < {effective_threshold:.2f}, overlap={overlap:.2f})",
                q=q,
            )

    n_removed = len(questions) - len(passed)
    n_checked = len(questions) - skipped_no_doc
    logger.info(
        "Source fact verification: %d/%d passed (threshold=%.2f, "
        "substring=%d, token_overlap=%d, skipped: too_short=%d, no_doc=%d)",
        len(passed),
        len(questions),
        threshold,
        n_substring_pass,
        n_token_overlap_pass,
        skipped_too_short,
        skipped_no_doc,
    )
    if similarities:
        logger.debug(
            "Embedding similarity stats: mean=%.3f, min=%.3f, max=%.3f",
            np.mean(similarities),
            np.min(similarities),
            np.max(similarities),
        )

    high_failure_threshold = 0.5
    if n_checked > 0 and (n_removed / n_checked) > high_failure_threshold:
        logger.warning(
            "%d questions removed — examiner may be hallucinating facts. Consider a more capable examiner model.",
            n_removed,
        )

    return passed


_LEAK_TEMPERATURES = (0.3, 0.7, 1.0)


async def check_parametric_leaks(
    questions: list[MCQQuestion],
    model: str,
    concurrency: int = 10,
    n_trials: int = 3,
) -> list[MCQQuestion]:
    """Layer 3: Remove questions answerable without any context (parametric leaks).

    Sends each question to the LLM with no context. Uses majority voting:
    a question is flagged as a leak when ``leak_threshold`` or more trials
    answer correctly (default: 2 out of 3). Each trial uses a different
    temperature to test both confident knowledge and lucky guesses.

    Transient LLM errors are retried after escalating cooldowns. Questions
    that permanently fail are removed conservatively (treated as potential leaks).
    """
    if not questions:
        return []

    leak_threshold = n_trials // 2 + 1  # majority: 1→1, 2→2, 3→2, 4→3, 5→3
    temperatures = list(_LEAK_TEMPERATURES[:n_trials])
    while len(temperatures) < n_trials:
        temperatures.append(0.7)

    _TRANSIENT = object()
    results: dict[int, int | object] = {}
    sem = asyncio.Semaphore(concurrency)

    async def _run_pass(indices: list[int], pbar: tqdm) -> None:  # type: ignore[type-arg]
        async def _check_one(idx: int) -> None:
            q = questions[idx]
            try:
                correct_count = 0
                for trial_idx in range(n_trials):
                    async with sem:
                        selected = await _call_mcq(
                            q,
                            context="No context available.",
                            model=model,
                            prompt_template=_PARAMETRIC_LEAK_ANSWER_PROMPT,
                            valid_keys=set(q.options.keys()) | {"E"},
                            temperature=temperatures[trial_idx],
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
                    results[idx] = n_trials  # treat as potential leak → remove
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

    # Permanently failed → remove conservatively (treat as potential leak)
    n_permanent = sum(1 for r in results.values() if r is _TRANSIENT)
    if n_permanent:
        logger.warning(
            "%d question(s) could not be leak-checked after all retries; removing them conservatively",
            n_permanent,
        )

    passed: list[MCQQuestion] = []
    for idx, q in enumerate(questions):
        correct_count = results.get(idx, 0)
        if correct_count is _TRANSIENT:
            correct_count = n_trials  # treat as potential leak
        if int(correct_count) < leak_threshold:
            passed.append(q)
        else:
            _log_rejection(
                logger,
                reason=f"parametric_leak ({correct_count}/{n_trials} correct, threshold={leak_threshold})",
                q=q,
            )

    n_removed = len(questions) - len(passed)
    logger.info(
        "Parametric leak check: %d questions removed (LLM answered without context, threshold=%d/%d)",
        n_removed,
        leak_threshold,
        n_trials,
    )

    leak_rate = n_removed / len(questions) if questions else 0.0
    if leak_rate > 0.30:
        logger.warning(
            "%.0f%% parametric leak rate — corpus may contain commonly known information.",
            leak_rate * 100,
        )

    return passed


def _extract_context_window(doc_text: str, source_fact: str, window_words: int = 300) -> str:
    """Extract a window of text around the source_fact from the document.

    Falls back to the source_fact itself when the anchor is not found (e.g.,
    synthesized table-derived source_facts).
    """
    if not source_fact or not doc_text:
        return source_fact or doc_text[:2000]

    anchor = source_fact[:50]
    pos = doc_text.find(anchor)
    if pos == -1:
        return source_fact

    pre_text = doc_text[:pos]
    words_before = pre_text.split()
    start_word = max(0, len(words_before) - window_words // 2)
    all_words = doc_text.split()
    end_word = min(len(all_words), start_word + window_words)
    return " ".join(all_words[start_word:end_word])


async def check_oracle(
    questions: list[MCQQuestion],
    model: str,
    concurrency: int = 10,
    documents: dict[str, str] | None = None,
    oracle_context_window_words: int = 300,
    oracle_retry_with_full_doc: bool = True,
) -> list[MCQQuestion]:
    """Layer 4: Remove questions that are broken even when given context.

    First tries a context window around the source_fact (broader than just
    the source_fact). If the LLM selects "E" (insufficient context) and
    ``oracle_retry_with_full_doc`` is enabled, retries with the full document.

    Transient LLM errors are retried after escalating cooldowns. Questions
    that permanently fail are removed conservatively.
    """
    if not questions:
        return []

    docs = documents or {}
    _TRANSIENT = object()
    results: dict[int, str | object] = {}
    sem = asyncio.Semaphore(concurrency)

    async def _run_pass(indices: list[int], pbar: tqdm) -> None:  # type: ignore[type-arg]
        async def _check_one(idx: int) -> None:
            q = questions[idx]
            doc_text = docs.get(q.source_doc_ids[0], "") if q.source_doc_ids else ""

            # Build context: broader window around source_fact
            if doc_text and q.source_fact:
                context = _extract_context_window(doc_text, q.source_fact, oracle_context_window_words)
            else:
                context = q.source_fact or "No source fact available."

            try:
                async with sem:
                    selected = await _call_mcq(
                        q,
                        context=context,
                        model=model,
                        prompt_template=_ORACLE_ANSWER_PROMPT,
                        valid_keys=set(q.options.keys()) | {"E"},
                    )

                # Retry with full document if LLM says "insufficient context"
                if selected == "E" and oracle_retry_with_full_doc and doc_text:
                    async with sem:
                        selected = await _call_mcq(
                            q,
                            context=doc_text,
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
    retrieval_filter_chunks: list[str] | None = None,
    retrieval_filter_embeddings: np.ndarray | None = None,
    retrieval_filter_embedder: object | None = None,
    retrieval_difficulty_top_k: int = 1,
) -> list[MCQQuestion]:
    """Run the full quality validation pipeline on candidate questions.

    Layers are applied sequentially (cheapest first):
      Layer 2:   Source fact verification (embedding similarity, no LLM)
      Layer 2.5: Retrieval difficulty filter (optional, no LLM)
      Layer 3:   Parametric leak check (multi-trial LLM, optional)
      Layer 4:   Oracle check (LLM)

    The retrieval difficulty filter runs before the expensive LLM checks,
    removing questions that are trivially retrievable by the weakest pipeline.

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
        retrieval_filter_chunks: Chunks from a weak index for retrieval difficulty filter.
        retrieval_filter_embeddings: Embeddings for those chunks.
        retrieval_filter_embedder: Embedder for encoding questions (can differ from Layer 2 embedder).
        retrieval_difficulty_top_k: Remove questions whose source_fact is in top-k chunks.

    Returns:
        Questions that passed all enabled layers.
    """
    run_logger = logging.getLogger("agentic_autorag.run")
    n_candidates = len(questions)
    run_logger.info("Starting validation pipeline with %d candidates", n_candidates)

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
    run_logger.info(
        "Source fact verification: %d/%d passed (%d removed)",
        n_after_source,
        n_candidates,
        n_candidates - n_after_source,
    )

    if not questions:
        logger.warning("No questions survived source fact verification")
        return []

    # Layer 2.5: Retrieval difficulty filter (no LLM, runs before expensive checks)
    n_after_retrieval = n_after_source
    if (
        retrieval_filter_chunks is not None
        and retrieval_filter_embeddings is not None
        and retrieval_filter_embedder is not None
    ):
        questions = filter_easy_retrieval(
            questions,
            chunks=retrieval_filter_chunks,
            chunk_embeddings=retrieval_filter_embeddings,
            embedder=retrieval_filter_embedder,
            max_easy_rank=retrieval_difficulty_top_k,
        )
        n_after_retrieval = len(questions)
        run_logger.info(
            "Retrieval difficulty filter: %d/%d passed (%d trivially retrievable, top_k=%d)",
            n_after_retrieval,
            n_after_source,
            n_after_source - n_after_retrieval,
            retrieval_difficulty_top_k,
        )

        if not questions:
            logger.warning("No questions survived retrieval difficulty filter")
            return []

    # Layer 3: Parametric leak check
    n_after_leak = n_after_retrieval
    if detect_parametric_leaks:
        questions = await check_parametric_leaks(
            questions,
            model=model,
            concurrency=concurrency,
            n_trials=parametric_leak_trials,
        )
        n_after_leak = len(questions)
        logger.info("After Layer 3 (parametric check): %d remaining", n_after_leak)
        run_logger.info(
            "Parametric leak check: %d/%d passed (%d removed, %.0f%% leak rate)",
            n_after_leak,
            n_after_retrieval,
            n_after_retrieval - n_after_leak,
            (n_after_retrieval - n_after_leak) / n_after_retrieval * 100 if n_after_retrieval else 0,
        )

    if not questions:
        logger.warning("No questions survived parametric leak check")
        return []

    # Layer 4: Oracle check
    questions = await check_oracle(
        questions,
        model=model,
        concurrency=concurrency,
        documents=documents,
    )
    n_after_oracle = len(questions)
    logger.info("After Layer 4 (oracle check): %d remaining", n_after_oracle)
    run_logger.info(
        "Oracle verification: %d/%d passed (%d removed)",
        n_after_oracle,
        n_after_leak,
        n_after_leak - n_after_oracle,
    )

    # Funnel summary
    funnel_parts = [f"{n_candidates} candidates"]
    funnel_parts.append(f"{n_after_source} source_fact")
    if n_after_retrieval != n_after_source:
        funnel_parts.append(f"{n_after_retrieval} retrieval_difficulty")
    if detect_parametric_leaks:
        funnel_parts.append(f"{n_after_leak} parametric")
    funnel_parts.append(f"{n_after_oracle} oracle (final)")
    run_logger.info("Validation funnel: %s", " → ".join(funnel_parts))

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


# ---------------------------------------------------------------------------
# Retrieval difficulty filter
# ---------------------------------------------------------------------------

_RETRIEVAL_OVERLAP_THRESHOLD = 0.5


def filter_easy_retrieval(
    questions: list[MCQQuestion],
    chunks: list[str],
    chunk_embeddings: np.ndarray,
    embedder: object,
    max_easy_rank: int = 1,
) -> list[MCQQuestion]:
    """Remove questions whose source_fact is trivially retrievable.

    Embeds each question, finds the most similar chunks via cosine similarity,
    and checks whether any of the top-``max_easy_rank`` chunks contain the
    question's source_fact (measured by token overlap). Questions where the
    answer is in the top-k of the weakest retrieval config are too easy —
    every pipeline will find them — so they add no discrimination value.

    No LLM calls are made; the cost is one batch embedding + matrix multiply.

    Args:
        questions: Validated candidate questions (must have ``source_fact``).
        chunks: Text chunks from a weak retrieval index.
        chunk_embeddings: Pre-computed embeddings for *chunks* (n_chunks, dim).
        embedder: SentenceTransformer (or compatible) for encoding questions.
        max_easy_rank: Remove questions whose source_fact appears in the
            top-N retrieved chunks. Default 1 = only remove if the single
            best-matching chunk contains the answer.

    Returns:
        Questions that passed the retrieval difficulty filter.
    """
    if not questions or len(chunks) == 0:
        return list(questions)

    q_texts = [q.question for q in questions]
    q_embeddings = np.asarray(embedder.encode(q_texts), dtype=np.float32)  # type: ignore[union-attr]
    sim_matrix = cosine_similarity(q_embeddings, chunk_embeddings)  # (n_questions, n_chunks)

    passed: list[MCQQuestion] = []
    for i, q in enumerate(questions):
        if not q.source_fact:
            passed.append(q)
            continue

        top_indices = np.argsort(sim_matrix[i])[::-1][:max_easy_rank]
        found_in_top = False
        for idx in top_indices:
            overlap = _token_overlap_ratio(q.source_fact, chunks[idx])
            if overlap >= _RETRIEVAL_OVERLAP_THRESHOLD:
                found_in_top = True
                break

        if not found_in_top:
            passed.append(q)

    logger.info(
        "Retrieval difficulty filter: %d/%d passed (%d trivially retrievable, top_k=%d)",
        len(passed),
        len(questions),
        len(questions) - len(passed),
        max_easy_rank,
    )
    return passed
