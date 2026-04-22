"""Quality validation pipeline for generated MCQ questions.

Runs candidate questions through four layers:
  Layer 1: Structural checks (handled by ExamAgent before this module)
  Layer 2: Source fact verify-and-locate (deterministic, no LLM, no embedder)
  Layer 3: Parametric leak check (LLM answers without context → remove)
  Layer 4: Oracle check (LLM can't answer WITH source_fact → remove)
"""

from __future__ import annotations

import asyncio
import logging
import re

import litellm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from agentic_autorag.config.models import MCQQuestion
from agentic_autorag.engine.pipeline import RetrievedDocument
from agentic_autorag.examiner._errors import format_llm_error, is_transient_llm_error
from agentic_autorag.examiner.evaluator import MCQEvaluator

logger = logging.getLogger(__name__)

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


# Unicode folding table for robustness against LLM whitespace/punctuation drift.
_UNICODE_FOLDS = str.maketrans(
    {
        " ": " ",  # non-breaking space
        " ": " ",
        " ": " ",
        " ": " ",
        "​": "",  # zero-width space
        "‐": "-",  # hyphen
        "‑": "-",
        "‒": "-",
        "–": "-",  # en dash
        "—": "-",  # em dash
        "−": "-",  # minus
        "‘": "'",  # left single quote
        "’": "'",  # right single quote
        "‚": "'",
        "“": '"',  # left double quote
        "”": '"',  # right double quote
        "„": '"',
        "…": "...",  # ellipsis
    }
)


def _fold_unicode(text: str) -> str:
    """Fold common Unicode punctuation to ASCII equivalents."""
    return text.translate(_UNICODE_FOLDS)


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace so multiline snippets can be matched robustly."""
    return " ".join(text.split())


def _normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching. Aggressively strips formatting artifacts."""
    text = _fold_unicode(text)
    text = re.sub(r"\|+|[+\-]{2,}", " ", text)  # pipes + runs of +/-
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()
    # Strip trailing punctuation from tokens so "93.4%," matches "93.4%"
    return " ".join(w.rstrip(",.;:!?)") for w in text.split())


def normalized_contains(needle: str, haystack: str) -> bool:
    """Return True when normalized needle is a substring of normalized haystack."""
    needle_norm = _normalize_for_matching(needle)
    haystack_norm = _normalize_for_matching(haystack)
    if not needle_norm:
        return False
    return needle_norm in haystack_norm


# ---------------------------------------------------------------------------
# Deterministic chunk-relevance matcher (offset primary, n-gram fallback).
# See plan: Tier 1 interval overlap, Tier 2 str.find for graph chunks,
# Tier 3 n-gram coverage + consecutive run for synthesized content.
# ---------------------------------------------------------------------------


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int], min_chars: int) -> bool:
    """Return True when the two half-open intervals overlap by ≥ min_chars."""
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return overlap >= min_chars


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Return word n-grams as ordered tuples."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _longest_consecutive_run(span_ngrams: list[tuple[str, ...]], chunk_set: set[tuple[str, ...]]) -> int:
    """Longest run of consecutive span n-grams that all appear in chunk_set."""
    best = 0
    run = 0
    for ng in span_ngrams:
        if ng in chunk_set:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def ngram_relevance(
    spans: list[str],
    chunk_text: str,
    *,
    ngram_size: int = 5,
    coverage_threshold: float = 0.5,
    min_run: int = 5,
) -> bool:
    """Deterministic n-gram relevance check for non-verbatim chunks.

    A chunk is relevant if any span has either:
      - n-gram set coverage |A∩B| / min(|A|,|B|) ≥ coverage_threshold, or
      - a longest consecutive-run of matching n-grams ≥ min_run.

    For spans too short to form a single n-gram, falls back to normalized
    substring containment.
    """
    chunk_norm_tokens = _normalize_for_matching(chunk_text).split()
    chunk_ngrams_set = set(_ngrams(chunk_norm_tokens, ngram_size))
    chunk_ngrams_union = chunk_ngrams_set

    for span in spans:
        span_norm_tokens = _normalize_for_matching(span).split()
        if len(span_norm_tokens) < ngram_size:
            if span_norm_tokens and " ".join(span_norm_tokens) in " ".join(chunk_norm_tokens):
                return True
            continue
        span_ngram_list = _ngrams(span_norm_tokens, ngram_size)
        span_ngrams_set = set(span_ngram_list)
        if not span_ngrams_set or not chunk_ngrams_union:
            continue
        inter = span_ngrams_set & chunk_ngrams_union
        if inter:
            coverage = len(inter) / min(len(span_ngrams_set), len(chunk_ngrams_union))
            if coverage >= coverage_threshold:
                return True
            if _longest_consecutive_run(span_ngram_list, chunk_ngrams_union) >= min_run:
                return True
    return False


def _is_verbatim_graph_chunk(chunk_id: str) -> bool:
    """Verbatim graph chunks from LightRAG are tagged with this prefix."""
    return chunk_id.startswith("lgchunk_")


def _is_synthesized_graph_content(chunk_id: str) -> bool:
    return chunk_id.startswith("lgentity_") or chunk_id.startswith("lgrel_")


def _locate_graph_chunk(
    chunk: RetrievedDocument,
    docs: dict[str, str],
    offset_cache: dict[str, tuple[str, int, int] | None],
) -> tuple[str, int, int] | None:
    """Locate a verbatim graph chunk in its source doc via str.find, cached.

    Returns ``(doc_id, start, end)`` or ``None`` when the content can't be
    found in the referenced document (e.g., LightRAG normalized the text).
    """
    if chunk.id in offset_cache:
        return offset_cache[chunk.id]

    file_path = str(chunk.metadata.get("file_path", "") or "")
    result: tuple[str, int, int] | None = None
    if file_path and file_path in docs:
        doc = docs[file_path]
        idx = doc.find(chunk.text)
        if idx >= 0:
            result = (file_path, idx, idx + len(chunk.text))

    offset_cache[chunk.id] = result
    return result


def chunk_contains_source_fact(
    question: MCQQuestion,
    chunk: RetrievedDocument,
    docs: dict[str, str] | None = None,
    offset_cache: dict[str, tuple[str, int, int] | None] | None = None,
    *,
    min_overlap_chars: int = 50,
    ngram_size: int = 5,
    coverage_threshold: float = 0.5,
    min_run: int = 5,
) -> bool:
    """Return True when the retrieved chunk contains (part of) the question's source_fact.

    Three-tier deterministic matcher:
      Tier 1 — chunk has ``char_range`` (vector/hybrid path): interval overlap
               against any of the question's ``source_fact_offsets`` with
               matching ``doc_id``.
      Tier 2 — verbatim graph chunk (``lgchunk_*``): locate the chunk's text
               in the source document (``str.find`` with an LRU cache keyed by
               chunk id), then interval overlap. Falls through on miss.
      Tier 3 — synthesized graph content (``lgentity_*`` / ``lgrel_*``) or any
               chunk we can't locate: n-gram coverage + consecutive-run match
               against the span text.
    """
    spans = list(question.source_fact)
    span_offsets = list(question.source_fact_offsets)
    doc_id = question.source_doc_ids[0] if question.source_doc_ids else ""

    # Tier 1: chunk carries its own offset from the vector store.
    if chunk.char_range is not None and span_offsets:
        chunk_doc = str(chunk.metadata.get("doc_id", ""))
        if chunk_doc == doc_id:
            for span_range in span_offsets:
                if _intervals_overlap(span_range, chunk.char_range, min_overlap_chars):
                    return True

    # Tier 2: verbatim graph chunk — resolve its offset via source doc.
    if _is_verbatim_graph_chunk(chunk.id) and docs is not None and offset_cache is not None and span_offsets:
        loc = _locate_graph_chunk(chunk, docs, offset_cache)
        if loc is not None and loc[0] == doc_id:
            chunk_range = (loc[1], loc[2])
            for span_range in span_offsets:
                if _intervals_overlap(span_range, chunk_range, min_overlap_chars):
                    return True

    # Tier 3: synthesized content or unlocatable — n-gram fallback.
    if spans:
        return ngram_relevance(
            spans,
            chunk.text,
            ngram_size=ngram_size,
            coverage_threshold=coverage_threshold,
            min_run=min_run,
        )
    return False


def _locate_span_in_doc(
    span: str,
    doc_text: str,
    fuzzy_threshold: float,
) -> tuple[int, int, str] | None:
    """Locate a verbatim span in the source document, returning (start, end, text).

    Three-tier cascade:
      1. Primary: ``doc_text.find(span)`` (exact match)
      2. Whitespace-tolerant: match after collapsing whitespace on both sides,
         mapping back to original offsets
      3. Fuzzy snap-to-source: find the best-matching region via n-gram
         localisation within a sliding window, replace the span with the actual
         verbatim doc substring at that window

    Returns ``None`` when no sufficiently good match exists. On success, the
    returned ``text`` equals ``doc_text[start:end]`` exactly — this invariant
    is relied on by downstream chunk-relevance scoring.
    """
    if not span or not doc_text:
        return None

    # Tier 1: exact substring match
    idx = doc_text.find(span)
    if idx >= 0:
        return (idx, idx + len(span), span)

    # Tier 2: whitespace-tolerant match
    # Build doc_text with whitespace collapsed and a parallel offset map.
    collapsed_chars: list[str] = []
    offset_map: list[int] = []  # collapsed_idx -> original_idx
    prev_ws = False
    for i, ch in enumerate(doc_text):
        if ch.isspace():
            if prev_ws:
                continue
            collapsed_chars.append(" ")
            offset_map.append(i)
            prev_ws = True
        else:
            collapsed_chars.append(ch)
            offset_map.append(i)
            prev_ws = False
    collapsed_doc = "".join(collapsed_chars)
    collapsed_span = re.sub(r"\s+", " ", span).strip()
    if collapsed_span:
        pos = collapsed_doc.find(collapsed_span)
        if pos >= 0:
            start = offset_map[pos]
            end_collapsed = pos + len(collapsed_span) - 1
            end = offset_map[end_collapsed] + 1 if end_collapsed < len(offset_map) else len(doc_text)
            # Snap to the actual substring in the original doc.
            actual = doc_text[start:end]
            return (start, end, actual)

    # Tier 3: fuzzy snap via n-gram localisation
    span_tokens = _normalize_for_matching(span).split()
    if len(span_tokens) < 5:
        return None
    span_ngrams = set(_ngrams(span_tokens, 5))
    if not span_ngrams:
        return None

    # Tokenize the doc with positions so we can snap to a word window.
    doc_words: list[tuple[str, int, int]] = []  # (token, start, end)
    for m in re.finditer(r"\S+", doc_text):
        doc_words.append((m.group(0), m.start(), m.end()))
    if len(doc_words) < 5:
        return None
    norm_doc_words = [_normalize_for_matching(w[0]) for w in doc_words]

    window_size = max(len(span_tokens), 5)
    best_overlap = 0.0
    best_window: tuple[int, int] | None = None
    for i in range(0, len(doc_words) - 4):
        end_i = min(i + window_size, len(doc_words))
        window_tokens = [t for t in norm_doc_words[i:end_i] if t]
        if len(window_tokens) < 5:
            continue
        window_ngrams = set(_ngrams(window_tokens, 5))
        if not window_ngrams:
            continue
        inter = span_ngrams & window_ngrams
        if not inter:
            continue
        overlap = len(inter) / min(len(span_ngrams), len(window_ngrams))
        if overlap > best_overlap:
            best_overlap = overlap
            best_window = (doc_words[i][1], doc_words[end_i - 1][2])

    if best_window is not None and best_overlap >= fuzzy_threshold:
        start, end = best_window
        return (start, end, doc_text[start:end])
    return None


def verify_source_facts(
    questions: list[MCQQuestion],
    documents: dict[str, str],
    min_source_fact_length: int = 150,
    fuzzy_threshold: float = 0.9,
) -> list[MCQQuestion]:
    """Layer 2: Verify source_fact spans are verbatim in the source document, record offsets.

    For each question, verifies every span. Each span must be locatable in the
    source document via exact match, whitespace-tolerant match, or fuzzy
    snap-to-source (n-gram coverage ≥ ``fuzzy_threshold``). On success, the
    question's ``source_fact`` is replaced with the exact verbatim text from
    the document and ``source_fact_offsets`` is populated.

    Questions where any span can't be located are rejected. Reports per-bucket
    counts so we can diagnose LLM drift.
    """
    if not questions:
        return []

    passed: list[MCQQuestion] = []
    skipped_no_doc = 0
    skipped_empty_spans = 0
    skipped_too_short = 0
    n_verbatim = 0
    n_tolerant = 0
    n_snap = 0
    n_rejected_missing = 0

    for q in questions:
        spans = list(q.source_fact)
        if not spans:
            skipped_empty_spans += 1
            _log_rejection(logger, reason="source_fact_empty", q=q)
            continue

        total_length = sum(len(_normalize_whitespace(s)) for s in spans)
        if total_length < min_source_fact_length:
            skipped_too_short += 1
            _log_rejection(
                logger,
                reason=f"source_fact_too_short (total_len={total_length} < min={min_source_fact_length})",
                q=q,
            )
            continue

        doc_id = q.source_doc_ids[0]
        if doc_id not in documents:
            passed.append(q)
            skipped_no_doc += 1
            continue

        doc_text = documents[doc_id]
        resolved_spans: list[str] = []
        resolved_offsets: list[tuple[int, int]] = []
        match_mode = "verbatim"  # tracks the weakest mode used
        rejected = False

        for span in spans:
            idx = doc_text.find(span)
            if idx >= 0:
                resolved_spans.append(span)
                resolved_offsets.append((idx, idx + len(span)))
                continue
            # Fall through to whitespace-tolerant + fuzzy snap
            loc = _locate_span_in_doc(span, doc_text, fuzzy_threshold=fuzzy_threshold)
            if loc is None:
                rejected = True
                break
            start, end, actual = loc
            # Infer whether tier 2 or tier 3 was used (cheap heuristic: exact length match ⇒ tier 2)
            if len(actual) == len(span):
                match_mode = "tolerant" if match_mode == "verbatim" else match_mode
            else:
                match_mode = "snap"
            resolved_spans.append(actual)
            resolved_offsets.append((start, end))

        if rejected:
            n_rejected_missing += 1
            _log_rejection(logger, reason="source_fact_not_in_doc", q=q)
            continue

        # Counter bookkeeping
        if match_mode == "verbatim":
            n_verbatim += 1
        elif match_mode == "tolerant":
            n_tolerant += 1
        else:
            n_snap += 1

        updated = q.model_copy(update={"source_fact": resolved_spans, "source_fact_offsets": resolved_offsets})
        # Sanity invariant
        for (start, end), expected in zip(resolved_offsets, resolved_spans, strict=True):
            if doc_text[start:end] != expected:
                logger.warning(
                    "Source fact offset invariant violated for %s: doc[%d:%d] != span",
                    q.id,
                    start,
                    end,
                )
        passed.append(updated)

    n_removed = len(questions) - len(passed)
    n_checked = len(questions) - skipped_no_doc - skipped_empty_spans - skipped_too_short
    logger.info(
        "Source fact verification: verbatim=%d tolerant=%d snap=%d rejected=%d too_short=%d no_doc=%d (of %d)",
        n_verbatim,
        n_tolerant,
        n_snap,
        n_rejected_missing,
        skipped_too_short,
        skipped_no_doc,
        len(questions),
    )

    high_failure_threshold = 0.5
    if n_checked > 0 and (n_removed / max(1, len(questions))) > high_failure_threshold:
        logger.warning(
            "%d questions removed — examiner may be producing non-verbatim source_facts. "
            "Consider a more capable examiner model.",
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


_ORACLE_SPAN_SEPARATOR = "\n\n---\n\n"


async def check_oracle(
    questions: list[MCQQuestion],
    model: str,
    concurrency: int = 10,
    documents: dict[str, str] | None = None,
    oracle_retry_with_full_doc: bool = True,
) -> list[MCQQuestion]:
    """Layer 4: Remove questions that are broken even when given their source_fact.

    Feeds the source_fact spans directly as context (joined with a visible
    separator). Since source_fact is now verbatim + contextual, no windowing
    is needed. If the LLM selects "E" (insufficient context) and
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

            context = _ORACLE_SPAN_SEPARATOR.join(q.source_fact) if q.source_fact else "No source fact available."

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
    model: str,
    concurrency: int = 10,
    detect_parametric_leaks: bool = True,
    source_fact_min_length: int = 150,
    source_fact_verify_fuzzy_threshold: float = 0.9,
    parametric_leak_trials: int = 3,
    retrieval_filter_chunks: list[str] | None = None,
    retrieval_filter_chunk_ranges: list[tuple[int, int]] | None = None,
    retrieval_filter_chunk_doc_ids: list[str] | None = None,
    retrieval_filter_embeddings: np.ndarray | None = None,
    retrieval_filter_embedder: object | None = None,
    retrieval_difficulty_top_k: int = 1,
    chunk_relevance_min_overlap_chars: int = 50,
    chunk_relevance_ngram_size: int = 5,
    chunk_relevance_overlap_threshold: float = 0.5,
    chunk_relevance_min_run: int = 5,
) -> list[MCQQuestion]:
    """Run the full quality validation pipeline on candidate questions.

    Layers are applied sequentially (cheapest first):
      Layer 2:   Source fact verify-and-locate (records offsets, no LLM)
      Layer 2.5: Retrieval difficulty filter (optional, no LLM)
      Layer 3:   Parametric leak check (multi-trial LLM, optional)
      Layer 4:   Oracle check (LLM)

    Args:
        questions: Candidate questions (already passed Layer 1 structural checks).
        documents: Mapping of doc_id to document text for Layer 2.
        model: LLM model for Layers 3-4.
        concurrency: Max concurrent LLM calls for Layers 3-4.
        detect_parametric_leaks: Whether to run Layer 3.
        source_fact_min_length: Minimum total span length (characters).
        source_fact_verify_fuzzy_threshold: Fuzzy n-gram threshold for snap-to-source.
        parametric_leak_trials: Number of independent trials for Layer 3.
        retrieval_filter_chunks: Chunks from a weak index for retrieval difficulty filter.
        retrieval_filter_chunk_ranges: (start, end) offsets for those chunks.
        retrieval_filter_chunk_doc_ids: doc_id per chunk.
        retrieval_filter_embeddings: Embeddings for those chunks.
        retrieval_filter_embedder: Embedder for encoding questions (can differ from Layer 2 embedder).
        retrieval_difficulty_top_k: Remove questions whose source_fact is in top-k chunks.

    Returns:
        Questions that passed all enabled layers.
    """
    run_logger = logging.getLogger("agentic_autorag.run")
    n_candidates = len(questions)
    run_logger.info("Starting validation pipeline with %d candidates", n_candidates)

    # Layer 2: Source fact verify-and-locate (records offsets on each question)
    questions = verify_source_facts(
        questions,
        documents,
        min_source_fact_length=source_fact_min_length,
        fuzzy_threshold=source_fact_verify_fuzzy_threshold,
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
            chunk_ranges=retrieval_filter_chunk_ranges,
            chunk_doc_ids=retrieval_filter_chunk_doc_ids,
            min_overlap_chars=chunk_relevance_min_overlap_chars,
            ngram_size=chunk_relevance_ngram_size,
            coverage_threshold=chunk_relevance_overlap_threshold,
            min_run=chunk_relevance_min_run,
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


def filter_easy_retrieval(
    questions: list[MCQQuestion],
    chunks: list[str],
    chunk_embeddings: np.ndarray,
    embedder: object,
    max_easy_rank: int = 1,
    chunk_ranges: list[tuple[int, int]] | None = None,
    chunk_doc_ids: list[str] | None = None,
    *,
    min_overlap_chars: int = 50,
    ngram_size: int = 5,
    coverage_threshold: float = 0.5,
    min_run: int = 5,
) -> list[MCQQuestion]:
    """Remove questions whose source_fact is trivially retrievable.

    Embeds each question, finds the most similar chunks via cosine similarity,
    and checks whether any of the top-``max_easy_rank`` chunks contain the
    question's source_fact. When ``chunk_ranges`` + ``chunk_doc_ids`` are
    provided, uses character-offset interval overlap (deterministic primary
    matcher); otherwise falls back to n-gram relevance on the chunk text.
    """
    if not questions or len(chunks) == 0:
        return list(questions)

    q_texts = [q.question for q in questions]
    q_embeddings = np.asarray(embedder.encode(q_texts), dtype=np.float32)  # type: ignore[union-attr]
    sim_matrix = cosine_similarity(q_embeddings, chunk_embeddings)  # (n_questions, n_chunks)

    def _chunk_relevant_by_offset(q: MCQQuestion, idx: int) -> bool:
        if chunk_ranges is None or chunk_doc_ids is None:
            return False
        cr = chunk_ranges[idx]
        doc_id = chunk_doc_ids[idx]
        if not q.source_doc_ids or doc_id != q.source_doc_ids[0]:
            return False
        return any(_intervals_overlap(span_range, cr, min_overlap_chars) for span_range in q.source_fact_offsets)

    passed: list[MCQQuestion] = []
    for i, q in enumerate(questions):
        if not q.source_fact:
            passed.append(q)
            continue

        top_indices = np.argsort(sim_matrix[i])[::-1][:max_easy_rank]
        found_in_top = False
        for idx in top_indices:
            if _chunk_relevant_by_offset(q, int(idx)):
                found_in_top = True
                break
            if ngram_relevance(
                list(q.source_fact),
                chunks[idx],
                ngram_size=ngram_size,
                coverage_threshold=coverage_threshold,
                min_run=min_run,
            ):
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
