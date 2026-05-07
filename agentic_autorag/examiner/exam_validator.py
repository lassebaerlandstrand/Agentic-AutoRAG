"""Validator for the open-ended exam pipeline.

  Oracle answerability (MUST succeed):
      Feed both gold spans concatenated as context to a strong validator
      model and ask the question. The expected answer is the canonical
      answer or one of its variants. Failures here = broken / ambiguous
      questions; reject them.

The discrimination dimension that previously lived in a "naive RAG must
fail" gate is now handled by the 4-probe filter in
``examiner.probe_selector``, called from ``orchestrator._generate_exam``
after this oracle gate. The probe filter measures discrimination directly
(by running diverse RAG configs over each candidate) instead of relying
on a single weak baseline.

Scoring uses the EM-or-judge stack from ``benchmark_eval.scoring``.
Judge calls fire whenever EM=0, since synthesized answers (counts,
comparatives, computed values) often paraphrase the canonical form.

This module also keeps the source-fact verifier (``verify_source_facts``)
and the retrieved-chunk relevance helpers (``chunk_contains_source_fact``,
``ngram_relevance``, ``filter_easy_retrieval``) — those are reused
unchanged by the open-ended evaluator and orchestrator.
"""

from __future__ import annotations

import asyncio
import logging
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from agentic_autorag.benchmark_eval.scoring import best_em, llm_judge
from agentic_autorag.config.models import OpenEndedQuestion
from agentic_autorag.engine.pipeline import RetrievedDocument
from agentic_autorag.examiner._errors import format_llm_error, is_transient_llm_error
from agentic_autorag.examiner.exam_agent import _call_completion
from agentic_autorag.examiner.prompts import ORACLE_OPEN_ENDED_PROMPT, answer_format_hint

logger = logging.getLogger(__name__)

_RETRY_COOLDOWNS = (10, 30, 60)
_ORACLE_SPAN_SEPARATOR = "\n\n---\n\n"


def _log_rejection(reason: str, q: OpenEndedQuestion, extra: str = "") -> None:
    logger.info("--- REMOVED: %s ---", reason)
    logger.info("  Q: %s", q.question)
    logger.info("  Canonical: %s", q.canonical_answer)
    logger.info("  Variants: %s", q.answer_variants)
    if extra:
        logger.info("  %s", extra)
    logger.info("")


# --- shared helpers (unchanged from prior MCQ pipeline) --------------------


# Build unicode-fold table programmatically to avoid F601 (visually-identical
# but byte-distinct space characters in source).
_UNICODE_FOLD_PAIRS: list[tuple[str, str]] = [
    (" ", " "),  # non-breaking space
    (" ", " "),  # figure space
    (" ", " "),  # narrow no-break space
    ("　", " "),  # ideographic space
    ("​", ""),  # zero-width space
    ("‐", "-"),  # hyphen
    ("‑", "-"),  # non-breaking hyphen
    ("‒", "-"),  # figure dash
    ("–", "-"),  # en dash
    ("—", "-"),  # em dash
    ("−", "-"),  # minus sign
    ("‘", "'"),  # left single quote
    ("’", "'"),  # right single quote
    ("‚", "'"),  # single low-9
    ("“", '"'),  # left double quote
    ("”", '"'),  # right double quote
    ("„", '"'),  # double low-9
    ("…", "..."),  # ellipsis
]
_UNICODE_FOLDS = str.maketrans(dict(_UNICODE_FOLD_PAIRS))


def _fold_unicode(text: str) -> str:
    return text.translate(_UNICODE_FOLDS)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _normalize_for_matching(text: str) -> str:
    text = _fold_unicode(text)
    text = re.sub(r"\|+|[+\-]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()
    return " ".join(w.rstrip(",.;:!?)") for w in text.split())


def normalized_contains(needle: str, haystack: str) -> bool:
    needle_norm = _normalize_for_matching(needle)
    haystack_norm = _normalize_for_matching(haystack)
    if not needle_norm:
        return False
    return needle_norm in haystack_norm


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int], min_chars: int) -> bool:
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return overlap >= min_chars


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _longest_consecutive_run(span_ngrams: list[tuple[str, ...]], chunk_set: set[tuple[str, ...]]) -> int:
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
    return chunk_id.startswith("lgchunk_")


def _locate_graph_chunk(
    chunk: RetrievedDocument,
    docs: dict[str, str],
    offset_cache: dict[str, tuple[str, int, int] | None],
) -> tuple[str, int, int] | None:
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
    question: OpenEndedQuestion,
    chunk: RetrievedDocument,
    docs: dict[str, str] | None = None,
    offset_cache: dict[str, tuple[str, int, int] | None] | None = None,
    *,
    min_overlap_chars: int = 50,
    ngram_size: int = 5,
    coverage_threshold: float = 0.5,
    min_run: int = 5,
    duplicate_alias_map: dict[str, str] | None = None,
    span_indices: tuple[int, ...] | None = None,
) -> bool:
    """True when the retrieved chunk overlaps any of the question's gold spans.

    ``duplicate_alias_map`` (alias_doc_id → canonical_doc_id) is consulted
    when present so a retrieved chunk from an aliased duplicate document
    counts toward the same source as a chunk from the canonical. Without
    canonicalization, retrieving ``paper_page_001.png`` for a question
    whose source is the canonical ``paper.pdf`` would falsely score zero.

    ``span_indices`` restricts the check to a subset of the question's
    spans (positions in ``source_spans``). ``None`` checks all spans.
    Used by the evaluator's per-span retrieval diagnostic to distinguish
    which gold span the retrieved chunk satisfies.
    """

    def _canon(doc_id: str) -> str:
        if duplicate_alias_map is None:
            return doc_id
        return duplicate_alias_map.get(doc_id, doc_id)

    all_spans = list(question.source_spans)
    all_span_offsets: list[tuple[int, int] | None] = list(question.source_span_offsets)
    all_doc_ids = [_canon(d) for d in question.source_doc_ids]

    active_indices = tuple(range(len(all_spans))) if span_indices is None else span_indices

    span_offsets = [all_span_offsets[i] for i in active_indices if i < len(all_span_offsets)]
    doc_ids = [all_doc_ids[i] for i in active_indices if i < len(all_doc_ids)]
    spans = [all_spans[i] for i in active_indices if i < len(all_spans)]

    if chunk.char_range is not None and any(o is not None for o in span_offsets):
        chunk_doc = _canon(str(chunk.metadata.get("doc_id", "")))
        for span_offset, doc_id in zip(span_offsets, doc_ids, strict=False):
            if span_offset is None:
                continue
            if chunk_doc == doc_id and _intervals_overlap(span_offset, chunk.char_range, min_overlap_chars):
                return True

    if (
        _is_verbatim_graph_chunk(chunk.id)
        and docs is not None
        and offset_cache is not None
        and any(o is not None for o in span_offsets)
    ):
        loc = _locate_graph_chunk(chunk, docs, offset_cache)
        if loc is not None:
            graph_doc = _canon(loc[0])
            for span_offset, doc_id in zip(span_offsets, doc_ids, strict=False):
                if span_offset is None:
                    continue
                if graph_doc == doc_id and _intervals_overlap(span_offset, (loc[1], loc[2]), min_overlap_chars):
                    return True

    if spans:
        return ngram_relevance(
            spans,
            chunk.text,
            ngram_size=ngram_size,
            coverage_threshold=coverage_threshold,
            min_run=min_run,
        )
    return False


def _locate_span_in_doc(span: str, doc_text: str, fuzzy_threshold: float) -> tuple[int, int, str] | None:
    if not span or not doc_text:
        return None
    idx = doc_text.find(span)
    if idx >= 0:
        return (idx, idx + len(span), span)

    collapsed_chars: list[str] = []
    offset_map: list[int] = []
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
            return (start, end, doc_text[start:end])

    span_tokens = _normalize_for_matching(span).split()
    if len(span_tokens) < 5:
        return None
    span_ngrams = set(_ngrams(span_tokens, 5))
    if not span_ngrams:
        return None
    doc_words: list[tuple[str, int, int]] = []
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
    questions: list[OpenEndedQuestion],
    documents: dict[str, str],
    fuzzy_threshold: float = 0.9,
) -> list[OpenEndedQuestion]:
    """Verify each gold span is verbatim in its document and record offsets.

    Each question carries parallel ``source_chunk_ids`` / ``source_doc_ids``
    / ``source_spans`` lists. We locate each span in its source doc via
    exact match, whitespace tolerance, or fuzzy n-gram snap. Questions
    where any span can't be located are rejected. On success the question's
    ``source_span_offsets`` is populated for downstream chunk-relevance
    scoring.

    The composition prompt asks the LLM for 4-5 sentence verbatim source
    spans, so we don't apply a separate min-length filter here — fragments
    that slip past the prompt rarely survive the oracle gate downstream
    anyway.
    """
    if not questions:
        return []
    passed: list[OpenEndedQuestion] = []
    n_verbatim = n_tolerant = n_snap = n_rejected = 0

    for q in questions:
        spans = list(q.source_spans)
        doc_ids = q.source_doc_ids
        if not spans or len(doc_ids) != len(spans):
            _log_rejection("source_fact_doc_id_count", q)
            n_rejected += 1
            continue

        offsets: list[tuple[int, int] | None] = [None] * len(spans)
        ok = True
        match_mode = "verbatim"
        for i, (span, doc_id) in enumerate(zip(spans, doc_ids, strict=True)):
            if doc_id not in documents:
                offsets[i] = None
                continue
            doc_text = documents[doc_id]
            idx = doc_text.find(span)
            if idx >= 0:
                offsets[i] = (idx, idx + len(span))
                continue
            loc = _locate_span_in_doc(span, doc_text, fuzzy_threshold)
            if loc is None:
                ok = False
                break
            start, end, _ = loc
            offsets[i] = (start, end)
            if end - start == len(span):
                if match_mode == "verbatim":
                    match_mode = "tolerant"
            else:
                match_mode = "snap"

        if not ok:
            n_rejected += 1
            _log_rejection("source_fact_not_in_doc", q)
            continue

        if match_mode == "verbatim":
            n_verbatim += 1
        elif match_mode == "tolerant":
            n_tolerant += 1
        else:
            n_snap += 1

        passed.append(q.model_copy(update={"source_span_offsets": offsets}))

    n_removed = len(questions) - len(passed)
    logger.info(
        "Source span verification: verbatim=%d tolerant=%d snap=%d rejected=%d (of %d)",
        n_verbatim,
        n_tolerant,
        n_snap,
        n_rejected,
        len(questions),
    )
    if len(questions) > 0 and n_removed / len(questions) > 0.5:
        logger.warning(
            "%d/%d questions rejected at source-span verification — examiner may be hallucinating spans",
            n_removed,
            len(questions),
        )
    return passed


# --- gate 1: oracle-pass ----------------------------------------------------


async def _judge_open_ended_answer(
    question: OpenEndedQuestion,
    pred: str,
    judge_model: str | None,
) -> bool:
    """Score a free-text answer: EM is the cheap fast path; judge decides everything else.

    Returns True iff the answer is correct (EM>0 or judge=YES). The judge
    returns 1/0/-1/None for YES/NO/NO_ANSWER/error; only YES counts.
    Empty predictions skip the judge — there's nothing to grade.
    """
    pred = (pred or "").strip()
    if not pred:
        return False
    em = best_em(pred, question.gold_answers)
    if em > 0:
        return True
    if judge_model is None:
        return False
    judge = await llm_judge(judge_model, question.question, pred, question.gold_answers)
    return judge == 1


async def _answer_question(
    model: str,
    prompt_template: str,
    question: str,
    context: str,
    answer_format_hint: str = "a short answer (at most 15 tokens)",
) -> str:
    prompt = prompt_template.format(
        context=context,
        question=question,
        answer_format_hint=answer_format_hint,
    )
    return (await _call_completion(model, prompt, temperature=0.0)).strip()


async def gate_oracle_pass(
    questions: list[OpenEndedQuestion],
    *,
    validator_model: str,
    judge_model: str | None,
    concurrency: int = 10,
) -> list[OpenEndedQuestion]:
    """Keep only questions a strong model answers correctly with both gold spans."""
    if not questions:
        return []
    sem = asyncio.Semaphore(concurrency)
    verdicts: list[bool] = [False] * len(questions)

    async def _run(idx: int, q: OpenEndedQuestion) -> None:
        context = _ORACLE_SPAN_SEPARATOR.join(q.source_spans)
        hint = answer_format_hint(q.reasoning_type, q.formula_kind)
        for attempt, cooldown in enumerate((0, *_RETRY_COOLDOWNS), start=0):
            if cooldown:
                await asyncio.sleep(cooldown)
            try:
                async with sem:
                    pred = await _answer_question(
                        validator_model,
                        ORACLE_OPEN_ENDED_PROMPT,
                        q.question,
                        context,
                        answer_format_hint=hint,
                    )
                verdicts[idx] = await _judge_open_ended_answer(q, pred, judge_model)
                return
            except Exception as exc:
                if not is_transient_llm_error(exc):
                    logger.info("oracle gate permanent error %s: %s", q.id, format_llm_error(exc))
                    verdicts[idx] = False
                    return
                if attempt == len(_RETRY_COOLDOWNS):
                    logger.warning("oracle gate exhausted retries for %s: %s", q.id, format_llm_error(exc))
                    verdicts[idx] = False
                    return

    with tqdm(total=len(questions), desc="Gate 1: oracle answerability", unit="q") as pbar:

        async def _bounded(idx: int, q: OpenEndedQuestion) -> None:
            await _run(idx, q)
            pbar.update(1)

        await asyncio.gather(*[_bounded(i, q) for i, q in enumerate(questions)])

    from collections import Counter

    kept = [q for q, ok in zip(questions, verdicts, strict=True) if ok]
    n_removed = len(questions) - len(kept)
    logger.info(
        "Gate 1 oracle-pass: %d/%d kept (%d removed as unanswerable)",
        len(kept),
        len(questions),
        n_removed,
    )
    # DIAG per-type and per-origin oracle removal breakdown.
    attempts_by_type: Counter[str] = Counter()
    removed_by_type: Counter[str] = Counter()
    attempts_by_origin: Counter[str] = Counter()
    removed_by_origin: Counter[str] = Counter()
    for q, ok in zip(questions, verdicts, strict=True):
        attempts_by_type[q.reasoning_type] += 1
        if q.num_hops == 1:
            origin_label = "single_chunk"
        elif q.is_multi_doc:
            origin_label = "cross_doc_pair"
        else:
            origin_label = "same_doc_pair"
        attempts_by_origin[origin_label] += 1
        if not ok:
            removed_by_type[q.reasoning_type] += 1
            removed_by_origin[origin_label] += 1
    if attempts_by_type:
        type_breakdown = ", ".join(
            f"{t}={removed_by_type[t]}/{attempts_by_type[t]}" for t in sorted(attempts_by_type.keys())
        )
        logger.info("DIAG Oracle gate by type: %s", type_breakdown)
    if attempts_by_origin:
        origin_breakdown = ", ".join(
            f"{o}={removed_by_origin[o]}/{attempts_by_origin[o]}" for o in sorted(attempts_by_origin.keys())
        )
        logger.info("DIAG Oracle gate by origin: %s", origin_breakdown)
    return kept


async def run_validation_pipeline(
    questions: list[OpenEndedQuestion],
    documents: dict[str, str],
    *,
    validator_model: str,
    judge_model: str | None = None,
    concurrency: int = 10,
    source_fact_verify_fuzzy_threshold: float = 0.9,
) -> list[OpenEndedQuestion]:
    """Source-fact verification → oracle answerability gate.

    The discrimination dimension (was: ``gate_naive_rag_fail``) is handled
    downstream by the 4-probe filter in ``orchestrator._generate_exam``
    after this pipeline returns.
    """
    run_logger = logging.getLogger("agentic_autorag.run")

    n_in = len(questions)
    questions = verify_source_facts(
        questions,
        documents,
        fuzzy_threshold=source_fact_verify_fuzzy_threshold,
    )
    n_after_spans = len(questions)
    run_logger.info("Source spans: %d/%d passed", n_after_spans, n_in)
    if not questions:
        return []

    questions = await gate_oracle_pass(
        questions,
        validator_model=validator_model,
        judge_model=judge_model,
        concurrency=concurrency,
    )
    n_after_oracle = len(questions)
    run_logger.info("Oracle answerability: %d/%d passed", n_after_oracle, n_after_spans)
    run_logger.info(
        "Validation funnel: %d candidates → %d source_spans → %d oracle (final)",
        n_in,
        n_after_spans,
        n_after_oracle,
    )
    return questions


# --- retrieval-difficulty filter (kept for orchestrator compatibility) -----


def filter_easy_retrieval(
    questions: list[OpenEndedQuestion],
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
) -> list[OpenEndedQuestion]:
    """Remove questions whose source spans are trivially retrievable.

    Used by the orchestrator as a quick optional filter before the LLM gates.
    For 2-hop questions, "trivially retrievable" means at least one of the
    two gold spans appears in the top-``max_easy_rank`` chunks.
    """
    if not questions or len(chunks) == 0:
        return list(questions)

    q_texts = [q.question for q in questions]
    q_embeddings = np.asarray(embedder.encode(q_texts), dtype=np.float32)  # type: ignore[union-attr]
    sim_matrix = cosine_similarity(q_embeddings, chunk_embeddings)

    def _chunk_relevant_by_offset(q: OpenEndedQuestion, idx: int) -> bool:
        if chunk_ranges is None or chunk_doc_ids is None:
            return False
        cr = chunk_ranges[idx]
        doc_id = chunk_doc_ids[idx]
        for offset, q_doc_id in zip(q.source_span_offsets, q.source_doc_ids, strict=True):
            if offset is None:
                continue
            if doc_id == q_doc_id and _intervals_overlap(offset, cr, min_overlap_chars):
                return True
        return False

    passed: list[OpenEndedQuestion] = []
    for i, q in enumerate(questions):
        if not any(span for span in q.source_spans):
            passed.append(q)
            continue
        top_indices = np.argsort(sim_matrix[i])[::-1][:max_easy_rank]
        found_in_top = False
        for idx in top_indices:
            if _chunk_relevant_by_offset(q, int(idx)):
                found_in_top = True
                break
            if ngram_relevance(
                list(q.source_spans),
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
        "Retrieval difficulty filter: %d/%d passed (top_k=%d)",
        len(passed),
        len(questions),
        max_easy_rank,
    )
    return passed
