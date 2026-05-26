"""Validator for the open-ended exam pipeline.

The oracle answerability gate feeds gold spans as context to a strong
validator and rejects questions whose canonical answer cannot be recovered.
Discrimination is filtered separately by ``examiner.probe_selector``.
Scoring uses the EM-or-judge stack from ``benchmark_eval.scoring``."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from agentic_autorag.benchmark_eval.scoring import best_em, llm_judge
from agentic_autorag.config.models import OpenEndedQuestion
from agentic_autorag.engine.pipeline import RetrievedDocument
from agentic_autorag.examiner._errors import RETRY_COOLDOWNS_S, format_llm_error, is_transient_llm_error
from agentic_autorag.examiner.exam_agent import _call_completion, _strip_markdown_fences
from agentic_autorag.examiner.prompts import (
    MULTI_HOP_DEPENDENCY_AND_ORACLE_PROMPT,
    ORACLE_OPEN_ENDED_PROMPT,
    answer_format_hint,
)

logger = logging.getLogger(__name__)

_ORACLE_SPAN_SEPARATOR = "\n\n---\n\n"


# Build unicode-fold table programmatically; the substitution table covers
# visually-identical but byte-distinct space and punctuation variants.
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
    *,
    report_path: Path | None = None,
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

    TEMPORARY DEBUG: when ``report_path`` is given, a pretty JSON file is
    written describing every question's outcome (which span matched
    verbatim/tolerant/snap and which failed to locate, with the offending
    span text and source doc length) so the high rejection rate can be
    debugged. Remove the ``report_path`` parameter, the ``report_records``
    accumulator, and the trailing write block once span-verification tuning
    is complete.
    """
    if not questions:
        return []
    passed: list[OpenEndedQuestion] = []
    n_verbatim = n_tolerant = n_snap = n_rejected = 0
    rejection_reasons: Counter[str] = Counter()
    report_records: list[dict[str, Any]] = []

    for q in questions:
        spans = list(q.source_spans)
        doc_ids = q.source_doc_ids
        if not spans or len(doc_ids) != len(spans):
            n_rejected += 1
            rejection_reasons["malformed_spans"] += 1
            report_records.append(
                {
                    "id": q.id,
                    "status": "rejected",
                    "rejection_reason": "malformed_spans",
                    "n_spans": len(spans),
                    "n_doc_ids": len(doc_ids),
                    "spans": [],
                }
            )
            continue

        offsets: list[tuple[int, int] | None] = [None] * len(spans)
        span_records: list[dict[str, Any]] = []
        ok = True
        match_mode = "verbatim"
        failing_index: int | None = None
        last_index = -1
        for i, (span, doc_id) in enumerate(zip(spans, doc_ids, strict=True)):
            last_index = i
            base: dict[str, Any] = {
                "doc_id": doc_id,
                "span": span,
                "span_len": len(span),
            }
            if doc_id not in documents:
                offsets[i] = None
                span_records.append({**base, "doc_in_corpus": False, "outcome": "doc_missing"})
                continue
            doc_text = documents[doc_id]
            idx = doc_text.find(span)
            if idx >= 0:
                offsets[i] = (idx, idx + len(span))
                span_records.append(
                    {
                        **base,
                        "doc_in_corpus": True,
                        "outcome": "verbatim",
                        "matched_offsets": [idx, idx + len(span)],
                    }
                )
                continue
            loc = _locate_span_in_doc(span, doc_text, fuzzy_threshold)
            if loc is None:
                ok = False
                failing_index = i
                span_records.append(
                    {
                        **base,
                        "doc_in_corpus": True,
                        "outcome": "not_found",
                        "doc_len": len(doc_text),
                    }
                )
                break
            start, end, matched_text = loc
            offsets[i] = (start, end)
            if end - start == len(span):
                per_span_mode = "tolerant"
                if match_mode == "verbatim":
                    match_mode = "tolerant"
            else:
                per_span_mode = "snap"
                match_mode = "snap"
            span_records.append(
                {
                    **base,
                    "doc_in_corpus": True,
                    "outcome": per_span_mode,
                    "matched_offsets": [start, end],
                    "matched_text": matched_text,
                }
            )

        if not ok:
            n_rejected += 1
            rejection_reasons["span_not_found"] += 1
            for j in range(last_index + 1, len(spans)):
                span_records.append(
                    {
                        "doc_id": doc_ids[j],
                        "span": spans[j],
                        "span_len": len(spans[j]),
                        "doc_in_corpus": doc_ids[j] in documents,
                        "outcome": "not_attempted",
                    }
                )
            report_records.append(
                {
                    "id": q.id,
                    "status": "rejected",
                    "rejection_reason": "span_not_found",
                    "failing_span_index": failing_index,
                    "spans": span_records,
                }
            )
            continue

        if match_mode == "verbatim":
            n_verbatim += 1
        elif match_mode == "tolerant":
            n_tolerant += 1
        else:
            n_snap += 1

        report_records.append(
            {
                "id": q.id,
                "status": "kept",
                "match_mode": match_mode,
                "spans": span_records,
            }
        )
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

    if report_path is not None:
        report = {
            "summary": {
                "n_total": len(questions),
                "n_kept": len(passed),
                "n_rejected": n_rejected,
                "match_modes": {
                    "verbatim": n_verbatim,
                    "tolerant": n_tolerant,
                    "snap": n_snap,
                },
                "rejection_reasons": dict(rejection_reasons),
                "fuzzy_threshold": fuzzy_threshold,
            },
            "questions": report_records,
        }
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            logger.info("Wrote span verification report to %s", report_path)
        except Exception:
            logger.warning("Failed to write span verification report", exc_info=True)

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


_MULTIHOP_PROMPT_VERSION = "multihop_dependency_oracle_v1"


def _render_spans_block(spans: list[str]) -> str:
    return "\n\n".join(f"Span {i}:\n{s}" for i, s in enumerate(spans))


def _parse_multihop_response(raw: str) -> dict[str, Any] | None:
    """Parse the JSON object returned by the unified multi-hop prompt.

    Tolerates fenced ```json blocks and trailing commas. Returns None on
    any parse failure so the caller can drop the candidate conservatively.
    """
    text = _strip_markdown_fences(raw).strip()
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


async def _unified_multihop_check(
    q: OpenEndedQuestion,
    validator_model: str,
) -> tuple[bool, list[int], dict[str, str], str, str] | None:
    """Run the unified decomposability + oracle-answer call for a multi-hop question.

    Returns ``(decomposable, sufficient_spans, supporting_quotes, reasoning, answer)``
    or ``None`` if the model response cannot be parsed as the expected JSON
    object. Raises on transient LLM errors so the caller can retry.
    """
    prompt = MULTI_HOP_DEPENDENCY_AND_ORACLE_PROMPT.format(
        num_spans=q.num_hops,
        answer_format_hint=answer_format_hint(q.reasoning_type, q.formula_kind),
        question=q.question,
        spans_block=_render_spans_block(list(q.source_spans)),
    )
    raw = await _call_completion(
        validator_model,
        prompt,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    parsed = _parse_multihop_response(raw)
    if parsed is None:
        return None
    sufficient_raw = parsed.get("sufficient_spans") or []
    sufficient_spans: list[int] = []
    if isinstance(sufficient_raw, list):
        for v in sufficient_raw:
            try:
                sufficient_spans.append(int(v))
            except (TypeError, ValueError):
                continue
    sufficient_spans = sorted(set(sufficient_spans))
    quotes_raw = parsed.get("supporting_quotes") or {}
    supporting_quotes: dict[str, str] = {}
    if isinstance(quotes_raw, dict):
        for k, v in quotes_raw.items():
            if isinstance(v, str):
                supporting_quotes[str(k)] = v
    reasoning = str(parsed.get("reasoning") or "")
    answer = str(parsed.get("answer") or "").strip()
    decomposable = 0 < len(sufficient_spans) < q.num_hops
    return decomposable, sufficient_spans, supporting_quotes, reasoning, answer


def _build_multihop_rejection_record(
    q: OpenEndedQuestion,
    *,
    reject_reason: str,
    reasoning: str,
    sufficient_spans: list[int],
    supporting_quotes: dict[str, str],
    llm_answer: str,
    oracle_judge_verdict: bool | None,
) -> dict[str, Any]:
    origin_label = "cross_doc_pair" if q.is_multi_doc else "same_doc_pair"
    return {
        "id": q.id,
        "question": q.question,
        "canonical_answer": q.canonical_answer,
        "answer_variants": list(q.answer_variants),
        "reasoning_type": q.reasoning_type,
        "origin": origin_label,
        "num_hops": q.num_hops,
        "reject_reason": reject_reason,
        "llm_reasoning": reasoning,
        "llm_sufficient_spans": sufficient_spans,
        "llm_supporting_quotes": supporting_quotes,
        "llm_answer": llm_answer,
        "oracle_judge_verdict": oracle_judge_verdict,
        "source_doc_ids": list(q.source_doc_ids),
        "source_chunk_ids": list(q.source_chunk_ids),
        "spans": [{"idx": i, "char_len": len(s), "text": s} for i, s in enumerate(q.source_spans)],
    }


async def gate_oracle_pass(
    questions: list[OpenEndedQuestion],
    *,
    validator_model: str,
    judge_model: str | None,
    concurrency: int = 10,
    cache_dir: Path | None = None,
) -> list[OpenEndedQuestion]:
    """Keep only questions a strong model answers correctly with both gold spans.

    Multi-hop questions (``num_hops >= 2``) go through a unified LLM call that
    judges decomposability and produces the oracle answer in one shot. Single-
    hop questions use the historical concatenated-context oracle path.
    """
    if not questions:
        return []
    sem = asyncio.Semaphore(concurrency)
    verdicts: list[bool] = [False] * len(questions)
    reject_reasons: list[str | None] = [None] * len(questions)
    multihop_rejections: list[dict[str, Any]] = []
    rejections_lock = asyncio.Lock()

    async def _record_multihop_rejection(record: dict[str, Any]) -> None:
        async with rejections_lock:
            multihop_rejections.append(record)

    async def _run_multihop(idx: int, q: OpenEndedQuestion) -> None:
        result = await _unified_multihop_check(q, validator_model)
        if result is None:
            logger.warning("multi-hop check: JSON parse failed for %s", q.id)
            verdicts[idx] = False
            reject_reasons[idx] = "parse_error"
            await _record_multihop_rejection(
                _build_multihop_rejection_record(
                    q,
                    reject_reason="parse_error",
                    reasoning="",
                    sufficient_spans=[],
                    supporting_quotes={},
                    llm_answer="",
                    oracle_judge_verdict=None,
                )
            )
            return
        decomposable, sufficient_spans, quotes, reasoning, pred = result
        if decomposable:
            verdicts[idx] = False
            reject_reasons[idx] = "decomposable"
            await _record_multihop_rejection(
                _build_multihop_rejection_record(
                    q,
                    reject_reason="decomposable",
                    reasoning=reasoning,
                    sufficient_spans=sufficient_spans,
                    supporting_quotes=quotes,
                    llm_answer=pred,
                    oracle_judge_verdict=None,
                )
            )
            return
        ok = await _judge_open_ended_answer(q, pred, judge_model)
        verdicts[idx] = ok
        if not ok:
            reject_reasons[idx] = "oracle_fail"
            await _record_multihop_rejection(
                _build_multihop_rejection_record(
                    q,
                    reject_reason="oracle_fail",
                    reasoning=reasoning,
                    sufficient_spans=sufficient_spans,
                    supporting_quotes=quotes,
                    llm_answer=pred,
                    oracle_judge_verdict=False,
                )
            )

    async def _run_single_hop(idx: int, q: OpenEndedQuestion) -> None:
        context = _ORACLE_SPAN_SEPARATOR.join(q.source_spans)
        hint = answer_format_hint(q.reasoning_type, q.formula_kind)
        pred = await _answer_question(
            validator_model,
            ORACLE_OPEN_ENDED_PROMPT,
            q.question,
            context,
            answer_format_hint=hint,
        )
        ok = await _judge_open_ended_answer(q, pred, judge_model)
        verdicts[idx] = ok
        if not ok:
            reject_reasons[idx] = "oracle_fail"

    async def _run(idx: int, q: OpenEndedQuestion) -> None:
        for attempt, cooldown in enumerate((0, *RETRY_COOLDOWNS_S), start=0):
            if cooldown:
                await asyncio.sleep(cooldown)
            try:
                async with sem:
                    if q.num_hops >= 2:
                        await _run_multihop(idx, q)
                    else:
                        await _run_single_hop(idx, q)
                return
            except Exception as exc:
                if not is_transient_llm_error(exc):
                    logger.info("oracle gate permanent error %s: %s", q.id, format_llm_error(exc))
                    verdicts[idx] = False
                    reject_reasons[idx] = "llm_error"
                    return
                if attempt == len(RETRY_COOLDOWNS_S):
                    logger.warning("oracle gate exhausted retries for %s: %s", q.id, format_llm_error(exc))
                    verdicts[idx] = False
                    reject_reasons[idx] = "llm_error"
                    return

    with tqdm(total=len(questions), desc="Validation", unit="q") as pbar:

        async def _bounded(idx: int, q: OpenEndedQuestion) -> None:
            await _run(idx, q)
            pbar.update(1)

        await asyncio.gather(*[_bounded(i, q) for i, q in enumerate(questions)])

    kept = [q for q, ok in zip(questions, verdicts, strict=True) if ok]

    sh_total = sh_kept = sh_oracle_fail = 0
    mh_total = mh_kept = mh_oracle_fail = mh_decomposable = mh_parse_error = 0
    kept_by_type: Counter[str] = Counter()
    total_by_type: Counter[str] = Counter()
    for q, ok, reason in zip(questions, verdicts, reject_reasons, strict=True):
        total_by_type[q.reasoning_type] += 1
        if ok:
            kept_by_type[q.reasoning_type] += 1
        if q.num_hops == 1:
            sh_total += 1
            if ok:
                sh_kept += 1
            elif reason == "oracle_fail":
                sh_oracle_fail += 1
        else:
            mh_total += 1
            if ok:
                mh_kept += 1
            elif reason == "oracle_fail":
                mh_oracle_fail += 1
            elif reason == "decomposable":
                mh_decomposable += 1
            elif reason == "parse_error":
                mh_parse_error += 1

    parts: list[str] = []
    total_oracle_fail = sh_oracle_fail + mh_oracle_fail
    if total_oracle_fail:
        parts.append(f"{total_oracle_fail} oracle couldn't answer")
    if mh_decomposable:
        parts.append(f"{mh_decomposable} multi-hop decomposable")
    if mh_parse_error:
        parts.append(f"{mh_parse_error} multi-hop parse error")
    rejected_segment = " · rejected: " + ", ".join(parts) if parts else ""

    logger.info(
        "Validation: %d/%d kept (single-hop %d/%d, multi-hop %d/%d)%s",
        len(kept),
        len(questions),
        sh_kept,
        sh_total,
        mh_kept,
        mh_total,
        rejected_segment,
    )
    if total_by_type:
        type_breakdown = ", ".join(f"{t}={kept_by_type[t]}/{total_by_type[t]}" for t in sorted(total_by_type.keys()))
        logger.info("By type kept: %s", type_breakdown)

    if cache_dir is not None and multihop_rejections:
        path = cache_dir / "multi_hop_rejections.json"
        payload = {
            "validator_model": validator_model,
            "judge_model": judge_model,
            "prompt_version": _MULTIHOP_PROMPT_VERSION,
            "rejections": multihop_rejections,
        }
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info(
                "Saved %d multi-hop rejections to %s",
                len(multihop_rejections),
                path.name,
            )
        except Exception:
            logger.warning("Failed to write multi-hop rejections file", exc_info=True)

    return kept


async def run_validation_pipeline(
    questions: list[OpenEndedQuestion],
    documents: dict[str, str],
    *,
    validator_model: str,
    judge_model: str | None = None,
    concurrency: int = 10,
    cache_dir: Path | None = None,
) -> list[OpenEndedQuestion]:
    """Oracle answerability gate.

    Source-fact verification runs upstream in ``ExamAgent.validate_compositions``;
    the multi-hop decomposability check now happens here, fused into the
    oracle call for multi-hop candidates. ``documents`` is unused but kept
    so the orchestrator has a single call site that owns the corpus map and
    the validator wiring. When ``cache_dir`` is supplied, multi-hop
    rejections are dumped to ``multi_hop_rejections.json`` there for audit.

    The discrimination dimension (was: ``gate_naive_rag_fail``) is handled
    downstream by the 4-probe filter in ``orchestrator._generate_exam``
    after this pipeline returns.
    """
    del documents  # unused — kept for orchestrator call-site stability
    return await gate_oracle_pass(
        questions,
        validator_model=validator_model,
        judge_model=judge_model,
        concurrency=concurrency,
        cache_dir=cache_dir,
    )


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
