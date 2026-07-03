"""Exam Agent — generates open-ended typed questions via per-neighborhood
LLM composition. One LLM call per neighborhood emits all the questions the
chunks support; each question cites the neighborhood positions it needs."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import re
import time as _time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
import numpy as np
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc.document import DoclingDocument
from tqdm import tqdm

from agentic_autorag.config.models import (
    QUESTION_TYPES,
    ExaminerConfig,
    OpenEndedQuestion,
)
from agentic_autorag.engine.section_classifier import (  # noqa: I001  (kept after docling imports)
    DEFAULT_ELIGIBLE_SECTIONS,
    SectionLabel,
    headings_to_label,
)
from agentic_autorag.examiner._errors import RETRY_COOLDOWNS_S, format_llm_error, is_transient_llm_error
from agentic_autorag.examiner.chunk_pair_index import ChunkRecord, Neighborhood
from agentic_autorag.examiner.composition_checks import check_selected_chunk_ids
from agentic_autorag.examiner.formula_verify import verify_formula
from agentic_autorag.examiner.neighborhoods import (
    NeighborhoodDiagnostic,
    build_neighborhood,
    build_tfidf_matrix,
)
from agentic_autorag.examiner.prompts import (
    COMPOSITION_BATCH_SYSTEM_PROMPT,
    COMPOSITION_BATCH_USER_PROMPT,
)
from agentic_autorag.examiner.seeders import _ANCHOR_MIN_TEXT_CHARS, emit_anchor_seeds
from agentic_autorag.litellm_runtime import acompletion_with_cost

logger = logging.getLogger(__name__)


class _WordCountTokenizer(BaseTokenizer):
    """Word-count tokenizer used by the examiner's HybridChunker.

    HybridChunker calls ``count_tokens`` for merge/split decisions and uses
    ``get_tokenizer`` as the callable handed to semchunk when an oversized
    section needs splitting. Counting whitespace-words rather than model
    tokens lets the user-facing chunk budget be specified directly in
    words — no model dependency, no token-per-word ratio.
    """

    max_tokens: int

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> Any:
        return self.count_tokens


def _build_examiner_chunker(max_chunk_words: int) -> HybridChunker:
    """Build a HybridChunker that merges/splits to a per-chunk word budget."""
    return HybridChunker(tokenizer=_WordCountTokenizer(max_tokens=max_chunk_words))


_DOC_TEXT_CHUNK_SEPARATOR = "\n\n"


def dl_doc_to_chunk_text(dl_doc: DoclingDocument, *, max_chunk_words: int) -> str:
    """Canonical text representation of a DoclingDocument.

    All coordinate-using subsystems (vector ``char_range`` from
    ``_chunk_docs_by_tokens``, graph-chunk lookup, span verifier
    ``source_span_offsets``) operate in this string's coordinate space.
    Composer spans (substrings of ``ChunkRecord.text``) are findable
    verbatim here because the chunker config is shared with
    ``chunk_documents`` via ``_build_examiner_chunker``.
    """
    chunker = _build_examiner_chunker(max_chunk_words)
    parts = [chunk.text for chunk in chunker.chunk(dl_doc=dl_doc) if chunk.text.strip()]
    return _DOC_TEXT_CHUNK_SEPARATOR.join(parts)


def dl_doc_to_index_text(dl_doc: DoclingDocument, *, max_chunk_words: int) -> str:
    """Retrieval-index text for a DoclingDocument: like ``dl_doc_to_chunk_text``
    but with each chunk's heading context prepended (via ``contextualize``), so
    section headers and the document title — high-signal retrieval terms the body
    often refers to only by pronoun — are embedded. Body-only
    ``dl_doc_to_chunk_text`` stays the coordinate frame for exam composition and
    span verification; this is used only for the parsed (PDF) retrieval index.
    """
    chunker = _build_examiner_chunker(max_chunk_words)
    parts: list[str] = []
    for chunk in chunker.chunk(dl_doc=dl_doc):
        contextualized = chunker.contextualize(chunk)
        if contextualized.strip():
            parts.append(contextualized)
    return _DOC_TEXT_CHUNK_SEPARATOR.join(parts)


def _greedy_merge_chunks(
    chunks: list[ChunkRecord],
    *,
    max_words: int,
) -> list[ChunkRecord]:
    """Greedy in-document chunk merging up to ``max_words``.

    HybridChunker emits one chunk per heading region, so a paper's
    author/affiliation block (~30 words, no heading) becomes its own chunk
    and gives the seeder nothing substantive to anchor on. This pass walks
    each document's chunks in original order and merges them into a
    running chunk while the running word count stays within budget; when
    adding the next chunk would exceed ``max_words`` the merged chunk is
    finalised and a new merge starts. The merged chunk inherits the first
    chunk's ``chunk_id`` and ``section``.
    """
    if not chunks or max_words <= 0:
        return list(chunks)

    out: list[ChunkRecord] = []
    for doc_id, group in itertools.groupby(chunks, key=lambda c: c.doc_id):
        doc_chunks = list(group)
        first = doc_chunks[0]
        merged_id = first.chunk_id
        merged_section = first.section
        merged_parts = [first.text]
        merged_words = len(first.text.split())
        for c in doc_chunks[1:]:
            c_words = len(c.text.split())
            if merged_words + c_words <= max_words:
                merged_parts.append(c.text)
                merged_words += c_words
            else:
                out.append(
                    ChunkRecord(
                        chunk_id=merged_id,
                        doc_id=doc_id,
                        text="\n\n".join(merged_parts),
                        section=merged_section,
                    )
                )
                merged_id = c.chunk_id
                merged_section = c.section
                merged_parts = [c.text]
                merged_words = c_words
        out.append(
            ChunkRecord(
                chunk_id=merged_id,
                doc_id=doc_id,
                text="\n\n".join(merged_parts),
                section=merged_section,
            )
        )
    return out


# Hard cap on canonical_answer length — matches H4 in the prompt. Words
# (whitespace-split) are tokenizer-independent and align with how a reader
# perceives length.
MAX_CANONICAL_WORDS = 15

# Strict regex bank that rejects question texts which proxy a source
# ("the document", "the study", ...).
SELF_CONTAINED_FILTERS = [
    re.compile(
        r'\b(documentation|paper|article|research|study|passage|text|excerpt)\b\s*"[^"]+"',
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(the\s+)?(above|given|provided|following)\s+(documentation|passage|text|excerpt|context)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\baccording\s+to\s+(the\s+)?(document|documentation|paper|article|passage|text|report|PDF|filing|contract)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^based\s+on\s+(the\s+)?(given\s+|provided\s+|above\s+)?"
        r"(text|passage|information|content|material|excerpt|context|document)",
        re.IGNORECASE,
    ),
    re.compile(r"\bin\s+the\s+(PDF|report|filing|contract|document)\b", re.IGNORECASE),
    re.compile(
        r"\b(in|within)\s+(this|that|these)\s+"
        r"(document|text|passage|report|study|article|paper|PDF|filing|contract|agreement|form)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfrom\s+(the|this)\s+"
        r"(document|text|passage|report|study|provided|given|following|attached|example|section|excerpt)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bas\s+(mentioned|discussed|stated|shown|noted|indicated|referenced|cited)\s+"
        r"(above|below|in\s+the\s+document|in\s+the\s+report|in\s+this\s+study)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(in|from)\s+(the\s+)?(following|preceding|previous|next|above|below)\s+"
        r"(section|paragraph|passage|example|excerpt|chapter|part|statement|clause|provision|text)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bthe\s+(?:study|research|trial|experiment|analysis|survey|review|"
        r"manuscript|findings|results|investigators?|authors?)(?:['’]s)?"
        r"\s*(?:[,;:.?!]|$)",
        re.IGNORECASE,
    ),
    # Internal scaffolding labels — the composer must never leak chunk
    # references into the question text.
    re.compile(
        r"\b(chunk\s*\d+|chunk_[a-z]|the\s+(first|second|third)\s+chunk|the\s+neighborhood)\b",
        re.IGNORECASE,
    ),
]


@dataclass
class CompositionResult:
    """One question (or refusal) emitted by the composer for a neighborhood.

    Each composer call emits a list of these — the composer is asked to
    produce as many questions as the chunks genuinely support.

    ``selected_chunk_ids`` are positions into ``neighborhood.chunks``
    (the same positions the composer saw as ``[Chunk N]`` labels).
    ``source_spans`` are aligned with ``selected_chunk_ids`` — one
    verbatim excerpt per cited chunk, in the same order.

    ``rejection_explanation`` is populated only when ``linkable=False``
    (the composer refused). ``reason`` is set by the harness for
    non-LLM rejection causes (parse errors, structural-check failures,
    transient LLM failures).
    """

    neighborhood: Neighborhood
    linkable: bool
    reasoning_type: str = "bridge"
    question: str = ""
    canonical_answer: str = ""
    answer_variants: list[str] = field(default_factory=list)
    selected_chunk_ids: list[int] = field(default_factory=list)
    source_spans: list[str] = field(default_factory=list)
    formula: str | None = None
    formula_kind: str | None = None
    rejection_explanation: str = ""
    reason: str = ""


def self_containment_failure(question_text: str) -> tuple[int, str] | None:
    """Return (pattern_index, matched_snippet) for the first failing filter, or None."""
    for idx, pattern in enumerate(SELF_CONTAINED_FILTERS):
        m = pattern.search(question_text)
        if m:
            return idx, m.group(0)
    return None


@dataclass
class _CompositionLogDiagnostic:
    """Per-corpus TF-IDF data needed to log top shared terms per chunk.

    Cached on the agent at corpus-prep time and consumed by
    ``_record_composition_call`` to attribute why each neighborhood
    member was picked.
    """

    tfidf: Any  # scipy.sparse.csr_matrix
    vocab: np.ndarray
    df_fraction: np.ndarray
    chunk_id_to_row: dict[str, int]
    diagnostics_by_anchor: dict[str, NeighborhoodDiagnostic]


_TOP_SHARED_TERMS_PER_CHUNK = 10


def _top_shared_terms(
    tfidf: Any,
    vocab: np.ndarray,
    df_fraction: np.ndarray,
    centroid: np.ndarray,
    chunk_row_idx: int,
    n: int = _TOP_SHARED_TERMS_PER_CHUNK,
) -> list[dict[str, Any]]:
    """Top-N terms by contribution to cosine(centroid, chunk_tfidf).

    Each entry's ``mass`` is the per-term elementwise product
    ``centroid[term] * chunk_tfidf[term]`` — the sum of these IS the
    cosine sim that drove the pick. ``df_fraction`` is the share of
    corpus chunks containing the term (compare against TfidfVectorizer
    ``max_df`` to judge whether the term is too common to be useful as
    a bridge signal). Returned in descending mass order.
    """
    row = tfidf[chunk_row_idx]
    cols = row.indices
    vals = row.data
    contributions = vals * centroid[cols]
    order = np.argsort(-contributions)[:n]
    return [
        {
            "term": str(vocab[cols[i]]),
            "mass": float(contributions[i]),
            "df_fraction": float(df_fraction[cols[i]]),
        }
        for i in order
        if contributions[i] > 0
    ]


@dataclass
class PreparedCorpus:
    """Result of one-time corpus preparation for exam generation.

    ``composition_results`` carries every per-question outcome from the
    most recent ``generate_exam`` run, including LLM refusals — the
    orchestrator persists those alongside accepted candidates so the
    user can audit why the LLM declined to compose a question.
    """

    chunks: list[ChunkRecord]
    neighborhoods: list[Neighborhood] = field(default_factory=list)
    composition_results: list[CompositionResult] = field(default_factory=list)


class ExamAgent:
    """Generates open-ended typed candidate questions via agentic composition.

    The agent is stateless across runs — each call to ``generate_exam``
    builds the chunk embeddings and neighborhoods fresh. A caller can
    also invoke ``prepare_corpus`` once and reuse the resulting
    ``PreparedCorpus`` across multiple composition passes (e.g.
    backfilling when too few candidates survive validation).
    """

    def __init__(
        self,
        config: ExaminerConfig,
        examiner_model: str,
        corpus_description: str = "",
        temperature: float | None = None,
        concurrency: int = 10,
        anchor_sampler_seed: int | str | None = None,
        reasoning_effort: str | None = None,
        composition_log_path: Path | None = None,
        span_verification_report_path: Path | None = None,
    ) -> None:
        self.config = config
        self.examiner_model = examiner_model
        self._supports_prompt_caching = _model_supports_prompt_caching(examiner_model)
        self.corpus_description = corpus_description or "General enterprise documents."
        self.temperature = temperature if temperature is not None else config.composition_temperature
        self.concurrency = concurrency
        # Reproducible anchor sampler — fed by the orchestrator from
        # project_name so the same corpus produces the same anchor set
        # across runs (tests pass an int directly).
        self._anchor_seed = anchor_sampler_seed
        self._reasoning_effort = _resolve_reasoning_effort(examiner_model, reasoning_effort)
        if self._reasoning_effort is not None:
            logger.info("ExamAgent using reasoning_effort=%s on %s", self._reasoning_effort, examiner_model)
        # Surfaced for the orchestrator's empty-exam guard so it can include
        # the top rejection reasons in ExamGenerationFailed. Populated by
        # ``_compositions_to_questions`` on every call.
        self.last_composition_rejections: Counter[str] = Counter()
        # Per-rejection records for post-LLM filter failures
        # (self_contained, structural_check, empty_span, formula_*,
        # pydantic_validation). Mirrors what the orchestrator already
        # persists for LLM refusals (linkable=False), so candidates.json
        # carries every rejection cause uniformly. Repopulated each call.
        self.last_downstream_rejections: list[dict] = []
        # When set, every composition LLM call (input neighborhood + raw
        # response + TF-IDF diagnostics) is accumulated in memory and dumped as
        # a pretty JSON array at the end of each ``compose_multihop_batched``
        # pass. This is the primary artifact for auditing composer behavior
        # (including the parsed-but-unstored ``reasoning`` field). The
        # orchestrator points it at ``output_dir/debug/composition_log.json``
        # when ``ExaminerConfig.save_debug_artifacts`` is on; None disables it.
        self._composition_log_path = composition_log_path
        self._composition_log_records: list[dict[str, Any]] = []
        self._composition_log_diag: _CompositionLogDiagnostic | None = None
        # When set, ``verify_source_facts`` writes a per-question span-
        # verification report here for diagnosing the source-span rejection
        # rate. None disables it.
        self._span_verification_report_path = span_verification_report_path

    def chunk_documents(
        self,
        documents: list[DoclingDocument],
        doc_ids: list[str],
    ) -> list[ChunkRecord]:
        """Chunk Docling documents with section-aware boundaries."""
        if len(documents) != len(doc_ids):
            raise ValueError(f"documents ({len(documents)}) and doc_ids ({len(doc_ids)}) must align")

        chunker = _build_examiner_chunker(self.config.max_chunk_words)
        min_words = self.config.min_doc_words

        chunks: list[ChunkRecord] = []
        for dl_doc, doc_id in zip(documents, doc_ids, strict=True):
            doc_chunks: list[ChunkRecord] = []
            doc_word_count = 0
            for i, dc in enumerate(chunker.chunk(dl_doc=dl_doc)):
                text = dc.text.strip()
                if not text:
                    continue
                doc_word_count += len(text.split())
                doc_chunks.append(
                    ChunkRecord(
                        chunk_id=f"{doc_id}::chunk_{i}",
                        doc_id=doc_id,
                        text=text,
                        section=headings_to_label(dc.meta.headings),
                    )
                )
            if doc_word_count >= min_words:
                chunks.extend(doc_chunks)
        return chunks

    def prepare_corpus(
        self,
        documents: list[DoclingDocument],
        doc_ids: list[str],
        *,
        eligible_sections: frozenset[SectionLabel] | None = DEFAULT_ELIGIBLE_SECTIONS,
    ) -> PreparedCorpus:
        """Build chunks → sample anchors → expand each anchor into a neighborhood.

        Anchor count is ``exam_size * initial_question_multiplier``;
        neighborhoods are grown adaptively per ``neighborhood_min_chunks``
        and ``neighborhood_min_words`` (whichever floor is reached first),
        split between same-document siblings and centroid-cosine-similar
        cross-document chunks per the normalized
        ``neighborhood_{same,cross}_doc_weight`` mix.
        """
        chunks = self.chunk_documents(documents, doc_ids)
        if eligible_sections is None:
            eligible_chunks = chunks
        else:
            eligible_chunks = [c for c in chunks if c.section is None or c.section in eligible_sections]
            if len(eligible_chunks) < len(chunks):
                logger.info(
                    "Section filter: %d/%d chunks eligible (dropped %d in excluded sections)",
                    len(eligible_chunks),
                    len(chunks),
                    len(chunks) - len(eligible_chunks),
                )

        pre_merge_count = len(eligible_chunks)
        eligible_chunks = _greedy_merge_chunks(eligible_chunks, max_words=self.config.max_chunk_words)
        if eligible_chunks:
            chunk_word_counts = np.asarray([len(c.text.split()) for c in eligible_chunks])
            logger.info(
                "Greedy-merged eligible chunks: %d → %d chunks "
                "(max_chunk_words=%d; words/chunk min=%d p25=%d median=%d p75=%d max=%d mean=%.0f; "
                "%d chunks <%d chars [anchor-eligibility floor])",
                pre_merge_count,
                len(eligible_chunks),
                self.config.max_chunk_words,
                int(chunk_word_counts.min()),
                int(np.percentile(chunk_word_counts, 25)),
                int(np.median(chunk_word_counts)),
                int(np.percentile(chunk_word_counts, 75)),
                int(chunk_word_counts.max()),
                float(chunk_word_counts.mean()),
                sum(1 for c in eligible_chunks if len(c.text) < _ANCHOR_MIN_TEXT_CHARS),
                _ANCHOR_MIN_TEXT_CHARS,
            )
        else:
            logger.info(
                "Greedy-merged eligible chunks: %d → 0 chunks (max_chunk_words=%d)",
                pre_merge_count,
                self.config.max_chunk_words,
            )

        target_anchor_count = max(
            1,
            int(self.config.exam_size * self.config.initial_question_multiplier),
        )
        anchors = emit_anchor_seeds(
            eligible_chunks,
            target_count=target_anchor_count,
            rng_seed=self._anchor_seed,
        )
        if not anchors:
            logger.warning(
                "Prepared corpus: 0 anchors from %d eligible chunks — corpus may be too short",
                len(eligible_chunks),
            )
            return PreparedCorpus(chunks=chunks, neighborhoods=[])

        # Build the neighborhood index: cross-doc cosine retrieval needs
        # a TF-IDF matrix over the full eligible-chunk text. TF-IDF
        # surfaces chunks sharing rare distinctive vocabulary with the
        # anchor — the multi-hop bridge signal that dense-embedding kNN
        # collapses into the trial-time retrieval signal.
        t_tfidf = _time.perf_counter()
        tfidf, vectorizer = build_tfidf_matrix(eligible_chunks)
        logger.info(
            "Built TF-IDF matrix (n_chunks=%d, vocab=%d) in %.1fs",
            tfidf.shape[0],
            tfidf.shape[1],
            _time.perf_counter() - t_tfidf,
        )

        chunk_id_to_index = {c.chunk_id: i for i, c in enumerate(eligible_chunks)}
        neighborhoods: list[Neighborhood] = []
        nh_diagnostics: dict[str, NeighborhoodDiagnostic] = {}
        nh_sizes: list[int] = []
        nh_words: list[int] = []
        for anchor in anchors:
            anchor_idx = chunk_id_to_index[anchor.chunk.chunk_id]
            nh, diag = build_neighborhood(
                anchor_idx,
                eligible_chunks,
                tfidf,
                min_chunks=self.config.neighborhood_min_chunks,
                min_words=self.config.neighborhood_min_words,
                same_doc_weight=self.config.neighborhood_same_doc_weight,
                cross_doc_weight=self.config.neighborhood_cross_doc_weight,
            )
            neighborhoods.append(nh)
            nh_diagnostics[anchor.chunk.chunk_id] = diag
            nh_sizes.append(len(nh.chunks))
            nh_words.append(sum(len(c.text.split()) for c in nh.chunks))

        if self._composition_log_path is not None:
            n_chunks = max(tfidf.shape[0], 1)
            doc_freq = np.asarray((tfidf > 0).sum(axis=0)).ravel()
            self._composition_log_diag = _CompositionLogDiagnostic(
                tfidf=tfidf,
                vocab=vectorizer.get_feature_names_out(),
                df_fraction=doc_freq / n_chunks,
                chunk_id_to_row=chunk_id_to_index,
                diagnostics_by_anchor=nh_diagnostics,
            )

        if neighborhoods:
            sizes = np.asarray(nh_sizes)
            words = np.asarray(nh_words)
            total_weight = self.config.neighborhood_same_doc_weight + self.config.neighborhood_cross_doc_weight
            same_ratio = self.config.neighborhood_same_doc_weight / total_weight
            logger.info(
                "Built %d neighborhoods (chunks/nh: min=%d median=%d max=%d; "
                "words/nh: min=%d median=%d max=%d; same_doc_ratio=%.2f)",
                len(neighborhoods),
                int(sizes.min()),
                int(np.median(sizes)),
                int(sizes.max()),
                int(words.min()),
                int(np.median(words)),
                int(words.max()),
                same_ratio,
            )

        logger.info(
            "Prepared corpus: %d raw chunks, %d after section-filter + merge, %d anchors → %d neighborhoods",
            len(chunks),
            len(eligible_chunks),
            len(anchors),
            len(neighborhoods),
        )
        return PreparedCorpus(chunks=chunks, neighborhoods=neighborhoods)

    async def compose_multihop_batched(self, neighborhoods: list[Neighborhood]) -> list[CompositionResult]:
        """Run one composition LLM call per neighborhood, concurrent up to ``concurrency``.

        Each call emits a list of ``CompositionResult`` — one per
        question the composer chose to produce (there is no upper cap).
        Per-call parse failures don't poison the rest — malformed entries
        are logged and skipped.
        """
        if not neighborhoods:
            return []

        if self._composition_log_path is not None:
            self._composition_log_records = []

        sem = asyncio.Semaphore(self.concurrency)

        results: list[CompositionResult] = []
        results_lock = asyncio.Lock()
        latencies: list[float] = []
        latency_lock = asyncio.Lock()

        async def _process_neighborhood(nh: Neighborhood) -> None:
            t0 = asyncio.get_event_loop().time()
            async with sem:
                nh_results = await self._compose_one_neighborhood(nh)
            elapsed = asyncio.get_event_loop().time() - t0
            async with results_lock:
                results.extend(nh_results)
            async with latency_lock:
                latencies.append(elapsed)

        try:
            with tqdm(total=len(neighborhoods), desc="Composing typed questions", unit="nh") as pbar:

                async def _bounded(nh: Neighborhood) -> None:
                    await _process_neighborhood(nh)
                    pbar.update(1)

                await asyncio.gather(*[_bounded(nh) for nh in neighborhoods])
        finally:
            self._flush_composition_log()

        if latencies:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            p50 = sorted_lat[n // 2]
            p95 = sorted_lat[min(n - 1, int(n * 0.95))]
            mean_lat = sum(sorted_lat) / n
            logger.info(
                "Composition latency: n=%d neighborhoods, mean=%.1fs p50=%.1fs p95=%.1fs",
                n,
                mean_lat,
                p50,
                p95,
            )

        return results

    async def _compose_one_neighborhood(self, nh: Neighborhood) -> list[CompositionResult]:
        """Compose one neighborhood with retries on transient LLM errors."""
        for attempt, cooldown in enumerate((0, *RETRY_COOLDOWNS_S), start=0):
            if cooldown:
                await asyncio.sleep(cooldown)
            try:
                raw = await self._call_composition_llm(nh)
                return self._parse_composition_neighborhood(raw, nh)
            except Exception as exc:
                if not is_transient_llm_error(exc):
                    logger.info("Neighborhood composition failed permanently: %s", format_llm_error(exc))
                    return [
                        CompositionResult(
                            neighborhood=nh,
                            linkable=False,
                            reason="composition_error",
                        )
                    ]
                if attempt == len(RETRY_COOLDOWNS_S):
                    logger.warning(
                        "Neighborhood composition exhausted retries (%s)",
                        format_llm_error(exc),
                    )
                    return [
                        CompositionResult(
                            neighborhood=nh,
                            linkable=False,
                            reason="composition_transient",
                        )
                    ]
        raise AssertionError("unreachable: retry loop must return")

    async def _call_composition_llm(self, nh: Neighborhood) -> str:
        chunk_blocks = []
        for pos, chunk in enumerate(nh.chunks):
            chunk_blocks.append(f"[Chunk {pos}] (doc_id={chunk.doc_id}, chunk_id={chunk.chunk_id})\n{chunk.text}")
        user = COMPOSITION_BATCH_USER_PROMPT.format(
            domain_description=self.corpus_description,
            anchor_chunk_id=nh.anchor.chunk_id,
            chunk_blocks="\n\n".join(chunk_blocks),
        )
        # The system prompt is stable across every composition call in a run.
        # Cache it on providers that support prompt caching; others get a plain
        # string (a cache_control block is only valid for caching providers).
        if self._supports_prompt_caching:
            system_content: Any = [
                {
                    "type": "text",
                    "text": COMPOSITION_BATCH_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                },
            ]
        else:
            system_content = COMPOSITION_BATCH_SYSTEM_PROMPT
        kwargs: dict = {
            "model": self.examiner_model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "num_retries": 0,
        }
        if self._reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._reasoning_effort
        response, _ = await acompletion_with_cost(cost_category="exam_generation", **kwargs)
        raw = response.choices[0].message.content or ""
        if self._composition_log_path is not None:
            self._record_composition_call(nh, raw)
        return raw

    def _record_composition_call(self, nh: Neighborhood, raw_response: str) -> None:
        """Capture one composition call into the in-memory log accumulator."""
        try:
            response: Any = json.loads(raw_response)
        except (ValueError, TypeError):
            response = raw_response

        diag = self._composition_log_diag
        nh_diag = diag.diagnostics_by_anchor.get(nh.anchor.chunk_id) if diag is not None else None

        neighborhood_entries: list[dict[str, Any]] = []
        for pos, c in enumerate(nh.chunks):
            entry: dict[str, Any] = {
                "pos": pos,
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
            }
            if diag is not None and nh_diag is not None:
                row_idx = diag.chunk_id_to_row.get(c.chunk_id)
                if row_idx is not None:
                    entry["position_kind"] = nh_diag.position_kinds[pos]
                    entry["top_shared_terms"] = _top_shared_terms(
                        diag.tfidf, diag.vocab, diag.df_fraction, nh_diag.centroid, row_idx
                    )
            neighborhood_entries.append(entry)

        self._composition_log_records.append(
            {
                "ts": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                "anchor_chunk_id": nh.anchor.chunk_id,
                "neighborhood": neighborhood_entries,
                "response": response,
            }
        )

    def _flush_composition_log(self) -> None:
        """Write the accumulated composition log to a pretty JSON file."""
        if self._composition_log_path is None or not self._composition_log_records:
            return
        self._composition_log_path.write_text(
            json.dumps(self._composition_log_records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _parse_composition_neighborhood(
        self,
        raw: str,
        nh: Neighborhood,
    ) -> list[CompositionResult]:
        """Parse a JSON array of question entries emitted for one neighborhood.

        Each entry is either an accepted question (with selected_chunk_ids,
        question, canonical_answer, source_spans, etc.) or a refusal
        (linkable=False, explanation). The composer is asked to emit a
        single refusal entry only when the entire neighborhood is unusable.
        """
        text = _strip_markdown_fences(raw)
        items = _try_parse_json_array(text)
        if items is None:
            items = _extract_json_array(text)
        if items is None:
            logger.info("Neighborhood %s composition parse failed; raw=%.200s", nh.anchor.chunk_id, raw)
            return [CompositionResult(neighborhood=nh, linkable=False, reason="parse_error")]

        valid_types = set(QUESTION_TYPES)
        out: list[CompositionResult] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue

            linkable_raw = entry.get("linkable", False)
            if not isinstance(linkable_raw, bool):
                linkable_raw = bool(linkable_raw)

            if not linkable_raw:
                explanation = str(entry.get("explanation") or entry.get("reason") or "llm_marked_not_linkable")[:480]
                out.append(
                    CompositionResult(
                        neighborhood=nh,
                        linkable=False,
                        rejection_explanation=explanation,
                    )
                )
                continue

            try:
                question = str(entry["question"]).strip()
                canonical = str(entry["canonical_answer"]).strip()
            except (KeyError, TypeError) as exc:
                logger.info("Neighborhood %s entry missing required fields: %s", nh.anchor.chunk_id, exc)
                out.append(
                    CompositionResult(
                        neighborhood=nh,
                        linkable=False,
                        reason="missing_fields",
                    )
                )
                continue

            if len(canonical.split()) > MAX_CANONICAL_WORDS:
                out.append(
                    CompositionResult(
                        neighborhood=nh,
                        linkable=False,
                        reason="answer_too_long",
                    )
                )
                continue

            reported_type = str(entry.get("reasoning_type") or "").strip()
            if reported_type not in valid_types:
                # Default to bridge — the most common multi-hop type. The
                # downstream gates will reject if the shape doesn't fit.
                reported_type = "bridge"

            variants_raw = entry.get("answer_variants", []) or []
            if isinstance(variants_raw, str):
                variants = [variants_raw]
            elif isinstance(variants_raw, list):
                variants = [str(v).strip() for v in variants_raw if isinstance(v, str) and v.strip()]
            else:
                variants = []

            # New schema: cited_chunks = [{"chunk_id": int, "span": str}, ...].
            # Co-located so the composer cannot drop spans independently of
            # citations. Falls back to the legacy flat shape
            # (selected_chunk_ids + source_spans) for one transition cycle so
            # cached composition_log.json files can still be replayed.
            selected_chunk_ids: list[int] = []
            source_spans: list[str] = []
            cited_raw = entry.get("cited_chunks")
            if isinstance(cited_raw, list):
                for obj in cited_raw:
                    if not isinstance(obj, dict):
                        continue
                    chunk_id_raw = obj.get("chunk_id")
                    span_raw = obj.get("span")
                    try:
                        chunk_id = int(chunk_id_raw)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(span_raw, str):
                        continue
                    selected_chunk_ids.append(chunk_id)
                    source_spans.append(span_raw.strip())
            else:
                selected_raw = entry.get("selected_chunk_ids", []) or []
                if isinstance(selected_raw, list):
                    for v in selected_raw:
                        with contextlib.suppress(TypeError, ValueError):
                            selected_chunk_ids.append(int(v))
                spans_raw = entry.get("source_spans", []) or []
                if isinstance(spans_raw, list):
                    source_spans = [str(s).strip() for s in spans_raw]

            formula_raw = entry.get("formula")
            formula = str(formula_raw).strip() if isinstance(formula_raw, str) and formula_raw.strip() else None
            formula_kind_raw = entry.get("formula_kind")
            formula_kind: str | None = None
            if isinstance(formula_kind_raw, str) and formula_kind_raw.strip():
                kind = formula_kind_raw.strip()
                if kind == "arithmetic":
                    formula_kind = kind
            if formula is None or formula_kind is None:
                formula = None
                formula_kind = None

            out.append(
                CompositionResult(
                    neighborhood=nh,
                    linkable=True,
                    reasoning_type=reported_type,
                    question=question,
                    canonical_answer=canonical,
                    answer_variants=variants,
                    selected_chunk_ids=selected_chunk_ids,
                    source_spans=source_spans,
                    formula=formula,
                    formula_kind=formula_kind,
                )
            )
        if not out:
            out.append(CompositionResult(neighborhood=nh, linkable=False, reason="empty_response"))
        return out

    async def generate_exam(
        self,
        documents: list[DoclingDocument],
        doc_ids: list[str],
        *,
        eligible_sections: frozenset[SectionLabel] | None = DEFAULT_ELIGIBLE_SECTIONS,
        doc_text_map: dict[str, str] | None = None,
        source_fact_verify_fuzzy_threshold: float = 0.9,
    ) -> tuple[list[OpenEndedQuestion], PreparedCorpus]:
        """Convenience wrapper: prepare corpus → compose → span verify."""
        corpus = self.prepare_corpus(documents, doc_ids, eligible_sections=eligible_sections)
        if not corpus.neighborhoods:
            return [], corpus

        composition_results = await self.compose_multihop_batched(corpus.neighborhoods)
        corpus.composition_results = composition_results
        questions = await self.validate_compositions(
            composition_results,
            documents=doc_text_map,
            source_fact_verify_fuzzy_threshold=source_fact_verify_fuzzy_threshold,
        )
        return questions, corpus

    async def validate_compositions(
        self,
        composition_results: list[CompositionResult],
        *,
        documents: dict[str, str] | None = None,
        source_fact_verify_fuzzy_threshold: float = 0.9,
    ) -> list[OpenEndedQuestion]:
        """Run the post-composition filters: structural → typed gates → source spans."""
        questions = self._compositions_to_questions(composition_results)
        if not questions:
            return []
        if documents is not None:
            # Lazy import to avoid the exam_validator ↔ exam_agent circular
            # (exam_validator imports `_call_completion` from this module).
            from agentic_autorag.examiner.exam_validator import verify_source_facts

            questions = verify_source_facts(
                questions,
                documents,
                fuzzy_threshold=source_fact_verify_fuzzy_threshold,
                report_path=self._span_verification_report_path,
            )
        return questions

    def _compositions_to_questions(self, results: list[CompositionResult]) -> list[OpenEndedQuestion]:
        """Convert composition results into validated ``OpenEndedQuestion``s.

        Pipeline (in order, per result):
          - linkable filter (LLM refusals + harness errors)
          - structural check (selected_chunk_ids in range, non-empty, unique)
          - source_spans aligned to selected_chunk_ids
          - self-containment regex check
          - non-empty source spans
          - numeric formula verification (if reasoning_type in {"numeric", "numeric_single"})

        Tracks rejections in a single ``Counter`` keyed by reason and a
        per-cause sample for diagnostic logging. Returns the questions
        that survive every gate.
        """
        kept: list[OpenEndedQuestion] = []
        reasons: Counter[str] = Counter()
        sample_rejections: dict[str, list[str]] = {}
        type_counts: Counter[str] = Counter()
        self.last_downstream_rejections = []

        def _reject(reason: str, *, sample: str = "", record: dict | None = None) -> None:
            reasons[reason] += 1
            if sample and len(sample_rejections.get(reason, [])) < 3:
                sample_rejections.setdefault(reason, []).append(sample)
            if record is not None:
                self.last_downstream_rejections.append(record)

        for i, r in enumerate(results, start=1):
            nh = r.neighborhood
            anchor_id = nh.anchor.chunk_id
            if not r.linkable:
                code = r.reason or ("llm_refused" if r.rejection_explanation else "unspecified")
                _reject(
                    code,
                    sample=f"{anchor_id} :: {r.rejection_explanation[:200]}" if code == "llm_refused" else anchor_id,
                )
                continue

            structural = check_selected_chunk_ids(r.selected_chunk_ids, r.source_spans, len(nh.chunks))
            if not structural.ok:
                _reject(
                    structural.reason,
                    sample=f"{anchor_id} :: selected={r.selected_chunk_ids} nh_size={len(nh.chunks)}",
                    record={
                        "anchor_chunk_id": anchor_id,
                        "reason": structural.reason,
                        "question": r.question,
                        "selected_chunk_ids": r.selected_chunk_ids,
                        "source_spans": r.source_spans,
                        "neighborhood_size": len(nh.chunks),
                    },
                )
                continue

            sc_fail = self_containment_failure(r.question)
            if sc_fail is not None:
                _reject(
                    "self_contained",
                    sample=f"{anchor_id} :: {r.question[:160]}",
                    record={
                        "anchor_chunk_id": anchor_id,
                        "reason": "self_contained",
                        "question": r.question,
                        "matched_phrase": sc_fail[1],
                    },
                )
                logger.info("self-contained-fail: %r", sc_fail[1])
                continue

            if any(not s.strip() for s in r.source_spans):
                _reject(
                    "empty_span",
                    sample=f"{anchor_id} :: {r.question[:160]}",
                    record={
                        "anchor_chunk_id": anchor_id,
                        "reason": "empty_span",
                        "question": r.question,
                    },
                )
                continue

            if r.reasoning_type in ("numeric", "numeric_single"):
                if not r.formula or not r.formula_kind:
                    _reject(
                        "formula_missing",
                        sample=f"{anchor_id} :: {r.question[:120]} -> {r.canonical_answer}",
                        record={
                            "anchor_chunk_id": anchor_id,
                            "reason": "formula_missing",
                            "question": r.question,
                            "canonical_answer": r.canonical_answer,
                        },
                    )
                    logger.info(
                        "%s question missing formula: q=%r answer=%r",
                        r.reasoning_type,
                        r.question[:120],
                        r.canonical_answer,
                    )
                    continue
                if not verify_formula(r.formula, r.formula_kind, r.canonical_answer):
                    _reject(
                        "formula_mismatch",
                        sample=f"{anchor_id} :: formula={r.formula!r} answer={r.canonical_answer!r}",
                        record={
                            "anchor_chunk_id": anchor_id,
                            "reason": "formula_mismatch",
                            "question": r.question,
                            "canonical_answer": r.canonical_answer,
                            "formula": r.formula,
                            "formula_kind": r.formula_kind,
                        },
                    )
                    continue

            source_doc_ids = [nh.chunks[idx].doc_id for idx in r.selected_chunk_ids]

            try:
                question = OpenEndedQuestion(
                    id=f"C{i:04d}",
                    question=r.question,
                    canonical_answer=r.canonical_answer,
                    answer_variants=r.answer_variants,
                    reasoning_type=r.reasoning_type,
                    source_doc_ids=source_doc_ids,
                    source_spans=list(r.source_spans),
                    formula=r.formula,
                    formula_kind=r.formula_kind,
                )
            except Exception as exc:  # noqa: BLE001
                _reject(
                    "pydantic_validation",
                    sample=f"{anchor_id} :: {exc}",
                    record={
                        "anchor_chunk_id": anchor_id,
                        "reason": "pydantic_validation",
                        "question": r.question,
                        "error": str(exc),
                    },
                )
                logger.info("OpenEndedQuestion validation failed: %s", exc)
                continue
            kept.append(question)
            type_counts[r.reasoning_type] += 1

        self.last_composition_rejections = Counter(reasons)
        if type_counts:
            actual_line = ", ".join(f"{t}={type_counts.get(t, 0)}" for t in QUESTION_TYPES if t in type_counts)
            logger.info("Per-actual-type kept counts: %s", actual_line)
        # Hop-count distribution: cited chunks per kept question.
        if kept:
            hop_counts = Counter(q.num_hops for q in kept)
            hop_breakdown = ", ".join(f"{hops}-hop={n}" for hops, n in sorted(hop_counts.items()))
            mean_hops = sum(h * n for h, n in hop_counts.items()) / sum(hop_counts.values())
            logger.info("Hop distribution: %s (mean=%.2f)", hop_breakdown, mean_hops)
        return kept


# --- helpers ---------------------------------------------------------------


def _model_supports_prompt_caching(model: str) -> bool:
    """Whether ``model`` accepts Anthropic-style ``cache_control`` blocks.

    Driven by LiteLLM's capability catalog. Unknown models — and any lookup
    error — default to False so the examiner sends a plain system prompt
    instead of a ``cache_control`` block a non-caching provider would reject.
    """
    try:
        return bool(litellm.utils.supports_prompt_caching(model=model))
    except Exception:
        return False


def _resolve_reasoning_effort(model: str, effort: str | None) -> str | None:
    """Return ``effort`` if the model supports reasoning, else None."""
    if not effort:
        return None
    try:
        supported = bool(litellm.supports_reasoning(model=model))
    except Exception:
        supported = True
    return effort if supported else None


async def _call_completion(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    cost_category: str = "exam_generation",
    response_format: dict | None = None,
) -> str:
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "num_retries": 0,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if response_format is not None:
        kwargs["response_format"] = response_format
    response, _ = await acompletion_with_cost(cost_category=cost_category, **kwargs)
    return response.choices[0].message.content or ""


def _strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def _try_parse_json_array(text: str) -> list[dict] | None:
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return None


def _extract_json_array(text: str) -> list[dict] | None:
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    data = json.loads(cleaned)
                except json.JSONDecodeError:
                    return None
                if isinstance(data, list):
                    return [item for item in data if isinstance(item, dict)]
                return None
    return None


__all__ = [
    "CompositionResult",
    "ExamAgent",
    "PreparedCorpus",
]
