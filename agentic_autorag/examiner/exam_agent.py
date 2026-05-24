"""Exam Agent — generates open-ended 2-hop questions via batched LLM composition.

Pipeline:
  1. Chunk the corpus into ``ChunkRecord``s and label sections.
  2. ``embedding_pair_index.emit_embedding_pairs()`` embeds every eligible
     chunk and emits cross-doc seeds via per-chunk top-K + round-robin
     selection over rank.
  3. ``ExamAgent.compose_multihop_batched()`` packs K seeds per LLM call and
     parses out per-seed outcomes — either a candidate question or a
     refusal with a free-text explanation.

The validation gates (oracle answerability + multi-hop decomposability, fused
into one LLM call for multi-hop candidates) live in ``exam_validator``; this
module hands off candidates that have cleared composition + span verification.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import random
import re
import time as _time
from collections import Counter
from collections.abc import Callable
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
    MULTI_HOP_QUESTION_TYPES,
    QUESTION_TYPES,
    SINGLE_HOP_QUESTION_TYPES,
    ExaminerConfig,
    OpenEndedQuestion,
)
from agentic_autorag.engine.section_classifier import (  # noqa: I001  (kept after docling imports)
    DEFAULT_ELIGIBLE_SECTIONS,
    SectionLabel,
    headings_to_label,
)
from agentic_autorag.examiner._errors import RETRY_COOLDOWNS_S, format_llm_error, is_transient_llm_error
from agentic_autorag.examiner.chunk_pair_index import ChunkRecord, Seed
from agentic_autorag.examiner.embedding_pair_index import make_pair_embedder
from agentic_autorag.examiner.formula_verify import verify_formula
from agentic_autorag.examiner.prompts import (
    COMPOSITION_BATCH_SYSTEM_PROMPT,
    COMPOSITION_BATCH_USER_PROMPT,
    answer_format_hint,
)
from agentic_autorag.examiner.seeders import (
    emit_cross_doc_pair_seeds,
    emit_same_doc_pair_seeds,
    emit_single_chunk_seeds,
)
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


def _greedy_merge_chunks(
    chunks: list[ChunkRecord],
    *,
    max_words: int,
) -> list[ChunkRecord]:
    """Greedy in-document chunk merging up to ``max_words``.

    HybridChunker emits one chunk per heading region, so a paper's
    author/affiliation block (~30 words, no heading) becomes its own chunk
    and gives the single-chunk seeder nothing substantive to ask about.
    This pass walks each document's chunks in original order and merges
    them into a running chunk while the running word count stays within
    budget; when adding the next chunk would exceed ``max_words`` the
    merged chunk is finalised and a new merge starts. Section labels are
    ignored during merging — the upstream section filter has already
    dropped excluded sections, and bigger chunks give the composer LLM
    more material per seed. The merged chunk inherits the first chunk's
    ``chunk_id`` and ``section``.
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


# Hard cap on canonical_answer length — matches R7 in the prompt. Words
# (whitespace-split) are tokenizer-independent and align with how a reader
# perceives length.
MAX_CANONICAL_WORDS = 15

# Re-exported for module-internal use. The schema's nested
# ``QuestionTypeWeights`` already partitions weights by lane, so the sampler
# only needs the tuples for fallback ordering and the multi-hop dependency
# check.
_SINGLE_HOP_TYPES = SINGLE_HOP_QUESTION_TYPES
_MULTI_HOP_TYPES = MULTI_HOP_QUESTION_TYPES

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
    # Bare-noun document proxies: 'the study,' / 'the authors.' / 'the experiment?'.
    # A trailing clause-end punct or EOL is the discriminator — qualified
    # references ('the study of X', 'the experiment where Y') carry a
    # disambiguator after the noun and survive.
    re.compile(
        r"\bthe\s+(?:study|research|trial|experiment|analysis|survey|review|"
        r"manuscript|findings|results|investigators?|authors?)(?:['’]s)?"
        r"\s*(?:[,;:.?!]|$)",
        re.IGNORECASE,
    ),
    # Internal scaffolding labels — should never appear in a closed-book
    # question. The digit boundary on "input N" and the underscore on
    # "chunk_a/b" keep this from biting legitimate domain uses of "input"
    # or "chunk" in ML/data-science questions.
    re.compile(
        r"\b(input\s+[12]|chunk_[ab]|the\s+(first|second)\s+input)\b",
        re.IGNORECASE,
    ),
]


@dataclass
class CompositionResult:
    """One per-seed outcome of a batched composition LLM call.

    ``rejection_explanation`` is the LLM's free-text reason for refusing
    to compose. Populated on ``linkable: false`` only. ``reason`` is an
    internal label set when the harness decides the result is not
    linkable for non-LLM reasons (parse errors, transient failures) —
    distinct from the LLM's own refusal text.

    ``preferred_type`` carries the type the orchestrator asked the LLM to
    generate. ``reasoning_type`` is the type the LLM actually produced
    (which may differ if the LLM fell back). ``preferred_type_used`` is
    used for per-type yield logging only — not persisted on the question.

    ``formula`` and ``formula_kind`` are populated for ``numeric`` and
    ``numeric_single`` questions; the harness verifies the math against
    ``canonical_answer``.
    """

    seed: Seed
    linkable: bool
    preferred_type: str = "bridge"
    reasoning_type: str = "bridge"
    preferred_type_used: bool = True
    question: str = ""
    canonical_answer: str = ""
    answer_variants: list[str] = field(default_factory=list)
    source_span_A: str = ""
    source_span_B: str = ""
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
class PreparedCorpus:
    """Result of one-time corpus preparation for exam generation.

    ``composition_results`` carries every per-seed outcome from the most
    recent ``generate_exam`` run, including LLM refusals — the orchestrator
    persists those alongside accepted candidates so the user can audit
    why the LLM declined to compose a question.
    """

    chunks: list[ChunkRecord]
    seeds: list[Seed] = field(default_factory=list)
    composition_results: list[CompositionResult] = field(default_factory=list)


class ExamAgent:
    """Generates open-ended typed 2-hop candidate questions from a corpus.

    The agent is stateless across runs — each call to ``generate_exam`` builds
    the embedding-pair index fresh. A user can also call ``prepare_corpus``
    once and reuse the resulting ``PreparedCorpus`` across multiple
    compositions (e.g. backfilling when too few candidates survive
    validation).

    Each seed is assigned a **preferred** question type, sampled from
    ``config.question_type_weights.single_hop`` for single-chunk seeds or
    ``.multi_hop`` for paired seeds, with a project-seeded RNG so the same
    corpus + project name yields the same type assignment across runs.
    """

    def __init__(
        self,
        config: ExaminerConfig,
        examiner_model: str,
        corpus_description: str = "",
        temperature: float | None = None,
        concurrency: int = 10,
        embed_callable: Callable[[list[str]], np.ndarray] | None = None,
        type_sampler_seed: int | str | None = None,
        reasoning_effort: str | None = None,
        composition_log_path: Path | None = None,
        span_verification_report_path: Path | None = None,
    ) -> None:
        self.config = config
        self.examiner_model = examiner_model
        self.corpus_description = corpus_description or "General enterprise documents."
        # Default temperature comes from config; explicit kwarg overrides it.
        self.temperature = temperature if temperature is not None else config.composition_temperature
        self.concurrency = concurrency
        # Tests inject a deterministic stub; production lazily loads a
        # SentenceTransformer the first time prepare_corpus runs.
        self._embed_callable = embed_callable
        # Reproducible per-seed preferred-type sampler. Fed by the orchestrator
        # from project_name so the same corpus produces the same type
        # assignment across runs; tests pass an int directly.
        self._type_rng = random.Random(type_sampler_seed)
        self._reasoning_effort = _resolve_reasoning_effort(examiner_model, reasoning_effort)
        if self._reasoning_effort is not None:
            logger.info("ExamAgent using reasoning_effort=%s on %s", self._reasoning_effort, examiner_model)
        # Surfaced for the orchestrator's empty-exam guard so it can include
        # the top rejection reasons in ExamGenerationFailed. Populated by
        # ``_compositions_to_questions`` on every call.
        self.last_composition_rejections: Counter[str] = Counter()
        # Per-rejection records for post-LLM filter failures (self_contained,
        # empty_span_a, empty_span_b, formula_mismatch, formula_missing,
        # pydantic_validation). Mirrors what the orchestrator already persists
        # for LLM refusals (linkable=False), so candidates.json carries every
        # rejection cause uniformly. Repopulated each call.
        self.last_downstream_rejections: list[dict] = []
        # TEMPORARY DEBUG: when set, accumulate every composition LLM call
        # (input chunks + raw response) in memory and dump a pretty JSON
        # array to this path at the end of each ``compose_multihop_batched``
        # pass, so we can inspect the parsed-but-unstored ``reasoning`` field
        # and audit whether prompt rules are being followed. Remove this
        # parameter, the records list, and the ``_record_composition_call`` /
        # ``_flush_composition_log`` helpers once composition-prompt
        # iteration is complete.
        self._composition_log_path = composition_log_path
        self._composition_log_records: list[dict[str, Any]] = []
        # TEMPORARY DEBUG: when set, ``validate_compositions`` asks
        # ``verify_source_facts`` to dump a per-question breakdown of
        # verbatim/tolerant/snap matches and rejection reasons to this path
        # for offline debugging. Remove this parameter, the field, and the
        # corresponding kwarg in ``validate_compositions`` once span-
        # verification tuning is complete.
        self._span_verification_report_path = span_verification_report_path

    def chunk_documents(
        self,
        documents: list[DoclingDocument],
        doc_ids: list[str],
    ) -> list[ChunkRecord]:
        """Chunk Docling documents with section-aware boundaries.

        Each chunk carries the deepest matching heading from its breadcrumb
        as a ``SectionLabel`` (via ``headings_to_label``) so the downstream
        pairing step can drop structurally non-substantive chunks (citation
        lists, acknowledgments, author/affiliation blocks).
        """
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
        """Build chunks → mixed-origin seeds for the corpus.

        The total seed pool is sized to ``exam_size *
        pair_overgeneration_factor`` and split across single-chunk,
        same-doc, and cross-doc origins by ``ExaminerConfig.seed_mix``.
        Embedding is shared across the same-doc and cross-doc paths.
        """
        chunks = self.chunk_documents(documents, doc_ids)
        # Section filter applied once at the boundary; downstream seeders
        # all operate on the pre-filtered set.
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
                "%d chunks <500 chars)",
                pre_merge_count,
                len(eligible_chunks),
                self.config.max_chunk_words,
                int(chunk_word_counts.min()),
                int(np.percentile(chunk_word_counts, 25)),
                int(np.median(chunk_word_counts)),
                int(np.percentile(chunk_word_counts, 75)),
                int(chunk_word_counts.max()),
                float(chunk_word_counts.mean()),
                sum(1 for c in eligible_chunks if len(c.text) < 500),
            )
        else:
            logger.info(
                "Greedy-merged eligible chunks: %d → 0 chunks (max_chunk_words=%d)",
                pre_merge_count,
                self.config.max_chunk_words,
            )

        target_seed_count = max(
            1,
            int(self.config.exam_size * self.config.pair_overgeneration_factor),
        )
        mix = self.config.seed_mix
        n_single = int(round(target_seed_count * mix["single_chunk"]))
        n_same = int(round(target_seed_count * mix["same_doc_pair"]))
        n_cross = max(0, target_seed_count - n_single - n_same)

        # Generate same-doc seeds first, then redistribute any deficit to
        # cross-doc, then redistribute cross-doc deficit to single-chunk.
        # This keeps the total seed pool full when one origin is structurally
        # infeasible (e.g. HotpotQA: same-doc impossible, cross-doc gets the
        # extra share; legal/medical with no cross-doc overlap: cross-doc
        # short, single-chunk gets the extra share).
        seeds: list[Seed] = []
        same_doc_seeds: list[Seed] = []
        cross_doc_seeds: list[Seed] = []

        if (n_same > 0 or n_cross > 0) and len(eligible_chunks) >= 2:
            embed_callable = self._embed_callable or make_pair_embedder(self.config.pair_embedding_model)
            t_embed = _time.perf_counter()
            embeddings = embed_callable([c.text for c in eligible_chunks])
            if embeddings.shape[0] != len(eligible_chunks):
                raise ValueError(
                    f"pair embedder returned {embeddings.shape[0]} vectors for {len(eligible_chunks)} eligible chunks"
                )
            logger.info(
                "Pair-embedded %d chunks via %s in %.1fs",
                len(eligible_chunks),
                self.config.pair_embedding_model,
                _time.perf_counter() - t_embed,
            )
            if n_same > 0:
                same_doc_seeds = emit_same_doc_pair_seeds(
                    eligible_chunks,
                    embeddings,
                    target_count=n_same,
                    cos_min=self.config.same_doc_pair_cosine_min,
                    cos_max=self.config.same_doc_pair_cosine_max,
                )
                same_doc_deficit = n_same - len(same_doc_seeds)
                if same_doc_deficit > 0:
                    logger.info(
                        "DIAG Seed budget redistribution: same_doc short by %d → cross_doc target +%d",
                        same_doc_deficit,
                        same_doc_deficit,
                    )
                    n_cross += same_doc_deficit
            if n_cross > 0:
                cross_doc_seeds = emit_cross_doc_pair_seeds(
                    eligible_chunks,
                    embeddings,
                    top_k_per_chunk=self.config.pair_top_k_per_chunk,
                    target_count=n_cross,
                )

        cross_doc_deficit = n_cross - len(cross_doc_seeds)
        if cross_doc_deficit > 0:
            logger.info(
                "DIAG Seed budget redistribution: cross_doc short by %d → single_chunk target +%d",
                cross_doc_deficit,
                cross_doc_deficit,
            )
            n_single += cross_doc_deficit

        single_chunk_seeds: list[Seed] = []
        if n_single > 0:
            single_chunk_seeds = emit_single_chunk_seeds(
                eligible_chunks,
                target_count=n_single,
            )

        seeds.extend(single_chunk_seeds)
        seeds.extend(same_doc_seeds)
        seeds.extend(cross_doc_seeds)

        logger.info(
            "Prepared corpus: %d raw chunks, %d after section-filter + merge (used for seeding), "
            "%d seeds (target=%d; single=%d, same_doc=%d, cross_doc=%d)",
            len(chunks),
            len(eligible_chunks),
            len(seeds),
            target_seed_count,
            len(single_chunk_seeds),
            len(same_doc_seeds),
            len(cross_doc_seeds),
        )
        return PreparedCorpus(chunks=chunks, seeds=seeds)

    def _sample_preferred_types(self, seeds: list[Seed]) -> list[str]:
        """Draw a preferred question type per seed, conditioned on its origin.

        Single-chunk seeds draw from ``question_type_weights.single_hop``;
        paired seeds draw from ``question_type_weights.multi_hop``. Each lane
        is normalised by ``random.choices`` at sample time, so the YAML need
        not pre-normalise.
        """
        weights = self.config.question_type_weights
        out: list[str] = []
        for seed in seeds:
            if seed.chunk_b is None:
                lane_weights = weights.single_hop
                fallback = _SINGLE_HOP_TYPES[0]
            else:
                lane_weights = weights.multi_hop
                fallback = _MULTI_HOP_TYPES[0]
            items = [(t, w) for t, w in lane_weights.items() if w > 0]
            if not items:
                # The schema validator already requires sum > 0 per lane, so
                # this branch is unreachable under a validated config. Kept as
                # a deterministic safety net for callers that bypass validation
                # (e.g. tests that mutate ``config`` post-construction).
                out.append(fallback)
                continue
            labels = [t for t, _ in items]
            ws = [w for _, w in items]
            out.append(self._type_rng.choices(labels, weights=ws, k=1)[0])
        return out

    async def compose_multihop_batched(self, seeds: list[Seed]) -> list[CompositionResult]:
        """Run batched composition LLM calls over the seed list.

        Seeds are partitioned into batches of ``composition_batch_size``. Each
        seed in a batch carries an independently sampled **preferred** question
        type, surfaced in the prompt as a hint the LLM may follow or fall back
        from.

        Per-element parse failures don't poison the batch — malformed entries
        are logged and skipped.
        """
        if not seeds:
            return []

        # TEMPORARY: reset the composition-log accumulator so each pass yields
        # a fresh debug file. Remove alongside the rest of the composition-log
        # plumbing once prompt iteration is complete.
        if self._composition_log_path is not None:
            self._composition_log_records = []

        preferred_types = self._sample_preferred_types(seeds)

        k = self.config.composition_batch_size
        batches: list[list[tuple[Seed, str]]] = [
            list(zip(seeds[i : i + k], preferred_types[i : i + k], strict=True)) for i in range(0, len(seeds), k)
        ]
        sem = asyncio.Semaphore(self.concurrency)

        results: list[CompositionResult] = []
        results_lock = asyncio.Lock()
        # per-batch wall-clock latency.
        batch_latencies: list[float] = []
        latency_lock = asyncio.Lock()

        async def _process_batch(batch: list[tuple[Seed, str]]) -> None:
            t0 = asyncio.get_event_loop().time()
            async with sem:
                batch_results = await self._compose_one_batch(batch)
            elapsed = asyncio.get_event_loop().time() - t0
            async with results_lock:
                results.extend(batch_results)
            async with latency_lock:
                batch_latencies.append(elapsed)

        try:
            with tqdm(total=len(batches), desc="Composing typed 2-hop questions", unit="batch") as pbar:

                async def _bounded(batch: list[tuple[Seed, str]]) -> None:
                    await _process_batch(batch)
                    pbar.update(1)

                await asyncio.gather(*[_bounded(b) for b in batches])
        finally:
            # TEMPORARY: flush the composition log even when composition aborts
            # mid-way, so a partial run still leaves a readable artifact.
            self._flush_composition_log()

        # composition latency p50/p95.
        if batch_latencies:
            sorted_lat = sorted(batch_latencies)
            n = len(sorted_lat)
            p50 = sorted_lat[n // 2]
            p95 = sorted_lat[min(n - 1, int(n * 0.95))]
            mean_lat = sum(sorted_lat) / n
            logger.info(
                "DIAG Composition latency: n=%d batches, mean=%.1fs p50=%.1fs p95=%.1fs",
                n,
                mean_lat,
                p50,
                p95,
            )

        return results

    async def _compose_one_batch(self, batch: list[tuple[Seed, str]]) -> list[CompositionResult]:
        """Compose one batch with up to N retries on transient LLM errors."""
        for attempt, cooldown in enumerate((0, *RETRY_COOLDOWNS_S), start=0):
            if cooldown:
                await asyncio.sleep(cooldown)
            try:
                raw = await self._call_composition_llm(batch)
                return self._parse_composition_batch(raw, batch)
            except Exception as exc:
                if not is_transient_llm_error(exc):
                    logger.info("Composition batch failed permanently: %s", format_llm_error(exc))
                    return [
                        CompositionResult(
                            seed=s,
                            preferred_type=pt,
                            linkable=False,
                            reason="composition_error",
                        )
                        for s, pt in batch
                    ]
                if attempt == len(RETRY_COOLDOWNS_S):
                    logger.warning(
                        "Composition batch exhausted retries (%s)",
                        format_llm_error(exc),
                    )
                    return [
                        CompositionResult(
                            seed=s,
                            preferred_type=pt,
                            linkable=False,
                            reason="composition_transient",
                        )
                        for s, pt in batch
                    ]
        raise AssertionError("unreachable: retry loop must return")

    async def _call_composition_llm(self, batch: list[tuple[Seed, str]]) -> str:
        seed_blocks = []
        for i, (seed, preferred_type) in enumerate(batch):
            # Surface the same per-type answer-shape hint that the eval-time
            # grader will use, so canonical_answer is produced in the shape
            # the downstream RAG pipeline is graded against.
            preferred_kind = "arithmetic" if preferred_type in ("numeric", "numeric_single") else None
            shape_hint = answer_format_hint(preferred_type, preferred_kind)
            block_lines = [
                f"Seed #{i}",
                f"  Origin: {seed.origin}",
                f"  Preferred reasoning type: {preferred_type}",
                f"  Expected canonical_answer shape: {shape_hint}",
                f"  === Input 1 === (doc_id={seed.chunk_a.doc_id})",
                f"  {seed.chunk_a.text}",
            ]
            if seed.chunk_b is not None:
                block_lines.append(f"  === Input 2 === (doc_id={seed.chunk_b.doc_id})")
                block_lines.append(f"  {seed.chunk_b.text}")
            seed_blocks.append("\n".join(block_lines))
        user = COMPOSITION_BATCH_USER_PROMPT.format(
            domain_description=self.corpus_description,
            k=len(batch),
            seed_blocks="\n\n".join(seed_blocks),
        )
        kwargs: dict = {
            "model": self.examiner_model,
            "messages": [
                # Anthropic prompt caching: the system prompt is stable across
                # every composition call in a run, so we attach the ephemeral
                # cache_control marker. LiteLLM strips/translates the marker
                # for non-Anthropic providers, so it's safe to send to anyone.
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": COMPOSITION_BATCH_SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
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
            self._record_composition_call(batch, raw)
        return raw

    def _record_composition_call(self, batch: list[tuple[Seed, str]], raw_response: str) -> None:
        """TEMPORARY: capture one composition call into the in-memory log accumulator.

        The accumulated list is flushed to ``self._composition_log_path`` as a
        pretty JSON array at the end of ``compose_multihop_batched``. The raw
        LLM response is pre-parsed when valid JSON so the file shows
        structured per-seed records instead of an escaped string blob.
        Remove this method, the ``composition_log_path`` constructor parameter,
        ``_flush_composition_log``, and the call sites once composition-prompt
        iteration is complete.
        """
        try:
            response: Any = json.loads(raw_response)
        except (ValueError, TypeError):
            response = raw_response
        self._composition_log_records.append(
            {
                "ts": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                "seeds": [
                    {
                        "preferred_type": preferred,
                        "origin": seed.origin,
                        "chunk_a_id": seed.chunk_a.chunk_id,
                        "chunk_a_text": seed.chunk_a.text,
                        "chunk_b_id": seed.chunk_b.chunk_id if seed.chunk_b is not None else None,
                        "chunk_b_text": seed.chunk_b.text if seed.chunk_b is not None else None,
                    }
                    for seed, preferred in batch
                ],
                "response": response,
            }
        )

    def _flush_composition_log(self) -> None:
        """TEMPORARY: write the accumulated composition log to a pretty JSON file."""
        if self._composition_log_path is None or not self._composition_log_records:
            return
        self._composition_log_path.write_text(
            json.dumps(self._composition_log_records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _parse_composition_batch(
        self,
        raw: str,
        batch: list[tuple[Seed, str]],
    ) -> list[CompositionResult]:
        """Parse a JSON array of K objects, one per (seed, preferred_type) pair.

        Schema for each entry:
          - refusal:  {"seed_id": i, "linkable": false, "explanation": "..."}
          - accepted: {"seed_id": i, "linkable": true,
                       "reasoning": "...",
                       "reasoning_type": "...",
                       "preferred_type_used": true|false,
                       "question": "...", "canonical_answer": "...",
                       "answer_variants": [...],
                       "formula": null | "...",
                       "formula_kind": null | "arithmetic",
                       "source_span_A": "...", "source_span_B": "..."}
        """
        text = _strip_markdown_fences(raw)
        items = _try_parse_json_array(text)
        if items is None:
            items = _extract_json_array(text)
        if items is None:
            logger.info("Composition batch parse failed; raw=%.200s", raw)
            return [
                CompositionResult(seed=s, preferred_type=pt, linkable=False, reason="parse_error") for s, pt in batch
            ]

        valid_types = set(QUESTION_TYPES)
        out: list[CompositionResult] = []
        for i, (seed, preferred_type) in enumerate(batch):
            entry: dict | None = None
            # Prefer entries that explicitly identify themselves by seed_id.
            for it in items:
                if isinstance(it, dict) and it.get("seed_id") == i:
                    entry = it
                    break
            if entry is None and i < len(items) and isinstance(items[i], dict):
                entry = items[i]

            if entry is None:
                out.append(
                    CompositionResult(
                        seed=seed,
                        preferred_type=preferred_type,
                        linkable=False,
                        reason="missing_in_batch",
                    )
                )
                continue

            linkable_raw = entry.get("linkable", False)
            if not isinstance(linkable_raw, bool):
                linkable_raw = bool(linkable_raw)

            if not linkable_raw:
                explanation = str(entry.get("explanation") or entry.get("reason") or "llm_marked_not_linkable")[:480]
                out.append(
                    CompositionResult(
                        seed=seed,
                        preferred_type=preferred_type,
                        linkable=False,
                        rejection_explanation=explanation,
                    )
                )
                continue

            try:
                question = str(entry["question"]).strip()
                canonical = str(entry["canonical_answer"]).strip()
                span_a = str(entry["source_span_A"]).strip()
                span_b = str(entry.get("source_span_B", "") or "").strip()
            except (KeyError, TypeError) as exc:
                logger.info("Seed %d composition entry missing required fields: %s", i, exc)
                out.append(
                    CompositionResult(
                        seed=seed,
                        preferred_type=preferred_type,
                        linkable=False,
                        reason="missing_fields",
                    )
                )
                continue

            # R7: canonical answers must fit in ≤ 15 words. Long abstractive
            # answers (mostly definitional) hurt EM scoring and force the
            # judge into work it shouldn't need to do.
            if len(canonical.split()) > MAX_CANONICAL_WORDS:
                out.append(
                    CompositionResult(
                        seed=seed,
                        preferred_type=preferred_type,
                        linkable=False,
                        reason="answer_too_long",
                    )
                )
                continue

            reported_type = str(entry.get("reasoning_type") or "").strip()
            if reported_type not in valid_types:
                reported_type = preferred_type
            preferred_used_raw = entry.get("preferred_type_used")
            if isinstance(preferred_used_raw, bool):
                preferred_type_used = preferred_used_raw
            else:
                preferred_type_used = reported_type == preferred_type

            variants_raw = entry.get("answer_variants", []) or []
            if isinstance(variants_raw, str):
                variants = [variants_raw]
            elif isinstance(variants_raw, list):
                variants = [str(v).strip() for v in variants_raw if isinstance(v, str) and v.strip()]
            else:
                variants = []

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
                    seed=seed,
                    preferred_type=preferred_type,
                    reasoning_type=reported_type,
                    preferred_type_used=preferred_type_used,
                    linkable=True,
                    question=question,
                    canonical_answer=canonical,
                    answer_variants=variants,
                    source_span_A=span_a,
                    source_span_B=span_b,
                    formula=formula,
                    formula_kind=formula_kind,
                )
            )
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
        """Convenience wrapper: prepare corpus → typed compose → span verify → single-hop probe.

        The returned questions still need to pass the oracle answerability gate
        in ``exam_validator`` and the 4-probe discrimination filter in
        ``orchestrator._generate_exam``. The corpus's ``composition_results``
        field carries every per-seed outcome — the orchestrator persists those
        alongside accepted candidates so the user can audit why the LLM
        declined to compose a question.

        ``doc_text_map`` (doc_id → ``dl_doc_to_chunk_text(dl_doc)``) is the
        canonical doc-text representation: HybridChunker chunk-text-concat,
        the same coordinate frame as vector ``char_range`` and graph chunk
        lookups. The source-span verifier searches this text to locate
        composer-extracted spans. When omitted (e.g. callers that only want
        composition results), span verification is skipped here and runs
        downstream instead.
        """
        corpus = self.prepare_corpus(documents, doc_ids, eligible_sections=eligible_sections)
        if not corpus.seeds:
            return [], corpus

        composition_results = await self.compose_multihop_batched(corpus.seeds)
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
        """Run the post-composition filters: typed gates → source spans.

        The multi-hop decomposability check now happens downstream in
        ``exam_validator.gate_oracle_pass`` (fused into the LLM oracle call
        for multi-hop candidates). When ``documents`` is None, span
        verification is skipped here and runs downstream.
        """
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
                # TEMPORARY DEBUG: dump per-span match/rejection details for
                # offline analysis. Drop this kwarg when the corresponding
                # ``report_path`` parameter on ``verify_source_facts`` is
                # removed.
                report_path=self._span_verification_report_path,
            )
        return questions

    def _compositions_to_questions(self, results: list[CompositionResult]) -> list[OpenEndedQuestion]:
        """Convert composition results into validated ``OpenEndedQuestion``s.

        Pipeline (in order):
          - linkable filter (LLM refusals + harness errors)
          - self-containment regex check
          - empty-span-B check on multi-hop seeds (LLM produced a single-hop
            answer for a 2-chunk seed → reject typed)
          - numeric formula verification (if reasoning_type in {"numeric", "numeric_single"})

        Span ↔ source verification is intentionally NOT performed here —
        ``exam_validator.verify_source_facts`` runs downstream against the
        full source doc with a multi-tier matcher (verbatim → whitespace-
        collapse → fuzzy 5-gram). Strict per-chunk substring matching here
        produced false positives on unicode drift (NBSP, smart quotes) and
        duplicated logic that lives correctly downstream.

        Tracks rejections in a single ``Counter`` keyed by reason and a
        per-origin × reason matrix for diagnostic logging. Returns the
        questions that survive every gate.
        """
        kept: list[OpenEndedQuestion] = []
        reasons: Counter[str] = Counter()
        # per-origin × reason matrix (Step 6 surfaces in the log).
        rejections_by_origin: dict[str, Counter[str]] = {}
        # per-origin survival counts (attempts → kept).
        origin_attempts: Counter[str] = Counter()
        origin_kept: Counter[str] = Counter()
        # sample up to 3 rejections per reason for human inspection.
        sample_rejections: dict[str, list[str]] = {}
        # Per-preferred-type {"attempts", "refused", "kept", "fallback"}.
        type_stats: dict[str, dict[str, int]] = {
            t: {"attempts": 0, "refused": 0, "kept": 0, "fallback": 0} for t in QUESTION_TYPES
        }
        # Reset per-rejection records for post-LLM filters; the orchestrator
        # reads this attribute after the call and concatenates with LLM-refusal
        # records when writing candidates.json.
        self.last_downstream_rejections = []

        def _reject(origin: str, reason: str, *, sample: str = "", record: dict | None = None) -> None:
            reasons[reason] += 1
            rejections_by_origin.setdefault(origin, Counter())[reason] += 1
            if sample and len(sample_rejections.get(reason, [])) < 3:
                sample_rejections.setdefault(reason, []).append(sample)
            if record is not None:
                self.last_downstream_rejections.append(record)

        for i, r in enumerate(results, start=1):
            origin = r.seed.origin
            origin_attempts[origin] += 1
            stats = type_stats.setdefault(
                r.preferred_type,
                {"attempts": 0, "refused": 0, "kept": 0, "fallback": 0},
            )
            stats["attempts"] += 1
            if not r.linkable:
                stats["refused"] += 1
                # Harness errors carry r.reason; LLM refusals carry only
                # r.rejection_explanation (free-text) — bucket those as
                # "llm_refused".
                code = r.reason or ("llm_refused" if r.rejection_explanation else "unspecified")
                _reject(
                    origin,
                    code,
                    sample=f"{r.seed.chunk_a.chunk_id} :: {r.rejection_explanation[:200]}"
                    if code == "llm_refused"
                    else f"{r.seed.chunk_a.chunk_id}",
                )
                continue

            seed_source_chunk_ids = [r.seed.chunk_a.chunk_id]
            if r.seed.chunk_b is not None:
                seed_source_chunk_ids.append(r.seed.chunk_b.chunk_id)

            sc_fail = self_containment_failure(r.question)
            if sc_fail is not None:
                _reject(
                    origin,
                    "self_contained",
                    sample=f"{r.seed.chunk_a.chunk_id} :: {r.question[:160]}",
                    record={
                        "source_chunk_ids": seed_source_chunk_ids,
                        "reason": "self_contained",
                        "question": r.question,
                        "matched_phrase": sc_fail[1],
                    },
                )
                logger.info("self-contained-fail: %r", sc_fail[1])
                continue

            # Empty span_A is an LLM error regardless of seed shape — every
            # accepted question must ground in chunk_A. Reject typed so the
            # funnel log shows the rate without falling through to pydantic.
            if not r.source_span_A.strip():
                _reject(
                    origin,
                    "empty_span_a",
                    sample=f"{r.seed.chunk_a.chunk_id} :: {r.question[:160]}",
                    record={
                        "source_chunk_ids": seed_source_chunk_ids,
                        "reason": "empty_span_a",
                        "question": r.question,
                    },
                )
                continue

            # Paired seed where LLM left span_B empty AND claimed a multi-hop
            # reasoning_type: R1 violation (claimed 2-hop but didn't ground in
            # chunk_B). A single-hop reasoning_type with empty span_B is an
            # internally-consistent fallback — falls through and is recorded
            # as single-hop downstream.
            if r.seed.chunk_b is not None and not r.source_span_B.strip() and r.reasoning_type in _MULTI_HOP_TYPES:
                _reject(
                    origin,
                    "empty_span_b_with_multi_hop_type",
                    sample=f"{r.seed.chunk_a.chunk_id} :: {r.question[:160]}",
                    record={
                        "source_chunk_ids": seed_source_chunk_ids,
                        "reason": "empty_span_b_with_multi_hop_type",
                        "question": r.question,
                        "reasoning_type": r.reasoning_type,
                    },
                )
                continue

            if r.reasoning_type in ("numeric", "numeric_single"):
                if not r.formula or not r.formula_kind:
                    _reject(
                        origin,
                        "formula_missing",
                        sample=f"{r.seed.chunk_a.chunk_id} :: {r.question[:120]} -> {r.canonical_answer}",
                        record={
                            "source_chunk_ids": seed_source_chunk_ids,
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
                        origin,
                        "formula_mismatch",
                        sample=f"{r.seed.chunk_a.chunk_id} :: formula={r.formula!r} answer={r.canonical_answer!r}",
                        record={
                            "source_chunk_ids": seed_source_chunk_ids,
                            "reason": "formula_mismatch",
                            "question": r.question,
                            "canonical_answer": r.canonical_answer,
                            "formula": r.formula,
                            "formula_kind": r.formula_kind,
                        },
                    )
                    logger.info(
                        "formula mismatch: formula=%r kind=%s answer=%r",
                        r.formula,
                        r.formula_kind,
                        r.canonical_answer,
                    )
                    continue

            # Branch on whether the LLM actually grounded in chunk_B, not on
            # seed shape. A paired seed with empty span_B (only reachable here
            # if reasoning_type is single-hop, per the gate above) is recorded
            # as single-hop with one source chunk.
            if r.seed.chunk_b is not None and r.source_span_B.strip():
                source_chunk_ids = [r.seed.chunk_a.chunk_id, r.seed.chunk_b.chunk_id]
                source_doc_ids = [r.seed.chunk_a.doc_id, r.seed.chunk_b.doc_id]
                source_spans = [r.source_span_A, r.source_span_B]
            else:
                source_chunk_ids = [r.seed.chunk_a.chunk_id]
                source_doc_ids = [r.seed.chunk_a.doc_id]
                source_spans = [r.source_span_A]

            try:
                question = OpenEndedQuestion(
                    id=f"C{i:04d}",
                    question=r.question,
                    canonical_answer=r.canonical_answer,
                    answer_variants=r.answer_variants,
                    reasoning_type=r.reasoning_type,
                    source_chunk_ids=source_chunk_ids,
                    source_doc_ids=source_doc_ids,
                    source_spans=source_spans,
                    formula=r.formula,
                    formula_kind=r.formula_kind,
                )
            except Exception as exc:  # noqa: BLE001
                _reject(
                    origin,
                    "pydantic_validation",
                    sample=f"{r.seed.chunk_a.chunk_id} :: {exc}",
                    record={
                        "source_chunk_ids": seed_source_chunk_ids,
                        "reason": "pydantic_validation",
                        "question": r.question,
                        "error": str(exc),
                    },
                )
                logger.info("OpenEndedQuestion validation failed: %s", exc)
                continue
            kept.append(question)
            origin_kept[origin] += 1
            stats["kept"] += 1
            if not r.preferred_type_used:
                stats["fallback"] += 1

        n_total = len(results)
        n_kept = len(kept)
        n_rejected = n_total - n_kept
        # Stash a copy so the orchestrator's empty-exam guard can surface the
        # top reasons in ExamGenerationFailed without re-walking results.
        self.last_composition_rejections = Counter(reasons)
        logger.info(
            "Composition → questions: %d kept / %d total (%d rejected)",
            n_kept,
            n_total,
            n_rejected,
        )
        if reasons:
            breakdown = ", ".join(f"{code}={n}" for code, n in reasons.most_common())
            logger.info("Composition rejections by reason: %s", breakdown)
        # per-origin survival funnel.
        if origin_attempts:
            survival = ", ".join(
                f"{origin}={origin_kept[origin]}/{origin_attempts[origin]}" for origin in sorted(origin_attempts.keys())
            )
            logger.info("DIAG Composition survival by origin: %s", survival)
        # per-origin × reason matrix.
        if rejections_by_origin:
            for origin in sorted(rejections_by_origin.keys()):
                ctr = rejections_by_origin[origin]
                origin_breakdown = ", ".join(f"{code}={n}" for code, n in ctr.most_common())
                logger.info("DIAG Composition rejections [%s]: %s", origin, origin_breakdown)
        # sample rejections (up to 3 per reason) for human inspection.
        if sample_rejections:
            for reason, samples in sorted(sample_rejections.items()):
                for j, s in enumerate(samples, start=1):
                    logger.info("DIAG Reject sample [%s #%d]: %s", reason, j, s)
        type_lines = []
        for t in QUESTION_TYPES:
            stats = type_stats.get(t)
            if not stats or stats["attempts"] == 0:
                continue
            type_lines.append(
                f"{t}: {stats['kept']} kept / {stats['attempts']} attempts "
                f"(refused={stats['refused']}, fallback={stats['fallback']})"
            )
        if type_lines:
            logger.info("Per-preferred-type yield: %s", " | ".join(type_lines))
        # Per-actual-type counts of kept questions (after possible fallback).
        actual_counts: dict[str, int] = {}
        for q in kept:
            actual_counts[q.reasoning_type] = actual_counts.get(q.reasoning_type, 0) + 1
        if actual_counts:
            actual_line = ", ".join(f"{t}={actual_counts.get(t, 0)}" for t in QUESTION_TYPES if t in actual_counts)
            logger.info("Per-actual-type kept counts: %s", actual_line)
        return kept


# --- helpers ---------------------------------------------------------------


def _resolve_reasoning_effort(model: str, effort: str | None) -> str | None:
    """Return ``effort`` if the model supports reasoning, else None.

    Mirrors ``ReasoningAgent._resolve_reasoning_effort`` so callers can ask
    LiteLLM to pass ``reasoning_effort`` through without crashing on models
    that don't support it.
    """
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
