"""Exam Agent — generates MCQ exams from full documents with diversity guarantees.

Uses a dedicated LLM (the examiner model) via LiteLLM to produce
multiple-choice questions from full document texts. Clustering and allocation
ensure the exam covers the full breadth of the corpus.

Questions are returned as candidates only — the calling code is responsible
for running the quality validation pipeline before freezing the exam.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import uuid
from dataclasses import dataclass, field

import litellm
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from agentic_autorag.config.models import MCQ_OPTION_LABELS, ExaminerConfig, MCQQuestion
from agentic_autorag.examiner._errors import format_llm_error, is_transient_llm_error
from agentic_autorag.examiner.clustering import (
    allocate_largest_remainder,
    compute_clusters,
    resolve_n_clusters,
)

logger = logging.getLogger(__name__)

_RETRY_COOLDOWNS = (10, 30, 60)

# Approximate tokens per word for budget estimation
_TOKENS_PER_WORD = 1.3

# Bloom's Revised Taxonomy levels (Anderson & Krathwohl, 2001).
# Level 6 (Create) is excluded — it's not feasible to assess via MCQ.
BLOOM_LEVELS = (
    {
        "level": "Remember",
        "instruction": (
            "Ask for a specific factual detail that requires locating a particular "
            "passage. The question should read like a realistic user query to a "
            "document search assistant. "
            "Choose a fact that is DISTINCTIVE to this document — something a domain "
            "expert would only know from reading this specific source. Avoid routine "
            "numeric outputs like generic accuracy or prevalence percentages unless "
            "the specific value is notable or unexpected. Prefer facts that identify "
            "a specific threshold, named entity, protocol parameter, or unusual outcome."
        ),
        "example": (
            "What percentage reduction in pulmonary vascular resistance was observed "
            "with inhaled iloprost during the Phase III trial?"
        ),
    },
    {
        "level": "Understand",
        "instruction": (
            "Ask a question that tests whether the reader can explain or interpret "
            "a finding, not just recall a single number. The question should read "
            "like someone trying to understand what a document means, not just what "
            "it says."
        ),
        "example": (
            "How does the interRAI ChYMH system improve care transitions across "
            "different age groups in mental health services?"
        ),
    },
    {
        "level": "Apply",
        "instruction": (
            "Ask a question where the reader must use information from the document "
            "to determine what would happen in a specific realistic scenario. The "
            "answer should follow logically from the document's stated rules, "
            "criteria, or procedures. "
            "Use scenario framing: start with 'You are a [role] responsible for ...' "
            "or 'A [entity] encounters ...' and embed the domain context in the "
            "scenario so the question is fully self-contained without referencing "
            "any document."
        ),
        "example": (
            "A patient presents with idiopathic granulomatous mastitis but has a "
            "history of poorly controlled diabetes. Which of the following treatment "
            "options would be contraindicated for this patient?"
        ),
    },
    {
        "level": "Analyze",
        "instruction": (
            "Ask a question that requires connecting multiple pieces of information "
            "from different parts of the document to identify a pattern, "
            "relationship, or distinction. The answer should not be found in any "
            "single sentence. "
            "Use 'Consider...' framing: embed the specific entities or measurements "
            "from the document into the question stem so no document reference is "
            "needed. Example stem: 'Consider two models, X and Y — how does "
            "difference Z affect which is more suitable for W?'"
        ),
        "example": (
            "Consider two naturally occurring canine models of inherited retinal "
            "degeneration — rcd1 and xlpra2. How does the difference in their "
            "photoreceptor degeneration timelines affect which model is more suitable "
            "for gene therapy trials targeting early-stage disease?"
        ),
    },
    {
        "level": "Evaluate",
        "instruction": (
            "Ask a question that requires making a judgment about the quality, "
            "significance, or appropriateness of something in the document. "
            "The reader should need to weigh evidence or compare alternatives. "
            "Frame as a decision-making scenario: 'A [decision-maker] must choose "
            "between...' or 'Which of the following approaches is most suitable "
            "for...?'. The judgment required should only be resolvable with the "
            "document's specific evidence."
        ),
        "example": (
            "A state corrections department is comparing community supervision programs "
            "for non-violent offenders. Which alternative program type offers both the "
            "highest per-inmate cost savings and the lowest recidivism risk?"
        ),
    },
)

# Weighted Bloom level distribution: ~10% Remember, 20% Understand, 25% Apply,
# 25% Analyze, 20% Evaluate. Remember-level questions are the most vulnerable to
# parametric leaks (isolated facts that may be common knowledge), so we down-weight them.
BLOOM_LEVEL_WEIGHTS = (0, 1, 2, 2, 3, 3, 4, 1, 2, 3)

MCQ_GENERATION_SYSTEM_PROMPT = """\
You are an expert at generating exam questions for evaluating AI document retrieval systems.

You are given a document from a real-world corpus. Your task is to write \
multiple-choice questions that a real user would ask when searching a document AI assistant.

Your questions MUST:
1. Be SELF-CONTAINED — never reference "the document", "the text", "the passage", \
"the paper", "the report", "the PDF", "this filing", "the above", "the contract", \
"based on the provided", or any phrase that implies the reader has the source in front of them.
2. Require retrieval of meaningful information — the question should test whether a \
retrieval system can surface the right context.
3. Sound like a real user query — practical, direct, and naturally phrased.
4. Be answerable from information in the document.

REWRITING RULE: If your first draft mentions "the study", "the research", "the trial", \
"the analysis", or any similar proxy for a source document, REWRITE the question by \
embedding the specific subject matter directly. Instead of "What did the study find \
about X?", write "What is the Y of X in Z context?" — where Y is the finding type \
and Z is the specific domain context from the document.
  Example rewrite:
  BAD:  "In the study evaluating dose-escalated radiation therapy, what was the Grade 2 GI toxicity rate?"
  GOOD: "What percentage of high-risk prostate cancer patients receiving dose-escalated \
whole pelvis IMRT experienced acute Grade 2 gastrointestinal toxicity?"

NEVER generate questions that ask for:
- URLs, web addresses, hyperlinks, or email addresses
- Case numbers, docket numbers, filing reference codes, or patent numbers
- The exact year or date something was established if that is the ONLY tested detail

CRITICAL — Parametric Leak Prevention:
Your question MUST NOT be answerable from general knowledge alone. To ensure this:
1. Target details unique to this document: specific measurements, relationships \
between entities, outcomes, conclusions, or procedures that are only found here.
2. Avoid questions about general concepts, definitions, or widely-known facts even if \
they appear in the document.
3. SELF-CHECK: Before finalizing, ask yourself: "Could an AI model answer this correctly \
without the document?" If yes, make the question more specific or choose a different fact.

UNIQUENESS TEST: The correct answer must contain at least one of:
  - A specific number, measurement, or date unique to THIS document
  - A proper noun in a relationship only described in THIS document
  - A technical procedure, criterion, or threshold specific to THIS document
If the correct answer is a general concept or widely-known fact, REJECT it \
and pick a different fact from the document.

For the 3 incorrect options (distractors):
- Each must be a plausible answer that could appear in a DIFFERENT document from \
the same domain. Think: "what would a similar document say instead?"
- Each must be clearly wrong when the actual document information is known.
- Do NOT rephrase the correct answer.
- Do NOT use obviously absurd or off-topic options.
- All 4 options should be approximately the same length and specificity.
- If the answer is a number, all options MUST be from the same order of magnitude.
- If the answer is a name, all options should be names from the same domain.

DISTRACTOR CALIBRATION: All 4 answer options MUST be equally plausible to someone who \
has NOT read the document. A reader without the document should have NO REASON to prefer \
one option over another. Given ONLY the question text and NO document, a reader should \
assign roughly equal probability (~25% each) to all four options. If one option "feels \
right" or three options are clearly absurd, rewrite the distractors.

Also output a "source_fact" field containing the information that answers the question.

CRITICAL source_fact requirements:
1. The source_fact must contain the INFORMATION needed to answer the question, \
written as clear, self-contained prose.
2. If the information comes from running text: copy the EXACT 3-5 consecutive \
sentences verbatim from the document.
3. If the information comes from a table, list, or structured data:
   a. First, look for any prose sentence in the document that states the same \
fact. If found, use that sentence (plus surrounding context).
   b. If NO prose sentence exists: write a clear prose summary of the relevant \
table/list data. Start the summary with "From the document's data:" so \
verification can identify it as a synthesis rather than a verbatim extract.
4. Include enough surrounding context so the fact stands on its own.
5. Do NOT output headers, labels, list-only snippets, ID-only lines, or bibliography/reference entries.
6. Do NOT output raw pipe-delimited table rows or formatting artifacts.

Domain context: {domain_description}

Good question examples (parametric-leak-resistant, self-contained):
- "What percentage reduction in pulmonary vascular resistance was observed with \
inhalation of iloprost?" — specific number, all options are similar percentages
- "Which court at St Carthage's House had only one shower for 16 residents?" — \
specific name, all options are plausible court names
- "You are a state corrections administrator looking to reduce incarceration costs \
for non-violent offenders. Which community supervision program type has been shown \
to produce the highest per-inmate savings compared to prison placement?" — \
scenario-framed, judgment requires comparing data only found in the document

For Apply, Analyze, and Evaluate questions, SCENARIO FRAMING is highly effective \
and keeps questions self-contained:
- "You are a [role] responsible for [task]. [Situation]. Which of the following...?"
- "Consider [specific technical entities from the document]. Which of the following...?"
- "A [decision-maker] must choose between [options from document]. Which...?"

Bad question examples (do NOT write these):
- "What is the URL for the voluntary government code on genetic testing?" — URL lookup
- "What were the case numbers for the Sixth Circuit opinion?" — case number lookup
- "What is the primary function of iloprost?" — general knowledge, no document needed
- "What type of organization is discussed?" — too vague, guessable from options
- "Based on the study's findings, what was the key result?" — 'the study's findings' \
is a document reference; state the specific domain context directly in the question
- "What was the reported specificity of the CART algorithm?" — 'the reported' implies \
a source document; rephrase as a direct factual question with the domain context stated
"""

MCQ_GENERATION_USER_PROMPT = """\
Document text:
{doc_text}

{avoid_section}\
Cognitive level for this question: {bloom_level}
{bloom_instruction}

Example of a question at this level: "{bloom_example}"

Generate one multiple-choice question at this cognitive level. \
For Apply, Analyze, and Evaluate levels, prefer questions whose answer draws on \
information from more than one sentence in the document. \
Make all 4 options equally plausible to someone who has not read this document.

REMINDER: The correct answer must NOT be guessable from general knowledge alone. \
All 4 options must be equally plausible without the document.

Return a valid JSON object with exactly these fields:
- "reasoning": brief explanation of why the correct answer is right, \
why each distractor is wrong, and why this question cannot be answered without the document
- "question": the question text (self-contained, realistic user query)
- "options": {{{option_dict_hint}}}
- "correct_answer": the letter of the correct option (e.g., "A")
- "source_fact": the passage from the document that answers the question \
(verbatim 3-5 sentences from running text, or a prose summary prefixed with \
"From the document's data:" if the answer comes from a table or list)

Return ONLY valid JSON, no markdown formatting or additional text.
"""

SELF_CONTAINED_FILTERS = [
    re.compile(
        r'\b(documentation|paper|article|research|study|passage|text|excerpt)\b\s*"[^"]+"',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(discussed in|addressed in|described in|mentioned in|according to|based on|stated in|of the)\b\s*"[^"]+"',
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(the\s+)?(above|given|provided|following)\s+(documentation|passage|text|excerpt|context)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\baccording\s+to\s+(the\s+)?(documentation|paper|article|passage|text|report|PDF|filing|contract)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^based\s+on\s+(the\s+)?(given\s+|provided\s+|above\s+)?"
        r"(text|passage|information|content|material|excerpt|context|document)",
        re.IGNORECASE,
    ),
    re.compile(r"\bin\s+the\s+(PDF|report|filing|contract|document)\b", re.IGNORECASE),
    re.compile(r"\bthe\s+report\s+states\b", re.IGNORECASE),
    re.compile(r"\bthis\s+filing\b", re.IGNORECASE),
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
        r"\b(Figure|Table|Exhibit|Schedule|Appendix|Annex|Chart|Graph|Diagram)\s+"
        r"(\d+(?:\.\d+)*|[A-Z](?:\.\d+)?)\b",
    ),
    # 'the study', 'the research', 'the trial' etc. used as document proxies.
    # Scoped to avoid blocking 'the study of X' (study as area of inquiry):
    # matches 'the study' as a standalone noun phrase (followed by 's, space, end, or punctuation).
    re.compile(
        r"\bthe\s+(?:study'?s?|research'?s?|trial'?s?|experiment'?s?|analysis'?s?|survey'?s?|review'?s?|findings?|results?|manuscript|investigators?|authors?)(?=[^\w]|$)",
        re.IGNORECASE,
    ),
    # 'based on the study / findings / results / analysis / paper / report'
    re.compile(
        r"\bbased\s+on\s+(?:the|this)\s+(?:study|research|trial|evidence|findings?|results?|analysis|review|paper|manuscript|report|survey|literature|article|publication)\b",
        re.IGNORECASE,
    ),
    # 'based on recent/current/presented/observed research or findings'
    re.compile(
        r"\bbased\s+on\s+(?:recent|current|available|presented|provided|observed)\s+(?:research|evidence|findings?|results?|data)\b",
        re.IGNORECASE,
    ),
    # 'the provided/presented/given case report/case study'
    re.compile(
        r"\bthe\s+(?:provided|presented|given|above)\s+(?:case\s+report|case\s+study|documentation|information|methodology)\b",
        re.IGNORECASE,
    ),
    # 'the reported X' — implies a source is reporting it
    re.compile(r"\bthe\s+reported\b", re.IGNORECASE),
]


@dataclass
class PreparedCorpus:
    """One-time corpus preparation result for exam generation.

    Created by ``ExamAgent.prepare_corpus()`` and reused across
    multiple ``generate_wave()`` calls during backfill.
    """

    doc_texts: list[str]
    expanded_ids: list[str]
    labels: np.ndarray = field(repr=False)
    n_clusters: int = 0
    cluster_sizes: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), repr=False)
    doc_embeddings: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)


# Minimum words per document to support one distinct question.
_MIN_WORDS_PER_QUESTION = 1500


def _log_quality_failure(logger_: logging.Logger, reason: str, q: MCQQuestion, extra: str = "") -> None:
    """Emit a structured multi-line QUALITY_FAIL log for a candidate question."""
    logger_.info("--- QUALITY_FAIL: %s ---", reason)
    logger_.info("  Q: %s", q.question)
    for option_key in sorted(q.options.keys()):
        logger_.info("  %s: %s", option_key, q.options[option_key])
    logger_.info("  Correct: %s", q.correct_answer)
    logger_.info("  Source fact: %s", q.source_fact or "(none)")
    if extra:
        logger_.info("  %s", extra)
    logger_.info("")


class ExamAgent:
    """Generates MCQ candidate questions from full documents with diversity guarantees."""

    def __init__(
        self,
        config: ExaminerConfig,
        examiner_model: str,
        embedding_model,
        corpus_description: str = "",
        temperature: float = 1.0,
        random_seed: int = 42,
        concurrency: int = 10,
    ) -> None:
        self.config = config
        self.examiner_model = examiner_model
        self.embedding_model = embedding_model
        self.corpus_description = corpus_description
        self.temperature = temperature
        self._rng = random.Random(random_seed)
        self.concurrency = concurrency

    def _split_large_document(self, doc_text: str, doc_id: str) -> list[tuple[str, str]]:
        """Split a large document into sections if it exceeds the word threshold.

        Returns a list of (section_text, section_id) tuples. Short documents
        return a single-element list with the original doc_id.
        """
        word_count = len(doc_text.split())
        if word_count <= self.config.doc_split_word_threshold:
            return [(doc_text, doc_id)]

        section_size = self.config.doc_section_word_size
        section_overlap = section_size // 10  # 10% overlap

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=section_size * 5,  # char estimate: 5 chars/word
            chunk_overlap=section_overlap * 5,
            separators=["\n\n\n", "\n\n", "\n", " ", ""],
        )
        sections = splitter.split_text(doc_text)
        logger.info("Document %s split into %d sections (word count: %d)", doc_id, len(sections), word_count)
        return [(section, f"{doc_id}_section_{i}") for i, section in enumerate(sections)]

    def _compute_doc_embedding(self, doc_text: str) -> np.ndarray:
        """Compute a single document embedding by mean-pooling chunk embeddings."""
        window_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        windows = window_splitter.split_text(doc_text)
        if not windows:
            windows = [doc_text[:2000]]
        window_embeddings = np.asarray(self.embedding_model.encode(windows), dtype=np.float32)
        return window_embeddings.mean(axis=0)

    def _doc_question_capacity(self, word_count: int) -> int:
        """Maximum distinct questions a document can support based on its length."""
        length_cap = max(1, word_count // _MIN_WORDS_PER_QUESTION)
        return min(length_cap, self.config.max_questions_per_doc)

    def prepare_corpus(
        self,
        documents: list[str],
        doc_ids: list[str],
    ) -> PreparedCorpus:
        """One-time corpus preparation: split, filter, embed, cluster.

        Returns a ``PreparedCorpus`` that can be reused across multiple
        ``generate_wave()`` calls during the backfill loop.
        """
        expanded: list[tuple[str, str]] = []
        for doc_text, doc_id in zip(documents, doc_ids, strict=False):
            expanded.extend(self._split_large_document(doc_text, doc_id))

        if self.config.min_doc_words > 0:
            before = len(expanded)
            expanded = [(t, d) for t, d in expanded if len(t.split()) >= self.config.min_doc_words]
            skipped = before - len(expanded)
            if skipped:
                logger.info("Skipped %d document(s) below %d-word minimum", skipped, self.config.min_doc_words)

        doc_texts = [t for t, _ in expanded]
        expanded_ids = [i for _, i in expanded]
        n_docs = len(doc_texts)

        if n_docs == 0:
            logger.warning("No documents provided for exam generation")
            return PreparedCorpus(doc_texts=[], expanded_ids=[], labels=np.array([], dtype=int))

        logger.info("Embedding %d documents for exam clustering", n_docs)
        doc_embeddings = np.vstack([self._compute_doc_embedding(text) for text in doc_texts])

        target_candidates = int(self.config.exam_size * self.config.initial_candidate_multiplier)
        n_clusters = resolve_n_clusters(n_docs, target_candidates)
        labels = compute_clusters(doc_embeddings, n_clusters)
        cluster_sizes = np.bincount(labels, minlength=n_clusters)

        logger.info("Clustered %d documents into %d clusters", n_docs, n_clusters)

        return PreparedCorpus(
            doc_texts=doc_texts,
            expanded_ids=expanded_ids,
            labels=labels,
            n_clusters=n_clusters,
            cluster_sizes=cluster_sizes,
            doc_embeddings=doc_embeddings,
        )

    async def generate_wave(
        self,
        corpus: PreparedCorpus,
        wave_size: int,
        exclude_questions: list[MCQQuestion] | None = None,
        cluster_deficits: dict[int, int] | None = None,
    ) -> list[MCQQuestion]:
        """Generate one wave of MCQ candidates.

        When ``cluster_deficits`` is provided, it is used as the per-cluster
        allocation directly (for backfill rounds targeting under-represented
        clusters). Otherwise, allocation is computed from ``wave_size``.

        Single-slot documents run concurrently. Multi-slot documents run
        sequentially within each doc (to avoid the race condition where
        concurrent calls to the same doc don't see each other's results)
        while different multi-slot docs run concurrently with each other.

        Returns:
            Candidate questions that passed structural checks, deduplication,
            and discriminator quality filtering (but NOT the validation pipeline).
        """
        if not corpus.doc_texts:
            return []

        n_docs = len(corpus.doc_texts)

        # Compute allocation
        if cluster_deficits is not None:
            allocations = np.array(
                [cluster_deficits.get(i, 0) for i in range(corpus.n_clusters)],
                dtype=int,
            )
        else:
            max_q_per_doc = max(1, -(-wave_size // n_docs))
            virtual_sizes = corpus.cluster_sizes * max_q_per_doc
            allocations = allocate_largest_remainder(virtual_sizes, wave_size)

        logger.info(
            "Wave allocation (target=%d): %s",
            wave_size,
            {i: int(allocations[i]) for i in range(corpus.n_clusters) if allocations[i] > 0},
        )

        # Build exclude sets from already-validated questions
        exclude_q_texts: set[str] = set()
        exclude_facts_by_doc: dict[str, list[str]] = {}
        exclude_questions_by_doc: dict[str, list[str]] = {}
        if exclude_questions:
            for q in exclude_questions:
                exclude_q_texts.add(q.question)
                for d_id in q.source_doc_ids:
                    exclude_questions_by_doc.setdefault(d_id, []).append(q.question)
                    if q.source_fact:
                        exclude_facts_by_doc.setdefault(d_id, []).append(q.source_fact)

        # Build per-cluster candidate lists with per-doc capacity caps
        candidates: list[tuple[str, str, int, int]] = []
        doc_slot_counts: dict[str, int] = {}

        for cluster_id in range(corpus.n_clusters):
            n_slots = int(allocations[cluster_id])
            if n_slots == 0:
                continue
            cluster_doc_indices = list(np.where(corpus.labels == cluster_id)[0])
            rng = np.random.default_rng(seed=42 + cluster_id)
            rng.shuffle(cluster_doc_indices)

            filled = 0
            cycle = 0
            while filled < n_slots and cycle < n_slots * 2:
                for doc_idx in cluster_doc_indices:
                    if filled >= n_slots:
                        break
                    d_id = corpus.expanded_ids[doc_idx]
                    word_count = len(corpus.doc_texts[doc_idx].split())
                    capacity = self._doc_question_capacity(word_count)
                    current = doc_slot_counts.get(d_id, 0)
                    if current >= capacity:
                        continue
                    doc_slot_counts[d_id] = current + 1
                    candidates.append((corpus.doc_texts[doc_idx], d_id, cluster_id, current))
                    filled += 1
                cycle += 1

        # Interleave round-robin across clusters
        per_cluster: dict[int, list[tuple[str, str, int, int]]] = {}
        for c in candidates:
            per_cluster.setdefault(c[2], []).append(c)

        interleaved: list[tuple[str, str, int, int]] = []
        max_len = max((len(v) for v in per_cluster.values()), default=0)
        for round_idx in range(max_len):
            for cluster_id in sorted(per_cluster.keys()):
                pool = per_cluster[cluster_id]
                if round_idx < len(pool):
                    interleaved.append(pool[round_idx])

        # Separate single-slot vs multi-slot docs for concurrency strategy
        multi_slot_doc_ids = {d_id for d_id, count in doc_slot_counts.items() if count > 1}

        single_slot = [(i, c) for i, c in enumerate(interleaved) if c[1] not in multi_slot_doc_ids]
        multi_slot_by_doc: dict[str, list[tuple[int, tuple[str, str, int, int]]]] = {}
        for i, c in enumerate(interleaved):
            if c[1] in multi_slot_doc_ids:
                multi_slot_by_doc.setdefault(c[1], []).append((i, c))

        # Tracking
        generated_by_doc: dict[str, list[str]] = {d_id: list(qs) for d_id, qs in exclude_questions_by_doc.items()}
        generated_facts_by_doc: dict[str, list[str]] = {d_id: list(fs) for d_id, fs in exclude_facts_by_doc.items()}

        logger.info(
            "Generating %d candidate MCQs from %d documents (concurrency=%d, multi-slot docs=%d)",
            len(interleaved),
            n_docs,
            self.concurrency,
            len(multi_slot_doc_ids),
        )

        _TRANSIENT_ERROR = object()
        results_by_idx: dict[int, MCQQuestion | None | object] = {}
        global_failures: dict[str, int] = {}

        # Phase 1: Single-slot docs — fully concurrent
        if single_slot:
            single_candidates = [c for _, c in single_slot]
            single_indices = [i for i, _ in single_slot]
            await self._run_generation_pass(
                single_candidates,
                results_by_idx,
                _TRANSIENT_ERROR,
                questions_per_doc=generated_by_doc,
                facts_per_doc=generated_facts_by_doc,
                global_failures=global_failures,
                total=len(single_candidates),
                concurrency=self.concurrency,
                desc="Generating exam questions",
                index_offset=single_indices,
            )

        # Phase 2: Multi-slot docs — sequential within each doc, concurrent across docs
        if multi_slot_by_doc:
            sem = asyncio.Semaphore(self.concurrency)
            pbar = tqdm(
                total=sum(len(slots) for slots in multi_slot_by_doc.values()),
                desc="Generating multi-slot questions",
                unit="q",
            )

            async def _generate_for_doc(doc_id: str, slots: list[tuple[int, tuple[str, str, int, int]]]) -> None:
                for idx, (doc_text, d_id, cluster_id, _slot_in_doc) in slots:
                    existing_q = list(generated_by_doc.get(doc_id, []))
                    existing_f = list(generated_facts_by_doc.get(doc_id, []))
                    async with sem:
                        result = await self._generate_single(
                            doc_text,
                            d_id,
                            cluster_id,
                            existing_q,
                            _TRANSIENT_ERROR,
                            global_failures=global_failures,
                            slot=idx,
                            existing_facts=existing_f,
                        )
                    if result is not _TRANSIENT_ERROR and result is not None:
                        q = result  # type: ignore[assignment]
                        generated_by_doc.setdefault(doc_id, []).append(q.question)
                        generated_facts_by_doc.setdefault(doc_id, []).append(q.source_fact)
                    results_by_idx[idx] = result
                    pbar.update(1)

            await asyncio.gather(*[_generate_for_doc(d_id, slots) for d_id, slots in multi_slot_by_doc.items()])
            pbar.close()

        # Retry transient errors
        for retry_round, cooldown in enumerate(_RETRY_COOLDOWNS, start=1):
            error_indices = [i for i, r in results_by_idx.items() if r is _TRANSIENT_ERROR]
            if not error_indices:
                break
            tqdm.write(
                f"\n  {len(error_indices)} generation(s) failed"
                f" — retrying after {cooldown}s cooldown"
                f" (round {retry_round}/{len(_RETRY_COOLDOWNS)})"
            )
            await asyncio.sleep(cooldown)
            retry_candidates = [(i, interleaved[i]) for i in sorted(error_indices)]
            await self._run_generation_pass_indexed(
                retry_candidates,
                results_by_idx,
                _TRANSIENT_ERROR,
                questions_per_doc=generated_by_doc,
                facts_per_doc=generated_facts_by_doc,
                global_failures=global_failures,
                desc=f"Retry round {retry_round}",
            )

        still_failed = sum(1 for r in results_by_idx.values() if r is _TRANSIENT_ERROR)
        if still_failed:
            tqdm.write(f"\n  {still_failed} generation(s) still failed after {len(_RETRY_COOLDOWNS)} retry rounds")

        questions: list[MCQQuestion] = [
            r  # type: ignore[misc]
            for r in (results_by_idx.get(i) for i in range(len(interleaved)))
            if r is not None and r is not _TRANSIENT_ERROR
        ]

        n_failed = len(interleaved) - len(questions)
        run_logger = logging.getLogger("agentic_autorag.run")
        run_logger.info(
            "Generated %d/%d candidate questions (%d failed generation)",
            len(questions),
            len(interleaved),
            n_failed,
        )
        if global_failures:
            failures_summary = ", ".join(f"{k}={v}" for k, v in sorted(global_failures.items()))
            run_logger.info("Generation failure statistics: %s", failures_summary)

        questions = self._deduplicate_exam(questions)

        doc_map = dict(zip(corpus.expanded_ids, corpus.doc_texts, strict=False))
        questions = self._filter_discriminator_quality(questions, doc_map)

        return questions

    async def generate_exam(
        self,
        documents: list[str],
        doc_ids: list[str],
    ) -> list[MCQQuestion]:
        """Convenience wrapper: prepare corpus then generate one wave.

        Equivalent to calling ``prepare_corpus()`` followed by a single
        ``generate_wave()`` with default allocation. Preserved for backward
        compatibility and simple usage.
        """
        corpus = self.prepare_corpus(documents, doc_ids)
        wave_size = int(self.config.exam_size * self.config.initial_candidate_multiplier)
        return await self.generate_wave(corpus, wave_size)

    async def _run_generation_pass(
        self,
        candidates: list[tuple[str, str, int, int]],
        results_by_idx: dict[int, MCQQuestion | None | object],
        transient_sentinel: object,
        *,
        questions_per_doc: dict[str, list[str]],
        facts_per_doc: dict[str, list[str]],
        global_failures: dict[str, int],
        total: int,
        concurrency: int,
        desc: str,
        index_offset: list[int] | None = None,
    ) -> None:
        sem = asyncio.Semaphore(concurrency)

        with tqdm(total=total, desc=desc, unit="q") as pbar:

            async def _bounded(result_idx: int, doc_text: str, doc_id: str, cluster_id: int, slot: int) -> None:
                async with sem:
                    existing_q = list(questions_per_doc.get(doc_id, []))
                    existing_f = list(facts_per_doc.get(doc_id, []))
                    result = await self._generate_single(
                        doc_text,
                        doc_id,
                        cluster_id,
                        existing_q,
                        transient_sentinel,
                        global_failures=global_failures,
                        slot=result_idx,
                        existing_facts=existing_f,
                    )
                if result is not transient_sentinel and result is not None:
                    q = result  # type: ignore[assignment]
                    questions_per_doc.setdefault(doc_id, []).append(q.question)
                    facts_per_doc.setdefault(doc_id, []).append(q.source_fact)
                    pbar.update(1)
                elif result is None:
                    pbar.update(1)
                results_by_idx[result_idx] = result

            indices = index_offset if index_offset is not None else list(range(len(candidates)))
            await asyncio.gather(
                *[
                    _bounded(result_idx, doc_text, doc_id, cluster_id, slot)
                    for result_idx, (doc_text, doc_id, cluster_id, slot) in zip(indices, candidates, strict=True)
                ]
            )

    async def _run_generation_pass_indexed(
        self,
        indexed_candidates: list[tuple[int, tuple[str, str, int, int]]],
        results_by_idx: dict[int, MCQQuestion | None | object],
        transient_sentinel: object,
        *,
        questions_per_doc: dict[str, list[str]],
        facts_per_doc: dict[str, list[str]],
        global_failures: dict[str, int],
        desc: str,
    ) -> None:
        sem = asyncio.Semaphore(self.concurrency)

        with tqdm(total=len(indexed_candidates), desc=desc, unit="q") as pbar:

            async def _bounded(idx: int, doc_text: str, doc_id: str, cluster_id: int, slot: int) -> None:
                async with sem:
                    existing_q = list(questions_per_doc.get(doc_id, []))
                    existing_f = list(facts_per_doc.get(doc_id, []))
                    result = await self._generate_single(
                        doc_text,
                        doc_id,
                        cluster_id,
                        existing_q,
                        transient_sentinel,
                        global_failures=global_failures,
                        slot=idx,
                        existing_facts=existing_f,
                    )
                if result is not transient_sentinel and result is not None:
                    q = result  # type: ignore[assignment]
                    questions_per_doc.setdefault(doc_id, []).append(q.question)
                    facts_per_doc.setdefault(doc_id, []).append(q.source_fact)
                    pbar.update(1)
                elif result is None:
                    pbar.update(1)
                results_by_idx[idx] = result

            await asyncio.gather(
                *[
                    _bounded(idx, doc_text, doc_id, cluster_id, slot)
                    for idx, (doc_text, doc_id, cluster_id, slot) in indexed_candidates
                ]
            )

    async def _generate_single(
        self,
        doc_text: str,
        doc_id: str,
        cluster_id: int,
        existing_questions: list[str],
        transient_sentinel: object,
        *,
        global_failures: dict[str, int],
        slot: int = 0,
        existing_facts: list[str] | None = None,
    ) -> MCQQuestion | None | object:
        """Wrapper around _generate_mcq_for_document that catches transient errors."""
        try:
            return await self._generate_mcq_for_document(
                doc_text,
                doc_id,
                cluster_id,
                existing_questions,
                global_failures=global_failures,
                slot=slot,
                existing_facts=existing_facts or [],
            )
        except Exception as exc:
            if is_transient_llm_error(exc):
                error_summary = format_llm_error(exc)
                tqdm.write(f"  TRANSIENT ERROR doc {doc_id} | {error_summary}")
                logger.debug("MCQ generation transient error for doc %s", doc_id, exc_info=True)
                return transient_sentinel
            error_summary = format_llm_error(exc)
            tqdm.write(f"  ERROR doc {doc_id} | {error_summary}")
            logger.debug("MCQ generation failed for doc %s", doc_id, exc_info=True)
            return None

    async def _generate_mcq_for_document(
        self,
        doc_text: str,
        doc_id: str,
        cluster_id: int,
        existing_questions: list[str],
        *,
        global_failures: dict[str, int],
        slot: int = 0,
        existing_facts: list[str] | None = None,
    ) -> MCQQuestion | None:
        """Generate one high-quality MCQ for a full document.

        Retries up to ``config.max_generation_retries`` times. Passes previously
        generated questions and source_facts for the same document to avoid
        correlated questions. The slot (global candidate index) selects the
        Bloom taxonomy level.
        """
        max_retries = self.config.max_generation_retries
        failures: dict[str, int] = {}
        for attempt in range(max_retries):
            try:
                mcq = await self._generate_mcq(
                    doc_text,
                    doc_id,
                    cluster_id,
                    existing_questions,
                    slot=slot,
                    existing_facts=existing_facts or [],
                )
                if mcq is None:
                    failures["parse"] = failures.get("parse", 0) + 1
                    global_failures["parse"] = global_failures.get("parse", 0) + 1
                    continue
                if not self._is_self_contained(mcq.question):
                    failures["self_contained"] = failures.get("self_contained", 0) + 1
                    global_failures["self_contained"] = global_failures.get("self_contained", 0) + 1
                    logger.info(
                        "SELF_CONTAINED_FAIL doc %s attempt %d: %s",
                        doc_id,
                        attempt + 1,
                        mcq.question,
                    )
                    continue
                if not self._is_source_fact_contextual(mcq.source_fact):
                    failures["source_fact"] = failures.get("source_fact", 0) + 1
                    global_failures["source_fact"] = global_failures.get("source_fact", 0) + 1
                    logger.info(
                        "SOURCE_FACT_FAIL doc %s attempt %d: %.120s",
                        doc_id,
                        attempt + 1,
                        mcq.source_fact,
                    )
                    continue

                mcq = self._shuffle_options(mcq)
                return mcq
            except Exception as exc:
                if is_transient_llm_error(exc):
                    raise
                failures["exception"] = failures.get("exception", 0) + 1
                global_failures["exception"] = global_failures.get("exception", 0) + 1
                logger.debug("MCQ generation attempt %d failed for doc %s", attempt + 1, doc_id, exc_info=True)

        failure_summary = ", ".join(f"{k}={v}" for k, v in sorted(failures.items()))
        logger.warning(
            "All %d MCQ generation attempts failed for doc %s: %s",
            max_retries,
            doc_id,
            failure_summary or "unknown",
        )
        return None

    async def _generate_mcq(
        self,
        doc_text: str,
        doc_id: str,
        cluster_id: int,
        existing_questions: list[str],
        *,
        slot: int = 0,
        existing_facts: list[str] | None = None,
    ) -> MCQQuestion | None:
        """Generate a single MCQ from a full document text using the examiner LLM.

        The slot (global candidate index) cycles through BLOOM_LEVELS using
        weighted distribution. When ``existing_facts`` are provided alongside
        ``existing_questions``, the avoid section shows both the fact and the
        question so the LLM targets a completely different passage.
        """
        labels = list(MCQ_OPTION_LABELS)
        option_dict_hint = ", ".join(f'"{lbl}": "..."' for lbl in labels)

        avoid_section = self._build_avoid_section(existing_questions, existing_facts or [])

        bloom = BLOOM_LEVELS[BLOOM_LEVEL_WEIGHTS[slot % len(BLOOM_LEVEL_WEIGHTS)]]

        system_prompt = MCQ_GENERATION_SYSTEM_PROMPT.format(
            domain_description=self.corpus_description or "General enterprise documents.",
        )
        user_prompt = MCQ_GENERATION_USER_PROMPT.format(
            doc_text=doc_text,
            avoid_section=avoid_section,
            option_dict_hint=option_dict_hint,
            bloom_level=bloom["level"],
            bloom_instruction=bloom["instruction"],
            bloom_example=bloom["example"],
        )

        response = await litellm.acompletion(
            model=self.examiner_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            num_retries=0,
        )
        raw = response.choices[0].message.content
        return self._parse_mcq_response(raw, doc_id, cluster_id, bloom_level=bloom["level"])

    def _extract_source_window(self, doc_text: str, source_fact: str, window_words: int = 100) -> str:
        """Extract a window of text around the source_fact for quality checks.

        Falls back to the first window_words words of the document if the
        source_fact is empty or not found.
        """
        if not source_fact:
            return " ".join(doc_text.split()[:window_words])

        # Find the source_fact in the document (first 50 chars as anchor)
        anchor = source_fact[:50]
        pos = doc_text.find(anchor)
        if pos == -1:
            # Anchor not found verbatim; use the source_fact itself as the reference
            return source_fact

        # Extract words around the found position
        pre_text = doc_text[:pos]
        words_before = pre_text.split()
        start_word = max(0, len(words_before) - window_words // 2)
        all_words = doc_text.split()
        end_word = min(len(all_words), start_word + window_words)
        return " ".join(all_words[start_word:end_word])

    def _parse_mcq_response(
        self,
        raw: str,
        doc_id: str,
        cluster_id: int,
        bloom_level: str = "",
    ) -> MCQQuestion | None:
        """Parse the LLM's JSON response into an MCQQuestion.

        Handles markdown code fences, trailing commas, and mixed
        text/JSON output. Returns None on any parse or validation failure.
        """
        try:
            text = raw.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)

            data = self._try_parse_json(text)

            if data is None:
                data = self._extract_json_object(text)

            if data is None:
                logger.info(
                    "JSON parse failed for doc %s: %.200s",
                    doc_id,
                    raw,
                )
                return None

            return MCQQuestion(
                id=str(uuid.uuid4()),
                question=data["question"],
                options=data["options"],
                correct_answer=data["correct_answer"],
                source_doc_ids=[doc_id],
                source_fact=data.get("source_fact", ""),
                bloom_level=bloom_level,
                cluster_id=cluster_id,
            )
        except (KeyError, ValueError) as exc:
            logger.info("MCQ response missing required fields for doc %s: %s", doc_id, exc)
            return None

    def _shuffle_options(self, mcq: MCQQuestion) -> MCQQuestion:
        """Shuffle answer option positions to reduce positional bias."""
        items = list(mcq.options.items())
        correct_text = mcq.options[mcq.correct_answer]
        self._rng.shuffle(items)

        new_options: dict[str, str] = {}
        new_correct: str | None = None
        for idx, (_, text) in enumerate(items):
            label = MCQ_OPTION_LABELS[idx]
            new_options[label] = text
            if text == correct_text:
                new_correct = label

        if new_correct is None:
            return mcq

        return mcq.model_copy(update={"options": new_options, "correct_answer": new_correct})

    @staticmethod
    def _build_avoid_section(existing_questions: list[str], existing_facts: list[str]) -> str:
        """Build the avoid section for the user prompt.

        When both questions and source_facts are available, pairs them so the
        LLM knows exactly which passages have been used and targets a different
        section of the document.
        """
        if existing_questions and existing_facts and len(existing_facts) == len(existing_questions):
            avoid_lines = []
            for q, f in zip(existing_questions, existing_facts, strict=True):
                fact_preview = f[:120] + "..." if len(f) > 120 else f
                q_preview = q[:100] + "..." if len(q) > 100 else q
                avoid_lines.append(f'  - Fact: "{fact_preview}" \u2192 Question: "{q_preview}"')
            return (
                "Do NOT generate a question about any of these already-used facts "
                "or passages. Target a COMPLETELY DIFFERENT section of the document:\n"
                + "\n".join(avoid_lines)
                + "\n\n"
            )
        if existing_questions:
            avoid_lines_simple = "\n".join(f"  - {q}" for q in existing_questions)
            return f"Do NOT generate a question similar to these already-generated questions:\n{avoid_lines_simple}\n\n"
        return ""

    @staticmethod
    def _is_self_contained(question_text: str) -> bool:
        """Return True when question text is self-contained."""
        return not any(pattern.search(question_text) for pattern in SELF_CONTAINED_FILTERS)

    def _is_source_fact_contextual(self, source_fact: str) -> bool:
        """Return True when source_fact has enough context to verify reliably."""
        normalized = " ".join(source_fact.split())
        if len(normalized) < self.config.source_fact_min_length:
            return False

        # Reject line-heavy label fragments that often come from table/header scraps.
        lines = [line.strip() for line in source_fact.splitlines() if line.strip()]
        if len(lines) >= 3:
            short_lines = sum(1 for line in lines if len(line.split()) <= 3)
            if short_lines / len(lines) >= 0.6:
                return False

        return True

    @staticmethod
    def _jaccard_ngram(text_a: str, text_b: str, n: int = 3) -> float:
        """Compute Jaccard similarity at word n-gram level."""
        tokens_a = text_a.lower().split()
        tokens_b = text_b.lower().split()

        if len(tokens_a) < n or len(tokens_b) < n:
            set_a, set_b = set(tokens_a), set(tokens_b)
            if not set_a or not set_b:
                return 0.0
            return len(set_a & set_b) / len(set_a | set_b)

        ngrams_a = set(tuple(tokens_a[i : i + n]) for i in range(len(tokens_a) - n + 1))
        ngrams_b = set(tuple(tokens_b[i : i + n]) for i in range(len(tokens_b) - n + 1))
        if not ngrams_a or not ngrams_b:
            return 0.0
        return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)

    def _deduplicate_exam(self, questions: list[MCQQuestion]) -> list[MCQQuestion]:
        """Remove near-duplicate question texts by cosine similarity."""
        if len(questions) <= 1:
            return questions

        question_texts = [question.question for question in questions]
        question_embeddings = np.asarray(self.embedding_model.encode(question_texts), dtype=np.float32)
        similarity_matrix = cosine_similarity(question_embeddings)

        kept_questions: list[MCQQuestion] = []
        removed_indices: set[int] = set()

        for idx in range(len(questions)):
            if idx in removed_indices:
                continue
            kept_questions.append(questions[idx])
            for jdx in range(idx + 1, len(questions)):
                if jdx in removed_indices:
                    continue
                if similarity_matrix[idx][jdx] > self.config.dedup_similarity_threshold:
                    removed_indices.add(jdx)

        return kept_questions

    def _compute_quality_metrics(
        self,
        mcq: MCQQuestion,
        source_text: str,
    ) -> dict[str, float]:
        """Compute the 4 discriminator quality metrics for a single question.

        Returns a dict with keys: extra_jaccard_gap, extra_embed_gap,
        intra_jaccard_max, intra_embed_max. Higher = worse for all metrics.
        """
        correct_text = mcq.options[mcq.correct_answer]
        discriminators = [text for key, text in mcq.options.items() if key != mcq.correct_answer]

        all_answers = list(mcq.options.values())
        mean_token_len = int(np.mean([len(a.split()) for a in all_answers]))
        n_gram = max(1, mean_token_len)

        batch_texts = [source_text, correct_text, *discriminators]
        batch_embeddings = np.asarray(self.embedding_model.encode(batch_texts), dtype=np.float32)
        source_emb = batch_embeddings[0:1]
        correct_emb = batch_embeddings[1:2]
        disc_embs = batch_embeddings[2:]

        j_correct = self._jaccard_ngram(source_text, correct_text, n_gram)
        e_correct = float(cosine_similarity(source_emb, correct_emb)[0][0])

        extra_j_gaps: list[float] = []
        extra_e_gaps: list[float] = []
        intra_j_vals: list[float] = []
        intra_e_vals: list[float] = []

        for idx, disc in enumerate(discriminators):
            j_disc = self._jaccard_ngram(source_text, disc, n_gram)
            e_disc = float(cosine_similarity(source_emb, disc_embs[idx : idx + 1])[0][0])
            extra_j_gaps.append(j_disc - j_correct)
            extra_e_gaps.append(e_disc - e_correct)

            j_intra = self._jaccard_ngram(correct_text, disc, n_gram)
            e_intra = float(cosine_similarity(correct_emb, disc_embs[idx : idx + 1])[0][0])
            intra_j_vals.append(j_intra)
            intra_e_vals.append(e_intra)

        return {
            "extra_jaccard_gap": max(extra_j_gaps) if extra_j_gaps else 0.0,
            "extra_embed_gap": max(extra_e_gaps) if extra_e_gaps else 0.0,
            "intra_jaccard_max": max(intra_j_vals) if intra_j_vals else 0.0,
            "intra_embed_max": max(intra_e_vals) if intra_e_vals else 0.0,
        }

    def _filter_discriminator_quality(
        self,
        questions: list[MCQQuestion],
        documents: dict[str, str],
    ) -> list[MCQQuestion]:
        """Batch-filter questions using percentile-based discriminator quality.

        Following the paper's guidance, thresholds are auto-calibrated at the
        (1 - target_removal_pct) percentile so that ~5% of questions are
        removed per metric. This avoids hard-coded thresholds that may not
        match the corpus.
        """
        if len(questions) < 5:
            logger.info("Too few candidates (%d) for batch quality filter, skipping", len(questions))
            return questions

        # Compute metrics for all questions
        metric_names = [
            "extra_jaccard_gap",
            "extra_embed_gap",
            "intra_jaccard_max",
            "intra_embed_max",
        ]
        all_metrics: list[dict[str, float]] = []
        for q in questions:
            doc_id = q.source_doc_ids[0]
            doc_text = documents.get(doc_id, "")
            source_window = self._extract_source_window(doc_text, q.source_fact)
            all_metrics.append(self._compute_quality_metrics(q, source_window))

        # Compute percentile threshold for each metric
        pct = (1.0 - self.config.discriminator_removal_pct) * 100.0
        thresholds: dict[str, float] = {}
        for name in metric_names:
            values = [m[name] for m in all_metrics]
            thresholds[name] = float(np.percentile(values, pct))

        logger.info(
            "Discriminator quality thresholds (p%.0f): %s",
            pct,
            ", ".join(f"{k}={v:.3f}" for k, v in thresholds.items()),
        )

        # Filter: remove questions exceeding ANY threshold
        passed: list[MCQQuestion] = []
        removed_counts: dict[str, int] = {name: 0 for name in metric_names}

        for q, metrics in zip(questions, all_metrics, strict=True):
            fail_reason = None
            for name in metric_names:
                if metrics[name] > thresholds[name]:
                    fail_reason = name
                    removed_counts[name] += 1
                    break  # report first failing metric

            if fail_reason is not None:
                _log_quality_failure(
                    logger,
                    reason=(f"{fail_reason} ({metrics[fail_reason]:.3f} > {thresholds[fail_reason]:.3f} threshold)"),
                    q=q,
                )
            else:
                passed.append(q)

        n_removed = len(questions) - len(passed)
        logger.info(
            "Discriminator quality filter: %d/%d passed (%d removed: %s)",
            len(passed),
            len(questions),
            n_removed,
            ", ".join(f"{k}={v}" for k, v in removed_counts.items() if v > 0) or "none",
        )
        return passed

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        """Attempt to parse JSON, fixing trailing commas first."""
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_json_object(text: str) -> dict | None:
        """Extract the first JSON object from mixed text."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        data = json.loads(cleaned)
                        return data if isinstance(data, dict) else None
                    except json.JSONDecodeError:
                        return None
        return None
