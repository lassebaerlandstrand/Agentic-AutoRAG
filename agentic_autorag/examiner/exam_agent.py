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
from dataclasses import dataclass, field

import litellm
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from agentic_autorag.config.models import MCQ_OPTION_LABELS, ExaminerConfig, MCQQuestion
from agentic_autorag.examiner._errors import format_llm_error, is_transient_llm_error
from agentic_autorag.examiner.clustering import (
    allocate_difficulty_weighted,
    allocate_largest_remainder,
    compute_clusters,
    compute_difficulty_scores,
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
            "Ask for a specific factual detail BURIED in a body paragraph, table, "
            "or sub-section — NOT from the abstract, conclusion, or headings. "
            "The question should read like a realistic user query to a "
            "document search assistant. "
            "Choose a fact that is DISTINCTIVE to this document — something a domain "
            "expert would only know from reading this specific source. Avoid routine "
            "numeric outputs like generic accuracy or prevalence percentages unless "
            "the specific value is notable or unexpected. Prefer facts that identify "
            "a specific threshold, named entity, protocol parameter, or unusual outcome. "
            "Use DIFFERENT WORDS than the source passage in your question."
        ),
        "example": (
            "What was the incidence of treatment-related gastrointestinal side effects "
            "among patients receiving the higher dose regimen in the iloprost inhalation trial?"
        ),
    },
    {
        "level": "Understand",
        "instruction": (
            "Ask a question that tests whether the reader can explain or interpret "
            "a SPECIFIC finding from a NON-PROMINENT section of the document — not "
            "the main conclusion. The question should read like someone trying to "
            "understand a detail, not the headline. "
            "Use different terminology than the source text to create a vocabulary gap."
        ),
        "example": (
            "Why does the interRAI ChYMH assessment system use different screening "
            "thresholds for adolescents compared to younger children when flagging "
            "mental health transitions?"
        ),
    },
    {
        "level": "Apply",
        "instruction": (
            "Ask a question where the reader must use a SPECIFIC rule, criterion, or "
            "procedure from the document to determine what would happen in a realistic "
            "scenario. The rule must come from a detail section, not the overview. "
            "Use scenario framing: start with 'You are a [role] responsible for ...' "
            "or 'A [entity] encounters ...' and embed the domain context in the "
            "scenario so the question is fully self-contained without referencing "
            "any document. "
            "Distractors should be outcomes that would result from applying a DIFFERENT "
            "rule or procedure from the SAME document."
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
            "Ask a question that requires connecting information from AT LEAST TWO "
            "DIFFERENT sections of the document to identify a pattern, relationship, "
            "or distinction. The answer should NOT be found in any single paragraph. "
            "Use 'Consider...' framing: embed the specific entities or measurements "
            "from the document into the question stem so no document reference is "
            "needed. Example stem: 'Consider two models, X and Y — how does "
            "difference Z affect which is more suitable for W?' "
            "Make each distractor a conclusion you would reach if you only had ONE "
            "of the required sections but not the other."
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
            "Ask a question that requires making a judgment by weighing SPECIFIC "
            "evidence from multiple parts of the document. The reader must compare "
            "alternatives using quantitative or qualitative details that are scattered "
            "across different sections. "
            "Frame as a decision-making scenario: 'A [decision-maker] must choose "
            "between...' or 'Which of the following approaches is most suitable "
            "for...?'. Each option should be a real alternative discussed in the "
            "document, making retrieval of the RIGHT comparison critical."
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
multiple-choice questions that TEST WHETHER A RETRIEVAL SYSTEM CAN FIND THE RIGHT \
PASSAGE. Easy questions that any search would surface are USELESS — you must create \
questions that ONLY a good retrieval system can answer.

Your questions MUST:
1. Be SELF-CONTAINED — never reference "the document", "the text", "the passage", \
"the paper", "the report", "the PDF", "this filing", "the above", "the contract", \
"based on the provided", or any phrase that implies the reader has the source in front of them.
2. CHALLENGE RETRIEVAL — the answer must come from a specific, non-obvious location \
in the document. Questions about main conclusions, abstracts, or repeated key points \
are TOO EASY because any search surfaces them.
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

RETRIEVAL DIFFICULTY — Target facts that are HARD to find:
1. DO target: supporting details buried in body paragraphs, table footnotes, \
methodology specifics, edge-case conditions, secondary outcomes, or nested sub-sections.
2. DO NOT target: main findings, conclusions, abstracts, executive summaries, \
section headings, or any fact that is stated multiple times in the document.
3. The best questions require locating a SPECIFIC paragraph or table cell — not \
information that appears across many sections.
4. If the fact appears in both a summary AND a detail section, target the DETAIL \
(e.g., a specific sub-group result rather than the overall finding).

VOCABULARY GAP: Phrase the question using DIFFERENT WORDS than the source passage. \
Do not copy key terms verbatim from the answer passage. Instead, use synonyms, \
paraphrases, or higher-level descriptions so that keyword-based search alone \
cannot find the answer. Example: if the document says "adverse event rate was 12%", \
ask about "side-effect incidence" or "safety outcome frequency", not "adverse event rate".

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

DOCUMENT-GROUNDED DISTRACTORS — This is critical for testing retrieval:
For the 3 incorrect options, draw them from OTHER FACTS IN THE SAME DOCUMENT \
whenever possible. This means if a retrieval system returns the WRONG passage \
from the same document, the reader will select a distractor instead of the \
correct answer. This is exactly the behavior we want to test.

Distractor rules:
- The remaining distractor(s) can be plausible domain alternatives.
- Each distractor must be clearly wrong when the CORRECT passage is retrieved.
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

Also output a "source_fact" field: an ARRAY of 1-3 VERBATIM excerpts from the \
document that contain the information answering the question.

CRITICAL source_fact requirements:
1. Each entry in the "source_fact" array MUST be a VERBATIM contiguous excerpt \
copied character-for-character from the document — including original punctuation, \
whitespace, line breaks, and markdown formatting. DO NOT paraphrase, summarize, \
rewrite, reflow, or "clean up" the text. The verification step checks that each \
excerpt appears as an exact substring of the source document.
2. PREFER a single excerpt (array of length 1). Use 2-3 excerpts ONLY when the \
answer genuinely requires combining non-adjacent parts of the document (e.g., \
a table row plus a qualifier sentence in a later section).
3. Each excerpt must include enough SURROUNDING CONTEXT to stand on its own:
   - For running prose: copy 3-5 consecutive sentences that include the answer.
   - For a table or list: copy the ENTIRE relevant table (or relevant rows with \
the header row) AS IT APPEARS in the document, INCLUDING the markdown pipe \
characters `|`, separator line like `|---|---|`, and any prose sentence from the \
paragraph immediately before or after the table. The reader must see both the \
column headers and the data rows so the numbers are interpretable.
4. Each excerpt should be long enough to be unambiguous — aim for at least 150 \
characters total across all excerpts. Upper guideline: ~2000 characters per excerpt.
5. Do NOT output excerpts that are only headers, only ID-like strings, or only \
bibliography entries — these don't carry enough information to answer a question.

WORKED EXAMPLE — table-sourced question.
Document (excerpt):
```
In our 2018 cohort, we observed substantial differences across treatment arms:

Table 3: Response rates by treatment and severity.

| Severity | Drug A | Drug B | Drug C |
|----------|--------|--------|--------|
| Mild     | 45.2%  | 52.1%  | 67.8%  |
| Moderate | 38.9%  | 44.3%  | 61.5%  |
| Severe   | 22.1%  | 28.7%  | 49.2%  |

These findings suggest Drug C is most effective in severe cases.
```
GOOD source_fact (array of length 1, verbatim with table + header + surrounding prose):
[
  "Table 3: Response rates by treatment and severity.\n\n\
| Severity | Drug A | Drug B | Drug C |\n\
|----------|--------|--------|--------|\n\
| Mild     | 45.2%  | 52.1%  | 67.8%  |\n\
| Moderate | 38.9%  | 44.3%  | 61.5%  |\n\
| Severe   | 22.1%  | 28.7%  | 49.2%  |\n\n\
These findings suggest Drug C is most effective in severe cases."
]
BAD source_fact (paraphrased, NOT verbatim — will be rejected):
[
  "From the document's data: Table 3 shows response rates were highest for Drug C in severe cases (49.2%)."
]
BAD source_fact (just the row without the header — numbers lose meaning):
[
  "| Severe | 22.1% | 28.7% | 49.2% |"
]

Domain context: {domain_description}

Good question examples (hard to retrieve, document-grounded distractors):
- "What percentage reduction in pulmonary vascular resistance was observed with \
inhalation of iloprost?" — targets a specific measurement buried in results, \
distractors are other percentages from the same study
- "Which court at St Carthage's House had only one shower for 16 residents?" — \
specific detail from a sub-section, distractors are other court names from the document
- "You are a state corrections administrator looking to reduce incarceration costs \
for non-violent offenders. Which community supervision program type has been shown \
to produce the highest per-inmate savings compared to prison placement?" — \
scenario-framed, distractors are other program types discussed in the document

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
- "What was the main finding of the analysis?" — TOO EASY, targets the conclusion \
which any retrieval system will surface. Target a SPECIFIC sub-finding instead.
- "Based on the study's findings, what was the key result?" — 'the study's findings' \
is a document reference; state the specific domain context directly in the question
- "What was the reported specificity of the CART algorithm?" — 'the reported' implies \
a source document; rephrase as a direct factual question with the domain context stated
"""

MCQ_BATCH_USER_PROMPT = """\
Document text:
{doc_text}

{exclude_section}\
Generate exactly {k} multiple-choice questions from this document.
Each question MUST target a DIFFERENT section and a DIFFERENT fact.
No two questions may share the same source_fact or test the same piece of information.

For each question, use the assigned cognitive level:
{bloom_instructions}

For Apply, Analyze, and Evaluate levels, prefer questions whose answer draws on \
information from more than one sentence in the document. \
Make all 4 options equally plausible to someone who has not read this document.

REMINDER: The correct answer must NOT be guessable from general knowledge alone. \
All 4 options must be equally plausible without the document. \
At least 1 distractor must come from OTHER facts in THIS SAME document. \
Target a fact that is NOT in the abstract, conclusion, or section headings — \
pick a specific detail buried in a body paragraph, table, or sub-section. \
Phrase the question using DIFFERENT WORDS than the source passage.

Return a valid JSON ARRAY of exactly {k} objects. Each object must have:
- "reasoning": brief explanation of why the correct answer is right, \
why each distractor is wrong, and why this question cannot be answered without the document
- "question": the question text (self-contained, realistic user query)
- "options": {{{option_dict_hint}}}
- "correct_answer": the letter of the correct option (e.g., "A")
- "source_fact": an ARRAY of 1-3 verbatim excerpts from the document that \
contain the answer. Each excerpt must appear as an exact substring of the \
document (including original whitespace and markdown). Prefer a single excerpt; \
use multiple only when the answer requires combining non-adjacent locations. \
For table-based answers, always include the table header row and at least one \
surrounding prose sentence. Example for a single-excerpt question: \
["Table 3 shows ... | header | ... | row 1 | ... These findings suggest ..."]

Return ONLY a valid JSON array, no markdown formatting or additional text.
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
    difficulty_scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32), repr=False)


# Minimum words per document to support one distinct question.
_MIN_WORDS_PER_QUESTION = 1500


def _format_failure_stats(counts: dict[str, int]) -> str:
    """Format failure counts as ``parent=N (sub_a=X, sub_b=Y), parent2=M``.

    Sub-buckets use ``parent.subname`` keys (e.g. ``source_fact.not_in_doc``) and
    are grouped under their parent in the output.
    """
    parents: dict[str, int] = {}
    subs: dict[str, list[tuple[str, int]]] = {}
    for key, value in counts.items():
        if "." in key:
            parent, sub = key.split(".", 1)
            subs.setdefault(parent, []).append((sub, value))
        else:
            parents[key] = value

    parts: list[str] = []
    for parent in sorted({*parents.keys(), *subs.keys()}):
        total = parents.get(parent)
        sub_list = sorted(subs.get(parent, []))
        if total is None and sub_list:
            total = sum(v for _, v in sub_list)
        if sub_list:
            sub_str = ", ".join(f"{s}={v}" for s, v in sub_list)
            parts.append(f"{parent}={total} ({sub_str})")
        else:
            parts.append(f"{parent}={total}")
    return ", ".join(parts)


def _log_quality_failure(logger_: logging.Logger, reason: str, q: MCQQuestion, extra: str = "") -> None:
    """Emit a structured multi-line QUALITY_FAIL log for a candidate question."""
    logger_.info("--- QUALITY_FAIL: %s ---", reason)
    logger_.info("  Q: %s", q.question)
    for option_key in sorted(q.options.keys()):
        logger_.info("  %s: %s", option_key, q.options[option_key])
    logger_.info("  Correct: %s", q.correct_answer)
    logger_.info("  Source fact: %s", "\n---\n".join(q.source_fact) if q.source_fact else "(none)")
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

        difficulty_scores = compute_difficulty_scores(doc_embeddings)
        mean_diff = float(difficulty_scores.mean()) if len(difficulty_scores) else 0.0
        logger.info(
            "Clustered %d documents into %d clusters (mean difficulty=%.3f)",
            n_docs,
            n_clusters,
            mean_diff,
        )

        return PreparedCorpus(
            doc_texts=doc_texts,
            expanded_ids=expanded_ids,
            labels=labels,
            n_clusters=n_clusters,
            cluster_sizes=cluster_sizes,
            doc_embeddings=doc_embeddings,
            difficulty_scores=difficulty_scores,
        )

    async def generate_wave(
        self,
        corpus: PreparedCorpus,
        wave_size: int,
        exclude_questions: list[MCQQuestion] | None = None,
        cluster_deficits: dict[int, int] | None = None,
    ) -> list[MCQQuestion]:
        """Generate one wave of MCQ candidates using batch generation.

        Each document gets a single LLM call that generates K questions at
        different Bloom taxonomy levels. All documents run fully concurrent.

        When ``cluster_deficits`` is provided, it is used as the per-cluster
        allocation directly (for backfill rounds targeting under-represented
        clusters). Otherwise, allocation is computed from ``wave_size``.

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
        elif self.config.difficulty_weighted_allocation and len(corpus.difficulty_scores) > 0:
            max_q_per_doc = max(1, -(-wave_size // n_docs))
            virtual_sizes = corpus.cluster_sizes * max_q_per_doc
            allocations = allocate_difficulty_weighted(
                virtual_sizes,
                difficulty_scores=corpus.difficulty_scores,
                labels=corpus.labels,
                exam_size=wave_size,
                min_per_cluster=self.config.min_questions_per_cluster,
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

        # Build exclude sets from already-validated questions (for backfill)
        exclude_facts_by_doc: dict[str, list[str]] = {}
        exclude_questions_by_doc: dict[str, list[str]] = {}
        if exclude_questions:
            for q in exclude_questions:
                for d_id in q.source_doc_ids:
                    exclude_questions_by_doc.setdefault(d_id, []).append(q.question)
                    if q.source_fact:
                        # Join span list into one string for display in the avoid-section prompt.
                        exclude_facts_by_doc.setdefault(d_id, []).append(" ... ".join(q.source_fact))

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

        allocated_total = int(allocations.sum())
        if len(candidates) < allocated_total:
            logger.info(
                "Filled %d/%d allocated slots (%d capacity-limited)",
                len(candidates),
                allocated_total,
                allocated_total - len(candidates),
            )

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

        # Group candidates by doc_id for batch generation
        doc_batches: dict[str, tuple[str, int, list[int]]] = {}
        for doc_text, doc_id, cluster_id, _slot in interleaved:
            if doc_id not in doc_batches:
                doc_batches[doc_id] = (doc_text, cluster_id, [])
            doc_batches[doc_id][2].append(0)  # placeholder, count slots

        # Assign Bloom levels with a global counter for consistent distribution
        doc_bloom_levels: dict[str, list[dict]] = {}
        global_slot = 0
        total_question_slots = 0
        for doc_id, (_, _, slot_list) in doc_batches.items():
            k = len(slot_list)
            total_question_slots += k
            blooms: list[dict] = []
            for _ in range(k):
                blooms.append(BLOOM_LEVELS[BLOOM_LEVEL_WEIGHTS[global_slot % len(BLOOM_LEVEL_WEIGHTS)]])
                global_slot += 1
            doc_bloom_levels[doc_id] = blooms

        n_unique_docs = len(doc_batches)
        logger.info(
            "Generating %d candidates from %d documents in batch mode (concurrency=%d)",
            total_question_slots,
            n_unique_docs,
            self.concurrency,
        )

        _TRANSIENT_ERROR = object()
        results_by_doc: dict[str, list[MCQQuestion] | object] = {}
        global_failures: dict[str, int] = {}

        sem = asyncio.Semaphore(self.concurrency)

        async def _bounded(doc_id: str, doc_text: str, cluster_id: int, bloom_levels: list[dict]) -> None:
            excl_q = list(exclude_questions_by_doc.get(doc_id, []))
            excl_f = list(exclude_facts_by_doc.get(doc_id, []))
            async with sem:
                result = await self._generate_batch_single(
                    doc_text,
                    doc_id,
                    cluster_id,
                    bloom_levels,
                    excl_q,
                    excl_f,
                    global_failures,
                    _TRANSIENT_ERROR,
                )
            results_by_doc[doc_id] = result

        with tqdm(total=n_unique_docs, desc="Generating exam questions", unit="doc") as pbar:

            async def _bounded_with_progress(
                doc_id: str, doc_text: str, cluster_id: int, bloom_levels: list[dict]
            ) -> None:
                await _bounded(doc_id, doc_text, cluster_id, bloom_levels)
                pbar.update(1)

            await asyncio.gather(
                *[
                    _bounded_with_progress(doc_id, doc_text, cluster_id, doc_bloom_levels[doc_id])
                    for doc_id, (doc_text, cluster_id, _) in doc_batches.items()
                ]
            )

        # Retry transient errors
        for retry_round, cooldown in enumerate(_RETRY_COOLDOWNS, start=1):
            error_doc_ids = [d_id for d_id, r in results_by_doc.items() if r is _TRANSIENT_ERROR]
            if not error_doc_ids:
                break
            tqdm.write(
                f"\n  {len(error_doc_ids)} batch generation(s) failed"
                f" — retrying after {cooldown}s cooldown"
                f" (round {retry_round}/{len(_RETRY_COOLDOWNS)})"
            )
            await asyncio.sleep(cooldown)

            with tqdm(total=len(error_doc_ids), desc=f"Retry round {retry_round}", unit="doc") as pbar:

                async def _retry_bounded(doc_id: str) -> None:
                    doc_text, cluster_id, _ = doc_batches[doc_id]
                    await _bounded(doc_id, doc_text, cluster_id, doc_bloom_levels[doc_id])
                    pbar.update(1)

                await asyncio.gather(*[_retry_bounded(d_id) for d_id in error_doc_ids])

        still_failed = sum(1 for r in results_by_doc.values() if r is _TRANSIENT_ERROR)
        if still_failed:
            tqdm.write(
                f"\n  {still_failed} batch generation(s) still failed after {len(_RETRY_COOLDOWNS)} retry rounds"
            )

        # Flatten results
        questions: list[MCQQuestion] = []
        for doc_id in doc_batches:
            result = results_by_doc.get(doc_id)
            if result is not _TRANSIENT_ERROR and isinstance(result, list):
                questions.extend(result)

        n_generated = len(questions)
        n_failed = total_question_slots - n_generated
        run_logger = logging.getLogger("agentic_autorag.run")
        run_logger.info(
            "Generated %d/%d candidate questions (%d failed generation)",
            n_generated,
            total_question_slots,
            n_failed,
        )
        if global_failures:
            run_logger.info("Generation failure statistics: %s", _format_failure_stats(global_failures))

        questions = self._deduplicate_exam(questions)
        n_after_dedup = len(questions)
        if n_after_dedup < n_generated:
            run_logger.info(
                "Deduplication: removed %d near-duplicate questions (%d remaining)",
                n_generated - n_after_dedup,
                n_after_dedup,
            )

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

    async def _generate_mcq_batch(
        self,
        doc_text: str,
        doc_id: str,
        cluster_id: int,
        bloom_levels: list[dict],
        exclude_questions: list[str],
        exclude_facts: list[str],
    ) -> list[MCQQuestion | None]:
        """Generate K MCQs from a single document in one LLM call.

        Returns a list of parsed questions (some may be None if individual
        elements failed validation).
        """
        labels = list(MCQ_OPTION_LABELS)
        option_dict_hint = ", ".join(f'"{lbl}": "..."' for lbl in labels)
        k = len(bloom_levels)

        exclude_section = self._build_avoid_section(exclude_questions, exclude_facts)

        bloom_lines: list[str] = []
        for i, bloom in enumerate(bloom_levels, start=1):
            bloom_lines.append(
                f"{i}. Question {i} — Cognitive level: {bloom['level']}\n"
                f"   {bloom['instruction']}\n"
                f'   Example: "{bloom["example"]}"'
            )
        bloom_instructions = "\n\n".join(bloom_lines)

        system_prompt = MCQ_GENERATION_SYSTEM_PROMPT.format(
            domain_description=self.corpus_description or "General enterprise documents.",
        )
        user_prompt = MCQ_BATCH_USER_PROMPT.format(
            doc_text=doc_text,
            exclude_section=exclude_section,
            k=k,
            bloom_instructions=bloom_instructions,
            option_dict_hint=option_dict_hint,
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
        bloom_level_names = [b["level"] for b in bloom_levels]
        return self._parse_batch_response(raw, doc_id, cluster_id, bloom_level_names)

    async def _generate_batch_for_document(
        self,
        doc_text: str,
        doc_id: str,
        cluster_id: int,
        bloom_levels: list[dict],
        exclude_questions: list[str],
        exclude_facts: list[str],
        global_failures: dict[str, int],
    ) -> list[MCQQuestion]:
        """Generate a batch of MCQs for one document, applying structural checks.

        No per-question retry loop — with K diverse questions generated
        simultaneously, retrying the same prompt is unlikely to help.
        Backfill rounds handle deficits.
        """
        parsed = await self._generate_mcq_batch(
            doc_text,
            doc_id,
            cluster_id,
            bloom_levels,
            exclude_questions,
            exclude_facts,
        )

        if not parsed:
            global_failures["parse"] = global_failures.get("parse", 0) + len(bloom_levels)
            return []

        passed: list[MCQQuestion] = []
        for mcq in parsed:
            if mcq is None:
                global_failures["parse"] = global_failures.get("parse", 0) + 1
                continue
            sc_failure = self._self_contained_failure(mcq.question)
            if sc_failure is not None:
                pattern_idx, matched = sc_failure
                global_failures["self_contained"] = global_failures.get("self_contained", 0) + 1
                bucket = f"self_contained.pattern_{pattern_idx}"
                global_failures[bucket] = global_failures.get(bucket, 0) + 1
                logger.info(
                    "SELF_CONTAINED_FAIL doc %s pattern=%d match=%r question=%s",
                    doc_id,
                    pattern_idx,
                    matched,
                    mcq.question,
                )
                continue
            sf_reason = self._source_fact_failure_reason(mcq.source_fact, doc_text)
            if sf_reason is not None:
                global_failures["source_fact"] = global_failures.get("source_fact", 0) + 1
                bucket = f"source_fact.{sf_reason}"
                global_failures[bucket] = global_failures.get(bucket, 0) + 1
                preview = mcq.source_fact[0][:160] if mcq.source_fact else "(empty)"
                logger.info(
                    "SOURCE_FACT_FAIL doc %s reason=%s n_spans=%d total_len=%d preview=%r",
                    doc_id,
                    sf_reason,
                    len(mcq.source_fact),
                    sum(len(s) for s in mcq.source_fact),
                    preview,
                )
                continue
            passed.append(self._shuffle_options(mcq))

        return passed

    async def _generate_batch_single(
        self,
        doc_text: str,
        doc_id: str,
        cluster_id: int,
        bloom_levels: list[dict],
        exclude_questions: list[str],
        exclude_facts: list[str],
        global_failures: dict[str, int],
        transient_sentinel: object,
    ) -> list[MCQQuestion] | object:
        """Wrapper around _generate_batch_for_document that catches transient errors."""
        try:
            return await self._generate_batch_for_document(
                doc_text,
                doc_id,
                cluster_id,
                bloom_levels,
                exclude_questions,
                exclude_facts,
                global_failures,
            )
        except Exception as exc:
            if is_transient_llm_error(exc):
                error_summary = format_llm_error(exc)
                tqdm.write(f"  TRANSIENT ERROR doc {doc_id} | {error_summary}")
                logger.debug("MCQ batch generation transient error for doc %s", doc_id, exc_info=True)
                return transient_sentinel
            error_summary = format_llm_error(exc)
            tqdm.write(f"  ERROR doc {doc_id} | {error_summary}")
            logger.debug("MCQ batch generation failed for doc %s", doc_id, exc_info=True)
            return []

    @staticmethod
    def _dict_to_mcq(
        data: dict,
        doc_id: str,
        cluster_id: int,
        bloom_level: str = "",
    ) -> MCQQuestion | None:
        """Convert a parsed JSON dict into an MCQQuestion.

        Returns None when required fields are missing or invalid.
        """
        try:
            # source_fact may arrive as list[str] (new format) or str (lenient fallback
            # for models that ignored the array instruction). Pydantic's coercer
            # wraps single strings into a single-element list.
            raw_source_fact = data.get("source_fact", [])
            return MCQQuestion(
                id="unset",
                question=data["question"],
                options=data["options"],
                correct_answer=data["correct_answer"],
                source_doc_ids=[doc_id],
                source_fact=raw_source_fact,
                bloom_level=bloom_level,
                cluster_id=cluster_id,
            )
        except (KeyError, ValueError) as exc:
            logger.info("MCQ dict missing required fields for doc %s: %s", doc_id, exc)
            return None

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
        text = self._strip_markdown_fences(raw)

        data = self._try_parse_json(text)
        if data is None:
            data = self._extract_json_object(text)
        if data is None:
            logger.info("JSON parse failed for doc %s: %.200s", doc_id, raw)
            return None

        return self._dict_to_mcq(data, doc_id, cluster_id, bloom_level)

    def _parse_batch_response(
        self,
        raw: str,
        doc_id: str,
        cluster_id: int,
        bloom_levels: list[str],
    ) -> list[MCQQuestion | None]:
        """Parse a JSON array of MCQ objects from a batch LLM response.

        Falls back through multiple strategies: direct JSON array parse,
        bracket-depth extraction, and single-object fallback. Returns a
        list where each element is either a parsed MCQQuestion or None
        for elements that failed validation.
        """
        text = self._strip_markdown_fences(raw)

        # Try direct JSON parse — accept both arrays and single objects.
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        items: list[dict] | None = None
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                items = [item for item in data if isinstance(item, dict)]
            elif isinstance(data, dict):
                # LLM ignored the array instruction and returned a single object.
                items = [data]
        except json.JSONDecodeError:
            pass

        # Fallback: bracket-depth array extraction
        if not items:
            items = self._extract_json_array(text)

        # Fallback: single object (LLM ignored array instruction + mixed text)
        if not items:
            single = self._try_parse_json(text)
            if single is None:
                single = self._extract_json_object(text)
            if single is not None:
                items = [single]

        if not items:
            logger.info("Batch JSON parse failed for doc %s: %.200s", doc_id, raw)
            return []

        results: list[MCQQuestion | None] = []
        for i, item in enumerate(items):
            bloom = bloom_levels[i] if i < len(bloom_levels) else ""
            results.append(self._dict_to_mcq(item, doc_id, cluster_id, bloom))
        return results

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
    def _self_contained_failure(question_text: str) -> tuple[int, str] | None:
        """Return (pattern_index, matched_snippet) for the first failing filter,
        or None when the question is self-contained."""
        for idx, pattern in enumerate(SELF_CONTAINED_FILTERS):
            m = pattern.search(question_text)
            if m:
                return idx, m.group(0)
        return None

    @classmethod
    def _is_self_contained(cls, question_text: str) -> bool:
        """Return True when the question text passes every self-containment filter."""
        return cls._self_contained_failure(question_text) is None

    def _is_source_fact_valid(self, source_fact: list[str], doc_text: str) -> bool:
        """Return True when source_fact passes every pre-filter check."""
        return self._source_fact_failure_reason(source_fact, doc_text) is None

    def _source_fact_failure_reason(self, source_fact: list[str], doc_text: str) -> str | None:
        """Return None when source_fact is a usable list of verbatim spans,
        otherwise a short reason string: ``empty``, ``too_short``,
        ``empty_span``, or ``not_in_doc``.

        Checks:
          - non-empty list
          - total span length ≥ ``source_fact_min_length``
          - every span's normalized form is findable in the normalized doc
            (primary: exact substring; fallback: whitespace-collapsed substring).

        Note: this is a cheap pre-filter. The full verify-and-locate step in
        ``exam_validator.verify_source_facts`` re-runs the check with a fuzzy
        snap-to-source fallback and records offsets.
        """
        if not source_fact:
            return "empty"

        total_len = sum(len(" ".join(span.split())) for span in source_fact)
        if total_len < self.config.source_fact_min_length:
            return "too_short"

        collapsed_doc = re.sub(r"\s+", " ", doc_text)
        for span in source_fact:
            if not span or not span.strip():
                return "empty_span"
            if doc_text.find(span) >= 0:
                continue
            collapsed_span = re.sub(r"\s+", " ", span).strip()
            if collapsed_span and collapsed_doc.find(collapsed_span) >= 0:
                continue
            return "not_in_doc"
        return None

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
        """Batch-filter questions using two discriminator filters (Guinet et al., A.2).

        Two independent filters, each targeting ~``discriminator_removal_pct`` removal:

        * **Extra-candidate**: a distractor is more similar to the source
          than the correct answer (Jaccard OR embedding).
        * **Intra-candidate**: a distractor is too similar to the correct
          answer itself (Jaccard OR embedding).

        Thresholds are calibrated per filter so that the combined OR
        condition within each filter removes ~5 % of questions.
        """
        if len(questions) < 5:
            logger.info("Too few candidates (%d) for batch quality filter, skipping", len(questions))
            return questions

        # Compute per-question metrics — use the joined source_fact spans as the
        # source text. Spans already include surrounding context by construction.
        all_metrics: list[dict[str, float]] = []
        for q in questions:
            source_text = "\n\n".join(q.source_fact) if q.source_fact else ""
            all_metrics.append(self._compute_quality_metrics(q, source_text))

        # --- Filter 1: Extra-candidate (distractor closer to source than correct answer) ---
        # Per-question worst-case across Jaccard and embedding
        extra_worst = [max(m["extra_jaccard_gap"], m["extra_embed_gap"]) for m in all_metrics]
        extra_threshold = float(np.percentile(extra_worst, (1.0 - self.config.discriminator_removal_pct) * 100.0))

        # --- Filter 2: Intra-candidate (distractor too similar to correct answer) ---
        intra_worst = [max(m["intra_jaccard_max"], m["intra_embed_max"]) for m in all_metrics]
        intra_threshold = float(np.percentile(intra_worst, (1.0 - self.config.discriminator_removal_pct) * 100.0))

        logger.info(
            "Discriminator thresholds (target %.0f%% removal each): extra=%.3f, intra=%.3f",
            self.config.discriminator_removal_pct * 100.0,
            extra_threshold,
            intra_threshold,
        )

        passed: list[MCQQuestion] = []
        removed_extra = 0
        removed_intra = 0

        for q, _metrics, ew, iw in zip(questions, all_metrics, extra_worst, intra_worst, strict=True):
            fail_reason = None
            if ew > extra_threshold:
                fail_reason = f"extra-candidate ({ew:.3f} > {extra_threshold:.3f})"
                removed_extra += 1
            elif iw > intra_threshold:
                fail_reason = f"intra-candidate ({iw:.3f} > {intra_threshold:.3f})"
                removed_intra += 1

            if fail_reason is not None:
                _log_quality_failure(logger, reason=fail_reason, q=q)
            else:
                passed.append(q)

        n_removed = len(questions) - len(passed)
        logger.info(
            "Discriminator quality filter: removed %d questions (extra=%d, intra=%d), %d remaining",
            n_removed,
            removed_extra,
            removed_intra,
            len(passed),
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
    def _strip_markdown_fences(raw: str) -> str:
        """Strip markdown code fences from LLM output."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return text

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

    @staticmethod
    def _extract_json_array(text: str) -> list[dict] | None:
        """Extract the first JSON array from mixed text."""
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
                        if isinstance(data, list):
                            return [item for item in data if isinstance(item, dict)]
                        return None
                    except json.JSONDecodeError:
                        return None
        return None
