"""Exam Agent — generates MCQ exams from corpus chunks with diversity guarantees.

Uses a dedicated LLM (the examiner model) via LiteLLM to produce
multiple-choice questions from sampled chunks. Clustering and allocation
ensure the exam covers the full breadth of the corpus.
"""

from __future__ import annotations

import json
import logging
import random
import re
import uuid

import litellm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from agentic_autorag.config.models import MCQ_OPTIONS, MCQ_OPTION_LABELS, ExaminerConfig, MCQQuestion
from agentic_autorag.examiner.clustering import (
    allocate_largest_remainder,
    compute_clusters,
    resolve_n_clusters,
)

logger = logging.getLogger(__name__)

MCQ_GENERATION_PROMPT = """\
Based on the following text, create a multiple-choice question that tests \
understanding of the content. The question should require comprehension, \
not just keyword matching.

TEXT:
{chunk}

Create a question with {n_options} options (labeled {option_labels}), \
where exactly one is correct and the others are plausible but incorrect \
distractors.

Return ONLY a valid JSON object with these exact fields:
- "question": the question text
- "options": {{{option_dict_hint}}}
- "correct_answer": the letter of the correct option (e.g., "A")

Return ONLY valid JSON, no markdown formatting or additional text."""

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
        r"\baccording\s+to\s+(the\s+)?(documentation|paper|article|passage|text)\b",
        re.IGNORECASE,
    ),
]


class ExamAgent:
    """Generates MCQ exams from corpus chunks with diversity guarantees.

    Uses a dedicated Examiner Agent for MCQ generation.
    """

    DEFAULT_MAX_RETRIES = 3
    JACCARD_EXTRA_THRESHOLD = 0.05
    EMBED_EXTRA_THRESHOLD = 0.05
    JACCARD_INTRA_THRESHOLD = 0.70
    EMBED_INTRA_THRESHOLD = 0.85

    def __init__(
        self,
        config: ExaminerConfig,
        examiner_model: str,
        embedding_model,
        temperature: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        self.config = config
        self.examiner_model = examiner_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self._rng = random.Random(random_seed)

    async def generate_exam(
        self,
        chunks: list[str],
        chunk_ids: list[str],
        embeddings: np.ndarray,
    ) -> list[MCQQuestion]:
        """Generate a diverse MCQ exam from corpus chunks.

        Steps:
          1. Cluster chunks into knowledge regions.
          2. Allocate question slots per cluster.
          3. For each cluster, sample chunks and generate MCQs with retry.
        """
        n_clusters = resolve_n_clusters(len(chunks), self.config.exam_size, self.config.diversity_clusters)
        labels = compute_clusters(embeddings, n_clusters)
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        allocations = allocate_largest_remainder(cluster_sizes, self.config.exam_size)

        questions: list[MCQQuestion] = []
        pbar = tqdm(total=self.config.exam_size, desc="Generating MCQs", unit="q")

        for cluster_id in range(n_clusters):
            target = allocations[cluster_id]
            if target == 0:
                continue

            cluster_indices = np.where(labels == cluster_id)[0]
            rng = np.random.default_rng(seed=42 + cluster_id)
            rng.shuffle(cluster_indices)
            chunk_pool = list(cluster_indices)

            generated = 0
            while generated < target and chunk_pool:
                idx = chunk_pool.pop(0)
                mcq = await self._generate_mcq_for_chunk(chunks[idx], chunk_ids[idx], cluster_id)
                if mcq is not None:
                    questions.append(mcq)
                    generated += 1
                    pbar.update(1)

        pbar.close()
        return questions

    async def _generate_mcq_for_chunk(
        self,
        chunk: str,
        chunk_id: str,
        cluster_id: int,
    ) -> MCQQuestion | None:
        """Generate one high-quality MCQ for a chunk.

        The quality pipeline follows Guinet et al. (ICML 2024, Appendix A.2):
        generation/parsing, self-contained filtering, option shuffling, and
        discriminator quality filtering.
        """
        for attempt in range(self.DEFAULT_MAX_RETRIES):
            try:
                mcq = await self._generate_mcq(chunk, chunk_id, cluster_id)
                if mcq is None:
                    continue
                if not self._is_self_contained(mcq.question):
                    continue

                mcq = self._shuffle_options(mcq)
                if not self._check_discriminator_quality(mcq, chunk):
                    return None

                return mcq
            except Exception:
                logger.debug("MCQ generation attempt %d failed", attempt + 1, exc_info=True)
        logger.warning("All %d MCQ generation attempts failed for chunk %s", self.DEFAULT_MAX_RETRIES, chunk_id)
        return None

    async def _generate_mcq_with_retry(
        self,
        chunk: str,
        chunk_id: str,
        cluster_id: int,
    ) -> MCQQuestion | None:
        """Backward-compatible wrapper around the quality-aware generation flow."""
        return await self._generate_mcq_for_chunk(chunk, chunk_id, cluster_id)

    async def _generate_mcq(
        self,
        chunk: str,
        chunk_id: str,
        cluster_id: int,
    ) -> MCQQuestion | None:
        """Generate a single MCQ from a document chunk using the examiner LLM."""
        labels = list(MCQ_OPTION_LABELS)
        option_labels = ", ".join(labels)
        option_dict_hint = ", ".join(f'"{lbl}": "..."' for lbl in labels)

        prompt = MCQ_GENERATION_PROMPT.format(
            chunk=chunk,
            n_options=MCQ_OPTIONS,
            option_labels=option_labels,
            option_dict_hint=option_dict_hint,
        )

        response = await litellm.acompletion(
            model=self.examiner_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        raw = response.choices[0].message.content
        return self._parse_mcq_response(raw, chunk_id, cluster_id)

    def _parse_mcq_response(
        self,
        raw: str,
        chunk_id: str,
        cluster_id: int,
    ) -> MCQQuestion | None:
        """Parse the LLM's JSON response into an MCQQuestion.

        Handles markdown code fences, trailing commas, and mixed
        text/JSON output.  Returns ``None`` on any parse or validation failure.
        """
        try:
            text = raw.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                lines = lines[1:]  # Drop opening fence
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)

            # Try direct parse first
            data = self._try_parse_json(text)

            # Fallback: extract first JSON object from mixed text
            if data is None:
                data = self._extract_json_object(text)

            if data is None:
                logger.debug("Could not parse JSON from MCQ response for chunk %s", chunk_id)
                return None

            return MCQQuestion(
                id=str(uuid.uuid4()),
                question=data["question"],
                options=data["options"],
                correct_answer=data["correct_answer"],
                source_chunk_id=chunk_id,
                cluster_id=cluster_id,
            )
        except (KeyError, ValueError) as exc:
            logger.debug("MCQ response missing required fields for chunk %s: %s", chunk_id, exc)
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
    def _is_self_contained(question_text: str) -> bool:
        """Return True when question text is self-contained."""
        return not any(pattern.search(question_text) for pattern in SELF_CONTAINED_FILTERS)

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

    def _embedding_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between embedding vectors."""
        emb_a = np.asarray(self.embedding_model.encode([text_a]), dtype=np.float32)
        emb_b = np.asarray(self.embedding_model.encode([text_b]), dtype=np.float32)
        return float(cosine_similarity(emb_a, emb_b)[0][0])

    def _check_discriminator_quality(
        self,
        mcq: MCQQuestion,
        source_chunk: str,
        jaccard_extra_threshold: float = JACCARD_EXTRA_THRESHOLD,
        embed_extra_threshold: float = EMBED_EXTRA_THRESHOLD,
        jaccard_intra_threshold: float = JACCARD_INTRA_THRESHOLD,
        embed_intra_threshold: float = EMBED_INTRA_THRESHOLD,
    ) -> bool:
        """Validate discriminator quality with extra- and intra-candidate checks.

        Thresholds are initialized to practical defaults and should be calibrated
        on the target corpus so each discriminator filter removes approximately 5%
        of generated candidate questions.
        """
        correct_text = mcq.options[mcq.correct_answer]
        discriminators = [text for key, text in mcq.options.items() if key != mcq.correct_answer]

        all_answers = list(mcq.options.values())
        mean_token_len = int(np.mean([len(answer.split()) for answer in all_answers]))
        n_gram = max(1, mean_token_len)

        j_correct = self._jaccard_ngram(source_chunk, correct_text, n_gram)
        e_correct = self._embedding_similarity(source_chunk, correct_text)

        for disc in discriminators:
            j_disc = self._jaccard_ngram(source_chunk, disc, n_gram)
            e_disc = self._embedding_similarity(source_chunk, disc)
            if j_correct + jaccard_extra_threshold < j_disc:
                return False
            if e_correct + embed_extra_threshold < e_disc:
                return False

        for disc in discriminators:
            j_intra = self._jaccard_ngram(correct_text, disc, n_gram)
            e_intra = self._embedding_similarity(correct_text, disc)
            if j_intra >= jaccard_intra_threshold:
                return False
            if e_intra >= embed_intra_threshold:
                return False

        return True

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        """Attempt to parse JSON, fixing trailing commas first."""
        # Remove trailing commas before } or ]
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_json_object(text: str) -> dict | None:
        """Extract the first JSON object from mixed text."""
        # Find the first { and try to parse from there
        start = text.find("{")
        if start == -1:
            return None

        # Walk forward to find matching closing brace
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    # Remove trailing commas and try parsing
                    cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        data = json.loads(cleaned)
                        return data if isinstance(data, dict) else None
                    except json.JSONDecodeError:
                        return None
        return None
