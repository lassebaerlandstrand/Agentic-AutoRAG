"""Exam Agent — generates MCQ exams from corpus chunks with diversity guarantees.

Uses a dedicated LLM (the examiner model) via LiteLLM to produce
multiple-choice questions from sampled chunks. Clustering and allocation
ensure the exam covers the full breadth of the corpus.
"""

from __future__ import annotations

import json
import logging
import uuid

import litellm
import numpy as np
from tqdm import tqdm

from agentic_autorag.config.models import ExaminerConfig, MCQQuestion
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


class ExamAgent:
    """Generates MCQ exams from corpus chunks with diversity guarantees.

    Uses a dedicated Examiner Agent for MCQ generation.
    """

    DEFAULT_MAX_RETRIES = 3

    def __init__(
        self,
        config: ExaminerConfig,
        examiner_model: str,
        temperature: float = 1.0,
    ) -> None:
        self.config = config
        self.examiner_model = examiner_model
        self.temperature = temperature

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
                mcq = await self._generate_mcq_with_retry(chunks[idx], chunk_ids[idx], cluster_id)
                if mcq is not None:
                    questions.append(mcq)
                    generated += 1
                    pbar.update(1)

        pbar.close()
        return questions

    async def _generate_mcq_with_retry(
        self,
        chunk: str,
        chunk_id: str,
        cluster_id: int,
    ) -> MCQQuestion | None:
        """Attempt to generate an MCQ, retrying on failure."""
        for attempt in range(self.DEFAULT_MAX_RETRIES):
            try:
                mcq = await self._generate_mcq(chunk, chunk_id, cluster_id)
                if mcq is not None:
                    return mcq
            except Exception:
                logger.warning("MCQ generation attempt %d failed", attempt + 1, exc_info=True)
        return None

    async def _generate_mcq(
        self,
        chunk: str,
        chunk_id: str,
        cluster_id: int,
    ) -> MCQQuestion | None:
        """Generate a single MCQ from a document chunk using the examiner LLM."""
        n = self.config.mcq_options_count
        labels = [chr(ord("A") + i) for i in range(n)]
        option_labels = ", ".join(labels)
        option_dict_hint = ", ".join(f'"{lbl}": "..."' for lbl in labels)

        prompt = MCQ_GENERATION_PROMPT.format(
            chunk=chunk,
            n_options=n,
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

        Handles optional markdown code fences (```json ... ```) that many
        models wrap around their output.  Returns ``None`` on any parse or
        validation failure.
        """
        try:
            text = raw.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                # Drop the opening fence line (e.g. ```json)
                lines = lines[1:]
                # Drop the closing fence if present
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)

            data = json.loads(text)

            return MCQQuestion(
                id=str(uuid.uuid4()),
                question=data["question"],
                options=data["options"],
                correct_answer=data["correct_answer"],
                source_chunk_id=chunk_id,
                cluster_id=cluster_id,
                guessing=1.0 / self.config.mcq_options_count,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("Failed to parse MCQ response", exc_info=True)
            return None
