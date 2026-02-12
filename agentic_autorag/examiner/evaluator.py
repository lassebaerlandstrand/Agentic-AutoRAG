"""MCQ evaluator — runs an exam against a RAG pipeline and scores the results."""

from __future__ import annotations

import re

from pydantic import BaseModel
from tqdm import tqdm

from agentic_autorag.config.models import MCQQuestion
from agentic_autorag.engine.pipeline import RAGPipeline


class QuestionResult(BaseModel):
    """Result of evaluating a single MCQ question."""

    question_id: str
    correct: bool
    selected_answer: str  # option letter or "INVALID"
    correct_answer: str
    retrieved_context: str
    generated_response: str


class ExamResult(BaseModel):
    """Aggregated result of evaluating a full MCQ exam."""

    score: float
    n_correct: int
    n_total: int
    question_results: list[QuestionResult]

    def failed_questions(self) -> list[QuestionResult]:
        """Return only the incorrect question results."""
        return [qr for qr in self.question_results if not qr.correct]


class MCQEvaluator:
    """Evaluates a RAG pipeline against an MCQ exam."""

    MCQ_ANSWER_PROMPT = """\
Answer the following multiple-choice question based ONLY on the provided context. \
Reply with just the letter ({option_labels}).

Context:
{context}

Question: {question}
{options}

Answer:"""

    async def evaluate(
        self,
        pipeline: RAGPipeline,
        exam: list[MCQQuestion],
    ) -> ExamResult:
        """Run every question through the pipeline and aggregate scores."""
        results: list[QuestionResult] = []

        for q in tqdm(exam, desc="Evaluating MCQs", unit="q"):
            retrieval_result = await pipeline.retrieve(q.question)
            context = "\n".join(doc.text for doc in retrieval_result.documents)

            option_labels = ", ".join(sorted(q.options.keys()))
            options_text = "\n".join(f"{k}) {v}" for k, v in q.options.items())
            prompt = self.MCQ_ANSWER_PROMPT.format(
                option_labels=option_labels,
                context=context,
                question=q.question,
                options=options_text,
            )

            answer = await pipeline.generate(prompt)
            selected = self._parse_answer(answer, valid_keys=set(q.options.keys()))

            results.append(
                QuestionResult(
                    question_id=q.id,
                    correct=selected == q.correct_answer,
                    selected_answer=selected,
                    correct_answer=q.correct_answer,
                    retrieved_context=context,
                    generated_response=answer,
                )
            )

        n_correct = sum(1 for r in results if r.correct)
        n_total = len(results)
        return ExamResult(
            score=n_correct / n_total if n_total else 0.0,
            n_correct=n_correct,
            n_total=n_total,
            question_results=results,
        )

    @staticmethod
    def _parse_answer(response: str, valid_keys: set[str]) -> str:
        """Extract the first valid option letter from a free-form LLM response.

        Handles formats like "B", "b", "The answer is B", "B)", "B.", etc.
        Returns "INVALID" if no valid letter is found.
        """
        text = response.strip()

        # Build pattern from valid keys (e.g. A|B|C|D)
        keys_upper = sorted(valid_keys)
        keys_pattern = "|".join(keys_upper)

        # Try common patterns in priority order
        patterns = [
            # "The answer is B" / "answer: B"
            rf"(?:the\s+)?answer\s*(?:is|:)\s*({keys_pattern})\b",
            # Standalone letter possibly followed by ) or . or :
            rf"\b({keys_pattern})\s*[).:]\s",
            # Letter at the very start or end of the response
            rf"^({keys_pattern})\b",
            rf"\b({keys_pattern})$",
            # Any occurrence as a fallback
            rf"\b({keys_pattern})\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return "INVALID"
