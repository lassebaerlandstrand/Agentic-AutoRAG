"""Tests for exam generation quality filters."""

from __future__ import annotations

import numpy as np

from agentic_autorag.config.models import ExaminerConfig, MCQQuestion
from agentic_autorag.examiner.exam_agent import ExamAgent


class DummyEmbeddingModel:
    """Simple deterministic embedding model for similarity checks."""

    def encode(self, texts: list[str]):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    float(lower.count("retrieval")),
                    float(lower.count("generation")),
                    float(len(lower.split())),
                    float(sum(ord(ch) for ch in lower) % 4099),
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


def _agent() -> ExamAgent:
    return ExamAgent(
        config=ExaminerConfig(exam_size=10, diversity_clusters=2),
        examiner_model="gemini/gemini-3-flash-preview",
        embedding_model=DummyEmbeddingModel(),
    )


def _question(correct_answer: str = "A") -> MCQQuestion:
    return MCQQuestion(
        id="q1",
        question="What is the primary goal of retrieval-augmented generation?",
        options={
            "A": "Ground model outputs with retrieved external context.",
            "B": "Increase GPU clock speed during decoding.",
            "C": "Replace retrieval with random sampling.",
            "D": "Train only on synthetic noise data.",
        },
        correct_answer=correct_answer,
        source_chunk_id="chunk_1",
        cluster_id=0,
    )


class TestExamQuality:
    def test_shuffle_options_updates_correct_answer(self) -> None:
        agent = _agent()
        mcq = _question(correct_answer="A")
        counts = {"A": 0, "B": 0, "C": 0, "D": 0}

        for _ in range(200):
            shuffled = agent._shuffle_options(mcq)
            counts[shuffled.correct_answer] += 1

        assert all(value > 20 for value in counts.values())

    def test_self_contained_filter_rejects_doc_reference(self) -> None:
        agent = _agent()
        assert not agent._is_self_contained("According to the documentation, what does this API do?")
        assert not agent._is_self_contained("Based on the provided text, what is the right answer?")
        assert agent._is_self_contained("What behavior does the API exhibit when retries are enabled?")

    def test_exam_deduplicates_near_identical_questions(self) -> None:
        agent = _agent()

        question_a = _question()
        question_b = question_a.model_copy(update={"id": "q2", "source_chunk_id": "chunk_2"})
        question_c = question_a.model_copy(
            update={
                "id": "q3",
                "question": "How should an engineer tune retrieval to reduce hallucinations?",
                "source_chunk_id": "chunk_3",
            }
        )

        def _encode(texts: list[str]):
            mapping = {
                question_a.question: np.array([1.0, 0.0, 0.0, 0.0]),
                question_c.question: np.array([0.0, 1.0, 0.0, 0.0]),
            }
            return np.asarray([mapping[text] for text in texts], dtype=np.float32)

        agent.embedding_model.encode = _encode
        deduped = agent._deduplicate_exam([question_a, question_b, question_c])
        assert [question.id for question in deduped] == ["q1", "q3"]

    def test_extra_candidate_similarity_rejects_bad_discriminator(self) -> None:
        agent = _agent()
        bad_mcq = MCQQuestion(
            id="q2",
            question="Which statement is correct?",
            options={
                "A": "Irrelevant short answer.",
                "B": "This source chunk says retrieval improves factual grounding.",
                "C": "Another unrelated sentence.",
                "D": "Completely different claim.",
            },
            correct_answer="A",
            source_chunk_id="chunk_2",
            cluster_id=1,
        )
        source_chunk = "This source chunk says retrieval improves factual grounding."

        assert not agent._check_discriminator_quality(bad_mcq, source_chunk)

    def test_intra_candidate_similarity_rejects_rephrased_correct_answer(self) -> None:
        agent = _agent()
        bad_mcq = MCQQuestion(
            id="q3",
            question="Which option is best?",
            options={
                "A": "Ground model outputs with retrieved external context.",
                "B": "Ground model outputs with retrieved external context.",
                "C": "Disable retrieval and rely on guessing.",
                "D": "Use random tokens to rank passages.",
            },
            correct_answer="A",
            source_chunk_id="chunk_3",
            cluster_id=1,
        )

        assert not agent._check_discriminator_quality(bad_mcq, "retrieval external context grounding")
