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
        config=ExaminerConfig(exam_size=10),
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
        source_doc_ids=["doc_1"],
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
        qs = [
            "According to the documentation, what does this API do?",
            "Based on the provided text, what is the right answer?",
            "In this report, what amount was disclosed?",
            "From the document, what date was recorded?",
            "As stated above, which requirement applies?",
            "What behavior does the API exhibit when retries are enabled?",
        ]

        results = [agent._is_self_contained(q) for q in qs]

        assert not results[0]
        assert not results[1]
        assert not results[2]
        assert not results[3]
        assert not results[4]
        assert results[5]

    def test_exam_deduplicates_near_identical_questions(self) -> None:
        agent = _agent()
        question_a = _question()
        question_b = question_a.model_copy(update={"id": "q2", "source_doc_ids": ["doc_2"]})
        question_c = question_a.model_copy(
            update={
                "id": "q3",
                "question": "How should an engineer tune retrieval to reduce hallucinations?",
                "source_doc_ids": ["doc_3"],
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

    def test_compute_quality_metrics_extra_gap_high_when_distractor_matches_source(self) -> None:
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
            source_doc_ids=["doc_2"],
            cluster_id=1,
        )
        source_chunk = "This source chunk says retrieval improves factual grounding."

        metrics = agent._compute_quality_metrics(bad_mcq, source_chunk)

        # Distractor B is nearly identical to source — extra gaps should be large
        assert metrics["extra_jaccard_gap"] > 0.1
        assert metrics["extra_embed_gap"] > 0.0

    def test_compute_quality_metrics_intra_high_when_options_identical(self) -> None:
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
            source_doc_ids=["doc_3"],
            cluster_id=1,
        )

        metrics = agent._compute_quality_metrics(bad_mcq, "retrieval external context grounding")

        # Option B is identical to correct A — intra similarity should be 1.0
        assert metrics["intra_jaccard_max"] >= 0.99
        assert metrics["intra_embed_max"] >= 0.99

    def test_batch_filter_removes_worst_questions(self) -> None:
        """The batch filter should remove questions with the worst metrics."""
        agent = _agent()
        source_text = "This source chunk says retrieval improves factual grounding."
        documents = {"doc_1": source_text, "doc_2": source_text, "doc_3": source_text}

        # Good question — options are distinct from source and each other
        good_q = MCQQuestion(
            id="q_good",
            question="What is the goal?",
            options={
                "A": "Ground model outputs with retrieved external context.",
                "B": "Increase GPU clock speed during decoding.",
                "C": "Replace retrieval with random sampling.",
                "D": "Train only on synthetic noise data.",
            },
            correct_answer="A",
            source_doc_ids=["doc_1"],
            cluster_id=0,
        )
        # Bad question — distractor B is identical to source
        bad_q = MCQQuestion(
            id="q_bad",
            question="Which statement is correct?",
            options={
                "A": "Irrelevant short answer.",
                "B": "This source chunk says retrieval improves factual grounding.",
                "C": "Another unrelated sentence.",
                "D": "Completely different claim.",
            },
            correct_answer="A",
            source_doc_ids=["doc_2"],
            cluster_id=1,
        )

        # With only 2 questions, batch filter skips (< 5 threshold)
        # So let's create enough questions to trigger filtering
        filler = [good_q.model_copy(update={"id": f"q_filler_{i}", "source_doc_ids": ["doc_3"]}) for i in range(8)]
        all_questions = [good_q, bad_q, *filler]

        result = agent._filter_discriminator_quality(all_questions, documents)

        # bad_q should be among those removed (worst extra_* metric)
        result_ids = {q.id for q in result}
        assert "q_good" in result_ids
        # With 10 questions and 5% removal per metric, ~2 get removed total
        assert len(result) < len(all_questions)
