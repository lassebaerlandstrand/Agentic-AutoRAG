"""Tests for the Orchestrator — corpus loading, loop mechanics, and fallback."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agentic_autorag.config.models import (
    MCQQuestion,
    ProjectConfig,
    TrialConfig,
)
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.orchestrator import Orchestrator


def _make_config_dict(corpus_path: str, output_dir: str, max_trials: int = 2) -> dict:
    """Return a minimal raw dict that converts to a valid ProjectConfig."""
    return {
        "meta": {
            "project_name": "test",
            "corpus_path": corpus_path,
            "corpus_description": "Test corpus",
            "output_dir": output_dir,
            "max_trials": max_trials,
            "index_registry": False,
        },
        "parsing": {
            "parser": "pymupdf4llm",
            "ocr": False,
            "table_structure": True,
        },
        "search_space": {
            "chunking": {
                "strategies": ["recursive", "fixed"],
                "chunk_token_size": {"min": 128, "max": 1024},
                "chunk_token_overlap": {"min": 0, "max": 128},
            },
            "embedding_models": ["sentence-transformers/all-MiniLM-L6-v2"],
            "index_types": ["vector_only"],
            "top_k": {"min": 3, "max": 10},
            "hybrid_alpha": {"min": 0.0, "max": 1.0},
            "reranker": {"models": ["none"], "top_n": {"min": 3, "max": 5}},
            "query_expansion": ["none"],
            "llm_models": ["ollama/llama3.2"],
            "temperature": {"min": 0.0, "max": 0.7},
        },
        "examiner": {
            "exam_size": 5,
        },
        "agent": {
            "optimizer_model": "test/model",
            "examiner_model": "test/model",
            "max_history_trials": 5,
        },
    }


def _make_trial_config() -> TrialConfig:
    return TrialConfig(
        chunking_strategy="recursive",
        chunk_token_size=512,
        chunk_token_overlap=64,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        top_k=5,
        reranker="none",
        llm_model="ollama/llama3.2",
        temperature=0.0,
    )


def _make_exam(n: int = 3) -> list[MCQQuestion]:
    return [
        MCQQuestion(
            id=f"q{i}",
            question=f"Question {i}?",
            options={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_answer="A",
            source_doc_ids=[f"doc_{i}"],
            cluster_id=0,
        )
        for i in range(n)
    ]


def _make_exam_result(n: int = 3, n_correct: int = 2) -> ExamResult:
    results = []
    for i in range(n):
        correct = i < n_correct
        results.append(
            QuestionResult(
                question_id=f"q{i}",
                correct=correct,
                selected_answer="A" if correct else "B",
                correct_answer="A",
                retrieved_context="some context",
                generated_response="A" if correct else "B",
            )
        )
    return ExamResult(
        score=n_correct / n,
        n_correct=n_correct,
        n_total=n,
        question_results=results,
    )


class TestLoadAndParseCorpus:
    @staticmethod
    def _make_orch(tmp_path: Path, corpus: Path, parser_extensions: set[str] = frozenset()) -> Orchestrator:
        """Build a minimal Orchestrator with corpus caching support."""
        out = tmp_path / "out"
        raw = _make_config_dict(str(corpus), str(out))
        cfg = ProjectConfig.model_validate(raw)
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = cfg
        orch.output_dir = Path(out)
        orch.output_dir.mkdir(parents=True, exist_ok=True)
        orch.logger = logging.getLogger("test")
        orch.parser = MagicMock()
        orch.parser.supported_extensions.return_value = parser_extensions
        return orch

    def test_loads_txt_and_md(self, tmp_path: Path) -> None:
        """Text and markdown files are read directly."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc1.txt").write_text("Hello world")
        (corpus / "doc2.md").write_text("# Heading\nContent")
        (corpus / "metadata.json").write_text("{}")
        (corpus / ".hidden").write_text("secret")

        orch = self._make_orch(tmp_path, corpus, {".pdf"})
        docs = orch._load_and_parse_corpus()
        assert len(docs) == 2
        assert "Hello world" in docs[0]
        assert "Content" in docs[1]

    def test_skips_metadata_and_hidden(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "metadata.json").write_text("{}")
        (corpus / ".hidden").write_text("x")
        (corpus / "real.txt").write_text("real content")

        orch = self._make_orch(tmp_path, corpus)
        docs = orch._load_and_parse_corpus()
        assert len(docs) == 1

    def test_empty_corpus_raises(self, tmp_path: Path) -> None:
        corpus = tmp_path / "empty_corpus"
        corpus.mkdir()

        orch = self._make_orch(tmp_path, corpus)
        docs = orch._load_and_parse_corpus()
        assert docs == []

    def test_corpus_cache_hit(self, tmp_path: Path) -> None:
        """Second call returns cached results without re-parsing."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc.txt").write_text("cached content")

        orch = self._make_orch(tmp_path, corpus)
        docs1 = orch._load_and_parse_corpus()
        assert len(docs1) == 1

        # Second call should hit cache — parser.parse should not be called
        orch.parser.parse.reset_mock()
        docs2 = orch._load_and_parse_corpus()
        assert docs2 == docs1
        orch.parser.parse.assert_not_called()

    def test_corpus_cache_invalidates_on_file_change(self, tmp_path: Path) -> None:
        """Cache invalidates when a file is added."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc.txt").write_text("original")

        orch = self._make_orch(tmp_path, corpus)
        docs1 = orch._load_and_parse_corpus()
        assert len(docs1) == 1

        # Add a new file — cache key should change
        (corpus / "doc2.txt").write_text("new file")
        docs2 = orch._load_and_parse_corpus()
        assert len(docs2) == 2


class TestRunLoop:
    @pytest.mark.asyncio
    async def test_basic_loop(self, tmp_path: Path) -> None:
        """Mock all expensive components and verify the loop runs to completion."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc.txt").write_text("Test document content for chunking.")

        out = tmp_path / "out"
        raw = _make_config_dict(str(corpus), str(out), max_trials=2)

        trial_config = _make_trial_config()
        exam = _make_exam(3)
        exam_result = _make_exam_result(3, 2)

        with (
            patch("agentic_autorag.orchestrator.load_config") as mock_load,
            patch("agentic_autorag.orchestrator.ExamAgent") as MockExamAgent,
            patch("agentic_autorag.orchestrator.run_validation_pipeline", new_callable=AsyncMock) as mock_validate,
            patch("agentic_autorag.orchestrator.IndexBuilder") as MockIndexBuilder,
            patch("agentic_autorag.orchestrator.MCQEvaluator") as MockEvaluator,
            patch("agentic_autorag.orchestrator.ReasoningAgent") as MockAgent,
            patch("agentic_autorag.orchestrator.build_parser") as mock_build_parser,
        ):
            mock_load.return_value = ProjectConfig.model_validate(raw)

            # Parser
            parser_mock = MagicMock()
            parser_mock.supported_extensions.return_value = {".pdf"}
            mock_build_parser.return_value = parser_mock

            # Embedder
            embedder_mock = MagicMock()
            embedder_mock.encode.return_value = np.random.rand(10, 384).astype(np.float32)

            # Exam agent: prepare_corpus (sync) + generate_wave (async)
            mock_exam_inst = MagicMock()
            mock_corpus = MagicMock()
            mock_corpus.doc_texts = ["doc text"]
            mock_corpus.n_clusters = 1
            mock_corpus.cluster_sizes = np.array([1])
            mock_exam_inst.prepare_corpus.return_value = mock_corpus
            mock_exam_inst.generate_wave = AsyncMock(return_value=exam)
            MockExamAgent.return_value = mock_exam_inst
            mock_validate.return_value = exam

            # Index builder
            mock_index = MagicMock()
            mock_index.vector_store = MagicMock()
            mock_index.graph_store = None
            mock_builder = AsyncMock()
            mock_builder.build.return_value = mock_index
            mock_builder.get_embedder = MagicMock(return_value=embedder_mock)
            mock_builder.get_cross_encoder = MagicMock(return_value=MagicMock())
            MockIndexBuilder.return_value = mock_builder

            # Evaluator
            mock_eval = AsyncMock()
            mock_eval.evaluate.return_value = exam_result
            MockEvaluator.return_value = mock_eval

            # Agent
            mock_agent = AsyncMock()
            mock_agent.propose_initial.return_value = trial_config
            next_config = _make_trial_config()
            next_config = next_config.model_copy(update={"top_k": 7})
            mock_agent.analyze_and_propose.return_value = ("error trace", next_config)
            MockAgent.return_value = mock_agent

            orch = Orchestrator(str(tmp_path / "fake_config.yaml"))
            best = await orch.run()

        assert best is not None
        assert best.score == exam_result.score
        assert len(orch.history.records) == 2
        assert (out / "exam.json").exists()


class TestRandomTweak:
    def test_produces_valid_config(self, tmp_path: Path) -> None:
        raw = _make_config_dict(str(tmp_path), str(tmp_path / "out"))
        cfg = ProjectConfig.model_validate(raw)

        orch = Orchestrator.__new__(Orchestrator)
        orch.config = cfg

        config = _make_trial_config()
        tweaked = orch._random_tweak(config)

        assert isinstance(tweaked, TrialConfig)
        # At least one param should differ (with very high probability)
        # but the index-building params should be identical
        assert tweaked.chunk_token_size == config.chunk_token_size
        assert tweaked.embedding_model == config.embedding_model
        assert tweaked.index_type == config.index_type


class TestPrintConfigDiff:
    def test_detects_changes(self, capsys) -> None:
        old = _make_trial_config()
        new = _make_trial_config().model_copy(update={"top_k": 10})

        Orchestrator._print_config_diff(old, new)
        captured = capsys.readouterr()
        assert "top_k" in captured.out
        assert "5 → 10" in captured.out

    def test_no_changes(self, capsys) -> None:
        config = _make_trial_config()
        Orchestrator._print_config_diff(config, config)
        captured = capsys.readouterr()
        assert "no changes" in captured.out


class TestSaveExam:
    def test_saves_valid_json(self, tmp_path: Path) -> None:
        orch = Orchestrator.__new__(Orchestrator)
        orch.output_dir = tmp_path

        exam = _make_exam(2)
        orch._save_exam(exam)

        exam_path = tmp_path / "exam.json"
        assert exam_path.exists()
        data = json.loads(exam_path.read_text())
        assert len(data) == 2
        assert data[0]["id"] == "q0"


class TestExamArtifacts:
    """Tests for canonical exam/candidate artifact behavior in _generate_exam()."""

    @staticmethod
    def _make_orch(tmp_path: Path) -> Orchestrator:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc.txt").write_text("Some content for exam generation.")
        out = tmp_path / "out"
        raw = _make_config_dict(str(corpus), str(out))
        cfg = ProjectConfig.model_validate(raw)
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = cfg
        orch.output_dir = Path(out)
        orch.output_dir.mkdir(parents=True, exist_ok=True)
        orch.logger = logging.getLogger("test")
        orch.index_builder = MagicMock()
        return orch

    @pytest.mark.asyncio
    async def test_loads_exam_from_existing_exam_file(self, tmp_path: Path) -> None:
        """When exam.json exists and is valid, generation is skipped."""
        orch = self._make_orch(tmp_path)

        # Pre-populate the canonical exam artifact
        cached_exam = _make_exam(2)
        exam_path = orch.output_dir / "exam.json"
        exam_path.write_text(
            json.dumps([q.model_dump(mode="json") for q in cached_exam], indent=2),
            encoding="utf-8",
        )

        with patch("agentic_autorag.orchestrator.ExamAgent") as MockExamAgent:
            exam, from_cache = await orch._generate_exam(["Some content."])

        assert from_cache is True
        assert len(exam) == 2
        assert exam[0].id == cached_exam[0].id
        assert exam[1].id == cached_exam[1].id
        MockExamAgent.assert_not_called()

    @pytest.mark.asyncio
    async def test_generates_and_saves_canonical_artifacts_on_miss(self, tmp_path: Path) -> None:
        """On miss, questions are generated and written to candidates.json/exam.json."""
        orch = self._make_orch(tmp_path)
        generated_exam = _make_exam(3)

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros((5, 384), dtype="float32")
        orch.index_builder.get_embedder.return_value = mock_embedder

        # Set exam_size to match generated count so backfill loop exits after wave 0
        orch.config.examiner.exam_size = 3
        orch.config.examiner.max_backfill_rounds = 0

        mock_exam_agent = MagicMock()
        mock_corpus = MagicMock()
        mock_corpus.doc_texts = ["Some content."]
        mock_corpus.n_clusters = 1
        mock_corpus.cluster_sizes = np.array([1])
        mock_exam_agent.prepare_corpus.return_value = mock_corpus
        mock_exam_agent.generate_wave = AsyncMock(return_value=generated_exam)

        with (
            patch("agentic_autorag.orchestrator.ExamAgent", return_value=mock_exam_agent),
            patch(
                "agentic_autorag.orchestrator.run_validation_pipeline",
                new_callable=AsyncMock,
                return_value=generated_exam,
            ),
        ):
            exam, from_cache = await orch._generate_exam(["Some content."])

        assert from_cache is False
        assert len(exam) == 3
        assert exam[0].id == generated_exam[0].id
        mock_exam_agent.prepare_corpus.assert_called_once()
        mock_exam_agent.generate_wave.assert_called_once()

        candidates_path = orch.output_dir / "candidates.json"
        exam_path = orch.output_dir / "exam.json"
        assert candidates_path.exists()
        assert exam_path.exists()

        saved_candidates = json.loads(candidates_path.read_text())
        saved_exam = json.loads(exam_path.read_text())
        assert len(saved_candidates) == 3
        assert len(saved_exam) == 3
        assert saved_candidates[0]["id"] == generated_exam[0].id
        assert saved_exam[0]["id"] == generated_exam[0].id
