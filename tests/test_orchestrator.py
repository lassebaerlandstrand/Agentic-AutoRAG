"""Tests for the Orchestrator — corpus loading, loop mechanics, and fallback."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agentic_autorag.config.models import (
    MCQQuestion,
    RuntimeConfig,
    SearchSpace,
    StructuralConfig,
    TrialConfig,
)
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.orchestrator import Orchestrator


def _make_search_space(corpus_path: str, output_dir: str, max_trials: int = 2) -> dict:
    """Return a minimal raw dict that converts to a valid SearchSpace."""
    return {
        "meta": {
            "project_name": "test",
            "corpus_path": corpus_path,
            "corpus_description": "Test corpus",
            "output_dir": output_dir,
            "max_trials": max_trials,
            "index_registry": False,
        },
        "structural": {
            "parsers": ["pymupdf4llm"],
            "chunking": {
                "strategies": ["recursive", "fixed"],
                "chunk_size": {"min": 128, "max": 1024},
                "chunk_overlap": {"min": 0, "max": 128},
            },
            "embedding_models": ["sentence-transformers/all-MiniLM-L6-v2"],
            "index_types": ["vector_only"],
        },
        "runtime": {
            "retrieval": {
                "top_k": {"min": 3, "max": 10},
                "hybrid_alpha": {"min": 0.0, "max": 1.0},
                "reranker": {"models": ["none"], "top_n": {"min": 3, "max": 5}},
                "query_expansion": ["none"],
            },
            "generation": {
                "llm_models": ["ollama/llama3.2"],
                "temperature": {"min": 0.0, "max": 0.7},
            },
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
        structural=StructuralConfig(
            parser="pymupdf4llm",
            chunking_strategy="recursive",
            chunk_size=512,
            chunk_overlap=64,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ),
        runtime=RuntimeConfig(
            top_k=5,
            reranker="none",
            llm_model="ollama/llama3.2",
            temperature=0.0,
        ),
    )


def _make_exam(n: int = 3) -> list[MCQQuestion]:
    return [
        MCQQuestion(
            id=f"q{i}",
            question=f"Question {i}?",
            options={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_answer="A",
            source_chunk_id=f"chunk_{i}",
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
    def test_loads_txt_and_md(self, tmp_path: Path) -> None:
        """Text and markdown files are read directly."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc1.txt").write_text("Hello world")
        (corpus / "doc2.md").write_text("# Heading\nContent")
        (corpus / "metadata.json").write_text("{}")
        (corpus / ".hidden").write_text("secret")

        raw = _make_search_space(str(corpus), str(tmp_path / "out"))
        with patch("agentic_autorag.orchestrator.load_config") as mock_load:
            mock_load.return_value = SearchSpace.model_validate(raw)
            orch = Orchestrator.__new__(Orchestrator)
            orch.search_space = mock_load.return_value
            orch.parser = MagicMock()
            orch.parser.supported_extensions.return_value = {".pdf"}

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

        raw = _make_search_space(str(corpus), str(tmp_path / "out"))
        with patch("agentic_autorag.orchestrator.load_config") as mock_load:
            mock_load.return_value = SearchSpace.model_validate(raw)
            orch = Orchestrator.__new__(Orchestrator)
            orch.search_space = mock_load.return_value
            orch.parser = MagicMock()
            orch.parser.supported_extensions.return_value = set()

        docs = orch._load_and_parse_corpus()
        assert len(docs) == 1

    def test_empty_corpus_raises(self, tmp_path: Path) -> None:
        corpus = tmp_path / "empty_corpus"
        corpus.mkdir()

        raw = _make_search_space(str(corpus), str(tmp_path / "out"))
        with patch("agentic_autorag.orchestrator.load_config") as mock_load:
            mock_load.return_value = SearchSpace.model_validate(raw)
            orch = Orchestrator.__new__(Orchestrator)
            orch.search_space = mock_load.return_value
            orch.parser = MagicMock()
            orch.parser.supported_extensions.return_value = set()

        docs = orch._load_and_parse_corpus()
        assert docs == []


class TestRunLoop:
    @pytest.mark.asyncio
    async def test_basic_loop(self, tmp_path: Path) -> None:
        """Mock all expensive components and verify the loop runs to completion."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc.txt").write_text("Test document content for chunking.")

        out = tmp_path / "out"
        raw = _make_search_space(str(corpus), str(out), max_trials=2)

        trial_config = _make_trial_config()
        exam = _make_exam(3)
        exam_result = _make_exam_result(3, 2)

        with (
            patch("agentic_autorag.orchestrator.load_config") as mock_load,
            patch("agentic_autorag.orchestrator.ExamAgent") as MockExamAgent,
            patch("agentic_autorag.orchestrator.IndexBuilder") as MockIndexBuilder,
            patch("agentic_autorag.orchestrator.SentenceTransformer") as MockST,
            patch("agentic_autorag.orchestrator.MCQEvaluator") as MockEvaluator,
            patch("agentic_autorag.orchestrator.ReasoningAgent") as MockAgent,
            patch("agentic_autorag.orchestrator.build_parser") as mock_build_parser,
        ):
            mock_load.return_value = SearchSpace.model_validate(raw)

            # Parser
            parser_mock = MagicMock()
            parser_mock.supported_extensions.return_value = {".pdf"}
            mock_build_parser.return_value = parser_mock

            # Exam agent
            mock_exam_inst = AsyncMock()
            mock_exam_inst.generate_exam.return_value = exam
            MockExamAgent.return_value = mock_exam_inst

            # Embedder
            embedder_mock = MagicMock()
            embedder_mock.encode.return_value = np.random.rand(10, 384).astype(np.float32)
            MockST.return_value = embedder_mock

            # Index builder
            mock_index = MagicMock()
            mock_index.vector_store = MagicMock()
            mock_index.graph_store = None
            mock_builder = AsyncMock()
            mock_builder.build.return_value = mock_index
            MockIndexBuilder.return_value = mock_builder

            # Evaluator
            mock_eval = AsyncMock()
            mock_eval.evaluate.return_value = exam_result
            MockEvaluator.return_value = mock_eval

            # Agent
            mock_agent = AsyncMock()
            mock_agent.propose_initial.return_value = trial_config
            next_config = _make_trial_config()
            next_config.runtime.top_k = 7
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
        raw = _make_search_space(str(tmp_path), str(tmp_path / "out"))
        ss = SearchSpace.model_validate(raw)

        orch = Orchestrator.__new__(Orchestrator)
        orch.search_space = ss

        config = _make_trial_config()
        tweaked = orch._random_tweak(config)

        assert isinstance(tweaked, TrialConfig)
        # At least one runtime param should differ (with very high probability)
        # but the structural config should be identical
        assert tweaked.structural == config.structural


class TestPrintConfigDiff:
    def test_detects_changes(self, capsys) -> None:
        old = _make_trial_config()
        new = _make_trial_config()
        new.runtime.top_k = 10

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
