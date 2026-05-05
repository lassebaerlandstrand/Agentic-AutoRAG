"""Tests for the Orchestrator — corpus loading, loop mechanics, and fallback."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agentic_autorag.config.models import (
    OpenEndedQuestion,
    ProjectConfig,
    TrialConfig,
)
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.orchestrator import Orchestrator


def _graph_build_config_dict(extraction_model: str = "azure/gpt-4.1-nano") -> dict:
    return {
        "extraction_model": extraction_model,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }


def _make_config_dict(corpus_path: str, output_dir: str, max_trials: int = 2) -> dict:
    """Return a minimal raw dict that converts to a valid ProjectConfig."""
    return {
        "meta": {
            "project_name": "test",
            "corpus_path": corpus_path,
            "corpus_description": "Test corpus",
            "output_dir": output_dir,
            "max_trials": max_trials,
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


def _make_exam(n: int = 3) -> list[OpenEndedQuestion]:
    return [
        OpenEndedQuestion(
            id=f"q{i}",
            question=f"Who founded the company that {i}?",
            canonical_answer=f"Person {i}",
            answer_variants=[],
            chunk_A_id=f"doc_{i}_a::chunk_0",
            chunk_B_id=f"doc_{i}_b::chunk_0",
            source_span_A=f"chunk A span for question {i}",
            source_span_B=f"chunk B span for question {i}",
            source_doc_ids=[f"doc_{i}_a", f"doc_{i}_b"],
            bridge_entity=f"bridge_{i}",
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
                selected_answer=f"Person {i}" if correct else "wrong",
                correct_answer=f"Person {i}",
                retrieved_context="some context",
                generated_response=f"Person {i}" if correct else "wrong",
                em=1.0 if correct else 0.0,
                f1=1.0 if correct else 0.0,
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
        assert docs[0] == ("doc1.txt", "Hello world")
        assert docs[1][0] == "doc2.md"
        assert "Content" in docs[1][1]

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
            patch("agentic_autorag.orchestrator.OpenEndedEvaluator") as MockEvaluator,
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

            # Exam agent: generate_exam returns (questions, prepared corpus)
            mock_exam_inst = MagicMock()
            mock_corpus = MagicMock()
            mock_corpus.chunks = []
            mock_corpus.seeds = []
            mock_exam_inst.generate_exam = AsyncMock(return_value=(exam, mock_corpus))
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
            from agentic_autorag.optimizer.diagnosis import (
                Bottleneck,
                Diagnosis,
                ProposalMeta,
                TrialMetrics,
            )

            mock_agent = AsyncMock()
            mock_agent.propose_initial.return_value = trial_config
            next_config = _make_trial_config()
            next_config = next_config.model_copy(update={"top_k": 7})
            trial_metrics = TrialMetrics()
            diagnosis = Diagnosis(
                trial_metrics=trial_metrics,
                bottlenecks=[Bottleneck(stage="retrieval", severity="primary", evidence="…")],
            )
            proposal_meta = ProposalMeta(
                changes=["top_k: 5 → 7"],
                rationale="diagnoser flagged retrieval primary",
            )
            mock_agent.analyze_and_propose.return_value = (
                trial_metrics,
                diagnosis,
                next_config,
                proposal_meta,
            )
            MockAgent.return_value = mock_agent

            orch = Orchestrator(str(tmp_path / "fake_config.yaml"))
            best = await orch.run()

        assert best is not None
        assert best.score == exam_result.score
        assert len(orch.history.records) == 2
        assert (out / "exam.json").exists()


class TestVLLMAutoManagementForGraph:
    """The vllm_manager is auto-created when any hosted_vllm/ model is configured —
    either in the trial search space or as the graph extraction model.
    """

    @staticmethod
    def _make_orch(tmp_path: Path, raw_config: dict) -> Orchestrator:
        """Build an Orchestrator with external deps mocked. Returns the instance."""
        with (
            patch("agentic_autorag.orchestrator.load_config", return_value=ProjectConfig.model_validate(raw_config)),
            patch("agentic_autorag.orchestrator._check_api_keys"),
            patch("agentic_autorag.orchestrator.IndexBuilder"),
            patch("agentic_autorag.orchestrator.OpenEndedEvaluator"),
            patch("agentic_autorag.orchestrator.ReasoningAgent"),
            patch("agentic_autorag.orchestrator.build_parser"),
            patch("agentic_autorag.orchestrator.KnowledgeBase", side_effect=Exception("no KB in test")),
            patch("agentic_autorag.orchestrator.VLLMServerManager") as MockVLLMCls,
            patch("agentic_autorag.orchestrator.LightRAGStore"),
        ):
            MockVLLMCls.return_value = MagicMock()
            orch = Orchestrator(str(tmp_path / "fake.yaml"))
            orch._mock_vllm_cls = MockVLLMCls  # type: ignore[attr-defined]
        return orch

    def test_vllm_manager_created_for_graph_extraction_model(self, tmp_path: Path) -> None:
        raw = _make_config_dict(str(tmp_path), str(tmp_path / "out"))
        raw["graph"] = _graph_build_config_dict(extraction_model="hosted_vllm/Qwen/Qwen3-30B-A3B")
        raw["search_space"]["index_types"] = ["vector_only", "graph_only"]

        orch = self._make_orch(tmp_path, raw)
        assert orch.vllm_manager is not None

    def test_no_vllm_manager_when_all_models_are_cloud(self, tmp_path: Path) -> None:
        raw = _make_config_dict(str(tmp_path), str(tmp_path / "out"))
        raw["graph"] = _graph_build_config_dict(extraction_model="azure/gpt-4.1-nano")
        raw["search_space"]["index_types"] = ["vector_only", "graph_only"]

        orch = self._make_orch(tmp_path, raw)
        assert orch.vllm_manager is None

    def test_vllm_manager_created_for_search_space_only(self, tmp_path: Path) -> None:
        """Existing behaviour: hosted_vllm/ in search_space triggers manager."""
        raw = _make_config_dict(str(tmp_path), str(tmp_path / "out"))
        raw["search_space"]["llm_models"] = ["hosted_vllm/Qwen/Qwen3-14B"]

        orch = self._make_orch(tmp_path, raw)
        assert orch.vllm_manager is not None


class TestGraphBuildEnsuresVLLMModel:
    """run() must start vLLM for the graph extraction model iff the graph needs building."""

    @staticmethod
    async def _run_graph_step(tmp_path: Path, raw: dict, graph_is_built: bool) -> MagicMock:
        """Execute just the graph-build section of run() with everything else mocked.

        Returns the VLLMServerManager mock instance so tests can inspect ensure_model calls.
        """
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc.txt").write_text("Test document.")

        exam = _make_exam(3)
        exam_result = _make_exam_result(3, 2)
        trial_config = _make_trial_config()

        with (
            patch("agentic_autorag.orchestrator.load_config") as mock_load,
            patch("agentic_autorag.orchestrator._check_api_keys"),
            patch("agentic_autorag.orchestrator.ExamAgent") as MockExamAgent,
            patch("agentic_autorag.orchestrator.run_validation_pipeline", new_callable=AsyncMock) as mock_validate,
            patch("agentic_autorag.orchestrator.IndexBuilder") as MockIndexBuilder,
            patch("agentic_autorag.orchestrator.OpenEndedEvaluator") as MockEvaluator,
            patch("agentic_autorag.orchestrator.ReasoningAgent") as MockAgent,
            patch("agentic_autorag.orchestrator.build_parser") as mock_build_parser,
            patch("agentic_autorag.orchestrator.KnowledgeBase", side_effect=Exception("no KB")),
            patch("agentic_autorag.orchestrator.VLLMServerManager") as MockVLLMCls,
            patch("agentic_autorag.orchestrator.LightRAGStore") as MockLightRAGCls,
        ):
            mock_load.return_value = ProjectConfig.model_validate(raw)

            parser_mock = MagicMock()
            parser_mock.supported_extensions.return_value = {".pdf"}
            mock_build_parser.return_value = parser_mock

            embedder_mock = MagicMock()
            embedder_mock.encode.return_value = np.random.rand(10, 384).astype(np.float32)

            mock_exam_inst = MagicMock()
            mock_corpus = MagicMock()
            mock_corpus.doc_texts = ["doc text"]
            mock_corpus.n_clusters = 1
            mock_corpus.cluster_sizes = np.array([1])
            mock_exam_inst.generate_exam = AsyncMock(return_value=(exam, mock_corpus))
            MockExamAgent.return_value = mock_exam_inst
            mock_validate.return_value = exam

            mock_index = MagicMock()
            mock_index.vector_store = MagicMock()
            mock_index.graph_store = None
            mock_builder = AsyncMock()
            mock_builder.build.return_value = mock_index
            mock_builder.get_embedder = MagicMock(return_value=embedder_mock)
            mock_builder.get_cross_encoder = MagicMock(return_value=MagicMock())
            MockIndexBuilder.return_value = mock_builder

            mock_eval = AsyncMock()
            mock_eval.evaluate.return_value = exam_result
            MockEvaluator.return_value = mock_eval

            from agentic_autorag.optimizer.diagnosis import (
                Bottleneck,
                Diagnosis,
                ProposalMeta,
                TrialMetrics,
            )

            mock_agent = AsyncMock()
            mock_agent.propose_initial.return_value = trial_config
            trial_metrics = TrialMetrics()
            diagnosis = Diagnosis(
                trial_metrics=trial_metrics,
                bottlenecks=[Bottleneck(stage="retrieval", severity="primary", evidence="…")],
            )
            proposal_meta = ProposalMeta(changes=["top_k: 5 → 7"], rationale="…")
            mock_agent.analyze_and_propose.return_value = (
                trial_metrics,
                diagnosis,
                trial_config,
                proposal_meta,
            )
            MockAgent.return_value = mock_agent

            vllm_mock = MagicMock()
            vllm_mock.ensure_model = AsyncMock()
            vllm_mock.shutdown = AsyncMock()
            MockVLLMCls.return_value = vllm_mock

            graph_mock = MagicMock()
            graph_mock.initialize = AsyncMock()
            graph_mock.build = AsyncMock()
            graph_mock.close = AsyncMock()
            graph_mock.is_built = MagicMock(return_value=graph_is_built)
            MockLightRAGCls.return_value = graph_mock

            orch = Orchestrator(str(tmp_path / "fake.yaml"))
            await orch.run()

        return vllm_mock

    @pytest.mark.asyncio
    async def test_ensure_model_called_when_graph_needs_building(self, tmp_path: Path) -> None:
        raw = _make_config_dict(str(tmp_path / "fake_corpus"), str(tmp_path / "out"))
        raw["meta"]["corpus_path"] = str(tmp_path / "corpus")
        raw["graph"] = _graph_build_config_dict(extraction_model="hosted_vllm/Qwen/Qwen3-30B-A3B")
        raw["search_space"]["index_types"] = ["vector_only", "graph_only"]

        vllm_mock = await self._run_graph_step(tmp_path, raw, graph_is_built=False)

        vllm_mock.ensure_model.assert_awaited_once_with("hosted_vllm/Qwen/Qwen3-30B-A3B")

    @pytest.mark.asyncio
    async def test_ensure_model_skipped_when_graph_already_built(self, tmp_path: Path) -> None:
        raw = _make_config_dict(str(tmp_path / "fake_corpus"), str(tmp_path / "out"))
        raw["meta"]["corpus_path"] = str(tmp_path / "corpus")
        raw["graph"] = _graph_build_config_dict(extraction_model="hosted_vllm/Qwen/Qwen3-30B-A3B")
        raw["search_space"]["index_types"] = ["vector_only", "graph_only"]

        vllm_mock = await self._run_graph_step(tmp_path, raw, graph_is_built=True)

        vllm_mock.ensure_model.assert_not_awaited()


class TestSetupLogger:
    """``_setup_logger`` must capture diagnostics from agentic_autorag.* modules
    into run.log. v2 attached the file handler only to the run logger, so
    INFO records from ``agentic_autorag.examiner.exam_agent`` etc. were
    silently dropped. Verify that breaking the wiring would be noticed.
    """

    def test_module_logger_record_lands_in_run_log(self, tmp_path: Path) -> None:
        # Snapshot logger state so we don't bleed config to sibling tests.
        parent = logging.getLogger("agentic_autorag")
        run = logging.getLogger("agentic_autorag.run")
        prev_parent_propagate = parent.propagate
        prev_run_propagate = run.propagate

        run_logger = Orchestrator._setup_logger(tmp_path)
        try:
            module_logger = logging.getLogger("agentic_autorag.examiner.smoke_test")
            module_logger.info("smoke-test-marker-xyz")
            for h in parent.handlers:
                h.flush()
            for h in run_logger.handlers:
                h.flush()
            log_path = tmp_path / "run.log"
            assert log_path.exists()
            assert "smoke-test-marker-xyz" in log_path.read_text(encoding="utf-8")
        finally:
            # Clean up so subsequent tests (and pytest's caplog at root)
            # see the same logger state as before _setup_logger was called.
            for h in list(parent.handlers):
                if isinstance(h, logging.FileHandler):
                    h.close()
                    parent.removeHandler(h)
            for h in list(run_logger.handlers):
                if isinstance(h, logging.FileHandler):
                    h.close()
                    run_logger.removeHandler(h)
            parent.propagate = prev_parent_propagate
            run.propagate = prev_run_propagate


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

    def test_detects_secondary_lever_change(self, capsys) -> None:
        """Secondary levers (reranker_top_n, overlap, graph_*) must show up in the diff."""
        old = _make_trial_config()
        new = _make_trial_config().model_copy(update={"reranker_top_n": old.reranker_top_n + 2})

        Orchestrator._print_config_diff(old, new)
        captured = capsys.readouterr()
        assert "reranker_top_n" in captured.out


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
            exam, from_cache = await orch._generate_exam(["Some content."], doc_ids=["doc.txt"])

        assert from_cache is True
        assert len(exam) == 2
        assert exam[0].id == cached_exam[0].id
        assert exam[1].id == cached_exam[1].id
        MockExamAgent.assert_not_called()

    @pytest.mark.asyncio
    async def test_generates_and_saves_canonical_artifacts_on_miss(self, tmp_path: Path) -> None:
        """On miss, questions are generated and written to candidates.json/exam.json."""
        from agentic_autorag.examiner.chunk_pair_index import ChunkRecord, Seed
        from agentic_autorag.examiner.exam_agent import CompositionResult

        orch = self._make_orch(tmp_path)
        generated_exam = _make_exam(3)

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros((5, 384), dtype="float32")
        orch.index_builder.get_embedder.return_value = mock_embedder

        orch.config.examiner.exam_size = 3

        # PreparedCorpus carrying one LLM refusal so we can verify the
        # rejections section of candidates.json gets populated.
        refusal_seed = Seed(
            chunk_a=ChunkRecord(chunk_id="refA::c0", doc_id="refA", text="t"),
            chunk_b=ChunkRecord(chunk_id="refB::c0", doc_id="refB", text="t"),
        )
        refusal_result = CompositionResult(
            seed=refusal_seed,
            linkable=False,
            rejection_explanation="Only overlap is institutional affiliation.",
        )

        mock_exam_agent = MagicMock()
        mock_corpus = MagicMock()
        mock_corpus.chunks = []
        mock_corpus.seeds = []
        mock_corpus.composition_results = [refusal_result]
        mock_exam_agent.generate_exam = AsyncMock(return_value=(generated_exam, mock_corpus))

        with (
            patch("agentic_autorag.orchestrator.ExamAgent", return_value=mock_exam_agent),
            patch(
                "agentic_autorag.orchestrator.run_validation_pipeline",
                new_callable=AsyncMock,
                return_value=generated_exam,
            ),
        ):
            exam, from_cache = await orch._generate_exam(["Some content."], doc_ids=["doc.txt"])

        assert from_cache is False
        assert len(exam) == 3
        assert exam[0].id == generated_exam[0].id
        mock_exam_agent.generate_exam.assert_awaited_once()

        candidates_path = orch.output_dir / "candidates.json"
        exam_path = orch.output_dir / "exam.json"
        assert candidates_path.exists()
        assert exam_path.exists()

        saved_payload = json.loads(candidates_path.read_text())
        saved_exam = json.loads(exam_path.read_text())
        # v3 shape: {"candidates": [...], "rejections": [...]}.
        assert isinstance(saved_payload, dict)
        assert len(saved_payload["candidates"]) == 3
        assert saved_payload["candidates"][0]["id"] == "C1"
        assert len(saved_payload["rejections"]) == 1
        assert "institutional affiliation" in saved_payload["rejections"][0]["explanation"]
        assert saved_payload["rejections"][0]["chunk_A_id"] == "refA::c0"
        assert len(saved_exam) == 3
        assert saved_exam[0]["id"] == "Q1"

    @pytest.mark.asyncio
    async def test_loads_v2_legacy_candidates_bare_list(self, tmp_path: Path) -> None:
        """A v2 candidates.json (bare list) still loads after the v3 schema change."""
        orch = self._make_orch(tmp_path)
        generated_exam = _make_exam(2)

        # v2-shape file: bare list of OpenEndedQuestion dicts.
        candidates_path = orch.output_dir / "candidates.json"
        candidates_path.write_text(
            json.dumps([q.model_dump(mode="json") for q in generated_exam], indent=2),
            encoding="utf-8",
        )

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros((5, 384), dtype="float32")
        orch.index_builder.get_embedder.return_value = mock_embedder
        orch.config.examiner.exam_size = 2

        with (
            patch("agentic_autorag.orchestrator.ExamAgent") as MockAgent,
            patch(
                "agentic_autorag.orchestrator.run_validation_pipeline",
                new_callable=AsyncMock,
                return_value=generated_exam,
            ),
        ):
            exam, from_cache = await orch._generate_exam(["Some content."], doc_ids=["doc.txt"])
            # ExamAgent must NOT be called — we hit the candidates cache.
            MockAgent.return_value.generate_exam.assert_not_called()

        assert from_cache is False
        assert len(exam) == 2
