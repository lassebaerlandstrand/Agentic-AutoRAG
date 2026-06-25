"""Tests for the Orchestrator — corpus loading, loop mechanics, and fallback."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agentic_autorag.config.models import (
    OpenEndedQuestion,
    ProjectConfig,
    TrialConfig,
)
from agentic_autorag.cost_ledger import CostLedger
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.orchestrator import Orchestrator


@pytest.fixture(autouse=True)
def _patch_dl_doc_to_chunk_text_for_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route ``dl_doc_to_chunk_text`` through ``export_to_markdown`` for stubs.

    Tests here use ``MagicMock`` DoclingDocuments via ``_stub_dl_doc``, but
    the real helper invokes ``HybridChunker.chunk(dl_doc=...)`` which cannot
    operate on a mock. The stub exposes its text via ``export_to_markdown``,
    so route the helper there to keep test mechanics identical regardless
    of which coordinate frame production uses.
    """
    monkeypatch.setattr(
        "agentic_autorag.orchestrator.dl_doc_to_chunk_text",
        lambda dl_doc, *, max_chunk_words: dl_doc.export_to_markdown(),
    )


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
            "ocr": False,
            "table_structure": True,
        },
        "search_space": {
            "chunking": {
                "strategies": ["recursive", "fixed"],
                "chunk_token_size": {"min": 128, "max": 1024},
                "chunk_token_overlap": {"min": 0, "max": 128},
            },
            "embedding": {"models": ["sentence-transformers/all-MiniLM-L6-v2"]},
            "retrieval": {
                "index_types": ["vector_only"],
                "top_k": {"min": 3, "max": 10},
                "hybrid_alpha": {"min": 0.0, "max": 1.0},
            },
            "reranker": {"models": ["none"], "top_n": {"min": 3, "max": 5}},
            "query_expansion": {"strategies": ["none"], "models": []},
            "generator": {"models": ["ollama/llama3.2"]},
            "temperature": {"min": 0.0, "max": 0.7},
        },
        "examiner": {
            "exam_size": 5,
        },
        "agent": {
            "optimizer_model": "test/model",
            "examiner_model": "test/model",
            "judge_model": "test/model",
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
        generator_llm="ollama/llama3.2",
        temperature=0.0,
    )


def _make_exam(n: int = 3) -> list[OpenEndedQuestion]:
    return [
        OpenEndedQuestion(
            id=f"q{i}",
            question=f"Who founded the company that {i}?",
            canonical_answer=f"Person {i}",
            answer_variants=[],
            reasoning_type="bridge",
            source_chunk_ids=[f"doc_{i}_a::chunk_0", f"doc_{i}_b::chunk_0"],
            source_doc_ids=[f"doc_{i}_a", f"doc_{i}_b"],
            source_spans=[f"chunk A span for question {i}", f"chunk B span for question {i}"],
        )
        for i in range(n)
    ]


def _make_exam_result(n: int = 3, n_correct: int = 2) -> ExamResult:
    return _make_exam_result_for_ids([f"q{i}" for i in range(n)], n_correct=n_correct)


def _make_exam_result_for_ids(question_ids: list[str], n_correct: int = 2) -> ExamResult:
    """Build an ExamResult whose question_ids match the supplied list.

    Used as a ``side_effect`` for mocked evaluators so the result IDs
    track whatever exam IDs the orchestrator constructs at runtime
    (which it mutates to ``C{i}`` form before probe evaluation).
    """
    results = []
    for i, qid in enumerate(question_ids):
        correct = i < n_correct
        results.append(
            QuestionResult(
                question_id=qid,
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
        answer_accuracy=n_correct / max(1, len(question_ids)),
        n_correct=n_correct,
        n_total=len(question_ids),
        n_valid=len(question_ids),
        question_results=results,
    )


def _varied_probe_eval_side_effect(trial_n_correct: int = 2):
    """Side effect that varies n_correct per probe call so the count-based
    selector sees mixed (non-saturated) outcome buckets.

    Order-agnostic exam selection drops k=N (all probes solved) items; if
    every probe returns the same n_correct, every question becomes either
    all-correct or all-wrong and the exam under-fills. This rotation
    guarantees at least one mixed bucket exists.

    Probe calls (identified by IDs not starting with ``Q``) rotate
    ``n_correct = 0, 1, 2, 3`` so every question is wrong in at least
    one probe and right in at least one — no question ends up saturated
    in either direction. Trial calls (the orchestrator renames exam IDs
    to ``Q{i}`` after probe selection) use ``trial_n_correct`` for
    stable scoring.
    """
    probe_calls = [0]

    def side_effect(pipeline, ex):
        ids = [q.id for q in ex]
        is_trial = bool(ids) and ids[0].startswith("Q")
        if is_trial:
            return _make_exam_result_for_ids(ids, n_correct=trial_n_correct)
        probe_calls[0] += 1
        return _make_exam_result_for_ids(ids, n_correct=probe_calls[0] - 1)

    return side_effect


def _stub_dl_doc(text: str) -> MagicMock:
    """Return a mock that quacks like a DoclingDocument for parse tests."""
    mock = MagicMock()
    mock.export_to_markdown.return_value = text
    mock.export_to_dict.return_value = {"_stub_markdown": text}
    return mock


class TestLoadAndParseCorpus:
    @staticmethod
    def _make_orch(
        tmp_path: Path,
        corpus: Path,
        parser_extensions: set[str] = frozenset({".pdf", ".txt", ".md"}),
    ) -> Orchestrator:
        """Build a minimal Orchestrator. The parser mock returns stub DoclingDocs."""
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
        orch.parser.parse.side_effect = lambda fp: _stub_dl_doc(fp.read_text(encoding="utf-8"))
        # Patch DoclingDocument.model_validate for cache hits so the round-trip
        # returns a stub with the cached text in export_to_markdown().
        return orch

    def test_loads_txt_and_md(self, tmp_path: Path) -> None:
        """All formats route through Docling — no direct-read path."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc1.txt").write_text("Hello world")
        (corpus / "doc2.md").write_text("# Heading\nContent")
        (corpus / "metadata.json").write_text("{}")
        (corpus / ".hidden").write_text("secret")

        orch = self._make_orch(tmp_path, corpus)
        docs = orch._load_and_parse_corpus()
        assert len(docs) == 2
        names = sorted(name for name, _ in docs)
        assert names == ["doc1.txt", "doc2.md"]

    def test_skips_metadata_and_hidden(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "metadata.json").write_text("{}")
        (corpus / ".hidden").write_text("x")
        (corpus / "real.txt").write_text("real content")

        orch = self._make_orch(tmp_path, corpus)
        docs = orch._load_and_parse_corpus()
        assert len(docs) == 1
        assert docs[0][0] == "real.txt"

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
        with patch(
            "agentic_autorag.orchestrator.DoclingDocument.model_validate",
            side_effect=lambda d: _stub_dl_doc(d["_stub_markdown"]),
        ):
            docs1 = orch._load_and_parse_corpus()
            assert len(docs1) == 1
            orch.parser.parse.reset_mock()
            docs2 = orch._load_and_parse_corpus()
            assert len(docs2) == 1
            assert docs2[0][0] == docs1[0][0]
            # A cache hit and a re-parse return identical docs, so the only
            # observable proof that the second call hit the cache is that the
            # parser was not invoked again.
            orch.parser.parse.assert_not_called()

    def test_corpus_cache_invalidates_on_file_change(self, tmp_path: Path) -> None:
        """Cache invalidates when a file is added."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc.txt").write_text("original")

        orch = self._make_orch(tmp_path, corpus)
        with patch(
            "agentic_autorag.orchestrator.DoclingDocument.model_validate",
            side_effect=lambda d: _stub_dl_doc(d["_stub_markdown"]),
        ):
            docs1 = orch._load_and_parse_corpus()
            assert len(docs1) == 1
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
            parser_mock.supported_extensions.return_value = {".pdf", ".txt", ".md"}
            parser_mock.parse.side_effect = lambda fp: _stub_dl_doc(fp.read_text(encoding="utf-8"))
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
            # No fingerprint → embedding-build cost crediting is skipped (the
            # MagicMock default would otherwise fail the ``> 0`` token check).
            mock_index.emb_fp = None
            mock_builder = AsyncMock()
            mock_builder.build.return_value = mock_index
            mock_builder.get_embedder = MagicMock(return_value=embedder_mock)
            mock_builder.get_cross_encoder = MagicMock(return_value=MagicMock())
            MockIndexBuilder.return_value = mock_builder

            # Evaluator — use side_effect so the returned ExamResult tracks
            # whatever exam IDs the orchestrator constructs at runtime (it
            # mutates candidate ids to "C{i}" form before probe evaluation).
            # Vary probe outcomes so the count-based selector sees mixed
            # (non-saturated) buckets; trial calls use n_correct=2 to match
            # ``exam_result.answer_accuracy`` below.
            mock_eval = AsyncMock()
            mock_eval.evaluate.side_effect = _varied_probe_eval_side_effect(trial_n_correct=2)
            MockEvaluator.return_value = mock_eval

            # Agent
            from agentic_autorag.optimizer.diagnosis import (
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
            )
            proposal_meta = ProposalMeta(
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
        assert best.answer_accuracy == exam_result.answer_accuracy
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
        raw["search_space"]["retrieval"]["index_types"] = ["vector_only", "graph_only"]

        orch = self._make_orch(tmp_path, raw)
        assert orch.vllm_manager is not None

    def test_no_vllm_manager_when_all_models_are_cloud(self, tmp_path: Path) -> None:
        raw = _make_config_dict(str(tmp_path), str(tmp_path / "out"))
        raw["graph"] = _graph_build_config_dict(extraction_model="azure/gpt-4.1-nano")
        raw["search_space"]["retrieval"]["index_types"] = ["vector_only", "graph_only"]

        orch = self._make_orch(tmp_path, raw)
        assert orch.vllm_manager is None

    def test_vllm_manager_created_for_search_space_only(self, tmp_path: Path) -> None:
        """Existing behaviour: hosted_vllm/ in search_space triggers manager."""
        raw = _make_config_dict(str(tmp_path), str(tmp_path / "out"))
        raw["search_space"]["generator"]["models"] = ["hosted_vllm/Qwen/Qwen3-14B"]

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
            parser_mock.supported_extensions.return_value = {".pdf", ".txt", ".md"}
            parser_mock.parse.side_effect = lambda fp: _stub_dl_doc(fp.read_text(encoding="utf-8"))
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
            mock_eval.evaluate.side_effect = _varied_probe_eval_side_effect(trial_n_correct=2)
            MockEvaluator.return_value = mock_eval

            from agentic_autorag.optimizer.diagnosis import (
                Diagnosis,
                ProposalMeta,
                TrialMetrics,
            )

            mock_agent = AsyncMock()
            mock_agent.propose_initial.return_value = trial_config
            trial_metrics = TrialMetrics()
            diagnosis = Diagnosis(
                trial_metrics=trial_metrics,
            )
            proposal_meta = ProposalMeta(rationale="…")
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
        raw["search_space"]["retrieval"]["index_types"] = ["vector_only", "graph_only"]

        vllm_mock = await self._run_graph_step(tmp_path, raw, graph_is_built=False)

        # Starting vLLM for the extraction model has no observable effect on
        # run()'s return value; the awaited call (with the exact model id) is
        # the behavior under test.
        vllm_mock.ensure_model.assert_awaited_once_with("hosted_vllm/Qwen/Qwen3-30B-A3B")

    @pytest.mark.asyncio
    async def test_ensure_model_skipped_when_graph_already_built(self, tmp_path: Path) -> None:
        raw = _make_config_dict(str(tmp_path / "fake_corpus"), str(tmp_path / "out"))
        raw["meta"]["corpus_path"] = str(tmp_path / "corpus")
        raw["graph"] = _graph_build_config_dict(extraction_model="hosted_vllm/Qwen/Qwen3-30B-A3B")
        raw["search_space"]["retrieval"]["index_types"] = ["vector_only", "graph_only"]

        vllm_mock = await self._run_graph_step(tmp_path, raw, graph_is_built=True)

        # Already-built graph must not spin up vLLM; the skipped call has no
        # observable effect on run()'s output, so not-awaited is the behavior.
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
        # First-use cache-credit bookkeeping (normally set by __init__,
        # which these tests bypass via __new__).
        orch._seen_emb_fps = set()
        orch._pending_cache_events = []
        orch._current_phase = "setup"
        return orch

    @pytest.mark.asyncio
    async def test_loads_exam_from_existing_exam_file(self, tmp_path: Path) -> None:
        """When exam.json exists and clears the minimum-fraction guard, generation is skipped."""
        orch = self._make_orch(tmp_path)
        # exam_size=5 ⇒ MIN_EXAM_FRACTION=0.5 ⇒ threshold=2 ⇒ cached size >=3
        # is the smallest that clears the guard.
        cached_exam = _make_exam(3)
        exam_path = orch.output_dir / "exam.json"
        exam_path.write_text(
            json.dumps([q.model_dump(mode="json") for q in cached_exam], indent=2),
            encoding="utf-8",
        )

        # ExamAgent is patched so a regression that ignored the cache would call
        # a mock rather than the real agent; the cache hit is proven by output.
        with patch("agentic_autorag.orchestrator.ExamAgent"):
            exam, from_cache = await orch._generate_exam([_stub_dl_doc("Some content.")], doc_ids=["doc.txt"])

        assert from_cache is True
        assert len(exam) == 3
        assert exam[0].id == cached_exam[0].id

    @pytest.mark.asyncio
    async def test_generates_and_saves_canonical_artifacts_on_miss(self, tmp_path: Path) -> None:
        """On miss, questions are generated and written to candidates.json/exam.json."""
        from agentic_autorag.examiner.chunk_pair_index import ChunkRecord, Neighborhood
        from agentic_autorag.examiner.exam_agent import CompositionResult

        orch = self._make_orch(tmp_path)
        generated_exam = _make_exam(3)

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros((5, 384), dtype="float32")
        orch.index_builder.get_embedder.return_value = mock_embedder

        orch.config.examiner.exam_size = 3

        # PreparedCorpus carrying two LLM refusals across distinct neighborhoods.
        nh_paired = Neighborhood(
            chunks=[
                ChunkRecord(chunk_id="refA::c0", doc_id="refA", text="t"),
                ChunkRecord(chunk_id="refB::c0", doc_id="refB", text="t"),
            ]
        )
        refusal_result = CompositionResult(
            neighborhood=nh_paired,
            linkable=False,
            rejection_explanation="Only overlap is institutional affiliation.",
        )
        nh_solo = Neighborhood(chunks=[ChunkRecord(chunk_id="solo::c0", doc_id="solo", text="t")])
        single_chunk_refusal_result = CompositionResult(
            neighborhood=nh_solo,
            linkable=False,
            rejection_explanation="Chunk lacks numeric content.",
        )

        mock_exam_agent = MagicMock()
        mock_corpus = MagicMock()
        mock_corpus.chunks = []
        mock_corpus.neighborhoods = []
        mock_corpus.composition_results = [refusal_result, single_chunk_refusal_result]
        mock_exam_agent.generate_exam = AsyncMock(return_value=(generated_exam, mock_corpus))

        with (
            patch("agentic_autorag.orchestrator.ExamAgent", return_value=mock_exam_agent),
            patch(
                "agentic_autorag.orchestrator.run_validation_pipeline",
                new_callable=AsyncMock,
                return_value=generated_exam,
            ),
        ):
            exam, from_cache = await orch._generate_exam([_stub_dl_doc("Some content.")], doc_ids=["doc.txt"])

        assert from_cache is False
        assert len(exam) == 3
        assert exam[0].id == generated_exam[0].id

        candidates_path = orch.output_dir / "details" / "candidates.json"
        exam_path = orch.output_dir / "exam.json"
        assert candidates_path.exists()
        assert exam_path.exists()

        saved_payload = json.loads(candidates_path.read_text())
        saved_exam = json.loads(exam_path.read_text())
        # v3 shape: {"candidates": [...], "rejections": [...]}.
        assert isinstance(saved_payload, dict)
        assert len(saved_payload["candidates"]) == 3
        assert saved_payload["candidates"][0]["id"] == "C1"
        assert len(saved_payload["rejections"]) == 2
        assert "institutional affiliation" in saved_payload["rejections"][0]["explanation"]
        assert saved_payload["rejections"][0]["anchor_chunk_id"] == "refA::c0"
        assert saved_payload["rejections"][0]["neighborhood_chunk_ids"] == ["refA::c0", "refB::c0"]
        assert saved_payload["rejections"][1]["anchor_chunk_id"] == "solo::c0"
        assert saved_payload["rejections"][1]["neighborhood_chunk_ids"] == ["solo::c0"]
        assert len(saved_exam) == 3
        assert saved_exam[0]["id"] == "Q1"

    @pytest.mark.asyncio
    async def test_raises_when_cached_exam_below_min_fraction(self, tmp_path: Path) -> None:
        """A stale on-disk exam.json with too few questions must fail-fast,
        not silently re-feed a degenerate exam to the optimizer."""
        from agentic_autorag.examiner._errors import ExamGenerationFailed

        orch = self._make_orch(tmp_path)
        orch.config.examiner.exam_size = 10

        # Cached exam carries only 2 questions; threshold = 5.
        cached_exam = _make_exam(2)
        exam_path = orch.output_dir / "exam.json"
        exam_path.write_text(
            json.dumps([q.model_dump(mode="json") for q in cached_exam], indent=2),
            encoding="utf-8",
        )

        with pytest.raises(ExamGenerationFailed) as exc_info:
            await orch._generate_exam([_stub_dl_doc("Some content.")], doc_ids=["doc.txt"])

        err = exc_info.value
        assert err.n_actual == 2
        assert err.n_target == 10
        assert err.stage_counts.get("loaded_from_cache") == 2

    @pytest.mark.asyncio
    async def test_raises_when_composition_yields_zero_candidates(self, tmp_path: Path) -> None:
        """Zero surviving candidates ⇒ ExamGenerationFailed, not a silent empty exam."""
        from agentic_autorag.examiner._errors import ExamGenerationFailed

        orch = self._make_orch(tmp_path)

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros((5, 384), dtype="float32")
        orch.index_builder.get_embedder.return_value = mock_embedder
        orch.config.examiner.exam_size = 5

        mock_corpus = MagicMock()
        mock_corpus.chunks = []
        mock_corpus.seeds = []
        mock_corpus.composition_results = []
        mock_exam_agent = MagicMock()
        mock_exam_agent.generate_exam = AsyncMock(return_value=([], mock_corpus))
        mock_exam_agent.last_composition_rejections = Counter({"llm_refused": 7, "empty_span_b_with_multi_hop_type": 2})

        with (
            patch("agentic_autorag.orchestrator.ExamAgent", return_value=mock_exam_agent),
            pytest.raises(ExamGenerationFailed) as exc_info,
        ):
            await orch._generate_exam([_stub_dl_doc("Some content.")], doc_ids=["doc.txt"])

        err = exc_info.value
        assert err.n_actual == 0
        assert err.n_target == 5
        assert "candidates.json" in err.candidates_path
        assert err.top_rejection_reasons[0] == ("llm_refused", 7)
        assert err.stage_counts.get("after_composition") == 0
        assert "llm_refused=7" in str(err)

    @pytest.mark.asyncio
    async def test_raises_when_exam_below_min_fraction(self, tmp_path: Path) -> None:
        """Filtering down to <50% of exam_size also triggers ExamGenerationFailed."""
        from agentic_autorag.examiner._errors import ExamGenerationFailed

        orch = self._make_orch(tmp_path)
        # exam_size=5 ⇒ MIN_EXAM_FRACTION=0.5 ⇒ threshold=2 ⇒ len<2 must raise.
        orch.config.examiner.exam_size = 5
        # probe_selection off so the final exam is whatever validation returns.
        orch.config.examiner.probe_selection = False

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros((5, 384), dtype="float32")
        orch.index_builder.get_embedder.return_value = mock_embedder

        # Composition produces 3 candidates; validation prunes to 1.
        all_candidates = _make_exam(3)
        validated = all_candidates[:1]

        mock_corpus = MagicMock()
        mock_corpus.chunks = []
        mock_corpus.seeds = []
        mock_corpus.composition_results = []
        mock_exam_agent = MagicMock()
        mock_exam_agent.generate_exam = AsyncMock(return_value=(all_candidates, mock_corpus))
        mock_exam_agent.last_composition_rejections = Counter()

        with (
            patch("agentic_autorag.orchestrator.ExamAgent", return_value=mock_exam_agent),
            patch(
                "agentic_autorag.orchestrator.run_validation_pipeline",
                new_callable=AsyncMock,
                return_value=validated,
            ),
            pytest.raises(ExamGenerationFailed) as exc_info,
        ):
            await orch._generate_exam([_stub_dl_doc("Some content.")], doc_ids=["doc.txt"])

        err = exc_info.value
        assert err.n_actual == 1
        assert err.n_target == 5
        assert err.stage_counts.get("after_composition") == 3
        assert err.stage_counts.get("after_validation") == 1
        assert err.stage_counts.get("after_selection") == 1

    @pytest.mark.asyncio
    async def test_probe_build_receives_markdown_strings_not_docling_docs(self, tmp_path: Path) -> None:
        """Regression: the probe-discrimination loop passes ``documents`` to
        ``index_builder.build``, which expects markdown strings. Earlier the
        same DoclingDocument list ExamAgent consumes was forwarded verbatim
        and crashed inside _chunk_docs_by_tokens (``doc.strip()`` on a
        Pydantic model). Verify the first positional arg is strings."""
        from agentic_autorag.config.models import TrialConfig

        orch = self._make_orch(tmp_path)
        orch.config.examiner.exam_size = 3
        orch.config.examiner.probe_selection = True

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros((5, 384), dtype="float32")
        orch.index_builder.get_embedder.return_value = mock_embedder
        # Short-circuit the per-probe pipeline build so we exit through the
        # try/except cleanly; we only care that the call args are strings.
        orch.index_builder.build = AsyncMock(side_effect=RuntimeError("stop after build"))

        generated_exam = _make_exam(3)
        mock_corpus = MagicMock()
        mock_corpus.chunks = []
        mock_corpus.seeds = []
        mock_corpus.composition_results = []
        mock_exam_agent = MagicMock()
        mock_exam_agent.generate_exam = AsyncMock(return_value=(generated_exam, mock_corpus))
        mock_exam_agent.last_composition_rejections = Counter()

        probe_trial = TrialConfig(
            chunking_strategy="recursive",
            chunk_token_size=256,
            chunk_token_overlap=0,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            top_k=3,
            reranker="none",
            generator_llm="ollama/llama3.2",
            temperature=0.0,
        )

        with (
            patch("agentic_autorag.orchestrator.ExamAgent", return_value=mock_exam_agent),
            patch(
                "agentic_autorag.orchestrator.run_validation_pipeline",
                new_callable=AsyncMock,
                return_value=generated_exam,
            ),
            patch(
                "agentic_autorag.orchestrator.select_probe_configs",
                return_value=[("probe-test", probe_trial)],
            ),
        ):
            await orch._generate_exam(
                [_stub_dl_doc("Markdown body of the document used at probe time.")],
                doc_ids=["doc.txt"],
            )

        first_arg = orch.index_builder.build.call_args.args[0]
        assert isinstance(first_arg, list) and first_arg, "probe build received empty/non-list documents"
        assert isinstance(first_arg[0], str), (
            f"probe build expected list[str] markdown documents, got list[{type(first_arg[0]).__name__}]"
        )

    @pytest.mark.asyncio
    async def test_loads_v2_legacy_candidates_bare_list(self, tmp_path: Path) -> None:
        """A v2 candidates.json (bare list) still loads after the v3 schema change."""
        orch = self._make_orch(tmp_path)
        generated_exam = _make_exam(2)

        # v2-shape file: bare list of OpenEndedQuestion dicts.
        candidates_path = orch.output_dir / "details" / "candidates.json"
        candidates_path.parent.mkdir(parents=True, exist_ok=True)
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
            exam, from_cache = await orch._generate_exam([_stub_dl_doc("Some content.")], doc_ids=["doc.txt"])
            # A candidates-cache hit also returns from_cache=False (that flag
            # tracks exam.json, not candidates.json), so the only signal that
            # the legacy candidates file was reused — rather than regenerated —
            # is that the exam agent was never asked to generate.
            MockAgent.return_value.generate_exam.assert_not_called()

        assert from_cache is False
        assert len(exam) == 2


class TestConfigDiff:
    """``_diff_pairs`` + ``_log_config_diff`` must report every changed lever.

    Regression: bm25_vector_fusion was missing from the hand-maintained pair
    list, so a swap from "alpha" → "rrf" rendered as "Config: no changes".
    The canonical CONFIG_LEVER_FIELDS tuple is now the single source of truth.
    """

    @staticmethod
    def _base() -> TrialConfig:
        return TrialConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            top_k=5,
            generator_llm="ollama/llama3.2",
            bm25_vector_fusion="alpha",
            long_context_reorder=False,
            passage_compressor="none",
            temperature=0.0,
        )

    def test_bm25_fusion_swap_appears_in_diff(self) -> None:
        old = self._base()
        new = old.model_copy(update={"bm25_vector_fusion": "rrf"})
        pairs = Orchestrator._diff_pairs(old, new)
        changes = [(n, a, b) for n, a, b in pairs if a != b]
        assert changes == [("bm25_vector_fusion", "alpha", "rrf")]

    def test_passage_compressor_swap_appears_in_diff(self) -> None:
        old = self._base()
        new = old.model_copy(update={"passage_compressor": "tree_summarize", "compressor_llm": "ollama/llama3.2"})
        pairs = Orchestrator._diff_pairs(old, new)
        changed_names = sorted(n for n, a, b in pairs if a != b)
        assert "passage_compressor" in changed_names
        assert "compressor_llm" in changed_names

    def test_long_context_reorder_swap_appears_in_diff(self) -> None:
        old = self._base()
        new = old.model_copy(update={"long_context_reorder": True})
        pairs = Orchestrator._diff_pairs(old, new)
        changes = [(n, a, b) for n, a, b in pairs if a != b]
        assert changes == [("long_context_reorder", False, True)]

    def test_log_emits_config_changes_on_real_diff(self, caplog) -> None:
        old = self._base()
        new = old.model_copy(update={"bm25_vector_fusion": "rrf"})
        orch = Orchestrator.__new__(Orchestrator)
        orch.logger = logging.getLogger("agentic_autorag.run")
        with caplog.at_level(logging.INFO, logger="agentic_autorag.run"):
            orch._log_config_diff(old, new)
        messages = [r.getMessage() for r in caplog.records]
        assert any("Config changes" in m and "bm25_vector_fusion: alpha -> rrf" in m for m in messages)
        assert not any("Config: no changes" in m for m in messages)


class TestResolveRecommendation:
    @staticmethod
    def _make_orch(tmp_path: Path, *, skip: bool = False) -> Orchestrator:
        from types import SimpleNamespace

        # Trial 1 is the top scorer (cheapest at that score → the mechanical
        # fallback pick); trial 2 is a cheaper, lower-scoring frontier point.
        cfg = SimpleNamespace(to_prompt_dump=lambda include_graph: {"generator_llm": "x/y", "top_k": 5})
        rec1 = SimpleNamespace(trial_number=1, answer_accuracy=0.9, mean_llm_cost_per_query_usd=0.002, config=cfg)
        rec2 = SimpleNamespace(trial_number=2, answer_accuracy=0.7, mean_llm_cost_per_query_usd=0.001, config=cfg)
        orch = Orchestrator.__new__(Orchestrator)
        orch.logger = logging.getLogger("agentic_autorag.run")
        orch.output_dir = tmp_path
        orch.skip_final_report = skip
        orch._exam = []
        orch.history = SimpleNamespace(records=[rec1, rec2])
        orch.config = SimpleNamespace(
            agent=SimpleNamespace(optimizer_model="test/model"),
            meta=SimpleNamespace(cost_aware=True, corpus_description="A corpus.", project_name="test"),
            uses_graph=lambda: False,
        )
        return orch

    @patch("agentic_autorag.optimizer.final_report.generate_final_report", new_callable=AsyncMock)
    async def test_writes_summary_and_returns_chosen_trial(self, mock_gen: AsyncMock, tmp_path: Path) -> None:
        mock_gen.return_value = (2, "## Recommendation\nIt worked.")
        orch = self._make_orch(tmp_path)

        recommended = await orch._resolve_recommendation(ledger=CostLedger())

        path = orch.output_dir / "optimization_summary.md"
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        # Structure, not exact title prose: the heading line names the project,
        # and the LLM-written body is embedded verbatim.
        title_line = text.splitlines()[0]
        assert title_line.startswith("# ")
        assert "test" in title_line  # project_name
        assert "## Recommendation\nIt worked." in text  # body from generate_final_report
        # The deterministic frontier block is folded into the same file …
        assert "## Pareto frontier" in text
        # … and there is no longer a separate frontier_report.md.
        assert not (orch.output_dir / "frontier_report.md").exists()
        # model is plumbed to the (mocked) report generator; not visible in the file.
        assert mock_gen.await_args.kwargs["model"] == "test/model"
        assert recommended.trial_number == 2

    @patch("agentic_autorag.optimizer.final_report.generate_final_report", new_callable=AsyncMock)
    async def test_skip_flag_writes_nothing(self, mock_gen: AsyncMock, tmp_path: Path) -> None:
        orch = self._make_orch(tmp_path, skip=True)

        recommended = await orch._resolve_recommendation(ledger=CostLedger())

        assert not (orch.output_dir / "optimization_summary.md").exists()
        # Skipping must avoid the (paid) LLM call entirely, not merely skip the
        # file write; the absent file alone wouldn't prove the model went unused.
        mock_gen.assert_not_awaited()
        assert recommended.trial_number == 1  # cheapest top scorer

    @patch("agentic_autorag.optimizer.final_report.generate_final_report", new_callable=AsyncMock)
    async def test_generation_failure_falls_back_to_max_score(self, mock_gen: AsyncMock, tmp_path: Path) -> None:
        mock_gen.side_effect = RuntimeError("model unreachable")
        orch = self._make_orch(tmp_path)

        recommended = await orch._resolve_recommendation(ledger=CostLedger())

        assert not (orch.output_dir / "optimization_summary.md").exists()
        assert recommended.trial_number == 1  # cheapest top scorer


class TestSaveFrontierArtifacts:
    @staticmethod
    def _make_orch(tmp_path: Path) -> Orchestrator:
        from types import SimpleNamespace

        cfg = SimpleNamespace(to_prompt_dump=lambda include_graph: {"generator_llm": "x/y", "top_k": 5})
        rec1 = SimpleNamespace(
            trial_number=1, answer_accuracy=0.9, mean_llm_cost_per_query_usd=0.002, total_llm_cost_usd=0.4, config=cfg
        )
        rec2 = SimpleNamespace(
            trial_number=2, answer_accuracy=0.7, mean_llm_cost_per_query_usd=0.001, total_llm_cost_usd=0.2, config=cfg
        )
        orch = Orchestrator.__new__(Orchestrator)
        orch.logger = logging.getLogger("agentic_autorag.run")
        orch.output_dir = tmp_path
        orch.history = SimpleNamespace(records=[rec1, rec2])
        orch.config = SimpleNamespace(uses_graph=lambda: False)
        return orch

    def test_writes_frontier_dir_and_recommended_without_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — frontier is both records; the recommended pick is trial 2.
        orch = self._make_orch(tmp_path)
        monkeypatch.setattr(
            "agentic_autorag.orchestrator.pareto.compute_frontier",
            lambda records: list(records),
        )

        # Act
        orch._save_frontier_artifacts(recommended_trial=2)

        # Assert — runnable per-member YAMLs at top level, recommended.yaml at
        # top level, and NO machine-readable frontier.json (removed).
        frontier_dir = tmp_path / "frontier"
        assert sorted(p.name for p in frontier_dir.iterdir()) == ["trial_01.yaml", "trial_02.yaml"]
        assert not (tmp_path / "frontier.json").exists()
        recommended = tmp_path / "recommended.yaml"
        assert recommended.exists()
        assert "trial 2" in recommended.read_text(encoding="utf-8")
        assert "recommended" in (frontier_dir / "trial_02.yaml").read_text(encoding="utf-8")


class TestAblationHooks:
    """``use_knowledge_base`` / ``use_diagnosis`` flags thread into the agent."""

    @staticmethod
    def _build(tmp_path: Path, *, use_knowledge_base: bool = True, use_diagnosis: bool = True):
        raw = _make_config_dict(str(tmp_path / "corpus"), str(tmp_path / "out"))
        mock_kb = MagicMock()
        mock_kb._embeddings = {"models": {"sentence-transformers/all-MiniLM-L6-v2": {"max_tokens": 256}}}
        with (
            patch("agentic_autorag.orchestrator.load_config", return_value=ProjectConfig.model_validate(raw)),
            patch("agentic_autorag.orchestrator._check_api_keys"),
            patch("agentic_autorag.orchestrator.IndexBuilder"),
            patch("agentic_autorag.orchestrator.OpenEndedEvaluator"),
            patch("agentic_autorag.orchestrator.ReasoningAgent") as MockRA,
            patch("agentic_autorag.orchestrator.build_parser"),
            patch("agentic_autorag.orchestrator.KnowledgeBase", return_value=mock_kb),
            patch("agentic_autorag.orchestrator.VLLMServerManager"),
            patch("agentic_autorag.orchestrator.LightRAGStore"),
        ):
            orch = Orchestrator(
                str(tmp_path / "fake.yaml"),
                use_knowledge_base=use_knowledge_base,
                use_diagnosis=use_diagnosis,
            )
        return orch, MockRA

    def test_kb_on_passes_kb_to_agent(self, tmp_path: Path) -> None:
        orch, MockRA = self._build(tmp_path, use_knowledge_base=True)
        assert orch.knowledge_base is not None
        assert MockRA.call_args.kwargs["knowledge_base"] is orch.knowledge_base
        assert MockRA.call_args.kwargs["use_diagnosis"] is True

    def test_kb_off_passes_none_but_keeps_token_limits(self, tmp_path: Path) -> None:
        orch, MockRA = self._build(tmp_path, use_knowledge_base=False)
        assert orch.knowledge_base is None
        assert MockRA.call_args.kwargs["knowledge_base"] is None
        # Fairness invariant: embedding token limits (a search-space feasibility
        # input) are still populated from the KB with the reasoning prior off,
        # so every method sees the identical feasible space.
        assert orch.config.embedding_token_limits["sentence-transformers/all-MiniLM-L6-v2"] == 256

    def test_diagnosis_off_threads_flag(self, tmp_path: Path) -> None:
        _, MockRA = self._build(tmp_path, use_diagnosis=False)
        assert MockRA.call_args.kwargs["use_diagnosis"] is False
