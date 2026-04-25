"""End-to-end test for the Bayesian (Optuna TPE) driver with stubbed evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import optuna
import pytest
import yaml

from agentic_autorag.baselines.bayesian import (
    _OPTUNA_DB_NAME,
    _OPTUNA_STUDY_NAME,
    _SAMPLER_PICKLE_NAME,
    run_bayesian_search,
)
from agentic_autorag.config.models import ProjectConfig, TrialConfig
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.orchestrator import Orchestrator


def _make_config_dict(corpus_path: str, output_dir: str, max_trials: int = 5) -> dict:
    return {
        "meta": {
            "project_name": "bayes-test",
            "corpus_path": corpus_path,
            "corpus_description": "Test corpus",
            "output_dir": output_dir,
            "max_trials": max_trials,
        },
        "parsing": {"parser": "pymupdf4llm", "ocr": False, "table_structure": True},
        "search_space": {
            "chunking": {
                "strategies": ["recursive"],
                "chunk_token_size": {"min": 128, "max": 512},
                "chunk_token_overlap": {"min": 0, "max": 64},
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
        "examiner": {"exam_size": 3},
        "agent": {
            "optimizer_model": "test/model",
            "examiner_model": "test/model",
            "max_history_trials": 5,
        },
    }


def _make_exam_result(score: float, n: int = 3) -> ExamResult:
    n_correct = round(score * n)
    results = [
        QuestionResult(
            question_id=f"q{i}",
            correct=i < n_correct,
            selected_answer="A" if i < n_correct else "B",
            correct_answer="A",
            retrieved_context="ctx",
            generated_response="A" if i < n_correct else "B",
        )
        for i in range(n)
    ]
    return ExamResult(score=score, n_correct=n_correct, n_total=n, question_results=results)


def _patch_orchestrator(score_iter):
    """Yield-style helper: patches Orchestrator's external deps + setup/eval/cleanup."""

    async def _stub_eval(_cfg: TrialConfig) -> ExamResult:
        return _make_exam_result(next(score_iter))

    return [
        patch("agentic_autorag.orchestrator._check_api_keys"),
        patch("agentic_autorag.orchestrator.IndexBuilder", return_value=AsyncMock()),
        patch("agentic_autorag.orchestrator.MCQEvaluator", return_value=AsyncMock()),
        patch("agentic_autorag.orchestrator.ReasoningAgent"),
        patch("agentic_autorag.orchestrator.KnowledgeBase", side_effect=Exception("no KB")),
        patch.object(Orchestrator, "setup", new_callable=AsyncMock),
        patch.object(Orchestrator, "evaluate_trial", new_callable=AsyncMock, side_effect=_stub_eval),
        patch.object(Orchestrator, "cleanup", new_callable=AsyncMock),
    ]


@pytest.mark.asyncio
async def test_run_bayesian_writes_outputs_and_sqlite(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.txt").write_text("Doc.")
    out_dir = tmp_path / "bayes_seed1"
    raw = _make_config_dict(str(corpus), str(tmp_path / "shared"), max_trials=5)
    cfg = ProjectConfig.model_validate(raw)
    scores = [0.40, 0.60, 0.55, 0.75, 0.65]
    score_iter = iter(scores)

    with (
        patch("agentic_autorag.orchestrator.load_config", return_value=cfg),
        patch("agentic_autorag.orchestrator.build_parser") as mock_parser,
    ):
        parser_mock = MagicMock()
        parser_mock.supported_extensions.return_value = set()
        mock_parser.return_value = parser_mock

        from contextlib import ExitStack

        with ExitStack() as stack:
            for p in _patch_orchestrator(score_iter):
                stack.enter_context(p)
            best = await run_bayesian_search(
                config_path=str(tmp_path / "fake.yaml"),
                output_dir=str(out_dir),
                seed=1,
                max_trials=5,
            )

    assert best is not None
    assert best.score == max(scores)

    # SQLite study exists with 5 trials
    db_path = out_dir / _OPTUNA_DB_NAME
    assert db_path.exists()
    study = optuna.load_study(study_name=_OPTUNA_STUDY_NAME, storage=f"sqlite:///{db_path}")
    assert len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) == 5

    # Per-run artefacts present
    assert (out_dir / "history.jsonl").exists()
    assert (out_dir / "best_config.yaml").exists()
    assert (out_dir / "optimizer_meta.json").exists()
    assert (out_dir / _SAMPLER_PICKLE_NAME).exists()

    payload = yaml.safe_load((out_dir / "best_config.yaml").read_text())
    rebuilt = TrialConfig.model_validate(payload)
    assert cfg.validate_trial(rebuilt) == []

    meta = json.loads((out_dir / "optimizer_meta.json").read_text())
    assert meta["algorithm"] == "bayesian"
    assert meta["seed"] == 1
    assert meta["n_trials_completed"] == 5


@pytest.mark.asyncio
async def test_run_bayesian_resume_extends_study(tmp_path: Path) -> None:
    """Re-running with a higher max_trials extends the existing sqlite study."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.txt").write_text("Doc.")
    out_dir = tmp_path / "bayes_resume"
    raw = _make_config_dict(str(corpus), str(tmp_path / "shared"), max_trials=10)
    cfg = ProjectConfig.model_validate(raw)
    score_iter = iter([0.4 + 0.05 * i for i in range(20)])

    from contextlib import ExitStack

    async def _do_run(target_trials: int) -> None:
        with (
            patch("agentic_autorag.orchestrator.load_config", return_value=cfg),
            patch("agentic_autorag.orchestrator.build_parser") as mock_parser,
        ):
            parser_mock = MagicMock()
            parser_mock.supported_extensions.return_value = set()
            mock_parser.return_value = parser_mock

            with ExitStack() as stack:
                for p in _patch_orchestrator(score_iter):
                    stack.enter_context(p)
                await run_bayesian_search(
                    config_path=str(tmp_path / "fake.yaml"),
                    output_dir=str(out_dir),
                    seed=1,
                    max_trials=target_trials,
                )

    await _do_run(3)
    db_path = out_dir / _OPTUNA_DB_NAME
    study1 = optuna.load_study(study_name=_OPTUNA_STUDY_NAME, storage=f"sqlite:///{db_path}")
    n_after_first = len([t for t in study1.trials if t.state == optuna.trial.TrialState.COMPLETE])
    assert n_after_first == 3

    await _do_run(5)
    study2 = optuna.load_study(study_name=_OPTUNA_STUDY_NAME, storage=f"sqlite:///{db_path}")
    n_after_second = len([t for t in study2.trials if t.state == optuna.trial.TrialState.COMPLETE])
    # Resume adds 5 new trials on top of the 3 already in the DB.
    assert n_after_second == 8
