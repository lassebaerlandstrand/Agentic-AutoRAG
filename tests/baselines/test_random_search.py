"""End-to-end test for the Random-search driver with a stubbed evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import yaml

from agentic_autorag.baselines.random_search import run_random_search
from agentic_autorag.config.models import ProjectConfig, TrialConfig
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.orchestrator import Orchestrator


def _make_config_dict(corpus_path: str, output_dir: str, max_trials: int = 5) -> dict:
    return {
        "meta": {
            "project_name": "random-test",
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


def _make_exam_result(score: float, n_correct: int = 2, n_total: int = 3) -> ExamResult:
    results = [
        QuestionResult(
            question_id=f"q{i}",
            correct=i < n_correct,
            selected_answer="A" if i < n_correct else "B",
            correct_answer="A",
            retrieved_context="ctx",
            generated_response="A" if i < n_correct else "B",
        )
        for i in range(n_total)
    ]
    return ExamResult(score=score, n_correct=n_correct, n_total=n_total, question_results=results)


@pytest.mark.asyncio
async def test_run_random_search_writes_outputs(tmp_path: Path) -> None:
    """Driver writes best_config.yaml + history.jsonl + optimizer_meta.json."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.txt").write_text("Doc content.")

    cache_dir = tmp_path / "shared"
    out_dir = tmp_path / "random_seed1"
    raw = _make_config_dict(str(corpus), str(cache_dir), max_trials=4)
    cfg = ProjectConfig.model_validate(raw)

    # Scores monotonically increase to make the best-trial assertion explicit.
    scores = [0.40, 0.55, 0.70, 0.65]
    score_iter = iter(scores)

    with (
        patch("agentic_autorag.orchestrator.load_config", return_value=cfg),
        patch("agentic_autorag.orchestrator._check_api_keys"),
        patch("agentic_autorag.orchestrator.IndexBuilder") as MockIndexBuilder,
        patch("agentic_autorag.orchestrator.OpenEndedEvaluator") as MockEvaluator,
        patch("agentic_autorag.orchestrator.ReasoningAgent"),
        patch("agentic_autorag.orchestrator.build_parser") as mock_parser,
        patch("agentic_autorag.orchestrator.KnowledgeBase", side_effect=Exception("no KB")),
        patch.object(
            Orchestrator,
            "setup",
            new_callable=AsyncMock,
        ),
        patch.object(
            Orchestrator,
            "evaluate_trial",
            new_callable=AsyncMock,
        ) as mock_eval_trial,
        patch.object(
            Orchestrator,
            "cleanup",
            new_callable=AsyncMock,
        ),
    ):
        # Parser supports nothing extra; corpus has only .txt which is direct-read.
        parser_mock = MagicMock()
        parser_mock.supported_extensions.return_value = set()
        mock_parser.return_value = parser_mock

        builder_mock = AsyncMock()
        builder_mock.build = AsyncMock()
        MockIndexBuilder.return_value = builder_mock

        eval_mock = AsyncMock()
        MockEvaluator.return_value = eval_mock

        async def _stub_eval_trial(trial_config: TrialConfig) -> ExamResult:
            return _make_exam_result(next(score_iter))

        mock_eval_trial.side_effect = _stub_eval_trial

        best = await run_random_search(
            config_path=str(tmp_path / "fake_config.yaml"),
            output_dir=str(out_dir),
            seed=1,
            max_trials=4,
        )

    # 1. Best record reflects max score
    assert best is not None
    assert best.score == max(scores)

    # 2. history.jsonl in per-run dir, with 4 entries
    history_path = out_dir / "history.jsonl"
    assert history_path.exists()
    lines = [ln for ln in history_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 4

    # 3. best_config.yaml is a valid TrialConfig
    best_path = out_dir / "best_config.yaml"
    assert best_path.exists()
    payload = yaml.safe_load(best_path.read_text())
    rebuilt = TrialConfig.model_validate(payload)
    assert cfg.validate_trial(rebuilt) == []

    # 4. optimizer_meta.json present and reasonable
    meta_path = out_dir / "optimizer_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["algorithm"] == "random"
    assert meta["seed"] == 1
    assert meta["max_trials"] == 4
    assert meta["n_trials_completed"] == 4
    assert meta["wall_clock_s"] >= 0.0


@pytest.mark.asyncio
async def test_run_random_search_reproducible(tmp_path: Path) -> None:
    """Same seed → identical history of TrialConfigs."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.txt").write_text("Doc content.")

    raw = _make_config_dict(str(corpus), str(tmp_path / "shared"), max_trials=5)
    cfg = ProjectConfig.model_validate(raw)

    async def _run(out_dir: Path) -> list[dict]:
        with (
            patch("agentic_autorag.orchestrator.load_config", return_value=cfg),
            patch("agentic_autorag.orchestrator._check_api_keys"),
            patch("agentic_autorag.orchestrator.IndexBuilder") as MockIndexBuilder,
            patch("agentic_autorag.orchestrator.OpenEndedEvaluator") as MockEvaluator,
            patch("agentic_autorag.orchestrator.ReasoningAgent"),
            patch("agentic_autorag.orchestrator.build_parser") as mock_parser,
            patch("agentic_autorag.orchestrator.KnowledgeBase", side_effect=Exception("no KB")),
            patch.object(
                Orchestrator,
                "setup",
                new_callable=AsyncMock,
            ),
            patch.object(
                Orchestrator,
                "evaluate_trial",
                new_callable=AsyncMock,
            ) as mock_eval_trial,
            patch.object(
                Orchestrator,
                "cleanup",
                new_callable=AsyncMock,
            ),
        ):
            parser_mock = MagicMock()
            parser_mock.supported_extensions.return_value = set()
            mock_parser.return_value = parser_mock
            MockIndexBuilder.return_value = AsyncMock()
            MockEvaluator.return_value = AsyncMock()
            score_iter = iter(np.linspace(0.4, 0.9, 10).tolist())

            async def _stub_eval(cfg_in: TrialConfig) -> ExamResult:
                return _make_exam_result(next(score_iter))

            mock_eval_trial.side_effect = _stub_eval

            await run_random_search(
                config_path=str(tmp_path / "fake_config.yaml"),
                output_dir=str(out_dir),
                seed=99,
                max_trials=5,
            )
        return [json.loads(ln) for ln in (out_dir / "history.jsonl").read_text().splitlines() if ln.strip()]

    run_a = await _run(tmp_path / "run_a")
    run_b = await _run(tmp_path / "run_b")

    # Trial configs (the proposer outputs) must be byte-identical across seeds.
    configs_a = [r["config"] for r in run_a]
    configs_b = [r["config"] for r in run_b]
    assert configs_a == configs_b
