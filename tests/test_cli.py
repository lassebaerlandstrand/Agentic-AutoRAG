"""CLI tests for validate command artifact path behavior."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from agentic_autorag.cli import app


def _candidate_payload() -> list[dict]:
    return [
        {
            "id": "q1",
            "question": "Which value is listed in the source?",
            "options": {"A": "10", "B": "20", "C": "30", "D": "40"},
            "correct_answer": "A",
            "source_doc_ids": ["doc_0"],
            "source_fact": "The value listed in the source is 10.",
            "cluster_id": 0,
        }
    ]


def _fake_orchestrator_factory(output_dir: Path):
    class _FakeOrchestrator:
        def __init__(self, config_path: str) -> None:  # noqa: ARG002
            self.config = SimpleNamespace(
                meta=SimpleNamespace(output_dir=str(output_dir)),
                examiner=SimpleNamespace(
                    source_fact_threshold=0.65,
                    detect_parametric_leaks=True,
                    source_fact_substring_fallback=True,
                    source_fact_min_length=60,
                    source_fact_window_chunk_size=300,
                    source_fact_window_chunk_overlap=150,
                    parametric_leak_trials=3,
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                ),
                agent=SimpleNamespace(examiner_model="test/model", concurrency=1),
            )
            self.index_builder = MagicMock()
            self.index_builder.get_embedder.return_value = MagicMock()

        def _load_and_parse_corpus(self) -> list[str]:
            return ["Some corpus document text for validation."]

    return _FakeOrchestrator


@pytest.mark.asyncio
async def _validation_passthrough(candidates, **kwargs):  # noqa: ANN001, ARG001
    return candidates


def test_validate_defaults_to_canonical_artifact_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidates.json"
    candidates_path.write_text(json.dumps(_candidate_payload()), encoding="utf-8")

    monkeypatch.setattr("agentic_autorag.cli.configure_litellm_runtime", lambda: None)
    monkeypatch.setattr("agentic_autorag.orchestrator.Orchestrator", _fake_orchestrator_factory(output_dir))
    monkeypatch.setattr("agentic_autorag.examiner.exam_validator.run_validation_pipeline", _validation_passthrough)

    runner = CliRunner()
    result = runner.invoke(app, ["validate", "--config", "configs/full.yaml"])

    assert result.exit_code == 0
    output_path = output_dir / "exam.json"
    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert saved[0]["id"] == "q1"


def test_validate_honors_explicit_input_output_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    explicit_candidates = tmp_path / "my_candidates.json"
    explicit_output = tmp_path / "my_exam.json"
    explicit_candidates.write_text(json.dumps(_candidate_payload()), encoding="utf-8")

    monkeypatch.setattr("agentic_autorag.cli.configure_litellm_runtime", lambda: None)
    monkeypatch.setattr("agentic_autorag.orchestrator.Orchestrator", _fake_orchestrator_factory(output_dir))
    monkeypatch.setattr("agentic_autorag.examiner.exam_validator.run_validation_pipeline", _validation_passthrough)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "validate",
            "--config",
            "configs/full.yaml",
            "--candidates",
            str(explicit_candidates),
            "--output",
            str(explicit_output),
        ],
    )

    assert result.exit_code == 0
    assert explicit_output.exists()
    saved = json.loads(explicit_output.read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert saved[0]["id"] == "q1"
    assert not (output_dir / "exam.json").exists()
