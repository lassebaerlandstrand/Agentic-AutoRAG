"""CLI behaviour: argument validation, friendly setup errors, and clean."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from agentic_autorag.cli import app

runner = CliRunner()


_MINIMAL_CONFIG = """\
meta:
  project_name: cli-test
  output_dir: "{output_dir}"
agent:
  optimizer_model: "ollama/llama3.2"
  examiner_model: "ollama/llama3.2"
  judge_model: "ollama/llama3.2"
search_space:
  embedding:
    models: ["sentence-transformers/all-MiniLM-L6-v2"]
  generator:
    models: ["openai/gpt-4o-mini"]
"""


def _write_config(path: Path, output_dir: Path) -> Path:
    path.write_text(_MINIMAL_CONFIG.format(output_dir=output_dir), encoding="utf-8")
    return path


class TestRemovedDebugPromptsFlag:
    def test_debug_prompts_option_is_gone(self) -> None:
        # The flag was removed; prompts are always logged to run.log.
        result = runner.invoke(app, ["optimize", "--debug-prompts"])
        assert result.exit_code == 2
        assert "--debug-prompts" in result.output or "No such option" in result.output

    def test_help_does_not_mention_debug_prompts(self) -> None:
        result = runner.invoke(app, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "--debug-prompts" not in result.output


class TestFriendlySetupErrors:
    def test_missing_config_file(self, tmp_path: Path) -> None:
        # Arrange
        missing = tmp_path / "does_not_exist.yaml"
        # Act
        result = runner.invoke(app, ["optimize", "--config", str(missing)])
        # Assert — concise message, exit 1, no raw traceback.
        assert result.exit_code == 1
        assert "Config error" in result.output
        assert "Traceback" not in result.output

    def test_empty_config_file(self, tmp_path: Path) -> None:
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("", encoding="utf-8")
        result = runner.invoke(app, ["optimize", "--config", str(cfg)])
        assert result.exit_code == 1
        assert "empty" in result.output.lower()
        assert "Traceback" not in result.output

    def test_invalid_config_schema(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad.yaml"
        # Valid YAML, but not a valid ProjectConfig (missing required search_space).
        cfg.write_text("meta:\n  project_name: x\n", encoding="utf-8")
        result = runner.invoke(app, ["optimize", "--config", str(cfg)])
        assert result.exit_code == 1
        assert "Invalid config" in result.output
        assert "Traceback" not in result.output


class TestClean:
    def test_nothing_to_clean_when_output_dir_absent(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path / "cfg.yaml", tmp_path / "never_created")
        result = runner.invoke(app, ["clean", "--config", str(cfg)])
        assert result.exit_code == 0
        assert "Nothing to clean" in result.output

    def test_removes_artifacts_including_details_dir(self, tmp_path: Path) -> None:
        # Arrange — a populated output dir with headline + secondary artifacts.
        out = tmp_path / "out"
        out.mkdir()
        (out / "run.log").write_text("log", encoding="utf-8")
        (out / "recommended.yaml").write_text("cfg", encoding="utf-8")
        debug = out / "details" / "debug"
        debug.mkdir(parents=True)
        (debug / "composition_log.json").write_text("[]", encoding="utf-8")
        (out / "details" / "cost_breakdown.json").write_text("{}", encoding="utf-8")
        cfg = _write_config(tmp_path / "cfg.yaml", out)

        # Act
        result = runner.invoke(app, ["clean", "--config", str(cfg), "--yes"])

        # Assert — the headline files and the whole details/ tree are gone.
        assert result.exit_code == 0
        assert not (out / "run.log").exists()
        assert not (out / "recommended.yaml").exists()
        assert not (out / "details").exists()


class TestExamSummaryLines:
    """`_exam_summary_lines` renders the saturation summary from a question list."""

    @staticmethod
    def _q(qid: str, probe_outcomes: list[int], reasoning_type: str = "bridge", n_hops: int = 2):
        from agentic_autorag.config.models import OpenEndedQuestion

        docs = [f"doc_{i}" for i in range(n_hops)]
        return OpenEndedQuestion(
            id=qid,
            question=f"Question {qid}?",
            canonical_answer=f"answer_{qid}",
            reasoning_type=reasoning_type,
            source_doc_ids=docs,
            source_spans=[f"span {i}" for i in range(n_hops)],
            probe_outcomes=probe_outcomes,
        )

    def test_monotone_ladder_has_no_warning(self) -> None:
        from agentic_autorag.cli import _exam_summary_lines

        exam = [
            self._q("Q1", [0, 0, 0, 1]),
            self._q("Q2", [0, 0, 1, 1]),
            self._q("Q3", [0, 1, 1, 1]),
            self._q("Q4", [1, 1, 1, 1]),
        ]
        text = "\n".join(_exam_summary_lines(exam, Path("/tmp/exam.json")))
        assert "4 questions" in text
        assert "NON-MONOTONE" not in text
        # ladder: P1=.25 P2=.50 P3=.75 P4=1.0
        assert "0.25, 0.50, 0.75, 1.00" in text
        assert "saturated (k=4, every probe solved): 1" in text
        assert "too-hard  (k=0, no probe solved):       0" in text

    def test_non_monotone_ladder_flags_warning(self) -> None:
        from agentic_autorag.cli import _exam_summary_lines

        # P3 > P4 — the Tier-4 throttling pattern seen in the live MultiHop exam.
        exam = [
            self._q("Q1", [0, 0, 1, 0]),
            self._q("Q2", [0, 0, 1, 0]),
            self._q("Q3", [0, 0, 1, 1]),
        ]
        text = "\n".join(_exam_summary_lines(exam, Path("/tmp/exam.json")))
        assert "NON-MONOTONE" in text

    def test_no_probe_outcomes(self) -> None:
        from agentic_autorag.cli import _exam_summary_lines

        exam = [self._q("Q1", []), self._q("Q2", [])]
        text = "\n".join(_exam_summary_lines(exam, Path("/tmp/exam.json")))
        assert "no probe outcomes recorded" in text
        assert "2 questions" in text

    def test_empty_exam(self) -> None:
        from agentic_autorag.cli import _exam_summary_lines

        lines = _exam_summary_lines([], Path("/tmp/exam.json"))
        assert lines == ["Exam: 0 questions  ->  /tmp/exam.json"]
