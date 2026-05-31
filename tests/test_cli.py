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

    def test_removes_artifacts_including_debug_dir(self, tmp_path: Path) -> None:
        # Arrange — a populated output dir with headline + debug artifacts.
        out = tmp_path / "out"
        out.mkdir()
        (out / "run.log").write_text("log", encoding="utf-8")
        (out / "recommended.yaml").write_text("cfg", encoding="utf-8")
        debug = out / "debug"
        debug.mkdir()
        (debug / "composition_log.json").write_text("[]", encoding="utf-8")
        cfg = _write_config(tmp_path / "cfg.yaml", out)

        # Act
        result = runner.invoke(app, ["clean", "--config", str(cfg), "--yes"])

        # Assert — both the headline files and the debug dir are gone.
        assert result.exit_code == 0
        assert not (out / "run.log").exists()
        assert not (out / "recommended.yaml").exists()
        assert not debug.exists()
