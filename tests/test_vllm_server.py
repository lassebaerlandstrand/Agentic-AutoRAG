"""Tests for VLLMServerManager subprocess lifecycle."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_autorag.config.models import VLLMConfig
from agentic_autorag.engine.vllm_server import VLLMServerManager


@pytest.fixture()
def tmp_output(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def _fake_vllm_binary():
    """Pretend the vllm binary exists on PATH."""
    with patch("shutil.which", return_value="/usr/bin/vllm"):
        yield


@pytest.fixture()
def manager(_fake_vllm_binary, tmp_output: Path) -> VLLMServerManager:
    return VLLMServerManager(VLLMConfig(), tmp_output)


def _mock_popen(returncode: int | None = None) -> MagicMock:
    proc = MagicMock()
    proc.poll.return_value = returncode  # None = still running
    proc.pid = 12345
    proc.returncode = returncode
    return proc


class TestInit:
    def test_missing_binary_raises(self, tmp_output: Path) -> None:
        with patch("shutil.which", return_value=None), pytest.raises(FileNotFoundError, match="vllm"):
            VLLMServerManager(VLLMConfig(), tmp_output)

    def test_custom_binary(self, tmp_output: Path) -> None:
        with patch("shutil.which", return_value="/opt/vllm"):
            mgr = VLLMServerManager(VLLMConfig(binary="/opt/vllm"), tmp_output)
            assert mgr._config.binary == "/opt/vllm"


class TestEnsureModel:
    @pytest.mark.asyncio
    async def test_starts_server(self, manager: VLLMServerManager) -> None:
        proc = _mock_popen()
        with (
            patch("subprocess.Popen", return_value=proc) as popen_mock,
            patch.object(manager, "_wait_ready"),
        ):
            await manager.ensure_model("hosted_vllm/Qwen/Qwen3-8B")
            cmd = popen_mock.call_args[0][0]
            assert cmd[0] == "vllm"
            assert cmd[1] == "serve"
            assert cmd[2] == "Qwen/Qwen3-8B"
            assert "--enforce-eager" in cmd
            assert manager._current_model == "Qwen/Qwen3-8B"

    @pytest.mark.asyncio
    async def test_same_model_is_noop(self, manager: VLLMServerManager) -> None:
        proc = _mock_popen()
        with (
            patch("subprocess.Popen", return_value=proc) as popen_mock,
            patch.object(manager, "_wait_ready"),
        ):
            await manager.ensure_model("hosted_vllm/Qwen/Qwen3-8B")
            assert popen_mock.call_count == 1
            # Second call with same model — no new process
            await manager.ensure_model("hosted_vllm/Qwen/Qwen3-8B")
            assert popen_mock.call_count == 1

    @pytest.mark.asyncio
    async def test_different_model_swaps(self, manager: VLLMServerManager) -> None:
        proc_a = _mock_popen()
        proc_b = _mock_popen()
        procs = iter([proc_a, proc_b])
        with (
            patch("subprocess.Popen", side_effect=lambda *a, **kw: next(procs)),
            patch.object(manager, "_wait_ready"),
        ):
            await manager.ensure_model("hosted_vllm/model-a")
            assert manager._current_model == "model-a"
            await manager.ensure_model("hosted_vllm/model-b")
            proc_a.terminate.assert_called_once()
            assert manager._current_model == "model-b"

    @pytest.mark.asyncio
    async def test_sets_api_base_env_var(self, manager: VLLMServerManager, monkeypatch) -> None:
        monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
        proc = _mock_popen()
        with (
            patch("subprocess.Popen", return_value=proc),
            patch.object(manager, "_wait_ready"),
        ):
            await manager.ensure_model("hosted_vllm/some/model")
            import os

            assert os.environ["HOSTED_VLLM_API_BASE"] == "http://localhost:8000/v1"

    @pytest.mark.asyncio
    async def test_max_model_len_only_when_set(self, _fake_vllm_binary, tmp_output: Path) -> None:
        mgr = VLLMServerManager(VLLMConfig(max_model_len=4096), tmp_output)
        proc = _mock_popen()
        with (
            patch("subprocess.Popen", return_value=proc) as popen_mock,
            patch.object(mgr, "_wait_ready"),
        ):
            await mgr.ensure_model("hosted_vllm/some/model")
            cmd = popen_mock.call_args[0][0]
            assert "--max-model-len" in cmd
            idx = cmd.index("--max-model-len")
            assert cmd[idx + 1] == "4096"

    @pytest.mark.asyncio
    async def test_no_max_model_len_by_default(self, manager: VLLMServerManager) -> None:
        proc = _mock_popen()
        with (
            patch("subprocess.Popen", return_value=proc) as popen_mock,
            patch.object(manager, "_wait_ready"),
        ):
            await manager.ensure_model("hosted_vllm/some/model")
            cmd = popen_mock.call_args[0][0]
            assert "--max-model-len" not in cmd


class TestShutdown:
    @pytest.mark.asyncio
    async def test_terminates_process(self, manager: VLLMServerManager) -> None:
        proc = _mock_popen()
        with (
            patch("subprocess.Popen", return_value=proc),
            patch.object(manager, "_wait_ready"),
        ):
            await manager.ensure_model("hosted_vllm/some/model")
            await manager.shutdown()
            proc.terminate.assert_called_once()
            assert manager._current_model is None
            assert manager._process is None

    @pytest.mark.asyncio
    async def test_shutdown_when_not_running_is_noop(self, manager: VLLMServerManager) -> None:
        await manager.shutdown()  # Should not raise


class TestWaitReady:
    @pytest.mark.asyncio
    async def test_process_dies_raises_runtime_error(self, manager: VLLMServerManager) -> None:
        proc = _mock_popen(returncode=1)  # Already exited
        manager._process = proc
        manager._current_model = "test"
        with pytest.raises(RuntimeError, match="exited with code 1"):
            await manager._wait_ready()

    @pytest.mark.asyncio
    async def test_timeout_raises(self, _fake_vllm_binary, tmp_output: Path) -> None:
        mgr = VLLMServerManager(VLLMConfig(startup_timeout=10), tmp_output)
        proc = _mock_popen()  # Alive but never ready
        mgr._process = proc
        mgr._current_model = "test"
        with (
            patch("urllib.request.urlopen", side_effect=ConnectionRefusedError),
            pytest.raises(TimeoutError, match="did not become ready"),
        ):
            await mgr._wait_ready()
