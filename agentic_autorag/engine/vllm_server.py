"""Managed vLLM server subprocess for hosted_vllm/ models."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import urllib.request
from io import TextIOWrapper
from pathlib import Path

from agentic_autorag.config.models import VLLMConfig

logger = logging.getLogger(__name__)

_HOSTED_VLLM_PREFIX = "hosted_vllm/"
_READINESS_POLL_INTERVAL_S = 2


class VLLMServerManager:
    """Manages a vLLM server subprocess for local model serving.

    Starts a ``vllm serve`` process, waits for it to become ready, and
    exposes a method to swap models between optimization trials.
    """

    def __init__(self, config: VLLMConfig | None, output_dir: Path) -> None:
        self._config = config or VLLMConfig()
        self._output_dir = output_dir
        self._process: subprocess.Popen[bytes] | None = None
        self._current_model: str | None = None
        self._log_file: TextIOWrapper | None = None
        self._validate_binary()

    def _validate_binary(self) -> None:
        binary = self._config.binary
        if shutil.which(binary) is None:
            raise FileNotFoundError(f"vLLM binary '{binary}' not found on PATH. Install with: uv sync --extra vllm")

    @property
    def api_base(self) -> str:
        return f"http://localhost:{self._config.port}/v1"

    async def ensure_model(self, litellm_model: str) -> None:
        """Ensure vLLM is serving the requested model.

        No-op if the model is already loaded. Otherwise stops the current
        server (if any) and starts a new one.
        """
        hf_model = litellm_model.removeprefix(_HOSTED_VLLM_PREFIX)
        if self._current_model == hf_model and self._is_alive():
            return
        logger.info("vLLM model swap: %s → %s", self._current_model or "(none)", hf_model)
        await self.shutdown()
        self._start(hf_model)
        await self._wait_ready()
        logger.info("vLLM server ready for %s", hf_model)

    async def shutdown(self) -> None:
        """Stop the current vLLM server process."""
        if self._process is None:
            return
        if self._process.poll() is None:
            logger.info("Stopping vLLM server (pid %d)", self._process.pid)
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None
        self._current_model = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _start(self, hf_model: str) -> None:
        cfg = self._config
        cmd: list[str] = [
            cfg.binary,
            "serve",
            hf_model,
            "--port",
            str(cfg.port),
            "--gpu-memory-utilization",
            str(cfg.gpu_memory_utilization),
        ]
        if cfg.max_model_len is not None:
            cmd.extend(["--max-model-len", str(cfg.max_model_len)])
        if cfg.enforce_eager:
            cmd.append("--enforce-eager")
        cmd.extend(cfg.extra_args)

        log_name = hf_model.replace("/", "_")
        log_path = self._output_dir / f"vllm_{log_name}.log"
        self._log_file = open(log_path, "w")  # noqa: SIM115

        logger.info("Starting vLLM: %s", " ".join(cmd))
        logger.info("vLLM log: %s", log_path)
        self._process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )
        self._current_model = hf_model
        os.environ["HOSTED_VLLM_API_BASE"] = self.api_base

    async def _wait_ready(self) -> None:
        """Poll /v1/models until the server is ready or timeout."""
        url = f"{self.api_base}/models"
        deadline = asyncio.get_event_loop().time() + self._config.startup_timeout
        while asyncio.get_event_loop().time() < deadline:
            if not self._is_alive():
                raise RuntimeError(
                    f"vLLM process exited with code {self._process.returncode} "
                    f"before becoming ready. Check log: {self._output_dir}"
                )
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(_READINESS_POLL_INTERVAL_S)
        raise TimeoutError(
            f"vLLM did not become ready within {self._config.startup_timeout}s. Check log: {self._output_dir}"
        )
