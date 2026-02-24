"""LiteLLM runtime configuration helpers."""

import os

DEFAULT_LOGGING_WORKER_TIMEOUT_SECONDS = 300.0


def configure_litellm_runtime() -> None:
    """Configure LiteLLM logging worker timeout for long-running workloads."""
    os.environ.setdefault(
        "LOGGING_WORKER_MAX_TIME_PER_COROUTINE",
        str(DEFAULT_LOGGING_WORKER_TIMEOUT_SECONDS),
    )
