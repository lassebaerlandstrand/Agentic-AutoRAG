"""LiteLLM runtime configuration helpers."""

import os

import litellm

DEFAULT_LOGGING_WORKER_TIMEOUT_SECONDS = 300.0


def configure_litellm_runtime() -> None:
    """Configure LiteLLM logging worker timeout for long-running workloads.

    Also enables ``drop_params`` so provider-specific parameters (e.g. OpenAI's
    ``seed``) are silently dropped for models that don't accept them, instead
    of raising UnsupportedParamsError. The whole point of LiteLLM here is
    cross-provider portability; strict per-param enforcement fights that.
    """
    os.environ.setdefault(
        "LOGGING_WORKER_MAX_TIME_PER_COROUTINE",
        str(DEFAULT_LOGGING_WORKER_TIMEOUT_SECONDS),
    )
    litellm.drop_params = True
