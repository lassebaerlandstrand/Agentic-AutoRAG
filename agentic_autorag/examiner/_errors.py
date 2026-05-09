"""Shared error formatting and retry constants for LiteLLM exceptions."""

from __future__ import annotations

import json

# Escalating cooldown (seconds) between retries on transient LLM errors.
RETRY_COOLDOWNS_S: tuple[int, ...] = (10, 30, 60)

# Sentinel strings stored on result rows when an LLM call fails.
TRANSIENT_ERROR_SENTINEL = "TRANSIENT_LLM_ERROR"
PERMANENT_ERROR_SENTINEL = "PERMANENT_LLM_ERROR"
ERROR_SENTINELS: tuple[str, str] = (TRANSIENT_ERROR_SENTINEL, PERMANENT_ERROR_SENTINEL)


def format_llm_error(exc: Exception) -> str:
    """Format an LLM exception into a concise one-liner with code and message.

    LiteLLM errors often embed a JSON body across multiple lines. This extracts
    the code and message fields when present, falling back to the raw first line.
    """
    raw = str(exc)
    brace_start = raw.find("{")
    if brace_start != -1:
        try:
            data = json.loads(raw[brace_start:])
            err = data.get("error", data)
            code = err.get("code", "")
            message = err.get("message", "")
            status = err.get("status", "")
            parts = [str(p) for p in (code, status, message) if p]
            if parts:
                return f"{type(exc).__name__}: {' / '.join(parts)}"
        except (json.JSONDecodeError, AttributeError):
            pass
    first_line = raw.split("\n", 1)[0]
    return f"{type(exc).__name__}: {first_line}"


def is_permanent_llm_error(exc: Exception) -> bool:
    """Return True if the exception is a permanent LLM error that retrying won't fix.

    Covers content policy violations, authentication errors, and other 4xx
    client errors that indicate the request itself is invalid.
    """
    type_name = type(exc).__name__
    if type_name in (
        "ContentPolicyViolationError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
    ):
        return True

    if hasattr(exc, "status_code") and exc.status_code in (400, 401, 403, 404):
        return True

    raw = str(exc)
    if "ContentPolicyViolation" in raw:
        return True

    brace_start = raw.find("{")
    if brace_start != -1:
        try:
            data = json.loads(raw[brace_start:])
            err = data.get("error", data)
            code = err.get("code")
            if isinstance(code, int) and code in (400, 401, 403, 404):
                return True
            if isinstance(code, str) and code in ("content_filter", "content_policy_violation"):
                return True
        except (json.JSONDecodeError, AttributeError):
            pass

    return False


def is_transient_llm_error(exc: Exception) -> bool:
    """Return True if the exception looks like a transient LLM provider error.

    Checks exception class names first (cheap), then falls back to looking for
    HTTP status codes in the structured error body that LiteLLM embeds.
    """
    type_name = type(exc).__name__
    if type_name in (
        "ServiceUnavailableError",
        "RateLimitError",
        "InternalServerError",
        "APIConnectionError",
        "Timeout",
    ):
        return True

    if hasattr(exc, "status_code") and exc.status_code in (429, 500, 502, 503):
        return True

    raw = str(exc)
    brace_start = raw.find("{")
    if brace_start != -1:
        try:
            data = json.loads(raw[brace_start:])
            err = data.get("error", data)
            code = err.get("code")
            if isinstance(code, int) and code in (429, 500, 502, 503):
                return True
        except (json.JSONDecodeError, AttributeError):
            pass

    return False
