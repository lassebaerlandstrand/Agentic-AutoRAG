"""Shared error formatting for LiteLLM exceptions."""

from __future__ import annotations

import json


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
