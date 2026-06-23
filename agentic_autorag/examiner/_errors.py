"""Shared error formatting and retry constants for LiteLLM exceptions."""

from __future__ import annotations

import json

# Escalating cooldown (seconds) between retries on transient LLM errors.
RETRY_COOLDOWNS_S: tuple[int, ...] = (10, 30, 60)

# Sentinel strings stored on result rows when an LLM call fails.
TRANSIENT_ERROR_SENTINEL = "TRANSIENT_LLM_ERROR"
PERMANENT_ERROR_SENTINEL = "PERMANENT_LLM_ERROR"
# Content-filter is a distinct category from PERMANENT because it identifies
# the *question* as unanswerable by this provider — every method evaluating
# the same hold-out should drop that question for a shared denominator.
# Authentication / 404 errors, by contrast, are system-wide and per-method.
CONTENT_FILTER_SENTINEL = "CONTENT_FILTER"
ERROR_SENTINELS: tuple[str, ...] = (
    TRANSIENT_ERROR_SENTINEL,
    PERMANENT_ERROR_SENTINEL,
    CONTENT_FILTER_SENTINEL,
)


class AllQuestionsErrored(RuntimeError):
    """Every exam question hit an error sentinel — usually a broken endpoint.

    The orchestrator raises this after evaluation when ``n_valid == 0`` so the
    existing failure-recovery branch routes through ``propose_after_failure``
    with a meaningful error summary instead of recording a misleading 0%
    trial.
    """

    def __init__(self, error_sentinel: str | None, n_total: int) -> None:
        self.error_sentinel = error_sentinel
        self.n_total = n_total
        msg = (
            f"All {n_total} exam questions hit an error sentinel "
            f"({error_sentinel or 'unknown'}). Likely a broken endpoint, "
            "credential failure, or model unavailability."
        )
        super().__init__(msg)


class ExamGenerationFailed(RuntimeError):
    """Exam generation produced too few questions to optimize against.

    Raised by ``orchestrator._generate_exam`` when the final exam is empty or
    falls below the minimum-fraction threshold of the configured ``exam_size``.
    Surfacing this early prevents the optimizer from billing many trials at
    Score 0.0 against a degenerate exam.
    """

    def __init__(
        self,
        n_actual: int,
        n_target: int,
        candidates_path: str,
        top_rejection_reasons: list[tuple[str, int]],
        stage_counts: dict[str, int] | None = None,
    ) -> None:
        self.n_actual = n_actual
        self.n_target = n_target
        self.candidates_path = candidates_path
        self.top_rejection_reasons = top_rejection_reasons
        self.stage_counts = stage_counts or {}

        if top_rejection_reasons:
            reasons_str = ", ".join(f"{reason}={count}" for reason, count in top_rejection_reasons)
        else:
            reasons_str = "no rejection counter available (check logs)"

        if self.stage_counts:
            stages_str = " ".join(f"{stage}={count}" for stage, count in self.stage_counts.items())
            stages_line = f"\nStage funnel: {stages_str}"
        else:
            stages_line = ""

        msg = (
            f"Exam generation produced {n_actual} question(s) — too few relative to the "
            f"target exam_size of {n_target} (it fell below the minimum-fraction floor). "
            f"The corpus may be too small, topically disjoint, the exam_size too large for "
            f"the corpus, or the LLM may be refusing under content-policy filters. "
            f"Raise initial_question_multiplier or lower exam_size.\n"
            f"See {candidates_path} for per-seed rejection explanations.\n"
            f"Top rejection reasons: {reasons_str}"
            f"{stages_line}"
        )
        super().__init__(msg)


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


def is_content_filter_error(exc: Exception) -> bool:
    """Return True if the exception is a provider content-policy rejection.

    Identifies a question that the provider refuses to answer regardless of
    pipeline configuration — every method evaluating the same hold-out will
    see the same rejection, so the question should be excluded from all
    score denominators (not just the method that observed it first).
    """
    if type(exc).__name__ == "ContentPolicyViolationError":
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
            if isinstance(code, str) and code in ("content_filter", "content_policy_violation"):
                return True
            innererror = err.get("innererror") if isinstance(err, dict) else None
            if isinstance(innererror, dict):
                inner_code = innererror.get("code")
                if isinstance(inner_code, str) and inner_code in (
                    "ResponsibleAIPolicyViolation",
                    "content_filter",
                ):
                    return True
        except (json.JSONDecodeError, AttributeError):
            pass

    return False


def is_permanent_llm_error(exc: Exception) -> bool:
    """Return True if the exception is a permanent LLM error that retrying won't fix.

    Covers content policy violations, authentication errors, and other 4xx
    client errors that indicate the request itself is invalid. Callers that
    need to single out content-filter rejections (so they can be excluded
    from cross-method denominators) should check ``is_content_filter_error``
    first.
    """
    if is_content_filter_error(exc):
        return True

    type_name = type(exc).__name__
    if type_name in (
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
    ):
        return True

    if hasattr(exc, "status_code") and exc.status_code in (400, 401, 403, 404):
        return True

    brace_start = str(exc).find("{")
    if brace_start != -1:
        try:
            data = json.loads(str(exc)[brace_start:])
            err = data.get("error", data)
            code = err.get("code")
            if isinstance(code, int) and code in (400, 401, 403, 404):
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
