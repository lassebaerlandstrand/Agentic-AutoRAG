"""Unit tests for the LLM-exception classifiers.

The bench-level union-exclusion of content-filtered questions depends on
``is_content_filter_error`` returning True for the exception shapes Azure
emits (typed class, inner-error JSON body, naked status_code), and on
``is_permanent_llm_error`` continuing to cover everything ``is_content_filter_error``
covers so the trial-time examiner path still treats them as terminal.
"""

from __future__ import annotations

from agentic_autorag.examiner._errors import (
    is_content_filter_error,
    is_permanent_llm_error,
    is_transient_llm_error,
)


class _ContentPolicy(Exception):
    pass


_ContentPolicy.__name__ = "ContentPolicyViolationError"


class _AuthError(Exception):
    pass


_AuthError.__name__ = "AuthenticationError"


class _StatusError(Exception):
    def __init__(self, status_code: int, msg: str = "") -> None:
        super().__init__(msg)
        self.status_code = status_code


def test_content_filter_matches_typed_exception() -> None:
    assert is_content_filter_error(_ContentPolicy("blocked"))


def test_content_filter_matches_raw_string() -> None:
    assert is_content_filter_error(Exception("ContentPolicyViolation: prompt rejected"))


def test_content_filter_matches_azure_inner_error_json() -> None:
    body = (
        'litellm.BadRequestError: AzureException: {"error": {"code": "content_filter", '
        '"innererror": {"code": "ResponsibleAIPolicyViolation"}}}'
    )
    assert is_content_filter_error(Exception(body))


def test_content_filter_rejects_auth_error() -> None:
    assert not is_content_filter_error(_AuthError("invalid api key"))


def test_permanent_subsumes_content_filter() -> None:
    """Trial-time examiner uses is_permanent_llm_error to decide whether to
    skip retries — it must still return True for content-filter so that path
    keeps its existing terminal behaviour."""
    assert is_permanent_llm_error(_ContentPolicy("blocked"))


def test_permanent_covers_auth() -> None:
    assert is_permanent_llm_error(_AuthError("invalid api key"))


def test_permanent_covers_400_status() -> None:
    assert is_permanent_llm_error(_StatusError(400, "bad request"))


def test_transient_does_not_pick_up_content_filter() -> None:
    assert not is_transient_llm_error(_ContentPolicy("blocked"))
