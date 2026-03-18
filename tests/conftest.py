"""Shared pytest configuration and fixtures."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _bypass_llm_model_probe():
    """Patch _probe_model to always succeed so unit tests don't hit live APIs.

    Tests that need to verify probe failure behaviour explicitly re-patch
    ``agentic_autorag.config.models._probe_model`` inside the test body.
    """
    with patch("agentic_autorag.config.models._probe_model", return_value=(True, None)):
        yield


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", default=False, help="Run tests marked as slow")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="Slow test — pass --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
