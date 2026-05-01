"""Shared pytest configuration and fixtures."""

import contextlib
import logging
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


_LOGGERS_TO_SNAPSHOT = ("agentic_autorag", "agentic_autorag.run")


@pytest.fixture(autouse=True)
def _isolate_agentic_autorag_loggers():
    """Snapshot/restore agentic_autorag.* logger state across tests.

    ``Orchestrator._setup_logger`` mutates both ``agentic_autorag`` and
    ``agentic_autorag.run`` (sets ``propagate=False``, attaches file
    handlers). Without isolation those mutations bleed into later tests
    that rely on caplog propagation, e.g. test_knowledge_base.
    """
    snapshots = []
    for name in _LOGGERS_TO_SNAPSHOT:
        log = logging.getLogger(name)
        snapshots.append((log, list(log.handlers), log.level, log.propagate))
    try:
        yield
    finally:
        for log, saved_handlers, saved_level, saved_propagate in snapshots:
            for handler in list(log.handlers):
                if handler not in saved_handlers:
                    log.removeHandler(handler)
                    with contextlib.suppress(Exception):
                        handler.close()
            for handler in saved_handlers:
                if handler not in log.handlers:
                    log.addHandler(handler)
            log.setLevel(saved_level)
            log.propagate = saved_propagate


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", default=False, help="Run tests marked as slow")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="Slow test — pass --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
