"""Test isolation for the baselines test suite.

The baseline drivers construct ``Orchestrator``, which calls ``_setup_logger`` —
that mutates the shared ``agentic_autorag.run`` logger AND the
``agentic_autorag`` parent logger (sets ``propagate=False``, swaps handlers,
attaches a file handler under the run output_dir). Without isolation, those
side effects bleed into later tests that rely on caplog propagation (e.g.
tests/test_knowledge_base.py, tests/test_exam_agent.py).

This autouse fixture snapshots the logger state before each baseline test and
restores it after, keeping tests deterministic regardless of run order.
"""

from __future__ import annotations

import contextlib
import logging

import pytest

_TRACKED_LOGGERS = ("agentic_autorag", "agentic_autorag.run")


@pytest.fixture(autouse=True)
def _isolate_run_logger():
    """Snapshot and restore agentic_autorag.* logger state per test."""
    snapshots = []
    for name in _TRACKED_LOGGERS:
        log = logging.getLogger(name)
        snapshots.append(
            (
                log,
                list(log.handlers),
                log.level,
                log.propagate,
            )
        )
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
