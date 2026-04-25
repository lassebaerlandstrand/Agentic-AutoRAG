"""Test isolation for the baselines test suite.

The baseline drivers construct ``Orchestrator``, which calls ``_setup_logger`` —
that mutates the shared ``agentic_autorag.run`` logger (sets ``propagate=False``,
swaps handlers, attaches a file handler under the run output_dir). Without
isolation, those side effects bleed into later tests that rely on caplog
propagation (e.g. tests/test_exam_agent.py).

This autouse fixture snapshots the logger state before each baseline test and
restores it after, keeping tests deterministic regardless of run order.
"""

from __future__ import annotations

import contextlib
import logging

import pytest


@pytest.fixture(autouse=True)
def _isolate_run_logger():
    """Snapshot and restore the ``agentic_autorag.run`` logger state per test."""
    run_logger = logging.getLogger("agentic_autorag.run")
    saved_handlers = list(run_logger.handlers)
    saved_level = run_logger.level
    saved_propagate = run_logger.propagate
    try:
        yield
    finally:
        # Drop anything the test added.
        for handler in list(run_logger.handlers):
            if handler not in saved_handlers:
                run_logger.removeHandler(handler)
                with contextlib.suppress(Exception):
                    handler.close()
        # Restore originals.
        for handler in saved_handlers:
            if handler not in run_logger.handlers:
                run_logger.addHandler(handler)
        run_logger.setLevel(saved_level)
        run_logger.propagate = saved_propagate
