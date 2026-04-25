"""Baseline optimizers for cross-framework comparison.

This sub-package implements competitor optimizers (Random search, Bayesian/Optuna
TPE, Marker-Inc AutoRAG with two QA-signal variants) that share the framework's
``SearchSpace`` + ``MCQEvaluator`` machinery via ``Orchestrator.setup()`` and
``Orchestrator.evaluate_trial()``. Only the proposal strategy differs across
baselines; every other moving part (corpus parse, exam generation, ingredient
cache, scoring) is identical to the agentic ``optimize`` path.

Optional dependencies (Optuna, pyarrow, pandas) live in the ``[baselines]``
extras group. AutoRAG itself is invoked via a separate venv (``AUTORAG_PYTHON``)
because its numpy<2 pin conflicts with our base dependencies.
"""
