"""Marker-Inc AutoRAG driver — wraps AutoRAG for fair cross-framework comparison.

Two QA-signal variants share most of the pipeline:

- ``autorag_ragas`` — AutoRAG's native ``autorag.data.qa`` bootstrap (factoid /
  RAGAS-style synthetic QA generated from the corpus alone).
- ``autorag_mcq`` — our cached ``exam.json`` converted to AutoRAG's
  ``qa.parquet`` schema, scored via a registered ``mcq_accuracy`` custom metric.

The two variants let the paper's ablation cleanly isolate exam-generation
quality with AutoRAG's algorithm held fixed.

AutoRAG itself is **not** a direct dependency of this package — its 0.3.x line
pins ``numpy<2``, which conflicts with the framework's base deps. The driver
shells out to a Python interpreter where AutoRAG is installed (path supplied
via the ``AUTORAG_PYTHON`` env var, e.g.
``AUTORAG_PYTHON=/path/to/autorag-venv/bin/python agentic-autorag baseline-optimize ...``).
"""
