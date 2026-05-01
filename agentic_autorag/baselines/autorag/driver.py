"""AutoRAG baseline driver — orchestrates both ``autorag_ragas`` and ``autorag_mcq``."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal

import yaml

from agentic_autorag.baselines.autorag.config_translator import translate_extracted_to_trial_config
from agentic_autorag.baselines.autorag.corpus_export import export_corpus_to_parquet
from agentic_autorag.baselines.autorag.native_config import generate_autorag_config
from agentic_autorag.baselines.autorag.qa_mcq import export_mcq_exam_to_parquet
from agentic_autorag.baselines.autorag.qa_ragas import export_ragas_qa_via_subprocess
from agentic_autorag.baselines.base import (
    BaselineMeta,
    save_best_config,
    write_optimizer_meta,
)
from agentic_autorag.optimizer.history import TrialRecord
from agentic_autorag.orchestrator import Orchestrator

logger = logging.getLogger("agentic_autorag.run")

QAVariant = Literal["ragas", "mcq"]


def _find_extracted_sample(project_dir: Path) -> Path | None:
    """Locate AutoRAG's resolved best-pipeline yaml under ``project_dir``."""
    for candidate in sorted(project_dir.rglob("extracted_sample.yaml")):
        return candidate
    return None


async def run_autorag_baseline(
    config_path: str,
    output_dir: str,
    qa_variant: QAVariant,
    *,
    autorag_python: str | None = None,
    sample_n: int | None = None,
) -> TrialRecord | None:
    """Drive an AutoRAG baseline (RAGAS or MCQ variant).

    Parameters
    ----------
    config_path:
        Path to the project YAML.
    output_dir:
        Per-run output directory (best_config.yaml, autorag_native_config.yaml,
        translation_notes.json, optimizer_meta.json, history.jsonl).
    qa_variant:
        ``"ragas"`` or ``"mcq"``.
    autorag_python:
        Path to the AutoRAG-venv Python. Falls back to ``AUTORAG_PYTHON`` env
        var. If unset, the driver stages artifacts and exits with instructions
        instead of executing AutoRAG.
    sample_n:
        QA-set size. For RAGAS, defaults to ``meta.examiner.exam_size`` so the
        signal volume matches our MCQ exam.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    autorag_dir = out_dir / "autorag_project"
    autorag_dir.mkdir(parents=True, exist_ok=True)

    orch = Orchestrator(config_path, output_dir_override=str(out_dir))

    # Baselines maintain their own history at output_dir/history.jsonl.
    history = orch.history
    history.clear()

    algorithm_name = f"autorag_{qa_variant}"
    meta = BaselineMeta(algorithm=algorithm_name, seed=0, max_trials=orch.config.meta.max_trials)
    autorag_python = autorag_python or os.environ.get("AUTORAG_PYTHON")
    meta.extras["autorag_python"] = autorag_python or "<not set>"
    meta.extras["qa_variant"] = qa_variant

    logger.info("=" * 60)
    logger.info("AutoRAG baseline (%s variant)", qa_variant)
    logger.info("=" * 60)

    t_start = time.monotonic()
    best: TrialRecord | None = None
    try:
        # Setup is required for the MCQ variant (needs cached exam.json) and is
        # cheap for the RAGAS variant when caches are warm.
        await orch.setup()

        # 1. Export corpus to parquet.
        corpus_parquet = autorag_dir / "corpus.parquet"
        n_corpus = export_corpus_to_parquet(Path(orch.config.meta.corpus_path), corpus_parquet)
        logger.info("Exported %d documents to %s", n_corpus, corpus_parquet.name)

        # 2. Build QA parquet (variant-specific).
        qa_parquet = autorag_dir / "qa.parquet"
        if qa_variant == "mcq":
            exam_json = orch.cache_dir / "exam.json"
            if not exam_json.exists():
                raise RuntimeError(
                    f"AutoRAG-MCQ requires a cached exam.json at {exam_json}. "
                    "Run `optimize` first or any baseline that triggers exam generation."
                )
            n_qa = export_mcq_exam_to_parquet(exam_json, qa_parquet)
            logger.info("Exported %d MCQ rows to %s", n_qa, qa_parquet.name)
            # Copy the metric file next to the AutoRAG config so the AutoRAG
            # subprocess can import it via PYTHONPATH (see step 5).
            shutil.copy2(Path(__file__).parent / "mcq_metric.py", autorag_dir / "mcq_metric.py")
        else:
            sample_n = sample_n or orch.config.examiner.exam_size
            export_ragas_qa_via_subprocess(
                corpus_parquet,
                qa_parquet,
                sample_n=sample_n,
                llm_model=orch.config.agent.examiner_model,
                autorag_python=autorag_python,
            )

        # 3. Generate AutoRAG config + translation notes (search-space mirroring).
        autorag_config_dict, notes = generate_autorag_config(orch.config.search_space, qa_variant=qa_variant)
        autorag_config_path = autorag_dir / "autorag_config.yaml"
        autorag_config_path.write_text(yaml.safe_dump(autorag_config_dict, sort_keys=False), encoding="utf-8")
        (autorag_dir / "translation_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
        logger.info("Wrote AutoRAG config to %s", autorag_config_path.name)

        # 4. Execute AutoRAG (or stage and exit with instructions).
        if not autorag_python:
            logger.warning(
                "AUTORAG_PYTHON not set — staged artifacts in %s. Set AUTORAG_PYTHON to a "
                "Python interpreter with AutoRAG installed and re-invoke this command.",
                autorag_dir,
            )
        elif _find_extracted_sample(autorag_dir) is None:
            logger.info("Invoking AutoRAG in %s", autorag_python)
            env = dict(os.environ)
            if qa_variant == "mcq":
                env["PYTHONPATH"] = f"{autorag_dir}:{env.get('PYTHONPATH', '')}"
            result = subprocess.run(
                [
                    autorag_python,
                    "-m",
                    "autorag",
                    "evaluate",
                    "--config",
                    str(autorag_config_path),
                    "--qa_data_path",
                    str(qa_parquet),
                    "--corpus_data_path",
                    str(corpus_parquet),
                    "--project_dir",
                    str(autorag_dir),
                ],
                check=False,
                env=env,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                logger.info(result.stdout.rstrip())
            if result.stderr:
                logger.warning(result.stderr.rstrip())
            if result.returncode != 0:
                raise RuntimeError(f"AutoRAG evaluate exited with rc={result.returncode}.")

        # 5. Translate winning pipeline → our TrialConfig.
        extracted = _find_extracted_sample(autorag_dir)
        if extracted is None:
            logger.warning(
                "No extracted_sample.yaml found under %s — skipping translation. "
                "Re-invoke after AutoRAG produces results.",
                autorag_dir,
            )
        else:
            logger.info("Translating %s → TrialConfig", extracted)
            trial_config = translate_extracted_to_trial_config(extracted, orch.config.search_space)
            violations = orch.config.validate_trial(trial_config)
            if violations:
                logger.warning(
                    "Translated config has validation issues (will save anyway): %s",
                    "; ".join(violations),
                )
            # Score the translated winning config on our MCQ exam so we get a
            # comparable score field in best_config.yaml's history.
            try:
                result = await orch.evaluate_trial(trial_config)
                record = TrialRecord(
                    trial_number=1,
                    config=trial_config,
                    score=result.score,
                    question_results=result.question_results,
                    answer_accuracy=result.answer_accuracy,
                    mean_retrieval_quality=result.mean_retrieval_quality,
                    n_em_correct=result.n_em_correct,
                    n_judge_correct=result.n_judge_correct,
                    n_judge_rejected=result.n_judge_rejected,
                    n_judge_failed=result.n_judge_failed,
                    n_no_answer=result.n_no_answer,
                    n_judge_calls=result.n_judge_calls,
                    mean_em=result.mean_em,
                    mean_f1=result.mean_f1,
                )
                history.add(record)
                meta.n_trials_completed = 1
                best = record
            except Exception:
                logger.exception("Failed to score the translated config; saving config anyway")
                record = TrialRecord(
                    trial_number=1,
                    config=trial_config,
                    score=0.0,
                    question_results=[],
                )
                history.add(record)
                best = record
            save_best_config(out_dir, best, include_graph=orch.config.uses_graph())
    finally:
        meta.wall_clock_s = round(time.monotonic() - t_start, 3)
        write_optimizer_meta(out_dir, meta)
        await orch.cleanup()

    if best:
        logger.info(
            "AutoRAG-%s complete | translated score=%.3f | %.1fs total",
            qa_variant,
            best.score,
            meta.wall_clock_s,
        )
    else:
        logger.warning("AutoRAG-%s did not produce a translatable winning pipeline", qa_variant)
    return best
