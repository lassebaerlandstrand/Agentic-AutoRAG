"""Random-search baseline driver.

Uniformly samples ``TrialConfig`` from the project's ``SearchSpace``, scores each
proposal via the orchestrator's ``evaluate_trial`` (same MCQ exam, same
evaluator, same caches as the agentic ``optimize`` path), and writes the
standard baseline outputs (``best_config.yaml`` + ``history.jsonl`` +
``optimizer_meta.json``) into the per-run ``output_dir``.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

from agentic_autorag.baselines.base import (
    BaselineMeta,
    save_best_config,
    write_optimizer_meta,
)
from agentic_autorag.baselines.sampler import sample_trial_config_random
from agentic_autorag.optimizer.history import TrialRecord
from agentic_autorag.orchestrator import Orchestrator

logger = logging.getLogger("agentic_autorag.run")

ALGORITHM_NAME = "random"


async def run_random_search(
    config_path: str,
    output_dir: str,
    seed: int = 42,
    max_trials: int | None = None,
) -> TrialRecord | None:
    """Drive a random-search baseline.

    Parameters
    ----------
    config_path:
        Path to the project YAML. Its ``meta.output_dir`` controls the shared
        cache (corpus parse, exam.json, ingredient cache, graph) — point
        multiple baselines at the same YAML to reuse all of these.
    output_dir:
        Per-run output directory. Receives ``best_config.yaml``,
        ``history.jsonl``, ``optimizer_meta.json``, ``run.log``. Distinct from
        the cache to keep per-baseline outputs isolated.
    seed:
        Seed for the ``random.Random`` instance driving sampling. Reproducible.
    max_trials:
        How many trials to draw. Defaults to ``meta.max_trials`` from the YAML.
    """
    out_dir = Path(output_dir)
    orch = Orchestrator(config_path, output_dir_override=str(out_dir))
    if max_trials is None:
        max_trials = orch.config.meta.max_trials

    # Baselines maintain their own history at output_dir/history.jsonl.
    # Orchestrator's __init__ already created a HistoryLog there; clear stale
    # records from a prior run so this run starts fresh.
    history = orch.history
    history.clear()

    rng = random.Random(seed)
    meta = BaselineMeta(algorithm=ALGORITHM_NAME, seed=seed, max_trials=max_trials)

    logger.info("=" * 60)
    logger.info("Random-search baseline | seed=%d | max_trials=%d", seed, max_trials)
    logger.info("=" * 60)

    t_start = time.monotonic()
    try:
        await orch.setup()
        for trial_num in range(1, max_trials + 1):
            logger.info("=" * 60)
            logger.info("RANDOM TRIAL %d/%d", trial_num, max_trials)
            logger.info("=" * 60)

            trial_config = sample_trial_config_random(rng, orch.config.search_space, orch.config.embedding_token_limits)
            violations = orch.config.validate_trial(trial_config)
            if violations:
                logger.warning("Random sample failed validation: %s — skipping", "; ".join(violations))
                meta.n_validation_rejects += 1
                continue

            orch._log_config_summary("Config", trial_config)

            try:
                result = await orch.evaluate_trial(trial_config)
            except Exception:
                logger.exception("Trial %d evaluation failed; skipping", trial_num)
                continue

            record = TrialRecord(
                trial_number=trial_num,
                config=trial_config,
                score=result.score,
                question_results=result.question_results,
                mcq_accuracy=result.mcq_accuracy,
                mean_retrieval_quality=result.mean_retrieval_quality,
            )
            history.add(record)
            meta.n_trials_completed += 1
            logger.info(
                "Random trial %d done | score=%.3f | best so far=%.3f",
                trial_num,
                result.score,
                history.get_best().score if history.get_best() else result.score,
            )

        best = history.get_best()
        save_best_config(out_dir, best, include_graph=orch.config.uses_graph())
    finally:
        meta.wall_clock_s = round(time.monotonic() - t_start, 3)
        write_optimizer_meta(out_dir, meta)
        await orch.cleanup()

    if best:
        logger.info("Random search complete | best score=%.3f | %.1fs total", best.score, meta.wall_clock_s)
    else:
        logger.warning("Random search produced no successful trials in %.1fs", meta.wall_clock_s)
    return best
