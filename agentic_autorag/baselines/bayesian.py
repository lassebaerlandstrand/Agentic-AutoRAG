"""Bayesian (Optuna TPE) baseline driver.

Drives an Optuna ``ask-and-tell`` loop so the async ``evaluate_trial`` integrates
naturally — pruned trials don't block subsequent suggestions, and the SQLite
storage backend lets killed runs resume by extending ``--max-trials``. The TPE
sampler's seed is set explicitly so a fresh sqlite gives reproducible
suggestions; we additionally pickle the sampler each trial so that on resume
the sampler internal state is restored (Optuna doesn't persist sampler state by
default — known gotcha).
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

from agentic_autorag.baselines.base import (
    BaselineMeta,
    save_best_config,
    write_optimizer_meta,
)
from agentic_autorag.baselines.sampler import sample_trial_config_optuna
from agentic_autorag.optimizer.history import TrialRecord
from agentic_autorag.orchestrator import Orchestrator

logger = logging.getLogger("agentic_autorag.run")

ALGORITHM_NAME = "bayesian"
_SAMPLER_PICKLE_NAME = "optuna_sampler.pkl"
_OPTUNA_DB_NAME = "optuna.db"
_OPTUNA_STUDY_NAME = "agentic_rag_bayesian"


async def run_bayesian_search(
    config_path: str,
    output_dir: str,
    seed: int = 42,
    max_trials: int | None = None,
) -> TrialRecord | None:
    """Drive an Optuna TPE baseline.

    Resumability: passing the same ``output_dir`` again with a higher
    ``max_trials`` resumes the study from the existing sqlite. Pickled sampler
    state is reloaded so the suggestion sequence after resume matches an
    uninterrupted run.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_dir = Path(output_dir)
    orch = Orchestrator(config_path, output_dir_override=str(out_dir))
    if max_trials is None:
        max_trials = orch.config.meta.max_trials

    history = orch.history
    history.clear()

    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / _OPTUNA_DB_NAME
    sampler_path = out_dir / _SAMPLER_PICKLE_NAME

    if sampler_path.exists():
        try:
            sampler = pickle.loads(sampler_path.read_bytes())
            logger.info("Resumed Optuna sampler state from %s", sampler_path.name)
        except Exception:
            logger.warning("Failed to load pickled sampler; starting fresh", exc_info=True)
            sampler = optuna.samplers.TPESampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=f"sqlite:///{db_path}",
        study_name=_OPTUNA_STUDY_NAME,
        load_if_exists=True,
    )

    meta = BaselineMeta(algorithm=ALGORITHM_NAME, seed=seed, max_trials=max_trials)
    meta.extras["sqlite_db"] = db_path.name

    logger.info("=" * 60)
    logger.info("Bayesian (Optuna TPE) baseline | seed=%d | max_trials=%d", seed, max_trials)
    logger.info("=" * 60)

    t_start = time.monotonic()
    try:
        await orch.setup()
        for trial_num in range(1, max_trials + 1):
            logger.info("=" * 60)
            logger.info("BAYESIAN TRIAL %d/%d", trial_num, max_trials)
            logger.info("=" * 60)

            trial = study.ask()
            try:
                trial_config = sample_trial_config_optuna(
                    trial, orch.config.search_space, orch.config.embedding_token_limits
                )
            except optuna.TrialPruned as exc:
                logger.warning("Optuna pruned trial %d before evaluation: %s", trial_num, exc)
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                meta.n_pruned += 1
                continue

            violations = orch.config.validate_trial(trial_config)
            if violations:
                logger.warning(
                    "Optuna sample failed validation (trial %d): %s — pruning",
                    trial_num,
                    "; ".join(violations),
                )
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                meta.n_validation_rejects += 1
                continue

            orch._log_config_summary("Config", trial_config)

            try:
                result = await orch.evaluate_trial(trial_config)
            except Exception:
                logger.exception("Trial %d evaluation failed; marking failed in study", trial_num)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                continue

            study.tell(trial, result.score)
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

            # Persist sampler so a kill/resume after this trial reproduces the
            # same future suggestions an uninterrupted run would have made.
            try:
                sampler_path.write_bytes(pickle.dumps(study.sampler))
            except Exception:
                logger.warning("Failed to persist Optuna sampler", exc_info=True)

            best_far = history.get_best()
            logger.info(
                "Bayesian trial %d done | score=%.3f | best so far=%.3f",
                trial_num,
                result.score,
                best_far.score if best_far else result.score,
            )

        best = history.get_best()
        save_best_config(out_dir, best, include_graph=orch.config.uses_graph())
    finally:
        meta.wall_clock_s = round(time.monotonic() - t_start, 3)
        write_optimizer_meta(out_dir, meta)
        await orch.cleanup()

    if best:
        logger.info("Bayesian search complete | best score=%.3f | %.1fs total", best.score, meta.wall_clock_s)
    else:
        logger.warning("Bayesian search produced no successful trials in %.1fs", meta.wall_clock_s)
    return best
