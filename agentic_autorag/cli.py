"""Typer CLI entry point for Agentic AutoRAG."""

from __future__ import annotations

import asyncio
import logging
import platform
import shutil
from pathlib import Path

import typer

from agentic_autorag.litellm_runtime import configure_litellm_runtime

app = typer.Typer(name="agentic-autorag", help="Agentic AutoRAG Optimizer")


@app.command()
def optimize(
    config: str = typer.Option("configs/starter.yaml", help="Path to YAML config"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose/debug logging"),
    debug_prompts: bool = typer.Option(
        False, "--debug-prompts", help="Log optimizer agent prompts and responses to run.log"
    ),
    debug_eval_samples: int = typer.Option(
        0,
        "--debug-eval-samples",
        help="Log N sampled question/retrieved-context/RAG-answer triples per trial to run.log "
        "(0 disables). Useful for diagnosing whether high accuracy reflects easy questions vs. "
        "real RAG quality.",
    ),
) -> None:
    """Run the optimization loop."""
    configure_litellm_runtime()

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(name)s: %(message)s",
    )
    from agentic_autorag.orchestrator import Orchestrator  # triggers lightrag import

    if not verbose:
        # LightRAG resets its own logger to INFO at module-load, so we silence it
        # AFTER the import above — otherwise our setLevel gets clobbered.
        logging.getLogger("lightrag").setLevel(logging.WARNING)

    orchestrator = Orchestrator(config, debug_prompts=debug_prompts, debug_eval_samples=debug_eval_samples)
    asyncio.run(orchestrator.run())


@app.command()
def info() -> None:
    """Print system info and check dependencies."""
    print(f"Python:   {platform.python_version()}")
    print(f"Platform: {platform.system()} {platform.machine()}")

    # Ollama
    ollama_path = shutil.which("ollama")
    if ollama_path:
        print(f"Ollama:   ✓ found at {ollama_path}")
    else:
        print("Ollama:   ✗ not found on PATH")

    # vLLM
    vllm_path = shutil.which("vllm")
    if vllm_path:
        print(f"vLLM:     ✓ found at {vllm_path}")
    else:
        print("vLLM:     ✗ not found on PATH (install: uv sync --extra vllm)")

    # Key packages
    for pkg in ("lancedb", "litellm", "sentence_transformers", "pydantic", "typer"):
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"{pkg:20s} ✓  {version}")
        except ImportError:
            print(f"{pkg:20s} ✗  not installed")


@app.command("benchmark-prepare")
def benchmark_prepare(
    name: str = typer.Argument(..., help="Benchmark name (e.g. 'hotpot_qa')"),
    output: str = typer.Option(..., "--output", "-o", help="Output directory for corpus + qa.json"),
    split: str = typer.Option("validation", help="Dataset split"),
    sample_size: int | None = typer.Option(500, "--sample-size", help="Deterministic sample size; omit for full split"),
    seed: int = typer.Option(42, help="Sampling seed"),
    hf_revision: str | None = typer.Option(None, "--hf-revision", help="Pin HuggingFace dataset revision"),
) -> None:
    """Download a benchmark and materialise it as corpus/ + qa.json + metadata.json."""
    from agentic_autorag.benchmarks import prepare as prepare_benchmark

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    manifest = prepare_benchmark(
        name=name,
        output_dir=Path(output),
        split=split,
        sample_size=sample_size,
        seed=seed,
        hf_revision=hf_revision,
    )
    print(f"Prepared {manifest.name} ({manifest.split}, n={manifest.sample_size})")
    print(f"  corpus:   {Path(output) / 'corpus'}  ({manifest.corpus_doc_count} docs)")
    print(f"  qa:       {Path(output) / 'qa.json'}")
    print(f"  metadata: {Path(output) / 'metadata.json'}  (hf_revision={manifest.hf_revision})")


@app.command("benchmark-evaluate")
def benchmark_evaluate(
    project_config: str = typer.Option(..., "--project-config", help="Path to the project YAML used for optimize"),
    trial_config: str = typer.Option(..., "--trial-config", help="Path to the best_config.yaml produced by optimize"),
    qa: str = typer.Option(..., "--qa", help="Path to benchmark qa.json"),
    output: str = typer.Option(..., "--output", "-o", help="Destination for benchmark_results.json"),
    judge_model: str | None = typer.Option(
        None, "--judge-model", help="LiteLLM model string for LLM-as-judge; omit to skip"
    ),
    concurrency: int = typer.Option(10, help="Concurrent questions under evaluation"),
    limit: int | None = typer.Option(
        None, "--limit", "-n", help="Evaluate only the first N questions; omit to use all"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Score a (project_config, trial_config) pair against held-out benchmark QA."""
    configure_litellm_runtime()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(name)s: %(message)s",
    )
    # Chatty libraries — we want their warnings, not their per-call INFO lines.
    if not verbose:
        for noisy in ("LiteLLM", "litellm", "sentence_transformers", "httpx"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    # User-facing progress goes through agentic_autorag.run — same dedicated
    # logger the orchestrator uses for `optimize`, with a bare %(message)s format
    # and propagate=False so root-handler prefixes don't leak in.
    run_logger = logging.getLogger("agentic_autorag.run")
    run_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    run_logger.propagate = False
    for handler in list(run_logger.handlers):
        run_logger.removeHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    run_logger.addHandler(console)

    from agentic_autorag.benchmark_eval.runner import run as run_eval

    if limit is not None and limit <= 0:
        raise typer.BadParameter("--limit must be a positive integer", param_hint="--limit")

    asyncio.run(
        run_eval(
            project_config_path=project_config,
            trial_config_path=trial_config,
            qa_path=qa,
            output_path=output,
            judge_model=judge_model,
            concurrency=concurrency,
            limit=limit,
        )
    )


_BASELINE_ALGORITHMS = ("random", "bayesian", "autorag_ragas", "autorag_mcq")


def _setup_run_logger(verbose: bool) -> None:
    """Wire user-facing progress through ``agentic_autorag.run`` (matches optimize/benchmark-evaluate)."""
    run_logger = logging.getLogger("agentic_autorag.run")
    run_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    run_logger.propagate = False
    for handler in list(run_logger.handlers):
        run_logger.removeHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    run_logger.addHandler(console)


@app.command("baseline-optimize")
def baseline_optimize(
    algorithm: str = typer.Option(
        ...,
        "--algorithm",
        "-a",
        help=f"Baseline algorithm to run. One of: {', '.join(_BASELINE_ALGORITHMS)}",
    ),
    config: str = typer.Option(..., "--config", "-c", help="Path to project YAML"),
    output_dir: str = typer.Option(
        ..., "--output-dir", "-o", help="Per-run output directory (best_config.yaml + history.jsonl + meta)"
    ),
    seed: int = typer.Option(42, "--seed", help="Single seed (use --seeds for multi-seed paper runs)"),
    seeds: str | None = typer.Option(
        None,
        "--seeds",
        help="Comma-separated seed list, e.g. '1,2,3'. When set, the driver runs once per seed "
        "into <output-dir>/seed_<n>/. Mutually exclusive with --seed for the random/bayesian "
        "baselines; both autorag variants are deterministic and ignore seeds.",
    ),
    max_trials: int | None = typer.Option(
        None, "--max-trials", help="Trial budget; defaults to meta.max_trials in the YAML"
    ),
    autorag_python: str | None = typer.Option(
        None,
        "--autorag-python",
        help="Path to a Python interpreter where AutoRAG is installed. Required for autorag_*. "
        "Falls back to AUTORAG_PYTHON env var.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a baseline optimizer (Random / Bayesian / AutoRAG-RAGAS / AutoRAG-MCQ).

    Each baseline reuses the project's shared cache_dir (``meta.output_dir``) for
    corpus parsing, exam generation, ingredient cache and graph store. Per-run
    outputs (``best_config.yaml``, ``history.jsonl``, ``optimizer_meta.json``)
    land in ``--output-dir``. Run multiple baselines against the same YAML to
    keep the cache warm across runs.
    """
    if algorithm not in _BASELINE_ALGORITHMS:
        raise typer.BadParameter(f"--algorithm must be one of: {', '.join(_BASELINE_ALGORITHMS)} (got {algorithm!r})")

    configure_litellm_runtime()

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(name)s: %(message)s",
    )
    if not verbose:
        for noisy in ("LiteLLM", "litellm", "sentence_transformers", "httpx"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    _setup_run_logger(verbose)

    seed_list = [int(s.strip()) for s in seeds.split(",")] if seeds else [seed]

    if algorithm.startswith("autorag_") and len(seed_list) > 1:
        logging.getLogger("agentic_autorag.run").warning(
            "%s is deterministic given identical input — running once and ignoring extra seeds %s",
            algorithm,
            seed_list[1:],
        )
        seed_list = seed_list[:1]

    for s in seed_list:
        target_dir = Path(output_dir) / f"seed_{s}" if len(seed_list) > 1 else Path(output_dir)
        if algorithm == "random":
            from agentic_autorag.baselines.random_search import run_random_search

            asyncio.run(
                run_random_search(
                    config_path=config,
                    output_dir=str(target_dir),
                    seed=s,
                    max_trials=max_trials,
                )
            )
        elif algorithm == "bayesian":
            try:
                import optuna  # noqa: F401
            except ImportError as exc:
                raise typer.BadParameter("Optuna is not installed. Install via: uv sync --extra baselines") from exc
            from agentic_autorag.baselines.bayesian import run_bayesian_search

            asyncio.run(
                run_bayesian_search(
                    config_path=config,
                    output_dir=str(target_dir),
                    seed=s,
                    max_trials=max_trials,
                )
            )
        elif algorithm in {"autorag_ragas", "autorag_mcq"}:
            from agentic_autorag.baselines.autorag.driver import run_autorag_baseline

            qa_variant = "ragas" if algorithm == "autorag_ragas" else "mcq"
            asyncio.run(
                run_autorag_baseline(
                    config_path=config,
                    output_dir=str(target_dir),
                    qa_variant=qa_variant,
                    autorag_python=autorag_python,
                )
            )


@app.command()
def clean(
    config: str = typer.Option("configs/starter.yaml", help="Path to YAML config"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Remove all generated artifacts for a fresh optimization run.

    Deletes the corpus cache, exam cache, ingredient cache, LanceDB working
    dir, history, exam, logs, and best config from the output directory.
    """
    from agentic_autorag.config.loader import load_config

    search_space = load_config(config)
    output_dir = Path(search_space.meta.output_dir)

    if not output_dir.exists():
        print(f"Nothing to clean — output directory does not exist: {output_dir}")
        raise typer.Exit()

    targets = [
        (".cache", "Corpus + exam + ingredient cache"),
        ("history.jsonl", "Trial history"),
        ("exam.json", "Exam questions"),
        ("best_config.yaml", "Best config"),
        ("benchmark_results.json", "Benchmark results"),
        ("run.log", "Run log"),
    ]

    found = [(output_dir / name, label) for name, label in targets if (output_dir / name).exists()]

    if not found:
        print(f"Nothing to clean in {output_dir}")
        raise typer.Exit()

    print(f"Will delete from {output_dir}:")
    for path, label in found:
        print(f"  {label:20s}  {path.name}")

    if not yes:
        typer.confirm("Proceed?", abort=True)

    for path, label in found:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"  Removed {label}")

    print("Done — ready for a fresh run.")


if __name__ == "__main__":
    app()
