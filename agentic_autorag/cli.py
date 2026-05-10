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
    objective: str = typer.Option(
        "max_score",
        "--objective",
        help="Selection policy applied to the Pareto frontier to pick recommended.yaml. "
        "One of: max_score | knee | cheapest_above:<score> | closest_to:<score>,<cost>. "
        "All policies operate on the frontier; alternative configs are always written to "
        "the frontier/ directory regardless.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Forwarded to the proposer LLM as ``seed=``. Providers that honour seed "
        "(OpenAI, Bedrock-Anthropic) become deterministic given identical inputs; "
        "others silently drop it via litellm.drop_params. Used by multi-seed "
        "benchmark runs to vary the agent across seeds.",
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

    orchestrator = Orchestrator(
        config,
        debug_prompts=debug_prompts,
        debug_eval_samples=debug_eval_samples,
        objective=objective,
        seed=seed,
    )
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
        print("vLLM:     ✗ not found on PATH (install: uv sync --extra dev)")

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
    trial_config: str = typer.Option(
        ..., "--trial-config", help="Path to a TrialConfig YAML (e.g. recommended.yaml or any frontier/trial_NN.yaml)"
    ),
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
        ("recommended.yaml", "Recommended config"),
        ("frontier", "Frontier configs directory"),
        ("frontier.json", "Frontier index"),
        ("frontier_report.md", "Frontier report"),
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
