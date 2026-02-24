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
) -> None:
    """Run the optimization loop."""
    configure_litellm_runtime()

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(name)s: %(message)s",
    )
    from agentic_autorag.orchestrator import Orchestrator

    orchestrator = Orchestrator(config)
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

    # Key packages
    for pkg in ("lancedb", "litellm", "sentence_transformers", "pydantic", "typer"):
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"{pkg:20s} ✓  {version}")
        except ImportError:
            print(f"{pkg:20s} ✗  not installed")


@app.command()
def clean(
    config: str = typer.Option("configs/starter.yaml", help="Path to YAML config"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Remove all generated artifacts for a fresh optimization run.

    Deletes the corpus cache, exam cache, index registry, LanceDB indices,
    history, exam, logs, and best config from the output directory.
    """
    from agentic_autorag.config.loader import load_config

    search_space = load_config(config)
    output_dir = Path(search_space.meta.output_dir)

    if not output_dir.exists():
        print(f"Nothing to clean — output directory does not exist: {output_dir}")
        raise typer.Exit()

    targets = [
        (".cache", "Corpus + exam cache"),
        ("indices", "Index registry"),
        (".index_staging", "Index staging"),
        ("lancedb", "LanceDB data"),
        ("history.jsonl", "Trial history"),
        ("exam.json", "Exam questions"),
        ("best_config.yaml", "Best config"),
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
