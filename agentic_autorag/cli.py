"""Typer CLI entry point for Agentic AutoRAG."""

from __future__ import annotations

import asyncio
import platform
import shutil

import typer

app = typer.Typer(name="agentic-autorag", help="Agentic AutoRAG Optimizer")


@app.command()
def optimize(
    config: str = typer.Option("configs/starter.yaml", help="Path to YAML config"),
) -> None:
    """Run the optimization loop."""
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


if __name__ == "__main__":
    app()
