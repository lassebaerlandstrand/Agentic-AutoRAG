"""Typer CLI entry point for Agentic AutoRAG."""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import shutil
from pathlib import Path
from typing import Annotated

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
) -> None:
    """Run the optimization loop."""
    configure_litellm_runtime()

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(name)s: %(message)s",
    )
    from agentic_autorag.orchestrator import Orchestrator

    orchestrator = Orchestrator(config, debug_prompts=debug_prompts)
    asyncio.run(orchestrator.run())


@app.command()
def validate(
    config: Annotated[str, typer.Option(help="Path to YAML config")] = "configs/full.yaml",
    candidates: Annotated[
        Path | None,
        typer.Option(
            "--candidates",
            help="Path to candidates JSON (defaults to {output_dir}/candidates.json)",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="Path to write validated exam JSON (defaults to {output_dir}/exam.json)",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose/debug logging")] = False,
) -> None:
    """Run only the validation pipeline from a candidates JSON file."""
    configure_litellm_runtime()

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(name)s: %(message)s",
    )

    from agentic_autorag.config.models import MCQQuestion
    from agentic_autorag.examiner.exam_validator import run_validation_pipeline
    from agentic_autorag.orchestrator import Orchestrator

    orchestrator = Orchestrator(config)
    output_dir = Path(orchestrator.config.meta.output_dir)

    candidates_path = candidates or (output_dir / "candidates.json")
    output_path = output or (output_dir / "exam.json")

    if not candidates_path.exists():
        raise typer.BadParameter(f"Candidates file not found: {candidates_path}")

    try:
        raw_candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
        candidate_questions = [MCQQuestion.model_validate(q) for q in raw_candidates]
    except Exception as exc:
        raise typer.BadParameter(f"Failed to parse candidates JSON: {exc}") from exc

    documents = orchestrator._load_and_parse_corpus()
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    doc_map = dict(zip(doc_ids, documents, strict=False))
    examiner = orchestrator.config.examiner
    embedder = orchestrator.index_builder.get_embedder(examiner.embedding_model)

    async def _run() -> list[MCQQuestion]:
        return await run_validation_pipeline(
            candidate_questions,
            documents=doc_map,
            embedder=embedder,
            model=orchestrator.config.agent.examiner_model,
            concurrency=orchestrator.config.agent.concurrency,
            source_fact_threshold=examiner.source_fact_threshold,
            detect_parametric_leaks=examiner.detect_parametric_leaks,
            source_fact_substring_fallback=examiner.source_fact_substring_fallback,
            source_fact_min_length=examiner.source_fact_min_length,
            source_fact_window_chunk_size=examiner.source_fact_window_chunk_size,
            source_fact_window_chunk_overlap=examiner.source_fact_window_chunk_overlap,
            parametric_leak_trials=examiner.parametric_leak_trials,
        )

    validated_questions = asyncio.run(_run())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([q.model_dump(mode="json") for q in validated_questions], indent=2),
        encoding="utf-8",
    )

    print(f"Validated {len(validated_questions)}/{len(candidate_questions)} candidates")
    print(f"Saved validated exam to {output_path}")


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
