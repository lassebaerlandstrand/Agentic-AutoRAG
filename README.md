# Agentic AutoRAG

Reasoning-driven agentic framework for RAG optimization.

## Getting Started

This project uses `uv` for dependency management.

### Installation

```bash
uv sync
```

### Essential Commands

#### Tests
Run all unit tests:
```bash
uv run pytest
```

Run tests and skip slow ones (e.g. PDF parsing):
```bash
uv run pytest -m "not slow"
```

#### Linting & Formatting
Check for lint errors:
```bash
uv run ruff check
```

Format code:
```bash
uv run ruff format
```

#### Data Preparation
Download the ArXiv development corpus (50 papers):
```bash
uv run python scripts/download_arxiv_corpus.py
```

Quick smoke test (1 paper per category):
```bash
uv run python scripts/download_arxiv_corpus.py --max-per-category 1 --output-dir data/corpus/smoke_test
```

## Project Structure

- `agentic_autorag/`: Core framework logic.
  - `engine/`: RAG pipeline, indexing, and parsing.
  - `examiner/`: MCQ generation and IRT analysis.
  - `config/`: YAML configuration and validation models.
- `scripts/`: Helper scripts for data and setup.
- `tests/`: Comprehensive test suite.
- `configs/`: Sample search space configurations.
