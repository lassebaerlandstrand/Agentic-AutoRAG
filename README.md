# Agentic AutoRAG

Reasoning-driven agentic framework for RAG optimization.

## Getting Started

This project uses `uv` for dependency management.

### Installation

```bash
uv sync
```

For development:

```bash
uv sync --extra dev
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

Run optimizer:

```bash
uv run python -m agentic_autorag optimize
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

## Development

### Run a single RAG trial

Parses the corpus, builds an index, generates an MCQ exam, and evaluates a pipeline:

```bash
uv run python scripts/run_single_trial.py --config configs/starter.yaml
```

### Test the reasoning agent

Tests the agent's diagnosis and proposal logic in isolation (uses mock exam data by default — no corpus or pipeline needed):

```bash
uv run python scripts/run_agent_once.py --config configs/starter.yaml
```

Optionally, feed it a real exam result from a previous trial:

```bash
# Step 1: run a trial and save the result
uv run python scripts/run_single_trial.py --config configs/starter.yaml --save-result experiments/exam_result.json

# Step 2: feed the real result to the agent
uv run python scripts/run_agent_once.py --config configs/starter.yaml --exam-result experiments/exam_result.json
```

## Project Structure

- `agentic_autorag/`: Core framework logic.
  - `engine/`: RAG pipeline, indexing, and parsing.
  - `examiner/`: MCQ generation and IRT analysis.
  - `optimizer/`: Reasoning agent and trial history.
  - `config/`: YAML configuration and validation models.
- `scripts/`: Helper scripts for data and setup.
- `tests/`: Comprehensive test suite.
- `configs/`: Sample search space configurations.
