# Agentic AutoRAG

Reasoning-driven framework for optimizing RAG pipelines with an LLM agent.

## Overview

Agentic AutoRAG runs an optimization loop:

1. Parse and chunk a corpus.
2. Build or load an index (with optional structural caching).
3. Generate and run MCQ-based evaluation.
4. Use an optimizer agent to diagnose failures and propose the next configuration.

## Prerequisites

- Python 3.12+
- `uv` package manager
- Ollama (for local `ollama/...` generation models)

Install dependencies:

```bash
uv sync
```

Install development dependencies:

```bash
uv sync --extra dev
```

## Setup

### 1) Configure the corpus

Set `meta.corpus_path` in your YAML config to your document directory.

Optional: download the ArXiv development corpus:

```bash
uv run python scripts/download_arxiv_corpus.py
```

### 2) Configure models and providers

- Generation search space: `runtime.generation.llm_models`
- Optimizer agent model: `agent.optimizer_model`
- Examiner agent model: `agent.examiner_model`

If you use cloud models via LiteLLM, export provider API keys before running (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or other provider-specific keys required by your selected model strings).

### 3) Install and run Ollama (required for `ollama/...` models)

If your selected config includes any `ollama/...` models, you must run Ollama and pull those exact models before optimization.

Install Ollama:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Start Ollama service:

```bash
ollama serve
```

In another terminal, pull every model referenced by your config. Example for `configs/full.yaml`:

```bash
ollama pull llama3.2
ollama pull llama3.1:8b
ollama pull mistral
ollama pull gemma2:9b
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b
ollama pull phi4
```

You only need to pull models that actually appear in the config you run.

Verify environment/tooling:

```bash
uv run agentic-autorag info
```

## Configuration files

- `configs/starter.yaml`: minimal search space for fast iteration.
- `configs/full.yaml`: broader search space for longer optimization runs.

Important config fields:

- `meta.corpus_path`: input documents.
- `meta.output_dir`: run artifacts (`history.jsonl`, `exam.json`, `run.log`, `best_config.yaml`).
- `meta.max_trials`: optimization budget.
- `meta.index_registry`:
  - `true`: cache and reuse structural indices by fingerprint.
  - `false`: rebuild structural index on every structural change.

## Run

Start with the starter configuration:

```bash
uv run agentic-autorag optimize --config configs/starter.yaml
```

Run broader optimization with the full search space:

```bash
uv run agentic-autorag optimize --config configs/full.yaml
```

## Outputs

Artifacts are written to `meta.output_dir`:

- `best_config.yaml`
- `history.jsonl`
- `run.log`
- `exam.json`

## Developer workflow

Run tests:

```bash
uv run pytest -q
```

Run lint and format:

```bash
uv run ruff check .
uv run ruff format .
```

Run a single trial during development:

```bash
uv run python scripts/run_single_trial.py --config configs/starter.yaml
```

Run the reasoning agent in isolation:

```bash
uv run python scripts/run_agent_once.py --config configs/starter.yaml
```

## Project structure

- `agentic_autorag/` core package
  - `engine/` indexing, retrieval, and pipeline logic
  - `examiner/` MCQ generation, evaluation, and IRT
  - `optimizer/` reasoning agent and trial history
  - `registry/` index caching
  - `config/` YAML models and loader
- `configs/` sample search-space configurations
- `scripts/` helper scripts
- `tests/` test suite
