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
- Ollama (for local `ollama/...` models) or vLLM (for local `hosted_vllm/...` models) — both optional

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

You can mix local (Ollama) and cloud models in the same search space. The agent will explore both.

**Model configuration files:**

- `runtime.generation.llm_models` — search space for RAG pipeline LLMs
- `agent.optimizer_model` — reasoning optimizer agent
- `agent.examiner_model` — MCQ generation agent

**Provider API keys:**

Export the API keys for any cloud providers you include in your config:

| Model prefix    | Required env var(s)                                               | Where to get it                             |
| --------------- | ----------------------------------------------------------------- | ------------------------------------------- |
| `ollama/...`    | none                                                              | Run Ollama locally (see step 3)             |
| `hosted_vllm/...` | none (framework-managed)                                       | Install vLLM (see step 3b)                  |
| `gemini/...`    | `GEMINI_API_KEY`                                                  | https://aistudio.google.com/apikey          |
| `openai/...`    | `OPENAI_API_KEY`                                                  | https://platform.openai.com/api-keys        |
| `anthropic/...` | `ANTHROPIC_API_KEY`                                               | https://console.anthropic.com/settings/keys |
| `mistral/...`   | `MISTRAL_API_KEY`                                                 | https://console.mistral.ai/api-keys         |
| `vertex_ai/...` | `VERTEXAI_PROJECT` + `VERTEXAI_LOCATION`                          | https://console.cloud.google.com            |
| `bedrock/...`   | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_REGION_NAME` | https://console.aws.amazon.com/iam          |
| `azure/...`     | `AZURE_API_KEY` + `AZURE_API_BASE`                                | https://portal.azure.com                    |
| `azure_ai/...`  | `AZURE_AI_API_KEY` + `AZURE_AI_API_BASE`                          | https://ai.azure.com                        |

**Validation:** The framework checks at startup for missing API keys and will tell you exactly which ones are needed for your configured models.

### Azure endpoint quick guide

- `azure/...` uses `AZURE_API_BASE` with `https://<resource>.cognitiveservices.azure.com/` (or `https://<resource>.openai.azure.com/`).
- `azure_ai/...` uses `AZURE_AI_API_BASE` with `https://<resource>.services.ai.azure.com/models`.
- Do not use project endpoints like `https://<resource>.services.ai.azure.com/api/projects/<project-name>` as `AZURE_API_BASE` / `AZURE_AI_API_BASE` for LiteLLM model calls.
- If Azure returns one shared key, use the same value for both `AZURE_API_KEY` and `AZURE_AI_API_KEY`.

**Cloud provider model examples:**

You can mix providers freely in the same config. Just set the required
env vars in `.env` for each provider you use:

```yaml
# Use Vertex AI Gemini instead of AI Studio Gemini
generation:
  llm_models:
    - "vertex_ai/gemini-2.5-flash"

# Use Bedrock Claude (us. prefix for cross-region inference)
generation:
  llm_models:
    - "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0"

# Mix providers — the optimizer agent explores all of them
generation:
  llm_models:
    - "gemini/gemini-2.5-flash"
    - "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    - "azure/my-gpt4o-deployment"
```

`configs/full.yaml` includes commented examples for every supported provider.

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

### 3b) Install vLLM (for `hosted_vllm/...` models)

vLLM provides higher throughput than Ollama through continuous batching, making it faster when the optimizer runs many parallel evaluations.

vLLM is bundled into the `dev` extra; if you ran `uv sync --extra dev` you already have it. Otherwise install with:

```bash
uv sync --extra dev
```

Pre-download models for faster first startup:

```bash
uv run hf download Qwen/Qwen3-8B
uv run hf download Qwen/Qwen2.5-7B-Instruct-AWQ
```

Then list models in your config with the `hosted_vllm/` prefix. The framework manages the vLLM server automatically — starting, stopping, and swapping models between trials:

```yaml
llm_models:
  - "hosted_vllm/Qwen/Qwen3-8B"
  - "hosted_vllm/Qwen/Qwen2.5-7B-Instruct-AWQ"
```

For 16GB VRAM GPUs, use quantized models (AWQ/GPTQ). If you hit OOM errors, set `vllm.max_model_len` in your config to limit context length. For reasoning models, add the parser via `vllm.extra_args`:

```yaml
vllm:
  extra_args: ["--reasoning-parser", "qwen3"]
```

Verify environment/tooling:

```bash
uv run agentic-autorag info
```

## Configuration files

- `configs/full.yaml`: full search space with all supported options (chunking, embeddings, rerankers, cloud + local LLMs, optional graph retrieval). Use this for ordinary optimization runs.
- `configs/hotpot_qa.yaml`: search space tuned for the HotpotQA benchmark workflow described under [Public benchmarks](#public-benchmarks).

Important config fields:

- `meta.corpus_path`: input documents.
- `meta.output_dir`: run artifacts (`history.jsonl`, `exam.json`, `run.log`, `best_config.yaml`).
- `meta.max_trials`: optimization budget.
- `meta.index_registry`:
  - `true`: cache and reuse structural indices by fingerprint.
  - `false`: rebuild structural index on every structural change.

## Run

```bash
uv run agentic-autorag optimize --config configs/full.yaml
```

## Clean up

Remove all generated artifacts (corpus cache, index registry, history, exam, logs) for a fresh run:

```bash
uv run agentic-autorag clean --config configs/full.yaml
```

## Outputs

Artifacts are written to `meta.output_dir`:

- `best_config.yaml`
- `history.jsonl`
- `run.log`
- `exam.json`

## Public benchmarks

Agentic AutoRAG ships utilities to evaluate a winning config end-to-end against
standard RAG benchmarks. The reasoning-agent optimization signal (internal MCQ
exam) is unchanged — the benchmark's own QA pairs are held out and used only
once, after optimization, to score the winning config.

Three-step workflow (HotpotQA distractor; NQ / MuSiQue / CRAG are future work).
The shipped `configs/hotpot_qa.yaml` points at `benchmark_data/hotpot_val_2000`
and writes optimization outputs to `experiments-hotspot/`; adjust if you want
a different sample size:

```bash
# 1. Download + materialise corpus/ + qa.json + metadata.json
uv run agentic-autorag benchmark-prepare hotpot_qa \
    --split validation --sample-size 2000 --seed 42 \
    --output benchmark_data/hotpot_val_2000

# 2. Optimize. The prepared corpus is a directory of plain markdown so the
#    normal `optimize` loop consumes it unchanged.
uv run agentic-autorag optimize --config configs/hotpot_qa.yaml
#    Writes: experiments-hotspot/best_config.yaml

# 3. Score the best config against the held-out QA. Use --limit / -n to
#    evaluate only the first N questions during iteration; omit to score
#    the full set.
uv run agentic-autorag benchmark-evaluate \
    --project-config configs/hotpot_qa.yaml \
    --trial-config experiments-hotspot/best_config.yaml \
    --qa benchmark_data/hotpot_val_2000/qa.json \
    --output experiments-hotspot/benchmark_results.json \
    --judge-model "gemini/gemini-3-flash-preview" \
    --limit 500
```

`benchmark-evaluate` writes EM, token-F1 (SQuAD/HotpotQA canonical), LLM-judge
accuracy (when `--judge-model` is set), Recall@1/2/5/10, MRR, average
retrieval/generation latency, and total LLM cost (USD + prompt/completion
tokens). Per-question records and `trial_config_hash` / `project_config_hash`
/ `corpus_hash` / `benchmark_manifest.hf_revision` are included for
reproducibility.

**Run in the same `meta.output_dir` as the preceding `optimize` run** so the
ingredient cache (chunks + embeddings) is reused — evaluation otherwise
rebuilds the index from scratch.

Note on the internal MCQ signal: `examiner.exam_size` (in the config) controls
how many MCQs the framework generates from the corpus for optimization. It is
independent from the benchmark `--sample-size` flag, which only controls how
many held-out QA pairs are prepared for the final evaluation.

Multi-method benchmark comparisons (Random / Bayesian / AutoRAG vs. ours) live
in a sibling repo, `agentic-autorag-bench`, which depends on this framework.

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

## Project structure

- `agentic_autorag/` core package
  - `engine/` indexing, retrieval, and pipeline logic
  - `examiner/` exam generation, validation, and evaluation
  - `optimizer/` reasoning agent, trial history, and Pareto frontier
  - `benchmark_eval/` end-to-end scoring against held-out benchmark QA
  - `benchmarks/` benchmark adapters (HotpotQA today; NQ / MuSiQue / CRAG planned)
  - `config/` Pydantic models and YAML loader
- `configs/` search-space configurations (`full.yaml`, `hotpot_qa.yaml`)
- `knowledge_base/` pre-computed model rankings auto-loaded by the optimizer
- `scripts/` corpus downloaders and one-off helpers
- `tests/` test suite
