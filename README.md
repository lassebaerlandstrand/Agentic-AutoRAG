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

```bash
uv sync --extra vllm
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

Agentic AutoRAG can be evaluated end-to-end against standard RAG benchmarks.
The design keeps the reasoning-agent optimization signal (internal MCQ exam)
unchanged — the benchmark's own QA pairs are held out and used only once, after
optimization, to score the winning config. Other frameworks (AutoRAG, Bayesian
search, grid/random search) optimize on the same corpus with the same search
space and their best configs go through the same evaluator for fair comparison.

Three-step workflow (HotpotQA distractor; NQ / MuSiQue / CRAG are future work):

```bash
# 1. Download + materialise corpus/ + qa.json + metadata.json
uv run agentic-autorag benchmark-prepare hotpot_qa \
    --split validation --sample-size 500 --seed 42 \
    --output benchmark_data/hotpot_val_500

# 2. Optimize. The prepared corpus is a directory of plain markdown so the
#    normal `optimize` loop consumes it unchanged.
uv run agentic-autorag optimize --config configs/hotpot_qa.yaml
#    Writes: experiments/hotpot/best_config.yaml

# 3. Score the best config against the held-out QA.
uv run agentic-autorag benchmark-evaluate \
    --project-config configs/hotpot_qa.yaml \
    --trial-config experiments/hotpot/best_config.yaml \
    --qa benchmark_data/hotpot_val_500/qa.json \
    --output experiments/hotpot/benchmark_results.json \
    --judge-model gemini/gemini-2.5-flash-lite
```

`benchmark-evaluate` writes EM, token-F1 (SQuAD/HotpotQA canonical), LLM-judge
accuracy (when `--judge-model` is set), Recall@1/2/5/10, and MRR. Per-question
records and `trial_config_hash` / `project_config_hash` / `corpus_hash` /
`benchmark_manifest.hf_revision` are included for reproducibility.

**Run in the same `meta.output_dir` as the preceding `optimize` run** so the
ingredient cache (chunks + embeddings) is reused — evaluation otherwise
rebuilds the index from scratch.

Note on the internal MCQ signal: `examiner.exam_size` (in the config) controls
how many MCQs the framework generates from the corpus for optimization. It is
independent from the benchmark `--sample-size` flag, which only controls how
many held-out QA pairs are prepared for the final evaluation.

### Baselines

For the EMNLP comparison table we run four competing baselines alongside the
reasoning-agent. Each one searches the **same** `TrialConfig` configuration
space defined in `configs/hotpot_qa.yaml`, produces a `best_config.yaml`, and
is then scored by the same `benchmark-evaluate` on the held-out HotpotQA QA.
That yields one row per `(method, seed)` pair on identical inference code:

| Row | Method | Proposal strategy | Optimization signal |
|---|---|---|---|
| 1 | Agentic AutoRAG (ours) | LLM reasoning agent | MCQ exam |
| 2 | Random search | Uniform sampling | MCQ exam |
| 3 | Bayesian (Optuna TPE) | TPE | MCQ exam |
| 4 | Marker-Inc AutoRAG (RAGAS) | Greedy node-by-node | AutoRAG's own RAGAS-style bootstrap QA |
| 5 | Marker-Inc AutoRAG (MCQ ablation) | Greedy node-by-node | Our MCQ exam, scored via custom `mcq_accuracy` metric |

Rows 4 and 5 use Marker-Inc AutoRAG's optimizer with two different exam
sources to isolate algorithm-vs-exam effects. Rows 2-5 share AutoRAG's strict
search-space mirror (auto-generated from our `SearchSpace`) so no row gets
extra knobs the others don't have.

Install the optional baseline dependencies (Optuna, pyarrow, pandas):

```bash
uv sync --extra baselines
```

`AutoRAG` itself is intentionally NOT in the extra: AutoRAG 0.3.x pins
`numpy<2`, which conflicts with our base deps (`opencv-python-headless>=4.13`
needs `numpy>=2`). Install AutoRAG into a separate venv and pass the path to
its Python interpreter when running the AutoRAG baselines:

```bash
python -m venv .autorag-venv
.autorag-venv/bin/pip install AutoRAG
export AUTORAG_PYTHON=$(pwd)/.autorag-venv/bin/python
```

Five-row workflow (paper-mode, sequential — vLLM is exclusive):

```bash
# Agentic AutoRAG (ours)
uv run agentic-autorag optimize --config configs/hotpot_qa.yaml
#  → experiments/hotpot/best_config.yaml + history.jsonl + exam.json

# Random + Bayesian (3 seeds each)
for SEED in 1 2 3; do
  uv run agentic-autorag baseline-optimize \
      --algorithm random --config configs/hotpot_qa.yaml \
      --output-dir experiments/hotpot_random/seed_$SEED \
      --seed $SEED
  uv run agentic-autorag baseline-optimize \
      --algorithm bayesian --config configs/hotpot_qa.yaml \
      --output-dir experiments/hotpot_bayesian/seed_$SEED \
      --seed $SEED
done

# AutoRAG-RAGAS (native QA bootstrap)
uv run agentic-autorag baseline-optimize \
    --algorithm autorag_ragas --config configs/hotpot_qa.yaml \
    --output-dir experiments/hotpot_autorag_ragas

# AutoRAG-MCQ (ablation: AutoRAG's algorithm with our MCQ exam)
uv run agentic-autorag baseline-optimize \
    --algorithm autorag_mcq --config configs/hotpot_qa.yaml \
    --output-dir experiments/hotpot_autorag_mcq

# Score every winning config on the held-out HotpotQA QA
for d in experiments/hotpot \
         experiments/hotpot_random/seed_{1,2,3} \
         experiments/hotpot_bayesian/seed_{1,2,3} \
         experiments/hotpot_autorag_ragas \
         experiments/hotpot_autorag_mcq; do
  [ -f "$d/best_config.yaml" ] || continue
  uv run agentic-autorag benchmark-evaluate \
      --project-config configs/hotpot_qa.yaml \
      --trial-config "$d/best_config.yaml" \
      --qa benchmark_data/hotpot_val_1000/qa.json \
      --output "$d/benchmark_results.json" \
      --judge-model gemini/gemini-2.5-flash-lite
done
```

**Cache reuse across baselines.** Every baseline reads `meta.output_dir` from
the YAML for the *shared* corpus parse, exam, ingredient cache, and graph
store; only per-run outputs (`best_config.yaml`, `history.jsonl`,
`optimizer_meta.json`) live under each `--output-dir`. The first baseline you
run generates `exam.json`; every subsequent run reuses it (hours saved per
run). The AutoRAG-MCQ ablation specifically requires this — it converts the
cached `exam.json` to AutoRAG's `qa.parquet` schema as a pure transform with
no LLM calls.

**AutoRAG search-space mirroring (deliberate).** The driver auto-generates
AutoRAG's `config.yaml` strictly from our `SearchSpace`. Knobs AutoRAG ships
that we don't tune (`passage_compressor`, `prompt_maker` template tuning,
LongLLMLingua, additional `query_expansion` modules) are explicitly excluded,
so AutoRAG's algorithm doesn't search dimensions the others can't. Recorded
in `translation_notes.json` next to each baseline's outputs.

**AutoRAG asymmetries to footnote in the paper:** (a) MCQ vs. free-form QA
in the RAGAS variant; (b) AutoRAG ships only single-hop QA sampling — its
RAGAS-bootstrap exam under-represents multi-hop and biases AutoRAG-RAGAS
*against* HotpotQA; (c) no parametric-leak filter in AutoRAG's bootstrap.

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
  - `baselines/` Random / Bayesian / Marker-Inc AutoRAG comparison drivers
  - `config/` Pydantic models and YAML loader
- `configs/` search-space configurations (`full.yaml`, `hotpot_qa.yaml`)
- `knowledge_base/` pre-computed model rankings auto-loaded by the optimizer
- `scripts/` corpus downloaders and one-off helpers
- `tests/` test suite
