<p align="center">
  <img src="https://raw.githubusercontent.com/lassebaerlandstrand/Agentic-AutoRAG/HEAD/assets/Agentic_AutoRAG.png" alt="Agentic AutoRAG">
</p>

# Agentic AutoRAG

Agentic AutoRAG is a reasoning-driven optimizer for Retrieval-Augmented
Generation (RAG) pipelines. Instead of grid search or Bayesian optimization,
it runs a two-stage LLM agent loop: a diagnoser analyses *why* a trial
configuration fails, and a proposer chooses *what to change* next based on
that diagnosis and the history of prior trials. The optimization signal is a
synthetic exam, typed open-ended questions plus ground-truth answers
generated from your corpus on the first run and cached for reuse. Retrieval
is database-agnostic: vector, hybrid BM25+vector, graph, or hybrid
graph-vector.

## How it works

Each run proceeds in three phases:

1. **Exam generation.** On the first run the examiner LLM reads your corpus
   and writes a synthetic exam — typed open-ended questions (extraction,
   definitional, inference, multi-hop bridge, comparison, numeric) paired with
   ground-truth answers grounded in specific source spans. Each question is
   validated (answer spans verified, an oracle gate confirms it's answerable)
   and the most discriminating items are kept. The exam is cached to
   `exam.json` and reused on later runs.
2. **Reasoning loop.** For each trial the pipeline is built and evaluated
   against the exam. A **diagnoser** analyses *why* the configuration scored as
   it did (retrieval misses vs. generation errors, which question types
   failed); a **proposer** then chooses *what to change* next, informed by the
   diagnosis and the full history of prior trials — not a grid or Bayesian
   sweep.
3. **Selection.** Trials are scored on two axes, answer quality and LLM cost
   per query; the non-dominated ones form a **Pareto frontier**, and the
   `--objective` policy picks the single `recommended.yaml` from it.

## Setup

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # runtime dependencies
uv sync --extra dev      # add tests, lint, and vLLM
```

Copy `.env.example` to `.env` and fill in API keys for the providers you
plan to use:

```bash
cp .env.example .env
```

Required keys are validated at startup based on the models you configure:

| Provider prefix   | Env vars                                                          |
| ----------------- | ----------------------------------------------------------------- |
| `openai/...`      | `OPENAI_API_KEY`                                                  |
| `anthropic/...`   | `ANTHROPIC_API_KEY`                                               |
| `azure/...`       | `AZURE_API_KEY`, `AZURE_API_BASE`                                 |
| `azure_ai/...`    | `AZURE_AI_API_KEY`, `AZURE_AI_API_BASE`                           |
| `gemini/...`      | `GEMINI_API_KEY`                                                  |
| `vertex_ai/...`   | `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`                           |
| `bedrock/...`     | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME`   |
| `mistral/...`     | `MISTRAL_API_KEY`                                                 |
| `ollama/...`      | none — start `ollama serve` and `ollama pull` each model          |
| `hosted_vllm/...` | none — vLLM is auto-managed (install via `uv sync --extra dev`)   |

Azure note: `AZURE_API_BASE` is `https://<resource>.cognitiveservices.azure.com/`
or `https://<resource>.openai.azure.com/`. `AZURE_AI_API_BASE` is
`https://<resource>.services.ai.azure.com/models`. If Azure returns one
shared key, use it for both `AZURE_API_KEY` and `AZURE_AI_API_KEY`.

Sanity-check your environment:

```bash
uv run agentic-autorag info
```

## Corpus

Point `meta.corpus_path` at a directory of documents. Supported formats:
PDF, DOCX, XLSX, PPTX, HTML, CSV, Markdown, plain text, AsciiDoc, and
images (PNG/JPG/TIFF/BMP/WEBP, OCR'd). Subdirectories are walked
recursively; one file per source document.

The example configs use `./data/corpus/unidoc/`; download it with:

```bash
uv run python scripts/download_unidoc_corpus.py
```

## Configure

Start from `configs/starter_example.yaml`, copy to a new file, and edit. The
fields you'll most often change:

- `meta.corpus_path`, `meta.project_name`, `meta.output_dir`
- `meta.max_trials` — optimization budget
- `agent.optimizer_model`, `agent.examiner_model`, `agent.judge_model`
- `search_space.embedding.models`
- `search_space.generator.models`

See `configs/full_example.yaml` for every supported field with its package
default, and `agentic_autorag/config/models.py` for the Pydantic schema (the
source of truth — invalid configs fail at parse time with a clear error).

## Bring your own exam

By default the exam is generated from your corpus. If you already have questions
with known answers, optimize against those instead — point the examiner at a
JSON file and generation is skipped:

```yaml
examiner:
  custom_exam_path: path/to/exam.json
```

The file is a list of question records; attaching supporting-document ids (and,
optionally, verbatim evidence spans) unlocks retrieval-level diagnostics. If your
questions carry document ids but no spans, `ground-exam` adds and verifies the
spans for you. See [docs/custom_exam.md](docs/custom_exam.md) for the format,
the three grounding tiers, and examples.

## Run

```bash
uv run agentic-autorag optimize --config configs/starter_example.yaml
```

To start over on a clean output directory:

```bash
uv run agentic-autorag clean --config configs/starter_example.yaml
```

## Outputs

Everything is written under `meta.output_dir`:

- `optimization_summary.md` — the run report: the recommended config and why,
  what the search found, and (cost-aware) the Pareto frontier table, score-vs-cost
  chart, tradeoffs, and per-config YAML. The prose is written by the optimizer
  model (disable with `--skip-final-report`); the tables are deterministic.
- `recommended.yaml` — the recommended pipeline configuration.
- `frontier/` — alternative configurations on the Pareto frontier, one
  YAML per frontier point.
- `frontier.json` — machine-readable frontier index.
- `history.jsonl` — one record per trial (config, scores, diagnosis, proposal).
- `cost_breakdown.json` — LLM spend per category (e.g. `rag_eval`,
  `exam_generation`, `judge`, `agent_proposal`, `final_report`).
- `exam.json` — the synthetic exam used for evaluation (generated on the
  first run, reused on subsequent runs).
- `run.log` — full run log, including every agent prompt.

## After optimization

`recommended.yaml` is a complete, ready-to-use pipeline config. To score it
(or any `frontier/trial_NN.yaml`) against held-out QA:

```bash
uv run agentic-autorag benchmark-evaluate \
  --project-config configs/starter_example.yaml \
  --trial-config experiments/unidoc-starter/recommended.yaml \
  --qa path/to/qa.json \
  --output results.json \
  --judge-model gemini/gemini-3-flash-preview
```

## Troubleshooting

- **Missing API key at startup.** Only the keys for models in your config are
  validated; the error names the missing variable — add it to `.env`.
- **An endpoint started working after a failed run.** Endpoint checks are
  cached — re-run with `--force-verify` to re-ping every model.
- **`.cache/` is large.** It holds the parsed corpus, embeddings, and exam so
  reruns are fast. `agentic-autorag clean` removes it along with the run
  artifacts.
- **Re-running** reuses the cached corpus and `exam.json` but starts the trial
  history fresh; use `clean` for a fully cold run.

## License

MIT — see [LICENSE](LICENSE).
