<p align="center">
  <img src="https://raw.githubusercontent.com/lassebaerlandstrand/Agentic-AutoRAG/HEAD/assets/Agentic_AutoRAG.png" alt="Agentic AutoRAG" width="500">
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
- `agent.optimizer_model`, `agent.examiner_model`
- `search_space.embedding.models`
- `search_space.generator.models`

See `configs/full_example.yaml` for every supported field with its package
default, and `agentic_autorag/config/models.py` for the Pydantic schema (the
source of truth — invalid configs fail at parse time with a clear error).

## Run

```bash
uv run agentic-autorag optimize --config configs/starter_example.yaml
uv run agentic-autorag optimize --config configs/starter_example.yaml --verbose
```

To start over on a clean output directory:

```bash
uv run agentic-autorag clean --config configs/starter_example.yaml
```

## Outputs

Everything is written under `meta.output_dir`:

- `recommended.yaml` — the recommended pipeline configuration.
- `frontier/` — alternative configurations on the Pareto frontier, one
  YAML per frontier point.
- `frontier.json`, `frontier_report.md` — frontier index and human-readable
  summary.
- `history.jsonl` — one record per trial (config, scores, diagnosis, proposal).
- `exam.json` — the synthetic exam used for evaluation (generated on the
  first run, reused on subsequent runs).
- `run.log` — full run log.

## License

MIT — see [LICENSE](LICENSE).
