# Cloud Provider Integration Plan — Agentic AutoRAG

## Context for coding agent

The Agentic AutoRAG codebase uses LiteLLM for all LLM calls. LiteLLM is provider-agnostic: it routes based on the model string prefix (e.g., `gemini/`, `ollama/`) and reads credentials from environment variables automatically. The codebase already works correctly with this pattern — no calls to `litellm.acompletion()` need to change.

**Goal**: Support Vertex AI (`vertex_ai/`), AWS Bedrock (`bedrock/`), Azure OpenAI (`azure/`), and Azure AI Foundry (`azure_ai/`) models by updating only the startup validation, env example, docs, and adding an example config. All provider auth is handled purely via `.env` — nothing provider-specific goes into YAML configs. The YAML configs just contain model strings like `vertex_ai/gemini-2.5-flash` or `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0`.

**Key principle**: LiteLLM reads env vars automatically based on the model prefix. The user sets env vars in `.env`, uses the right prefix in the YAML config, and everything just works. We do NOT add any provider-specific fields to the YAML config models.

---

## Files to change (4 total)

| # | File | Action | Why |
|---|------|--------|-----|
| 1 | `agentic_autorag/orchestrator.py` | Modify | Refactor `_PROVIDER_ENV_VARS` and `_check_api_keys()` to handle new providers with multi-variable auth |
| 2 | `.env.example` | Modify | Add sections for Vertex AI, Bedrock, Azure, Azure AI Foundry |
| 3 | `README.md` | Modify | Expand provider table, add cloud model YAML examples |
| 4 | `configs/cloud.yaml` | Create | New example config showing cloud provider usage |

## Files that must NOT change

These are already provider-agnostic. Do not touch them:

- `agentic_autorag/engine/pipeline.py` — calls `litellm.acompletion(model=self.config.llm_model, ...)`. LiteLLM handles all routing. No changes.
- `agentic_autorag/optimizer/reasoning_agent.py` — calls `litellm.acompletion(model=self.model, ...)`. No changes.
- `agentic_autorag/examiner/exam_agent.py` — calls `litellm.acompletion(model=self.examiner_model, ...)`. No changes.
- `agentic_autorag/config/models.py` — model strings are just `str` fields. The `SearchSpace.validate_trial()` method validates values against user-defined YAML lists, not against hardcoded provider names. No changes.
- `pyproject.toml` — `litellm` is already a dependency and handles provider-specific sub-dependencies (boto3 for Bedrock, google-cloud-aiplatform for Vertex, etc.) internally. No new deps needed.

---

## Step 1: Refactor `_PROVIDER_ENV_VARS` and `_check_api_keys()` in `orchestrator.py`

### 1a. Replace `_PROVIDER_ENV_VARS` dict

**Location**: `agentic_autorag/orchestrator.py`, around line 30.

**Current code**:
```python
_PROVIDER_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}
```

**Replace with**: A dict where each value is a list of alternative auth sets. Each inner list is a set of env vars that together form one valid authentication method. If ANY one set is fully present in the environment, the provider passes validation.

```python
# Provider prefix → list of alternative auth methods.
# Each inner list is a set of env vars that together satisfy auth.
# The provider passes if ANY one alternative is fully present.
# The first alternative is shown to the user in error messages.
_PROVIDER_ENV_VARS: dict[str, list[list[str]]] = {
    # Direct API key providers
    "gemini":     [["GEMINI_API_KEY"]],
    "openai":     [["OPENAI_API_KEY"]],
    "anthropic":  [["ANTHROPIC_API_KEY"]],
    "cohere":     [["COHERE_API_KEY"]],
    "mistral":    [["MISTRAL_API_KEY"]],
    # Cloud platform providers
    "vertex_ai":  [
        ["VERTEXAI_PROJECT", "VERTEXAI_LOCATION"],
        # Actual credentials come from GOOGLE_APPLICATION_CREDENTIALS
        # or ADC (gcloud auth application-default login) — both are
        # picked up by google-auth automatically, so we only require
        # project + location here.
    ],
    "bedrock":    [
        ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"],
        ["AWS_PROFILE", "AWS_REGION_NAME"],  # named profile (SSO/multi-account)
        ["AWS_REGION_NAME"],  # IAM role on EC2/ECS/Lambda (boto3 auto-discovers creds)
    ],
    "azure":      [["AZURE_API_KEY", "AZURE_API_BASE"]],
    "azure_ai":   [["AZURE_AI_API_KEY", "AZURE_AI_API_BASE"]],
}
```

**Why the Bedrock entry has 3 alternatives**: On EC2/ECS/Lambda with an IAM role attached, boto3 discovers credentials from instance metadata — the user only needs `AWS_REGION_NAME`. With SSO, they use `AWS_PROFILE`. Explicit keys are the third path. We don't want to force users on EC2 to set access keys they don't need.

**Why Vertex AI doesn't require `GOOGLE_APPLICATION_CREDENTIALS`**: The google-auth library has its own credential discovery chain (ADC). Developers who run `gcloud auth application-default login` don't need a JSON file at all. But they always need project + location.

### 1b. Rewrite `_check_api_keys()` function

**Location**: `agentic_autorag/orchestrator.py`, the `_check_api_keys` function (starts around line 40).

**Current code** iterates over models, extracts the provider prefix, looks up a single env var string, and checks `os.getenv(env_var)`.

**Replace the full function** with this logic:

```python
def _check_api_keys(search_space: SearchSpace) -> None:
    """Check that required API keys / env vars are set for all configured models.

    Each provider can have multiple alternative auth methods (e.g., Bedrock
    supports explicit keys, named profiles, or IAM roles). The check passes
    if ANY alternative is fully satisfied.

    Raises EnvironmentError with a clear message listing what's missing.
    """
    missing: list[tuple[str, list[str]]] = []

    # Collect every model string that needs a provider check
    models_to_check: list[str] = []
    models_to_check.extend(search_space.runtime.generation.llm_models)
    models_to_check.append(search_space.agent.optimizer_model)
    models_to_check.append(search_space.agent.examiner_model)

    # Deduplicate prefixes so we don't report the same provider twice
    checked_prefixes: set[str] = set()

    for model_str in models_to_check:
        if "/" not in model_str:
            continue
        provider_prefix = model_str.split("/")[0]

        # Skip local providers and already-checked prefixes
        if provider_prefix in ("ollama", "sentence-transformers"):
            continue
        if provider_prefix in checked_prefixes:
            continue
        if provider_prefix not in _PROVIDER_ENV_VARS:
            continue

        checked_prefixes.add(provider_prefix)
        auth_alternatives = _PROVIDER_ENV_VARS[provider_prefix]

        # Pass if ANY alternative auth set is fully present
        provider_ok = any(
            all(os.getenv(var) for var in required_set)
            for required_set in auth_alternatives
        )

        if not provider_ok:
            # Report the first (most common) auth method as the suggestion
            primary_set = auth_alternatives[0]
            missing_vars = [v for v in primary_set if not os.getenv(v)]
            missing.append((model_str, missing_vars))

    if missing:
        lines = ["Missing environment variables for configured models:"]
        for model_str, vars_list in missing:
            vars_str = ", ".join(vars_list)
            lines.append(f"  {model_str:<45} →  set {vars_str}")
        lines.append("")
        lines.append("See .env.example for all supported providers and auth methods.")
        raise EnvironmentError("\n".join(lines))
```

**Key behavioral changes**:
- The `missing` list now stores `tuple[str, list[str]]` instead of `tuple[str, str]` (multiple env vars per provider).
- Provider prefixes are deduplicated — if the user has 3 Bedrock models, we only check Bedrock auth once.
- The error message footer points to `.env.example`.
- The formatting uses 45-char left-alignment for the model string (wider than before, since `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0` is long).

### 1c. Update the type annotation for the `missing` variable in error formatting

The error message block at the end of the function now iterates over `list[str]` instead of a single string. Make sure the error raise uses `", ".join(vars_list)` as shown above.

### 1d. Do NOT change anything else in `orchestrator.py`

The `Orchestrator` class, `_load_and_parse_corpus()`, `_generate_exam()`, the optimization loop, and all other methods are provider-agnostic and must not be touched.

---

## Step 2: Update `.env.example`

**Location**: `.env.example` (project root).

**Replace entire file contents** with:

```bash
# Agentic AutoRAG — Environment Variables
#
# Uncomment and fill in the keys for the providers you use.
# LiteLLM reads these automatically from the environment.
# You only need to configure providers that appear in your YAML config.

# ─── Direct API providers ────────────────────────────────────────────

# Google AI Studio (model prefix: gemini/)
# Get key: https://aistudio.google.com/apikey
# GEMINI_API_KEY=your-gemini-api-key-here

# OpenAI (model prefix: openai/)
# Get key: https://platform.openai.com/api-keys
# OPENAI_API_KEY=your-openai-api-key-here

# Anthropic (model prefix: anthropic/)
# Get key: https://console.anthropic.com/settings/keys
# ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Mistral (model prefix: mistral/)
# Get key: https://console.mistral.ai/api-keys
# MISTRAL_API_KEY=your-mistral-api-key-here

# ─── Cloud platform providers ────────────────────────────────────────

# Google Vertex AI (model prefix: vertex_ai/)
# Supports Gemini + partner models (Claude, Llama, Mistral) via GCP.
# Auth: set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON,
#        or run `gcloud auth application-default login` for local dev.
# VERTEXAI_PROJECT=your-gcp-project-id
# VERTEXAI_LOCATION=us-central1
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# AWS Bedrock (model prefix: bedrock/)
# Supports Claude, Llama, Mistral, Amazon Nova, Cohere via AWS.
# On EC2/ECS/Lambda with an IAM role attached, only AWS_REGION_NAME is needed.
# AWS_ACCESS_KEY_ID=AKIA...
# AWS_SECRET_ACCESS_KEY=your-secret-key
# AWS_REGION_NAME=us-east-1

# Azure OpenAI (model prefix: azure/)
# Supports GPT-4o, o-series, GPT-5 via Azure OpenAI Service.
# Note: the model string uses your Azure deployment name, not the model name.
# AZURE_API_KEY=your-azure-api-key
# AZURE_API_BASE=https://your-resource.openai.azure.com/

# Azure AI Foundry (model prefix: azure_ai/)
# Supports non-OpenAI models (Mistral, Cohere, Llama) via Azure AI Foundry.
# AZURE_AI_API_KEY=your-azure-ai-key
# AZURE_AI_API_BASE=https://your-endpoint.inference.ai.azure.com/
```

---

## Step 3: Update `README.md`

### 3a. Replace the provider API keys table

**Location**: `README.md`, under the "Provider API keys" heading (the markdown table).

**Replace the existing table** with:

```markdown
| Model prefix       | Required env var(s)                                          | Where to get it                          |
|--------------------|--------------------------------------------------------------|------------------------------------------|
| `ollama/...`       | none                                                         | Run Ollama locally (see step 3)          |
| `gemini/...`       | `GEMINI_API_KEY`                                             | https://aistudio.google.com/apikey       |
| `openai/...`       | `OPENAI_API_KEY`                                             | https://platform.openai.com/api-keys     |
| `anthropic/...`    | `ANTHROPIC_API_KEY`                                          | https://console.anthropic.com/settings/keys |
| `mistral/...`      | `MISTRAL_API_KEY`                                            | https://console.mistral.ai/api-keys      |
| `vertex_ai/...`    | `VERTEXAI_PROJECT` + `VERTEXAI_LOCATION`                     | https://console.cloud.google.com         |
| `bedrock/...`      | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_REGION_NAME` | https://console.aws.amazon.com/iam  |
| `azure/...`        | `AZURE_API_KEY` + `AZURE_API_BASE`                           | https://portal.azure.com                 |
| `azure_ai/...`     | `AZURE_AI_API_KEY` + `AZURE_AI_API_BASE`                     | https://ai.azure.com                     |
```

### 3b. Add a cloud provider examples section

**Location**: `README.md`, immediately after the provider table (before the "Validation" line).

**Add**:

```markdown
**Cloud provider model examples:**

You can mix providers freely in the same config. Just set the required
env vars in `.env` for each provider you use:

```yaml
# Use Vertex AI Gemini instead of AI Studio Gemini
generation:
  llm_models:
    - "vertex_ai/gemini-2.5-flash"

# Use Bedrock Claude
generation:
  llm_models:
    - "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"

# Mix providers — the optimizer agent explores all of them
generation:
  llm_models:
    - "gemini/gemini-2.5-flash"
    - "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
    - "azure/my-gpt4o-deployment"
```

See `configs/cloud.yaml` for a complete example configuration using cloud providers.
```

### 3c. Update the configuration files list

**Location**: `README.md`, under "Configuration files" (the bullet list of configs).

**Add** a third bullet:

```markdown
- `configs/cloud.yaml`: demonstrates using cloud providers (Vertex AI, Bedrock, Azure).
```

---

## Step 4: Create `configs/cloud.yaml`

**Location**: `configs/cloud.yaml` (new file).

This is a working example config showing cloud providers. It exists alongside `starter.yaml` and `full.yaml` as documentation-by-example.

```yaml
# Agentic AutoRAG — Cloud Providers Configuration
# Demonstrates using Vertex AI, Bedrock, and Azure models.
# Set the required env vars in .env for each provider you use.
# See .env.example for details.

meta:
  project_name: "my-rag-project"
  corpus_path: "./data/corpus/"
  corpus_description: |
    A collection of ~50 ArXiv papers across 5 CS and physics categories.
    The corpus covers NLP, computer vision, machine learning,
    statistics, and optics. Documents are academic papers
    with mathematical notation, tables, figures, and references.
  output_dir: "./experiments/"
  max_trials: 15
  index_registry: true

structural:
  parsers:
    - "docling"
  chunking:
    strategies: ["recursive"]
    chunk_size: { min: 256, max: 1024 }
    chunk_overlap: { min: 0, max: 128 }
  embedding_models:
    - "sentence-transformers/all-MiniLM-L6-v2"
  index_types:
    - "vector_only"
    - "hybrid_bm25_vector"

runtime:
  retrieval:
    top_k: { min: 3, max: 15 }
    hybrid_alpha: { min: 0.0, max: 1.0 }
    reranker:
      models:
        - "none"
        - "cross-encoder/ms-marco-MiniLM-L-6-v2"
      top_n: { min: 3, max: 10 }
    query_expansion: ["none", "hyde"]
  generation:
    llm_models:
      # Mix cloud providers — the optimizer agent explores all of them.
      # Uncomment only the providers you have configured in .env.

      # Google Vertex AI (requires VERTEXAI_PROJECT + VERTEXAI_LOCATION)
      - "vertex_ai/gemini-2.5-flash"
      # - "vertex_ai/gemini-2.5-pro"

      # AWS Bedrock (requires AWS credentials + AWS_REGION_NAME)
      - "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
      # - "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0"
      # - "bedrock/amazon.nova-pro-v1:0"

      # Azure OpenAI (requires AZURE_API_KEY + AZURE_API_BASE)
      # Note: replace "my-gpt4o-deployment" with your actual deployment name
      # - "azure/my-gpt4o-deployment"

    temperature: { min: 0.0, max: 0.7 }

examiner:
  exam_size: 50
  irt_discrimination_threshold: 0.3
  refresh_interval_trials: 5

agent:
  # Agent models can also use cloud providers
  optimizer_model: "vertex_ai/gemini-2.5-flash"
  examiner_model: "vertex_ai/gemini-2.5-flash"
  max_history_trials: 10
```

---

## Testing checklist

After implementing the changes, verify:

1. **Existing behavior is preserved**: Run `uv run agentic-autorag optimize --config configs/starter.yaml` with `GEMINI_API_KEY` set. It should work exactly as before.

2. **Missing key detection works for new providers**: Create a test YAML that references `vertex_ai/gemini-2.5-flash` without setting `VERTEXAI_PROJECT` / `VERTEXAI_LOCATION`. Run `uv run agentic-autorag optimize --config <test.yaml>`. It should raise `EnvironmentError` listing the missing vars.

3. **Alternative auth detection works for Bedrock**: Set only `AWS_REGION_NAME` (simulating an EC2 IAM role). The Bedrock check should pass (third alternative satisfied). Unset it — should fail.

4. **No false positives**: A config with only `ollama/` models should not trigger any env var checks.

5. **Run existing tests**: `uv run pytest -q` should pass unchanged — no test files need modification since the tests don't mock `_check_api_keys` and the config models are untouched.

---

## What this plan intentionally does NOT do

- **No LiteLLM Router integration**: The research doc mentions `litellm.Router` for load balancing/failover. This is out of scope — it would require significant refactoring of `pipeline.py` and `reasoning_agent.py`. Can be a follow-up.
- **No provider-specific YAML fields**: We don't add `vertex_project`, `aws_region`, etc. to the YAML config models. Everything goes through `.env`. This keeps configs portable.
- **No live credential validation at startup**: `litellm.validate_environment()` is unreliable for some providers. `litellm.check_valid_key()` costs tokens. The current env-var-presence check is the right tradeoff — if the var is set but the credential is invalid, the user gets a clear LiteLLM error on the first actual API call.
- **No new Python dependencies**: LiteLLM handles provider-specific deps internally (boto3 for Bedrock, google-cloud-aiplatform for Vertex). Users who don't use those providers don't need them installed.
