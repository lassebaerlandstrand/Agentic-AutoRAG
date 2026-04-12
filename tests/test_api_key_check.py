"""Behavior-based tests for _check_api_keys() provider validation.

Each test sets up env vars (input), calls _check_api_keys(), and asserts
the outcome — either an OSError with the right message, or no error.
"""

import pytest

from agentic_autorag.config.models import (
    AgentConfig,
    ProjectConfig,
    SearchSpace,
)
from agentic_autorag.orchestrator import _check_api_keys


def _make_config(
    llm_models: list[str],
    optimizer_model: str = "ollama/llama3.2",
    examiner_model: str = "ollama/llama3.2",
) -> ProjectConfig:
    """Build a minimal ProjectConfig with the given model strings."""
    return ProjectConfig(
        search_space=SearchSpace(
            embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
            llm_models=llm_models,
        ),
        agent=AgentConfig(
            optimizer_model=optimizer_model,
            examiner_model=examiner_model,
        ),
    )


# ── Env var names used across providers ──────────────────────────────

_ALL_PROVIDER_VARS = [
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "COHERE_API_KEY",
    "MISTRAL_API_KEY",
    "VERTEXAI_PROJECT",
    "VERTEXAI_LOCATION",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION_NAME",
    "AWS_PROFILE",
    "AZURE_API_KEY",
    "AZURE_API_BASE",
    "AZURE_AI_API_KEY",
    "AZURE_AI_API_BASE",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure no provider env vars leak between tests."""
    for var in _ALL_PROVIDER_VARS:
        monkeypatch.delenv(var, raising=False)


# ── Single-key providers ─────────────────────────────────────────────


def test_gemini_missing_raises(monkeypatch):
    cfg = _make_config(["gemini/gemini-2.5-flash"])
    with pytest.raises(OSError, match="GEMINI_API_KEY"):
        _check_api_keys(cfg)


def test_gemini_present_passes(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg = _make_config(["gemini/gemini-2.5-flash"])
    _check_api_keys(cfg)


def test_openai_missing_raises(monkeypatch):
    cfg = _make_config(["openai/gpt-4o"])
    with pytest.raises(OSError, match="OPENAI_API_KEY"):
        _check_api_keys(cfg)


def test_anthropic_missing_raises(monkeypatch):
    cfg = _make_config(["anthropic/claude-sonnet-4-20250514"])
    with pytest.raises(OSError, match="ANTHROPIC_API_KEY"):
        _check_api_keys(cfg)


# ── Multi-variable provider: Vertex AI ───────────────────────────────


def test_vertex_ai_missing_both_raises(monkeypatch):
    cfg = _make_config(["vertex_ai/gemini-2.5-flash"])
    with pytest.raises(OSError, match="VERTEXAI_PROJECT") as exc_info:
        _check_api_keys(cfg)
    assert "VERTEXAI_LOCATION" in str(exc_info.value)


def test_vertex_ai_partial_raises(monkeypatch):
    monkeypatch.setenv("VERTEXAI_PROJECT", "my-project")
    cfg = _make_config(["vertex_ai/gemini-2.5-flash"])
    with pytest.raises(OSError, match="VERTEXAI_LOCATION"):
        _check_api_keys(cfg)


def test_vertex_ai_fully_present_passes(monkeypatch):
    monkeypatch.setenv("VERTEXAI_PROJECT", "my-project")
    monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")
    cfg = _make_config(["vertex_ai/gemini-2.5-flash"])
    _check_api_keys(cfg)


# ── Alternative auth: Bedrock ────────────────────────────────────────


def test_bedrock_all_missing_raises(monkeypatch):
    cfg = _make_config(["bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"])
    with pytest.raises(OSError, match="AWS_ACCESS_KEY_ID"):
        _check_api_keys(cfg)


def test_bedrock_explicit_keys_passes(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA...")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    cfg = _make_config(["bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"])
    _check_api_keys(cfg)


def test_bedrock_named_profile_passes(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "my-sso-profile")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    cfg = _make_config(["bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"])
    _check_api_keys(cfg)


def test_bedrock_iam_role_passes(monkeypatch):
    """On EC2/ECS/Lambda with an IAM role, only AWS_REGION_NAME is needed."""
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    cfg = _make_config(["bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"])
    _check_api_keys(cfg)


# ── Azure providers ──────────────────────────────────────────────────


def test_azure_missing_raises(monkeypatch):
    cfg = _make_config(["azure/my-gpt4o-deployment"])
    with pytest.raises(OSError, match="AZURE_API_KEY") as exc_info:
        _check_api_keys(cfg)
    assert "AZURE_API_BASE" in str(exc_info.value)


def test_azure_present_passes(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "key")
    monkeypatch.setenv("AZURE_API_BASE", "https://my-resource.openai.azure.com/")
    cfg = _make_config(["azure/my-gpt4o-deployment"])
    _check_api_keys(cfg)


def test_azure_ai_missing_raises(monkeypatch):
    cfg = _make_config(["azure_ai/mistral-large"])
    with pytest.raises(OSError, match="AZURE_AI_API_KEY"):
        _check_api_keys(cfg)


def test_azure_ai_present_passes(monkeypatch):
    monkeypatch.setenv("AZURE_AI_API_KEY", "key")
    monkeypatch.setenv("AZURE_AI_API_BASE", "https://endpoint.inference.ai.azure.com/")
    cfg = _make_config(["azure_ai/mistral-large"])
    _check_api_keys(cfg)


# ── Local-only configs ───────────────────────────────────────────────


def test_local_only_passes(monkeypatch):
    cfg = _make_config(["ollama/llama3.2"])
    _check_api_keys(cfg)


def test_sentence_transformers_skipped(monkeypatch):
    cfg = _make_config(["sentence-transformers/all-MiniLM-L6-v2"])
    _check_api_keys(cfg)


def test_vllm_skipped(monkeypatch):
    """hosted_vllm/ models are framework-managed, no env var check needed."""
    cfg = _make_config(["hosted_vllm/Qwen/Qwen3-8B"])
    _check_api_keys(cfg)


# ── Edge cases ───────────────────────────────────────────────────────


def test_unknown_provider_skipped(monkeypatch):
    cfg = _make_config(["some_new_provider/model-v1"])
    _check_api_keys(cfg)


def test_model_without_slash_skipped(monkeypatch):
    cfg = _make_config(["bare-model-name"])
    _check_api_keys(cfg)


def test_deduplication_single_error(monkeypatch):
    """Multiple models from the same provider should produce one error, not three."""
    cfg = _make_config(
        [
            "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
            "bedrock/amazon.nova-pro-v1:0",
        ]
    )
    with pytest.raises(OSError) as exc_info:
        _check_api_keys(cfg)
    msg = str(exc_info.value)
    assert msg.count("bedrock/") == 1


def test_mixed_providers_only_missing_reported(monkeypatch):
    """Only the provider with missing vars should appear in the error."""
    monkeypatch.setenv("VERTEXAI_PROJECT", "proj")
    monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")
    cfg = _make_config(
        ["gemini/gemini-2.5-flash", "vertex_ai/gemini-2.5-flash"],
    )
    with pytest.raises(OSError) as exc_info:
        _check_api_keys(cfg)
    msg = str(exc_info.value)
    assert "GEMINI_API_KEY" in msg
    assert "VERTEXAI" not in msg


def test_agent_models_checked(monkeypatch):
    """Agent models (optimizer/examiner) are also validated."""
    cfg = _make_config(
        ["ollama/llama3.2"],
        optimizer_model="anthropic/claude-sonnet-4-20250514",
        examiner_model="ollama/llama3.2",
    )
    with pytest.raises(OSError, match="ANTHROPIC_API_KEY"):
        _check_api_keys(cfg)


def test_error_message_points_to_env_example(monkeypatch):
    cfg = _make_config(["gemini/gemini-2.5-flash"])
    with pytest.raises(OSError) as exc_info:
        _check_api_keys(cfg)
    assert ".env.example" in str(exc_info.value)
