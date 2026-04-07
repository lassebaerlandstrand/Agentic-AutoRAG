"""Knowledge base loader and formatter for the optimizer agent.

Provides rich context about models (LLMs, embeddings, rerankers) and parameter
semantics so the agent can make informed optimization decisions.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_KB_DIR = Path(__file__).parent.parent.parent / "knowledge_base"


def _normalize(name: str) -> str:
    """Reduce a model name to a canonical form for cross-convention matching."""
    s = name.lower()
    s = re.sub(r"^[a-z0-9_\-]+/", "", s)
    s = re.sub(r"^(us|eu|apac|global|jp|au)\.[a-z0-9]+\.", "", s)
    s = re.sub(r"[-@:]\d{6,}", "", s)
    s = re.sub(r"-v?\d+:\d+$", "", s)
    s = re.sub(r":\d+$", "", s)
    s = s.replace(".", "-").replace("_", "-")
    return s


class KnowledgeBase:
    """Loads knowledge base YAMLs and formats filtered context for agent prompts."""

    def __init__(self, kb_dir: Path = _KB_DIR) -> None:
        self._llms: dict = {}
        self._embeddings: dict = {}
        self._rerankers: dict = {}
        self._params: dict = {}
        self._base_to_variants: dict[str, list[dict]] = {}
        self._load(kb_dir)

    def _load(self, kb_dir: Path) -> None:
        for filename, attr in [
            ("llms.yaml", "_llms"),
            ("embeddings.yaml", "_embeddings"),
            ("rerankers.yaml", "_rerankers"),
            ("parameter_descriptions.yaml", "_params"),
        ]:
            path = kb_dir / filename
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    setattr(self, attr, yaml.safe_load(f) or {})
            else:
                logger.warning("Knowledge base file not found: %s", path)
        self._build_variant_index()

    def _build_variant_index(self) -> None:
        """Build base_slug → [variant entries] lookup for reasoning benchmark display."""
        self._base_to_variants = {}
        models = self._llms.get("models", {})
        for entry in models.values():
            base_slug = entry.get("base_slug")
            if base_slug and base_slug in models:
                self._base_to_variants.setdefault(base_slug, []).append(entry)

    def format_for_prompt(
        self,
        llm_models: list[str],
        embedding_models: list[str],
        reranker_models: list[str],
        reasoning_allowed: dict[str, bool] | None = None,
    ) -> str:
        """Return a markdown-formatted knowledge base section, filtered to search space models."""
        sections: list[str] = []

        llm_section = self._format_llm_section(llm_models, reasoning_allowed or {})
        if llm_section:
            sections.append(llm_section)

        embed_section = self._format_embedding_section(embedding_models)
        if embed_section:
            sections.append(embed_section)

        reranker_section = self._format_reranker_section(reranker_models)
        if reranker_section:
            sections.append(reranker_section)

        param_section = self._format_param_section()
        if param_section:
            sections.append(param_section)

        if not sections:
            return ""

        return "## Knowledge Base\n\n" + "\n\n".join(sections)

    def _find_llm_entry(self, litellm_name: str) -> dict | None:
        """Find the base LLM entry whose litellm_ids list contains the given name.

        Variant entries (those with a ``base_slug`` field) are skipped — they are
        always accessed via ``_base_to_variants``, never returned directly here.
        """
        models = self._llms.get("models", {})
        norm = _normalize(litellm_name)
        sig = frozenset(t for t in norm.split("-") if t != "0")

        for entry in models.values():
            if "base_slug" in entry:  # skip variants
                continue
            ids = entry.get("litellm_ids") or []
            slug_norm = _normalize(entry.get("slug", ""))
            slug_sig = frozenset(t for t in slug_norm.split("-") if t != "0")

            if litellm_name in ids:
                return entry
            if any(_normalize(lid) == norm for lid in ids):
                return entry
            if slug_norm == norm:
                return entry
            # Token-set equality fallback: handles word-order and minor version gaps
            if len(sig) >= 2 and sig == slug_sig:
                return entry

        return None

    def _get_model_display_rows(self, model_name: str, entry: dict, reasoning_allowed: bool) -> list[dict]:
        """Return the ordered display rows for a single model.

        Rules:
        - If reasoning_allowed AND a reasoning + non-reasoning pair exists in the KB:
            row 1: ``{model_name} (non-reasoning)`` with non-reasoning benchmarks
            row 2: ``{model_name} (reasoning)``     with reasoning benchmarks
        - Otherwise: single plain-name row using the non-reasoning entry's benchmarks.

        Handles two KB shapes:
        - Base is non-reasoning + has a ``reasoning`` variant
        - Base is reasoning-default + has a ``non-reasoning`` variant
        """
        slug = entry.get("slug", "")
        variants = self._base_to_variants.get(slug, [])

        reasoning_variant = next((v for v in variants if v.get("variant_type") == "reasoning"), None)
        non_reasoning_variant = next(
            (v for v in variants if "non-reasoning" in (v.get("variant_type") or "")),
            None,
        )

        # Determine which entry holds the non-reasoning and reasoning benchmarks
        if non_reasoning_variant and not reasoning_variant:
            # Base is the reasoning-default; non-reasoning variant is the low-mode entry
            non_r_entry = non_reasoning_variant
            r_entry = entry
        elif reasoning_variant:
            # Base is non-reasoning; variant is the high-mode entry
            non_r_entry = entry
            r_entry = reasoning_variant
        else:
            # No variants — single plain row
            return [{"litellm_name": model_name, **entry}]

        if reasoning_allowed:
            return [
                {"litellm_name": f"{model_name} (non-reasoning)", **non_r_entry},
                {"litellm_name": f"{model_name} (reasoning)", **r_entry},
            ]
        # reasoning not allowed — show only the non-reasoning row, plain name
        return [{"litellm_name": model_name, **non_r_entry}]

    def _format_llm_section(self, llm_models: list[str], reasoning_allowed: dict[str, bool]) -> str:
        if not self._llms:
            return ""

        rows: list[dict] = []
        for model_name in llm_models:
            entry = self._find_llm_entry(model_name)
            if entry:
                allowed = reasoning_allowed.get(model_name, False)
                rows.extend(self._get_model_display_rows(model_name, entry, allowed))

        if not rows:
            return ""

        cols = [
            "LiteLLM Name",
            "Creator",
            "MMLU Pro",
            "GPQA",
            "IFBench",
            "Intel. Index",
            "Input $/1M",
            "Output $/1M",
            "Tokens/s",
            "Max Input",
        ]
        lines = ["### LLM Models", "", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for r in rows:
            b = r.get("benchmarks") or {}
            p = r.get("pricing") or {}
            perf = r.get("performance") or {}

            cells = [
                f"`{r['litellm_name']}`",
                r.get("creator", "?"),
                _fmt(b.get("mmlu_pro")),
                _fmt(b.get("gpqa")),
                _fmt(b.get("ifbench")),
                _fmt(b.get("artificial_analysis_intelligence_index")),
                _fmt_price(p.get("input_per_1m_tokens")),
                _fmt_price(p.get("output_per_1m_tokens")),
                _fmt(perf.get("median_output_tokens_per_second"), decimals=0),
                _fmt_tokens(p.get("max_input_tokens")),
            ]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def _format_embedding_section(self, embedding_models: list[str]) -> str:
        if not self._embeddings:
            return ""

        models = self._embeddings.get("models", {})
        rows = [models[m] for m in embedding_models if m in models]
        if not rows:
            return ""

        lines = ["### Embedding Models", ""]
        lines.append("| Model | Dim | Max Tokens | Params (B) | Retrieval | STS | Reranking | Memory (MB) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            s = r.get("scores") or {}
            lines.append(
                f"| `{r['hf_id']}` "
                f"| {r.get('embedding_dimensions', '?')} "
                f"| {r.get('max_tokens', '?')} "
                f"| {_fmt(r.get('parameters_billions'))} "
                f"| {_fmt(s.get('retrieval'))} "
                f"| {_fmt(s.get('sts'))} "
                f"| {_fmt(s.get('reranking'))} "
                f"| {_fmt(r.get('memory_usage_mb'), decimals=0)} |"
            )

        return "\n".join(lines)

    def _format_reranker_section(self, reranker_models: list[str]) -> str:
        if not self._rerankers:
            return ""

        models = self._rerankers.get("models", {})
        active = [m for m in reranker_models if m != "none" and m in models]
        if not active:
            return ""

        lines = ["### Reranker Models", ""]
        lines.append("| Model | Params | MTEB-R | MMTEB-R | FollowIR |")
        lines.append("|---|---|---|---|---|")
        for name in active:
            r = models[name]
            s = r.get("scores") or {}
            lines.append(
                f"| `{name}` "
                f"| {r.get('parameters', '?')} "
                f"| {_fmt(s.get('mteb_reranking'))} "
                f"| {_fmt(s.get('mmteb_reranking'))} "
                f"| {_fmt(s.get('followir'))} |"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Model ranking for probe-based question selection
    # ------------------------------------------------------------------

    def _llm_quality_score(self, entry: dict) -> float:
        """Extract a single quality score from an LLM KB entry.

        Prefers Intelligence Index (pre-computed aggregate on ~0-50 scale).
        Falls back to mean of available benchmarks (MMLU Pro, GPQA, IFBench)
        scaled to a comparable range. Only averages over benchmarks present
        for this model so models with missing fields are not penalised.
        """
        b = entry.get("benchmarks") or {}
        intel_idx = b.get("artificial_analysis_intelligence_index")
        if intel_idx is not None:
            try:
                return float(intel_idx)
            except (TypeError, ValueError):
                pass

        # Fair average of available 0-1 benchmarks, scaled to ~0-50 range
        benchmark_keys = ("mmlu_pro", "gpqa", "ifbench")
        values = []
        for key in benchmark_keys:
            v = b.get(key)
            if v is not None:
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    continue
        if not values:
            return 0.0
        return sum(values) / len(values) * 50.0  # scale 0-1 average to ~0-50

    def rank_llms(self, model_names: list[str]) -> tuple[list[str], list[str]]:
        """Rank LLM model names by quality (weakest first).

        Returns (ranked known models, list of unknown models not in KB).
        """
        scored: list[tuple[float, str]] = []
        unknown: list[str] = []
        for name in model_names:
            entry = self._find_llm_entry(name)
            if entry:
                scored.append((self._llm_quality_score(entry), name))
            else:
                unknown.append(name)
        scored.sort(key=lambda t: t[0])
        return [name for _, name in scored], unknown

    def rank_embeddings(self, model_names: list[str]) -> tuple[list[str], list[str]]:
        """Rank embedding model names by retrieval quality (weakest first).

        Returns (ranked known models, list of unknown models not in KB).
        """
        models = self._embeddings.get("models", {})
        scored: list[tuple[float, str]] = []
        unknown: list[str] = []
        for name in model_names:
            entry = models.get(name)
            if entry:
                scores = entry.get("scores") or {}
                retrieval = scores.get("retrieval")
                try:
                    scored.append((float(retrieval), name))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    unknown.append(name)
            else:
                unknown.append(name)
        scored.sort(key=lambda t: t[0])
        return [name for _, name in scored], unknown

    def rank_rerankers(self, model_names: list[str]) -> tuple[list[str], list[str]]:
        """Rank reranker model names by quality (weakest first).

        ``"none"`` always sorts first (weakest).
        Returns (ranked known models, list of unknown models not in KB).
        """
        models = self._rerankers.get("models", {})
        scored: list[tuple[float, str]] = []
        unknown: list[str] = []
        has_none = False
        for name in model_names:
            if name == "none":
                has_none = True
                continue
            entry = models.get(name)
            if entry:
                scores = entry.get("scores") or {}
                mteb = scores.get("mteb_reranking")
                try:
                    scored.append((float(mteb), name))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    unknown.append(name)
            else:
                unknown.append(name)
        scored.sort(key=lambda t: t[0])
        ranked = [name for _, name in scored]
        if has_none:
            ranked.insert(0, "none")
        return ranked, unknown

    def _format_param_section(self) -> str:
        params = self._params.get("parameters", {})
        if not params:
            return ""

        lines = ["### Parameter Guide", ""]
        for name, info in params.items():
            desc = info.get("description", "")
            guidance = info.get("guidance", "")
            # Collapse multi-line guidance to a single line for table-friendliness
            guidance_flat = " ".join(guidance.strip().splitlines()).strip()
            lines.append(f"- **{name}**: {desc} {guidance_flat}")

        return "\n".join(lines)


def _fmt(value: object, decimals: int = 3) -> str:
    if value is None:
        return "—"
    try:
        f = float(value)  # type: ignore[arg-type]
        if math.isnan(f):
            return "—"
        if decimals == 0:
            return str(int(round(f)))
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_price(value: object) -> str:
    if value is None:
        return "—"
    try:
        f = float(value)  # type: ignore[arg-type]
        if math.isnan(f):
            return "—"
        if f < 0.01:
            return f"${f:.4f}"
        return f"${f:.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_tokens(value: object) -> str:
    if value is None:
        return "—"
    try:
        n = int(float(value))  # type: ignore[arg-type]
        if n >= 1_000_000:
            return f"{n // 1_000_000}M"
        if n >= 1_000:
            return f"{n // 1_000}K"
        return str(n)
    except (TypeError, ValueError):
        return "—"
