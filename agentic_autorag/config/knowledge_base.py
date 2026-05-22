"""Knowledge base loader and formatter for the optimizer agent.

Provides rich context about models (LLMs, embeddings, rerankers) and parameter
semantics so the agent can make informed optimization decisions.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import yaml

from agentic_autorag.config.aa_matcher import find_best_aa_slug
from agentic_autorag.config.reasoning_mode import select_pair as _select_reasoning_pair

logger = logging.getLogger(__name__)

_KB_DIR = Path(__file__).parent.parent.parent / "knowledge_base"
_GRAPH_PARAM_NAMES = frozenset({"graph_query_mode", "graph_top_k"})


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
        reasoning_enabled: bool = True,
        reasoning_effort: str = "medium",
        include_graph: bool = False,
        skip_params: set[str] | None = None,
        option_filter: dict[str, set[str]] | None = None,
    ) -> str:
        """Return a markdown-formatted knowledge base section, filtered to search space models.

        ``reasoning_enabled`` mirrors ``SearchSpace.reasoning``: when False the
        agent cannot pick reasoning=true for any model, so the
        ``Supports Reasoning`` column and the ``reasoning`` parameter guide
        entry are suppressed to stop the proposer wasting tokens on a knob it
        can't actually move.

        ``reasoning_effort`` mirrors ``GeneratorSearchSpace.reasoning_effort``
        (``"low"``, ``"medium"``, ``"high"``). It controls which ON variant
        the reasoning row displays so the agent's view of model strength
        matches the effort the engine will actually request at runtime.

        ``skip_params`` is the set of TrialConfig field names whose parameter-
        guide entries should be suppressed. The intended caller is "every
        pinned field" so the agent never reads guidance for a knob it cannot
        turn — the pinned values themselves still appear in the search-space
        "Fixed values" block above the guide.

        ``option_filter`` maps a parameter name to the set of option keys that
        survive in the configured search space. Option entries outside the set
        are dropped before rendering. Parameters absent from the mapping show
        every option.
        """
        sections: list[str] = []

        llm_section = self._format_llm_section(llm_models, reasoning_allowed or {}, reasoning_enabled, reasoning_effort)
        if llm_section:
            sections.append(llm_section)

        embed_section = self._format_embedding_section(embedding_models)
        if embed_section:
            sections.append(embed_section)

        reranker_section = self._format_reranker_section(reranker_models)
        if reranker_section:
            sections.append(reranker_section)

        param_section = self._format_param_section(
            include_graph=include_graph,
            include_reasoning=reasoning_enabled,
            skip_params=skip_params or set(),
            option_filter=option_filter or {},
        )
        if param_section:
            sections.append(param_section)

        if not sections:
            return ""

        return "## Knowledge Base\n\n" + "\n\n".join(sections)

    def _find_llm_entry(self, litellm_name: str) -> dict | None:
        """Find the base LLM entry that matches a given LiteLLM model id.

        Order:
          1. Exact membership in any base entry's ``litellm_ids`` list
             (the precomputed mapping from build time).
          2. Fallback: re-run the AA matcher against base AA slugs. Catches
             models the user added after the last knowledge-base rebuild.

        Variant entries (those with a ``base_slug`` field) are skipped — they
        are accessed via ``_base_to_variants``.
        """
        models = self._llms.get("models", {})

        for entry in models.values():
            if "base_slug" in entry:
                continue
            if litellm_name in (entry.get("litellm_ids") or []):
                return entry

        candidate_slugs = [
            entry.get("slug", "") for entry in models.values() if "base_slug" not in entry and entry.get("slug")
        ]
        best = find_best_aa_slug(litellm_name, candidate_slugs)
        if best is not None:
            return models.get(best)

        return None

    def _get_model_display_rows(
        self,
        model_name: str,
        entry: dict | None,
        reasoning_allowed: bool,
        reasoning_effort: str = "medium",
    ) -> list[dict]:
        """Return the ordered display rows for a single model.

        - If ``entry`` is None the model is in the search space but missing from
          the KB: emit a single row with the litellm name and no benchmarks.
        - If ``reasoning_allowed`` is True (model supports reasoning AND search
          space allows it), always emit two rows labelled
          ``(non-reasoning)`` / ``(reasoning)``. Missing variant data shows as
          blank cells so the agent still sees that the row exists. Each row is
          tagged with ``__supports_reasoning__=True`` for the
          ``Supports Reasoning`` column.
        - If ``reasoning_allowed`` is False, emit a single row using the OFF
          entry when available (falling back to the base entry), tagged with
          ``__supports_reasoning__=False``.

        See :mod:`agentic_autorag.config.reasoning_mode` for the AA→on/off
        classification — it covers `-reasoning`/`-non-reasoning` pairs as well
        as `-thinking`, `-adaptive`, and `-low/-medium/-high` effort variants.
        """
        if entry is None:
            return [{"litellm_name": model_name, "__supports_reasoning__": reasoning_allowed}]
        slug = entry.get("slug", "")
        variants = self._base_to_variants.get(slug, [])

        off_entry, on_entry = _select_reasoning_pair(entry, variants, reasoning_effort)

        if reasoning_allowed:
            return [
                {
                    "litellm_name": f"{model_name} (non-reasoning)",
                    "__supports_reasoning__": True,
                    **(off_entry or {}),
                },
                {
                    "litellm_name": f"{model_name} (reasoning)",
                    "__supports_reasoning__": True,
                    **(on_entry or {}),
                },
            ]

        if off_entry is not None:
            return [{"litellm_name": model_name, "__supports_reasoning__": False, **off_entry}]

        return [{"litellm_name": model_name, "__supports_reasoning__": False, **entry}]

    def _format_llm_section(
        self,
        llm_models: list[str],
        reasoning_allowed: dict[str, bool],
        reasoning_enabled: bool,
        reasoning_effort: str = "medium",
    ) -> str:
        if not self._llms:
            return ""

        rows: list[dict] = []
        for model_name in llm_models:
            entry = self._find_llm_entry(model_name)
            allowed = reasoning_allowed.get(model_name, False)
            rows.extend(self._get_model_display_rows(model_name, entry, allowed, reasoning_effort))

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
        if reasoning_enabled:
            cols.append("Supports Reasoning")
        lines = ["### LLM Models", "", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for r in rows:
            b = r.get("benchmarks") or {}
            p = r.get("pricing") or {}
            perf = r.get("performance") or {}

            cells = [
                f"`{r['litellm_name']}`",
                r.get("creator", "—"),
                _fmt(b.get("mmlu_pro")),
                _fmt(b.get("gpqa")),
                _fmt(b.get("ifbench")),
                _fmt(b.get("artificial_analysis_intelligence_index")),
                _fmt_price(p.get("input_per_1m_tokens")),
                _fmt_price(p.get("output_per_1m_tokens")),
                _fmt(perf.get("median_output_tokens_per_second"), decimals=0),
                _fmt_tokens(p.get("max_input_tokens")),
            ]
            if reasoning_enabled:
                cells.append("✓" if r.get("__supports_reasoning__") else "✗")
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def _format_embedding_section(self, embedding_models: list[str]) -> str:
        if not self._embeddings or not embedding_models:
            return ""

        models = self._embeddings.get("models", {})

        lines = ["### Embedding Models", ""]
        lines.append("| Model | Dim | Max Tokens | Params (B) | Retrieval | STS | Reranking | Memory (MB) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for name in embedding_models:
            r = models.get(name)
            if r is None:
                # Model is in the search space but missing from the KB: show it so the
                # Proposer sees it exists, but with — in every data column.
                lines.append(f"| `{name}` | — | — | — | — | — | — | — |")
                continue
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
        # "none" is a valid sentinel, not a model: never render it as a table row.
        active = [m for m in reranker_models if m != "none"]
        if not active:
            return ""

        lines = ["### Reranker Models", ""]
        lines.append("| Model | Params | MTEB-R | MMTEB-R | FollowIR |")
        lines.append("|---|---|---|---|---|")
        for name in active:
            r = models.get(name)
            if r is None:
                # In the search space but missing from the KB — show the name with —
                # so the Proposer knows the option exists.
                lines.append(f"| `{name}` | — | — | — | — |")
                continue
            s = r.get("scores") or {}
            lines.append(
                f"| `{name}` "
                f"| {r.get('parameters', '?')} "
                f"| {_fmt(s.get('mteb_reranking'))} "
                f"| {_fmt(s.get('mmteb_reranking'))} "
                f"| {_fmt(s.get('followir'))} |"
            )

        return "\n".join(lines)

    def _llm_quality_score(
        self,
        entry: dict,
        reasoning_allowed: bool = False,
        reasoning_effort: str = "medium",
    ) -> float:
        """Extract a single quality score from an LLM KB entry.

        The score targets the variant that will actually be deployed under
        the project's reasoning setting — same selection rule the display
        path uses, so the agent's KB view and the probe ranker agree on
        which model is stronger. With ``reasoning_allowed=False`` we score
        the OFF variant (e.g. ``gpt-5-4-nano-non-reasoning`` II=24.4)
        rather than the base xhigh entry; with ``True`` we score the ON
        variant whose effort matches ``reasoning_effort`` (one of
        ``"low"``, ``"medium"``, ``"high"``). Falls back to the base
        entry when no variant exists.

        Prefers Intelligence Index (pre-computed aggregate on ~0-50 scale).
        Falls back to mean of available benchmarks (MMLU Pro, GPQA, IFBench)
        scaled to a comparable range. Only averages over benchmarks present
        for this model so models with missing fields are not penalised.
        """
        scored_entry = self._select_variant_entry(entry, reasoning_allowed, reasoning_effort)
        b = scored_entry.get("benchmarks") or {}
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

    def _select_variant_entry(
        self,
        base_entry: dict,
        reasoning_allowed: bool,
        reasoning_effort: str = "medium",
    ) -> dict:
        """Pick the KB entry whose benchmarks reflect the deployed reasoning mode.

        Mirrors ``_get_model_display_rows``: under ``reasoning_allowed=False``
        prefer the OFF variant if one exists, else fall back to the base
        entry; under ``True`` prefer the ON variant whose effort variant
        matches ``reasoning_effort`` (passed through to
        ``_select_reasoning_pair``) and fall back to base. The base entry
        is the fallback for both modes because models without sibling
        variants (e.g. ``o4-mini``, ``gpt-4o-mini``) only have a base
        entry to score against.
        """
        slug = base_entry.get("slug", "")
        variants = self._base_to_variants.get(slug, [])
        off_entry, on_entry = _select_reasoning_pair(base_entry, variants, reasoning_effort)
        if reasoning_allowed:
            return on_entry or base_entry
        return off_entry or base_entry

    def rank_llms(
        self,
        model_names: list[str],
        reasoning_allowed: dict[str, bool] | None = None,
        reasoning_effort: str = "medium",
    ) -> tuple[list[str], list[str]]:
        """Rank LLM model names by quality (weakest first).

        ``reasoning_allowed`` maps each litellm id to whether reasoning may
        run for that model under the current search space (use
        ``SearchSpace.is_reasoning_allowed``). Missing entries default to
        False — i.e. score the OFF variant — which matches the
        ``reasoning: false`` bench projects. Pass the same map the display
        path uses so the agent's KB view and this ranking stay in sync.

        ``reasoning_effort`` is the project's
        ``GeneratorSearchSpace.reasoning_effort`` (``"low"``, ``"medium"``,
        or ``"high"``). It selects which ON variant is scored when reasoning
        is allowed for a model.

        Returns (ranked known models, list of unknown models not in KB).
        """
        scored: list[tuple[float, str]] = []
        unknown: list[str] = []
        for name in model_names:
            entry = self._find_llm_entry(name)
            if entry:
                allowed = (reasoning_allowed or {}).get(name, False)
                scored.append((self._llm_quality_score(entry, allowed, reasoning_effort), name))
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

    def _format_param_section(
        self,
        include_graph: bool = False,
        include_reasoning: bool = True,
        skip_params: set[str] | None = None,
        option_filter: dict[str, set[str]] | None = None,
    ) -> str:
        """Render the parameter guide.

        - Preserves bullet/newline structure in ``guidance`` (no flattening).
        - Renders the YAML ``options:`` map as a nested sub-list, optionally
          filtered to the option keys present in ``option_filter[name]``.
        - Drops parameters in ``skip_params`` and parameters whose post-filter
          option set has fewer than two entries (effectively pinned — the agent
          gets the value from the "Fixed values" block).
        """
        params = self._params.get("parameters", {})
        if not params:
            return ""

        skip: set[str] = set()
        if not include_graph:
            skip.update(_GRAPH_PARAM_NAMES)
        if not include_reasoning:
            skip.add("reasoning")
        if skip_params:
            skip.update(skip_params)

        filt = option_filter or {}

        lines = ["### Parameter Guide", ""]
        for name, info in params.items():
            if name in skip:
                continue
            desc = (info.get("description") or "").strip()
            guidance = info.get("guidance") or ""
            opts = info.get("options") or {}
            if name in filt and opts:
                allowed = filt[name]
                opts = {k: v for k, v in opts.items() if k in allowed}

            lines.append(f"- **{name}**: {desc}")
            for gline in guidance.splitlines():
                gline = gline.rstrip()
                if gline:
                    lines.append(f"    {gline}")
            if opts:
                lines.append("    Options:")
                for opt_name, opt_desc in opts.items():
                    lines.append(f"      - {opt_name}: {opt_desc}")

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
