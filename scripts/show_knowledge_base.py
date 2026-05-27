"""Render the framework's knowledge-base markdown for a project config.

``ReasoningAgent`` injects a ``## Knowledge Base`` block into every
optimizer / diagnoser prompt and, with ``debug_prompts: true`` (the bench
default), mirrors it to ``run.log``. This script wraps the same
``KnowledgeBase.format_for_prompt`` call so the output can be inspected
without spinning up an optimization run — useful for paper-appendix
dumps and for auditing what models the agent actually sees after a
search-space edit.

Usage::

    uv run python scripts/show_knowledge_base.py configs/hotpot_qa.yaml
    uv run python scripts/show_knowledge_base.py path/to/project.yaml -o kb.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentic_autorag.config import load_config
from agentic_autorag.config.knowledge_base import KnowledgeBase


def render(config_path: Path) -> str:
    cfg = load_config(config_path)
    ss = cfg.search_space
    llms = ss.all_llm_models()
    kb = KnowledgeBase()
    return kb.format_for_prompt(
        llm_models=llms,
        embedding_models=list(ss.embedding.models),
        reranker_models=list(ss.reranker.models),
        reasoning_allowed={m: ss.is_reasoning_allowed(m) for m in llms},
        reasoning_enabled=bool(ss.generator.reasoning),
        reasoning_effort=ss.generator.reasoning_effort,
        include_graph=cfg.uses_graph(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render the framework's debug-prompts knowledge base for a "
            "given project config, without starting an optimization run."
        )
    )
    parser.add_argument("config", type=Path, help="path to a project YAML")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="write markdown to this file instead of stdout",
    )
    args = parser.parse_args()

    markdown = render(args.config)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {len(markdown)} chars to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(markdown)
        if not markdown.endswith("\n"):
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
