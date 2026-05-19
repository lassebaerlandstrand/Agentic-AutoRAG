"""Reachability check for every LLM in a config's search space.

Loads a project config, installs its ``model_aliases``, and sends a tiny
"Say only Hello" completion to every LLM declared in the search-space stage pools.
Prints OK / FAIL per model with timing, token counts, and the alias-resolved
target so deployment-name mismatches are obvious.

Usage:
    uv run scripts/test_llm.py                                # tests configs/hotpot_qa.yaml
    uv run scripts/test_llm.py --config configs/full.yaml     # any config
    uv run scripts/test_llm.py --agent                        # also test optimizer/examiner/judge
    uv run scripts/test_llm.py --graph                        # also test graph extraction model
    uv run scripts/test_llm.py --model azure/gpt-5-nano       # test a single model and exit
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import ProjectConfig
from agentic_autorag.litellm_runtime import (
    acompletion_with_cost,
    configure_litellm_runtime,
    install_model_aliases,
    resolve_model,
)

PROMPT = "Say only 'Hello'"
# Reasoning models need budget for hidden reasoning tokens before any visible
# output; 4000 is enough for "Hello" even at reasoning_effort=high. Non-reasoning
# models cap at their own max well below this and ignore the excess.
MAX_TOKENS = 4000
DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "hotpot_qa.yaml"


async def _probe(model: str) -> dict:
    target, _ = resolve_model(model)
    start = time.perf_counter()
    try:
        response, usage = await acompletion_with_cost(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_completion_tokens=MAX_TOKENS,
            reasoning_effort="minimal",
        )
        elapsed = time.perf_counter() - start
        content = (response.choices[0].message.content or "").strip()
        return {
            "model": model,
            "target": target,
            "ok": True,
            "elapsed_s": elapsed,
            "content": content,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "usd": usage["usd"],
        }
    except Exception as e:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        return {
            "model": model,
            "target": target,
            "ok": False,
            "elapsed_s": elapsed,
            "error": str(e).splitlines()[0][:200],
        }


def _format_row(result: dict) -> str:
    status = "OK  " if result["ok"] else "FAIL"
    arrow = "" if result["model"] == result["target"] else f" → {result['target']}"
    head = f"  [{status}] {result['model']}{arrow}  ({result['elapsed_s']:.1f}s)"
    if result["ok"]:
        return (
            f"{head}\n"
            f"        reply={result['content']!r}  "
            f"tokens={result['prompt_tokens']}/{result['completion_tokens']}  "
            f"cost=${result['usd']:.6f}"
        )
    return f"{head}\n        error: {result['error']}"


def _collect_models(config: ProjectConfig, *, agent: bool, graph: bool) -> list[str]:
    models: list[str] = config.search_space.all_llm_models()
    if agent:
        models.append(config.agent.optimizer_model)
        models.append(config.agent.examiner_model)
        if config.agent.judge_model:
            models.append(config.agent.judge_model)
    if graph and config.graph is not None:
        models.append(config.graph.extraction_model)
    # Stable de-dup preserving first occurrence.
    seen: set[str] = set()
    out: list[str] = []
    for m in models:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


async def _run(models: list[str], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(m: str) -> dict:
        async with sem:
            result = await _probe(m)
            print(_format_row(result), flush=True)
            return result

    return await asyncio.gather(*(_bounded(m) for m in models))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to project YAML")
    parser.add_argument("--agent", action="store_true", help="Also test agent models (optimizer/examiner/judge)")
    parser.add_argument("--graph", action="store_true", help="Also test graph.extraction_model when defined")
    parser.add_argument("--model", type=str, default=None, help="Test a single model (still honours --config aliases)")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel calls")
    args = parser.parse_args()

    load_dotenv()
    configure_litellm_runtime()

    config = load_config(args.config)
    install_model_aliases(config.model_aliases)

    models = [args.model] if args.model else _collect_models(config, agent=args.agent, graph=args.graph)

    print(f"Testing {len(models)} model(s) from {args.config} (concurrency={args.concurrency})")
    print(f"Aliases installed: {len(config.model_aliases)}")
    print("=" * 72)
    results = asyncio.run(_run(models, args.concurrency))

    print("=" * 72)
    ok_count = sum(1 for r in results if r["ok"])
    total_cost = sum(r.get("usd", 0.0) for r in results)
    print(f"Summary: {ok_count}/{len(results)} reachable  (total cost ${total_cost:.6f})")

    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
