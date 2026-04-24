"""EM / F1 / retrieval metrics and LLM-as-judge for free-form QA scoring.

The ``normalize_answer`` / ``exact_match`` / ``f1`` implementations are
deliberately copied from the HotpotQA / SQuAD official evaluation scripts
so our numbers are directly leaderboard-comparable. Source:
https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
"""

from __future__ import annotations

import logging
import re
import string
from collections import Counter

import litellm

from agentic_autorag.benchmark_eval.prompts import JUDGE_PROMPT

logger = logging.getLogger(__name__)

_JUDGE_PARSE_RE = re.compile(r"\s*(YES|NO)\b", re.IGNORECASE)


def normalize_answer(s: str) -> str:
    """SQuAD/HotpotQA canonical normalization: lowercase, strip punct, drop articles, collapse ws."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 after canonical normalization.

    Mirrors the HotpotQA evaluator: yes/no/noanswer answers require exact
    string match — partial overlap on those tokens would be misleading.
    """
    norm_pred = normalize_answer(pred)
    norm_gold = normalize_answer(gold)

    if norm_pred in ("yes", "no", "noanswer") or norm_gold in ("yes", "no", "noanswer"):
        return 1.0 if norm_pred == norm_gold else 0.0

    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_em(pred: str, gold_answers: list[str]) -> float:
    return max((exact_match(pred, g) for g in gold_answers), default=0.0)


def best_f1(pred: str, gold_answers: list[str]) -> float:
    return max((token_f1(pred, g) for g in gold_answers), default=0.0)


def retrieval_metrics(
    retrieved_doc_ids: list[str],
    supporting_doc_ids: list[str],
    ks: tuple[int, ...] = (1, 2, 5, 10),
) -> tuple[dict[int, float], int | None]:
    """Compute Recall@k for each k and the 1-indexed rank of the first gold doc.

    Returns ``({k: recall_at_k}, first_gold_rank_or_None)``. Duplicate
    retrieved doc_ids are deduplicated in rank order before scoring so a
    single gold doc appearing at ranks 1 and 3 still counts once.
    """
    if not supporting_doc_ids:
        return {k: 0.0 for k in ks}, None

    gold = set(supporting_doc_ids)
    seen: set[str] = set()
    dedup: list[str] = []
    for d in retrieved_doc_ids:
        if d in seen:
            continue
        seen.add(d)
        dedup.append(d)

    first_rank: int | None = None
    for rank, d in enumerate(dedup, start=1):
        if d in gold:
            first_rank = rank
            break

    recalls: dict[int, float] = {}
    for k in ks:
        hits = sum(1 for d in dedup[:k] if d in gold)
        recalls[k] = hits / len(gold)
    return recalls, first_rank


async def llm_judge(
    judge_model: str,
    question: str,
    pred: str,
    gold_answers: list[str],
    timeout_s: float = 30.0,
) -> int | None:
    """Ask an LLM whether ``pred`` matches any answer in ``gold_answers``.

    Returns 1 (correct), 0 (incorrect), or None when the model's response
    doesn't start with YES/NO — in which case the caller records the
    judge as invalid for this question and excludes it from the accuracy
    denominator.
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold=" | ".join(gold_answers),
        pred=pred,
    )
    try:
        response = await litellm.acompletion(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            seed=42,
            max_tokens=4,
            num_retries=0,
            timeout=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Judge call failed: %s", exc)
        return None
    text = response.choices[0].message.content or ""
    match = _JUDGE_PARSE_RE.match(text)
    if not match:
        return None
    return 1 if match.group(1).upper() == "YES" else 0
