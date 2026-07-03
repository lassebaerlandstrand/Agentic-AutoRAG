"""Add verbatim evidence spans to a doc-level exam (tier B -> tier C).

A custom exam can be supplied at three grounding tiers (see ``OpenEndedQuestion``):
tier A (question + answer only), tier B (answer + the ids of the corpus
documents that support it), and tier C (answer + a verbatim evidence span in
each supporting document). Tier C unlocks span-level retrieval diagnostics; the
other tiers score answer accuracy and — for tier B — doc-level retrieval.

``ground_exam`` upgrades the tier-B questions in an exam to tier C: for every
(question, supporting document) pair it asks an LLM to copy the verbatim
evidence span out of that document, then runs the existing ``verify_source_facts``
gate so only spans that are actually present in their document survive. A
question is upgraded only when *every* one of its supporting spans verifies;
otherwise it is kept unchanged as tier B. Tier-A and already-tier-C questions
pass through untouched. Nothing is ever dropped — ``len(out) == len(in)``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from pydantic import BaseModel, ValidationError

from agentic_autorag.config.models import OpenEndedQuestion
from agentic_autorag.examiner.exam_agent import _strip_markdown_fences
from agentic_autorag.examiner.exam_validator import verify_source_facts
from agentic_autorag.litellm_runtime import acompletion_with_cost

logger = logging.getLogger(__name__)

# A one-time offline build, so we can afford to show the extractor a large slice
# of each document. Evidence beyond this bound is missed and the question then
# stays tier B.
MAX_DOC_CHARS = 32_000
COST_CATEGORY = "ground_exam"
# Cloud endpoints return transient 5xx under load; a dropped span costs a whole
# upgrade, so retry the extraction call a few times.
EXTRACTION_NUM_RETRIES = 3

_EXTRACTION_PROMPT = """You are grounding an exam question in its supporting documents.

Question: {question}
Answer: {answer}

The answer is supported by evidence spread across one or more documents. Below \
is ONE of those documents. Copy the single shortest verbatim span (1-3 \
consecutive sentences) FROM THIS DOCUMENT that this document contributes toward \
answering the question: the fact(s) in THIS document that support the answer. \
This document may supply only an intermediate step (an entity, date, or link \
another document builds on) rather than the answer itself; extract that \
intermediate evidence. If no single sentence is clearly relevant, copy the \
sentence most related to the question's entities.

The span must be an EXACT substring of this document, copied character for \
character: no paraphrasing, no ellipses, no added or removed words, and never \
stitch together separate sentences.

Return ONLY a JSON object of the form {{"span": "<verbatim span>"}}.

Document:
\"\"\"
{document}
\"\"\"
"""


class GroundExamProvenance(BaseModel):
    """Reproducibility record written next to a grounded exam file."""

    n_input: int
    n_upgraded_to_c: int
    n_kept_b: int
    n_already_c: int
    n_tier_a: int
    extractor_model: str
    fuzzy_threshold: float


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_span(raw: str) -> str | None:
    """Pull the ``span`` string out of the extractor's reply, tolerantly."""
    text = _strip_markdown_fences(raw).strip()
    if not text:
        return None
    for candidate in (text, _first_json_object(text)):
        if candidate is None:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("span"), str):
            span = obj["span"].strip()
            return span or None
    return None


async def _extract_span(
    *,
    question: str,
    answer: str,
    document: str,
    model: str,
    reasoning_effort: str | None,
    semaphore: asyncio.Semaphore,
) -> str | None:
    prompt = _EXTRACTION_PROMPT.format(question=question, answer=answer, document=document[:MAX_DOC_CHARS])
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "num_retries": EXTRACTION_NUM_RETRIES,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    async with semaphore:
        try:
            response, _ = await acompletion_with_cost(cost_category=COST_CATEGORY, **kwargs)
        except Exception:
            logger.warning("span extraction call failed after retries", exc_info=True)
            return None
    return _parse_span(response.choices[0].message.content or "")


async def _tier_c_candidate(
    question: OpenEndedQuestion,
    corpus: dict[str, str],
    *,
    model: str,
    reasoning_effort: str | None,
    semaphore: asyncio.Semaphore,
) -> OpenEndedQuestion | None:
    """Extract one verbatim span per supporting doc and assemble a tier-C question.

    Returns ``None`` if a supporting doc is absent from the corpus or its span
    can't be extracted — the question is then not groundable and stays tier B.
    """
    doc_ids = list(question.supporting_doc_ids)
    if not doc_ids or any(d not in corpus for d in doc_ids):
        return None
    spans = await asyncio.gather(
        *(
            _extract_span(
                question=question.question,
                answer=question.canonical_answer,
                document=corpus[d],
                model=model,
                reasoning_effort=reasoning_effort,
                semaphore=semaphore,
            )
            for d in doc_ids
        )
    )
    if any(s is None for s in spans):
        return None
    try:
        return OpenEndedQuestion(
            id=question.id,
            question=question.question,
            canonical_answer=question.canonical_answer,
            answer_variants=list(question.answer_variants),
            reasoning_type=question.reasoning_type,
            source_doc_ids=list(doc_ids),
            source_spans=[s for s in spans if s is not None],
            supporting_doc_ids=list(doc_ids),
        )
    except ValidationError:
        logger.warning("tier-C assembly failed for %s", question.id, exc_info=True)
        return None


async def ground_exam(
    questions: list[OpenEndedQuestion],
    corpus: dict[str, str],
    *,
    extractor_model: str,
    reasoning_effort: str | None = None,
    fuzzy_threshold: float = 0.9,
    concurrency: int = 10,
) -> tuple[list[OpenEndedQuestion], GroundExamProvenance]:
    """Upgrade the tier-B questions in ``questions`` to tier C where possible.

    ``corpus`` maps document id -> full text (the same ids used in each
    question's ``supporting_doc_ids``). A tier-B question is upgraded only when a
    verbatim span is extracted and ``verify_source_facts``-confirmed for *every*
    supporting document; otherwise it is returned unchanged as tier B. Tier-A and
    tier-C questions pass through untouched. Order and count are preserved.
    """
    semaphore = asyncio.Semaphore(concurrency)
    tier_b = [q for q in questions if q.grounding_tier == "B"]
    candidates = await asyncio.gather(
        *(
            _tier_c_candidate(q, corpus, model=extractor_model, reasoning_effort=reasoning_effort, semaphore=semaphore)
            for q in tier_b
        )
    )
    buildable = [c for c in candidates if c is not None]
    grounded = verify_source_facts(buildable, corpus, fuzzy_threshold)
    grounded_by_id = {q.id: q for q in grounded}

    out = [grounded_by_id.get(q.id, q) for q in questions]

    n_already_c = sum(1 for q in questions if q.grounding_tier == "C")
    n_tier_a = sum(1 for q in questions if q.grounding_tier == "A")
    provenance = GroundExamProvenance(
        n_input=len(questions),
        n_upgraded_to_c=len(grounded_by_id),
        n_kept_b=len(tier_b) - len(grounded_by_id),
        n_already_c=n_already_c,
        n_tier_a=n_tier_a,
        extractor_model=extractor_model,
        fuzzy_threshold=fuzzy_threshold,
    )
    return out, provenance


def write_grounded_exam(
    exam: list[OpenEndedQuestion],
    provenance: GroundExamProvenance,
    output_path: Path,
) -> Path:
    """Write the grounded exam JSON and a sibling ``<stem>_provenance.json``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([q.model_dump(mode="json") for q in exam], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    prov_path = output_path.with_name(output_path.stem + "_provenance.json")
    prov_path.write_text(provenance.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return prov_path
