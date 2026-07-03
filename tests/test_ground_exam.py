"""Tests for the tier-B -> tier-C exam grounder (LLM span extraction mocked)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agentic_autorag.config.models import OpenEndedQuestion
from agentic_autorag.examiner.ground_exam import ground_exam

# Every groundable document contains this exact substring; the mocked extractor
# always returns it, so a question upgrades iff all its supporting docs contain it.
GROUNDABLE_SPAN = "GROUNDABLE evidence sentence"


def _completion_response(content: str):
    """Mimic the (response, cost) tuple returned by acompletion_with_cost."""
    message = SimpleNamespace(content=content)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    return response, {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}


def _mock_extractor() -> AsyncMock:
    return AsyncMock(return_value=_completion_response(json.dumps({"span": GROUNDABLE_SPAN})))


def _corpus(n_groundable: int, n_ungroundable: int) -> dict[str, str]:
    docs = {f"g{i}": f"lead {GROUNDABLE_SPAN} trailing {i}" for i in range(n_groundable)}
    docs.update({f"u{i}": f"lead nothing relevant trailing {i}" for i in range(n_ungroundable)})
    return docs


def _tier_b(qid: str, doc_ids: list[str], *, reasoning_type: str | None = None) -> OpenEndedQuestion:
    return OpenEndedQuestion(
        id=qid,
        question=f"q {qid}?",
        canonical_answer=f"answer {qid}",
        supporting_doc_ids=doc_ids,
        reasoning_type=reasoning_type,
    )


@pytest.mark.asyncio
async def test_tier_b_upgrades_to_c() -> None:
    questions = [_tier_b("q1", ["g0", "g1"])]
    corpus = _corpus(2, 0)
    with patch("agentic_autorag.examiner.ground_exam.acompletion_with_cost", _mock_extractor()):
        out, prov = await ground_exam(questions, corpus, extractor_model="test/model", concurrency=2)
    assert len(out) == 1
    q = out[0]
    assert q.grounding_tier == "C"
    assert q.source_spans == [GROUNDABLE_SPAN, GROUNDABLE_SPAN]
    assert q.source_doc_ids == ["g0", "g1"]
    assert q.supporting_doc_ids == ["g0", "g1"]  # doc lane carried alongside spans
    assert q.source_span_offsets and all(o is not None for o in q.source_span_offsets)
    assert prov.n_upgraded_to_c == 1
    assert prov.n_kept_b == 0


@pytest.mark.asyncio
async def test_ungroundable_stays_tier_b() -> None:
    # q2's second doc does not contain the span => verification fails => stays tier B.
    questions = [_tier_b("q1", ["g0"]), _tier_b("q2", ["g0", "u0"])]
    corpus = _corpus(1, 1)
    with patch("agentic_autorag.examiner.ground_exam.acompletion_with_cost", _mock_extractor()):
        out, prov = await ground_exam(questions, corpus, extractor_model="test/model", concurrency=2)
    assert len(out) == 2  # never drops
    by_id = {q.id: q for q in out}
    assert by_id["q1"].grounding_tier == "C"
    assert by_id["q2"].grounding_tier == "B"
    assert by_id["q2"].supporting_doc_ids == ["g0", "u0"]  # original doc lane intact
    assert prov.n_upgraded_to_c == 1
    assert prov.n_kept_b == 1


@pytest.mark.asyncio
async def test_missing_doc_stays_tier_b() -> None:
    questions = [_tier_b("q1", ["g0", "absent"])]  # 'absent' is not in the corpus
    corpus = _corpus(1, 0)
    with patch("agentic_autorag.examiner.ground_exam.acompletion_with_cost", _mock_extractor()):
        out, prov = await ground_exam(questions, corpus, extractor_model="test/model", concurrency=2)
    assert out[0].grounding_tier == "B"
    assert prov.n_kept_b == 1


@pytest.mark.asyncio
async def test_reasoning_type_preserved_on_upgrade() -> None:
    questions = [_tier_b("q1", ["g0"], reasoning_type="bridge")]
    corpus = _corpus(1, 0)
    with patch("agentic_autorag.examiner.ground_exam.acompletion_with_cost", _mock_extractor()):
        out, _ = await ground_exam(questions, corpus, extractor_model="test/model", concurrency=2)
    assert out[0].grounding_tier == "C"
    assert out[0].reasoning_type == "bridge"


@pytest.mark.asyncio
async def test_tier_a_and_c_pass_through_untouched() -> None:
    tier_a = OpenEndedQuestion(id="a1", question="q?", canonical_answer="x")
    tier_c = OpenEndedQuestion(
        id="c1",
        question="q?",
        canonical_answer="x",
        reasoning_type="bridge",
        source_doc_ids=["g0"],
        source_spans=[GROUNDABLE_SPAN],
    )
    corpus = _corpus(1, 0)
    extractor = _mock_extractor()
    with patch("agentic_autorag.examiner.ground_exam.acompletion_with_cost", extractor):
        out, prov = await ground_exam([tier_a, tier_c], corpus, extractor_model="test/model", concurrency=2)
    assert [q.id for q in out] == ["a1", "c1"]
    assert out[0].grounding_tier == "A"
    assert out[1].grounding_tier == "C"
    assert prov.n_tier_a == 1
    assert prov.n_already_c == 1
    assert prov.n_upgraded_to_c == 0
    extractor.assert_not_called()  # no tier-B questions => no extraction calls
