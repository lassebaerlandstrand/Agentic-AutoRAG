"""Tests for agentic_autorag.engine.pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agentic_autorag.config.models import IndexType, RuntimeConfig
from agentic_autorag.engine.pipeline import RAGPipeline, RetrievalResult, RetrievalTiming, RetrievedDocument


def _make_doc(doc_id: str, text: str = "doc text", score: float = 0.9) -> dict:
    """Return a minimal raw document dict (as returned by LanceDB / graph_store)."""
    return {"id": doc_id, "text": text, "score": score}


def _mock_embedder():
    """Return a mock embedder whose `encode` returns a fixed numpy vector."""
    embedder = MagicMock()
    embedder.encode = MagicMock(return_value=np.zeros(384))
    return embedder


def _default_config(**overrides) -> RuntimeConfig:
    defaults = {"llm_model": "test/model"}
    defaults.update(overrides)
    return RuntimeConfig(**defaults)


def _pipeline(
    *,
    index_type: IndexType = IndexType.VECTOR_ONLY,
    config: RuntimeConfig | None = None,
    vector_store: MagicMock | None = None,
    graph_store: MagicMock | None = None,
    embedder: MagicMock | None = None,
    cross_encoder: MagicMock | None = None,
) -> RAGPipeline:
    return RAGPipeline(
        vector_store=vector_store or MagicMock(),
        graph_store=graph_store,
        config=config or _default_config(),
        embedder=embedder or _mock_embedder(),
        index_type=index_type,
        cross_encoder=cross_encoder,
    )


class TestRetrieveVectorOnly:
    async def test_returns_documents(self):
        vs = MagicMock()
        vs.search_vector = MagicMock(return_value=[_make_doc("a"), _make_doc("b")])
        pipe = _pipeline(vector_store=vs, config=_default_config(top_k=2))

        result = await pipe.retrieve("hello")

        assert isinstance(result, RetrievalResult)
        assert len(result.documents) == 2
        assert all(isinstance(d, RetrievedDocument) for d in result.documents)
        vs.search_vector.assert_called_once()

    async def test_timing_populated(self):
        vs = MagicMock()
        vs.search_vector = MagicMock(return_value=[_make_doc("a")])
        pipe = _pipeline(vector_store=vs, config=_default_config(top_k=2))

        result = await pipe.retrieve("hello")

        assert isinstance(result.timing, RetrievalTiming)
        assert result.timing.total_s >= 0
        assert result.timing.embed_search_s >= 0

    async def test_respects_top_k(self):
        vs = MagicMock()
        vs.search_vector = MagicMock(return_value=[_make_doc(f"d{i}") for i in range(10)])
        pipe = _pipeline(vector_store=vs, config=_default_config(top_k=3))

        result = await pipe.retrieve("q")

        assert len(result.documents) == 3


class TestRetrieveHybridBM25:
    async def test_dispatches_to_hybrid(self):
        vs = MagicMock()
        vs.search_hybrid = MagicMock(return_value=[_make_doc("h1")])
        pipe = _pipeline(
            index_type=IndexType.HYBRID_BM25_VECTOR,
            vector_store=vs,
            config=_default_config(top_k=5, hybrid_alpha=0.7),
        )

        result = await pipe.retrieve("query")

        assert len(result.documents) == 1
        vs.search_hybrid.assert_called_once()
        assert vs.search_hybrid.call_args.kwargs["hybrid_alpha"] == 0.7


class TestRetrieveGraphOnly:
    async def test_dispatches_to_graph_store(self):
        gs = AsyncMock()
        gs.query = AsyncMock(return_value=[_make_doc("g1"), _make_doc("g2")])
        pipe = _pipeline(
            index_type=IndexType.GRAPH_ONLY,
            graph_store=gs,
            config=_default_config(top_k=5, graph_query_mode="hybrid", graph_top_k=60),
        )

        result = await pipe.retrieve("graph query")

        assert len(result.documents) == 2
        gs.query.assert_called_once_with("graph query", mode="hybrid", top_k=60)

    async def test_returns_empty_when_no_graph_store(self):
        pipe = _pipeline(
            index_type=IndexType.GRAPH_ONLY,
            graph_store=None,
            config=_default_config(top_k=5),
        )

        result = await pipe.retrieve("no graph")

        assert len(result.documents) == 0


class TestRetrieveHybridGraphVector:
    async def test_merges_vector_and_graph_results(self):
        vs = MagicMock()
        vs.search_hybrid = MagicMock(return_value=[_make_doc("v1"), _make_doc("v2")])
        gs = AsyncMock()
        gs.query = AsyncMock(return_value=[_make_doc("g1"), _make_doc("g2")])

        pipe = _pipeline(
            index_type=IndexType.HYBRID_GRAPH_VECTOR,
            vector_store=vs,
            graph_store=gs,
            config=_default_config(top_k=5, hybrid_alpha=0.3, graph_query_mode="hybrid", graph_top_k=60),
        )

        result = await pipe.retrieve("hybrid")

        # All 4 unique docs should be returned (< top_k=5).
        assert len(result.documents) == 4
        vs.search_hybrid.assert_called_once()
        gs.query.assert_called_once_with("hybrid", mode="hybrid", top_k=60)
        assert vs.search_hybrid.call_args.kwargs["hybrid_alpha"] == 0.3


class TestDeduplication:
    async def test_removes_duplicate_ids(self):
        """When query expansion returns multiple queries, duplicates are removed."""
        vs = MagicMock()
        # Both calls return the same doc.
        vs.search_vector = MagicMock(return_value=[_make_doc("dup")])

        config = _default_config(top_k=5, query_expansion="hyde")
        pipe = _pipeline(vector_store=vs, config=config)

        with patch.object(
            pipe,
            "generate",
            new_callable=AsyncMock,
            return_value=("hypothetical", {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}),
        ):
            result = await pipe.retrieve("q")

        # Two queries (original + HyDE), but the duplicate should be collapsed.
        assert len(result.documents) == 1
        assert result.documents[0].id == "dup"


class TestReranking:
    def test_requires_cross_encoder_when_reranker_enabled(self):
        config = _default_config(
            reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

        with pytest.raises(ValueError, match="cross_encoder"):
            _pipeline(config=config)

    async def test_fetches_more_candidates_and_truncates(self):
        """When reranking is active, fetch top_k*3, rerank, return reranker_top_n."""
        vs = MagicMock()
        docs = [_make_doc(f"d{i}") for i in range(15)]
        vs.search_vector = MagicMock(return_value=docs)

        mock_ce = MagicMock()
        mock_ce.predict = MagicMock(return_value=list(range(15)))

        config = _default_config(
            top_k=5,
            reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
            reranker_top_n=3,
        )
        pipe = _pipeline(vector_store=vs, config=config, cross_encoder=mock_ce)

        result = await pipe.retrieve("rerank me")

        assert len(result.documents) == 3
        # The vector store should have been asked for top_k*3 = 15 candidates.
        call_args = vs.search_vector.call_args
        assert call_args.kwargs.get("top_k") == 15


def _mock_response(content: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> MagicMock:
    """Build a litellm-shaped response mock with usable .choices and .usage."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


class TestGenerate:
    async def test_calls_litellm_and_returns_content_and_cost(self):
        pipe = _pipeline(config=_default_config(llm_model="ollama/llama3.2", temperature=0.1))

        with (
            patch(
                "agentic_autorag.litellm_runtime.litellm.acompletion",
                new_callable=AsyncMock,
                return_value=_mock_response("answer", prompt_tokens=10, completion_tokens=4),
            ) as mock_llm,
            patch(
                "agentic_autorag.litellm_runtime.litellm.completion_cost",
                return_value=0.0123,
            ),
        ):
            content, cost = await pipe.generate("prompt text")

        assert content == "answer"
        assert cost == {"usd": 0.0123, "prompt_tokens": 10, "completion_tokens": 4}
        mock_llm.assert_called_once_with(
            model="ollama/llama3.2",
            messages=[{"role": "user", "content": "prompt text"}],
            temperature=0.1,
            num_retries=0,
            timeout=100.0,
        )

    async def test_passes_reasoning_effort_when_reasoning_enabled(self):
        config = _default_config(
            llm_model="vertex_ai/gemini-2.5-flash",
            temperature=0.0,
            reasoning=True,
            reasoning_effort="high",
        )
        pipe = _pipeline(config=config)

        with (
            patch(
                "agentic_autorag.litellm_runtime.litellm.acompletion",
                new_callable=AsyncMock,
                return_value=_mock_response("reasoned answer"),
            ) as mock_llm,
            patch("agentic_autorag.litellm_runtime.litellm.completion_cost", return_value=0.0),
        ):
            content, _ = await pipe.generate("complex question")

        assert content == "reasoned answer"
        mock_llm.assert_called_once_with(
            model="vertex_ai/gemini-2.5-flash",
            messages=[{"role": "user", "content": "complex question"}],
            temperature=0.0,
            num_retries=0,
            timeout=100.0,
            reasoning_effort="high",
        )

    async def test_no_reasoning_effort_when_reasoning_disabled(self):
        config = _default_config(
            llm_model="vertex_ai/gemini-2.5-flash",
            reasoning=False,
        )
        pipe = _pipeline(config=config)

        with (
            patch(
                "agentic_autorag.litellm_runtime.litellm.acompletion",
                new_callable=AsyncMock,
                return_value=_mock_response("answer"),
            ) as mock_llm,
            patch("agentic_autorag.litellm_runtime.litellm.completion_cost", return_value=0.0),
        ):
            await pipe.generate("simple question")

        call_kwargs = mock_llm.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs

    async def test_cost_falls_back_to_zero_on_completion_cost_error(self):
        """When LiteLLM has no pricing for a model, cost gracefully degrades to 0."""
        pipe = _pipeline(config=_default_config(llm_model="ollama/local-model"))

        with (
            patch(
                "agentic_autorag.litellm_runtime.litellm.acompletion",
                new_callable=AsyncMock,
                return_value=_mock_response("answer"),
            ),
            patch(
                "agentic_autorag.litellm_runtime.litellm.completion_cost",
                side_effect=Exception("no pricing for model"),
            ),
        ):
            _, cost = await pipe.generate("prompt")

        assert cost["usd"] == 0.0


class TestExpandQuery:
    async def test_none_returns_original(self):
        pipe = _pipeline(config=_default_config(query_expansion="none"))

        queries, cost = await pipe._expand_query("hello")

        assert queries == ["hello"]
        assert cost == {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    async def test_hyde_returns_two_queries(self):
        pipe = _pipeline(config=_default_config(query_expansion="hyde"))

        gen_cost = {"usd": 0.001, "prompt_tokens": 5, "completion_tokens": 7}
        with patch.object(
            pipe,
            "generate",
            new_callable=AsyncMock,
            return_value=("hypothetical answer", gen_cost),
        ):
            queries, cost = await pipe._expand_query("question")

        assert len(queries) == 2
        assert queries[0] == "question"
        assert queries[1] == "hypothetical answer"
        assert cost == gen_cost

    async def test_multi_query_returns_up_to_four(self):
        pipe = _pipeline(config=_default_config(query_expansion="multi_query"))

        rephrasings = "rephrasing 1\nrephrasing 2\nrephrasing 3\nrephrasing 4"
        with patch.object(
            pipe,
            "generate",
            new_callable=AsyncMock,
            return_value=(rephrasings, {"usd": 0.002, "prompt_tokens": 8, "completion_tokens": 16}),
        ):
            queries, cost = await pipe._expand_query("original")

        # original + 3 rephrasings (cap at 3)
        assert len(queries) == 4
        assert queries[0] == "original"
        assert cost["usd"] == 0.002


class TestRRFMerge:
    def test_merges_two_lists(self):
        list_a = [_make_doc("a"), _make_doc("b")]
        list_b = [_make_doc("b"), _make_doc("c")]

        merged = RAGPipeline._rrf_merge(list_a, list_b, k=60)

        ids = [d["id"] for d in merged]
        # "b" appears in both lists, so it should have the highest fused score.
        assert ids[0] == "b"
        assert set(ids) == {"a", "b", "c"}

    def test_empty_lists(self):
        assert RAGPipeline._rrf_merge([], []) == []

    def test_one_empty_list(self):
        list_a = [_make_doc("x")]
        merged = RAGPipeline._rrf_merge(list_a, [])
        assert len(merged) == 1
        assert merged[0]["id"] == "x"

    def test_preserves_doc_data(self):
        doc = _make_doc("z", text="important", score=0.99)
        merged = RAGPipeline._rrf_merge([doc], [])
        assert merged[0]["text"] == "important"
