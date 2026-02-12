"""Tests for agentic_autorag.engine.pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from agentic_autorag.config.models import IndexType, RuntimeConfig
from agentic_autorag.engine.pipeline import RAGPipeline, RetrievalResult, RetrievedDocument


def _make_doc(doc_id: str, text: str = "doc text", score: float = 0.9) -> dict:
    """Return a minimal raw document dict (as returned by LanceDB / graph_store)."""
    return {"id": doc_id, "text": text, "score": score}


def _mock_embedder():
    """Return a mock embedder whose `encode` returns a fixed numpy vector."""
    embedder = MagicMock()
    embedder.encode = MagicMock(return_value=np.zeros(384))
    return embedder


def _default_config(**overrides) -> RuntimeConfig:
    return RuntimeConfig(**overrides)


def _pipeline(
    *,
    index_type: IndexType = IndexType.VECTOR_ONLY,
    config: RuntimeConfig | None = None,
    vector_store: MagicMock | None = None,
    graph_store: MagicMock | None = None,
    embedder: MagicMock | None = None,
) -> RAGPipeline:
    return RAGPipeline(
        vector_store=vector_store or MagicMock(),
        graph_store=graph_store,
        config=config or _default_config(),
        embedder=embedder or _mock_embedder(),
        index_type=index_type,
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
            config=_default_config(top_k=5),
        )

        result = await pipe.retrieve("query")

        assert len(result.documents) == 1
        vs.search_hybrid.assert_called_once()


class TestRetrieveGraphOnly:
    async def test_dispatches_to_graph_store(self):
        gs = AsyncMock()
        gs.search = AsyncMock(return_value=[_make_doc("g1"), _make_doc("g2")])
        pipe = _pipeline(
            index_type=IndexType.GRAPH,
            graph_store=gs,
            config=_default_config(top_k=5),
        )

        result = await pipe.retrieve("graph query")

        assert len(result.documents) == 2
        gs.search.assert_called_once()

    async def test_returns_empty_when_no_graph_store(self):
        pipe = _pipeline(
            index_type=IndexType.GRAPH,
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
        gs.search = AsyncMock(return_value=[_make_doc("g1"), _make_doc("g2")])

        pipe = _pipeline(
            index_type=IndexType.HYBRID_GRAPH_VECTOR,
            vector_store=vs,
            graph_store=gs,
            config=_default_config(top_k=5),
        )

        result = await pipe.retrieve("hybrid")

        # All 4 unique docs should be returned (< top_k=5).
        assert len(result.documents) == 4
        vs.search_hybrid.assert_called_once()
        gs.search.assert_called_once()


class TestDeduplication:
    async def test_removes_duplicate_ids(self):
        """When query expansion returns multiple queries, duplicates are removed."""
        vs = MagicMock()
        # Both calls return the same doc.
        vs.search_vector = MagicMock(return_value=[_make_doc("dup")])

        config = _default_config(top_k=5, query_expansion="hyde")
        pipe = _pipeline(vector_store=vs, config=config)

        with patch.object(pipe, "generate", new_callable=AsyncMock, return_value="hypothetical"):
            result = await pipe.retrieve("q")

        # Two queries (original + HyDE), but the duplicate should be collapsed.
        assert len(result.documents) == 1
        assert result.documents[0].id == "dup"


class TestReranking:
    async def test_fetches_more_candidates_and_truncates(self):
        """When reranking is active, fetch top_k*3, rerank, return reranker_top_n."""
        vs = MagicMock()
        docs = [_make_doc(f"d{i}") for i in range(15)]
        vs.search_vector = MagicMock(return_value=docs)

        config = _default_config(
            top_k=5,
            reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
            reranker_top_n=3,
        )
        pipe = _pipeline(vector_store=vs, config=config)

        # Mock the cross-encoder so we don't download a model.
        mock_ce = MagicMock()
        mock_ce.predict = MagicMock(return_value=list(range(15)))
        pipe._cross_encoder = mock_ce

        result = await pipe.retrieve("rerank me")

        assert len(result.documents) == 3
        # The vector store should have been asked for top_k*3 = 15 candidates.
        call_args = vs.search_vector.call_args
        assert call_args.kwargs.get("top_k") == 15


class TestGenerate:
    async def test_calls_litellm_and_returns_content(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "answer"

        pipe = _pipeline(config=_default_config(llm_model="ollama/llama3.2", temperature=0.1))

        with patch(
            "agentic_autorag.engine.pipeline.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_llm:
            result = await pipe.generate("prompt text")

        assert result == "answer"
        mock_llm.assert_called_once_with(
            model="ollama/llama3.2",
            messages=[{"role": "user", "content": "prompt text"}],
            temperature=0.1,
        )


class TestExpandQuery:
    async def test_none_returns_original(self):
        pipe = _pipeline(config=_default_config(query_expansion="none"))

        result = await pipe._expand_query("hello")

        assert result == ["hello"]

    async def test_hyde_returns_two_queries(self):
        pipe = _pipeline(config=_default_config(query_expansion="hyde"))

        with patch.object(pipe, "generate", new_callable=AsyncMock, return_value="hypothetical answer"):
            result = await pipe._expand_query("question")

        assert len(result) == 2
        assert result[0] == "question"
        assert result[1] == "hypothetical answer"

    async def test_multi_query_returns_up_to_four(self):
        pipe = _pipeline(config=_default_config(query_expansion="multi_query"))

        rephrasings = "rephrasing 1\nrephrasing 2\nrephrasing 3\nrephrasing 4"
        with patch.object(pipe, "generate", new_callable=AsyncMock, return_value=rephrasings):
            result = await pipe._expand_query("original")

        # original + 3 rephrasings (cap at 3)
        assert len(result) == 4
        assert result[0] == "original"


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
