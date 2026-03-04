"""Tests for LightRAGStore wrapper.

All LightRAG internals are mocked — no real LLM or embedding calls are made.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_autorag.config.models import GraphBuildConfig
from agentic_autorag.engine.graph_store import _GRAPH_MARKER_FILES, LightRAGStore


def _make_build_config(**kwargs) -> GraphBuildConfig:
    defaults = {"extraction_model": "gemini/gemini-2.5-flash-lite"}
    defaults.update(kwargs)
    return GraphBuildConfig(**defaults)


def _make_store(tmp_path: Path, **config_kwargs) -> LightRAGStore:
    return LightRAGStore(
        working_dir=tmp_path / "lightrag",
        build_config=_make_build_config(**config_kwargs),
    )


class TestIsBuilt:
    def test_returns_false_when_working_dir_is_empty(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert not store.is_built()

    def test_returns_true_when_all_marker_files_exist(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        wd = tmp_path / "lightrag"
        wd.mkdir(parents=True, exist_ok=True)
        for fname in _GRAPH_MARKER_FILES:
            (wd / fname).write_text("", encoding="utf-8")

        assert store.is_built()

    def test_returns_false_when_only_some_markers_exist(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        wd = tmp_path / "lightrag"
        wd.mkdir(parents=True, exist_ok=True)
        # Write only the first marker file
        (wd / _GRAPH_MARKER_FILES[0]).write_text("", encoding="utf-8")

        assert not store.is_built()


class TestBuild:
    @pytest.mark.asyncio
    async def test_skips_build_when_already_built(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        # Pre-create marker files so is_built() returns True
        wd = tmp_path / "lightrag"
        wd.mkdir(parents=True, exist_ok=True)
        for fname in _GRAPH_MARKER_FILES:
            (wd / fname).write_text("", encoding="utf-8")

        mock_rag = AsyncMock()
        store._rag = mock_rag

        await store.build(["doc1", "doc2"])

        mock_rag.ainsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_ainsert_when_not_built(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        mock_rag = AsyncMock()
        store._rag = mock_rag

        await store.build(["doc1", "doc2"])

        mock_rag.ainsert.assert_called_once_with(["doc1", "doc2"])


class TestQuery:
    @pytest.mark.asyncio
    async def test_returns_docs_on_success(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        mock_rag = AsyncMock()
        mock_rag.aquery_data = AsyncMock(
            return_value={
                "status": "success",
                "message": "ok",
                "data": {
                    "chunks": [
                        {"content": "chunk text A", "chunk_id": "c1"},
                        {"content": "chunk text B", "chunk_id": "c2"},
                    ],
                    "entities": [],
                    "relationships": [],
                },
            }
        )
        store._rag = mock_rag

        docs = await store.query("what is photovoltaics?", mode="hybrid", top_k=60)

        assert len(docs) == 2
        assert docs[0]["id"] == "c1"
        assert docs[0]["text"] == "chunk text A"
        assert docs[0]["score"] > docs[1]["score"]  # first chunk ranked higher
        mock_rag.aquery_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_empty_on_failure_status(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        mock_rag = AsyncMock()
        mock_rag.aquery_data = AsyncMock(
            return_value={
                "status": "failure",
                "message": "empty result",
                "data": {},
            }
        )
        store._rag = mock_rag

        docs = await store.query("anything")

        assert docs == []

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        # _rag is None (not initialized)
        with pytest.raises(RuntimeError, match="initialize"):
            await store.query("test")

    @pytest.mark.asyncio
    async def test_passes_mode_and_top_k_to_aquery_data(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        mock_rag = AsyncMock()
        mock_rag.aquery_data = AsyncMock(
            return_value={
                "status": "success",
                "data": {"chunks": [], "entities": [], "relationships": []},
            }
        )
        store._rag = mock_rag

        await store.query("q", mode="local", top_k=40)

        call_args = mock_rag.aquery_data.call_args
        param = call_args.args[1]  # QueryParam is the second positional arg
        assert param.mode == "local"
        assert param.top_k == 40


class TestNormaliseResult:
    def test_chunks_come_first_with_highest_scores(self) -> None:
        data = {
            "chunks": [
                {"content": "main text", "chunk_id": "c0"},
            ],
            "entities": [
                {"entity_name": "RAG", "description": "retrieval augmented generation"},
            ],
            "relationships": [
                {"src_id": "RAG", "tgt_id": "LLM", "description": "RAG uses LLM"},
            ],
        }

        docs = LightRAGStore._normalise_result(data)

        # Chunk, entity, relation — all three present
        assert len(docs) == 3
        ids = [d["id"] for d in docs]
        assert "c0" in ids
        assert "lgentity_RAG" in ids

        # Chunk should have the highest score
        chunk_score = next(d["score"] for d in docs if d["id"] == "c0")
        entity_score = next(d["score"] for d in docs if d["id"] == "lgentity_RAG")
        assert chunk_score > entity_score

    def test_empty_data_returns_empty_list(self) -> None:
        docs = LightRAGStore._normalise_result({})
        assert docs == []

    def test_skips_empty_content(self) -> None:
        data = {
            "chunks": [{"content": "", "chunk_id": "c0"}],
            "entities": [{"entity_name": "E", "description": ""}],
            "relationships": [],
        }
        docs = LightRAGStore._normalise_result(data)
        assert docs == []


class TestLlmFunc:
    @pytest.mark.asyncio
    async def test_wraps_litellm_correctly(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "extracted entities"

        llm_func = LightRAGStore._make_llm_func("gemini/test-model")

        with patch(
            "agentic_autorag.engine.graph_store.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_llm:
            result = await llm_func("extract entities from this text", system_prompt="You are helpful.")

        assert result == "extracted entities"
        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["model"] == "gemini/test-model"
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
