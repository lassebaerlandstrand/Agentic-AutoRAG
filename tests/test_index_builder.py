"""Tests for the index builder."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from agentic_autorag.config.models import GraphConfig, IndexType, StructuralConfig
from agentic_autorag.engine.index_builder import IndexBuilder


def _make_documents() -> list[str]:
    return [
        (
            "Solar photovoltaic panels convert sunlight into electrical energy using semiconductor cells. "
            "Engineers monitor panel angle and irradiance to improve power output in rooftop systems.\n\n"
            "In grid-connected installations, an inverter converts direct current to alternating current. "
            "Well-designed systems pair battery storage with forecasting to reduce evening demand spikes."
        ),
        (
            "Database indexing improves lookup speed by organizing keys for fast retrieval paths. "
            "B-tree indexes support range filters while hash indexes target exact matches.\n\n"
            "When index maintenance is ignored, write amplification increases and query latency drifts upward. "
            "Careful schema design balances read efficiency with update costs."
        ),
        (
            "Sourdough bread fermentation uses wild yeast and lactic acid bacteria to develop flavor. "
            "Dough hydration, proof timing, and oven spring control crust and crumb structure.\n\n"
            "Bakers often score the loaf to guide expansion and improve heat transfer in the first bake phase."
        ),
        (
            "Wetland ecosystems provide habitat for migratory birds and filter runoff before it reaches rivers. "
            "Conservation plans track biodiversity, nutrient loading, and seasonal flood patterns.\n\n"
            "Long-term restoration combines native planting with monitoring to stabilize water quality."
        ),
    ]


def _make_config(
    *,
    chunk_size: int,
    chunk_overlap: int = 20,
    chunking_strategy: str = "recursive",
    index_type: IndexType = IndexType.VECTOR_ONLY,
) -> StructuralConfig:
    return StructuralConfig(
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_type=index_type,
    )


@pytest.fixture(scope="module")
def db_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("index_builder")


@pytest.fixture(scope="module")
def builder(db_root: Path) -> IndexBuilder:
    return IndexBuilder(db_path=db_root / "lancedb", table_name="chunks")


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class TestIndexBuilder:
    @pytest.mark.asyncio
    async def test_build_recursive_chunking_creates_reasonable_chunks(self, builder: IndexBuilder) -> None:
        documents = _make_documents()
        config = _make_config(chunk_size=140, chunk_overlap=20, chunking_strategy="recursive")

        index = await builder.build(documents, config)

        assert len(index.chunks) > len(documents)
        assert index.embeddings.shape[0] == len(index.chunks)
        assert index.embeddings.shape[1] > 0
        assert max(len(chunk) for chunk in index.chunks) <= config.chunk_size + 40

    @pytest.mark.asyncio
    async def test_build_index_and_search_returns_relevant_chunks(
        self,
        builder: IndexBuilder,
        embedder: SentenceTransformer,
    ) -> None:
        documents = _make_documents()
        config = _make_config(chunk_size=180, chunk_overlap=20, chunking_strategy="recursive")

        index = await builder.build(documents, config)
        query = "How do photovoltaic panels turn sunlight into electricity?"
        query_embedding = np.asarray(embedder.encode(query), dtype=np.float32)

        results = index.vector_store.search_hybrid("photovoltaic sunlight electricity", query_embedding, top_k=3)

        assert len(results) > 0
        assert any(
            ("solar" in row["text"].lower() or "photovoltaic" in row["text"].lower())
            and "electric" in row["text"].lower()
            for row in results
        )

    @pytest.mark.asyncio
    async def test_different_chunk_sizes_produce_different_chunk_counts(self, builder: IndexBuilder) -> None:
        documents = _make_documents()
        small_config = _make_config(chunk_size=110, chunk_overlap=20, chunking_strategy="recursive")
        large_config = _make_config(chunk_size=280, chunk_overlap=20, chunking_strategy="recursive")

        small_index = await builder.build(documents, small_config)
        large_index = await builder.build(documents, large_config)

        assert len(small_index.chunks) > len(large_index.chunks)

    @pytest.mark.asyncio
    async def test_graph_index_type_logs_warning_and_does_not_crash(
        self,
        builder: IndexBuilder,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        documents = _make_documents()
        config = _make_config(chunk_size=180, chunk_overlap=20, index_type=IndexType.GRAPH)

        with caplog.at_level(logging.WARNING):
            index = await builder.build(documents, config, graph_config=GraphConfig())

        assert index.index_type == IndexType.GRAPH
        assert len(index.chunks) > 0
        assert "graph indexing is not implemented yet" in caplog.text.lower()
