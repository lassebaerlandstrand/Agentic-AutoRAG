"""Tests for the index builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agentic_autorag.config.models import IndexType, StructuralConfig
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
    chunk_token_size: int,
    chunk_token_overlap: int = 20,
    chunking_strategy: str = "recursive",
    index_type: IndexType = IndexType.VECTOR_ONLY,
) -> StructuralConfig:
    return StructuralConfig(
        chunking_strategy=chunking_strategy,
        chunk_token_size=chunk_token_size,
        chunk_token_overlap=chunk_token_overlap,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_type=index_type,
    )


class DummyEmbeddingModel:
    def __init__(self, model_name: str = "", **kwargs):
        self.model_name = model_name

    def encode(self, texts: list[str], **kwargs):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    float(lower.count("photovoltaic") + lower.count("solar")),
                    float(lower.count("electric")),
                    float(lower.count("sunlight") + lower.count("energy")),
                    1.0,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


@pytest.fixture(scope="module")
def db_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("index_builder")


@pytest.fixture(scope="module")
def builder(db_root: Path) -> IndexBuilder:
    return IndexBuilder(db_path=db_root / "lancedb", table_name="chunks")


@pytest.fixture(scope="module")
def embedder() -> DummyEmbeddingModel:
    return DummyEmbeddingModel()


@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    from unittest.mock import patch

    with patch("agentic_autorag.engine.index_builder.SentenceTransformer", new=DummyEmbeddingModel):
        yield


class TestIndexBuilder:
    @pytest.mark.asyncio
    async def test_build_recursive_chunking_creates_reasonable_chunks(self, builder: IndexBuilder) -> None:
        documents = _make_documents()
        config = _make_config(chunk_token_size=140, chunk_token_overlap=20, chunking_strategy="recursive")

        index = await builder.build(documents, config)

        assert len(index.chunks) > len(documents)
        assert index.embeddings.shape[0] == len(index.chunks)
        assert index.embeddings.shape[1] > 0
        assert max(len(chunk) for chunk in index.chunks) <= config.chunk_token_size + 40

    @pytest.mark.asyncio
    async def test_build_index_and_search_returns_relevant_chunks(
        self,
        builder: IndexBuilder,
        embedder: DummyEmbeddingModel,
    ) -> None:
        documents = _make_documents()
        config = _make_config(chunk_token_size=180, chunk_token_overlap=20, chunking_strategy="recursive")
        query = "How do photovoltaic panels turn sunlight into electricity?"
        query_embedding = np.asarray(embedder.encode([query])[0], dtype=np.float32)

        index = await builder.build(documents, config)
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
        small_config = _make_config(chunk_token_size=110, chunk_token_overlap=20, chunking_strategy="recursive")
        large_config = _make_config(chunk_token_size=280, chunk_token_overlap=20, chunking_strategy="recursive")

        small_index = await builder.build(documents, small_config)
        large_index = await builder.build(documents, large_config)

        assert len(small_index.chunks) > len(large_index.chunks)

    @pytest.mark.asyncio
    async def test_graph_index_type_builds_without_graph_store(
        self,
        builder: IndexBuilder,
    ) -> None:
        """Building a graph-typed index only creates the vector side; graph_store is None."""
        documents = _make_documents()
        config = _make_config(chunk_token_size=180, chunk_token_overlap=20, index_type=IndexType.GRAPH_ONLY)

        index = await builder.build(documents, config)

        assert index.index_type == IndexType.GRAPH_ONLY
        assert len(index.chunks) > 0
        # The graph store is always None after build — orchestrator attaches it later.
        assert index.graph_store is None
