"""Tests for the index builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer

from agentic_autorag.config.models import IndexType, StructuralConfig
from agentic_autorag.cost_ledger import CostLedger, reset_active_ledger, set_active_ledger
from agentic_autorag.engine.index_builder import IndexBuilder, IngredientCache

TEST_TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
    embedding_model: str = TEST_TOKENIZER_MODEL,
) -> StructuralConfig:
    return StructuralConfig(
        chunking_strategy=chunking_strategy,
        chunk_token_size=chunk_token_size,
        chunk_token_overlap=chunk_token_overlap,
        embedding_model=embedding_model,
        index_type=index_type,
    )


@pytest.fixture(scope="module")
def real_tokenizer():
    tok = AutoTokenizer.from_pretrained(TEST_TOKENIZER_MODEL)
    tok.model_max_length = 10**7
    return tok


class DummyEmbeddingModel:
    """Deterministic stand-in for SentenceTransformer.

    Exposes a real fast tokenizer so the offset-mapping chunker has something
    valid to work with, while the embeddings themselves are trivial keyword
    counts — cheap, deterministic, and good enough for retrieval tests.
    """

    _shared_tokenizer = None

    def __init__(self, model_name: str = "", **kwargs):
        self.model_name = model_name
        if DummyEmbeddingModel._shared_tokenizer is None:
            DummyEmbeddingModel._shared_tokenizer = AutoTokenizer.from_pretrained(TEST_TOKENIZER_MODEL)
            DummyEmbeddingModel._shared_tokenizer.model_max_length = 10**7
        self.tokenizer = DummyEmbeddingModel._shared_tokenizer
        self.max_seq_length = 512

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
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
def builder() -> IndexBuilder:
    return IndexBuilder(table_name="chunks")


@pytest.fixture(scope="module")
def embedder() -> DummyEmbeddingModel:
    return DummyEmbeddingModel()


@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    from unittest.mock import patch

    with patch("agentic_autorag.engine.index_builder.SentenceTransformer", new=DummyEmbeddingModel):
        yield


def _max_tokens(chunks: list[str], tokenizer) -> int:
    return max((len(tokenizer.encode(c, add_special_tokens=False)) for c in chunks), default=0)


class TestIndexBuilder:
    @pytest.mark.asyncio
    async def test_build_recursive_chunking_respects_token_budget(self, builder: IndexBuilder, real_tokenizer) -> None:
        documents = _make_documents()
        config = _make_config(chunk_token_size=40, chunk_token_overlap=8, chunking_strategy="recursive")

        index = await builder.build(documents, config, corpus_hash="test")

        assert len(index.chunks) > len(documents)
        assert index.embeddings.shape[0] == len(index.chunks)
        assert index.embeddings.shape[1] > 0
        assert _max_tokens(index.chunks, real_tokenizer) <= config.chunk_token_size

    @pytest.mark.asyncio
    async def test_build_index_and_search_returns_relevant_chunks(
        self,
        builder: IndexBuilder,
        embedder: DummyEmbeddingModel,
    ) -> None:
        documents = _make_documents()
        config = _make_config(chunk_token_size=60, chunk_token_overlap=8, chunking_strategy="recursive")
        query = "How do photovoltaic panels turn sunlight into electricity?"
        query_embedding = np.asarray(embedder.encode([query])[0], dtype=np.float32)

        index = await builder.build(documents, config, corpus_hash="test")
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
        small_config = _make_config(chunk_token_size=30, chunk_token_overlap=4, chunking_strategy="recursive")
        large_config = _make_config(chunk_token_size=120, chunk_token_overlap=4, chunking_strategy="recursive")

        small_index = await builder.build(documents, small_config, corpus_hash="test")
        large_index = await builder.build(documents, large_config, corpus_hash="test")

        assert len(small_index.chunks) > len(large_index.chunks)

    @pytest.mark.asyncio
    async def test_graph_index_type_builds_without_graph_store(self, builder: IndexBuilder) -> None:
        """Building a graph-typed index only creates the vector side; graph_store is None."""
        documents = _make_documents()
        config = _make_config(chunk_token_size=60, chunk_token_overlap=8, index_type=IndexType.GRAPH_ONLY)

        index = await builder.build(documents, config, corpus_hash="test")

        assert index.index_type == IndexType.GRAPH_ONLY
        assert len(index.chunks) > 0
        # The graph store is always None after build — orchestrator attaches it later.
        assert index.graph_store is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("chunk_size", "overlap", "strategy"),
        [
            (30, 0, "recursive"),
            (30, 8, "recursive"),
            (80, 12, "recursive"),
            (30, 8, "fixed"),
            (80, 0, "fixed"),
        ],
    )
    async def test_token_budget_never_exceeded(
        self, builder: IndexBuilder, real_tokenizer, chunk_size: int, overlap: int, strategy: str
    ) -> None:
        documents = _make_documents()
        config = _make_config(chunk_token_size=chunk_size, chunk_token_overlap=overlap, chunking_strategy=strategy)
        index = await builder.build(documents, config, corpus_hash=f"tb-{chunk_size}-{overlap}-{strategy}")
        assert _max_tokens(index.chunks, real_tokenizer) <= chunk_size

    @pytest.mark.asyncio
    async def test_overlap_is_applied_in_separator_merge_path(self, builder: IndexBuilder, real_tokenizer) -> None:
        """Adjacent chunks produced via separator merging should share close to chunk_overlap tokens."""
        # Use a doc with many well-defined paragraph breaks so the separator-merge
        # path (not the hard-split fallback) is exercised.
        paragraph = "This is a self-contained sentence about photovoltaic energy conversion. "
        doc = "\n\n".join(paragraph * 3 for _ in range(40))
        documents = [doc]
        config = _make_config(chunk_token_size=50, chunk_token_overlap=12, chunking_strategy="recursive")

        index = await builder.build(documents, config, corpus_hash="overlap-test")
        assert len(index.chunks) >= 2

        # Check adjacent chunks share at least (overlap - 2) tokens at the boundary,
        # allowing a small margin for token-realignment when a chunk is re-tokenized
        # after char slicing.
        min_shared = config.chunk_token_overlap - 2
        any_overlap_found = False
        for i in range(len(index.chunks) - 1):
            t_prev = real_tokenizer.encode(index.chunks[i], add_special_tokens=False)
            t_next = real_tokenizer.encode(index.chunks[i + 1], add_special_tokens=False)
            longest = 0
            for k in range(1, min(len(t_prev), len(t_next), config.chunk_token_size) + 1):
                if t_prev[-k:] == t_next[:k]:
                    longest = k
            if longest >= min_shared:
                any_overlap_found = True
                break
        assert any_overlap_found, "no adjacent chunk pair carried the requested token overlap"

    @pytest.mark.asyncio
    async def test_missing_fast_tokenizer_raises(self, builder: IndexBuilder) -> None:
        documents = _make_documents()
        config = _make_config(chunk_token_size=40)

        class NoTokenizer:
            def __init__(self, *a, **kw):
                pass

            def encode(self, texts, **kw):
                return np.zeros((len(texts), 4), dtype=np.float32)

        from unittest.mock import patch

        with patch("agentic_autorag.engine.index_builder.SentenceTransformer", new=NoTokenizer):
            local_builder = IndexBuilder(table_name="chunks")
            with pytest.raises(ValueError, match="fast HuggingFace tokenizer"):
                await local_builder.build(documents, config, corpus_hash="nt")


class TestIngredientCache:
    @pytest.mark.asyncio
    async def test_full_hit_on_second_build(self, tmp_path: Path) -> None:
        cache = IngredientCache(tmp_path / "cache", max_bytes=10**9)
        builder = IndexBuilder(cache=cache, table_name="chunks")
        documents = _make_documents()
        config = _make_config(chunk_token_size=50)

        first = await builder.build(documents, config, corpus_hash="c")
        chunks_fp = config.chunks_fingerprint("c")
        emb_fp = config.embeddings_fingerprint("c")
        assert cache.has_chunks(chunks_fp)
        assert cache.has_embeddings(emb_fp)

        second = await builder.build(documents, config, corpus_hash="c")
        assert second.chunks == first.chunks
        np.testing.assert_array_equal(second.embeddings, first.embeddings)

    @pytest.mark.asyncio
    async def test_chunks_are_reused_across_embedding_models(self, tmp_path: Path) -> None:
        """Same chunking config + different embedding model → chunks cache hit, embeddings rebuilt."""
        cache = IngredientCache(tmp_path / "cache", max_bytes=10**9)
        builder = IndexBuilder(cache=cache, table_name="chunks")
        documents = _make_documents()
        config_a = _make_config(chunk_token_size=50, embedding_model=TEST_TOKENIZER_MODEL)
        # Pretend a second embedding model by changing the string — DummyEmbeddingModel
        # ignores the model name.
        config_b = _make_config(chunk_token_size=50, embedding_model="pretend/other-embedding-model")

        await builder.build(documents, config_a, corpus_hash="c")
        chunks_fp = config_a.chunks_fingerprint("c")
        emb_fp_a = config_a.embeddings_fingerprint("c")
        emb_fp_b = config_b.embeddings_fingerprint("c")
        assert config_a.chunks_fingerprint("c") == config_b.chunks_fingerprint("c")
        assert emb_fp_a != emb_fp_b

        # After the second build with a different embedding model, chunks entry is
        # shared and both embeddings entries exist.
        await builder.build(documents, config_b, corpus_hash="c")
        assert cache.has_chunks(chunks_fp)
        assert cache.has_embeddings(emb_fp_a)
        assert cache.has_embeddings(emb_fp_b)

    @pytest.mark.asyncio
    async def test_index_type_variants_share_one_cache_entry(self, tmp_path: Path) -> None:
        """VECTOR_ONLY and HYBRID_BM25_VECTOR produce the same embeddings fingerprint."""
        cache = IngredientCache(tmp_path / "cache", max_bytes=10**9)
        builder = IndexBuilder(cache=cache, table_name="chunks")
        documents = _make_documents()

        vector_config = _make_config(chunk_token_size=50, index_type=IndexType.VECTOR_ONLY)
        hybrid_config = _make_config(chunk_token_size=50, index_type=IndexType.HYBRID_BM25_VECTOR)

        assert vector_config.embeddings_fingerprint("c") == hybrid_config.embeddings_fingerprint("c")

        await builder.build(documents, vector_config, corpus_hash="c")
        await builder.build(documents, hybrid_config, corpus_hash="c")

        # One chunks entry + one embeddings entry = two manifest keys.
        assert len(cache.manifest) == 2

    @pytest.mark.asyncio
    async def test_eviction_prefers_embeddings_over_chunks(self, tmp_path: Path) -> None:
        """Under budget pressure, embeddings entries evict before chunks."""
        cache = IngredientCache(tmp_path / "cache", max_bytes=1)  # force eviction on every store
        builder = IndexBuilder(cache=cache, table_name="chunks")
        documents = _make_documents()
        config_a = _make_config(chunk_token_size=50, embedding_model=TEST_TOKENIZER_MODEL)
        config_b = _make_config(chunk_token_size=50, embedding_model="pretend/other-embedding-model")

        await builder.build(documents, config_a, corpus_hash="c")
        # After the second store, only config_b's entries are protected; config_a's
        # embeddings can be evicted but its chunks are still referenced by b.
        await builder.build(documents, config_b, corpus_hash="c")

        chunks_fp = config_a.chunks_fingerprint("c")
        emb_fp_b = config_b.embeddings_fingerprint("c")
        assert cache.has_chunks(chunks_fp), "chunks entry must survive while an embeddings entry references it"
        assert cache.has_embeddings(emb_fp_b)

    @pytest.mark.asyncio
    async def test_corpus_hash_isolates_cache_entries(self, tmp_path: Path) -> None:
        cache = IngredientCache(tmp_path / "cache", max_bytes=10**9)
        builder = IndexBuilder(cache=cache, table_name="chunks")
        documents = _make_documents()
        config = _make_config(chunk_token_size=50)

        await builder.build(documents, config, corpus_hash="corpus_a")
        await builder.build(documents, config, corpus_hash="corpus_b")

        assert cache.has_chunks(config.chunks_fingerprint("corpus_a"))
        assert cache.has_chunks(config.chunks_fingerprint("corpus_b"))
        assert cache.has_embeddings(config.embeddings_fingerprint("corpus_a"))
        assert cache.has_embeddings(config.embeddings_fingerprint("corpus_b"))

    @pytest.mark.asyncio
    async def test_doc_set_isolates_cache_entries(self, tmp_path: Path) -> None:
        """A cache built over one doc-id set must NOT be hit by a different set.

        Guards the corruption where two corpus loaders disagreed on the document
        set (e.g. 17626 vs 17629) yet shared one cache key, so cached doc_indices
        were positionally misresolved against a longer doc_ids list. A doc-id set
        differing in count must miss (forcing a correct rebuild), never hit.
        """
        cache = IngredientCache(tmp_path / "cache", max_bytes=10**9)
        builder = IndexBuilder(cache=cache, table_name="chunks")
        documents = _make_documents()
        config = _make_config(chunk_token_size=50)
        doc_ids = [f"doc_{i}" for i in range(len(documents))]

        await builder.build(documents, config, corpus_hash="c", doc_ids=doc_ids)

        longer = doc_ids + ["doc_extra"]
        assert not cache.has_chunks(config.chunks_fingerprint("c", longer))
        assert not cache.has_embeddings(config.embeddings_fingerprint("c", longer))
        # The exact set that built the cache still hits.
        assert cache.has_chunks(config.chunks_fingerprint("c", doc_ids))
        assert cache.has_embeddings(config.embeddings_fingerprint("c", doc_ids))


def test_doc_set_fingerprint_distinguishes_doc_sets() -> None:
    """chunks/embeddings fingerprints are sensitive to the indexed doc-id set
    (count, order, presence) — the property that makes a subset-built cache
    unhittable by a different doc-id universe."""
    config = _make_config(chunk_token_size=50)
    base = ["a", "b", "c"]
    assert config.chunks_fingerprint("h", base) == config.chunks_fingerprint("h", base)
    assert config.chunks_fingerprint("h", base) != config.chunks_fingerprint("h", base + ["d"])
    assert config.chunks_fingerprint("h", base) != config.chunks_fingerprint("h", ["c", "b", "a"])
    assert config.chunks_fingerprint("h", base) != config.chunks_fingerprint("h", None)
    assert config.embeddings_fingerprint("h", base) != config.embeddings_fingerprint("h", base + ["d"])


class TestEmbeddingTokenAccounting:
    """First-use-per-(method, seed) cache credit + meta sidecar persistence."""

    @pytest.mark.asyncio
    async def test_meta_sidecar_records_deterministic_token_count(self, tmp_path: Path) -> None:
        cache = IngredientCache(tmp_path / "cache", max_bytes=10**9)
        builder = IndexBuilder(cache=cache, table_name="chunks")
        documents = _make_documents()
        config = _make_config(chunk_token_size=50)

        index = await builder.build(documents, config, corpus_hash="c")

        emb_fp = config.embeddings_fingerprint("c")
        meta = cache.load_embeddings_meta(emb_fp)
        assert meta is not None
        assert meta["embedding_input_tokens"] == index.embedding_input_tokens
        assert meta["embedding_input_tokens"] > 0
        assert meta["n_chunks"] == len(index.chunks)
        assert meta["embedding_model"] == config.embedding_model
        assert index.emb_fp == emb_fp
        assert index.embedding_model == config.embedding_model

    @pytest.mark.asyncio
    async def test_cache_hit_recovers_token_count_from_sidecar(self, tmp_path: Path) -> None:
        cache = IngredientCache(tmp_path / "cache", max_bytes=10**9)
        builder = IndexBuilder(cache=cache, table_name="chunks")
        documents = _make_documents()
        config = _make_config(chunk_token_size=50)

        first = await builder.build(documents, config, corpus_hash="c")
        second = await builder.build(documents, config, corpus_hash="c")

        assert first.emb_fp == second.emb_fp
        assert first.embedding_input_tokens == second.embedding_input_tokens
        assert second.embedding_input_tokens > 0

    @pytest.mark.asyncio
    async def test_missing_meta_sidecar_raises_loudly(self, tmp_path: Path) -> None:
        cache = IngredientCache(tmp_path / "cache", max_bytes=10**9)
        builder = IndexBuilder(cache=cache, table_name="chunks")
        documents = _make_documents()
        config = _make_config(chunk_token_size=50)

        await builder.build(documents, config, corpus_hash="c")
        emb_fp = config.embeddings_fingerprint("c")
        cache._embeddings_meta_path(emb_fp).unlink()

        with pytest.raises(RuntimeError, match="meta.json sidecar"):
            cache.load_embeddings_meta(emb_fp)

    @pytest.mark.asyncio
    async def test_first_use_rule_credits_ledger_once_per_run(self, tmp_path: Path) -> None:
        """Simulate (method=random, seed=0) seeing the same emb_fp across three trials."""
        cache = IngredientCache(tmp_path / "cache", max_bytes=10**9)
        builder = IndexBuilder(cache=cache, table_name="chunks")
        documents = _make_documents()
        config = _make_config(chunk_token_size=50)

        ledger = CostLedger()
        token = set_active_ledger(ledger)
        seen: set[str] = set()
        try:
            for _ in range(3):
                index = await builder.build(documents, config, corpus_hash="c")
                if index.emb_fp not in seen:
                    seen.add(index.emb_fp)
                    ledger.record(
                        "embedding_build",
                        usd=0.0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        embedding_input_tokens=index.embedding_input_tokens,
                    )
        finally:
            reset_active_ledger(token)

        bucket = ledger.buckets["embedding_build"]
        assert bucket.n_calls == 1, "first-use rule must credit exactly once per (method, seed) per emb_fp"
        assert bucket.embedding_input_tokens > 0

    @pytest.mark.asyncio
    async def test_first_use_rule_per_seed_resets_credit(self, tmp_path: Path) -> None:
        """A fresh ledger + fresh ``seen`` set (= new seed) re-credits the same cache key."""
        cache = IngredientCache(tmp_path / "cache", max_bytes=10**9)
        builder = IndexBuilder(cache=cache, table_name="chunks")
        documents = _make_documents()
        config = _make_config(chunk_token_size=50)

        # Seed 0
        ledger_seed0 = CostLedger()
        token0 = set_active_ledger(ledger_seed0)
        seen0: set[str] = set()
        try:
            index0 = await builder.build(documents, config, corpus_hash="c")
            seen0.add(index0.emb_fp)
            ledger_seed0.record(
                "embedding_build",
                usd=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                embedding_input_tokens=index0.embedding_input_tokens,
            )
        finally:
            reset_active_ledger(token0)

        # Seed 1 — same cache, different (method, seed) ledger + seen set
        ledger_seed1 = CostLedger()
        token1 = set_active_ledger(ledger_seed1)
        seen1: set[str] = set()
        try:
            index1 = await builder.build(documents, config, corpus_hash="c")
            if index1.emb_fp not in seen1:
                seen1.add(index1.emb_fp)
                ledger_seed1.record(
                    "embedding_build",
                    usd=0.0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    embedding_input_tokens=index1.embedding_input_tokens,
                )
        finally:
            reset_active_ledger(token1)

        assert ledger_seed0.buckets["embedding_build"].embedding_input_tokens > 0
        assert ledger_seed1.buckets["embedding_build"].embedding_input_tokens > 0
        assert (
            ledger_seed0.buckets["embedding_build"].embedding_input_tokens
            == ledger_seed1.buckets["embedding_build"].embedding_input_tokens
        ), "different (method, seed) runs must each pay the full deterministic token cost"
