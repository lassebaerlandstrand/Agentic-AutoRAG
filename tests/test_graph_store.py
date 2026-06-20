"""Tests for LightRAGStore wrapper.

All LightRAG internals are mocked — no real LLM or embedding calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from agentic_autorag.config.models import GraphBuildConfig
from agentic_autorag.engine.graph_store import (
    MANIFEST_FILENAME,
    LightRAGStore,
    _is_retryable_error,
)

CORPUS_HASH = "abc123def4567890"


def _make_build_config(**kwargs) -> GraphBuildConfig:
    defaults = {"extraction_model": "gemini/gemini-2.5-flash-lite"}
    defaults.update(kwargs)
    return GraphBuildConfig(**defaults)


def _make_store(tmp_path: Path, **config_kwargs) -> LightRAGStore:
    return LightRAGStore(
        working_dir=tmp_path / "lightrag",
        build_config=_make_build_config(**config_kwargs),
    )


def _write_manifest(store: LightRAGStore, **fields) -> None:
    store.working_dir.mkdir(parents=True, exist_ok=True)
    store.manifest_path.write_text(json.dumps(fields), encoding="utf-8")


class TestIsBuilt:
    def test_returns_false_when_no_manifest(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert not store.is_built(CORPUS_HASH)

    def test_returns_true_when_manifest_matches(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _write_manifest(
            store,
            status="complete",
            corpus_hash=CORPUS_HASH,
            build_config_hash=store._build_config.config_hash(),
        )
        assert store.is_built(CORPUS_HASH)

    def test_returns_false_when_status_in_progress(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _write_manifest(
            store,
            status="in_progress",
            corpus_hash=CORPUS_HASH,
            build_config_hash=store._build_config.config_hash(),
        )
        assert not store.is_built(CORPUS_HASH)

    def test_returns_false_when_corpus_hash_differs(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _write_manifest(
            store,
            status="complete",
            corpus_hash="different_corpus",
            build_config_hash=store._build_config.config_hash(),
        )
        assert not store.is_built(CORPUS_HASH)

    def test_returns_false_when_config_hash_differs(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _write_manifest(
            store,
            status="complete",
            corpus_hash=CORPUS_HASH,
            build_config_hash="different_config",
        )
        assert not store.is_built(CORPUS_HASH)

    def test_corrupt_manifest_is_treated_as_absent(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.working_dir.mkdir(parents=True, exist_ok=True)
        store.manifest_path.write_text("not json{", encoding="utf-8")
        assert not store.is_built(CORPUS_HASH)


class TestBuild:
    @pytest.mark.asyncio
    async def test_skips_build_when_manifest_says_complete(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _write_manifest(
            store,
            status="complete",
            corpus_hash=CORPUS_HASH,
            build_config_hash=store._build_config.config_hash(),
            n_documents_total=2,
            elapsed_s=1.2,
        )
        mock_rag = AsyncMock()
        store._rag = mock_rag

        await store.build(["doc1", "doc2"], CORPUS_HASH)

        # Skip path: the complete manifest is left untouched and no docs are
        # re-inserted. "No insert" is the behavior; it has no other output.
        manifest = json.loads(store.manifest_path.read_text())
        assert manifest["status"] == "complete"
        assert manifest["n_documents_total"] == 2
        mock_rag.ainsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_builds_when_no_manifest(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, build_batch_size=20)
        mock_rag = AsyncMock()
        store._rag = mock_rag

        await store.build(["doc1", "doc2"], CORPUS_HASH)

        # build() has no return value; the docs forwarded to the (mocked) store
        # are its observable effect, alongside the manifest written below.
        mock_rag.ainsert.assert_called_once_with(["doc1", "doc2"])
        manifest = json.loads(store.manifest_path.read_text())
        assert manifest["status"] == "complete"
        assert manifest["corpus_hash"] == CORPUS_HASH
        assert manifest["n_documents_total"] == 2
        assert manifest["n_documents_inserted"] == 2
        assert manifest["inserted_doc_indices"] == [0, 1]
        assert manifest["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_initial_manifest_written_before_first_batch(self, tmp_path: Path) -> None:
        """If the very first batch crashes, a manifest with status=in_progress still exists."""
        store = _make_store(tmp_path, build_batch_size=2)
        mock_rag = AsyncMock()
        mock_rag.ainsert = AsyncMock(side_effect=RuntimeError("died on first batch"))
        store._rag = mock_rag

        with pytest.raises(RuntimeError, match="died on first batch"):
            await store.build(["d0", "d1", "d2"], CORPUS_HASH)

        manifest = json.loads(store.manifest_path.read_text())
        assert manifest["status"] == "in_progress"
        assert manifest["inserted_doc_indices"] == []
        assert manifest["n_documents_total"] == 3
        assert manifest["corpus_hash"] == CORPUS_HASH

    @pytest.mark.asyncio
    async def test_batches_large_document_lists(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, build_batch_size=3)
        mock_rag = AsyncMock()
        store._rag = mock_rag

        docs = [f"doc{i}" for i in range(7)]
        await store.build(docs, CORPUS_HASH)

        # 7 docs, batch_size=3 → 3 batches of [3, 3, 1]. The per-batch doc lists
        # forwarded to the (mocked) store are build()'s observable effect.
        batch_sizes = [len(call.args[0]) for call in mock_rag.ainsert.call_args_list]
        assert batch_sizes == [3, 3, 1]

    @pytest.mark.asyncio
    async def test_resumes_from_partial_manifest(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, build_batch_size=2)
        _write_manifest(
            store,
            status="in_progress",
            corpus_hash=CORPUS_HASH,
            build_config_hash=store._build_config.config_hash(),
            n_documents_total=5,
            n_documents_inserted=2,
            inserted_doc_indices=[0, 1],
            started_at="2026-04-15T00:00:00+00:00",
            batch_size=2,
            extraction_model="gemini/gemini-2.5-flash-lite",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            elapsed_s=50.0,
        )
        mock_rag = AsyncMock()
        store._rag = mock_rag

        docs = [f"doc{i}" for i in range(5)]
        await store.build(docs, CORPUS_HASH)

        # Only docs 2, 3, 4 should be inserted (2 batches: [doc2, doc3], [doc4]).
        # The exact per-batch doc lists are build()'s observable effect on resume.
        inserted = [call.args[0] for call in mock_rag.ainsert.call_args_list]
        assert inserted == [["doc2", "doc3"], ["doc4"]]

        manifest = json.loads(store.manifest_path.read_text())
        assert manifest["status"] == "complete"
        assert manifest["inserted_doc_indices"] == [0, 1, 2, 3, 4]
        assert manifest["elapsed_s"] >= 50.0  # prior elapsed preserved

    @pytest.mark.asyncio
    async def test_raises_on_corpus_hash_mismatch(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _write_manifest(
            store,
            status="complete",
            corpus_hash="old_corpus_hash",
            build_config_hash=store._build_config.config_hash(),
            n_documents_total=3,
        )
        mock_rag = AsyncMock()
        store._rag = mock_rag

        with pytest.raises(RuntimeError, match="different corpus or config"):
            await store.build(["d"], CORPUS_HASH)
        # Safety contract: we abort before inserting into a mismatched store.
        mock_rag.ainsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_on_config_hash_mismatch(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _write_manifest(
            store,
            status="in_progress",
            corpus_hash=CORPUS_HASH,
            build_config_hash="stale_config_hash",
            n_documents_total=3,
            inserted_doc_indices=[0],
        )
        mock_rag = AsyncMock()
        store._rag = mock_rag

        with pytest.raises(RuntimeError, match="different corpus or config"):
            await store.build(["d"], CORPUS_HASH)
        # Safety contract: we abort before inserting into a mismatched store.
        mock_rag.ainsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_manifest_written_after_each_batch(self, tmp_path: Path) -> None:
        """If ainsert fails on batch N, manifest reflects completion of batches [0, N-1]."""
        store = _make_store(tmp_path, build_batch_size=2)

        mock_rag = AsyncMock()

        async def ainsert_side_effect(batch):
            # Succeed on the first call, raise on the second.
            if mock_rag.ainsert.call_count == 2:
                raise RuntimeError("simulated crash")

        mock_rag.ainsert = AsyncMock(side_effect=ainsert_side_effect)
        store._rag = mock_rag

        docs = [f"doc{i}" for i in range(4)]
        with pytest.raises(RuntimeError, match="simulated crash"):
            await store.build(docs, CORPUS_HASH)

        manifest = json.loads(store.manifest_path.read_text())
        assert manifest["status"] == "in_progress"
        assert manifest["inserted_doc_indices"] == [0, 1]
        assert manifest["n_documents_inserted"] == 2


class TestManifestFilename:
    def test_manifest_path_lives_in_working_dir(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert store.manifest_path == store.working_dir / MANIFEST_FILENAME


class TestInitializeValidates:
    @pytest.mark.asyncio
    async def test_raises_on_mismatch_before_loading_models(self, tmp_path: Path) -> None:
        """A stale manifest must raise our clear error before SentenceTransformer / LightRAG load."""
        store = _make_store(tmp_path)
        _write_manifest(
            store,
            status="complete",
            corpus_hash="old_corpus",
            build_config_hash="old_config",
            n_documents_total=1,
        )

        # If validation fails early, these patches' targets should never be called.
        with (
            patch("agentic_autorag.engine.graph_store.SentenceTransformer") as mock_st,
            patch("agentic_autorag.engine.graph_store.LightRAG") as mock_rag_cls,
            pytest.raises(RuntimeError, match="different corpus or config"),
        ):
            await store.initialize(CORPUS_HASH)

        # The contract under test is "validate before loading heavy models";
        # the early exit has no output, so the not-called guard is the behavior.
        mock_st.assert_not_called()
        mock_rag_cls.assert_not_called()


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
        # graph_store prefixes chunk_ids with 'lgchunk_' so the evaluator can
        # identify verbatim graph chunks for offset lookup.
        assert docs[0]["id"] == "lgchunk_c1"
        assert docs[0]["text"] == "chunk text A"
        assert docs[0]["score"] > docs[1]["score"]

    @pytest.mark.asyncio
    async def test_returns_empty_on_failure_status(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        mock_rag = AsyncMock()
        mock_rag.aquery_data = AsyncMock(return_value={"status": "failure", "message": "empty result", "data": {}})
        store._rag = mock_rag

        docs = await store.query("anything")

        assert docs == []

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        with pytest.raises(RuntimeError, match="initialize"):
            await store.query("test")

    @pytest.mark.asyncio
    async def test_passes_mode_and_top_k_to_aquery_data(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        mock_rag = AsyncMock()
        mock_rag.aquery_data = AsyncMock(
            return_value={"status": "success", "data": {"chunks": [], "entities": [], "relationships": []}}
        )
        store._rag = mock_rag

        await store.query("q", mode="local", top_k=40)

        call_args = mock_rag.aquery_data.call_args
        param = call_args.args[1]
        assert param.mode == "local"
        assert param.top_k == 40


class TestNormaliseResult:
    def test_chunks_come_first_with_highest_scores(self) -> None:
        data = {
            "chunks": [{"content": "main text", "chunk_id": "c0"}],
            "entities": [{"entity_name": "RAG", "description": "retrieval augmented generation"}],
            "relationships": [{"src_id": "RAG", "tgt_id": "LLM", "description": "RAG uses LLM"}],
        }

        docs = LightRAGStore._normalise_result(data)

        assert len(docs) == 3
        ids = [d["id"] for d in docs]
        assert "lgchunk_c0" in ids
        assert "lgentity_RAG" in ids

        chunk_score = next(d["score"] for d in docs if d["id"] == "lgchunk_c0")
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


class TestRetryClassification:
    def test_400_is_not_retryable(self) -> None:
        exc = RuntimeError("bad request")
        exc.status_code = 400  # type: ignore[attr-defined]
        assert not _is_retryable_error(exc)

    def test_401_is_not_retryable(self) -> None:
        exc = RuntimeError("auth failed")
        exc.status_code = 401  # type: ignore[attr-defined]
        assert not _is_retryable_error(exc)

    def test_403_is_not_retryable(self) -> None:
        exc = RuntimeError("forbidden")
        exc.status_code = 403  # type: ignore[attr-defined]
        assert not _is_retryable_error(exc)

    def test_404_is_not_retryable(self) -> None:
        exc = RuntimeError("not found")
        exc.status_code = 404  # type: ignore[attr-defined]
        assert not _is_retryable_error(exc)

    def test_422_is_not_retryable(self) -> None:
        exc = RuntimeError("unprocessable")
        exc.status_code = 422  # type: ignore[attr-defined]
        assert not _is_retryable_error(exc)

    def test_429_is_retryable(self) -> None:
        exc = RuntimeError("rate limited")
        exc.status_code = 429  # type: ignore[attr-defined]
        assert _is_retryable_error(exc)

    def test_503_is_retryable(self) -> None:
        exc = RuntimeError("service unavailable")
        exc.status_code = 503  # type: ignore[attr-defined]
        assert _is_retryable_error(exc)

    def test_unknown_error_is_retryable(self) -> None:
        """Connection / timeout errors often don't set status_code."""
        assert _is_retryable_error(RuntimeError("connection reset"))


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

    @pytest.mark.asyncio
    async def test_fails_fast_on_non_retryable_error(self) -> None:
        """A 401 auth error should raise immediately, not sleep through retries."""
        llm_func = LightRAGStore._make_llm_func("gemini/test-model", num_retries=5)
        err = RuntimeError("bad auth")
        err.status_code = 401  # type: ignore[attr-defined]

        with (
            patch(
                "agentic_autorag.engine.graph_store.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=err,
            ) as mock_llm,
            pytest.raises(RuntimeError, match="bad auth"),
        ):
            await llm_func("prompt")

        # Called exactly once — no retries
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_retryable_error(self) -> None:
        """A 429 should trigger retries; a later success should return."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        err = RuntimeError("rate limited")
        err.status_code = 429  # type: ignore[attr-defined]

        llm_func = LightRAGStore._make_llm_func("gemini/test-model", num_retries=2)

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise err
            return mock_response

        # Patch asyncio.sleep so the test doesn't actually wait.
        with (
            patch(
                "agentic_autorag.engine.graph_store.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=side_effect,
            ),
            patch("agentic_autorag.engine.graph_store.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await llm_func("prompt")

        assert result == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_passes_timeout_to_acompletion(self) -> None:
        """The per-call timeout must reach litellm.acompletion so hangs are cut short."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        llm_func = LightRAGStore._make_llm_func("gemini/test-model", call_timeout_s=37.5)

        with patch(
            "agentic_autorag.engine.graph_store.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_llm:
            await llm_func("prompt")

        assert mock_llm.call_args.kwargs["timeout"] == 37.5

    @pytest.mark.asyncio
    async def test_caller_timeout_overrides_default(self) -> None:
        """If LightRAG passes its own timeout kwarg, we honor it rather than the default."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        llm_func = LightRAGStore._make_llm_func("gemini/test-model", call_timeout_s=45.0)

        with patch(
            "agentic_autorag.engine.graph_store.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_llm:
            await llm_func("prompt", timeout=10.0)

        assert mock_llm.call_args.kwargs["timeout"] == 10.0

    @pytest.mark.asyncio
    async def test_backoff_capped_and_jittered(self) -> None:
        """Every retry sleep must be ≤ backoff_max * 1.5 (jitter upper bound)."""
        err = RuntimeError("rate limited")
        err.status_code = 429  # type: ignore[attr-defined]

        llm_func = LightRAGStore._make_llm_func(
            "gemini/test-model",
            num_retries=3,
            backoff_base_s=5.0,
            backoff_max_s=30.0,
        )

        sleeps: list[float] = []

        async def record_sleep(wait: float) -> None:
            sleeps.append(wait)

        with (
            patch(
                "agentic_autorag.engine.graph_store.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=err,
            ),
            patch("agentic_autorag.engine.graph_store.asyncio.sleep", side_effect=record_sleep),
            pytest.raises(RuntimeError, match="rate limited"),
        ):
            await llm_func("prompt")

        # 3 retries → 3 sleeps (between attempts 1→2, 2→3, 3→4)
        assert len(sleeps) == 3
        upper_bound = 30.0 * 1.5  # backoff_max * max jitter
        for s in sleeps:
            assert 0.0 <= s <= upper_bound, f"sleep {s} outside [0, {upper_bound}]"

    @pytest.mark.asyncio
    async def test_does_not_use_litellm_internal_retries(self) -> None:
        """We must never ask LiteLLM to retry — those retries hold LightRAG's semaphore.

        Our explicit async loop is the only retry path.
        """
        err = RuntimeError("rate limited")
        err.status_code = 429  # type: ignore[attr-defined]

        llm_func = LightRAGStore._make_llm_func("gemini/test-model", num_retries=2)

        with (
            patch(
                "agentic_autorag.engine.graph_store.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=err,
            ) as mock_llm,
            patch("agentic_autorag.engine.graph_store.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError, match="rate limited"),
        ):
            await llm_func("prompt")

        # Our loop retries 3 times (initial + 2 retries); every call must be
        # clean of LiteLLM's own retry knobs.
        assert mock_llm.call_count == 3
        for call in mock_llm.call_args_list:
            assert "num_retries" not in call.kwargs
            assert "retry_policy" not in call.kwargs
            assert "max_retries" not in call.kwargs


class TestConfigHash:
    def test_stable_across_reads(self) -> None:
        cfg = _make_build_config()
        assert cfg.config_hash() == cfg.config_hash()

    def test_changes_with_extraction_model(self) -> None:
        a = _make_build_config(extraction_model="gemini/a")
        b = _make_build_config(extraction_model="gemini/b")
        assert a.config_hash() != b.config_hash()

    def test_changes_with_embedding_model(self) -> None:
        a = _make_build_config(embedding_model="sentence-transformers/m1")
        b = _make_build_config(embedding_model="sentence-transformers/m2")
        assert a.config_hash() != b.config_hash()

    def test_changes_with_entity_types(self) -> None:
        a = _make_build_config(entity_types=["person"])
        b = _make_build_config(entity_types=["person", "location"])
        assert a.config_hash() != b.config_hash()

    def test_unchanged_by_concurrency_knobs(self) -> None:
        a = _make_build_config(
            max_parallel_insert=2,
            llm_model_max_async=4,
            llm_model_max_retries=3,
            embedding_func_max_async=4,
        )
        b = _make_build_config(
            max_parallel_insert=8,
            llm_model_max_async=16,
            llm_model_max_retries=0,
            embedding_func_max_async=16,
        )
        assert a.config_hash() == b.config_hash()

    def test_unchanged_by_build_batch_size(self) -> None:
        a = _make_build_config(build_batch_size=10)
        b = _make_build_config(build_batch_size=50)
        assert a.config_hash() == b.config_hash()

    def test_unchanged_by_embedding_batch_size(self) -> None:
        a = _make_build_config(embedding_batch_size=32)
        b = _make_build_config(embedding_batch_size=256)
        assert a.config_hash() == b.config_hash()

    def test_unchanged_by_timeout_knobs(self) -> None:
        a = _make_build_config(
            default_llm_timeout=180,
            default_embedding_timeout=30,
            extraction_call_timeout_s=45.0,
            extraction_retry_backoff_base_s=5.0,
            extraction_retry_backoff_max_s=30.0,
        )
        b = _make_build_config(
            default_llm_timeout=600,
            default_embedding_timeout=120,
            extraction_call_timeout_s=90.0,
            extraction_retry_backoff_base_s=1.0,
            extraction_retry_backoff_max_s=10.0,
        )
        assert a.config_hash() == b.config_hash()


class TestRetryBudgetValidator:
    def test_accepts_default_budget(self) -> None:
        cfg = _make_build_config()
        # Worst-case: 4 * 45 + 1.5 * (5+10+20) = 232.5 < 360
        attempts = cfg.llm_model_max_retries + 1
        sleeps = sum(
            min(cfg.extraction_retry_backoff_base_s * 2**i, cfg.extraction_retry_backoff_max_s)
            for i in range(cfg.llm_model_max_retries)
        )
        budget = cfg.extraction_call_timeout_s * attempts + sleeps * 1.5
        assert budget < cfg.default_llm_timeout * 2

    def test_rejects_retries_overflowing_worker_cap(self) -> None:
        with pytest.raises(ValidationError, match="retry budget"):
            _make_build_config(llm_model_max_retries=20)

    def test_rejects_per_call_timeout_overflowing_worker_cap(self) -> None:
        with pytest.raises(ValidationError, match="retry budget"):
            _make_build_config(extraction_call_timeout_s=500.0)

    def test_rejects_low_llm_timeout(self) -> None:
        with pytest.raises(ValidationError, match="retry budget"):
            _make_build_config(default_llm_timeout=30)


class TestInitializePassesKwargs:
    @pytest.mark.asyncio
    async def test_initialize_passes_new_lightrag_kwargs(self, tmp_path: Path) -> None:
        store = _make_store(
            tmp_path,
            default_llm_timeout=150,
            default_embedding_timeout=45,
            embedding_func_max_async=6,
            max_parallel_insert=3,
            llm_model_max_async=5,
        )
        mock_embedder = MagicMock()
        mock_embedder.get_sentence_embedding_dimension.return_value = 384
        mock_rag_instance = MagicMock()
        mock_rag_instance.initialize_storages = AsyncMock()

        with (
            patch("agentic_autorag.engine.graph_store.SentenceTransformer", return_value=mock_embedder),
            patch("agentic_autorag.engine.graph_store.LightRAG", return_value=mock_rag_instance) as mock_rag_cls,
        ):
            await store.initialize(CORPUS_HASH)

        kwargs = mock_rag_cls.call_args.kwargs
        assert kwargs["default_llm_timeout"] == 150
        assert kwargs["default_embedding_timeout"] == 45
        assert kwargs["embedding_func_max_async"] == 6
        assert kwargs["max_parallel_insert"] == 3
        assert kwargs["llm_model_max_async"] == 5
