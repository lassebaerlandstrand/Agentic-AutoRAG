"""LightRAG knowledge graph store wrapper.

Handles building and querying the LightRAG graph. The graph is built once
and persisted to disk in LightRAG's ``working_dir``. A ``build_manifest.json``
alongside the graph records the corpus + build config it was built from;
subsequent runs verify the manifest matches before reusing the graph, and
resume a partial build when it does.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import litellm
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from agentic_autorag.config.models import GraphBuildConfig
from agentic_autorag.engine._io import atomic_write_text
from agentic_autorag.litellm_runtime import acompletion_with_cost

# Silence LiteLLM's "Give Feedback / Get Help" banners — we log retries ourselves.
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "build_manifest.json"

# HTTP status codes that should NOT trigger a retry — the request itself is wrong.
# Everything else (including no status code at all) is treated as retryable, since
# connection/timeout errors from LiteLLM typically don't expose a status code.
_NON_RETRYABLE_STATUS_CODES = frozenset({400, 401, 403, 404, 422})


def _is_retryable_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return not (isinstance(status, int) and status in _NON_RETRYABLE_STATUS_CODES)


class LightRAGStore:
    """Manages a LightRAG knowledge graph instance.

    Wraps LightRAG's async API and normalises its output into the standard
    retrieval dict format used by the rest of the engine.

    Persistence:  LightRAG writes all state (graph, vectors, KV cache) to
    ``working_dir``. Alongside that, this wrapper writes ``build_manifest.json``
    recording the corpus + config the graph was built from, plus per-batch
    progress so a crashed build resumes where it left off.
    """

    def __init__(
        self,
        working_dir: str | Path,
        build_config: GraphBuildConfig,
    ) -> None:
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self._build_config = build_config
        self._rag: LightRAG | None = None
        self._embedder: SentenceTransformer | None = None

    @property
    def manifest_path(self) -> Path:
        return self.working_dir / MANIFEST_FILENAME

    async def initialize(self, corpus_hash: str) -> None:
        """Create the LightRAG instance and initialise its storage backends.

        Validates the manifest BEFORE touching LightRAG's on-disk state or
        loading the embedding model, so a corpus/config mismatch raises our
        clear error instead of a deep ``AssertionError`` from LightRAG when
        it discovers (e.g.) a stale embedding dimension.
        """
        if self._rag is not None:
            return

        cfg = self._build_config

        manifest = self._read_manifest()
        if manifest is not None:
            self._validate_manifest_compatible(manifest, corpus_hash, cfg.config_hash())

        loop = asyncio.get_running_loop()
        self._embedder = await loop.run_in_executor(None, SentenceTransformer, cfg.embedding_model)
        embedding_dim = self._embedder.get_sentence_embedding_dimension()

        rag_kwargs: dict[str, Any] = {
            "working_dir": str(self.working_dir),
            "llm_model_func": self._make_llm_func(
                cfg.extraction_model,
                cfg.llm_model_max_retries,
                cfg.extraction_call_timeout_s,
                cfg.extraction_retry_backoff_base_s,
                cfg.extraction_retry_backoff_max_s,
            ),
            "llm_model_name": cfg.extraction_model,
            "llm_model_max_async": cfg.llm_model_max_async,
            "embedding_func": self._make_embedding_func(self._embedder, embedding_dim, cfg.embedding_batch_size),
            "embedding_func_max_async": cfg.embedding_func_max_async,
            "max_parallel_insert": cfg.max_parallel_insert,
            "default_llm_timeout": cfg.default_llm_timeout,
            "default_embedding_timeout": cfg.default_embedding_timeout,
        }

        if cfg.chunk_token_size is not None:
            rag_kwargs["chunk_token_size"] = cfg.chunk_token_size
        if cfg.chunk_overlap_token_size is not None:
            rag_kwargs["chunk_overlap_token_size"] = cfg.chunk_overlap_token_size
        if cfg.entity_types is not None:
            rag_kwargs["addon_params"] = {"entity_types": cfg.entity_types}

        self._rag = LightRAG(**rag_kwargs)
        await self._rag.initialize_storages()
        logger.info("LightRAGStore initialised (working_dir=%s)", self.working_dir)

    async def close(self) -> None:
        """Finalise storages and release resources."""
        if self._rag is not None:
            await self._rag.finalize_storages()
            self._rag = None

    def _read_manifest(self) -> dict | None:
        if not self.manifest_path.exists():
            return None
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt manifest at %s (%s); treating as absent", self.manifest_path, exc)
            return None

    def _write_manifest(self, data: dict) -> None:
        atomic_write_text(self.manifest_path, json.dumps(data, indent=2, sort_keys=True))

    def is_built(self, corpus_hash: str) -> bool:
        """Return True iff a completed graph for the given corpus exists on disk.

        Checks the manifest's ``status``, ``corpus_hash``, and ``build_config_hash``.
        A manifest mismatch returns False here and is reported loudly in ``build()``.
        """
        manifest = self._read_manifest()
        if not manifest or manifest.get("status") != "complete":
            return False
        return (
            manifest.get("corpus_hash") == corpus_hash
            and manifest.get("build_config_hash") == self._build_config.config_hash()
        )

    async def build(self, documents: list[str], corpus_hash: str) -> None:
        """Insert ``documents`` into LightRAG in batches, persisting progress.

        - If the manifest shows a completed build for the same corpus+config, returns immediately.
        - If it shows a completed or in-progress build for a *different* corpus/config, raises.
        - If it shows an in-progress build for the same corpus+config, resumes from there.
        - Otherwise, starts a fresh build.

        LightRAG deduplicates documents by content hash internally, so re-inserting
        a doc is a no-op — the manifest's ``inserted_doc_indices`` is authoritative
        for *our* progress tracking but safe even when LightRAG's state drifts.
        """
        self._assert_initialized()

        cfg = self._build_config
        cfg_hash = cfg.config_hash()
        manifest = self._read_manifest()

        if manifest is not None:
            self._validate_manifest_compatible(manifest, corpus_hash, cfg_hash)
            if manifest.get("status") == "complete":
                logger.info(
                    "Graph already built: %d docs in %.1fs (manifest at %s)",
                    manifest.get("n_documents_total", 0),
                    manifest.get("elapsed_s", 0.0),
                    self.manifest_path,
                )
                return

        inserted: set[int] = set(manifest.get("inserted_doc_indices", [])) if manifest else set()
        total = len(documents)
        batch_size = cfg.build_batch_size

        started_at = (manifest or {}).get("started_at") or datetime.now(UTC).isoformat()
        elapsed_prior = (manifest or {}).get("elapsed_s", 0.0)

        base_manifest = {
            "corpus_hash": corpus_hash,
            "build_config_hash": cfg_hash,
            "n_documents_total": total,
            "batch_size": batch_size,
            "started_at": started_at,
            "extraction_model": cfg.extraction_model,
            "embedding_model": cfg.embedding_model,
        }

        remaining = [(i, documents[i]) for i in range(total) if i not in inserted]
        if not remaining:
            self._finalize_manifest(base_manifest, inserted, elapsed_prior)
            logger.info("Graph build: all %d docs already inserted; finalised manifest.", total)
            return

        if inserted:
            logger.info(
                "Resuming graph build: %d/%d docs already inserted (batch_size=%d)",
                len(inserted),
                total,
                batch_size,
            )
        else:
            logger.info("Building graph: %d documents in batches of %d", total, batch_size)

        self._write_in_progress_manifest(base_manifest, inserted, elapsed_prior)

        t_start = time.monotonic()
        with tqdm(total=total, initial=len(inserted), unit="doc", desc="Graph build", smoothing=0.1) as pbar:
            for offset in range(0, len(remaining), batch_size):
                batch = remaining[offset : offset + batch_size]
                batch_indices = [i for i, _ in batch]
                batch_docs = [d for _, d in batch]

                t_batch = time.monotonic()
                await self._rag.ainsert(batch_docs)  # type: ignore[union-attr]
                batch_elapsed = time.monotonic() - t_batch

                inserted.update(batch_indices)
                elapsed_now = elapsed_prior + (time.monotonic() - t_start)
                self._write_in_progress_manifest(base_manifest, inserted, elapsed_now)

                pbar.update(len(batch_indices))
                eta = pbar.format_dict.get("remaining")
                eta_str = pbar.format_interval(eta) if isinstance(eta, (int, float)) else "?"
                logger.info(
                    "Graph build: %d/%d docs | batch %.1fs | ETA %s",
                    len(inserted),
                    total,
                    batch_elapsed,
                    eta_str,
                )

        total_elapsed = elapsed_prior + (time.monotonic() - t_start)
        self._finalize_manifest(base_manifest, inserted, total_elapsed)
        logger.info("Graph build complete: %d docs in %.1fs", total, total_elapsed)

    def _validate_manifest_compatible(self, manifest: dict, corpus_hash: str, cfg_hash: str) -> None:
        """Raise with a clear message when an existing manifest doesn't match current corpus/config."""
        m_corpus = manifest.get("corpus_hash")
        m_config = manifest.get("build_config_hash")
        if m_corpus == corpus_hash and m_config == cfg_hash:
            return
        raise RuntimeError(
            f"Graph at {self.working_dir} was built with a different corpus or config:\n"
            f"  manifest: corpus={m_corpus} config={m_config} status={manifest.get('status')}\n"
            f"  current : corpus={corpus_hash} config={cfg_hash}\n"
            f"Delete the directory to rebuild from scratch."
        )

    def _write_in_progress_manifest(self, base: dict, inserted: set[int], elapsed_s: float) -> None:
        self._write_manifest(
            {
                **base,
                "status": "in_progress",
                "n_documents_inserted": len(inserted),
                "inserted_doc_indices": sorted(inserted),
                "completed_at": None,
                "elapsed_s": elapsed_s,
            }
        )

    def _finalize_manifest(self, base: dict, inserted: set[int], elapsed_s: float) -> None:
        self._write_manifest(
            {
                **base,
                "status": "complete",
                "n_documents_inserted": len(inserted),
                "inserted_doc_indices": sorted(inserted),
                "completed_at": datetime.now(UTC).isoformat(),
                "elapsed_s": elapsed_s,
            }
        )

    async def query(self, query: str, mode: str = "hybrid", top_k: int = 60) -> list[dict]:
        """Query the graph and return results as standard retrieval dicts.

        Uses ``aquery_data`` which returns structured entities, relationships,
        and chunks without invoking an LLM — our pipeline handles generation.
        """
        self._assert_initialized()

        param = QueryParam(mode=mode, top_k=top_k, enable_rerank=False)
        result = await self._rag.aquery_data(query, param)  # type: ignore[union-attr]

        if result.get("status") != "success":
            logger.warning(
                "LightRAG query returned non-success status '%s': %s",
                result.get("status"),
                result.get("message"),
            )
            return []

        return self._normalise_result(result.get("data", {}))

    def _assert_initialized(self) -> None:
        if self._rag is None:
            raise RuntimeError("LightRAGStore.initialize() must be called before insert or query operations.")

    @staticmethod
    def _make_llm_func(
        model: str,
        num_retries: int = 3,
        call_timeout_s: float = 45.0,
        backoff_base_s: float = 5.0,
        backoff_max_s: float = 30.0,
    ):
        """Return a LightRAG-compatible async LLM function backed by LiteLLM.

        Retries on transient errors with capped, jittered exponential back-off.
        Fails fast on 4xx client errors so a misconfigured build doesn't waste
        minutes sleeping between doomed attempts.

        This retry loop is the *only* retry path: we deliberately do not pass
        ``num_retries`` / ``retry_policy`` to ``litellm.acompletion``. LiteLLM's
        internal retries would run inside LightRAG's worker semaphore without
        giving it back, deadlocking external concurrency control.
        """
        _log = logging.getLogger(__name__)

        async def llm_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict] | None = None,
            **kwargs: Any,
        ) -> str:
            messages: list[dict] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            allowed = {k: v for k, v in kwargs.items() if k in {"temperature", "max_tokens", "timeout"}}
            allowed.setdefault("timeout", call_timeout_s)

            last_exc: Exception | None = None
            for attempt in range(num_retries + 1):
                try:
                    response, _ = await acompletion_with_cost(
                        cost_category="graph_build",
                        model=model,
                        messages=messages,
                        **allowed,
                    )
                    return response.choices[0].message.content
                except Exception as exc:
                    last_exc = exc
                    if not _is_retryable_error(exc):
                        _log.error(
                            "LLM call failed with non-retryable error (%s): %s",
                            getattr(exc, "status_code", type(exc).__name__),
                            exc,
                        )
                        raise
                    if attempt < num_retries:
                        wait = min(backoff_base_s * 2**attempt, backoff_max_s) * random.uniform(0.5, 1.5)
                        _log.warning(
                            "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                            attempt + 1,
                            num_retries + 1,
                            exc,
                            wait,
                        )
                        await asyncio.sleep(wait)
            raise last_exc  # type: ignore[misc]

        return llm_func

    @staticmethod
    def _make_embedding_func(embedder: SentenceTransformer, embedding_dim: int, batch_size: int) -> EmbeddingFunc:
        """Return a LightRAG EmbeddingFunc backed by a local SentenceTransformer."""

        async def embed_func(texts: list[str]) -> np.ndarray:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: embedder.encode(texts, show_progress_bar=False, batch_size=batch_size),
            )

        return EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=embed_func,
        )

    @staticmethod
    def _normalise_result(data: dict) -> list[dict]:
        """Convert LightRAG's structured query result to our standard format.

        Combines text chunks (primary context), entity descriptions, and
        relationship descriptions into a ranked list of retrieval dicts.
        The ordering preserves LightRAG's internal ranking: chunks first
        (highest relevance), then entities, then relationships.

        For verbatim chunks, ``file_path`` is carried through so the evaluator
        can look up the source document and compute character offsets for
        interval-overlap chunk relevance. Prefixes on entity/relationship ids
        (``lgentity_``/``lgrel_``) let the evaluator route synthesized content
        to its n-gram fallback.
        """
        docs: list[dict] = []

        for i, chunk in enumerate(data.get("chunks", [])):
            content = chunk.get("content", "").strip()
            if not content:
                continue
            chunk_id = chunk.get("chunk_id", f"lgchunk_{i}")
            # Prefix plain LightRAG chunk_ids so the evaluator can detect
            # verbatim graph chunks reliably regardless of ID scheme.
            if not chunk_id.startswith("lgchunk_"):
                chunk_id = f"lgchunk_{chunk_id}"
            docs.append(
                {
                    "id": chunk_id,
                    "text": content,
                    "score": 1.0 / (i + 1),
                    "file_path": chunk.get("file_path", ""),
                }
            )

        for i, entity in enumerate(data.get("entities", [])):
            description = entity.get("description", "").strip()
            if not description:
                continue
            entity_name = entity.get("entity_name", f"entity_{i}")
            docs.append(
                {
                    "id": f"lgentity_{entity_name}",
                    "text": f"[Entity: {entity_name}] {description}",
                    "score": 0.5 / (i + 1),
                }
            )

        for i, rel in enumerate(data.get("relationships", [])):
            description = rel.get("description", "").strip()
            if not description:
                continue
            src = rel.get("src_id", "?")
            tgt = rel.get("tgt_id", "?")
            docs.append(
                {
                    "id": f"lgrel_{src}_{tgt}_{i}",
                    "text": f"[Relation: {src} → {tgt}] {description}",
                    "score": 0.25 / (i + 1),
                }
            )

        return docs
