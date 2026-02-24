"""Iterative exam refinement using IRT-based question culling."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from tqdm import tqdm

from agentic_autorag.config.models import MCQQuestion
from agentic_autorag.examiner.clustering import compute_clusters, resolve_n_clusters
from agentic_autorag.examiner.exam_agent import _RETRY_COOLDOWNS, ExamAgent
from agentic_autorag.examiner.irt import IRTAnalyzer, IRTResult

logger = logging.getLogger(__name__)


class ExamRefiner:
    """Refines an exam by culling weak questions and generating replacements."""

    def __init__(
        self,
        irt_analyzer: IRTAnalyzer,
        exam_agent: ExamAgent,
        drop_ratio: float = 0.1,
    ) -> None:
        self.irt = irt_analyzer
        self.exam_agent = exam_agent
        self.drop_ratio = drop_ratio

    def analyze(
        self,
        exam: list[MCQQuestion],
        response_matrix: np.ndarray,
    ) -> tuple[IRTResult, list[MCQQuestion]]:
        """Run IRT and update question parameters on the exam."""
        irt_result = self.irt.fit(response_matrix)

        exam_sorted = sorted(exam, key=lambda q: q.id)
        for idx, question in enumerate(exam_sorted):
            question.discrimination = float(irt_result.discriminations[idx])
            question.difficulty = float(irt_result.difficulties[idx])
            question.guessing = float(irt_result.guessings[idx])

        return irt_result, exam

    def cull(
        self,
        exam: list[MCQQuestion],
        irt_result: IRTResult,
    ) -> tuple[list[MCQQuestion], list[MCQQuestion]]:
        """Remove the least discriminative questions from the exam."""
        exam_sorted = sorted(exam, key=lambda q: q.id)
        cull_indices = self.irt.identify_cull_candidates(irt_result.discriminations, self.drop_ratio)
        culled_ids = {exam_sorted[idx].id for idx in cull_indices}

        surviving = [question for question in exam if question.id not in culled_ids]
        culled = [question for question in exam if question.id in culled_ids]
        return surviving, culled

    async def refine(
        self,
        exam: list[MCQQuestion],
        response_matrix: np.ndarray,
        chunks: list[str],
        chunk_ids: list[str],
        embeddings: np.ndarray,
    ) -> list[MCQQuestion]:
        """Full refinement cycle: analyze, cull, and replace questions."""
        if len(exam) <= 1:
            return exam

        irt_result, updated_exam = self.analyze(exam, response_matrix)
        surviving, culled = self.cull(updated_exam, irt_result)

        if not culled:
            return updated_exam

        target_clusters = [question.cluster_id for question in culled]
        replacements = await self._generate_replacements(
            surviving_exam=surviving,
            n_replacements=len(culled),
            target_clusters=target_clusters,
            chunks=chunks,
            chunk_ids=chunk_ids,
            embeddings=embeddings,
        )

        refined_exam = surviving + replacements
        if len(refined_exam) < len(exam):
            logger.warning(
                "Exam refinement produced fewer questions than requested (%d/%d)",
                len(refined_exam),
                len(exam),
            )
        return refined_exam

    async def _generate_replacements(
        self,
        surviving_exam: list[MCQQuestion],
        n_replacements: int,
        target_clusters: list[int],
        chunks: list[str],
        chunk_ids: list[str],
        embeddings: np.ndarray,
    ) -> list[MCQQuestion]:
        """Generate replacement questions in batches, prioritizing culled-question clusters."""
        if n_replacements <= 0:
            return []

        n_clusters = resolve_n_clusters(
            n_chunks=len(chunks),
            exam_size=self.exam_agent.config.exam_size,
            explicit=self.exam_agent.config.diversity_clusters,
        )
        labels = compute_clusters(embeddings, n_clusters)

        cluster_to_indices: dict[int, list[int]] = {cluster_id: [] for cluster_id in range(n_clusters)}
        for idx, cluster_id in enumerate(labels):
            cluster_to_indices[int(cluster_id)].append(idx)

        used_chunk_ids = {question.source_chunk_id for question in surviving_exam}

        prioritized_clusters = [cluster for cluster in target_clusters if cluster in cluster_to_indices]
        remaining_clusters = [cluster for cluster in cluster_to_indices if cluster not in prioritized_clusters]
        ordered_clusters = prioritized_clusters + remaining_clusters

        candidates: list[tuple[str, str, int]] = []
        for cluster_id in ordered_clusters:
            candidate_indices = list(cluster_to_indices[cluster_id])
            self.exam_agent._rng.shuffle(candidate_indices)
            for idx in candidate_indices:
                source_chunk_id = chunk_ids[idx]
                if source_chunk_id in used_chunk_ids:
                    continue
                candidates.append((chunks[idx], source_chunk_id, cluster_id))
                used_chunk_ids.add(source_chunk_id)

        _TRANSIENT_ERROR = object()
        results_by_idx: dict[int, MCQQuestion | None | object] = {}

        await self._run_replacement_pass(
            candidates,
            results_by_idx,
            _TRANSIENT_ERROR,
            n_target=n_replacements,
            desc="Generating replacements",
        )

        for retry_round, cooldown in enumerate(_RETRY_COOLDOWNS, start=1):
            error_indices = [i for i, r in results_by_idx.items() if r is _TRANSIENT_ERROR]
            if not error_indices:
                break
            tqdm.write(
                f"\n  {len(error_indices)} replacement(s) failed"
                f" — retrying after {cooldown}s cooldown"
                f" (round {retry_round}/{len(_RETRY_COOLDOWNS)})"
            )
            await asyncio.sleep(cooldown)
            retry_items = [(i, candidates[i]) for i in sorted(error_indices)]
            await self._run_replacement_pass_indexed(
                retry_items,
                results_by_idx,
                _TRANSIENT_ERROR,
                desc=f"Replacement retry {retry_round}",
            )

        replacements: list[MCQQuestion] = []
        for idx in range(len(candidates)):
            result = results_by_idx.get(idx)
            if result is not None and result is not _TRANSIENT_ERROR:
                replacements.append(result)  # type: ignore[arg-type]
            if len(replacements) >= n_replacements:
                break

        return replacements

    async def _run_replacement_pass(
        self,
        candidates: list[tuple[str, str, int]],
        results_by_idx: dict[int, MCQQuestion | None | object],
        transient_sentinel: object,
        *,
        n_target: int,
        desc: str,
    ) -> None:
        """Process candidates with a semaphore, stopping once n_target successes are collected."""
        sem = asyncio.Semaphore(self.exam_agent.concurrency)
        n_accepted = 0

        with tqdm(total=n_target, desc=desc, unit="q") as pbar:
            async def _bounded(idx: int, chunk: str, chunk_id: str, cluster_id: int) -> None:
                nonlocal n_accepted
                async with sem:
                    if n_accepted >= n_target:
                        results_by_idx[idx] = None
                        return
                    result = await self.exam_agent._generate_single(chunk, chunk_id, cluster_id, transient_sentinel)

                if result is not transient_sentinel and result is not None:
                    n_accepted += 1
                    pbar.update(1)
                results_by_idx[idx] = result

            await asyncio.gather(*[
                _bounded(i, chunk, chunk_id, cluster_id)
                for i, (chunk, chunk_id, cluster_id) in enumerate(candidates)
            ])

    async def _run_replacement_pass_indexed(
        self,
        indexed_candidates: list[tuple[int, tuple[str, str, int]]],
        results_by_idx: dict[int, MCQQuestion | None | object],
        transient_sentinel: object,
        desc: str,
    ) -> None:
        sem = asyncio.Semaphore(self.exam_agent.concurrency)

        with tqdm(total=len(indexed_candidates), desc=desc, unit="q") as pbar:
            async def _bounded(idx: int, chunk: str, chunk_id: str, cluster_id: int) -> None:
                async with sem:
                    result = await self.exam_agent._generate_single(chunk, chunk_id, cluster_id, transient_sentinel)
                results_by_idx[idx] = result
                pbar.update(1)

            await asyncio.gather(*[
                _bounded(idx, chunk, chunk_id, cluster_id)
                for idx, (chunk, chunk_id, cluster_id) in indexed_candidates
            ])
