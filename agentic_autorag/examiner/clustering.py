"""Corpus clustering and sqrt-proportional question allocation.

Provides KMeans clustering over chunk embeddings and Hamilton's method
(largest remainder) for distributing exam question slots across clusters
with square-root proportional weights.
"""

import math

import numpy as np
from sklearn.cluster import KMeans


def resolve_n_clusters(n_chunks: int, exam_size: int, explicit: int | None = None) -> int:
    """Determine the number of clusters for corpus diversity.

    If *explicit* is provided (from ``ExaminerConfig.diversity_clusters``),
    use it directly.  Otherwise auto-compute as ``min(sqrt(n_chunks), exam_size)``,
    with a floor of 1.
    """
    if explicit is not None:
        return explicit
    return max(1, min(int(math.sqrt(n_chunks)), exam_size))


def compute_clusters(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cluster embeddings using KMeans and return the label array."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings)


def allocate_largest_remainder(cluster_sizes: np.ndarray, exam_size: int) -> np.ndarray:
    """Distribute exactly *exam_size* question slots across clusters.

    Uses square-root proportional weights and the largest remainder method
    (Hamilton's method). Each cluster's allocation is capped at its actual
    size so we never try to sample more chunks than exist.
    """
    weights = np.sqrt(cluster_sizes.astype(float))
    total_weight = weights.sum()

    if total_weight == 0:
        return np.zeros(len(cluster_sizes), dtype=int)

    quotas = exam_size * weights / total_weight
    floors = np.floor(quotas).astype(int)
    remainders = quotas - floors

    # Cap each cluster's allocation at its actual size
    floors = np.minimum(floors, cluster_sizes)

    deficit = exam_size - floors.sum()

    # Award remaining slots to clusters with the largest remainders
    # that still have capacity
    remainder_order = np.argsort(-remainders)
    for idx in remainder_order:
        if deficit <= 0:
            break
        if floors[idx] < cluster_sizes[idx]:
            floors[idx] += 1
            deficit -= 1

    return floors
