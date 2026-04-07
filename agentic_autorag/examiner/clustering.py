"""Corpus clustering and question allocation.

Provides KMeans clustering over document embeddings, per-document retrieval
difficulty scoring, and two allocation strategies:

- ``allocate_largest_remainder``: sqrt-proportional diversity-first (legacy).
- ``allocate_difficulty_weighted``: concentrates questions in dense
  neighborhoods where retrieval is genuinely hard, with a per-cluster floor.
"""

import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

_DIFFICULTY_N_NEIGHBORS = 5


def resolve_n_clusters(n_items: int, target_size: int) -> int:
    """Determine the number of clusters for corpus diversity.

    Auto-computes as ``min(sqrt(n_items), target_size)`` with a floor of 1.
    """
    return max(1, min(int(math.sqrt(n_items)), target_size))


def compute_clusters(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cluster embeddings using KMeans and return the label array."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings)


def compute_difficulty_scores(
    doc_embeddings: np.ndarray,
    n_neighbors: int = _DIFFICULTY_N_NEIGHBORS,
) -> np.ndarray:
    """Per-document retrieval difficulty based on neighbor similarity.

    Documents with many similar neighbors are hard to distinguish via
    retrieval.  Isolated documents (unique topic) are trivially retrievable.

    Returns an array of shape ``(n_docs,)`` with scores in [0, 1].
    """
    n_docs = doc_embeddings.shape[0]
    if n_docs <= 1:
        return np.zeros(n_docs, dtype=np.float32)

    normed = normalize(doc_embeddings, norm="l2")
    sim = normed @ normed.T
    np.fill_diagonal(sim, -1.0)

    k = min(n_neighbors, n_docs - 1)
    # Partition instead of full sort — faster for large n_docs
    top_k_indices = np.argpartition(sim, -k, axis=1)[:, -k:]
    top_k_sims = np.take_along_axis(sim, top_k_indices, axis=1)
    scores = top_k_sims.mean(axis=1).astype(np.float32)

    # Clamp to [0, 1] — cosine similarity of normalised vectors is in [-1, 1]
    return np.clip(scores, 0.0, 1.0)


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


def allocate_difficulty_weighted(
    cluster_sizes: np.ndarray,
    difficulty_scores: np.ndarray,
    labels: np.ndarray,
    exam_size: int,
    min_per_cluster: int = 1,
) -> np.ndarray:
    """Distribute *exam_size* slots weighted by cluster difficulty with a floor.

    Each cluster receives at least ``min_per_cluster`` questions (capped at
    cluster size).  Remaining slots are distributed via Hamilton's method
    using ``cluster_difficulty * sqrt(cluster_size)`` as weights, where
    ``cluster_difficulty`` is the mean document difficulty of its members.
    """
    n_clusters = len(cluster_sizes)
    alloc = np.zeros(n_clusters, dtype=int)

    # Per-cluster mean difficulty
    cluster_diff = np.zeros(n_clusters, dtype=np.float64)
    for cid in range(n_clusters):
        mask = labels == cid
        if mask.any():
            cluster_diff[cid] = float(difficulty_scores[mask].mean())

    # Floor allocation
    for cid in range(n_clusters):
        alloc[cid] = min(min_per_cluster, int(cluster_sizes[cid]))
    remaining = exam_size - int(alloc.sum())

    if remaining <= 0:
        return alloc

    # Difficulty-weighted allocation for the remaining slots
    capacity = cluster_sizes - alloc
    weights = cluster_diff * np.sqrt(cluster_sizes.astype(float))
    # Zero out clusters with no remaining capacity
    weights[capacity <= 0] = 0.0
    total_weight = weights.sum()

    if total_weight == 0:
        return alloc

    quotas = remaining * weights / total_weight
    floors = np.floor(quotas).astype(int)
    floors = np.minimum(floors, capacity)
    remainders = quotas - floors

    deficit = remaining - int(floors.sum())
    remainder_order = np.argsort(-remainders)
    for idx in remainder_order:
        if deficit <= 0:
            break
        if floors[idx] < capacity[idx]:
            floors[idx] += 1
            deficit -= 1

    alloc += floors
    return alloc
