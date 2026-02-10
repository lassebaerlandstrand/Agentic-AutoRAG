"""Tests for examiner clustering and allocation logic."""

import numpy as np

from agentic_autorag.examiner.clustering import (
    allocate_largest_remainder,
    compute_clusters,
    resolve_n_clusters,
)


class TestResolveNClusters:
    def test_explicit_value_used(self) -> None:
        assert resolve_n_clusters(1000, 50, explicit=10) == 10

    def test_explicit_one(self) -> None:
        assert resolve_n_clusters(500, 50, explicit=1) == 1

    def test_auto_sqrt(self) -> None:
        # sqrt(100) = 10, exam_size=50 → 10
        assert resolve_n_clusters(100, 50) == 10

    def test_auto_capped_by_exam_size(self) -> None:
        # sqrt(10000) = 100, but exam_size=20 → 20
        assert resolve_n_clusters(10000, 20) == 20

    def test_auto_single_chunk(self) -> None:
        # sqrt(1) = 1
        assert resolve_n_clusters(1, 50) == 1

    def test_auto_floor_of_one(self) -> None:
        # Even with 0 chunks (edge case), floor is 1
        assert resolve_n_clusters(0, 50) >= 1

    def test_auto_large_chunk_count(self) -> None:
        # sqrt(2500) = 50, exam_size=100 → 50
        assert resolve_n_clusters(2500, 100) == 50


class TestComputeClusters:
    def test_returns_correct_shape(self) -> None:
        embeddings = np.random.default_rng(42).standard_normal((50, 8))
        labels = compute_clusters(embeddings, n_clusters=5)
        assert labels.shape == (50,)

    def test_label_range(self) -> None:
        embeddings = np.random.default_rng(42).standard_normal((30, 4))
        labels = compute_clusters(embeddings, n_clusters=3)
        assert set(labels).issubset({0, 1, 2})

    def test_deterministic(self) -> None:
        embeddings = np.random.default_rng(42).standard_normal((40, 4))
        labels_a = compute_clusters(embeddings, n_clusters=4)
        labels_b = compute_clusters(embeddings, n_clusters=4)
        np.testing.assert_array_equal(labels_a, labels_b)

    def test_well_separated_clusters(self) -> None:
        """Three clearly separated groups should each get their own label."""
        rng = np.random.default_rng(0)
        cluster_a = rng.standard_normal((20, 2)) + np.array([0, 0])
        cluster_b = rng.standard_normal((20, 2)) + np.array([50, 50])
        cluster_c = rng.standard_normal((20, 2)) + np.array([-50, 50])
        embeddings = np.vstack([cluster_a, cluster_b, cluster_c])

        labels = compute_clusters(embeddings, n_clusters=3)
        # All points in each group should share the same label
        assert len(set(labels[:20])) == 1
        assert len(set(labels[20:40])) == 1
        assert len(set(labels[40:])) == 1
        # And those labels should be distinct
        assert len({labels[0], labels[20], labels[40]}) == 3


class TestAllocateLargestRemainder:
    def test_sums_to_exam_size(self) -> None:
        sizes = np.array([100, 50, 30, 20])
        alloc = allocate_largest_remainder(sizes, exam_size=50)
        assert alloc.sum() == 50

    def test_equal_clusters(self) -> None:
        sizes = np.array([100, 100, 100, 100])
        alloc = allocate_largest_remainder(sizes, exam_size=20)
        assert alloc.sum() == 20
        # Equal clusters → equal (or near-equal) allocation
        assert alloc.min() >= 4
        assert alloc.max() <= 6

    def test_unequal_clusters(self) -> None:
        sizes = np.array([400, 25, 25, 25])
        alloc = allocate_largest_remainder(sizes, exam_size=40)
        assert alloc.sum() == 40
        # Largest cluster should get more, but sqrt damps it
        assert alloc[0] > alloc[1]

    def test_single_cluster(self) -> None:
        sizes = np.array([200])
        alloc = allocate_largest_remainder(sizes, exam_size=30)
        assert alloc.sum() == 30
        assert alloc[0] == 30

    def test_cluster_smaller_than_allocation(self) -> None:
        """If a cluster has fewer chunks than its allocation, cap at cluster size."""
        sizes = np.array([3, 100, 100])
        alloc = allocate_largest_remainder(sizes, exam_size=50)
        assert alloc[0] <= 3
        assert alloc.sum() == 50

    def test_all_zero_clusters(self) -> None:
        sizes = np.array([0, 0, 0])
        alloc = allocate_largest_remainder(sizes, exam_size=10)
        assert alloc.sum() == 0

    def test_exam_size_exceeds_total_chunks(self) -> None:
        """When exam_size > total chunks, allocation should cap at total available."""
        sizes = np.array([5, 5, 5])
        alloc = allocate_largest_remainder(sizes, exam_size=100)
        # Can only allocate at most 15 total
        assert alloc.sum() <= 15

    def test_no_negative_allocations(self) -> None:
        sizes = np.array([10, 1, 50, 2, 100])
        alloc = allocate_largest_remainder(sizes, exam_size=30)
        assert (alloc >= 0).all()
        assert alloc.sum() == 30

    def test_many_small_clusters(self) -> None:
        sizes = np.ones(20, dtype=int) * 5  # 20 clusters, 5 each
        alloc = allocate_largest_remainder(sizes, exam_size=50)
        assert alloc.sum() == 50
        # Each cluster has capacity 5, so no cluster should exceed that
        assert (alloc <= 5).all()
