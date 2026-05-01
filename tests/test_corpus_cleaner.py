"""Tests for the near-duplicate detection helper."""

from __future__ import annotations

import pytest

from agentic_autorag.engine.corpus_cleaner import (
    DuplicateClusters,
    detect_near_duplicates,
)

# Synthetic body of text long enough that 5-shingles produce a meaningful
# containment signal. Reused across tests so we can produce variants.
_BASE_TEXT = (
    "The patient presented with chest pain and shortness of breath. "
    "Initial workup included an EKG, chest X-ray, and basic labs. "
    "The EKG showed ST elevation in leads II, III, and aVF, suggestive "
    "of an inferior myocardial infarction. Cardiac catheterization was "
    "performed urgently and revealed a thrombotic occlusion of the right "
    "coronary artery. A drug-eluting stent was deployed with restoration "
    "of normal flow."
) * 4


def _slightly_changed(text: str) -> str:
    """OCR-ish noise: drop ~1 word per 200 to simulate lossy OCR."""
    words = text.split()
    keep = [w for i, w in enumerate(words) if i % 200 != 137]
    return " ".join(keep)


class TestDetectNearDuplicates:
    def test_no_documents_returns_empty_clusters(self) -> None:
        result = detect_near_duplicates([], [])
        assert result.canonical_doc_ids == []
        assert result.alias_to_canonical == {}
        assert result.n_clusters == 0
        assert result.n_duplicates == 0

    def test_distinct_documents_form_singleton_clusters(self) -> None:
        docs = [
            "The mitochondrion is the powerhouse of the cell.",
            "France's capital city is Paris on the river Seine.",
            "Photosynthesis converts light into chemical energy.",
        ]
        ids = ["a.txt", "b.txt", "c.txt"]
        result = detect_near_duplicates(docs, ids, threshold=0.85)
        assert sorted(result.canonical_doc_ids) == sorted(ids)
        assert result.n_clusters == 3
        assert result.n_duplicates == 0
        for d in ids:
            assert result.alias_to_canonical[d] == d

    def test_exact_duplicates_cluster(self) -> None:
        docs = [_BASE_TEXT, _BASE_TEXT, "Totally unrelated text " * 100]
        ids = ["paper.pdf", "paper_page_001.png", "other.pdf"]
        result = detect_near_duplicates(docs, ids, threshold=0.85)
        assert result.n_clusters == 2
        assert result.n_duplicates == 1
        assert result.alias_to_canonical["paper.pdf"] == result.alias_to_canonical["paper_page_001.png"]
        assert result.alias_to_canonical["other.pdf"] == "other.pdf"

    def test_near_duplicates_with_minor_noise_cluster(self) -> None:
        docs = [_BASE_TEXT, _slightly_changed(_BASE_TEXT)]
        ids = ["clean.pdf", "ocr.pdf"]
        result = detect_near_duplicates(docs, ids, threshold=0.85)
        assert result.n_clusters == 1
        assert result.alias_to_canonical["clean.pdf"] == result.alias_to_canonical["ocr.pdf"]

    def test_threshold_separates_partial_overlap(self) -> None:
        # Two docs that share ~half their content. At a low threshold
        # they cluster; at a high threshold they don't.
        common = "the patient had chest pain and shortness of breath " * 20
        a = common + " " + "extra unique content about clinical labs and tests " * 20
        b = common + " " + "completely different text about chlorophyll and plants " * 20
        loose = detect_near_duplicates([a, b], ["a.pdf", "b.pdf"], threshold=0.3)
        strict = detect_near_duplicates([a, b], ["a.pdf", "b.pdf"], threshold=0.9)
        assert loose.n_clusters == 1
        assert strict.n_clusters == 2

    def test_canonical_is_longest_document(self) -> None:
        long_text = _BASE_TEXT
        short_text = " ".join(_BASE_TEXT.split()[:60])  # truncated alias
        docs = [short_text, long_text]
        ids = ["short.pdf", "long.pdf"]
        result = detect_near_duplicates(docs, ids, threshold=0.5)
        if result.n_clusters == 1:
            canon = result.canonical_doc_ids[0]
            assert canon == "long.pdf"

    def test_canonicalize_returns_input_for_unknown_docs(self) -> None:
        result = DuplicateClusters(canonical_doc_ids=["a"], alias_to_canonical={"a": "a"})
        assert result.canonicalize("a") == "a"
        assert result.canonicalize("not_in_map") == "not_in_map"

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            detect_near_duplicates(["a", "b"], ["only_one"])

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            detect_near_duplicates(["a"], ["a"], threshold=1.5)

    def test_three_way_cluster(self) -> None:
        docs = [_BASE_TEXT, _slightly_changed(_BASE_TEXT), _slightly_changed(_BASE_TEXT)]
        ids = ["v1.pdf", "v2.pdf", "v3.pdf"]
        result = detect_near_duplicates(docs, ids, threshold=0.85)
        assert result.n_clusters == 1
        canon = result.canonical_doc_ids[0]
        for d in ids:
            assert result.alias_to_canonical[d] == canon


class TestContainment:
    """Containment metric catches 'small-fully-inside-large' that symmetric
    Jaccard would miss (a one-page image inside a multi-page PDF has
    Jaccard ≈ 1/N even when its shingles are a strict subset of the PDF's).
    """

    def test_subset_clusters(self) -> None:
        # A small fragment whose shingles are a strict subset of a longer
        # document. Containment ≈ 1.0 even when the docs differ in length.
        long_doc = _BASE_TEXT
        short_doc = " ".join(_BASE_TEXT.split()[:30])
        result = detect_near_duplicates(
            [long_doc, short_doc],
            ["pdf.pdf", "page1.png"],
            threshold=0.85,
        )
        assert result.n_clusters == 1
        assert result.canonical_doc_ids[0] == "pdf.pdf"

    def test_unrelated_short_doc_not_clustered(self) -> None:
        # A short document that is genuinely unrelated to the long one
        # must not get clustered just because the threshold is loose.
        result = detect_near_duplicates(
            [_BASE_TEXT, "Photosynthesis converts light into chemical energy. " * 10],
            ["pdf.pdf", "biology.pdf"],
            threshold=0.85,
        )
        assert result.n_clusters == 2


class TestNormalizedTokenization:
    """Token normalisation (lowercase + word-only regex) makes shingling
    robust to punctuation, capitalisation, and Unicode marks like daggers
    that show up in OCR-of-PDF page images.
    """

    def test_capitalisation_does_not_break_clustering(self) -> None:
        # Same text, different capitalisation. Without normalisation,
        # shingles differ; with normalisation, they collide.
        text_a = _BASE_TEXT
        text_b = _BASE_TEXT.upper()
        result = detect_near_duplicates([text_a, text_b], ["a.pdf", "b.pdf"], threshold=0.95)
        assert result.n_clusters == 1

    def test_punctuation_differences_do_not_break_clustering(self) -> None:
        # Identical content with/without dagger marks ('†') after author
        # names — typical OCR-of-PDF artefact. Word-only normalisation
        # drops the daggers entirely so shingle sets match.
        author_pdf = "Author Name †, Coauthor One †, Coauthor Two †. " + _BASE_TEXT
        author_png = "Author Name, Coauthor One, Coauthor Two. " + _BASE_TEXT
        result = detect_near_duplicates(
            [author_pdf, author_png],
            ["paper.pdf", "page1.png"],
            threshold=0.95,
        )
        assert result.n_clusters == 1
