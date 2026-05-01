"""Tests for the heuristic section classifier."""

from __future__ import annotations

from agentic_autorag.engine.section_classifier import (
    DEFAULT_ELIGIBLE_SECTIONS,
    SectionLabel,
    classify_chunks_in_document,
    detect_section_label,
)


class TestDetectSectionLabel:
    def test_markdown_references_header(self) -> None:
        chunk = "## References\n\n1. Foo et al., 2019..."
        assert detect_section_label(chunk) is SectionLabel.REFERENCES

    def test_plain_references_header(self) -> None:
        chunk = "References\n\n1. Foo et al., 2019..."
        assert detect_section_label(chunk) is SectionLabel.REFERENCES

    def test_acknowledgments_variants(self) -> None:
        for header in ("Acknowledgements", "Acknowledgments", "## Acknowledgements"):
            assert detect_section_label(f"{header}\n\nWe thank...") is SectionLabel.ACKNOWLEDGMENTS

    def test_methods_header(self) -> None:
        chunk = "# Materials and Methods\n\nWe enrolled 50 patients..."
        assert detect_section_label(chunk) is SectionLabel.METHODS

    def test_results_header(self) -> None:
        assert detect_section_label("Results\n\nThe primary endpoint was met.") is SectionLabel.RESULTS

    def test_discussion_header(self) -> None:
        assert detect_section_label("Discussion\n\nOur findings...") is SectionLabel.DISCUSSION

    def test_abstract_header(self) -> None:
        assert detect_section_label("Abstract\n\nBackground: ...") is SectionLabel.ABSTRACT

    def test_author_info_header(self) -> None:
        for header in ("Author Information", "Affiliations", "Corresponding Author"):
            assert detect_section_label(f"{header}\nDr. Lin, MIT") is SectionLabel.AUTHOR_INFO

    def test_numbered_section(self) -> None:
        chunk = "3. References\n\n[1] ..."
        assert detect_section_label(chunk) is SectionLabel.REFERENCES

    def test_body_text_returns_none(self) -> None:
        chunk = (
            "Compound XR-12 was first synthesised by Dr. Lin's group in 2018 "
            "to selectively inhibit kinase TRK-A in murine models of melanoma."
        )
        assert detect_section_label(chunk) is None

    def test_empty_chunk(self) -> None:
        assert detect_section_label("") is None
        assert detect_section_label("   \n\n  ") is None

    def test_header_must_be_near_start(self) -> None:
        # Long body prefix means a 'References' line buried deep doesn't flip
        # the label for this chunk — only chunks that *start* with the header.
        prefix = "Body content. " * 50  # ~700 chars
        chunk = prefix + "\n## References\n[1] ..."
        assert detect_section_label(chunk) is None

    def test_caps_only_threshold_for_plain_header(self) -> None:
        # Sentence-like line doesn't pass plain-header heuristic.
        assert detect_section_label("results were positive overall.") is None
        # Ends with period → not header-shaped.
        assert detect_section_label("Results: A short summary follows here.") is None


class TestClassifyChunksInDocument:
    def test_first_chunk_defaults_to_body(self) -> None:
        chunks = ["Some intro text without any section header to speak of."]
        labels = classify_chunks_in_document(chunks)
        assert labels == [SectionLabel.BODY]

    def test_label_inherits_from_predecessor(self) -> None:
        chunks = [
            "## References\n[1] First citation.",
            "[2] Second citation continued from previous chunk.",
            "[3] More citations.",
        ]
        labels = classify_chunks_in_document(chunks)
        assert labels == [
            SectionLabel.REFERENCES,
            SectionLabel.REFERENCES,
            SectionLabel.REFERENCES,
        ]

    def test_label_flips_on_new_header(self) -> None:
        chunks = [
            "Body content describing experiments.",
            "## References\n[1] ...",
            "[2] another citation",
            "## Acknowledgments\nWe thank...",
            "Funding by NIH grant...",
        ]
        labels = classify_chunks_in_document(chunks)
        assert labels == [
            SectionLabel.BODY,
            SectionLabel.REFERENCES,
            SectionLabel.REFERENCES,
            SectionLabel.ACKNOWLEDGMENTS,
            SectionLabel.ACKNOWLEDGMENTS,
        ]

    def test_no_headers_means_all_body(self) -> None:
        chunks = ["Body 1", "Body 2", "Body 3"]
        labels = classify_chunks_in_document(chunks)
        assert all(label is SectionLabel.BODY for label in labels)


class TestDefaultEligibleSections:
    def test_excludes_references_acks_authors(self) -> None:
        assert SectionLabel.REFERENCES not in DEFAULT_ELIGIBLE_SECTIONS
        assert SectionLabel.ACKNOWLEDGMENTS not in DEFAULT_ELIGIBLE_SECTIONS
        assert SectionLabel.AUTHOR_INFO not in DEFAULT_ELIGIBLE_SECTIONS

    def test_includes_substantive_sections(self) -> None:
        for label in (
            SectionLabel.BODY,
            SectionLabel.ABSTRACT,
            SectionLabel.METHODS,
            SectionLabel.RESULTS,
            SectionLabel.DISCUSSION,
            SectionLabel.OTHER,
        ):
            assert label in DEFAULT_ELIGIBLE_SECTIONS
