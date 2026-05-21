"""Tests for the heading → SectionLabel mapper."""

from __future__ import annotations

from agentic_autorag.engine.section_classifier import (
    DEFAULT_ELIGIBLE_SECTIONS,
    SectionLabel,
    heading_to_label,
    headings_to_label,
)


class TestHeadingToLabelReferences:
    def test_plain_references(self) -> None:
        assert heading_to_label("References") is SectionLabel.REFERENCES

    def test_numeric_prefix(self) -> None:
        assert heading_to_label("7. References") is SectionLabel.REFERENCES
        assert heading_to_label("2.3 References") is SectionLabel.REFERENCES

    def test_roman_prefix(self) -> None:
        assert heading_to_label("IV. References") is SectionLabel.REFERENCES

    def test_singular_reference(self) -> None:
        assert heading_to_label("Reference") is SectionLabel.REFERENCES

    def test_references_cited(self) -> None:
        assert heading_to_label("References Cited") is SectionLabel.REFERENCES

    def test_literature(self) -> None:
        # The user's failure case: healthcare_1011958.pdf uses bare "Literature"
        assert heading_to_label("Literature") is SectionLabel.REFERENCES
        assert heading_to_label("Literature:") is SectionLabel.REFERENCES

    def test_literature_cited(self) -> None:
        assert heading_to_label("Literature Cited") is SectionLabel.REFERENCES

    def test_bibliography(self) -> None:
        assert heading_to_label("Bibliography") is SectionLabel.REFERENCES
        assert heading_to_label("Bibliographies") is SectionLabel.REFERENCES

    def test_works_cited(self) -> None:
        assert heading_to_label("Works Cited") is SectionLabel.REFERENCES
        assert heading_to_label("Work Cited") is SectionLabel.REFERENCES

    def test_citations(self) -> None:
        assert heading_to_label("Citations") is SectionLabel.REFERENCES
        assert heading_to_label("Citation") is SectionLabel.REFERENCES

    def test_sources(self) -> None:
        assert heading_to_label("Sources") is SectionLabel.REFERENCES
        assert heading_to_label("Source") is SectionLabel.REFERENCES

    def test_notes_and_references(self) -> None:
        assert heading_to_label("Notes and references") is SectionLabel.REFERENCES

    def test_trailing_colon_stripped(self) -> None:
        assert heading_to_label("REFERENCES:") is SectionLabel.REFERENCES


class TestHeadingToLabelAcknowledgments:
    def test_basic_variants(self) -> None:
        for h in ("Acknowledgments", "Acknowledgements", "Acknowledgement", "Acknowledgment"):
            assert heading_to_label(h) is SectionLabel.ACKNOWLEDGMENTS

    def test_funding(self) -> None:
        assert heading_to_label("Funding") is SectionLabel.ACKNOWLEDGMENTS
        assert heading_to_label("Funding sources") is SectionLabel.ACKNOWLEDGMENTS

    def test_conflict_of_interest(self) -> None:
        assert heading_to_label("Conflict of Interest") is SectionLabel.ACKNOWLEDGMENTS
        assert heading_to_label("Conflicts of interest") is SectionLabel.ACKNOWLEDGMENTS

    def test_competing_interests(self) -> None:
        assert heading_to_label("Competing interests") is SectionLabel.ACKNOWLEDGMENTS

    def test_disclosures(self) -> None:
        assert heading_to_label("Disclosures") is SectionLabel.ACKNOWLEDGMENTS

    def test_author_contributions(self) -> None:
        assert heading_to_label("Author Contributions") is SectionLabel.ACKNOWLEDGMENTS

    def test_data_availability(self) -> None:
        assert heading_to_label("Data Availability") is SectionLabel.ACKNOWLEDGMENTS
        assert heading_to_label("Data availability statement") is SectionLabel.ACKNOWLEDGMENTS

    def test_ethics_statement(self) -> None:
        assert heading_to_label("Ethics Statement") is SectionLabel.ACKNOWLEDGMENTS


class TestHeadingToLabelAuthorInfo:
    def test_author_information(self) -> None:
        assert heading_to_label("Author Information") is SectionLabel.AUTHOR_INFO

    def test_affiliations(self) -> None:
        assert heading_to_label("Affiliations") is SectionLabel.AUTHOR_INFO
        assert heading_to_label("Affiliation") is SectionLabel.AUTHOR_INFO

    def test_corresponding_author(self) -> None:
        assert heading_to_label("Corresponding Author") is SectionLabel.AUTHOR_INFO


class TestHeadingToLabelStandardSections:
    def test_abstract(self) -> None:
        assert heading_to_label("Abstract") is SectionLabel.ABSTRACT
        assert heading_to_label("Summary") is SectionLabel.ABSTRACT
        assert heading_to_label("Executive Summary") is SectionLabel.ABSTRACT
        assert heading_to_label("TL;DR") is SectionLabel.ABSTRACT

    def test_methods(self) -> None:
        assert heading_to_label("Methods") is SectionLabel.METHODS
        assert heading_to_label("Methodology") is SectionLabel.METHODS
        assert heading_to_label("Materials and methods") is SectionLabel.METHODS
        assert heading_to_label("Experimental Setup") is SectionLabel.METHODS

    def test_results(self) -> None:
        assert heading_to_label("Results") is SectionLabel.RESULTS
        assert heading_to_label("Findings") is SectionLabel.RESULTS

    def test_discussion(self) -> None:
        assert heading_to_label("Discussion") is SectionLabel.DISCUSSION
        assert heading_to_label("Conclusion") is SectionLabel.DISCUSSION
        assert heading_to_label("Conclusions") is SectionLabel.DISCUSSION
        assert heading_to_label("Limitations") is SectionLabel.DISCUSSION
        assert heading_to_label("Future work") is SectionLabel.DISCUSSION


class TestHeadingToLabelBody:
    def test_empty_or_none(self) -> None:
        assert heading_to_label(None) is SectionLabel.BODY
        assert heading_to_label("") is SectionLabel.BODY
        assert heading_to_label("   ") is SectionLabel.BODY

    def test_arbitrary_body_heading(self) -> None:
        assert heading_to_label("Patient Demographics") is SectionLabel.BODY
        assert heading_to_label("Background and Motivation") is SectionLabel.BODY
        assert heading_to_label("How does the device work?") is SectionLabel.BODY


class TestHeadingsToLabel:
    def test_empty_breadcrumb(self) -> None:
        assert headings_to_label([]) is SectionLabel.BODY
        assert headings_to_label(None) is SectionLabel.BODY

    def test_single_heading(self) -> None:
        assert headings_to_label(["References"]) is SectionLabel.REFERENCES

    def test_deepest_match_wins(self) -> None:
        # Subsection under a body parent — the deepest heading carries the label.
        assert headings_to_label(["Discussion", "References cited"]) is SectionLabel.REFERENCES

    def test_shallowest_match_wins_when_deeper_is_body(self) -> None:
        # Deepest is body-like; classifier falls back to the parent heading.
        assert headings_to_label(["References", "Author Index"]) is SectionLabel.REFERENCES

    def test_all_body_falls_through(self) -> None:
        assert headings_to_label(["Background", "Motivation"]) is SectionLabel.BODY


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
        ):
            assert label in DEFAULT_ELIGIBLE_SECTIONS
