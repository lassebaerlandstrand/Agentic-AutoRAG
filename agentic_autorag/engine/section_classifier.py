"""Map Docling heading text to a closed ``SectionLabel`` taxonomy so the
chunk-pair indexer can drop structurally non-substantive chunks (references,
acknowledgments, author blocks) from exam-eligible seeds."""

from __future__ import annotations

import re
from enum import StrEnum


class SectionLabel(StrEnum):
    """Closed taxonomy of section labels.

    Ordered from "almost certainly substantive" down to "almost certainly
    bibliographic / structural noise". The chunk-pair indexer's default
    eligibility set (``DEFAULT_ELIGIBLE_SECTIONS``) excludes the bottom
    three.
    """

    BODY = "body"
    ABSTRACT = "abstract"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    REFERENCES = "references"
    ACKNOWLEDGMENTS = "acknowledgments"
    AUTHOR_INFO = "author_info"


# Heading-text patterns. Order matters: REFERENCES is checked before
# ACKNOWLEDGMENTS so a heading like "References and Acknowledgments" lands
# under REFERENCES (we'd rather drop a borderline chunk than admit one).
# Patterns match the *normalised* heading (numeric prefix and trailing
# colons stripped, lowercased).
_HEADING_PATTERNS: tuple[tuple[SectionLabel, re.Pattern[str]], ...] = (
    (
        SectionLabel.REFERENCES,
        re.compile(
            r"^("
            r"references?(?:\s+(?:cited|and\s+notes))?"
            r"|bibliograph(?:y|ies)"
            r"|literature(?:\s+cited)?"
            r"|works?\s+cited"
            r"|citations?"
            r"|sources?"
            r"|notes?\s+and\s+references"
            r")\b"
        ),
    ),
    (
        SectionLabel.ACKNOWLEDGMENTS,
        re.compile(
            r"^("
            r"acknowledg(?:e?ment|ements)s?"
            r"|funding(?:\s+(?:sources?|information))?"
            r"|conflict[s]?\s+of\s+interest"
            r"|disclosures?"
            r"|competing\s+interests?"
            r"|author\s+contributions?"
            r"|authors'?\s+contributions?"
            r"|ethics\s+(?:statement|approval|declaration)"
            r"|data\s+availability(?:\s+statement)?"
            r"|consent\s+(?:statement|for\s+publication)"
            r")\b"
        ),
    ),
    (
        SectionLabel.AUTHOR_INFO,
        re.compile(
            r"^("
            r"author\s+(?:information|affiliations?|details?|notes?)"
            r"|affiliations?"
            r"|corresponding\s+author"
            r"|authors?\s+address(?:es)?"
            r"|about\s+the\s+authors?"
            r")\b"
        ),
    ),
    (
        SectionLabel.ABSTRACT,
        re.compile(r"^(abstract|summary|tl;dr|tldr|executive\s+summary)\b"),
    ),
    (
        SectionLabel.METHODS,
        re.compile(
            r"^("
            r"methods?|methodology"
            r"|materials?\s+and\s+methods?"
            r"|experimental(?:\s+(?:setup|design|procedure|methods?))?"
            r"|study\s+design"
            r"|data\s+collection"
            r"|participants?"
            r")\b"
        ),
    ),
    (
        SectionLabel.RESULTS,
        re.compile(r"^(results?|findings?|outcomes?)\b"),
    ),
    (
        SectionLabel.DISCUSSION,
        re.compile(r"^(discussions?|conclusions?|limitations?|future\s+work|implications?)\b"),
    ),
)


# Numeric or roman section prefix to strip before matching.
# Matches "1.", "2.3.4", "IV.", "7. " (with optional trailing dot/space).
_PREFIX_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[ivxlcdm]+\.)\s+", re.IGNORECASE)


def _normalise_heading(heading: str) -> str:
    """Lowercase + strip numeric prefix and trailing colons / whitespace."""
    h = heading.strip()
    h = _PREFIX_RE.sub("", h, count=1)
    h = h.rstrip(":").rstrip()
    return h.lower()


def heading_to_label(heading: str | None) -> SectionLabel:
    """Map a single heading text to a ``SectionLabel``.

    Returns ``BODY`` when the heading is absent or matches no pattern —
    BODY is the safe default that keeps chunks eligible for exam seeding.
    """
    if not heading:
        return SectionLabel.BODY
    normalised = _normalise_heading(heading)
    if not normalised:
        return SectionLabel.BODY
    for label, pattern in _HEADING_PATTERNS:
        if pattern.match(normalised):
            return label
    return SectionLabel.BODY


def headings_to_label(headings: list[str] | None) -> SectionLabel:
    """Map a heading breadcrumb to a label, preferring the deepest match.

    Docling's ``HybridChunker`` carries the full heading stack (``[H1,
    H2, ...]``). We walk from deepest to shallowest so a subsection like
    ``["Discussion", "References cited in this section"]`` lands under
    REFERENCES rather than DISCUSSION. Falls through to BODY when no
    heading matches a non-body pattern.
    """
    if not headings:
        return SectionLabel.BODY
    for heading in reversed(headings):
        label = heading_to_label(heading)
        if label is not SectionLabel.BODY:
            return label
    return SectionLabel.BODY


# Sections excluded from entity-cooccurrence indexing by default. These
# chunks rarely contain the kind of subject-matter facts a 2-hop question
# should bridge — they're structural metadata about the document, not its
# content. Source of truth for both the user-facing ``excluded_section_types``
# config default and the derived ``DEFAULT_ELIGIBLE_SECTIONS`` frozenset.
DEFAULT_EXCLUDED_SECTIONS: frozenset[SectionLabel] = frozenset(
    {
        SectionLabel.REFERENCES,
        SectionLabel.ACKNOWLEDGMENTS,
        SectionLabel.AUTHOR_INFO,
    }
)

DEFAULT_ELIGIBLE_SECTIONS: frozenset[SectionLabel] = frozenset(SectionLabel) - DEFAULT_EXCLUDED_SECTIONS


__all__ = [
    "DEFAULT_ELIGIBLE_SECTIONS",
    "DEFAULT_EXCLUDED_SECTIONS",
    "SectionLabel",
    "heading_to_label",
    "headings_to_label",
]
