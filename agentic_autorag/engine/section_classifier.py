"""Heuristic section classifier for document chunks.

Labels each chunk with one of a fixed taxonomy of section types so the
chunk-pair index can drop structurally non-substantive chunks (citation
lists, acknowledgments, author/affiliation blocks) before entity extraction
ever runs.

The classifier is regex-on-headers only — no LLM, no learned model. It walks
chunks in document order, maintains a "current section" state, and updates
that state whenever it sees a markdown-style header or a recognised
section-title line near the top of a chunk. Chunks without a header
inherit the section of their predecessor in the same document.

Limitations (acceptable on purpose):
  - A document with no explicit header markers labels everything as ``body``.
  - Headers buried mid-chunk (rare in practice for parsed PDFs/markdown)
    only flip the label for the *next* chunk, not retroactively.
  - The label set is closed; unknown sections fall through to ``other``.

The downstream LLM composition gate is the second line of defence — if a
mention-only bridge slips through the section filter, the LLM is asked to
state what fact each chunk asserts about the bridge entity, and refuses to
compose a question when no real fact exists.
"""

from __future__ import annotations

import re
from enum import StrEnum

# Cap how far into a chunk we look for a header line. Real markdown/PDF
# parses put headers at the start of a section; scanning further drags in
# headers from neighbouring sections that happen to have been merged into
# one chunk by the splitter.
_MAX_HEADER_SCAN_CHARS = 200


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
    OTHER = "other"


# Section-title patterns. Keys are the canonical label; values are
# alternation-free regexes that match a line whose *content* (after
# stripping markdown header markers and numbering) starts with one of the
# listed phrases. Order doesn't matter for correctness, only for tie-breaks.
_SECTION_TITLE_PATTERNS: tuple[tuple[SectionLabel, re.Pattern[str]], ...] = (
    (
        SectionLabel.REFERENCES,
        re.compile(
            r"^\s*(references?|bibliograph(y|ies)|works?\s+cited|literature\s+cited|citations?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        SectionLabel.ACKNOWLEDGMENTS,
        re.compile(
            r"^\s*(acknowledg(e?ments?|ements?)|funding|conflict\s+of\s+interest|disclosures?|"
            r"competing\s+interests?|author\s+contributions?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        SectionLabel.AUTHOR_INFO,
        re.compile(
            r"^\s*(author\s+(information|affiliations?|details?|notes?)|affiliations?|"
            r"corresponding\s+author|authors?\s+address(es)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        SectionLabel.ABSTRACT,
        re.compile(r"^\s*(abstract|summary|tl;dr)\b", re.IGNORECASE),
    ),
    (
        SectionLabel.METHODS,
        re.compile(
            r"^\s*(methods?|methodology|materials?\s+and\s+methods?|experimental(\s+(setup|design|procedure))?|"
            r"study\s+design|data\s+collection|participants?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        SectionLabel.RESULTS,
        re.compile(r"^\s*(results?|findings?)\b", re.IGNORECASE),
    ),
    (
        SectionLabel.DISCUSSION,
        re.compile(r"^\s*(discussions?|conclusions?|limitations?|future\s+work)\b", re.IGNORECASE),
    ),
)


# Strip markdown header markers (``#``, ``##``, …) and leading section
# numbers (``1.``, ``1.2.3 ``, ``IV.``) so the title-pattern regex can match
# the bare phrase.
_HEADER_MARKER_RE = re.compile(r"^\s*#{1,6}\s+")
_NUMERIC_PREFIX_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[ivxlcdm]+\.)\s+", re.IGNORECASE)


def _classify_header_line(line: str) -> SectionLabel | None:
    """Return the label a single line implies, or None if it isn't a header.

    A "header" here is either a markdown ``#`` line, or a short standalone
    line whose content matches one of the recognised section-title patterns.
    """
    stripped = line.strip()
    if not stripped:
        return None

    is_markdown_header = bool(_HEADER_MARKER_RE.match(stripped))
    if is_markdown_header:
        stripped = _HEADER_MARKER_RE.sub("", stripped, count=1)

    bare = _NUMERIC_PREFIX_RE.sub("", stripped, count=1).strip()
    if not bare:
        return None

    # Non-markdown lines must be plausibly header-shaped: short, capitalised,
    # no terminal punctuation that would suggest a sentence. Markdown
    # headers (``## results``) bypass this check — the marker itself is the
    # signal.
    if not is_markdown_header:
        if len(bare) > 80 or bare.endswith((".", ",", ";", "?", "!")):
            return None
        first_alpha = next((c for c in bare if c.isalpha()), "")
        if not first_alpha.isupper():
            return None

    for label, pattern in _SECTION_TITLE_PATTERNS:
        if pattern.match(bare):
            return label

    return None


def detect_section_label(chunk_text: str) -> SectionLabel | None:
    """Inspect the start of a chunk for a section header.

    Returns the implied label when one is found, else ``None``. Callers
    use this to update the running "current section" state as they walk
    chunks in document order.
    """
    if not chunk_text:
        return None
    head = chunk_text[:_MAX_HEADER_SCAN_CHARS]
    for line in head.splitlines():
        label = _classify_header_line(line)
        if label is not None:
            return label
    return None


# Sections eligible for entity-cooccurrence indexing by default. References,
# acknowledgments, and author_info chunks rarely contain the kind of
# subject-matter facts a 2-hop question should bridge — they're structural
# metadata about the document, not its content.
DEFAULT_ELIGIBLE_SECTIONS: frozenset[SectionLabel] = frozenset(
    {
        SectionLabel.BODY,
        SectionLabel.ABSTRACT,
        SectionLabel.METHODS,
        SectionLabel.RESULTS,
        SectionLabel.DISCUSSION,
        SectionLabel.OTHER,
    }
)


def classify_chunks_in_document(chunk_texts: list[str]) -> list[SectionLabel]:
    """Label each chunk in a document by walking them in order.

    The first chunk defaults to ``BODY``. Subsequent chunks inherit the
    label of their predecessor unless they begin with a recognised header.
    Documents without any detected header therefore receive ``BODY``
    everywhere — which is the correct default: any chunk that *could* be
    a body chunk should be treated as one.
    """
    current: SectionLabel = SectionLabel.BODY
    out: list[SectionLabel] = []
    for text in chunk_texts:
        label = detect_section_label(text)
        if label is not None:
            current = label
        out.append(current)
    return out


__all__ = [
    "DEFAULT_ELIGIBLE_SECTIONS",
    "SectionLabel",
    "classify_chunks_in_document",
    "detect_section_label",
]
