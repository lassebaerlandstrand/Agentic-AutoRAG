"""End-to-end test: Docling MD parsing + HybridChunker + section labeling.

Verifies the full pipeline catches the section variants that broke the old
regex-on-chunk-prefix classifier — Literature, References Cited, numeric-
prefixed headers, references buried past a 200-char scan window, etc.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument

from agentic_autorag.config.models import ExaminerConfig
from agentic_autorag.engine.section_classifier import SectionLabel
from agentic_autorag.examiner.exam_agent import ExamAgent

_CONVERTER = DocumentConverter()


def _md_to_dl(markdown: str) -> DoclingDocument:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(markdown)
        path = Path(f.name)
    try:
        return _CONVERTER.convert(str(path)).document
    finally:
        path.unlink()


def _make_agent() -> ExamAgent:
    return ExamAgent(
        config=ExaminerConfig(exam_size=2, min_doc_words=1),
        examiner_model="test/model",
        corpus_description="t",
        concurrency=1,
    )


class TestSectionLabelingOnDoclingMarkdown:
    """Each markdown variant must produce at least one REFERENCES-labeled chunk."""

    def test_plain_references_header(self) -> None:
        md = (
            "# Paper Title\n\nMain body content discussing the topic in depth.\n\n"
            "## Introduction\nIntroduction body text here that elaborates on the motivation.\n\n"
            "## Results\nThe primary endpoint was met and the effect size was substantial.\n\n"
            "## References\n\n1. Smith J. Study X. Journal A, 2020.\n"
            "2. Jones K. Study Y. Journal B, 2021.\n3. Lee Q. Study Z. Journal C, 2022.\n"
        )
        chunks = _make_agent().chunk_documents([_md_to_dl(md)], ["doc.md"])
        labels = {c.section for c in chunks}
        assert SectionLabel.REFERENCES in labels

    def test_literature_header(self) -> None:
        """User's exact failure case: bare 'Literature' (not 'Literature Cited')."""
        md = (
            "# User Guide\n\nDescribes a chromatography column workflow with detailed steps.\n\n"
            "## Procedure\nStep-by-step procedure for column setup and operation in the lab.\n\n"
            "## Literature\n\nAffinity Separations: A Practical Approach 1997 (Matejtschuk Ed.).\n"
            "Protein Purification Protocols 1996 (Doonan Ed.). Recombinant Protein 1997 (Tuan Ed.).\n"
        )
        chunks = _make_agent().chunk_documents([_md_to_dl(md)], ["doc.md"])
        labels = {c.section for c in chunks}
        assert SectionLabel.REFERENCES in labels

    def test_numeric_prefix_references(self) -> None:
        md = (
            "# Article\n\nAbstract and body content here describing the work in detail.\n\n"
            "## 1. Introduction\nMotivation and prior work review of the field.\n\n"
            "## 2. Methods\nWe describe the experimental setup and procedure in detail here.\n\n"
            "## 7. References\n\n[1] Smith 2020. [2] Jones 2021. [3] Lee 2022.\n"
        )
        chunks = _make_agent().chunk_documents([_md_to_dl(md)], ["doc.md"])
        labels = {c.section for c in chunks}
        assert SectionLabel.REFERENCES in labels

    def test_acknowledgments_header(self) -> None:
        md = (
            "# Paper\n\nMain body discussing the topic in some detail at length.\n\n"
            "## Methods\nWe used a randomized controlled trial design over several months.\n\n"
            "## Acknowledgements\n\nWe thank our colleagues for helpful discussions.\n"
            "Funding was provided by grants from NIH and the company foundation directly.\n"
        )
        chunks = _make_agent().chunk_documents([_md_to_dl(md)], ["doc.md"])
        labels = {c.section for c in chunks}
        assert SectionLabel.ACKNOWLEDGMENTS in labels


class TestNoReferencesNoFalsePositive:
    """A doc without a refs section should not get any REFERENCES chunks."""

    def test_manual_with_only_body_sections(self) -> None:
        md = (
            "# Spa Controller User Manual\n\nThis manual describes setup and operation.\n\n"
            "## Error Codes\n- Code FL: Flow problem. Power off and check filter.\n"
            "- Code OH: Overheat. Allow cooling time and reset breaker.\n"
            "- Code dr: Low water. Refill the tank to the proper level mark.\n\n"
            "## Maintenance\nReplace the filter every 90 days and clean the basket weekly.\n"
        )
        chunks = _make_agent().chunk_documents([_md_to_dl(md)], ["manual.md"])
        labels = {c.section for c in chunks}
        assert SectionLabel.REFERENCES not in labels


class TestChunkerProducesUsableChunks:
    """Make sure the new chunker actually produces non-empty chunks with the
    expected ChunkRecord shape downstream consumers depend on."""

    def test_chunks_have_required_fields(self) -> None:
        md = "# Title\n\nBody paragraph with substantive content for question generation.\n"
        chunks = _make_agent().chunk_documents([_md_to_dl(md)], ["doc.md"])
        assert chunks
        for c in chunks:
            assert c.chunk_id.startswith("doc.md::chunk_")
            assert c.doc_id == "doc.md"
            assert c.text
            assert c.section is not None
