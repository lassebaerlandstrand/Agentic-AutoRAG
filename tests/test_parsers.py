"""Tests for engine/parsers.py — Docling parser + corpus validation."""

from pathlib import Path

import pytest
from docling_core.types.doc.document import DoclingDocument

from agentic_autorag.engine.parsers import (
    DoclingParser,
    build_parser,
    get_corpus_extensions,
    validate_parser_for_corpus,
)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal PDF with known text content."""
    import pymupdf

    doc = pymupdf.Document()
    page = doc.new_page()
    page.insert_text((72, 72), "Retrieval-Augmented Generation is a technique.")
    pdf_path = tmp_path / "sample.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def mixed_corpus(tmp_path: Path) -> Path:
    """Corpus directory with several file types."""
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "report.docx").write_bytes(b"fake docx")
    (tmp_path / "notes.txt").write_text("plain text")
    (tmp_path / "readme.md").write_text("# Readme")
    (tmp_path / "metadata.json").write_text("{}")
    return tmp_path


@pytest.fixture
def pdf_only_corpus(tmp_path: Path) -> Path:
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 fake")
    return tmp_path


class TestDoclingParser:
    def test_build_parser_returns_docling(self) -> None:
        parser = build_parser()
        assert isinstance(parser, DoclingParser)

    def test_build_parser_forwards_kwargs(self) -> None:
        parser = build_parser(ocr=False, table_structure=False)
        assert isinstance(parser, DoclingParser)


class TestSupportedExtensions:
    def test_extensions_cover_all_target_formats(self) -> None:
        exts = DoclingParser().supported_extensions()
        # Document formats
        assert {".pdf", ".docx", ".xlsx", ".pptx", ".html", ".xhtml", ".csv", ".adoc", ".asciidoc"} <= exts
        # Markdown / plain text
        assert {".md", ".txt", ".text"} <= exts
        # Image formats (OCR)
        assert {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"} <= exts


@pytest.mark.slow
class TestDoclingParsePdf:
    def test_parse_returns_docling_document(self, sample_pdf: Path) -> None:
        parser = DoclingParser()
        dl_doc = parser.parse(sample_pdf)
        assert isinstance(dl_doc, DoclingDocument)
        assert dl_doc.export_to_markdown().strip()

    def test_parse_contains_expected_content(self, sample_pdf: Path) -> None:
        parser = DoclingParser()
        dl_doc = parser.parse(sample_pdf)
        assert "Retrieval-Augmented Generation" in dl_doc.export_to_markdown()


class TestParseMarkdown:
    def test_parse_md_returns_docling_document_with_heading(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Methods\n\nWe describe the procedure.\n\n## References\n\n1. Smith 2020")
        parser = DoclingParser()
        dl_doc = parser.parse(md)
        assert isinstance(dl_doc, DoclingDocument)
        out = dl_doc.export_to_markdown()
        assert "Methods" in out
        assert "References" in out


class TestGetCorpusExtensions:
    def test_mixed_corpus(self, mixed_corpus: Path) -> None:
        exts = get_corpus_extensions(mixed_corpus)
        # Now md/txt count too because Docling handles them.
        assert {".pdf", ".docx", ".txt", ".md", ".json"} == exts

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert get_corpus_extensions(tmp_path) == set()

    def test_nested_directories(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "deep.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "top.html").write_text("<html></html>")
        assert get_corpus_extensions(tmp_path) == {".pdf", ".html"}


class TestValidateParserForCorpus:
    def test_pdf_only_is_compatible(self, pdf_only_corpus: Path) -> None:
        assert validate_parser_for_corpus(pdf_only_corpus) == []

    def test_mixed_corpus_flags_unsupported(self, tmp_path: Path) -> None:
        # .json is not in DoclingParser.supported_extensions().
        (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "metadata.json").write_text("{}")
        assert validate_parser_for_corpus(tmp_path) == [".json"]
