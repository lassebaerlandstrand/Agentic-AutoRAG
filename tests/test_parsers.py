"""Tests for engine/parsers.py — parser classes, registry, and corpus validation."""

from pathlib import Path

import pytest

from agentic_autorag.engine.parsers import (
    PARSER_REGISTRY,
    BaseParser,
    DoclingParser,
    PyMuPDF4LLMParser,
    build_parser,
    get_corpus_extensions,
    validate_parsers_for_corpus,
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
    """Create a corpus directory with several file types."""
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "report.docx").write_bytes(b"fake docx")
    (tmp_path / "notes.txt").write_text("plain text")
    (tmp_path / "readme.md").write_text("# Readme")
    (tmp_path / "metadata.json").write_text("{}")
    return tmp_path


@pytest.fixture
def pdf_only_corpus(tmp_path: Path) -> Path:
    """Create a corpus directory with only PDF files."""
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "metadata.json").write_text("{}")
    return tmp_path


class TestBaseParser:
    def test_parse_not_implemented(self, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError):
            BaseParser().parse(tmp_path / "any.pdf")

    def test_supported_extensions_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            BaseParser().supported_extensions()


class TestParserRegistry:
    def test_registry_keys(self) -> None:
        assert "pymupdf4llm" in PARSER_REGISTRY
        assert "docling" in PARSER_REGISTRY

    def test_registry_values_are_subclasses(self) -> None:
        for cls in PARSER_REGISTRY.values():
            assert issubclass(cls, BaseParser)

    def test_build_parser_pymupdf(self) -> None:
        parser = build_parser("pymupdf4llm")
        assert isinstance(parser, PyMuPDF4LLMParser)

    def test_build_parser_docling(self) -> None:
        parser = build_parser("docling")
        assert isinstance(parser, DoclingParser)

    def test_build_parser_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown parser 'nonexistent'"):
            build_parser("nonexistent")


class TestSupportedExtensions:
    def test_pymupdf4llm_extensions(self) -> None:
        assert PyMuPDF4LLMParser().supported_extensions() == {".pdf"}

    def test_docling_extensions(self) -> None:
        exts = DoclingParser().supported_extensions()
        # Document formats
        assert {".pdf", ".docx", ".xlsx", ".pptx", ".html", ".xhtml", ".csv", ".adoc", ".asciidoc"} <= exts
        # Image formats (OCR)
        assert {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"} <= exts


@pytest.mark.slow
class TestPyMuPDF4LLMParsePdf:
    def test_parse_returns_nonempty_text(self, sample_pdf: Path) -> None:
        parser = PyMuPDF4LLMParser()
        text = parser.parse(sample_pdf)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_parse_contains_expected_content(self, sample_pdf: Path) -> None:
        parser = PyMuPDF4LLMParser()
        text = parser.parse(sample_pdf)
        assert "Retrieval-Augmented Generation" in text


@pytest.mark.slow
class TestDoclingParsePdf:
    def test_parse_returns_nonempty_text(self, sample_pdf: Path) -> None:
        parser = DoclingParser()
        text = parser.parse(sample_pdf)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_parse_contains_expected_content(self, sample_pdf: Path) -> None:
        parser = DoclingParser()
        text = parser.parse(sample_pdf)
        assert "Retrieval-Augmented Generation" in text


class TestGetCorpusExtensions:
    def test_mixed_corpus(self, mixed_corpus: Path) -> None:
        exts = get_corpus_extensions(mixed_corpus)
        assert exts == {".pdf", ".docx"}

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert get_corpus_extensions(tmp_path) == set()

    def test_only_bypass_files(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("text")
        (tmp_path / "readme.md").write_text("# hi")
        (tmp_path / "metadata.json").write_text("{}")
        assert get_corpus_extensions(tmp_path) == set()

    def test_nested_directories(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "deep.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "top.html").write_text("<html></html>")
        assert get_corpus_extensions(tmp_path) == {".pdf", ".html"}


class TestValidateParsersForCorpus:
    def test_all_pdf_corpus(self, pdf_only_corpus: Path) -> None:
        result = validate_parsers_for_corpus(["pymupdf4llm", "docling"], pdf_only_corpus)
        assert result["pymupdf4llm"] == []
        assert result["docling"] == []

    def test_mixed_corpus_pymupdf_incompatible(self, mixed_corpus: Path) -> None:
        result = validate_parsers_for_corpus(["pymupdf4llm", "docling"], mixed_corpus)
        assert result["pymupdf4llm"] == [".docx"]
        assert result["docling"] == []

    def test_unknown_parser_raises(self, pdf_only_corpus: Path) -> None:
        with pytest.raises(ValueError, match="Unknown parser"):
            validate_parsers_for_corpus(["nonexistent"], pdf_only_corpus)
