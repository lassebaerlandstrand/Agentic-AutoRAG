"""Document parsing backends for converting raw files to plain text.

Each parser implements a common interface so they are swappable via the
``structural.parsers`` list in the YAML config. Markdown and plain text
files bypass the parser entirely (handled by the orchestrator/corpus loader).
"""

from pathlib import Path

# Extensions that bypass the parser (ingested directly or are metadata).
BYPASS_EXTENSIONS: set[str] = {".md", ".txt", ".json"}


class BaseParser:
    """Common interface for document parsers."""

    def parse(self, file_path: Path) -> str:
        """Convert a file to plain text. Returns the extracted text."""
        raise NotImplementedError

    def supported_extensions(self) -> set[str]:
        """Return the set of file extensions this parser can handle."""
        raise NotImplementedError


class PyMuPDF4LLMParser(BaseParser):
    """PDF parser using pymupdf4llm (Markdown output with headings/tables)."""

    def parse(self, file_path: Path) -> str:
        import pymupdf4llm

        return pymupdf4llm.to_markdown(str(file_path))

    def supported_extensions(self) -> set[str]:
        return {".pdf"}


class DoclingParser(BaseParser):
    """Multi-format parser using IBM Docling.

    Supports documents (PDF, Office, HTML, CSV, AsciiDoc),
    images (via OCR), and several schema-specific XML formats.
    """

    def parse(self, file_path: Path) -> str:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        return result.document.export_to_markdown()

    def supported_extensions(self) -> set[str]:
        return {
            # Documents
            ".pdf",
            ".docx",
            ".xlsx",
            ".pptx",
            ".html",
            ".xhtml",
            ".csv",
            ".adoc",
            ".asciidoc",
            # Images (OCR)
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".tif",
            ".bmp",
            ".webp",
        }


PARSER_REGISTRY: dict[str, type[BaseParser]] = {
    "pymupdf4llm": PyMuPDF4LLMParser,
    "docling": DoclingParser,
}


def build_parser(parser_name: str) -> BaseParser:
    """Instantiate a parser by its registry name."""
    if parser_name not in PARSER_REGISTRY:
        raise ValueError(f"Unknown parser '{parser_name}'. Available: {sorted(PARSER_REGISTRY.keys())}")
    return PARSER_REGISTRY[parser_name]()


def get_corpus_extensions(corpus_path: Path) -> set[str]:
    """Return the set of file extensions in *corpus_path* that need parsing.

    Extensions that bypass the parser (``.md``, ``.txt``) and metadata
    files (``.json``) are excluded.
    """
    extensions = {p.suffix.lower() for p in corpus_path.rglob("*") if p.is_file() and p.suffix}
    return extensions - BYPASS_EXTENSIONS


def validate_parsers_for_corpus(
    parser_names: list[str],
    corpus_path: Path,
) -> dict[str, list[str]]:
    """Check each parser against the file types present in *corpus_path*.

    Returns a dict mapping each parser name to the list of corpus file
    extensions it **cannot** handle.  An empty list means the parser is
    fully compatible with the corpus.

    Raises ``ValueError`` if a parser name is not in the registry.
    """
    extensions = get_corpus_extensions(corpus_path)

    result: dict[str, list[str]] = {}
    for name in parser_names:
        parser = build_parser(name)
        unsupported = sorted(extensions - parser.supported_extensions())
        result[name] = unsupported
    return result
