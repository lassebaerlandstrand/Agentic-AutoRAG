"""Document parser: every supported file format becomes a ``DoclingDocument``.

Going through Docling for everything (PDF, DOCX, MD, TXT, HTML, …) gives us
typed structural items (``SectionHeaderItem``, ``TableItem``, …) downstream
chunkers and section classifiers can consume directly — without re-parsing
markdown text with regex.
"""

from __future__ import annotations

import logging
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument


def _silence_ocr_loggers() -> None:
    """Mute Docling OCR warnings and RapidOCR's INFO banner.

    RapidOCR installs its own colored StreamHandler at module-import time
    only when the ``RapidOCR`` logger has no handlers yet (see
    ``rapidocr/utils/log.py``); pre-attaching a NullHandler here pre-empts
    that handler so nothing is emitted even when its module init later
    forces the level back to INFO.
    """
    logging.getLogger("docling").setLevel(logging.ERROR)

    rapid = logging.getLogger("RapidOCR")
    for handler in list(rapid.handlers):
        if not isinstance(handler, logging.NullHandler):
            rapid.removeHandler(handler)
    if not any(isinstance(h, logging.NullHandler) for h in rapid.handlers):
        rapid.addHandler(logging.NullHandler())
    rapid.setLevel(logging.ERROR)
    rapid.propagate = False


_silence_ocr_loggers()


class DoclingParser:
    """Single parser for all supported formats.

    PDFs and images go through the PDF pipeline (with optional OCR and
    table-structure recovery). Markdown, plain text, HTML, DOCX, XLSX,
    PPTX, AsciiDoc, CSV all dispatch automatically by file extension.
    """

    DEFAULT_DOCUMENT_TIMEOUT = 120

    def __init__(self, *, ocr: bool = True, table_structure: bool = True) -> None:
        pdf_options = PdfPipelineOptions(
            do_ocr=ocr,
            do_table_structure=table_structure,
            document_timeout=self.DEFAULT_DOCUMENT_TIMEOUT,
        )
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
            },
        )

    def parse(self, file_path: Path) -> DoclingDocument:
        """Convert a file to a ``DoclingDocument`` with typed structure."""
        result = self._converter.convert(str(file_path))
        return result.document

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
            # Markdown / plain text via Docling's MD backend
            ".md",
            ".txt",
            ".text",
            # Images (OCR)
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".tif",
            ".bmp",
            ".webp",
        }


def build_parser(**kwargs) -> DoclingParser:
    """Instantiate the parser. Extra kwargs forward to ``DoclingParser``."""
    return DoclingParser(**kwargs)


def get_corpus_extensions(corpus_path: Path) -> set[str]:
    """Return the set of file extensions present in *corpus_path*.

    Used to validate corpus contents against parser support before a run.
    """
    return {p.suffix.lower() for p in corpus_path.rglob("*") if p.is_file() and p.suffix}


def validate_parser_for_corpus(corpus_path: Path) -> list[str]:
    """Return the list of corpus extensions the parser cannot handle.

    An empty list means the parser handles every file type in the corpus.
    """
    extensions = get_corpus_extensions(corpus_path)
    parser = DoclingParser()
    return sorted(extensions - parser.supported_extensions())
