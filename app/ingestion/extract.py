"""Deterministic document ingestion.

Format handling (PDF/DOCX/plain text) is done here in code, not by an LLM —
per the "LLMs reason, tools do deterministic work" principle. OCR/image
input is intentionally out of scope for this slice; a scanned/near-empty
PDF surfaces a warning instead of silently failing.
"""

from pathlib import Path

import docx
import pymupdf as fitz

from app.schemas.document import ExtractedDocument, SourceFormat

MIN_CHARS_PER_PAGE_WARNING = 20


def extract_pdf_text(path: str | Path) -> ExtractedDocument:
    warnings: list[str] = []
    doc = fitz.open(path)
    try:
        page_texts = [page.get_text() for page in doc]
        page_count = doc.page_count
    finally:
        doc.close()

    raw_text = "\n".join(page_texts).strip()
    avg_chars = len(raw_text) / max(page_count, 1)
    if avg_chars < MIN_CHARS_PER_PAGE_WARNING:
        warnings.append(
            "Very little text extracted per page — this PDF may be a scanned "
            "image without a text layer. OCR is not supported in this build."
        )

    return ExtractedDocument(
        source_format=SourceFormat.PDF,
        raw_text=raw_text,
        page_count=page_count,
        filename=Path(path).name,
        extraction_warnings=warnings,
    )


def extract_docx_text(path: str | Path) -> ExtractedDocument:
    document = docx.Document(str(path))

    parts: list[str] = []
    for para in document.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    raw_text = "\n".join(parts).strip()
    warnings = []
    if not raw_text:
        warnings.append("No extractable text found in DOCX file.")

    return ExtractedDocument(
        source_format=SourceFormat.DOCX,
        raw_text=raw_text,
        page_count=1,
        filename=Path(path).name,
        extraction_warnings=warnings,
    )


def extract_plain_text(text: str, filename: str | None = None) -> ExtractedDocument:
    raw_text = text.strip()
    warnings = [] if raw_text else ["Empty text input."]
    return ExtractedDocument(
        source_format=SourceFormat.TEXT,
        raw_text=raw_text,
        page_count=1,
        filename=filename,
        extraction_warnings=warnings,
    )


def _extract_txt_file(path: str | Path) -> ExtractedDocument:
    return extract_plain_text(Path(path).read_text(), filename=Path(path).name)


EXTENSION_HANDLERS = {
    ".pdf": extract_pdf_text,
    ".docx": extract_docx_text,
    ".txt": _extract_txt_file,
}


def extract_from_file(path: str | Path) -> ExtractedDocument:
    ext = Path(path).suffix.lower()
    handler = EXTENSION_HANDLERS.get(ext)
    if handler is None:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: {sorted(EXTENSION_HANDLERS)}"
        )
    return handler(path)
