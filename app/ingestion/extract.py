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
from app.security import clean_untrusted_text, neutralize_injection

MIN_CHARS_PER_PAGE_WARNING = 20
MIN_VISIBLE_FONT_PT = 2.0
WHITE_RGB = 0xFFFFFF


def _hidden_text(doc) -> tuple[list[str], int]:
    """Returns (spans smaller than anyone can read, number of characters of
    pure-white text). Both are the usual ways to stuff keywords into a resume
    for software while a person sees nothing."""
    tiny: list[str] = []
    white_chars = 0
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    if span.get("size", 12) < MIN_VISIBLE_FONT_PT:
                        tiny.append(text)
                    elif span.get("color") == WHITE_RGB:
                        white_chars += len(text)
    return tiny, white_chars


def _sanitize(doc: ExtractedDocument) -> ExtractedDocument:
    """Single choke point: every document is stripped of invisible characters
    and instruction-like phrases before any agent sees it."""
    text, replaced = neutralize_injection(clean_untrusted_text(doc.raw_text))
    warnings = list(doc.extraction_warnings)
    if replaced:
        warnings.append(f"Removed {replaced} instruction-like phrase(s) aimed at AI systems.")
    return doc.model_copy(update={"raw_text": text, "extraction_warnings": warnings})


def extract_pdf_text(path: str | Path) -> ExtractedDocument:
    warnings: list[str] = []
    doc = fitz.open(path)
    try:
        page_texts = [page.get_text() for page in doc]
        page_count = doc.page_count
        tiny_spans, white_chars = _hidden_text(doc)
    finally:
        doc.close()

    raw_text = "\n".join(page_texts).strip()
    if tiny_spans:
        for span_text in tiny_spans:
            raw_text = raw_text.replace(span_text, "")
        raw_text = raw_text.strip()
        warnings.append(
            f"Removed {sum(len(s) for s in tiny_spans)} characters of microscopic text "
            f"(under {MIN_VISIBLE_FONT_PT}pt) that a person cannot read but software can."
        )
    if white_chars:
        warnings.append(
            f"Found {white_chars} characters of white text. It was kept because white text "
            "can be legitimate (for example on a coloured header), but it may be hidden keywords."
        )
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
    return _sanitize(
        ExtractedDocument(
            source_format=SourceFormat.TEXT,
            raw_text=raw_text,
            page_count=1,
            filename=filename,
            extraction_warnings=warnings,
        )
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
    return _sanitize(handler(path))
