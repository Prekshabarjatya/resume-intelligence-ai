from enum import Enum

from pydantic import BaseModel, Field


class SourceFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TEXT = "text"


class ExtractedDocument(BaseModel):
    """Normalized representation of any uploaded resume/JD, regardless of
    original format. All downstream agents work off this, never the raw
    bytes, so extraction is the single place format differences are absorbed.
    """

    source_format: SourceFormat
    raw_text: str
    page_count: int = 1
    filename: str | None = None
    extraction_warnings: list[str] = Field(default_factory=list)


class DocumentClassification(BaseModel):
    document_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
