from dataclasses import dataclass, field

from pydantic import BaseModel


class ExtractedDocument(BaseModel):
    text: str
    url: str
    title: str | None = None
    brand: str
    extractor: str


class ChunkMetadata(BaseModel):
    tenant_id: str
    doc_id: str
    chunk_index: int
    source: str
    brand: str


@dataclass
class IngestDocument:
    url: str | None = None
    text: str | None = None
    brand: str = ""
    source: str | None = None


@dataclass
class DocumentError:
    document: IngestDocument
    code: str
    message: str


@dataclass
class IngestResult:
    documents_stored: int = 0
    chunks_stored: int = 0
    errors: list[DocumentError] = field(default_factory=list)