from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from langchain_core.documents import Document

from app.chunking.document_chunker import DocumentChunker, get_document_chunker
from app.config import Settings, get_settings
from app.extraction import (
    EmptyExtractionError,
    WebBaseExtractor,
    get_extractor,
)
from app.ingestion.document_id import generate_text_doc_id, generate_url_doc_id
from app.ingestion.models import (
    DocumentError,
    ExtractedDocument,
    IngestDocument,
    IngestResult,
)


class IngestValidationError(ValueError):
    """Raised when raw document input is structurally invalid (before doc_id)."""


@dataclass
class IngestionService:
    extractor: WebBaseExtractor = field(init=False)
    settings: Settings = field(init=False)
    document_chunker: DocumentChunker = field(init=False)
    persist: Callable[[list[Document]], Awaitable[None]] | None = None
    min_content_length: int | None = None

    def __post_init__(self) -> None:
        self.settings = get_settings()
        self.extractor = get_extractor(self.settings)
        self.document_chunker = get_document_chunker(self.settings)
        self.min_content_length = self.min_content_length or self.settings.min_content_length

    async def run(self, tenant_id: str, documents: list[IngestDocument]) -> IngestResult:
        result = IngestResult()

        # Deduplicate within the batch by doc_id; first occurrence wins.
        seen: set[str] = set()
        for document in documents:
            try:
                self._validate(document)
            except IngestValidationError as exc:
                result.errors.append(DocumentError(document, "invalid_input", str(exc)))
                continue

            doc_id = self._doc_id(document)
            if doc_id in seen:
                continue
            seen.add(doc_id)

            try:
                chunk_count = await self._process_document(tenant_id, document, doc_id)
                result.documents_stored += 1
                result.chunks_stored += chunk_count
            except Exception as exc:
                result.errors.append(
                    DocumentError(document, getattr(exc, "code", "ingest_failed"), str(exc))
                )
        return result

    # ------------------------------------------------------------------

    def _validate(self, document: IngestDocument) -> None:
        if document.url and document.text:
            raise IngestValidationError("document must provide either url or text, not both")
        if not document.url and not document.text:
            raise IngestValidationError("document must provide a url or text")
        if not document.brand or not document.brand.strip():
            raise IngestValidationError("document.brand is required")

    def _doc_id(self, document: IngestDocument) -> str:
        if document.url:
            return generate_url_doc_id(document.url)
        return generate_text_doc_id(document.text or "")

    def _source(self, document: IngestDocument) -> str:
        if document.url:
            return document.url
        return document.source or "raw_text"

    async def _process_document(self, tenant_id: str, document: IngestDocument, doc_id: str) -> int:
        extracted = await self._extract(document)
        chunks = self.document_chunker.create_chunks(
            text=extracted.text,
            tenant_id=tenant_id,
            doc_id=doc_id,
            source=self._source(document),
            brand=extracted.brand,
        )
        if not chunks:
            raise ValueError(f"Document {doc_id} produced no chunks")
        if self.persist is not None:
            await self.persist(chunks)
        return len(chunks)

    async def _extract(self, document: IngestDocument) -> ExtractedDocument:
        if document.url:
            return await asyncio.to_thread(
                self.extractor.extract, document.url, document.brand
            )

        text = document.text or ""
        if len(text) < (self.min_content_length or 0):
            source = self._source(document)
            raise EmptyExtractionError(
                source,
                f"text content too short ({len(text)} chars, min {self.min_content_length})",
            )

        return ExtractedDocument(
            text=text,
            url="raw_text",
            title=None,
            brand=document.brand,
            extractor="manual",
        )


async def get_ingestion_service() -> IngestionService:
    return IngestionService()