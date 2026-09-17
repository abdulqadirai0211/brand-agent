from langchain_core.documents import Document

from app.chunking.splitter import TextChunker
from app.config import Settings, get_settings
from app.ingestion.models import ChunkMetadata


class DocumentChunker:
    """Split a document and attach required metadata to every chunk."""

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ):
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "DocumentChunker":
        settings = settings or get_settings()
        return cls(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def create_chunks(
        self,
        text: str,
        tenant_id: str,
        doc_id: str,
        source: str,
        brand: str,
    ) -> list[Document]:
        chunks = self.chunker.split(text)

        documents = []
        for chunk_index, chunk in enumerate(chunks):
            metadata = ChunkMetadata(
                tenant_id=tenant_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
                source=source,
                brand=brand,
            )

            documents.append(
                Document(
                    page_content=chunk,
                    metadata=metadata.model_dump(),
                )
            )

        return documents


def get_document_chunker(settings: Settings | None = None) -> DocumentChunker:
    return DocumentChunker.from_settings(settings)