from langchain_core.documents import Document

from app.chunking.document_chunker import DocumentChunker, get_document_chunker
from app.chunking.splitter import TextChunker
from app.config import Settings

TEXT = "Asana is a project management platform. " * 60


def make_chunker(chunk_size: int = 400, chunk_overlap: int = 50) -> TextChunker:
    return TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def make_document_chunker(chunk_size: int = 400, chunk_overlap: int = 50) -> DocumentChunker:
    return DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def make_chunks(dc: DocumentChunker | None = None, text: str = TEXT) -> list[Document]:
    dc = dc or make_document_chunker()
    return dc.create_chunks(
        text=text,
        tenant_id="acme",
        doc_id="doc-123",
        source="https://example.com/asana",
        brand="Asana",
    )


# ---------------------------------------------------------------------------
# TextChunker
# ---------------------------------------------------------------------------


def test_split_returns_text_chunks():
    chunks = make_chunker().split(TEXT)

    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)
    assert len(chunks) > 1


def test_chunks_are_character_bounded():
    chunker = make_chunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split(TEXT)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 100


def test_short_text_stays_single_chunk():
    chunks = make_chunker().split("Small content.")

    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# DocumentChunker
# ---------------------------------------------------------------------------


def test_create_chunks_returns_documents():
    chunks = make_chunks()

    assert len(chunks) > 1
    assert all(isinstance(c, Document) for c in chunks)


def test_chunks_have_required_metadata():
    chunks = make_chunks()

    for c in chunks:
        md = c.metadata
        assert md["tenant_id"] == "acme"
        assert md["doc_id"] == "doc-123"
        assert md["source"] == "https://example.com/asana"
        assert md["brand"] == "Asana"
        assert isinstance(md["chunk_index"], int)
        assert set(md) == {"tenant_id", "doc_id", "chunk_index", "source", "brand"}


def test_chunk_index_sequential_from_zero():
    chunks = make_chunks()

    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_chunking_is_deterministic():
    a = [c.page_content for c in make_chunks()]
    b = [c.page_content for c in make_chunks()]

    assert a == b


def test_content_preserved_across_chunks():
    chunks = make_chunks()

    joined = " ".join(c.page_content for c in chunks)
    assert "project management" in joined


def test_from_settings_uses_settings_values(settings: Settings):
    settings.chunk_size = 300
    settings.chunk_overlap = 50
    dc = DocumentChunker.from_settings(settings)

    chunks = dc.create_chunks(
        text=TEXT,
        tenant_id="acme",
        doc_id="doc-123",
        source="https://example.com/asana",
        brand="Asana",
    )

    for c in chunks:
        assert len(c.page_content) <= 300


def test_get_document_chunker_returns_document_chunker(settings: Settings):
    dc = get_document_chunker(settings)

    assert isinstance(dc, DocumentChunker)
    assert dc.chunker.splitter._chunk_size == settings.chunk_size