import asyncio
import hashlib

from langchain_core.documents import Document

from app.extraction import FetchFailedError
from app.ingestion.models import ExtractedDocument, IngestDocument
from app.ingestion.service import IngestionService

TEXT = "Acme pricing page with lots of text. " * 30


class FakeExtractor:
    def __init__(self, doc=None, exc=None):
        self.doc = doc or ExtractedDocument(
            text=TEXT,
            url="https://example.com/asana",
            title="Asana",
            brand="Asana",
            extractor="webbase_loader",
        )
        self.exc = exc
        self.calls = []

    def extract(self, url, brand):
        self.calls.append((url, brand))
        if self.exc is not None:
            raise self.exc
        return self.doc


class FakeDocumentChunker:
    def __init__(self, n=2):
        self.n = n
        self.calls = []

    def create_chunks(self, *, text, tenant_id, doc_id, source, brand):
        self.calls.append(
            {"text": text, "tenant_id": tenant_id, "doc_id": doc_id, "source": source, "brand": brand}
        )
        return [
            Document(
                page_content=f"chunk {i}",
                metadata={
                    "tenant_id": tenant_id,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "source": source,
                    "brand": brand,
                },
            )
            for i in range(self.n)
        ]


class FakePersist:
    def __init__(self):
        self.batches = []

    async def __call__(self, chunks):
        self.batches.append(list(chunks))


def make_service(*, document_chunker=None, persist=None, extractor=None, min_length=10):
    service = IngestionService(min_content_length=min_length)
    service.document_chunker = document_chunker or FakeDocumentChunker()
    service.persist = persist
    if extractor is not None:
        service.extractor = extractor
    return service


def run(service, tenant_id, documents):
    return asyncio.run(service.run(tenant_id, documents))


# ---------------------------------------------------------------------------
# text path
# ---------------------------------------------------------------------------


def test_text_document_ingested():
    persist = FakePersist()
    service = make_service(persist=persist)

    result = run(service, "acme", [IngestDocument(text=TEXT, brand="Acme")])

    assert result.errors == []
    assert result.documents_stored == 1
    assert result.chunks_stored == 2
    assert persist.batches[0][0].metadata["source"] == "raw_text"


def test_text_document_uses_explicit_source():
    service = make_service()
    result = run(service, "acme", [IngestDocument(text=TEXT, brand="Acme", source="my-source")])

    assert result.errors == []
    assert service.document_chunker.calls[0]["source"] == "my-source"


def test_text_doc_id_deterministic():
    persist = FakePersist()
    service = make_service(persist=persist)
    run(service, "acme", [IngestDocument(text=TEXT, brand="Acme")])
    d1 = persist.batches[0][0].metadata["doc_id"]

    persist2 = FakePersist()
    service2 = make_service(persist=persist2)
    run(service2, "acme", [IngestDocument(text=TEXT, brand="Acme")])
    d2 = persist2.batches[0][0].metadata["doc_id"]

    expected = hashlib.sha256(TEXT.strip().encode("utf-8")).hexdigest()
    assert d1 == d2 == expected


def test_text_too_short_reported_as_error():
    service = make_service(min_length=200)
    result = run(service, "acme", [IngestDocument(text="tiny", brand="Acme")])

    assert result.documents_stored == 0
    assert result.errors[0].code == "empty_extraction"


# ---------------------------------------------------------------------------
# url path
# ---------------------------------------------------------------------------


def test_url_document_uses_extractor():
    extractor = FakeExtractor()
    persist = FakePersist()
    service = make_service(extractor=extractor, persist=persist)

    result = run(service, "acme", [IngestDocument(url="https://example.com/asana", brand="Asana")])

    assert service.extractor.calls == [("https://example.com/asana", "Asana")]
    assert result.errors == []
    assert result.documents_stored == 1
    chunk = persist.batches[0][0]
    assert chunk.metadata["source"] == "https://example.com/asana"
    assert chunk.metadata["brand"] == "Asana"
    expected = hashlib.sha256(b"https://example.com/asana").hexdigest()
    assert chunk.metadata["doc_id"] == expected


def test_url_extractor_failure_reported_and_others_continue():
    extractor = FakeExtractor(exc=FetchFailedError("https://bad.com", "timed out"))
    service = make_service(extractor=extractor)

    result = run(
        service,
        "acme",
        [
            IngestDocument(url="https://bad.com", brand="A"),
            IngestDocument(text=TEXT, brand="B"),
        ],
    )

    assert result.documents_stored == 1
    assert result.errors[0].code == "fetch_failed"
    assert result.errors[0].document.url == "https://bad.com"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_both_url_and_text_rejected():
    service = make_service()
    result = run(service, "acme", [IngestDocument(url="https://x.com", text=TEXT, brand="A")])

    assert result.errors[0].code == "invalid_input"


def test_neither_url_nor_text_rejected():
    service = make_service()
    result = run(service, "acme", [IngestDocument(brand="A")])

    assert result.errors[0].code == "invalid_input"


def test_missing_brand_rejected():
    service = make_service()
    result = run(service, "acme", [IngestDocument(text=TEXT, brand="")])

    assert result.errors[0].code == "invalid_input"


def test_bad_scheme_rejected():
    service = make_service()
    result = run(service, "acme", [IngestDocument(url="ftp://x.com/a", brand="A")])

    assert result.errors[0].code == "invalid_url"


# ---------------------------------------------------------------------------
# dedupe + persistence
# ---------------------------------------------------------------------------


def test_duplicate_documents_in_batch_collapse():
    persist = FakePersist()
    service = make_service(persist=persist)

    result = run(
        service,
        "acme",
        [
            IngestDocument(text=TEXT, brand="Acme"),
            IngestDocument(text=TEXT, brand="Acme"),
        ],
    )

    assert result.documents_stored == 1
    assert len(persist.batches) == 1


def test_persist_receives_all_chunks_in_order():
    persist = FakePersist()
    service = make_service(persist=persist)

    result = run(service, "acme", [IngestDocument(text=TEXT, brand="Acme")])

    assert len(persist.batches) == 1
    assert result.chunks_stored == 2
    doc_id = persist.batches[0][0].metadata["doc_id"]
    assert [c.metadata["chunk_index"] for c in persist.batches[0]] == [0, 1]
    assert all(c.metadata["doc_id"] == doc_id for c in persist.batches[0])


def test_chunk_metadata_contract():
    persist = FakePersist()
    service = make_service(persist=persist)

    run(service, "acme", [IngestDocument(text=TEXT, brand="Acme", source="manual-x")])
    md = persist.batches[0][0].metadata

    assert set(md) == {"tenant_id", "doc_id", "chunk_index", "source", "brand"}
    assert md["tenant_id"] == "acme"
    assert md["doc_id"] == persist.batches[0][0].metadata["doc_id"]
    assert md["chunk_index"] == 0
    assert md["brand"] == "Acme"
    assert md["source"] == "manual-x"