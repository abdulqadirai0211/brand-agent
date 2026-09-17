import pytest
from types import SimpleNamespace

from app.config import Settings
from app.extraction import (
    EmptyExtractionError,
    FetchFailedError,
    UnsafeURLError,
    WebBaseExtractor,
    get_extractor,
    validate_fetch_url,
)
from app.ingestion.models import ExtractedDocument

CONTENT = "WebBaseLoader extracted this long enough piece of text. " * 20


class FakeLoader:
    def __init__(self, docs=None, exc=None):
        self.docs = docs if docs is not None else [fake_doc()]
        self.exc = exc

    def load(self):
        if self.exc is not None:
            raise self.exc
        return self.docs


def fake_doc(content=CONTENT, title="Asana Reviews"):
    return SimpleNamespace(
        page_content=content,
        metadata={"title": title, "language": "en", "description": "desc"},
    )


def make_extractor(*, min_len=200, loader_factory=None):
    return WebBaseExtractor(
        min_content_length=min_len,
        loader_factory=loader_factory,
    )


def make_loader_factory(exc=None, docs=None):
    calls = []

    def factory(url):
        calls.append(url)
        return FakeLoader(docs=docs, exc=exc)

    return calls, factory


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------


def test_extract_returns_document():
    calls, factory = make_loader_factory()
    extractor = make_extractor(loader_factory=factory)

    doc = extractor.extract("https://example.com/asana", "Asana")

    assert isinstance(doc, ExtractedDocument)
    assert doc.text == CONTENT.strip()
    assert doc.url == "https://example.com/asana"
    assert doc.title == "Asana Reviews"
    assert doc.brand == "Asana"
    assert doc.extractor == "webbase_loader"
    assert calls == ["https://example.com/asana"]


def test_extract_uses_first_document_only():
    docs = [fake_doc(content="first part " * 30, title="t1"), fake_doc(content="second part " * 30, title="t2")]
    _, factory = make_loader_factory(docs=docs)
    extractor = make_extractor(loader_factory=factory)

    doc = extractor.extract("https://example.com/asana", "Asana")

    assert "first part" in doc.text
    assert doc.title == "t1"


def test_empty_content_raises_empty_extraction():
    _, factory = make_loader_factory(docs=[fake_doc(content="")])
    extractor = make_extractor(loader_factory=factory)

    with pytest.raises(EmptyExtractionError) as excinfo:
        extractor.extract("https://example.com/asana", "Asana")

    assert excinfo.value.code == "empty_extraction"


def test_short_content_raises_empty_extraction():
    _, factory = make_loader_factory(docs=[fake_doc(content="tiny")])
    extractor = make_extractor(min_len=200, loader_factory=factory)

    with pytest.raises(EmptyExtractionError) as excinfo:
        extractor.extract("https://example.com/asana", "Asana")

    assert excinfo.value.code == "empty_extraction"


def test_loader_exception_wrapped_as_fetch_failed():
    _, factory = make_loader_factory(exc=TimeoutError("request timed out"))
    extractor = make_extractor(loader_factory=factory)

    with pytest.raises(FetchFailedError) as excinfo:
        extractor.extract("https://example.com/asana", "Asana")

    assert excinfo.value.code == "fetch_failed"
    assert "timed out" in str(excinfo.value)


def test_no_documents_raises_fetch_failed():
    _, factory = make_loader_factory(docs=[])
    extractor = make_extractor(loader_factory=factory)

    with pytest.raises(FetchFailedError):
        extractor.extract("https://example.com/asana", "Asana")


def test_metadata_missing_fields_tolerated():
    _, factory = make_loader_factory(docs=[SimpleNamespace(page_content=CONTENT, metadata={})])
    extractor = make_extractor(loader_factory=factory)

    doc = extractor.extract("https://example.com/asana", "Asana")

    assert doc.title is None
    assert doc.text == CONTENT.strip()


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "javascript:alert(1)",
        "",
        "not a url",
        "http:///nohost",
    ],
)
def test_invalid_urls_rejected(url):
    with pytest.raises(UnsafeURLError):
        validate_fetch_url(url)


def test_extract_rejects_invalid_url_before_fetching():
    calls, factory = make_loader_factory()
    extractor = make_extractor(loader_factory=factory)

    with pytest.raises(UnsafeURLError):
        extractor.extract("http:///nohost", "Asana")

    assert calls == []


@pytest.mark.parametrize(
    "url",
    [
        "https://www.langchain.com",
        "http://example.com/",
        "https://example.com:8080/x",
        "http://localhost/asana",
    ],
)
def test_valid_http_urls_allowed(url):
    assert validate_fetch_url(url) == url


# ---------------------------------------------------------------------------
# construction / factory
# ---------------------------------------------------------------------------


def test_from_settings(settings: Settings):
    extractor = WebBaseExtractor.from_settings(settings)

    assert extractor.timeout == 30
    assert extractor.min_content_length == 200
    assert extractor.user_agent == settings.web_user_agent


def test_get_extractor_returns_webbaseloader(settings: Settings):
    extractor = get_extractor(settings)

    assert isinstance(extractor, WebBaseExtractor)


def test_loader_uses_configured_user_agent(settings: Settings):
    extractor = WebBaseExtractor.from_settings(settings)
    loader = extractor._build_loader("https://x.test/")

    headers = loader.requests_kwargs["headers"]

    assert headers["User-Agent"] == settings.web_user_agent