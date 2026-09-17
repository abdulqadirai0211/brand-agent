from __future__ import annotations

import os
from typing import Any, Callable
from urllib.parse import urlparse

from app.config import Settings, get_settings
from app.ingestion.models import ExtractedDocument

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 "
    "(Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/151.0 Safari/537.36"
)

os.environ.setdefault("USER_AGENT", DEFAULT_USER_AGENT)


class ExtractionError(Exception):
    code: str = "extraction_failed"

    def __init__(self, url: str, reason: str | None = None) -> None:
        self.url = url
        self.reason = reason
        message = f"Extraction failed for {url}"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class UnsafeURLError(ExtractionError):
    code = "invalid_url"


class FetchFailedError(ExtractionError):
    code = "fetch_failed"


class EmptyExtractionError(ExtractionError):
    code = "empty_extraction"


def validate_fetch_url(url: str) -> str:
    """Rejects non-HTTP(s) URLs and URLs without a hostname."""
    if not url or not url.strip():
        raise UnsafeURLError(url, "URL is empty")
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise UnsafeURLError(url, f"unsupported scheme {parsed.scheme!r}")
    if not parsed.hostname:
        raise UnsafeURLError(url, "missing hostname")
    return url


class WebBaseExtractor:
    """
    Extract readable text from web pages using LangChain WebBaseLoader.

    Mirrors the ingestion pipeline in ``experiment.ipynb``.
    """

    def __init__(
        self,
        timeout: int = 30,
        min_content_length: int = 200,
        user_agent: str = DEFAULT_USER_AGENT,
        loader_factory: Callable[[str], Any] | None = None,
    ):
        self.timeout = timeout
        self.min_content_length = min_content_length
        self.user_agent = user_agent
        self._loader_factory = loader_factory or self._build_loader

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "WebBaseExtractor":
        settings = settings or get_settings()
        return cls(
            timeout=max(1, settings.web_loader_timeout_ms // 1000),
            min_content_length=settings.min_content_length,
            user_agent=settings.web_user_agent,
        )

    def _build_loader(self, url: str) -> Any:
        from langchain_community.document_loaders import WebBaseLoader

        return WebBaseLoader(
            web_paths=(url,),
            requests_kwargs={
                "timeout": self.timeout,
                "headers": {
                    "User-Agent": self.user_agent,
                },
            },
        )

    def extract(self, url: str, brand: str) -> ExtractedDocument:
        """
        Fetch a URL and return an ``ExtractedDocument``.

        ``brand`` is attached so every downstream chunk carries it.
        """
        validate_fetch_url(url)
        loader = self._loader_factory(url)

        try:
            documents = loader.load()
        except Exception as exc:
            raise FetchFailedError(url=url, reason=str(exc)) from exc

        if not documents:
            raise FetchFailedError(url=url, reason="no content extracted")

        document = documents[0]
        text = (getattr(document, "page_content", "") or "").strip()
        if len(text) < self.min_content_length:
            raise EmptyExtractionError(
                url,
                f"content too short ({len(text)} chars, min {self.min_content_length})",
            )

        return ExtractedDocument(
            text=text,
            url=url,
            title=(getattr(document, "metadata", {}) or {}).get("title"),
            brand=brand,
            extractor="webbase_loader",
        )


def get_extractor(settings: Settings | None = None) -> WebBaseExtractor:
    return WebBaseExtractor.from_settings(settings)