import hashlib


def _sha256(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


def generate_url_doc_id(url: str) -> str:
    """Generate a stable doc_id from a document URL (idempotent across re-ingest)."""
    return _sha256(url)


def generate_text_doc_id(text: str) -> str:
    """Generate a stable doc_id from raw text content (idempotent across re-ingest)."""
    return _sha256(text)