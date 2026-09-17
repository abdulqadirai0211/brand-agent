"""Seed a tenant's corpus through the production ingestion service.

The only ingestion implementation lives in `app.ingestion.service`; this script
drives it (it does not reimplement extraction, chunking, or storage).

Usage:
    venv/bin/python scripts/seed_corpus.py
    venv/bin/python scripts/seed_corpus.py --tenant acme
    venv/bin/python scripts/seed_corpus.py --manifest data/corpus.json

A manifest is JSON:

    {
      "tenant_id": "acme",
      "documents": [
        {"url": "https://example.com/best-pm-tools", "brand": "Asana"},
        {"text": "Trello is ...", "brand": "Trello", "source": "notes.txt"}
      ]
    }

With no manifest, a small built-in project-management corpus is used as a
starting point (see BUILTIN_DOCUMENTS below).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.models import IngestDocument
from app.ingestion.service import get_ingestion_service
from app.vectorstore.pinecone import get_vector_store

BUILTIN_TENANT = "acme"

# Project-management tools: comparison/review pages covering two or more brands.
# Replace or extend with your own URLs; a manifest keeps the corpus out of code.
BUILTIN_DOCUMENTS: list[dict[str, str]] = [
    {"url": "https://asana.com/pricing", "brand": "Asana"},
    {"url": "https://trello.com/pricing", "brand": "Trello"},
]


def _load_manifest(path: Path) -> tuple[str, list[IngestDocument]]:
    payload = json.loads(path.read_text())
    tenant_id = payload.get("tenant_id", BUILTIN_TENANT)
    documents = [
        IngestDocument(
            url=item.get("url"),
            text=item.get("text"),
            brand=item.get("brand", ""),
            source=item.get("source"),
        )
        for item in payload.get("documents", [])
    ]
    return tenant_id, documents


def _builtin_documents() -> list[IngestDocument]:
    return [
        IngestDocument(url=item.get("url"), text=item.get("text"), brand=item["brand"])
        for item in BUILTIN_DOCUMENTS
    ]


async def main() -> int:
    parser = argparse.ArgumentParser(description="Seed a tenant corpus via the ingestion service")
    parser.add_argument("--tenant", default=BUILTIN_TENANT, help="tenant_id (overridden by manifest)")
    parser.add_argument("--manifest", type=Path, default=None, help="path to a JSON corpus manifest")
    args = parser.parse_args()

    if args.manifest is not None:
        tenant_id, documents = _load_manifest(args.manifest)
    else:
        tenant_id, documents = args.tenant, _builtin_documents()

    if not documents:
        print("No documents to ingest.")
        return 1

    store = get_vector_store()
    await asyncio.to_thread(store.ensure_index)

    service = await get_ingestion_service()
    service.persist = store.persist_chunks

    print(f"Ingesting {len(documents)} document(s) into tenant '{tenant_id}' ...")
    result = await service.run(tenant_id, documents)

    print(f"documents_stored : {result.documents_stored}")
    print(f"chunks_stored    : {result.chunks_stored}")
    for error in result.errors:
        target = error.document.url or error.document.source or "<text>"
        print(f"  error [{error.code}] {target}: {error.message}")

    total = await asyncio.to_thread(store.count, tenant_id)
    print(f"tenant vectors   : {total}")
    return 0 if not result.errors else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
