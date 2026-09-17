"""Demonstrate the ingestion pipeline stage by stage:

    URL -> extraction -> chunking -> embedding -> Pinecone storage

Run with:
    venv/bin/python scripts/show_pipeline.py --url https://asana.com/pricing
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking.document_chunker import get_document_chunker
from app.config import get_settings
from app.embeddings.service import get_embedding_service
from app.extraction.web_loader import get_extractor
from app.ingestion.document_id import generate_url_doc_id
from app.vectorstore.pinecone import get_vector_store


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title.upper())
    print("=" * 78)


def preview(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} more chars]"


def show_floats(values: list[float], n: int = 8) -> str:
    return "[" + ", ".join(f"{v:.6f}" for v in values[:n]) + (", ...]" if len(values) > n else "]")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Show ingestion pipeline I/O")
    parser.add_argument("--url", default="https://asana.com/pricing", help="URL to ingest")
    parser.add_argument("--tenant", default="demo-tenant", help="tenant_id")
    parser.add_argument("--brand", default="asana", help="brand")
    parser.add_argument("--chunks", type=int, default=4, help="how many chunk previews to print")
    args = parser.parse_args()

    settings = get_settings()
    banner("Settings")
    print(json.dumps({
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "min_content_length": settings.min_content_length,
        "pinecone_index": settings.pinecone_index,
        "pinecone_region": settings.pinecone_region,
        "alpha": settings.pinecone_alpha,
    }, indent=2))

    # ------------------------------------------------------------------ STEP 1
    banner(f"Step 1: EXTRACTION   ({args.url})")
    extractor = get_extractor()
    print(f"extractor class : {type(extractor).__name__}")
    document = extractor.extract(args.url, brand=args.brand)
    print(f"title           : {document.title}")
    print(f"url             : {document.url}")
    print(f"brand           : {document.brand}")
    print(f"extractor       : {document.extractor}")
    print(f"text length     : {len(document.text):,} characters")
    print()
    print("---- extracted text preview ----")
    print(preview(document.text))

    # ------------------------------------------------------------------ STEP 2
    banner("Step 2: CHUNKING")
    doc_id = generate_url_doc_id(args.url)
    chunker = get_document_chunker()
    chunks = chunker.create_chunks(
        document.text,
        tenant_id=args.tenant,
        doc_id=doc_id,
        source=document.url,
        brand=document.brand,
    )
    print(f"doc_id          : {doc_id}")
    print(f"splitter        : RecursiveCharacterTextSplitter(chunk_size={settings.chunk_size}, "
          f"chunk_overlap={settings.chunk_overlap})")
    print(f"total chunks    : {len(chunks):,}")
    print()
    shown = set()
    indices = list(range(min(args.chunks, len(chunks))))
    if len(chunks) > args.chunks:
        indices += list(range(max(args.chunks, len(chunks) - 2), len(chunks)))
    for i in indices:
        if i in shown:
            continue
        shown.add(i)
        chunk = chunks[i]
        print(f"---- chunk [{i}]  ({len(chunk.page_content)} chars) ----")
        print(f"metadata {json.dumps(chunk.metadata, indent=2)}")
        print(preview(chunk.page_content, limit=600))
        print()

    # ------------------------------------------------------------------ STEP 3
    banner("Step 3: EMBEDDING  (display only — the store re-embeds the full batch)")
    embedder = get_embedding_service()
    demo_texts = [c.page_content for c in chunks[:2]]
    vectors = await embedder.embed_documents(demo_texts)
    for i, (text, vector) in enumerate(zip(demo_texts, vectors)):
        norm = sum(x * x for x in vector) ** 0.5
        print(f"text[{i}]            : {preview(text, 80)!r}")
        print(f"vector[{i}] dimension: {len(vector)}")
        print(f"vector[{i}] values   : {show_floats(vector)}")
        print(f"vector[{i}] l2-norm  : {norm:.6f}  (1.0 => cosine-safe for dotproduct index)")
        print()

    # ------------------------------------------------------------------ STEP 4
    banner(f"Step 4: STORE IN PINECONE   ({settings.pinecone_index} / namespace={args.tenant})")
    store = get_vector_store()
    index = await asyncio.to_thread(store.ensure_index)
    print(f"index ready     : {settings.pinecone_index}")
    print(f"metric          : dotproduct")
    print(f"dimension       : {settings.embedding_dimension}")

    try:
        stored = await store.persist_chunks(chunks)
    except Exception as exc:
        print(f"\npersist failed: {type(exc).__name__}: {exc}")
        print("\nCheck that PINECONE_API_KEY is set and Ollama is running (ollama serve).")
        raise SystemExit(1)

    print(f"upserted vectors : {stored:,}  (vector_id = {{doc_id}}-{{chunk_index}})")
    print(f"sparse encoder   : Bm25SparseVectorizer (fit-free: hash term ids, term-frequency values)")
    print(f"sparse layer     : indices + term-frequency values per vector")

    stats = await asyncio.to_thread(index.describe_index_stats)
    ns = stats["namespaces"].get(args.tenant, {})
    print()
    print(f"confirm on dashboard -> describe_index_stats:")
    print(f"  total vectors in index   : {stats['total_vector_count']:,}")
    print(f"  vectors in namespace     : {ns.get('vector_count', 0):,}")

    vector_ids = [f"{doc_id}-0", f"{doc_id}-1"]
    fetched = await asyncio.to_thread(index.fetch, ids=vector_ids, namespace=args.tenant)
    print()
    print("---- what a stored record contains (what the Pinecone console shows) ----")
    for vid, record in (fetched.get("vectors") or {}).items():
        values = record.get("values", [])
        sparse = record.get("sparse_values", {}) or {}
        print(f"id             : {vid}")
        print(f"metadata       : {json.dumps(record.get('metadata', {}), indent=2)}")
        print(f"dense values   : {len(values)} dims -> {show_floats(values, 6)}")
        idx = sparse.get("indices", [])
        print(f"sparse {len(idx)} terms  : "
              + ", ".join(f"#{i}->{v:.1f}" for i, v in zip(idx[:6], sparse.get("values", [])[:6])) + (" ..." if len(idx) > 6 else ""))
        print()

    print("idempotency     : re-running overwrites the same {doc_id}-{chunk_index} ids (count stays the same)")
    print("\nSee it in Pinecone: console.cloud.pinecone.io -> index 'brand-chunks' -> browse vectors in namespace '" + args.tenant + "'")


if __name__ == "__main__":
    asyncio.run(main())