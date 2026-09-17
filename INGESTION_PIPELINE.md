# Ingestion Pipeline Architecture

Here's the ingestion pipeline, drawn from the actual code. Use the top diagram as the visual, then the "data envelope" for the per-chunk walkthrough.

## 1. Pipeline architecture

```
                         POST /tenants/{tenant_id}/ingest
                         app/api/routes/ingest.py:14
                         body: { documents: [ {url | text, brand, source?} ] }
                                        │
                     Pydantic IngestRequest  (url XOR text; brand required)
                                        │
                     Depends(get_ingestion)  → singleton IngestionService
                     dependencies.py:21      → service.persist = store.persist_chunks
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ ORCHESTRATION   IngestionService.run(tenant_id, documents)  service.py:43      │
│                                                                                │
│   for each document:                                                           │
│     1  _validate ── url XOR text? brand present? ──✗──► error "invalid_input"  │
│     2  _doc_id = sha256(url | text)  ──► batch dedupe (first occurrence wins)  │
│     3  _process_document ▼                                                     │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌───────────────────┐   ┌───────────────────────────┐   ┌───────────────────────────────┐
│ 3a  EXTRACT       │   │ 3b  CHUNK                 │   │ 3c  PERSIST                   │
│ service.py:105    │   │ document_chunker.py:29    │   │ pinecone.py:116               │
│                   │   │                           │   │                               │
│ url → WebBase     │   │ RecursiveCharacter        │   │ dense : embed_documents        │
│   Extractor       │   │ TextSplitter              │   │   → Ollama qwen3-embedding:4b  │
│   web_loader.py:98│   │  size=400 / overlap=50    │   │   → L2-normalize (dim 1024)    │
│   (to_thread)     │   │                           │   │                               │
│   → fetch, strip  │   │ per chunk → ChunkMetadata:│   │ sparse: Bm25SparseVectorizer   │
│   → len ≥ min?    │   │   tenant_id, doc_id,      │   │   TF, hashed 24-bit term ids   │
│                   │   │   chunk_index, source,    │   │   (fit-free, stateless)        │
│ text → len ≥ min? │   │   brand                   │   │                               │
│                   │   │                           │   │ group records by               │
│ ─✗→ empty_        │   │ → Document(page_content,  │   │   metadata["tenant_id"]        │
│      extraction   │   │            metadata)      │   │ id = f"{doc_id}-{chunk_index}" │
│                   │   │                           │   │                               │
│ ▼                 │   │                           │   │ index.upsert(                 │
│ ExtractedDocument │   │                           │   │   namespace = tenant_id, ◄──── ISOLATION
│ {text,url,title,  │   │                           │   │   batch_size = 100)           │
│  brand,extractor} │   │                           │   │                               │
└───────────────────┘   └───────────────────────────┘   └───────────────────────────────┘
                                        │
                                        ▼
                 IngestResponse { documents_stored, chunks_stored, errors[] }
                 ingest.py:30
```

## 2. Data envelope — one document becoming N searchable records

```
 input          { url: "https://asana.com/pricing", brand: "Asana" }   (tenant_id from path = "acme")
   │
   ▼  extract
 ExtractedDocument { text: "<~4k chars>", url, title, brand: "Asana", extractor: "webbase_loader" }
   │
   ▼  chunk  (400/50, character-based)
 Document(page_content="<chunk text>",
          metadata={ tenant_id:"acme", doc_id:"<sha256>", chunk_index:0,
                     source:"https://asana.com/pricing", brand:"Asana" })
   │
   ▼  embed + sparse-encode
 vector { id: "<sha256>-0",
          values: [1024 floats L2-normalized],
          sparse_values: { indices:[...], values:[term-freq...] },
          metadata: { tenant_id, doc_id, chunk_index, source, brand, text } }
   │
   ▼  upsert
 Pinecone  └─ namespace = "acme"   ← every record lands only here
```

## 3. Talking points for the video

- **Two input modes, one path** — `url` is fetched with `WebBaseLoader`; raw `text` is used directly. Both converge on `ExtractedDocument` (`service.py:105`), so all downstream stages are input-agnostic.
- **Idempotent by construction** — `doc_id = sha256(url)` (or sha256 of text), and the vector id is `{doc_id}-{chunk_index}` (`document_id.py:8`, `pinecone.py:135`). Re-ingesting the same doc overwrites the same ids rather than duplicating.
- **Per-document fault isolation** — each doc is processed in its own `try/except` (`service.py:60-67`); one bad URL doesn't abort the batch. Errors are collected, never raised.
- **Tenant routing happens at the write boundary** — `tenant_id` comes only from the URL path (`ingest.py:21`), is stamped into every chunk's metadata (`document_chunker.py:41`), and is the namespace used by `upsert` (`pinecone.py:143`). This is the same key the query path filters on.
- **Dual representation** — each chunk is stored as both a dense vector (semantic, normalized) and a sparse vector (lexical TF, hashed term ids), which is what enables hybrid search at query time.
- **Sync libraries kept off the event loop** — the web fetch, embedding client, and sparse encoding all run via `asyncio.to_thread` (`service.py:107`, `pinecone.py:119,123`).

## 4. Error taxonomy (per-document `errors[]`)

| Stage | Condition | Code |
|---|---|---|
| validate | url+text both, neither, or missing brand | `invalid_input` |
| extract | non-http(s) scheme / no hostname | `invalid_url` |
| extract | fetch failed / no content | `fetch_failed` |
| extract | text below `min_content_length` (200) | `empty_extraction` |
| chunk/persist | no chunks, or any unexpected failure | `ingest_failed` |
