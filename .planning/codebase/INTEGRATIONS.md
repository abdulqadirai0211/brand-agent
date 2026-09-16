# External Integrations

**Analysis Date:** 2026-09-16

## APIs & External Services

**Web Extraction — Firecrawl (primary, implemented):**
- Cloud scraping API `https://api.firecrawl.dev` (default `DEFAULT_API_URL` in `app/ingestion/extractors/firecrawl.py` line 18)
- SDK/Client: `firecrawl-py` 4.43.0 — async client `AsyncV1FirecrawlApp` (`app/ingestion/extractors/firecrawl.py` line 12)
- Auth: API key via `FIRECRAWL_API_KEY` env var (required — `ValueError` raised if empty, line 37); URL override `FIRECRAWL_API_URL`
- Call pattern: `client.scrape_url(url, formats=["markdown"], only_main_content=..., exclude_tags=["img","svg","picture","video","script"], max_age=0, timeout=30000)` (`app/ingestion/extractors/firecrawl.py` lines 77-84)
- Response: `result.markdown` (min length guard `min_content_length=200`, raises `EmptyExtractionError`), `result.title`, `result.url`, `result.error` → `FetchFailedError`
- Errors: typed exceptions `UnsafeURLError` (`invalid_url`), `FetchFailedError` (`fetch_failed`), `EmptyExtractionError` (`empty_extraction`) in `app/ingestion/extractor.py`; error codes map to API `502` responses with `{"detail": {"error": ...}}` (spec §6)
- ⚠️ **Secret leak:** `experiment2.ipynb` cell 2 (line 63) hardcodes a live Firecrawl API key in source. The key value must NOT be reused/committed; rotate it and remove from the notebook.
- Prototype parity: `experiment2.ipynb` used the sync client `from firecrawl import Firecrawl` with identical `scrape()` params — same semantics as the async implementation.

**Web Extraction — Crawl4AI (planned fallback, NOT implemented):**
- SDK: `crawl4ai` 0.9.3 installed in venv; prototype in `experimentation.ipynb` (`AsyncWebCrawler`, `CrawlerRunConfig`, `CacheMode`, `PruningContentFilter`/`BM25ContentFilter` content filters, `DefaultMarkdownGenerator`)
- Selecting `WEB_EXTRACTOR=crawl4ai` raises `NotImplementedError` (`app/ingestion/extractor.py` line 98); scaffold `app/ingestion/extractors/crawl4ai.py` is empty
- Extraction factory: `get_extractor()` maps `firecrawl`/`auto` → `FirecrawlExtractor`, `crawl4ai` → not implemented (`app/ingestion/extractor.py` lines 90-99)

**LLM — OpenAI (planned; config present, not yet wired):**
- Provider-auth: `OPENAI_API_KEY` env var; model `OPENAI_MODEL=gpt-4o-mini` (`.env.example`)
- SDK: `openai` 3.14.0 installed
- Uses in the QA graph (per `brand_qa_agent_file_spec.md` §15-16): answer generation, per-chunk relevance grading (`grade_node`), with async calls required and `llm_unavailable` → `502`
- Health endpoint must report `"llm": "ok"` reachability (spec §6, `app/api/routes/health.py` stub)

## Data Storage

**Vector Database — Qdrant:**
- Provider: local Docker `qdrant/qdrant:latest` (`docker-compose.yml`), or remote instance
- Connection: `QDRANT_URL=http://localhost:6333`, optional `QDRANT_API_KEY`, collection `QDRANT_COLLECTION=brand_chunks` (`.env.example`)
- Client: `qdrant-client` (declared in `requirements.txt`; prototype pattern in `experiment2.ipynb` cell 11: `QdrantClient(location=":memory:")` for tests)
- Schema (spec §11 + notebook): named vectors config —
  - `dense`: `VectorParams(size=1024, distance=COSINE)` (embeddings from `qwen3-embedding:4b` at 1024 dims; experiment used 1536 OpenAI dims — migrated to Ollama)
  - `sparse`: `SparseVectorParams(modifier=Modifier.IDF)` for BM25 lexical retrieval
  - Payload index: `create_payload_index(field_name="metadata.tenant_id", field_schema=PayloadSchemaType.KEYWORD)` — tenant isolation enforced in-vectorstore, never post-filter (spec §11)
  - Point payload: `tenant_id`, `doc_id`, `chunk_index`, `source`, `brand`, `heading_path`, `text`
- Operations required: `ensure_collection`, `upsert_chunks`, `delete_document`, `dense_search`, `sparse_search`, `hybrid_search` (spec §11; `app/vectorstore/qdrant.py` empty stub)

**File Storage:** None. Only in-memory/local (Qdrant memory mode in tests).

**Caching:** None. Extraction caching delegated to Firecrawl `max_age` param (default 0 = fresh).

## Embedding & Reranking Infrastructure

**Embeddings — Ollama (config present; not yet wired):**
- Service: local Ollama server on `OLLAMA_BASE_URL=http://localhost:11434`
- Model: `EMBEDDING_MODEL=qwen3-embedding:4b`, `EMBEDDING_DIMENSION=1024` via `langchain_ollama.OllamaEmbeddings(model="qwen3-embedding:4b", base_url="http://localhost:11434", dimensions=1024)` (spec §10)
- Note: model defaults to 2560 dims; `dimensions=1024` must be set explicitly to match `EMBEDDING_DIMENSION` and the Qdrant `dense` vector size
- Interface: `embed_documents(list[str])`, `embed_query(str)` (spec §10; `app/embeddings/service.py` empty stub)

**Reranker — Hugging Face (config present; not yet wired):**
- Model: `RERANKER_MODEL=BAAI/bge-reranker-v2-m3`, device `RERANKER_DEVICE=cpu` (`.env.example`, spec §13)
- Required packages (spec §24): `transformers`, `sentence-transformers`, `torch` — none declared in `requirements.txt` yet; `tokenizers 0.23.2` + `huggingface_hub 1.31.0` present in venv
- Separate from the LLM relevance grader: reranker ranks retrieval candidates (top `FINAL_TOP_K=5`), grade node is an agent decision (spec §13)

## Authentication & Identity

**Auth Provider:** None — no user auth in the stack. Multi-tenancy via path param `tenant_id` (`POST /tenants/{tenant_id}/ingest`, `POST /tenants/{tenant_id}/ask`) enforced at the Qdrant filter level (spec §6, §11). `ADMIN_API_KEY` appeared in an early scaffold draft in `template.py` but is absent from the current `.env.example`/`app/config.py`.

## Monitoring & Observability

**Error Tracking:** None.

**Logs:** Standard logging configured via `LOG_LEVEL` env (`app/config.py`); no structured logging or tracing setup. QA graph includes an execution `trace` returned in the API response (`AskResponse.trace`, spec §5) but no external sink.

## CI/CD & Deployment

**Hosting:** Docker container — `Dockerfile` (python:3.11-slim, uvicorn `app.main:app` on 8000).

**CI Pipeline:** None (no GitHub Actions/other CI config in repo).

**Orchestration:** `docker-compose.yml` runs `qdrant` (6333) + `app` (8000, `env_file: .env`, `depends_on: qdrant`).

## Environment Configuration

**Required env vars (per `.env.example`):**
- `FIRECRAWL_API_KEY` — Firecrawl API (no default; empty → `ValueError` at extractor construction)
- `OPENAI_API_KEY` — LLM provider
- `OLLAMA_BASE_URL` (default `http://localhost:11434`) — local embedding server
- `QDRANT_URL` (default `http://localhost:6333`), `QDRANT_API_KEY`, `QDRANT_COLLECTION` (default `brand_chunks`)
- Optional tunables: `WEB_EXTRACTOR` (firecrawl|crawl4ai|auto), `CHUNK_SIZE`/`CHUNK_OVERLAP`, `EMBEDDING_MODEL`/`EMBEDDING_DIMENSION`, `RERANKER_MODEL`/`RERANKER_DEVICE`/`RERANK_TOP_K`, `DENSE_TOP_K`/`SPARSE_TOP_K`/`RRF_TOP_K`/`FINAL_TOP_K`, `APP_NAME`/`ENVIRONMENT`/`LOG_LEVEL`

**Secrets location:**
- `.env` at repo root (gitignored). `.env.example` committed with empty key placeholders.
- ⚠️ A Firecrawl API key is hardcoded in `experiment2.ipynb` (line 63) — treat as compromised; rotate.

## Webhooks & Callbacks

**Incoming:** None.

**Outgoing:** None (ingestion is pull-based: app calls Firecrawl/LLM/Qdrant on request).

## External Service Dependency Map

| Service | Status | Config key | Client |
|---|---|---|---|
| Firecrawl API | Implemented | `FIRECRAWL_API_KEY` | `firecrawl.v1.AsyncV1FirecrawlApp` |
| Ollama (embeddings) | Planned | `OLLAMA_BASE_URL` | `langchain_ollama.OllamaEmbeddings` |
| Qdrant | Planned | `QDRANT_URL` | `qdrant_client.QdrantClient` |
| OpenAI | Planned | `OPENAI_API_KEY` | `openai` SDK |
| Hugging Face (reranker) | Planned | `RERANKER_MODEL` | `sentence-transformers` |
| Crawl4AI | Planned/blocked | `WEB_EXTRACTOR=crawl4ai` | `crawl4ai` (NotImplementedError) |

---

*Integration audit: 2026-09-16*