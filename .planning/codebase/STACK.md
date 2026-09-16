# Technology Stack

**Analysis Date:** 2026-09-16

## Languages

**Primary:**
- Python — entire backend. Docker image targets `python:3.11-slim` (`Dockerfile`); local venv runs Python 3.14.4 (see `experiment2.ipynb` kernel metadata: `venv (3.14.4)`). Type hints (`str | None`, `list[str]`) used throughout, requiring Python 3.10+.

**Secondary:**
- None (no JS/TS frontend; notebooks: Markdown/JSON)

## Runtime

**Environment:**
- ASGI via Uvicorn — `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]` in `Dockerfile`. Note: `app/main.py` is currently an empty stub (0 lines), so the app object does not exist yet.
- Local venv at `venv/` (gitignored, `.gitignore`).

**Package Manager:**
- pip with `requirements.txt` (no version pins — unpinned range installs)
- Lockfile: missing. Installed versions (from `venv/lib/python3.14/site-packages` dist-info): `firecrawl-py 4.43.0`, `langchain_ollama 1.1.0`, `langchain_core 1.6.3`, `openai 3.14.0`, `pydantic 2.13.5`, `pydantic_settings 2.15.0`, `pytest 9.1.1`, `pytest_asyncio 1.4.0`, `httpx 0.28.1`, `python_dotenv 1.2.3`, `certifi 2026.7.22`, `crawl4ai 0.9.3`, `tokenizers 0.23.2`.

## Frameworks

**Core:**
- FastAPI — REST framework (declared in `requirements.txt`; NOT yet installed in venv; planned app composition in `app/main.py` + routers in `app/api/routes/{health,ingest,ask}.py`, all currently empty stubs)
- Pydantic v2 / Pydantic Settings — config + schemas (`app/config.py` implemented, `app/api/schemas.py` empty stub)
- LangChain — `langchain-text-splitters` for chunking (`app/chunking/chunker.py` stub; real usage pattern in `experiment2.ipynb` cell using `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`), `langchain-ollama` for embeddings (`app/embeddings/service.py` stub)
- LangGraph — planned QA agent graph (`app/graph/{state,nodes,edges,graph}.py`, all empty stubs; package not yet in requirements.txt — spec `brand_qa_agent_file_spec.md` §24 lists it as required)

**Testing:**
- pytest 9.1.1 + pytest-asyncio 1.4.0 — `tests/conftest.py` provides a `settings` fixture; `tests/test_firecrawl_extractor.py` (277 lines) is the only implemented test module. No `pyproject.toml`/`pytest.ini`/`setup.cfg` — default pytest config + `conftest.py`.

**Build/Dev:**
- Docker + Docker Compose — `Dockerfile` (Python 3.11-slim, uvicorn) and `docker-compose.yml` (services: `qdrant` image `qdrant/qdrant:latest` on port 6333, `app` built from Dockerfile on port 8000 with `env_file: .env`)
- Jupyter notebooks for prototyping: `experimentation.ipynb` (crawl4ai trials), `experiment2.ipynb` (firecrawl + chunking + qdrant trials)

## Key Dependencies

**Critical:**
- `openai` (3.14.0 installed) — LLM generation + relevance grading in the QA agent (`OPENAI_MODEL=gpt-4o-mini` in `.env.example`). Not yet imported in `app/`
- `qdrant-client` — vector DB client (declared; not yet installed in venv; usage pattern in `experiment2.ipynb`: `QdrantClient(location=":memory:")`, `create_collection` with named vectors `dense` + sparse `models.SparseVectorParams(modifier=models.Modifier.IDF)`, `create_payload_index` on `metadata.tenant_id`)
- `firecrawl-py` (4.43.0) — web extraction; sync prototype used `from firecrawl import Firecrawl`; the implementation in `app/ingestion/extractors/firecrawl.py` uses `from firecrawl.v1 import AsyncV1FirecrawlApp`
- `langchain-ollama` (1.1.0) — `OllamaEmbeddings` for `qwen3-embedding:4b` (see spec §10)
- `langchain-text-splitters` — `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` (verified in `experiment2.ipynb`)

**Infrastructure:**
- `pydantic-settings` — env-driven `Settings` in `app/config.py`
- `python-dotenv` — `.env` loading (also a transitive dep of firecrawl-py)
- `httpx` — async HTTP (transitive dep of firecrawl-py; declared for custom HTTP needs)
- `certifi` — `os.environ.setdefault("SSL_CERT_FILE", certifi.where())` set at import time in `app/config.py` (line 8)
- `tiktoken` — used in `experiment2.ipynb` cell 9 for token counting (`cl100k_base`); spec §9 requires the Qwen3 tokenizer via `transformers`/`tokenizers` instead (Ollama has no tokenize endpoint)
- `crawl4ai` (0.9.3 installed) — alternative extractor; factory raises `NotImplementedError` (`app/ingestion/extractor.py` line 98), scaffold file `app/ingestion/extractors/crawl4ai.py` is empty
- `pytest-asyncio` — async test support

## Configuration

**Environment:**
- pydantic-settings `Settings` class in `app/config.py` reads `.env` (`SettingsConfigDict(env_file=".env", extra="ignore")`), cached via `@lru_cache(maxsize=1)` `get_settings()`
- `.env` file present at repo root (gitignored); `.env.example` documents all 38 config keys: APP_NAME, ENVIRONMENT, LOG_LEVEL, WEB_EXTRACTOR, FIRECRAWL_API_KEY/URL, CHUNK_SIZE=400, CHUNK_OVERLAP=50, EMBEDDING_MODEL=`qwen3-embedding:4b`, EMBEDDING_DIMENSION=1024, OLLAMA_BASE_URL=`http://localhost:11434`, QDRANT_URL=`http://localhost:6333`, QDRANT_API_KEY, QDRANT_COLLECTION=`brand_chunks`, RERANKER_MODEL=`BAAI/bge-reranker-v2-m3`, RERANKER_DEVICE=cpu, RERANK_TOP_K=10, DENSE_TOP_K=10, SPARSE_TOP_K=10, RRF_TOP_K=10, FINAL_TOP_K=5, OPENAI_API_KEY, OPENAI_MODEL=`gpt-4o-mini`
- Custom validator: `firecrawl_exclude_tags` accepts comma-separated string (`app/config.py` lines 38-43)
- Defaults for non-secret values only; `firecrawl_api_key` has empty default (`app/config.py` line 26)

**Build:**
- `Dockerfile`: installs unpinned `requirements.txt`, copies only `app/`, EXPOSE 8000
- `docker-compose.yml`: qdrant on 6333 (no volume, no auth), app on 8000, `env_file: .env`, `depends_on: qdrant`

## Platform Requirements

**Development:**
- Python 3.11+ (Docker target) / 3.14 (current local venv)
- Local services: Ollama on `localhost:11434` (serving `qwen3-embedding:4b`), Qdrant on `localhost:6333` (or via `docker compose up`), Firecrawl cloud API key
- `docker compose up` per `docker-compose.yml` to run Qdrant + app

**Production:**
- Deployment target: Docker container (uvicorn on 0.0.0.0:8000). No cloud/CI config present beyond Docker.

## Implementation Status (important context)

Most of the scaffold is EMPTY stubs generated by `template.py`. Fully implemented today:
- `app/config.py` (47 lines) — settings
- `app/ingestion/extractor.py` (98 lines) — `WebExtractor` protocol, SSRF-safe `validate_fetch_url`, `get_extractor` factory
- `app/ingestion/extractors/firecrawl.py` (109 lines) — `FirecrawlExtractor`
- `app/ingestion/models.py` (8 lines) — `ExtractedDocument` dataclass
- `tests/conftest.py`, `tests/test_firecrawl_extractor.py`
All other modules (API routes, chunking, embeddings, retrieval, vectorstore, graph, preprocessing) are empty files. Specification for the full system: `brand_qa_agent_file_spec.md`.

---

*Stack analysis: 2026-09-16*