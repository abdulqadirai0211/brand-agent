# Codebase Structure

**Analysis Date:** 2026-09-16

## Directory Layout

```
omnibound-assignment/
├── app/                     # Application package (layered RAG backend)
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point (⬜ empty — Docker CMD target)
│   ├── config.py            # ✅ pydantic-settings Settings + get_settings()
│   ├── dependencies.py      # ⬜ FastAPI dependency wiring (service singletons)
│   ├── api/                 # HTTP layer (⬜ all stubs)
│   │   ├── schemas.py       # ⬜ request/response Pydantic models
│   │   └── routes/
│   │       ├── health.py    # ⬜ GET /health
│   │       ├── ingest.py    # ⬜ POST /tenants/{tenant_id}/ingest
│   │       └── ask.py       # ⬜ POST /tenants/{tenant_id}/ask
│   ├── ingestion/           # Document acquisition (partially ✅)
│   │   ├── models.py        # ✅ ExtractedDocument dataclass
│   │   ├── extractor.py     # ✅ WebExtractor Protocol, SSRF guard, factory
│   │   ├── service.py       # ⬜ batch orchestration pipeline
│   │   └── extractors/
│   │       ├── firecrawl.py # ✅ FirecrawlExtractor (Markdown via API)
│   │       └── crawl4ai.py  # ⬜ second WebExtractor implementation
│   ├── preprocessing/       # ⬜ cleaner.py / markdown_parser.py / validator.py
│   ├── chunking/            # ⬜ chunker.py (structure-aware, 400/50 tokens)
│   ├── embeddings/          # ⬜ service.py (Ollama qwen3-embedding:4b, 1024-d)
│   ├── vectorstore/         # ⬜ qdrant.py (tenant-scoped persistence/search)
│   ├── retrieval/           # ⬜ hybrid.py (dense+BM25→RRF), reranker.py (BGE)
│   └── graph/               # ⬜ state.py / nodes.py / edges.py / graph.py
├── tests/                   # pytest suite
│   ├── conftest.py          # ✅ settings fixture
│   ├── test_firecrawl_extractor.py  # ✅ 37 tests (extractor + SSRF)
│   ├── test_tenant_isolation.py     # ⬜ retrieval-layer isolation test
│   └── test_agent_decision.py       # ⬜ LangGraph routing tests (mocked LLM)
├── scripts/
│   └── seed_corpus.py       # ⬜ corpus loader reusing ingestion service
├── .planning/               # GSD workflow artifacts (codebase maps, plans)
├── venv/                    # Local virtualenv (gitignored) — Python 3.14.4
├── template.py              # Scaffold generator (created the stub tree)
├── brand_qa_agent_file_spec.md  # Authoritative 2051-line design contract
├── experimentation.ipynb    # Exploratory notebook (repo root)
├── experiment2.ipynb        # Exploratory notebook (repo root)
├── requirements.txt         # Dependencies (unpinned)
├── .env.example             # Documented env vars (38 lines)
├── .env                     # Local secrets (gitignored — do not read/commit)
├── Dockerfile               # python:3.11-slim + uvicorn app.main:app
├── docker-compose.yml       # qdrant + app services
└── README.md                # Title only ("Brand QA Agent")
```

Status legend: ✅ implemented · ⬜ empty stub, target defined in `brand_qa_agent_file_spec.md`

## Directory Purposes

**`app/`** — Python application package. Every subpackage has `__init__.py` (package markers, no logic). Layered per responsibility boundary chain: `api → ingestion → preprocessing → chunking → embeddings → vectorstore → retrieval → graph`.

**`app/api/`** — HTTP adapters only. `routes/` contains one module per endpoint named after the endpoint (`health.py`, `ingest.py`, `ask.py`). No business logic allowed (spec §6). Currently all stubs.

**`app/ingestion/`** — Document acquisition. `models.py` = internal data types; `extractor.py` = abstraction + safety + factory; `extractors/` = one module per provider; `service.py` = batch orchestrator. This is the only layer with implemented code besides config.

**`app/preprocessing/`, `app/chunking/`, `app/embeddings/`** — Content normalization, segmentation, and vector generation respectively. All stubs.

**`app/vectorstore/`** — Qdrant-specific persistence/search. All persistence knowledge lives here; the rest of the app must not import `qdrant-client` directly.

**`app/retrieval/`** — Candidate retrieval/ranking: `hybrid.py` (dense + BM25 + RRF) and `reranker.py` (BGE cross-encoder → top-5). Reranker is distinct from the graph's LLM grade node (spec §13).

**`app/graph/`** — LangGraph agent: `state.py` (`QAState` TypedDict), `nodes.py` (node functions), `edges.py` (conditional routing), `graph.py` (compile once, reuse). All stubs.

**`tests/`** — pytest suite. One test module per concern (`test_firecrawl_extractor.py`, `test_tenant_isolation.py`, `test_agent_decision.py`), shared fixtures in `conftest.py`. Unit tests must not require Firecrawl/Qdrant/LLM/internet (spec §18).

**`scripts/`** — Operational one-offs; currently only `seed_corpus.py` (stub). Must reuse production ingestion, not re-implement (spec §21).

**`.planning/`** — GSD workflow artifacts (phase plans, codebase maps). Not application code; ignore for runtime.

## Key File Locations

**Entry Points:**
- `app/main.py`: FastAPI composition (⬜ stub; Docker CMD `uvicorn app.main:app` targets it — `Dockerfile:11`)
- `scripts/seed_corpus.py`: corpus seeding CLI (⬜ stub)
- `tests/`: pytest discovery root — no `pytest.ini`/`pyproject.toml`, run via `pytest` or `venv/bin/python -m pytest tests/`

**Configuration:**
- `app/config.py`: `Settings` (env-driven, `.env` via `SettingsConfigDict`, `extra="ignore"`), `get_settings()` cached
- `.env.example`: canonical env var documentation (extraction, chunking, embeddings, Qdrant, reranking, retrieval, LLM groups)
- `requirements.txt`: runtime deps (unpinned; spec §24 lists additional required deps not yet present — `langgraph`, `transformers`, `sentence-transformers`, `torch`, `crawl4ai`)
- `Dockerfile` + `docker-compose.yml`: container build (`python:3.11-slim`) and local stack (`qdrant` on 6333 + `app` on 8000)

**Core Logic (implemented):**
- `app/ingestion/extractor.py`: `validate_fetch_url` (SSRF), `WebExtractor` Protocol, exception hierarchy, `get_extractor` factory
- `app/ingestion/extractors/firecrawl.py`: `FirecrawlExtractor` + `from_settings` classmethod
- `app/ingestion/models.py`: `ExtractedDocument`

**Core Logic (stubs awaiting implementation, per spec sequence §32):**
- `app/ingestion/service.py` → `app/preprocessing/` → `app/chunking/chunker.py` → `app/embeddings/service.py` → `app/vectorstore/qdrant.py` → `app/retrieval/` → `app/graph/` → `app/api/` → `app/dependencies.py` → `app/main.py`

**Testing:**
- `tests/conftest.py`: `settings` fixture (overrides all Firecrawl fields, test key) — pattern for future mocks (`mock_embeddings`, `mock_qdrant`, `mock_llm`, `mock_reranker`, `test_graph` per spec §18)
- `tests/test_firecrawl_extractor.py`: parametrized SSRF matrix + extractor behavior with `FakeFirecrawlClient`

## Naming Conventions

**Files:**
- snake_case matching the module's single responsibility: `extractor.py`, `extractors/firecrawl.py`, `chunker.py`, `hybrid.py`, `reranker.py`
- One route module per endpoint: `health.py`, `ingest.py`, `ask.py` under `app/api/routes/`
- Tests: `test_<subject>.py` (_firecrawl_extractor, _tenant_isolation, _agent_decision)

**Directories:**
- Lowercase, plural for collections of providers (`app/ingestion/extractors/`, `app/api/routes/`), singular for single-module layers (`app/chunking/`, `app/graph/`, `app/vectorstore/`)
- Layer order in `app/` mirrors the pipeline: `ingestion → preprocessing → chunking → embeddings → vectorstore → retrieval → graph`

**Functions:**
- snake_case verb-first: `get_extractor`, `get_settings`, `validate_fetch_url`, `_is_blocked_ip`, `_validate_fetch_url_sync` (`app/ingestion/extractor.py`)
- LangGraph nodes will carry the `_node` suffix: `retrieve_node`, `grade_node`, `rewrite_node`, `answer_node` (spec §15)
- Private helpers prefixed `_` (`_get_client`, `_is_blocked_ip`, `_validate_fetch_url_sync`)
- Async methods `async def` + `asyncio.to_thread` for blocking DNS (`app/ingestion/extractor.py:87`)

**Types:**
- `class`-level error codes as uppercase constants: `code = "invalid_url"`, `code = "fetch_failed"` (`app/ingestion/extractor.py:33-41`)
- Dataclasses for internal transfer objects: `ExtractedDocument` (`app/ingestion/models.py`)
- Protocol for provider abstraction: `WebExtractor` (`app/ingestion/extractor.py:13`)
- Configuration via `BaseSettings` subclass: `Settings` (`app/config.py:11`)
- API models via Pydantic `BaseModel` (spec §5, stubs)

## Where to Add New Code

**New Feature (e.g., a new endpoint):**
- Route handler: `app/api/routes/<name>.py` (HTTP-only adapter)
- Schemas: extend `app/api/schemas.py`
- Register router in `app/main.py`
- Wire service instance in `app/dependencies.py`

**New Extractor (provider):**
- Implementation: `app/ingestion/extractors/<provider>.py` — must satisfy `WebExtractor` Protocol (`app/ingestion/extractor.py:13`), raise the typed extraction errors, and expose `from_settings(settings)` (pattern: `app/ingestion/extractors/firecrawl.py:47`)
- Register backend string in `get_extractor` dispatch (`app/ingestion/extractor.py:90`)
- Tests: `tests/test_<provider>_extractor.py` with a fake client (pattern: `FakeFirecrawlClient` in `tests/test_firecrawl_extractor.py:24`)

**New LangGraph Node (agent behavior):**
- Node function: `app/graph/nodes.py` (`<name>_node`, appends one trace entry, receives dependencies via closure/factory — never construct clients inside)
- Routing: `app/graph/edges.py` conditional function
- State fields: extend `QAState` in `app/graph/state.py` (keep `retry_count` bounded)
- Graph assembly: `app/graph/graph.py` (`compile` once)

**Utilities / helpers:**
- Shared helpers: `app/` layer that owns the concern (e.g., validation → `app/preprocessing/validator.py`, Qdrant client → `app/vectorstore/qdrant.py`)
- Do NOT add a top-level `utils/` package; the spec mandates strict per-layer ownership (§30)

**Tests:**
- New test module: `tests/test_<concern>.py`, fixtures in `tests/conftest.py`
- Mock external services; never hit real Firecrawl/Qdrant/LLM (spec §18)

## Special Directories

**`venv/`:**
- Purpose: Local virtualenv (Python 3.14.4 — newer than the Docker image's 3.11)
- Generated: Yes · Committed: No (`.gitignore`)

**`__pycache__/` (inside `app/`, `tests/`):**
- Purpose: Python bytecode cache
- Generated: Yes · Committed: No (`.gitignore`)

**`.planning/`:**
- Purpose: GSD workflow artifacts (codebase maps, phase plans, milestones)
- Generated: Yes (by GSD) · Committed: Yes (GSD convention)

**`.pytest_cache/`:**
- Purpose: pytest cache
- Generated: Yes · Committed: No (`.gitignore`)

**Notebooks (`experimentation.ipynb`, `experiment2.ipynb`):**
- Purpose: Exploration artifacts at repo root; not part of the `app/` package
- Generated: Manual · Committed: Yes (should move to a gitignored `notebooks/`)

---

*Structure analysis: 2026-09-16*