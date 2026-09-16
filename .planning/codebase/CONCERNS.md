# Codebase Concerns

**Analysis Date:** 2026-09-16

## Tech Debt

### Application is an empty skeleton — cannot boot

- Issue: `app/main.py` is 0 bytes, yet `Dockerfile` runs `CMD ["uvicorn", "app.main:app", ...]` and `docker-compose.yml` starts the `app` service. The container crashes at startup: `app.main` has no `app` object. The entire FastAPI surface is unimplemented — `app/api/routes/health.py`, `app/api/routes/ingest.py`, `app/api/routes/ask.py`, `app/api/schemas.py`, and `app/dependencies.py` are all 0 bytes.
- Files: `app/main.py`, `app/dependencies.py`, `app/api/schemas.py`, `app/api/routes/health.py`, `app/api/routes/ingest.py`, `app/api/routes/ask.py`, `Dockerfile`, `docker-compose.yml`
- Impact: Nothing runs. The spec (`brand_qa_agent_file_spec.md`, 2,051 lines) defines the full contract, but only the extraction slice exists.
- Fix approach: Implement per spec — `app/main.py` (FastAPI app + lifespan), `app/dependencies.py` (settings/vectorstore deps), the three routes, schemas. Use `test_agent_decision.py` / `test_tenant_isolation.py` (currently empty) to drive behavior.

### ~24 module stubs for the core pipeline

- Issue: The whole RAG pipeline is scaffolded as empty files with `__init__.py` only. Zero logic exists for preprocessing, chunking, embeddings, retrieval, vector store, or the LangGraph agent — despite `brand_qa_agent_file_spec.md` specifying exact responsibilities, interfaces, and expected behaviors (e.g., chunk_size=400 tokens at line 867, RRF strategy, tenant isolation rules).
- Files: `app/preprocessing/cleaner.py`, `app/preprocessing/markdown_parser.py`, `app/preprocessing/validator.py`, `app/chunking/chunker.py`, `app/embeddings/service.py`, `app/retrieval/hybrid.py`, `app/retrieval/reranker.py`, `app/vectorstore/qdrant.py`, `app/graph/state.py`, `app/graph/nodes.py`, `app/graph/edges.py`, `app/graph/graph.py`, `app/ingestion/service.py`, `app/ingestion/extractors/crawl4ai.py`, `scripts/seed_corpus.py`, plus `app/api/*`
- Impact: The product cannot perform ingestion, search, or Q&A. Every downstream feature is blocked.
- Fix approach: Implement modules in dependency order: preprocessing → chunking (`app/chunking/chunker.py`) → embeddings (`app/embeddings/service.py` via `langchain-ollama`) → vectorstore (`app/vectorstore/qdrant.py`) → retrieval (`app/retrieval/hybrid.py`, `app/retrieval/reranker.py`) → graph (`app/graph/*`) → API. Align names/signatures exactly with the spec to avoid rework.

### Empty test files with deleted-content remnants

- Issue: `tests/test_tenant_isolation.py` and `tests/test_agent_decision.py` are 0 bytes, but `tests/__pycache__/test_tenant_isolation.cpython-314-pytest-9.1.1.pyc` and `tests/__pycache__/test_agent_decision.cpython-314-pytest-9.1.1.pyc` exist — proof these files previously contained collected tests and were emptied/truncated. The spec requires both test suites.
- Files: `tests/test_tenant_isolation.py`, `tests/test_agent_decision.py`, `tests/__pycache__/` (stale `.pyc` artifacts)
- Impact: Tenant-isolation and agent-decision behavior is both unimplemented AND untested — the two most safety-critical behaviors of the product.
- Fix approach: Restore/rewrite both suites alongside the tenant-scoped vector store and graph implementation. Do not copy from stale `.pyc`.

### Config drift between three sources of truth

- Issue: `.env.example`, `app/config.py`, and `template.py` disagree. `.env.example` declares `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RERANKER_MODEL`, `RERANKER_DEVICE`, `RERANK_TOP_K`, `DENSE_TOP_K`, `SPARSE_TOP_K`, `RRF_TOP_K`, `FINAL_TOP_K`, `OPENAI_API_KEY`, `OPENAI_MODEL` — none of which exist as fields on `Settings` in `app/config.py`. With `extra="ignore"` in `SettingsConfigDict` (line 15), those env vars are silently dropped. Meanwhile `template.py`'s embedded `.env.example` (lines 55–61) lists yet a third set (`ADMIN_API_KEY`, `DEFAULT_EMBEDDING_MODEL`, `DEFAULT_LLM_MODEL`).
- Files: `app/config.py` (lines 11–43), `.env.example`, `template.py` (lines 55–61)
- Impact: Silent misconfiguration — a developer sets `QDRANT_URL` in `.env` and the app never reads it; no error, just wrong behavior once the pipeline exists.
- Fix approach: Extend `Settings` with all the groups the spec defines (Configuration groups at spec lines 176–245): chunking, qdrant, reranking, retrieval, LLM. Delete `template.py`'s stale `.env.example` content or regenerate it from `config.py`.

### Unpinned dependencies, no lockfile

- Issue: `requirements.txt` has zero version pins (`fastapi`, `uvicorn[standard]`, `openai`, `qdrant-client`, `firecrawl-py`, `langchain-ollama`, ...). `docker-compose.yml` uses `qdrant/qdrant:latest`. No lockfile (no `pip-tools`, `uv`, or `poetry`).
- Files: `requirements.txt`, `docker-compose.yml`
- Impact: Builds are non-reproducible; a breaking `firecrawl-py` or `langchain-ollama` release silently changes behavior between environments.
- Fix approach: Pin exact versions (or use `pip-compile`/`uv lock`) and pin the Qdrant image tag in `docker-compose.yml`.

## Known Bugs

### Docker networking is wired wrong for localhost services

- Symptoms: Once implemented, the app container will try to reach Qdrant and Ollama at `localhost`, which inside Docker resolves to the container itself, not the host or the `qdrant` service.
- Files: `docker-compose.yml`, `.env.example` (`QDRANT_URL=http://localhost:6333`, `OLLAMA_BASE_URL=http://localhost:11434`)
- Trigger: `docker compose up` with the pipeline implemented.
- Workaround: Run services on host and the app in host-network mode; or edit `.env` to `http://qdrant:6333` and `http://host.docker.internal:11434` (macOS/Windows) — but `.env` is gitignored so every fresh clone silently hits this.
- Additionally: `env_file: .env` in `docker-compose.yml` hard-fails or warns on a fresh clone because `.env` is gitignored and absent; there is no bootstrap step to copy `.env.example`.

### Fresh clone cannot start anything

- Symptoms: `docker compose up` errors on missing `.env`; `pip install -r requirements.txt` succeeds but `uvicorn app.main:app` fails with "no module app.main" content issue (empty `app.main`).
- Files: `docker-compose.yml`, `Dockerfile`, `app/main.py`
- Trigger: Clone + `docker compose up` or `uvicorn app.main:app`.
- Workaround: None (short of implementing `app/main.py`).

## Security Considerations

### SSRF guard is validation-only, with TOCTOU and redirect gaps

- Risk: `validate_fetch_url()` in `app/ingestion/extractor.py` (lines 57–87) resolves the hostname locally and blocks loopback/private/link-local/reserved addresses. But Firecrawl's SaaS fetches the URL server-side: (a) the address Firecrawl's servers resolve can differ from the local resolver's answer (DNS rebinding window — TOCTOU between check and fetch); (b) Firecrawl follows redirects after the initial URL passed validation, and a redirect chain can end at `http://169.254.169.254/...` without re-validation.
- Files: `app/ingestion/extractor.py` (`_validate_fetch_url_sync`, `validate_fetch_url`), `app/ingestion/extractors/firecrawl.py` (`extract`, line 74)
- Current mitigation: Blocks `169.254.169.254` explicitly plus `ip.is_loopback / is_link_local / is_private / is_reserved / is_multicast / is_unspecified` (lines 44–54). Decent first line of defense.
- Recommendations: Enforce Firecrawl `max_redirects`/redirect policy at the API layer; validate the *final* resolved URL; or run ingestion against a self-hosted Firecrawl in a network-isolated zone. Document that this guard does not fully prevent SSRF once Firecrawl is involved.

### No authentication anywhere

- Risk: The spec mandates admin-only ingestion via `ADMIN_API_KEY` (spec section on `ingest.py`); zero auth exists because the API is unimplemented. When routes get built, ingestion and ask endpoints would otherwise be open.
- Files: `app/api/routes/ingest.py`, `app/api/routes/ask.py`, `app/config.py` (no `admin_api_key` field)
- Current mitigation: None.
- Recommendations: Add `admin_api_key: str` to `Settings` (fail-fast validation), enforce on `/ingest` (and future admin endpoints), and keep `/ask` behind tenant-scoped auth per spec.

### Config loads secrets with silent empty defaults

- Risk: `firecrawl_api_key: str = Field(default="", ...)` in `app/config.py` (line 26) means a misconfigured deployment loads fine and only fails later inside `FirecrawlExtractor.__init__` with `ValueError("FIRECRAWL_API_KEY is required...")` at request time, not at startup. Same pattern will repeat for `OPENAI_API_KEY`/`QDRANT_API_KEY` when added.
- Files: `app/config.py` (line 26), `app/ingestion/extractors/firecrawl.py` (lines 36–37)
- Current mitigation: `FirecrawlExtractor.from_settings` raises early when an extractor is built; `.env` is gitignored (good).
- Recommendations: In `Settings`, treat empty secret values as startup errors (a `model_validator` that fails when `environment == "production"`), and never log secrets.

### Repo hygiene: secrets and bloat risk

- Risk: First commit `742bc72` contains only `.gitignore` + `experimentation.ipynb` (210 KB, 1,614 lines) — no leaked keys found in a scan of both notebooks, but these untracked artifacts remain: `experimentation.ipynb`, `experiment2.ipynb`, the PDF `omnibound-ai-engineer-take-home-v2_7642_2672 (1).pdf` (131 KB), and `brand_qa_agent_file_spec.md`. None are in `.gitignore`, so `git add .` commits them all. Notebooks commonly accumulate API keys in cell outputs — a future risk.
- Files: `.gitignore`, `experimentation.ipynb`, `experiment2.ipynb`, `omnibound-ai-engineer-take-home-v2_7642_2672 (1).pdf`
- Current mitigation: `.env`, `venv/`, `__pycache__/`, `.pytest_cache/` are gitignored.
- Recommendations: Add `*.ipynb` and the PDF to `.gitignore` (keep the markdown spec as the source of truth); run a secret scan before any commit.

## Performance Bottlenecks

### Reranker on CPU is inherently slow (latent)

- Problem: The spec mandates `BAAI/bge-reranker-v2-m3` with `RERANKER_DEVICE=cpu` (`app/config.py` has no field yet; `.env.example` declares it). Cross-encoder reranking on CPU is typically 10–100× slower than the dense/BM25 retrieval steps.
- Files: `.env.example` (`RERANKER_MODEL`, `RERANKER_DEVICE`, `RERANK_TOP_K`), `app/retrieval/reranker.py` (empty stub)
- Cause: Cross-encoder must run every candidate pair through the transformer; no batching or caching is specified.
- Improvement path: Cap `RERANK_TOP_K` (spec says 10) tightly, batch scoring, cache by query/chunk hash, or expose a GPU/ONNX option; make `device` configurable at runtime.

### No vector-store or embedding-layer design review yet

- Problem: `app/vectorstore/qdrant.py` and `app/embeddings/service.py` are empty; there is no evidence of payload schema (tenant_id metadata), index settings (HNSW params, dense dim 1024 + BM25 sparse payload), or an idempotence strategy for `deterministic/idempotent document ingestion` (spec requirement).
- Files: `app/vectorstore/qdrant.py`, `app/embeddings/service.py`, `scripts/seed_corpus.py` (empty)
- Cause: Not implemented.
- Improvement path: Design payload + point-id hashing (e.g., sha256 of tenant+doc hash) before implementation to avoid a reindex later.

## Fragile Areas

### Global `SSL_CERT_FILE` mutation on import

- Files: `app/config.py` line 8: `os.environ.setdefault("SSL_CERT_FILE", certifi.where())`
- Why fragile: Importing `app.config` mutates process-global environment for every library in the process (OpenAI, httpx, Ollama, Firecrawl all inherit it). It exists as a workaround for Ollama's local HTTPS, but it can override an operator's deliberately set `SSL_CERT_FILE` — no, it uses `setdefault`, so it only fills an *unset* var — still, it silently changes TLS trust for the whole app and can mask real hostname/cert misconfigurations on local Ollama.
- Safe modification: Scope cert handling to the Ollama/embeddings client only (its own `verify`/SSL arg) instead of process-wide.
- Test coverage: None — no test asserts cert behavior (and none should mutate env; note `render_tests` currently don't touch this).

### Firecrawl errors collapse into one type

- Files: `app/ingestion/extractors/firecrawl.py` lines 85–86: `except Exception as exc: raise FetchFailedError(url=url, reason=str(exc))`
- Why fragile: Catches *all* exceptions — including programming errors (e.g., `AttributeError` inside `_get_client`, type mismatches in kwargs) — and re-labels them `fetch_failed`. Debugging becomes guessing. The `reason` string may embed SDK/response internals.
- Safe modification: Catch `Exception` only around the `scrape_url` call (as now) but log the original traceback with `logger.exception()`; let non-`scrape_url` errors propagate.
- Test coverage: `tests/test_firecrawl_extractor.py` covers SDK exception wrapping (`test_sdk_exception_wrapped_as_fetch_failed`) but not code-error propagation.

### Settings fixture duplicates config defaults

- Files: `tests/conftest.py` (lines 7–19) hardcodes every setting that `app/config.py` defaults. If a default changes in `Settings`, the fixture silently diverges and tests pass against stale values.
- Why fragile: Duplicated constants are a classic drift source; `test_from_settings` in `tests/test_firecrawl_extractor.py` only checks extractor fields, not config defaults.
- Safe modification: Build the fixture from `Settings()` defaults overridden by a small explicit diff, or assert fixture == defaults.

### Local Python 3.14 vs Docker Python 3.11

- Files: `venv/` (CPython 3.14.4 — evidenced by `cpython-314` pycache artifacts), `Dockerfile` (`FROM python:3.11-slim`)
- Why fragile: Code is developed and tested on 3.14 but deployed on 3.11. Today's implemented code (dataclasses, `asyncio.to_thread`, `list[str]`) is 3.11-safe, but as the pipeline grows (LangGraph, langchain-ollama, qdrant-client) version-specific behavior differences (e.g., `asyncio` semantics, SSL defaults changed in 3.10+) can surface only in production.
- Safe modification: Pin the local toolchain: add `.python-version` (`3.11`) and rebuild `venv`, or bump `Dockerfile` to match 3.14 once LangGraph/ollama support it.

## Scaling Limits

### Single-container, single-tenant-bucket design not yet defined

- Current capacity: N/A — the store layer is unimplemented.
- Limit: The spec assumes one Qdrant collection (`QDRANT_COLLECTION=brand_chunks`) with tenant isolation via metadata filtering. At high tenant/corpus counts, collection-wide HNSW + BM25 scans and unfiltered RRF degrade; per-tenant payload indexing and sharding are not designed.
- Scaling path: Use Qdrant payload indexes on `tenant_id`, per-tenant collections or tenants-as-groups if isolation becomes hard, and add `FINAL_TOP_K`/resource quotas per request. `rate limits` are not in the spec and not implemented.

## Dependencies at Risk

### firecrawl-py (unpinned, API-coupled)

- Risk: No version pin in `requirements.txt`; `app/ingestion/extractors/firecrawl.py` uses `from firecrawl.v1 import AsyncV1FirecrawlApp` guarded by `# type: ignore[attr-defined]` (line 12) — a version-sensitive import path that has already changed across SDK releases (v0 → v1 API named "v1"). `AttributeError`-style breakage on upgrade is likely.
- Impact: Ingestion (the only implemented feature) breaks on any `pip install` upgrade.
- Migration plan: Pin the exact SDK version; wrap SDK interaction behind `WebExtractor` protocol (already the pattern in `app/ingestion/extractor.py`) and add a contract test against a recorded SDK response.

### langchain-ollama / Ollama

- Risk: `langchain-ollama` in `requirements.txt` but no `OllamaEmbeddings` code exists yet; `OLLAMA_BASE_URL=http://localhost:11434` requires a locally running Ollama with the `qwen3-embedding:4b` model pulled (`EMBEDDING_DIMENSION=1024` must match the model). Nothing validates model presence or dimension at startup — a mismatched dimension will surface as a Qdrant hard error later.
- Impact: Future embeddings layer fails in confusing ways (empty/dim-mismatch responses).
- Migration plan: Add a startup or first-use health check that verifies Ollama reachability, model list, and embedding dimension before writing to the vector store.

## Missing Critical Features

### Entire API + pipeline absent

- Problem: No FastAPI app (`app/main.py`), no routes (`/health`, `/ingest`, `/ask`), no schemas (`app/api/schemas.py`), no DI (`app/dependencies.py`). Spec defines exact endpoints and error codes.
- Blocks: Everything user-facing; the deployment artifacts (`Dockerfile`, `docker-compose.yml`) cannot run.

### Tenant isolation

- Problem: Multi-tenant ingestion and query isolation (spec requirement, and the subject of the emptied `tests/test_tenant_isolation.py`) is unimplemented — no tenant header handling, no `tenant_id` in `ExtractedDocument` metadata (`app/ingestion/models.py` has only content/source/title/metadata), no filtered Qdrant queries.
- Blocks: Safe multi-tenant use; the product is single-tenant-capable only.

### Idempotent ingestion

- Problem: Spec requires deterministic/idempotent document ingestion; `app/ingestion/service.py` is empty; there is no content-hash or `document_id` field in `ExtractedDocument`.
- Blocks: Re-ingesting the same URL creates duplicates; no update/delete semantics.

### Agent trace + citations

- Problem: Spec requires an agent execution trace and source citations on `/ask`; `app/graph/*` is empty.
- Blocks: The core Q&A value proposition and the `test_agent_decision.py` suite.

## Test Coverage Gaps

### Tenant isolation and agent decision suites are empty

- What's not tested: Everything in `tests/test_tenant_isolation.py` (0 bytes) and `tests/test_agent_decision.py` (0 bytes) — the two most important behavior suites per the spec.
- Files: `tests/test_tenant_isolation.py`, `tests/test_agent_decision.py`
- Risk: Tenant data leakage and wrong agent routing/termination decisions ship unnoticed.
- Priority: High

### No tests for the remaining implemented slice

- What's not tested: `app/config.py` — no test for `.env` parsing, `field_validator` tag-splitting, or the missing-settings silent-drop behavior.
- Files: `app/config.py`, `tests/conftest.py`
- Risk: Config drift (documented above) goes completely unnoticed; `extra="ignore"` hides typos.
- Priority: Medium

### No integration tests

- What's not tested: Any real Firecrawl API call, Qdrant interaction, or Ollama embedding call. `tests/test_firecrawl_extractor.py` (the only live suite, 23 tests, all passing per `.pytest_cache`) uses `FakeFirecrawlClient` and monkeypatched `socket.getaddrinfo` exclusively.
- Files: `tests/test_firecrawl_extractor.py`, `tests/conftest.py`
- Risk: SDK contract drift (firecrawl-py) and OOM/network error handling in the real path are unverified; response-format regressions in the SDK break extraction silently.
- Priority: Medium

---

*Concerns audit: 2026-09-16*