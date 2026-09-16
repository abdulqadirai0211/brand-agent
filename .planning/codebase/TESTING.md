# Testing Patterns

**Analysis Date:** 2026-09-16

## Test Framework

**Runner:**
- pytest 9.1.1 (venv: `venv/bin/pytest`; `venv/pyvenv.cfg` shows Python 3.14.4)
- Config: **none** — no `pytest.ini`, `pyproject.toml`, or `setup.cfg` exists. pytest defaults are in effect (rootdir auto-detected, `tests/` discovered via glob).

**Assertion Library:**
- Builtin `assert` statements only. No `assertpy`, `hamcrest`, or `expect` library.

**Run Commands:**
```bash
./venv/bin/python -m pytest tests/ -q          # Run all tests (37 passed, ~0.23s)
./venv/bin/python -m pytest tests/test_firecrawl_extractor.py -q   # Single file
./venv/bin/python -m pytest -k "unsafe or ssrf"                    # Name filter
```

**Dependencies:**
- `pytest` and `pytest-asyncio` are both listed in `requirements.txt`, **but pytest-asyncio is not used** — no `asyncio_mode` config and no `@pytest.mark.asyncio` anywhere. Async code is driven synchronously with `asyncio.run(...)` inside test bodies (see Async Testing below).

## Test File Organization

**Location:**
- A single `tests/` directory mirroring `app/` — NOT co-located with source. `tests/__init__.py` exists (package-style layout).
- Implemented: `tests/test_firecrawl_extractor.py` (277 lines) covers `app/ingestion/extractors/firecrawl.py` + `app/ingestion/extractor.py`.
- Stubs awaiting tests: `tests/test_agent_decision.py`, `tests/test_tenant_isolation.py` (both 0 bytes). Follow the `test_<module>.py` naming.

**Naming:**
- Files: `test_<target_module>.py`.
- Functions: `test_<behavior_under_test>` in flattening snake_case, e.g., `test_extract_returns_document`, `test_empty_markdown_raises_empty_extraction`, `test_blocks_host_that_resolves_to_private_ip` (`tests/test_firecrawl_extractor.py:60, 96, 211`).
- No test classes — plain module-level functions.

**Structure:**
```
tests/
├── conftest.py                     # Shared fixtures (settings)
├── test_firecrawl_extractor.py     # extract() behavior + SSRF guard + factory
├── test_agent_decision.py          # (stub)
└── test_tenant_isolation.py        # (stub)
```

## Test Structure

**Suite Organization:**
```python
# Module-level test fixtures/helpers first (MARKDOWN, FakeFirecrawlClient,
# fake_result, make_extractor), then section-divided test groups:

# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------
def test_extract_returns_document():
    ...
```
Grouping matches the code's public surface: `extract()` → SSRF guard → construction/factory (`tests/test_firecrawl_extractor.py:55-57, 171-173, 245-247`).

**Patterns:**
- **Arrange → Act → Assert** with `asyncio.run()` as the Act boundary:
  ```python
  def test_extract_returns_document():
      client = FakeFirecrawlClient(result=fake_result())
      extractor = make_extractor(client=client)

      doc = asyncio.run(extractor.extract("https://example.com/asana"))

      assert isinstance(doc, ExtractedDocument)
      assert doc.source == "https://example.com/asana"
  ```
  (`tests/test_firecrawl_extractor.py:60-71`)
- One behavior per test, asserted via multiple `assert` lines (no chained/soft asserts).
- Keyword args and recorded calls verify exactly what the SUT passes downstream: `assert call["formats"] == ["markdown"]` (`tests/test_firecrawl_extractor.py:80-84`) and `assert client.calls == []` to prove early-exit ordering (`tests/test_firecrawl_extractor.py:208`).
- `SimpleNamespace` builds minimal fake API responses with only the fields the code reads (`tests/test_firecrawl_extractor.py:37-43`).

**Teardown:**
- None needed — no test writes files, opens servers, or creates global state. Network is never touched (all clients are fakes or `getaddrinfo` is monkeypatched).

## Mocking

**Framework:** None. No `unittest.mock`, no `mocker`, no `responses`/`respx`. Two hand-rolled techniques:

**Technique 1 — Fake client via dependency injection** (preferred). The extractor accepts `client` in its constructor, so tests inject a recording fake that implements the async interface:
```python
class FakeFirecrawlClient:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = []

    async def scrape_url(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if self.exc is not None:
            raise self.exc
        return self.result
```
(`tests/test_firecrawl_extractor.py:24-34`)

**Technique 2 — `monkeypatch` fixture** for module-level functions that can't be injected:
```python
def test_blocks_host_that_resolves_to_private_ip(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, proto: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 0))
        ],
    )
```
(`tests/test_firecrawl_extractor.py:211-221`)

**What to Mock:**
- External services (Firecrawl API), DNS resolution, anything with network I/O.
- Failure injection through the fake (`exc=TimeoutError(...)`) to test error wrapping.

**What NOT to Mock:**
- The code under test's own logic — the SSRF validator runs for real against the faked `getaddrinfo`, and the factory `get_extractor()` constructs real objects.

## Fixtures and Factories

**Test Data:**
- Module-level constants for repeated payloads:
  ```python
  MARKDOWN = (
      "# Asana Reviews\n\nAsana is a work management tool used by teams.\n\n"
      "It supports projects, tasks, and timelines.\n"
  ) * 40
  ```
  (`tests/test_firecrawl_extractor.py:18-21`)
- Builder helpers with overridable defaults:
  ```python
  def fake_result(markdown=MARKDOWN, title="Asana Reviews", error=None, url=None):
      return SimpleNamespace(markdown=markdown, title=title, error=error, url=url or "...")
  ```
  (`tests/test_firecrawl_extractor.py:37-43`)

**Shared fixture — `conftest.py`:**
```python
@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_name="brand-qa-agent",
        environment="test",
        log_level="WARNING",
        web_extractor="firecrawl",
        firecrawl_api_key="fc-test-key",
        ...
    )
```
(`tests/conftest.py:6-19`) — a single typed `Settings` fixture used directly (no factories/parametrization of settings at the fixture level; per-test overrides are constructed inline, e.g., `Settings(web_extractor="crawl4ai", firecrawl_api_key="fc-test-key")` at `tests/test_firecrawl_extractor.py:268`).

**Location:** In-file for ephemeral helpers, `tests/conftest.py` for shared fixtures.

## Coverage

**Requirements:** None enforced — no coverage config, no `--cov` anywhere, no `.coveragerc`.

**View Coverage:**
```bash
./venv/bin/python -m pytest tests/ --cov=app --cov-report=term-missing   # requires pytest-cov (not installed)
```
Not currently runnable: `pytest-cov` is not in `requirements.txt`. Install it if a coverage gate is added.

**Observed effective coverage:** high for the implemented surface — every branch of `FirecrawlExtractor.extract` and the SSRF validator has a corresponding test (short-content, empty, SDK exception, None result, missing optional fields, blocked IP families, DNS-resolve-to-private, public URLs, factory error paths).

## Test Types

**Unit Tests:**
- The only type present. Scope: `app/ingestion/extractor.py` + `app/ingestion/extractors/firecrawl.py`, exercised entirely in-process with fakes. 37 tests, all passing in 0.23s (no I/O).

**Integration Tests:**
- None. `scripts/seed_corpus.py` (stub), `app/main.py` (0 bytes), and all API routes (`app/api/routes/`) are unimplemented, so there is no app/endpoint integration surface yet.

**E2E Tests:**
- Not used.

## Common Patterns

**Async Testing:**
- Run coroutines to completion inside sync test functions with `asyncio.run()`:
  ```python
  with pytest.raises(EmptyExtractionError) as excinfo:
      asyncio.run(extractor.extract("https://example.com/asana"))
  assert excinfo.value.code == "empty_extraction"
  ```
  (`tests/test_firecrawl_extractor.py:96-103`)
- This avoids async-fixture plumbing entirely; the code under test is constructed with injected clients, so a running loop is unnecessary.
- Do **not** introduce `@pytest.mark.asyncio` / `asyncio_mode = auto` — it would diverge from the established pattern and require enabling the currently-inert pytest-asyncio plugin.

**Error Testing:**
- `pytest.raises(<DomainError>)` with the `match=` regex for message pinning on factory errors: `pytest.raises(ValueError, match="FIRECRAWL_API_KEY")` (`tests/test_firecrawl_extractor.py:157`).
- `excinfo.value.code` asserts the machine-readable error code; `str(excinfo.value)` asserts the human message (`tests/test_firecrawl_extractor.py:103, 124-135`).
- Security-oriented error test: assert sensitive values never appear in error text (`test_api_key_never_leaked_in_errors`, `tests/test_firecrawl_extractor.py:146-153`).

**Parametrized Data Tables:**
- `@pytest.mark.parametrize` with raw-tuple lists for input/output tables — SSRF blocklist (13 URLs), host variants, public-URL allowlist (`tests/test_firecrawl_extractor.py:176-196, 199-208, 224-242`). Keep the parametrize list adjacent to the test and read it as a data table.

**Negative-Path Proof:**
- For ordering guarantees, assert the side effect DID NOT happen: `assert client.calls == []` after a rejected URL proves the SSRF guard runs before any network call (`tests/test_firecrawl_extractor.py:208`).

**Private attribute assertions:**
- Tests read private state of the SUT directly for construction verification: `assert extractor._only_main_content is True` (`tests/test_firecrawl_extractor.py:254-258`). Acceptable for fixture-style assertions on the extracted class; prefer public behavior assertions in the general case.

---

*Testing analysis: 2026-09-16*