# Brand Q&A Agent — File-by-File Technical Specification

## 1. Purpose

This document defines the exact responsibility, interfaces, inputs, outputs, and implementation boundaries of every source, test, configuration, and deployment file in the Brand Q&A Agent backend.

The implementation follows the project requirements:

- Multi-tenant web/raw-text ingestion
- Structure-aware chunking
- Ollama `qwen3-embedding:4b` dense embeddings (via `langchain-ollama`)
- BM25 sparse retrieval
- Pinecone vector storage (serverless, hybrid dense + BM25 in one index)
- Hybrid retrieval via `alpha` query weighting (no RRF)
- `cross-encoder/ms-marco-MiniLM-L6-v2` reranking
- LangGraph-based question answering agent
- FastAPI REST API
- Strict tenant isolation
- Deterministic/idempotent document ingestion
- Source citations
- Agent execution trace
- Automated tests
- Docker-ready deployment

---

# 2. Project Structure

```text
brand-qa-agent/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       ├── ingest.py
│   │       └── ask.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── extractor.py
│   │   ├── service.py
│   │   └── extractors/
│   │       ├── __init__.py
│   │       └── webbaseloader.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── markdown_parser.py
│   │   └── validator.py
│   │
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── splitter.py
│   │   └── document_chunker.py
│   │
│   ├── extraction/
│   │   ├── __init__.py
│   │   └── web_loader.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── service.py
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── pinecone.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── service.py
│   │
│   └── graph/
│       ├── __init__.py
│       ├── state.py
│       ├── nodes.py
│       ├── edges.py
│       └── graph.py
│
├── tests/
│   ├── conftest.py
│   ├── test_tenant_isolation.py
│   └── test_agent_decision.py
│
├── scripts/
│   └── seed_corpus.py
│
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

# 3. Application Layer

## `app/__init__.py`

### Responsibility

Marks `app` as the Python application package.

### Requirements

- No application logic.
- No environment loading.
- No imports with side effects.

---

## `app/main.py`

### Responsibility

FastAPI application entry point.

### Responsibilities

1. Create the `FastAPI` application.
2. Register API routers.
3. Configure application-level metadata.
4. Configure startup/shutdown lifecycle if required.
5. Expose the application object for:
   - Uvicorn
   - Docker
   - tests

### Expected interface

```python
app = FastAPI(...)
```

### Routes

```text
GET  /health
POST /tenants/{tenant_id}/ingest
POST /tenants/{tenant_id}/ask
```

### Boundaries

`main.py` must not contain:

- Pinecone logic
- embedding logic
- scraping logic
- LangGraph node implementation
- business logic

It only composes the application.

---

# 4. Configuration

## `app/config.py`

### Responsibility

Centralized typed application configuration.

Use Pydantic Settings.

### Configuration groups

#### Application

```text
APP_NAME
ENVIRONMENT
LOG_LEVEL
```

#### Extraction

```text
WEB_LOADER_TIMEOUT_MS
WEB_USER_AGENT
MIN_CONTENT_LENGTH
```

Single `WebBaseLoader` backend (`app/extraction/web_loader.py`).

#### Chunking

```text
CHUNK_SIZE=400
CHUNK_OVERLAP=50
```

#### Embeddings (Ollama)

```text
EMBEDDING_MODEL=qwen3-embedding:4b
EMBEDDING_DIMENSION=1024
OLLAMA_BASE_URL=http://localhost:11434
```

#### Pinecone

Serverless index storing both dense and sparse vectors per record; metric MUST
be `dotproduct` (the only metric accepting sparse). `dimension` must match
`EMBEDDING_DIMENSION` (1024).

```text
PINECONE_API_KEY
PINECONE_INDEX=brand-chunks
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_ALPHA=0.5
```

`PINECONE_ALPHA` controls the dense/sparse balance at query time:
`score = alpha * dense + (1 - alpha) * sparse` (0 = sparse only, 1 = dense only).

#### Reranking

```text
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
RERANKER_DEVICE=cpu
```

Small (~80 MB), CPU-friendly MS MARCO passage-ranking cross-encoder
(Apache-2.0); downloads once on first run via `sentence-transformers`. The
survivor count is `FINAL_TOP_K`; over-retrieval is `HYBRID_TOP_K`.

#### Retrieval

```text
HYBRID_TOP_K=10
FINAL_TOP_K=5
```

#### LLM

Provider/model/API configuration required by the selected LLM. Groq
(`langchain-groq`) is the primary provider; an OpenAI / OpenAI-compatible
endpoint (`langchain-openai`, honoring `OPENAI_BASE_URL`) is the optional
fallback. If neither is configured, `/ask` responds `503`.

```text
GROQ_API_KEY
GROQ_MODEL=openai/gpt-oss-120b
OPENAI_API_KEY
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=
LLM_STRUCTURED_OUTPUT_METHOD=json_schema
TEMPERATURE=0.0
```

### Requirements

- No hardcoded secrets.
- `.env` support.
- Strong validation.
- Defaults only for non-secret values.
- Configuration should be injectable into services.

---

## `app/dependencies.py`

### Responsibility

FastAPI dependency wiring and application service construction.

### Responsibilities

Provide reusable instances for:

```text
Settings
EmbeddingService
PineconeVectorStore
Retriever
Reranker
QA Graph
IngestionService
```

### Important requirement

Avoid constructing expensive models on every request.

Embedding and reranker models should be initialized once and reused.

---

# 5. API Layer

## `app/api/schemas.py`

### Responsibility

All API request/response Pydantic models.

---

## Ingestion request

The PDF requires a batch endpoint. The request body contains a list of documents:

```python
class DocumentInput(BaseModel):
    url: str | None = None
    text: str | None = None
    brand: str
    source: str | None = None


class IngestRequest(BaseModel):
    documents: list[DocumentInput]
```

Validation:

- `documents` must be non-empty.
- For every document, at least one of `url` or `text` must be provided.
- `brand` is required per document.
- `source` is optional; when absent, the URL (for URL documents) or the stable
  `"raw_text"` (for text documents) is used, matching the notebook pipeline.

---

## Ingestion response

Aggregate result matching the PDF contract:

```python
class IngestResponse(BaseModel):
    documents_stored: int
    chunks_stored: int
    errors: list[DocumentError] = []
```

- `documents_stored` counts the unique documents upserted in the request.
- `chunks_stored` counts the total chunks upserted across all documents.
- Re-ingesting the same document must not increase `chunks_stored` (same
  `doc_id` → same `{doc_id}-{chunk_index}` vector ids → upsert overwrites).
- `errors` is additive: a document that fails extraction (unreachable URL,
  too-short content, …) is reported here instead of being silently dropped or
  aborting the batch.

---

## Ask request

```python
class AskRequest(BaseModel):
    question: str
```

---

## Citation

```python
class Citation(BaseModel):
    source: str
    doc_id: str
    chunk_index: int
    brand: str
    chunk_text: str
```

`chunk_text` matches the PDF citation example and shows the exact text the
answer is based on.

---

## Ask response

Required structure:

```python
class AskResponse(BaseModel):
    answer: str
    found_in_corpus: bool
    citations: list[Citation]
    trace: list[TraceStep]
```

---

## Trace

```python
class TraceStep(BaseModel):
    node: str
    status: str
    details: dict
```

Trace should expose execution information without exposing secrets or sensitive internal configuration.

---

# 6. API Routes

## `app/api/routes/health.py`

### Responsibility

Health endpoint.

### Endpoint

```text
GET /health
```

### Expected behavior

The PDF requires HTTP 200 with a small JSON showing whether the vector
database and the LLM provider are reachable. Example:

```json
{
  "status": "ok",
  "pinecone": "ok",
  "llm": "ok"
}
```

The route is lightweight: reachability checks must be fast (timeouts in the
low single-digit seconds) and must not block on full model initialization.

---

## `app/api/routes/ingest.py`

### Responsibility

HTTP adapter for document ingestion.

### Endpoint

```text
POST /tenants/{tenant_id}/ingest
```

### Flow

```text
HTTP request
    ↓
Validate request
    ↓
IngestionService (async, batched)
    ↓
response (aggregate)
```

### Async requirement

The handler and the whole ingestion pipeline must be async — URL fetching,
embedding, and Pinecone calls must not block the event loop.

### Error responses

- A document providing neither `url` nor `text`, providing both, or missing
  `brand` → `422` from Pydantic (`DocumentInput` validator).
- A document that fails extraction/validation is reported per document in the
  aggregate `errors` list and does not abort the rest of the batch (e.g.
  `{"url": "...", "code": "fetch_failed" | "invalid_url" | "empty_extraction",
  "message": "..."}`).

### Must not contain

- scraping implementation
- chunking implementation
- embedding implementation
- Pinecone implementation

---

## `app/api/routes/ask.py`

### Responsibility

HTTP adapter for question answering.

### Endpoint

```text
POST /tenants/{tenant_id}/ask
```

### Flow

```text
HTTP request
    ↓
validate
    ↓
QA Graph (async)
    ↓
AskResponse
```

### Async requirement

The handler and every LLM call inside the graph nodes must be async.

### Error responses

- Empty question → `422` (Pydantic validation).
- LLM not configured at all (no `GROQ_API_KEY` / `OPENAI_API_KEY`) → `503`
  with `{"detail": {"error": "llm_unconfigured", "message": "..."}}`.
- LLM timeout / provider error at request time → `502` with
  `{"detail": {"error": "llm_unavailable", "message": "..."}}`.
- A tenant with no ingested data is **not** an error. Per the spec's Example 3
  (`globex`, which ingested nothing, asks and gets "not in corpus"), the route
  returns HTTP 200 with `found_in_corpus=false`, an honest answer stating the
  corpus does not cover the question, and an empty `citations` list. The trace
  shows `retrieve` (0 chunks) → `grade` (0 relevant) → answer.
- When the corpus genuinely cannot answer, this is likewise **not** an error:
  HTTP 200 with `found_in_corpus=false`, an honest answer, and empty
  `citations`.

### Important

`tenant_id` must be passed into the graph state and must never be silently replaced by a default tenant.

---

# 7. Ingestion Layer

## `app/ingestion/models.py`

### Responsibility

Typed internal representations of extracted documents.

### `ExtractedDocument`

Should contain at least:

```text
text
url
title
brand
extractor
```

Example:

```python
class ExtractedDocument(BaseModel):
    text: str
    url: str
    title: str | None = None
    brand: str
    extractor: str
```

### Purpose

Decouple extraction providers from preprocessing/chunking.

### `ChunkMetadata`

Structured metadata attached to every chunk (the PDF B.4 contract):

```python
class ChunkMetadata(BaseModel):
    tenant_id: str
    doc_id: str
    chunk_index: int
    source: str
    brand: str
```

---

## `app/ingestion/document_id.py`

### Responsibility

Deterministic, idempotent document IDs (the PDF B.4 contract).

`SHA-256` of the input so the same content always produces the same `doc_id`:

```python
def generate_doc_id(content: str) -> str: ...
def generate_url_doc_id(url: str) -> str: ...   # sha256(url.strip())
def generate_text_doc_id(text: str) -> str: ...  # sha256(text.strip())
```

URLs hash the canonical URL (not the fetched text), so re-ingesting a URL
overwrites old chunks with freshly extracted content instead of duplicating.

---

## `app/extraction/web_loader.py`

### Responsibility

`WebBaseLoader` extraction — the ingestion pipeline exactly as built in
`experiment.ipynb`.

### Responsibilities

1. Accept URL + brand.
2. Validate URL before fetching (reject empty / non-HTTP(s) scheme / missing
   hostname) — `UnsafeURLError` (`invalid_url`).
3. Fetch via `WebBaseLoader.load()` with `requests_kwargs` `timeout=30` and a
   realistic Chrome `User-Agent` header.
4. Return a single `ExtractedDocument` (`extractor="webbase_loader"`).

### Must handle

- unreachable URL / HTTP error / timeout → `FetchFailedError` (`fetch_failed`)
- empty list / empty content (`EmptyExtractionError`, `empty_extraction`)
- short/empty content below `min_content_length`

### Headers

```python
requests_kwargs={"timeout": 30, "headers": {"User-Agent": "Mozilla/5.0 (Macintosh; ...) Chrome/151.0 ..."}}
```

Configurable via `web_loader_timeout_ms` / `web_user_agent`.

### Security

Validate URLs before fetching (scheme + hostname check). Per the deliverable,
this is a simple bad-URL guard, not a DNS/SSRF resolver.

Note: G2/digest-style pages behind Cloudflare may still return 403 to any
scripted client. Ingest should report a per-document fetch error rather than
failing the whole batch.

---

## `app/ingestion/service.py`

### Responsibility

Orchestrate the complete ingestion pipeline.

### Input

A batch of documents (per the PDF):

```text
tenant_id
documents: [{url OR text, brand, optional source}, ...]
```

### Pipeline

Runs per document (concurrently where safe):

```text
Input
  ↓
Validate
  ↓
Generate deterministic doc_id
  ↓
Extract if URL
  ↓
Clean
  ↓
Parse structure
  ↓
Chunk
  ↓
Embed
  ↓
Upsert into tenant scope
```

### Batch contract

- Process every document in the request.
- Return aggregate counts (`documents_stored`, `chunks_stored`) per the PDF.
- Duplicate documents inside the same request collapse into one via `doc_id`.
- A failing document is reported, not silently dropped.

### Idempotency

Document ID should be deterministic.

For URL:

```text
SHA-256(canonical URL)
```

For raw text:

```text
SHA-256(normalized text)
```

If the same document is ingested again:

```text
same doc_id
same chunk IDs
upsert
```

rather than creating duplicate vectors. The vector ID `{doc_id}-{chunk_index}`
is reused on re-ingest, so the chunk count stays flat (the PDF explicitly tests
this: ingesting the same documents twice must not grow the stored count).

### Async requirement

The pipeline must be async end-to-end. URL fetching uses `httpx.AsyncClient`
(or equivalent); embedding and Pinecone calls are non-blocking (async clients or
threaded executors where the SDK is sync).

### Important

Document IDs and vector IDs must be deterministic.

---

# 8. Preprocessing Layer

## `app/preprocessing/cleaner.py`

### Responsibility

Normalize extracted content before chunking.

### Operations

Potential operations:

- normalize whitespace
- remove obvious extraction artifacts
- normalize line endings
- preserve Markdown structure
- remove empty sections
- preserve links
- preserve tables
- preserve lists

### Must NOT

- summarize content
- rewrite factual content
- use an LLM to clean content
- remove meaningful source information

The source should remain faithful to the original page.

---

## `app/preprocessing/markdown_parser.py`

### Responsibility

Handle Markdown structural information.

### Output

Represent content with heading context.

Example:

```text
H1: Pricing
H2: Enterprise
content...
```

### Metadata to preserve

Chunk metadata is exactly the five fields from the PDF (B.4):

```text
tenant_id
doc_id
chunk_index
source
brand
```

This context should be available to retrieval and citations.

---

## `app/preprocessing/validator.py`

### Responsibility

Validate extracted documents before ingestion.

### Checks

- content is non-empty
- content has minimum useful length
- source is valid
- document metadata is valid
- URL is allowed
- extraction did not return an error page

### Failure behavior

Raise typed application exceptions rather than generic exceptions.

---

# 9. Chunking

## `app/chunking/splitter.py`

### Responsibility

Character-bounded chunking.

### Required strategy

Single-stage:

```text
RecursiveCharacterTextSplitter
            ↓
~400 characters
50 character overlap
```

### Length function

Chunk limits use Python's builtin `len` (character count) as the splitter's
`length_function`. No external tokenizer is required, so there is no Hugging
Face / local tokenizer dependency and no tokenization network call.

Do not rely on an unrelated tokenizer when determining the final embedding
chunk size.

### Target

```text
chunk_size = 400 characters
chunk_overlap = 50 characters
```

### Metadata required on every chunk

```text
tenant_id
doc_id
chunk_index
source
brand
```

### Vector ID

```text
{doc_id}-{chunk_index}
```

### Design constraints

- Preserve semantic boundaries.
- Avoid splitting tables unnecessarily.
- Avoid splitting lists unnecessarily where possible.
- Keep heading context.
- Do not force every chunk to be exactly 400 characters.

The default separators split on `\n\n`, `\n`, space, and `` so paragraphs and
words are rarely cut mid-sentence. No metadata beyond the five B.4 fields is
added (no `heading_path`, no `token_count`). To change sizes without changing
architecture, pass different `chunk_size`/`chunk_overlap` to
`TextChunker`/`DocumentChunker`.

---

## `app/chunking/document_chunker.py`

### Responsibility

Attach the B.4 metadata to every split.

### Interface

```python
class DocumentChunker:
    def create_chunks(
        self,
        text: str,
        tenant_id: str,
        doc_id: str,
        source: str,
        brand: str,
    ) -> list[Document]:  # langchain Document with ChunkMetadata
```

- Delegates splitting to `TextChunker` (`app/chunking/splitter.py`).
- Returns LangChain `Document`s whose `metadata` is exactly
  `tenant_id, doc_id, chunk_index, source, brand` (via `ChunkMetadata.model_dump()`).
- `vector_id = f"{doc_id}-{chunk_index}"` is derived in the vector store, so
  re-ingesting the same URL/text overwrites the same vector IDs (idempotent).

### Optional tuning

The implementation should make it easy to benchmark:

```text
300/50
400/50
500/50
```

without changing the architecture.

---

# 10. Embeddings

## `app/embeddings/service.py`

### Responsibility

Generate dense embeddings.

### Model

Ollama embeddings via LangChain:

```text
qwen3-embedding:4b
OLLAMA_BASE_URL=http://localhost:11434
```

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:4b",
    base_url="http://localhost:11434",
    dimensions=1024,
)
```

### Dimension

```text
1024
```

Note: `qwen3-embedding:4b` defaults to 2560 dimensions. The Matryoshka-style
`dimensions=1024` request must be set explicitly so vectors match
`EMBEDDING_DIMENSION`.

### Interface

```python
embed_documents(texts: list[str]) -> list[list[float]]

embed_query(query: str) -> list[float]
```

### Requirements

- Batch document embedding.
- Separate query embedding method.
- Normalize embeddings consistently.
- Reuse model instance.
- Configurable Ollama endpoint (`base_url`) and `dimensions`.
- Avoid re-initializing the client per request.

### Expected flow

```text
chunks
  ↓
qwen3-embedding:4b (Ollama)
  ↓
1024-dimensional vectors
```

---

# 11. Vector Store

## `app/vectorstore/pinecone.py`

### Responsibility

All Pinecone-specific persistence and search operations. Backed by a serverless
index created with the client pattern `pc.has_index` / `pc.create_index(...,
Spec=ServerlessSpec(...))` / `pc.Index(index_name)` — the same idiom LangChain's
`PineconeVectorStore` is built on, kept as a thin wrapper so the hybrid upsert,
namespace isolation, and `{doc_id}-{chunk_index}` ID scheme stay under our
control.

### Index

Configured using:

```text
PINECONE_INDEX
PINECONE_CLOUD
PINECONE_REGION
```

```text
dimension = 1024        # must equal EMBEDDING_DIMENSION
metric    = dotproduct  # the only metric that accepts sparse vectors
```

### Dense + sparse in one record

Each record carries both a dense vector and a BM25 sparse vector:

```python
{"id": "{doc_id}-{chunk_index}",
 "values": [...1024 floats...],
 "sparse_values": {"indices": [...], "values": [...]},
 "metadata": {...}}
```

Sparse vectors are produced by `Bm25SparseVectorizer` in `app/retrieval/bm25.py`
via the `SparseEncoder` protocol. It is fit-free and deterministic (stable
hash `term -> 24-bit feature id`, raw term frequencies as values), so document
and query vectors align without any persisted corpus vocabulary or hidden
state. `pinecone_text.sparse.BM25Encoder` remains available as an injectable
alternative encoder (fitted per ingest batch); when used it requires the NLTK
`stopwords`, `punkt_tab`, and `snowball_data` corpora.

### Metadata/payload

Every record should contain:

```json
{
  "tenant_id": "...",
  "doc_id": "...",
  "chunk_index": 0,
  "source": "...",
  "brand": "...",
  "text": "..."
}
```

### Required operations

```python
ensure_index()
persist_chunks(chunks) -> int
delete_document(tenant_id, doc_id)
count(tenant_id=None) -> int
dense_search(query_vector, tenant_id, limit)
sparse_search(query_vector, tenant_id, limit)
hybrid_search(query_vector, sparse_vector, tenant_id, limit, alpha)  # single query
embed_query(text)
encode_query(text)
```

`hybrid_search` scales the two query vectors with
`pinecone_text.hybrid.hybrid_convex_scale` (convex `alpha` combination) and
sends ONE `index.query(vector=..., sparse_vector=...)` request. No client-side
fusion is required — Pinecone dot-products the weighted dense and sparse query
vectors against each stored record.

### Tenant isolation

Two mechanisms, both applied ("do both", per PDF B.3):

1. **Namespace per tenant** — every record is upserted and searched under
   `namespace=tenant_id`; different tenants can never overlap.
2. **Metadata filter** — every query also passes
   `filter={"tenant_id": {"$eq": tenant_id}}` as defense-in-depth.

Never retrieve globally and filter afterward.

The vector database itself must enforce the isolation.

---

# 12. Hybrid Retrieval

## `app/retrieval/hybrid.py`

### Responsibility

Turn a `question` into a single hybrid query and return the winning chunks for
one tenant.

### Pipeline

```text
question
   │
   ├────────────────┐
   ↓                ↓
dense (Ollama)   BM25 (Bm25SparseVectorizer)
   │                │
   └───┬────────────┘
       ↓
 hybrid_convex_scale(dense, sparse, alpha)
       ↓
 Pinecone index.query(vector, sparse_vector)   # one request
       ↓
candidate set (HYBRID_TOP_K)
```

### Candidate counts

```text
HYBRID_TOP_K=10   # initial candidates handed to the reranker
FINAL_TOP_K=5      # post-rerank context used by the answer node
```

### Fusion

No RRF. Dense and BM25 scores live on incomparable scales (cosine-normed dense
in `[-1, 1]` vs unbounded BM25), so the store applies a convex combination on
the query vectors before the (already scaled for dotproduct) server-side score:
`score = alpha * dense + (1 - alpha) * sparse`, with `alpha = PINECONE_ALPHA`
(0.5 default).

### Output

Return ranked candidate chunks with metadata.

### Tenant isolation

The retriever receives:

```text
tenant_id
```

and passes it into every store call (namespace + metadata filter).

---

# 13. Reranker

## `app/retrieval/reranker.py`

### Responsibility

Cross-encoder reranking of hybrid candidates.

### Model

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

Small, fast, CPU-friendly cross-encoder trained on the MS MARCO passage
ranking task (~80 MB, 22.7M params, Apache-2.0). Downloads once on first run
and is loaded through `sentence_transformers.CrossEncoder`.

Over-retrieve `fetch_k = HYBRID_TOP_K` (10), then keep `top_k = FINAL_TOP_K`
(5) after reranking — the same fetch_k/top_k 10/5 split as the hybrid search.

### Input

```text
query
candidate documents
```

### Output

Candidates sorted by reranker relevance.

### Flow

```text
Hybrid candidates
      ↓
cross-encoder reranker
      ↓
top 5
```

### Important distinction

The reranker is NOT the LangGraph relevance grader.

There are two separate responsibilities:

```text
Reranker
→ retrieval ranking

LLM grade node
→ agent decision: relevant / not relevant
```

---

# 14. LLM Providers

## `app/llm/service.py`

### Responsibility

Construct the chat model and expose the three LLM-backed callables the graph
nodes depend on. Keeping them behind injected callables lets the graph be tested
with deterministic fakes (no network) and lets the provider be swapped.

### Provider selection

```text
get_chat_model()
  |-- GROQ_API_KEY set            -> langchain_groq.ChatGroq(GROQ_MODEL)
  |-- else OPENAI_API_KEY set     -> langchain_openai.ChatOpenAI(OPENAI_MODEL, base_url=OPENAI_BASE_URL)
  +-- neither                     -> raise LLMUnconfiguredError (route -> 503)
```

### Callables

```python
Grader    = Callable[[str, list[Document]], Awaitable[list[int]]]
Rewriter  = Callable[[str], Awaitable[str]]
Generator = Callable[[str, str], Awaitable[str]]
```

- `build_grader(chat, method=LLM_STRUCTURED_OUTPUT_METHOD)` - structured output
  (`json_schema` by default) returning `relevant_indices` + `explanation`.
- `build_rewriter(chat)` - returns an improved, searchable question.
- `build_generator(chat)` - grounded answer from the relevant context only.

### Requirements

- Async invocation (`ainvoke`) for LLM calls.
- `temperature` from settings.
- No secrets in code; keys come from `.env`.

---

# 15. LangGraph State

## `app/graph/state.py`

### Responsibility

Define the state passed between graph nodes.

Required fields:

```python
from typing import Annotated, TypedDict
import operator

class QAState(TypedDict):
    tenant_id: str
    question: str
    active_query: str            # original question, or the rewritten query after a retry
    retrieved_chunks: list
    relevant_chunks: list
    retry_count: int
    answer: str
    found_in_corpus: bool
    trace: Annotated[list, operator.add]   # appended by each node
```

### State rules

- `tenant_id` must remain unchanged.
- `retry_count` must prevent infinite loops.
- `trace` is an `Annotated[list, operator.add]` channel so every node appends
  exactly one entry without overwriting earlier steps.
- Nodes receive dependencies (store, retriever, LLM callables) via injection;
  no hidden global state.

---

# 16. LangGraph Nodes

## `app/graph/nodes.py`

### Responsibility

Implement graph node functions.

---

## `retrieve_node`

### Input

```text
tenant_id
question
```

### Operation

Call hybrid retrieval + reranking.

### Output

```text
retrieved_chunks
```

Append trace entry.

---

## `grade_node`

### Responsibility

Determine whether retrieved chunks answer the question.

### Operation

LLM evaluates each retrieved chunk.

Expected conceptual result:

```text
yes
no
```

### Output

```text
relevant_chunks
```

### Important

The LLM must not generate the final answer here.

This node only determines relevance.

---

## `rewrite_node`

### Responsibility

Rewrite the user's question when the initial retrieval produces no relevant chunks.

### Input

```text
original question
```

### Output

```text
rewritten question
```

### Constraints

- Preserve user intent.
- Make the query retrieval-friendly.
- Do not invent facts.

Increment:

```text
retry_count
```

---

## `answer_node`

### Responsibility

Generate the final answer exclusively from relevant chunks.

### Rules

1. Use only retrieved relevant context.
2. Do not introduce unsupported facts.
3. Include citations.
4. If context does not support the answer, do not fabricate.
5. Set `found_in_corpus=True` only when sufficient evidence exists.

### Output

```text
answer
found_in_corpus
citations
```

Append trace.

---

# 17. LangGraph Routing

## `app/graph/edges.py`

### Responsibility

Define conditional routing logic.

### Required logic

```text
GRADE
  │
  ├── relevant chunks exist
  │        ↓
  │      ANSWER
  │
  └── no relevant chunks
           ↓
      retry_count == 0?
        │
        ├── yes → REWRITE → RETRIEVE
        │
        └── no → END
```

### Important

No infinite retry loop.

Maximum retry:

```text
1
```

for the required flow.

---

# 18. LangGraph Construction

## `app/graph/graph.py`

### Responsibility

Construct and compile the LangGraph.

### Graph

```text
START
  ↓
RETRIEVE
  ↓
GRADE
  ├──────────────→ ANSWER
  │
  └→ REWRITE → RETRIEVE
                    ↓
                  GRADE
                    ↓
                   END
```

### Requirements

- Compile graph once.
- Reuse graph instance.
- Inject dependencies rather than creating global model clients inside nodes.
- Use async node functions for LLM calls.
- Every node appends exactly one entry to `trace` describing what it did
  (e.g. `{"node": "retrieve", "status": "ok", "details": {"count": 5}}`).
- `graph.get_graph().draw_ascii()` must work for debugging and for the
  walkthrough video.

---

# 19. Tests

## `tests/conftest.py`

### Responsibility

Shared pytest fixtures.

Fixtures may include:

```text
mock_embeddings
mock_pinecone
mock_llm
mock_reranker
test_graph
```

### Requirements

Tests should not require:

- real WebBaseLoader fetch
- real Pinecone Cloud
- real LLM API
- external internet

---

# 20. Tenant Isolation Test

## `tests/test_tenant_isolation.py`

### Purpose

Prove that tenant A cannot retrieve tenant B's documents.

### Scenario

```text
Tenant A
  └── Asana document

Tenant B
  └── Other brand document
```

Query:

```text
tenant A → can retrieve A
tenant B → cannot retrieve A
```

### Test requirement

Assert isolation at the retrieval layer, not only at the API response layer.

This is important because filtering results after retrieval would not be sufficient isolation.

---

# 21. Agent Decision Test

## `tests/test_agent_decision.py`

### Purpose

Test LangGraph decision logic.

### Cases

#### Case 1

Relevant context exists:

```text
RETRIEVE
→ GRADE: relevant
→ ANSWER
```

#### Case 2

No relevant context initially:

```text
RETRIEVE
→ GRADE: none
→ REWRITE
→ RETRIEVE
```

#### Case 3

Still no relevant context:

```text
RETRIEVE
→ GRADE: none
→ REWRITE
→ RETRIEVE
→ GRADE: none
→ END
```

Expected:

```text
found_in_corpus = false
```

### LLM

Mock the LLM responses.

Do not use real model calls.

---

# 22. Corpus Seeding

## `scripts/seed_corpus.py`

### Responsibility

Convenience script for loading the evaluation corpus.

### Input

A JSON manifest (or the small built-in project-management corpus):

```json
{
  "tenant_id": "acme",
  "documents": [
    {"url": "https://example.com/best-pm-tools", "brand": "Asana"},
    {"text": "Trello is ...", "brand": "Trello", "source": "notes.txt"}
  ]
}
```

```text
--tenant acme            # tenant_id (manifest value wins)
--manifest data/corpus.json
```

### Purpose

Useful for:

- local development
- demo
- evaluation
- reproducible testing

### Must use

The same production ingestion service.

Do not create a second ingestion implementation in the script.

---

# 23. Environment

## `.env.example`

### Responsibility

Document all environment variables required to run the system.

Example categories:

```text
APP
EXTRACTOR
PINECONE
EMBEDDING
RERANKER
LLM
RETRIEVAL
CHUNKING
```

### Requirements

- No real secrets.
- Clearly mark required vs optional values.
- Use safe example placeholders.

---

# 24. Git Ignore

## `.gitignore`

Must exclude:

```text
.env
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.pyc
models/
logs/
.DS_Store
```

Do not commit API keys or local model artifacts.

---

# 25. Python Dependencies

## `requirements.txt`

Dependencies should cover:

### API

```text
fastapi
uvicorn
pydantic
pydantic-settings
```

### LangChain/LangGraph

```text
langchain
langchain-community
langchain-ollama
langgraph
```

### Hugging Face

Required for the reranker (`cross-encoder/ms-marco-MiniLM-L6-v2`):

```text
transformers
sentence-transformers
torch
```

### Vector database

```text
pinecone
pinecone-text
nltk
```

`pinecone_text` is used for `hybrid_convex_scale` and remains the source of the
optional `BM25Encoder`. The default sparse encoder is the fit-free
`Bm25SparseVectorizer`, so NLTK corpora are **not** required for normal
operation; the start script still downloads `stopwords`, `punkt_tab`, and
`snowball_data` so the injectable `BM25Encoder` alternative works if selected.
`nltk` is pinned explicitly so the install step is reproducible.

### Extraction

```text
langchain-community
beautifulsoup4
```

`langchain-community` is sunset upstream (WebBaseLoader lives there). The
deprecation is accepted for this assignment; the sole usage is the URL fetch
path. Migrating off it would only require swapping `_build_loader`.

### Testing

```text
pytest
pytest-asyncio
```

Exact versions should be pinned after validating the final environment.

Avoid unnecessary dependencies.

---

# 26. Dockerfile

## `Dockerfile`

### Responsibility

Build the application container.

### Requirements

- Python 3.11+
- Install dependencies.
- Copy application source.
- Run FastAPI/Uvicorn.
- Support environment variables.
- Do not store secrets in the image.

### SSR note

Some corpus pages (G2 and similar review platforms) front HTTP content with
anti-bot/Cloudflare and will return 403 to any scripted client. Ingest must
surface this as a per-document fetch error rather than failing the whole
batch.

---

# 27. Docker Compose

## `docker-compose.yml`

### Responsibility

Local development environment.

Potential services:

```text
brand-qa-api
```

Pinecone is a managed cloud service (serverless index), so no local vector
database container is needed — the app talks to it directly with the API key in
`.env`.

### Goal

One command should provide a reproducible local environment.

```text
scripts/run.sh
        ↓
FastAPI (main.py)
        ↓
Pinecone Cloud (serverless index)
```

---

# 28. README

## `README.md`

### Required sections

```text
1. Project overview
2. Architecture
3. Features
4. Tech stack
5. Project structure
6. Setup
7. Environment variables
8. Running locally
9. API documentation
10. Ingestion example
11. Ask example
12. Tenant isolation
13. RAG pipeline
14. LangGraph workflow
15. Testing
16. Docker
17. Deployment architecture
18. Design trade-offs
19. Evaluation approach
20. Known limitations
```

### README should explain

Why the system uses:

```text
qwen3-embedding:4b (Ollama)
BM25 (Pinecone hybrid, alpha-weighted)
Pinecone (dense + sparse, namespace-per-tenant)
cross-encoder/ms-marco-MiniLM-L6-v2
LangGraph
```

rather than merely listing technologies.

---

# 29. End-to-End Data Contracts

## Ingestion (batch, idempotent)

```text
POST /tenants/{tenant_id}/ingest

        ↓

IngestRequest {documents: [...]}

        ↓

IngestionService (per document, async)

        ↓

ExtractedDocument

        ↓

ProcessedDocument

        ↓

Chunks  ({doc_id}-{chunk_index})

        ↓

Embeddings

        ↓

Pinecone upsert (namespace per tenant + tenant_id metadata filter)

        ↓

IngestResponse {documents_stored, chunks_stored}
```

---

## Question answering

```text
POST /tenants/{tenant_id}/ask

        ↓

AskRequest

        ↓

QAState

        ↓

RETRIEVE

        ↓

Hybrid Retrieval

        ↓

Hybrid candidates
        ↓

Reranker

        ↓

GRADE

        ↓
   ┌────┴────┐
   │         │
 relevant   none
   │         │
   ↓         ↓
ANSWER     REWRITE
             │
             ↓
          RETRIEVE

        ↓

AskResponse
```

---

# 30. Metadata Contract

Every chunk must carry:

```json
{
  "tenant_id": "tenant-123",
  "doc_id": "sha256...",
  "chunk_index": 0,
  "source": "https://example.com/page",
  "brand": "Example",
  "text": "..."
}
```

These fields support:

- tenant isolation
- deterministic IDs
- citations
- debugging
- traceability
- source attribution

---

# 31. Responsibility Boundaries

The following boundaries should remain strict:

```text
API
 └── HTTP only

Ingestion
 └── document acquisition/orchestration

Preprocessing
 └── source normalization

Chunking
 └── document segmentation

Embeddings
 └── vector generation

VectorStore
 └── Pinecone persistence/search

Retrieval
 └── candidate retrieval/ranking

Graph
 └── agent decision/orchestration

Tests
 └── behavior verification
```

No layer should bypass another layer unnecessarily.

---

# 32. Important Non-Goals

The first implementation should NOT include:

- LLM-based document chunking
- LLM-based source extraction
- arbitrary autonomous web browsing during QA
- unrestricted agent tool access
- multiple retry loops
- cross-tenant search
- storing entire documents in the LLM context
- sending unrelated chunks to the answer model
- hardcoded API keys
- duplicated ingestion logic in scripts

---

# 33. Implementation Sequence

Implement in this exact order:

```text
01. config.py
02. .env.example
03. requirements.txt
04. ingestion/models.py
05. ingestion/document_id.py
06. extraction/web_loader.py
07. preprocessing/cleaner.py
08. preprocessing/markdown_parser.py
09. preprocessing/validator.py
10. chunking/splitter.py
11. chunking/document_chunker.py
12. embeddings/service.py
13. vectorstore/pinecone.py
14. retrieval/bm25.py
15. retrieval/hybrid.py
16. retrieval/reranker.py
17. llm/service.py
18. ingestion/service.py
19. graph/state.py
20. graph/nodes.py
21. graph/edges.py
22. graph/graph.py
23. api/schemas.py
24. api/routes/health.py
25. api/routes/ingest.py
26. api/routes/ask.py
27. dependencies.py
28. main.py
29. tests/conftest.py
30. test_tenant_isolation.py
31. test_agent_decision.py
32. seed_corpus.py
33. Dockerfile
34. docker-compose.yml
35. README.md
```

---

# 34. Definition of Done

The backend is considered functionally complete when:

### Ingestion

- [ ] URL ingestion works.
- [ ] Raw text ingestion works.
- [ ] WebBaseLoader works.
- [ ] Extraction failures return controlled errors.
- [ ] Same document can be ingested repeatedly without duplicate chunks.
- [ ] Stable `doc_id` is generated.

### Chunking

- [ ] Markdown headings are preserved.
- [ ] Recursive character-based chunking works.
- [ ] Target is 400 characters.
- [ ] Overlap is 50 characters.
- [ ] Chunk sizing uses a character length function (no external tokenizer).
- [ ] Required metadata exists on every chunk.

### Retrieval

- [ ] Dense retrieval works.
- [ ] BM25 sparse retrieval works.
- [ ] Pinecone hybrid search (alpha-weighted) works.
- [ ] Reranking works.
- [ ] Final context is limited to approximately top 5 chunks.
- [ ] Tenant filter is applied during retrieval (namespace + metadata filter).

### Agent

- [ ] LangGraph is used.
- [ ] Retrieve node exists.
- [ ] Grade node exists.
- [ ] Rewrite/retry exists.
- [ ] Answer node exists.
- [ ] Retry is bounded.
- [ ] Answer is grounded in relevant context.
- [ ] `found_in_corpus` is correctly populated.
- [ ] Citations are returned.
- [ ] Trace is returned.

### API

- [ ] `/health` returns 200 and reports vector DB and LLM reachability.
- [ ] `/tenants/{tenant_id}/ingest` accepts a batch of documents and returns
      `documents_stored` / `chunks_stored`.
- [ ] `/tenants/{tenant_id}/ask` returns answer, `found_in_corpus`, citations, trace.
- [ ] Clear error responses for unreachable URL, tenant with no data, and LLM timeout.
- [ ] Async used for LLM calls and URL fetching.
- [ ] Not-in-corpus answers return 200 with `found_in_corpus=false`.

### Tests

- [ ] Tenant isolation test passes.
- [ ] Agent decision test passes.
- [ ] LLM calls are mocked.
- [ ] External services are not required for unit tests.

### Deployment

- [ ] Docker build succeeds.
- [ ] Local Docker environment works.
- [ ] Secrets are environment-based.
- [ ] Application is deployable to AWS.
