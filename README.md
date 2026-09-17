# Brand Q&A Agent

A multi-tenant RAG backend that ingests web pages / raw text about brands and answers
natural-language questions about them, citing its sources and refusing to invent answers
when the corpus does not cover the question.

Built for the Omnibound.ai Backend AI Engineer take-home.

- **Language:** Python 3.11+
- **Web framework:** FastAPI
- **Agent:** LangChain + LangGraph `StateGraph`
- **Vector DB:** Pinecone (serverless, hybrid dense + sparse)
- **Embeddings:** Ollama `qwen3-embedding:4b` (1024-d)
- **LLM:** Groq `openai/gpt-oss-120b` (OpenAI / OpenAI-compatible fallback)

---

## 1. What it does

1. **Ingest** — accepts web pages or raw text about brands, splits them into chunks, and
   upserts them into Pinecone under the tenant's namespace.
2. **Ask** — runs a LangGraph agent: retrieve candidate chunks, grade their relevance,
   answer from the relevant ones (or rewrite and retry once), and return an answer with
   citations and a readable trace.
3. **Health** — reports whether Pinecone, Ollama, and the LLM provider are reachable.

---

## 2. Quick start (one command)

### Option A — shell script

```bash
cp .env.example .env      # then fill in PINECONE_API_KEY and GROQ_API_KEY (or OPENAI_API_KEY)
scripts/run.sh
```

`scripts/run.sh` creates/activates the virtualenv, installs dependencies, ensures the NLTK
corpora, and starts FastAPI on `http://0.0.0.0:8000` (docs at `/docs`).
Override with env vars: `HOST=127.0.0.1 PORT=9000 INSTALL_DEPS=1 scripts/run.sh`.

### Option B — Docker

```bash
docker build -t brand-qa-agent:latest .

docker run --rm -p 8000:8000 \
  --env-file .env \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  brand-qa-agent:latest
```

Or, with the included compose file:

```bash
docker compose up --build
```

The image uses **CPU-only PyTorch** (avoids the ~2 GB CUDA wheels), pre-downloads the
cross-encoder reranker at build time, and runs as a non-root user.

### Prerequisites

- A Pinecone API key and a serverless index (created automatically on first use:
  dimension 1024, metric `dotproduct`).
- Ollama running with the embedding model: `ollama pull qwen3-embedding:4b`.
- A Groq API key (or OpenAI / OpenAI-compatible endpoint).

---

## 3. Configuration

All configuration is read from environment variables (`.env`); no keys are committed.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PINECONE_API_KEY` | — | Pinecone credentials |
| `PINECONE_INDEX` | `brand-chunks` | Index name |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` | Serverless spec |
| `PINECONE_ALPHA` | `0.5` | Hybrid dense weight (1.0 = dense only, 0.0 = sparse only) |
| `EMBEDDING_MODEL` | `qwen3-embedding:4b` | Ollama embedding model |
| `EMBEDDING_DIMENSION` | `1024` | Must match the Pinecone index |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `GROQ_API_KEY` / `GROQ_MODEL` | — / `openai/gpt-oss-120b` | Primary LLM |
| `OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_BASE_URL` | — | Optional fallback / compatible endpoint |
| `LLM_STRUCTURED_OUTPUT_METHOD` | `json_schema` | Structured-output mode for the grader |
| `TEMPERATURE` | `0.0` | Chat temperature |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `400` / `50` | Characters (see §6 Q4) |
| `HYBRID_TOP_K` / `FINAL_TOP_K` | `10` / `5` | Candidates before / after reranking |
| `RERANKER_MODEL` / `RERANKER_DEVICE` | `cross-encoder/ms-marco-MiniLM-L6-v2` / `cpu` | Reranker |
| `MIN_CONTENT_LENGTH` | `200` | Reject documents whose extracted text is shorter |
| `WEB_LOADER_TIMEOUT_MS` | `30000` | URL fetch timeout |

---

## 4. Architecture

### Request flow

```
                 POST /tenants/{tenant_id}/ask
                              |
                              v
                    LangGraph StateGraph
                              |
                    retrieve  (hybrid search, top 10)
                              |
                     grade    (LLM, structured yes/no)
                              |
              +---------------+----------------+
              |               |                |
         relevant         none, no         none, already
         chunks           retry yet        retried (max 1)
              |               |                |
              v               v                v
           answer         rewrite          not_in_corpus
              |           (retry+1)             |
              |               |                |
              |        back to retrieve        |
              v                                v
             END------------------------------END
```

Ingestion flow:

```
documents[] ──▶ extract (WebBaseLoader | raw text)
            ──▶ doc_id = sha256(url | text)
            ──▶ RecursiveCharacterTextSplitter (400/50)
            ──▶ attach metadata (tenant_id, doc_id, chunk_index, source, brand)
            ──▶ embed dense (Ollama) + sparse (BM25) 
            ──▶ upsert id="{doc_id}-{chunk_index}" into namespace={tenant_id}
```

### Project structure

```
app/
  main.py                     FastAPI app factory
  config.py                   Pydantic Settings (env-based)
  dependencies.py             cached store / ingestion / graph wiring
  api/
    schemas.py                Pydantic request/response models
    routes/{health,ingest,ask}.py
  ingestion/
    service.py                orchestrates extract -> chunk -> persist
    document_id.py            stable sha256 doc ids
    models.py                 ingest data models
  extraction/web_loader.py    URL fetch + text extraction
  chunking/
    splitter.py               RecursiveCharacterTextSplitter wrapper
    document_chunker.py       chunk + attach metadata
  embeddings/service.py       Ollama embeddings (L2-normalized)
  retrieval/
    bm25.py                   fit-free BM25 sparse vectorizer
    reranker.py               cross-encoder reranker
  vectorstore/pinecone.py     hybrid upsert / query, namespace + metadata filter
  llm/service.py              Groq/OpenAI chat models, grader/rewriter/generator
  graph/
    state.py                  QAState TypedDict (+ trace reducer)
    nodes.py                  retrieve / grade / rewrite / answer / not_in_corpus
    edges.py                  route_after_grade conditional edge
    graph.py                  StateGraph assembly
scripts/
  run.sh                      one-command start
  seed_corpus.py              seed a tenant via the production ingestion path
  show_pipeline.py            stage-by-stage ingestion demo
tests/                        pytest suite (LLM mocked)
```

---

## 5. API reference

### `POST /tenants/{tenant_id}/ingest`

```bash
curl -X POST http://localhost:8000/tenants/acme/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents":[
        {"url":"https://asana.com/pricing","brand":"Asana"},
        {"text":"Trello is a kanban tool ...","source":"manual-1","brand":"Trello"}
      ]}'
```

```json
{"documents_stored": 2, "chunks_stored": 37, "errors": []}
```

Each document has **either** `url` **or** `text` (not both), plus a required `brand` and an
optional `source`. Per-document failures are returned in `errors` and do not abort the
batch. Running the same document twice does not grow the chunk count (idempotent).

### `POST /tenants/{tenant_id}/ask`

```bash
curl -X POST http://localhost:8000/tenants/acme/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What do reviewers say about Asana pricing compared to Trello?"}'
```

```json
{
  "answer": "Reviewers generally describe Asana as ...",
  "found_in_corpus": true,
  "citations": [
    {"source": "https://asana.com/pricing", "brand": "Asana", "chunk_text": "..."}
  ],
  "trace": [
    {"node": "retrieve", "detail": "retrieved 10 chunk(s)"},
    {"node": "grade", "detail": "graded 10 chunk(s); 3 relevant"},
    {"node": "answer", "detail": "answered from 3 relevant chunk(s)"}
  ]
}
```

When nothing relevant is found, `found_in_corpus` is `false`, `citations` is empty, and the
trace shows the rewrite/retry path.

### `GET /health`

```json
{"status": "ok", "pinecone": "ok", "ollama": "ok", "llm": "ok"}
```

### Error responses

| Situation | Status | Body |
| --- | --- | --- |
| Unreachable / invalid URL | 200 (per-document) | `errors[]` entry with `code` = `invalid_url`, `fetch_failed`, or `empty_extraction` |
| Tenant with no data | 200 | `found_in_corpus: false`, `citations: []` (not an error) |
| Empty `question` / malformed body | 422 | FastAPI validation error |
| LLM not configured | 503 | `{"error": "llm_unconfigured", "message": "..."}` |
| LLM timeout / provider error | 502 | `{"error": "llm_unavailable", "message": "..."}` |

---

## 6. Answers to the assignment's questions

### Q1 — Which web framework did you choose, and why? (B.1)

**FastAPI.** It is async-first (the whole service is `async` — URL fetches and LLM calls are
awaited, not blocked), Pydantic-native (every request/response body is a Pydantic model, as
B.6 requires), and it validates input and generates OpenAPI docs at `/docs` for free.
Django/DRF would add an ORM and admin layer this service has no use for.

### Q2 — Which vector database, and what trade-off? (B.1, B.3)

**Pinecone (serverless).** The free tier is enough, and it gives us three things in one
place: per-tenant **namespaces**, a serverless index, and **native hybrid search** (dense +
sparse) in a single query. The trade-off is a managed-service dependency with no local
emulation; Chroma/Qdrant would run locally for free but would not provide the same
namespace/hybrid story out of the box.

### Q3 — Which LLM and embedding provider, and why? (B.1)

- **LLM: Groq (`openai/gpt-oss-120b`).** Extremely low latency and a generous free tier,
  which matters because the graph makes several LLM calls per question (grade, rewrite,
  answer). A `langchain-openai` path (with optional `OPENAI_BASE_URL`) is wired in as a
  fallback so the provider can be swapped without code changes.
- **Embeddings: local Ollama `qwen3-embedding:4b` (1024-d).** No per-token cost and no
  document-egress to a third party; the vectors are L2-normalized so dot-product == cosine
  similarity.

### Q4 — Why this chunk size and overlap? (B.4.3)

**400 characters with 50 characters of overlap**, via
`RecursiveCharacterTextSplitter`.

- `RecursiveCharacterTextSplitter` tries paragraph, then sentence, then word boundaries, so
  chunks stay semantically coherent instead of being cut mid-sentence.
- The 50-character overlap carries a little context across boundaries so a sentence split
  across two chunks is still retrievable from either side.
- Chunks are **character**-based (the built-in `len`) rather than token-based. This is a
  deliberate trade-off: it is deterministic, needs no tokenizer dependency, and is fast. The
  cost — chunks are not aligned to a specific model's tokenizer — is noted in
  [§9](#9-reflection-trade-offs-rough-edges-next-steps).

### Q5 — What metadata does each chunk carry? (B.4.4)

Every chunk stores exactly the contract metadata, plus the text payload:

`tenant_id`, `doc_id`, `chunk_index`, `source`, `brand` (+ `text` for the payload returned
in citations). Vector IDs are `{doc_id}-{chunk_index}`.

### Q6 — How is the graph structured? (B.5)

Implemented as a LangGraph `StateGraph` (not a single chain).

**State** (`QAState`, a `TypedDict`): `tenant_id`, `question`, `active_query`,
`retrieved_chunks`, `relevant_chunks`, `retry_count`, `answer`, `found_in_corpus`, and
`trace`. `trace` uses an `Annotated[list, operator.add]` reducer so each node appends one
entry instead of overwriting the list.

**Nodes:**

- `retrieve` — embeds `active_query`, runs the hybrid search scoped to the tenant, reranks,
  and stores the survivors in `retrieved_chunks`.
- `grade` — asks the LLM (structured output) which chunks actually help answer the question;
  the "yes" chunks go into `relevant_chunks`.
- `rewrite` — asks the LLM to rephrase the question, increments `retry_count`, then loops
  back to `retrieve`.
- `answer` — generates the answer **only** from `relevant_chunks` and sets
  `found_in_corpus = true`. Temperature is `0` and the prompt forbids outside knowledge;
  if the model judges the context insufficient it returns the `NOT_IN_CORPUS` sentinel,
  which the node maps to the deterministic not-in-corpus reply with
  `found_in_corpus = false` and empty citations.
- `not_in_corpus` — terminal node that returns an honest "the corpus does not cover this"
  answer with `found_in_corpus = false`.

**Conditional edge** (`route_after_grade`):

- `relevant_chunks` non-empty → `answer`
- empty and `retry_count == 0` → `rewrite` (retry at most once)
- empty and already retried → `not_in_corpus`

Every node appends one `{node, detail}` entry to `trace`, which is exactly what the `/ask`
response returns.

### Q7 — How is tenant isolation implemented? (B.3)

**Both** mechanisms the assignment allows, together:

1. One Pinecone **namespace per `tenant_id`** on upsert and query.
2. A `tenant_id` **metadata filter** (`{"tenant_id": {"$eq": tenant_id}}`) on every query.

`tenant_id` always comes from the URL path; it is never defaulted and never taken from the
request body. Namespacing alone would isolate data; the metadata filter is a second lock so
a mis-scoped namespace can still not leak another tenant's chunks. This is directly covered
by `tests/test_tenant_isolation.py`.

### Q8 — How is ingest idempotent? (B.2, B.4)

`doc_id` is a stable SHA-256 of the document's URL (for `url` documents) or of its text (for
`text` documents). Vector IDs are `{doc_id}-{chunk_index}` and the store uses Pinecone
**upsert**, so re-sending the same document overwrites the same vectors instead of adding
duplicates. (The service also de-duplicates repeated documents within a single batch.)

### Q9 — What are the required error behaviours? (B.6)

- **Unreachable URL** — reported per document in `errors[]` with a code
  (`invalid_url` / `fetch_failed` / `empty_extraction`); the rest of the batch still
  processes.
- **Tenant with no data** — HTTP 200 with `found_in_corpus: false` and empty citations
  (this is a valid answer, not an error — see Example 3).
- **LLM timeout / provider failure** — HTTP 502 `llm_unavailable`.
- **LLM not configured** — HTTP 503 `llm_unconfigured`.
- **Malformed request** — HTTP 422 with FastAPI's validation detail.

### Q10 — The two automated tests (B.6)

Both run with the LLM (and store) mocked, so no API key is needed:

1. **Tenant isolation** — `tests/test_tenant_isolation.py`: data ingested for tenant A is
   never returned when tenant B asks; also covers the empty-tenant "not in corpus" path, the
   503 on an unconfigured LLM, and the 502 on an LLM timeout.
2. **Decision logic** — `tests/test_agent_decision.py`: relevant chunks → answer without a
   rewrite; no relevant chunks → retry once; still nothing after the retry → not found.

Run them with:

```bash
venv/bin/python -m pytest -q
```

The full suite is **95 tests**; the two required ones are the files above.

### Q11 — How does the finished service behave on the three examples? (A.4)

- **Example 1 (normal question)** — `acme` asks about Asana vs Trello pricing; the service
  returns an answer grounded in retrieved chunks with citations and a `retrieve → grade →
  answer` trace.
- **Example 2 (off-topic question)** — "What is the capital of France?" finds nothing,
  rewrites once, finds nothing again, and returns `found_in_corpus: false` with an explicit
  "not in corpus" answer and empty citations. It does not say "Paris".
- **Example 3 (different tenant)** — `globex` (no ingested data) asks the Example 1 question
  and gets `found_in_corpus: false` with empty citations; `acme`'s data is never visible.

---

## 7. Tests

```bash
venv/bin/python -m pytest -q          # 95 tests, LLM mocked
```

Coverage includes ingestion (validation, per-document errors, idempotency), chunking and
metadata, embeddings, BM25 sparse vectorization, the Pinecone store (hybrid search, tenant
filtering, persistence), the reranker, the graph decision logic, and the API routes.

---

## 8. Reflection: trade-offs, rough edges, next steps (D.1.8)

**One trade-off.** I used Pinecone's **native hybrid search** (`hybrid_convex_scale`,
`alpha=0.5`) rather than running dense and BM25 as two separate retrievals and fusing them
with Reciprocal Rank Fusion (RRF). Native hybrid is one round trip, less code, and lets
Pinecone do the fusion — at the cost of direct control over (and inspectability of) the
fusion step. The sparse encoder itself is a dependency-free `Bm25SparseVectorizer` (stable
24-bit hash term ids) rather than a fitted `BM25Encoder`, so it needs no global vocabulary
and no NLTK download at runtime.

**One rough edge.** Chunking is character-based (400/50) rather than token-based, so chunk
boundaries do not correspond to any model's tokenizer. `found_in_corpus` has two gates — the
grader decides whether any chunk is relevant, and the generator can still decline with the
`NOT_IN_CORPUS` sentinel when it judges the context insufficient — but the grader can
occasionally mark a weak chunk as relevant, so a re-calibrated grader would tighten precision
further.

**One thing I would do with more time.** Add token-aware chunking (e.g. the Qwen tokenizer),
a stronger/re-calibrated grader, and true RRF over separate dense and sparse retrievers so
fusion is tunable and observable. Streaming the answer and adding per-tenant corpus manifests
for reproducible seeding would be next.
