# Retrieval Pipeline + Agent Orchestration

Drawn from the actual code. Use the end-to-end diagram as the visual, then walk top-to-bottom through retrieval, the graph, and the per-node behaviour.

## 1. End-to-end request flow

```
POST /tenants/{tenant_id}/ask                     app/api/routes/ask.py:15
body: { question }
   │
   │  initial QAState  (ask.py:21)
   │    tenant_id, question, active_query = question,
   │    retrieved_chunks = [], relevant_chunks = [], retry_count = 0,
   │    answer = "", found_in_corpus = False, trace = []
   ▼
┌──────────────────────── LangGraph (compiled)  graph.py:16 ──────────────────────┐
│                                                                                 │
│   START ──► retrieve ──► grade ──►(route_after_grade)  edges.py:8               │
│                             │            │                                      │
│                             │            ├── relevant ───────────► answer ──► END│
│                             │            ├── none  & retry < 1 ─► rewrite       │
│                             │            │                          │           │
│                             │            │                          └► retrieve │
│                             │            └── none  & retried ───► not_in_corpus─►END
│                             │                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
AskResponse { answer, found_in_corpus, citations[], trace[] }
```

## 2. Retrieval pipeline (inside the `retrieve` node)

```
retrieve node   nodes.py:32
  query = state["active_query"]        # original question, or the rewritten query after a retry
  │
  ├─ dense  : store.embed_query(query)  → Ollama qwen3-embedding:4b → 1024-d, L2-normalized
  ├─ sparse : to_thread(store.encode_query) → Bm25SparseVectorizer (TF, hashed 24-bit term ids)
  │
  ▼
  store.hybrid_search(vector, sparse, tenant_id, limit=hybrid_top_k=10)   pinecone.py:200
  │     _query():                                                        pinecone.py:219
  │        namespace   = tenant_id                                        ◄── ISOLATION
  │        filter      = { "tenant_id": { "$eq": tenant_id } }            ◄── ISOLATION (defense in depth)
  │        hybrid_convex_scale(vector, sparse, alpha=0.5)   ← weighted sum, NOT true RRF
  │        → top 10 candidates { id, score, payload(metadata + text) }
  ▼
  RerankingRetriever.rerank(query, candidates)                          reranker.py:42
  │     cross-encoder/ms-marco-MiniLM-L6-v2
  │     scores (query, chunk) pairs together → sort desc → keep top final_top_k=5
  ▼
  retrieved_chunks = [ Document(page_content=payload["text"], metadata=payload) ]
  trace += { "retrieve", "retrieved 5 chunk(s)" }
```

## 3. Graph topology and state

**Edges** (`graph.py:43-59`)

| From | To | Kind |
|---|---|---|
| `START` | `retrieve` | fixed |
| `retrieve` | `grade` | fixed |
| `grade` | `answer` / `rewrite` / `not_in_corpus` | conditional (`route_after_grade`, `edges.py:8`) |
| `rewrite` | `retrieve` | fixed (the retry loop) |
| `answer` | `END` | fixed |
| `not_in_corpus` | `END` | fixed |

**Routing rule** (`edges.py:15-19`)

```
if relevant_chunks            → "answer"
elif retry_count < 1          → "rewrite"      # max 1 retry
else                          → "not_in_corpus"
```

**Shared state** — `QAState` (`graph/state.py:9`)

| Field | Meaning |
|---|---|
| `tenant_id` | from the URL path only, never defaulted; scopes every retrieval |
| `question` | the original question (used by grade + rewrite) |
| `active_query` | original, or the rewritten query after a retry (used by retrieve) |
| `retrieved_chunks` | post-rerank candidates for the current attempt |
| `relevant_chunks` | subset the grader judged relevant |
| `retry_count` | number of rewrites so far (caps the loop at 1) |
| `answer` | final answer text |
| `found_in_corpus` | grounding flag driving the response |
| `trace` | `Annotated[list, operator.add]` — one entry appended per node |

## 4. Node-by-node

| Node | Reads | Does | Writes | Trace entry |
|---|---|---|---|---|
| `retrieve` (`nodes.py:32`) | `active_query`, `tenant_id` | dense + sparse query → hybrid top-10 → cross-encoder rerank → top-5 | `retrieved_chunks` | `retrieved N chunk(s)` |
| `grade` (`nodes.py:57`) | `question`, `retrieved_chunks` | if none → empty; else LLM structured verdict → relevant indices | `relevant_chunks` | `graded N chunk(s); M relevant` |
| `rewrite` (`nodes.py:76`) | `question`, `retry_count` | LLM reformulates a more searchable query; `retry_count += 1` | `active_query`, `retry_count` | `rewrote query (attempt N)` |
| `answer` (`nodes.py:85`) | `question`, `relevant_chunks` | join context → generator; if it returns `NOT_IN_CORPUS` → deterministic decline | `answer`, `found_in_corpus` | `answered from N relevant chunk(s)` **or** `generator declined: …` |
| `not_in_corpus` (`nodes.py:110`) | — | deterministic terminal reply | `answer`, `found_in_corpus=False` | `no relevant chunks after mitigated retrieval; not in corpus` |

**LLM boundary** (`app/llm/service.py`)

- Provider: **Groq primary** (`openai/gpt-oss-120b`), **OpenAI / OpenAI-compatible fallback** if `OPENAI_API_KEY` set (`get_chat_model`, `service.py:122`). Neither configured → `LLMUnconfiguredError`.
- **Grader** uses structured output: `GradeOutput { relevant_indices, explanation }` via `with_structured_output(..., method="json_schema")` (`service.py:148`). Out-of-range/duplicate indices are filtered and de-duped.
- **Rewriter** and **generator** are plain chat calls.
- **Grounding sentinel**: the generator returns `NOT_IN_CORPUS` when the context is insufficient; the `answer` node maps that to a fixed reply and sets `found_in_corpus=False` (`nodes.py:89`).
- All three prompts treat retrieved context as **data only** (prompt-injection guard) and the generator forbids outside knowledge.

## 5. Response assembly and citations

```
grounded = result["found_in_corpus"]
cited_chunks = relevant_chunks if grounded else []      # no citations when not grounded

AskResponse {
  answer,
  found_in_corpus,
  citations: [ { source, doc_id, chunk_index, brand, chunk_text } ]   # from chunk metadata
  trace:     [ { node, detail } ]                                     # ordered node log
}
```

## 6. Example traces (the three A.4 behaviours)

```
1) Answerable (acme / Asana workload)     retrieve → grade(3 relevant) → answer           found_in_corpus = true
2) Not in corpus (acme / capital of France) retrieve → grade(0) → rewrite → retrieve
                                            → grade(0) → not_in_corpus                     found_in_corpus = false
3) Generator declines (context judged insufficient)
                                          retrieve → grade(n) → answer(declined)           found_in_corpus = false
```

## 7. Error mapping (`ask.py:34-43`)

| Condition | HTTP | `detail.error` |
|---|---|---|
| no provider configured | `503` | `llm_unconfigured` |
| provider fails at request time (incl. timeout) | `502` | `llm_unavailable` |
| empty `question` | `422` | FastAPI validation |
| tenant with no data | `200` | `found_in_corpus: false` (empty corpus is not an error) |

## 8. Talking points for the video

- **Bounded self-correction loop** — one grade, at most one rewrite, then a deterministic terminal branch. The `retry_count` cap makes latency and cost predictable and prevents infinite retrieval.
- **Retrieval is rerank-then-grade** — a cheap cross-encoder narrows 10 hybrid candidates to 5 before the (more expensive) LLM grader ever sees them.
- **Two independent relevance gates** — the LLM **grader** decides whether chunks are relevant at all; the **generator sentinel** is a second gate that can still decline if the kept context is insufficient. Both must pass for `found_in_corpus = true`.
- **Hybrid, then convex combination** — dense (semantic) and sparse (lexical BM25) are merged with `hybrid_convex_scale(alpha=0.5)`, a weighted sum — deliberately not true RRF.
- **Isolation survives the whole graph** — `tenant_id` is injected once from the path and carried in state; every `retrieve` re-applies namespace + metadata filter, so no node can widen the search space.
- **Grounding is enforced twice at the edges** — citations are emitted only when `found_in_corpus` is true, and the decline path returns a fixed message instead of improvised text.
- **Testable by construction** — nodes are built around injected `grader`/`rewriter`/`generator`/`retriever` (`graph.py:16`), so the mocked decision-logic tests never touch a real LLM.
