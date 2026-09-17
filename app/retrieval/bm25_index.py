from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.retrieval.bm25 import _STOPWORDS, _TOKEN_RE


def tokenize(text: str) -> list[str]:
    """Consistent lexical tokenization (lowercase alnum, no stopwords / singles)."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def chunk_id(doc: Document) -> str:
    """Canonical chunk identity, matching the Pinecone vector id."""
    return f"{doc.metadata['doc_id']}-{doc.metadata['chunk_index']}"


class BM25Index:
    """Per-tenant in-memory BM25 index over the same chunks stored in Pinecone.

    Isolation mirrors the vector store: documents are grouped per tenant and a
    tenant's query only ever scores its own corpus. Indices are rebuilt lazily
    after each ingest batch and kept in memory (`rank_bm25` has no incremental
    update), which is the pragmatic trade-off for this take-home.
    """

    def __init__(self) -> None:
        self._corpora: dict[str, list[Document]] = {}
        self._indices: dict[str, BM25Okapi | None] = {}
        self._dirty: set[str] = set()

    def add_documents(self, tenant_id: str, documents: list[Document]) -> None:
        if not documents:
            return
        self._corpora.setdefault(tenant_id, []).extend(documents)
        self._dirty.add(tenant_id)

    def _rebuild(self, tenant_id: str) -> None:
        corpus = self._corpora.get(tenant_id, [])
        tokenized = [tokenize(doc.page_content) for doc in corpus]
        # rank_bm25 divides by average document length; a corpus with no
        # tokenizable content must degrade to "no lexical index" rather than raise.
        if corpus and any(tokenized):
            self._indices[tenant_id] = BM25Okapi(tokenized)
        else:
            self._indices[tenant_id] = None
        self._dirty.discard(tenant_id)

    def retrieve(
        self, tenant_id: str, query: str, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Top-k tenant-scoped BM25 results in ScoredChunk shape (id/score/payload)."""
        if tenant_id in self._dirty:
            self._rebuild(tenant_id)
        index = self._indices.get(tenant_id)
        if index is None:
            return []
        corpus = self._corpora[tenant_id]
        scores = index.get_scores(tokenize(query))
        ranked = sorted(zip(corpus, scores), key=lambda pair: pair[1], reverse=True)
        # rank_bm25's Lucene-style idf goes negative for terms present in every
        # corpus document, so matching docs legitimately score < 0. Only docs
        # with *no* query-term overlap score exactly 0.0; drop only those so the
        # BM25 branch contributes nothing when the query shares no terms.
        ranked = [(doc, score) for doc, score in ranked if score != 0]
        return [
            {
                "id": chunk_id(doc),
                "score": float(score),
                "payload": {**doc.metadata, "text": doc.page_content},
            }
            for doc, score in ranked[:top_k]
        ]

    def tenant_ids(self) -> list[str]:
        return list(self._corpora.keys())

    def is_built(self, tenant_id: str) -> bool:
        return self._indices.get(tenant_id) is not None