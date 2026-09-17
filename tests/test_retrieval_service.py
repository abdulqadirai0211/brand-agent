import asyncio
from typing import Any

from app.retrieval.bm25_index import BM25Index
from app.retrieval.service import RetrievalService
from tests.fakes import make_chunk


class _FakeDenseStore:
    """ScoredChunk results per tenant, exactly like Pinecone's dense_search."""

    def __init__(self, results_by_tenant: dict[str, list[dict[str, Any]]]) -> None:
        self.results_by_tenant = results_by_tenant
        self.queries: list[tuple[str, str]] = []

    async def embed_query(self, query: str) -> list[float]:
        return [1.0]

    async def dense_search(
        self, vector: list[float], tenant_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        self.queries.append((tenant_id, vector[0]))
        return list(self.results_by_tenant.get(tenant_id, []))[:limit]


def _scored(chunk, score: float = 1.0) -> dict[str, Any]:
    return {
        "id": f"{chunk.metadata['doc_id']}-{chunk.metadata['chunk_index']}",
        "score": score,
        "payload": {**chunk.metadata, "text": chunk.page_content},
    }


class _RecordingReranker:
    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.calls: list[list[str]] = []

    def rerank(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.calls.append([candidate["id"] for candidate in candidates])
        return candidates[: self.top_k]


def test_retrieve_merges_dense_and_bm25_by_rrf_then_reranks() -> None:
    a = make_chunk("Asana workload gantt charts", "acme", "a")
    b = make_chunk("Asana gantt boards", "acme", "b")
    c = make_chunk("gantt overview guide", "acme", "c")

    dense = [_scored(a, 0.9), _scored(b, 0.7)]  # dense ranked: a, b
    bm25 = BM25Index()
    bm25.add_documents("acme", [b, c])  # lexical for "gantt boards": b, c

    reranker = _RecordingReranker(top_k=2)
    service = RetrievalService(
        _FakeDenseStore({"acme": dense}),
        bm25,
        reranker,
        dense_top_k=10,
        bm25_top_k=10,
        final_top_k=2,
    )

    results = asyncio.run(service.retrieve("acme", "gantt boards"))

    # RRF order: b (in both) > a (dense only, rank 1) > c (bm25 only)
    assert reranker.calls == [["b-0", "a-0", "c-0"]]
    assert [result["id"] for result in results] == ["b-0", "a-0"]


def test_retrieve_falls_back_to_dense_only_when_bm25_empty() -> None:
    a = make_chunk("Asana workload", "acme", "a")
    dense = [_scored(a)]
    reranker = _RecordingReranker()
    service = RetrievalService(
        _FakeDenseStore({"acme": dense}),
        BM25Index(),  # no tenant corpus at all
        reranker,
    )

    results = asyncio.run(service.retrieve("acme", "asana"))

    assert [result["id"] for result in results] == ["a-0"]
    assert reranker.calls == [["a-0"]]  # single-list fusion preserves order


def test_retrieve_is_tenant_scoped_on_both_branches() -> None:
    acme = make_chunk("Asana gantt workload", "acme", "a")
    beta = make_chunk("secret beta cloud doc", "beta", "s")

    bm25 = BM25Index()
    bm25.add_documents("acme", [acme])  # beta has no lexical corpus

    store = _FakeDenseStore({"acme": [_scored(acme)], "beta": [_scored(beta)]})
    reranker = _RecordingReranker()
    service = RetrievalService(store, bm25, reranker)

    acme_hits = asyncio.run(service.retrieve("acme", "asana gantt"))
    beta_hits = asyncio.run(service.retrieve("beta", "asana gantt"))

    assert [result["payload"]["tenant_id"] for result in acme_hits] == ["acme"]
    assert all(result["payload"]["tenant_id"] == "beta" for result in beta_hits)
    assert store.queries == [("acme", 1.0), ("beta", 1.0)]


def test_retrieve_service_add_documents_feeds_bm25_branch() -> None:
    a = make_chunk("Asana gantt boards", "acme", "a")
    dense = [_scored(a)]

    bm25 = BM25Index()
    reranker = _RecordingReranker()
    service = RetrievalService(_FakeDenseStore({"acme": dense}), bm25, reranker)
    service.add_documents("acme", [a])

    results = asyncio.run(service.retrieve("acme", "gantt"))

    assert [result["id"] for result in results] == ["a-0"]
    assert bm25.is_built("acme") is True