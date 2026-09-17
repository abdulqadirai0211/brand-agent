from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from app.graph.graph import build_ask_graph


def make_chunk(
    text: str,
    tenant_id: str,
    doc_id: str,
    chunk_index: int = 0,
    brand: str = "Acme",
    source: str = "https://example.com/acme",
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "tenant_id": tenant_id,
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "source": source,
            "brand": brand,
        },
    )


class FakeStore:
    """In-memory store recording every tenant-scoped query it receives.

    ``hybrid_search`` only ever returns chunks belonging to the requested
    tenant, mirroring the namespace + metadata-filter isolation the real store
    performs against Pinecone.
    """

    def __init__(self, chunks_by_tenant: dict[str, list[Document]]) -> None:
        self.chunks_by_tenant = chunks_by_tenant
        self.queries: list[dict[str, Any]] = []

    async def embed_query(self, query: str) -> list[float]:
        return [1.0] * 4

    def encode_query(self, text: str) -> dict[str, list]:
        return {"indices": [0], "values": [1.0]}

    async def hybrid_search(
        self,
        query_vector: list[float],
        sparse_vector: dict[str, list],
        tenant_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        self.queries.append({"tenant_id": tenant_id, "query": None})
        chunks = self.chunks_by_tenant.get(tenant_id, [])
        return [
            {
                "id": chunk.metadata["doc_id"],
                "score": 1.0,
                "payload": {**chunk.metadata, "text": chunk.page_content},
            }
            for chunk in chunks[:limit]
        ]


class FakeRetriever:
    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return candidates[: self.top_k]


class FakeLLMs:
    """Deterministic scripted stand-ins for grader / rewriter / generator."""

    def __init__(
        self,
        relevant_by_call: list[list[int]] | None = None,
        *,
        rewritten: str = "improved search query",
        answer: str = "Here is the grounded answer.",
    ) -> None:
        self.relevant_by_call = relevant_by_call or [[]]
        self.rewritten = rewritten
        self.answer = answer
        self.grade_calls = 0
        self.rewrite_calls = 0
        self.generate_calls = 0

    async def grader(self, query: str, chunks: list[Document]) -> list[int]:
        index = self.grade_calls
        self.grade_calls += 1
        if index < len(self.relevant_by_call):
            return list(self.relevant_by_call[index])
        return []

    async def rewriter(self, query: str) -> str:
        self.rewrite_calls += 1
        return self.rewritten

    async def generator(self, query: str, context: str) -> str:
        self.generate_calls += 1
        return self.answer


def build_test_graph(
    store: FakeStore,
    llms: FakeLLMs,
    retriever: FakeRetriever | None = None,
) -> Any:
    return build_ask_graph(
        store,
        retriever or FakeRetriever(),
        grader=llms.grader,
        rewriter=llms.rewriter,
        generator=llms.generator,
    )