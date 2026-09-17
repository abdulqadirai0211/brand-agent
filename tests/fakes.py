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


class FakeRetrievalService:
    """In-memory retrieval stand-in recording every tenant-scoped query it receives.

    ``retrieve`` only ever returns chunks belonging to the requested tenant,
    mirroring the isolation the real service performs (Pinecone namespace +
    metadata filter on the dense branch, a per-tenant BM25 corpus on the lexical
    branch).
    """

    def __init__(self, chunks_by_tenant: dict[str, list[Document]], top_k: int = 5) -> None:
        self.chunks_by_tenant = chunks_by_tenant
        self.top_k = top_k
        self.queries: list[dict[str, Any]] = []

    async def retrieve(self, tenant_id: str, query: str) -> list[dict[str, Any]]:
        self.queries.append({"tenant_id": tenant_id, "query": query})
        chunks = self.chunks_by_tenant.get(tenant_id, [])
        return [
            {
                "id": f"{chunk.metadata['doc_id']}-{chunk.metadata['chunk_index']}",
                "score": 1.0,
                "payload": {**chunk.metadata, "text": chunk.page_content},
            }
            for chunk in chunks[: self.top_k]
        ]


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
    retrieval_service: FakeRetrievalService,
    llms: FakeLLMs,
) -> Any:
    return build_ask_graph(
        retrieval_service,
        grader=llms.grader,
        rewriter=llms.rewriter,
        generator=llms.generator,
    )