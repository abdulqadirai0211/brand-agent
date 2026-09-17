from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.documents import Document

from app.retrieval.bm25_index import BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import RerankingRetriever
from app.vectorstore.pinecone import PineconeVectorStore


class RetrievalService:
    """Orchestrates dense (Pinecone) + local BM25 retrieval, RRF fusion, rerank.

    ``retrieve`` is the single entry point the LangGraph ``retrieve`` node calls.
    Tenant isolation is enforced on both branches: dense goes through the
    tenant's namespace + metadata filter, and BM25 only scores the tenant's own
    corpus.
    """

    def __init__(
        self,
        store: PineconeVectorStore,
        bm25_index: BM25Index,
        reranker: RerankingRetriever,
        dense_top_k: int = 10,
        bm25_top_k: int = 10,
        final_top_k: int = 5,
    ) -> None:
        self._store = store
        self._bm25 = bm25_index
        self._reranker = reranker
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k

    async def retrieve(self, tenant_id: str, query: str) -> list[dict[str, Any]]:
        embedding = await self._store.embed_query(query)
        dense_results = await self._store.dense_search(
            embedding, tenant_id, limit=self.dense_top_k
        )
        bm25_results = self._bm25.retrieve(tenant_id, query, top_k=self.bm25_top_k)

        fused = reciprocal_rank_fusion([dense_results, bm25_results])
        fused = fused[: self.dense_top_k]
        if not fused:
            return []

        reranked = await asyncio.to_thread(self._reranker.rerank, query, fused)
        return reranked[: self.final_top_k]

    def add_documents(self, tenant_id: str, documents: list[Document]) -> None:
        self._bm25.add_documents(tenant_id, documents)