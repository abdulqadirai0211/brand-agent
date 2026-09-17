from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_core.documents import Document

from app.config import get_settings
from app.ingestion.service import IngestionService, get_ingestion_service
from app.llm.service import LLMUnconfiguredError
from app.retrieval.bm25_index import BM25Index
from app.retrieval.reranker import RerankingRetriever
from app.retrieval.service import RetrievalService
from app.vectorstore.pinecone import PineconeVectorStore, get_vector_store
from fastapi import HTTPException

_store: PineconeVectorStore | None = None
_bm25_index: BM25Index | None = None
_retrieval_service: RetrievalService | None = None
_ingestion: IngestionService | None = None
_ask_graph: Any | None = None


async def get_store() -> PineconeVectorStore:
    global _store
    if _store is None:
        _store = get_vector_store()
    return _store


async def _rebuild_bm25_from_store(store: PineconeVectorStore) -> BM25Index:
    """Rehydrate the per-tenant in-memory BM25 index from Pinecone at startup."""
    bm25 = BM25Index()
    try:
        index = store.ensure_index()
        stats = index.describe_index_stats()
    except Exception:
        return bm25  # store unavailable now; BM25 is populated at ingest time

    for tenant_id, namespace_info in stats.get("namespaces", {}).items():
        if not namespace_info.get("vector_count"):
            continue
        ids: list[str] = []
        try:
            for page in index.list(namespace=tenant_id):
                vectors = (
                    page.get("vectors", [])
                    if isinstance(page, dict)
                    else getattr(page, "vectors", [])
                )
                ids.extend(
                    vec["id"] if isinstance(vec, dict) else getattr(vec, "id")
                    for vec in vectors
                )
        except Exception:
            continue
        for offset in range(0, len(ids), 100):
            try:
                fetched = index.fetch(ids=ids[offset : offset + 100], namespace=tenant_id)
            except Exception:
                continue
            documents: list[Document] = []
            for metadata in (vector.get("metadata", {}) for vector in fetched.get("vectors", {}).values()):
                payload = dict(metadata)
                text = payload.pop("text", "")
                documents.append(Document(page_content=text, metadata=payload))
            bm25.add_documents(tenant_id, documents)
    return bm25


async def _ensure_bm25_index(store: PineconeVectorStore) -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = await _rebuild_bm25_from_store(store)
    return _bm25_index


async def get_retrieval_service() -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        store = await get_store()
        settings = get_settings()
        _retrieval_service = RetrievalService(
            store,
            await _ensure_bm25_index(store),
            RerankingRetriever.from_settings(),
            dense_top_k=settings.hybrid_top_k,
            bm25_top_k=settings.hybrid_top_k,
            final_top_k=settings.final_top_k,
        )
    return _retrieval_service


async def get_ingestion() -> IngestionService:
    global _ingestion
    if _ingestion is None:
        service = await get_ingestion_service()
        store = await get_store()
        bm25_index = await _ensure_bm25_index(store)

        async def persist_with_index(chunks: list[Document]) -> None:
            await store.persist_chunks(chunks)
            by_tenant: dict[str, list[Document]] = defaultdict(list)
            for chunk in chunks:
                by_tenant[chunk.metadata.get("tenant_id", "")].append(chunk)
            for tenant_id, tenant_chunks in by_tenant.items():
                bm25_index.add_documents(tenant_id, tenant_chunks)

        service.persist = persist_with_index
        _ingestion = service
    return _ingestion


async def get_ask_graph() -> Any:
    global _ask_graph
    if _ask_graph is None:
        from app.graph.graph import build_ask_graph

        retrieval_service = await get_retrieval_service()
        try:
            _ask_graph = build_ask_graph(retrieval_service)
        except LLMUnconfiguredError as exc:
            raise HTTPException(
                status_code=503,
                detail={"error": "llm_unconfigured", "message": str(exc)},
            ) from exc
    return _ask_graph