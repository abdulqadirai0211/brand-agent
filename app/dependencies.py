from typing import Any

from app.ingestion.service import IngestionService, get_ingestion_service
from app.llm.service import LLMUnconfiguredError
from app.retrieval.reranker import RerankingRetriever
from app.vectorstore.pinecone import PineconeVectorStore, get_vector_store
from fastapi import HTTPException

_store: PineconeVectorStore | None = None
_ingestion: IngestionService | None = None
_ask_graph: Any | None = None


async def get_store() -> PineconeVectorStore:
    global _store
    if _store is None:
        _store = get_vector_store()
    return _store


async def get_ingestion() -> IngestionService:
    global _ingestion
    if _ingestion is None:
        service = await get_ingestion_service()
        store = await get_store()
        service.persist = store.persist_chunks
        _ingestion = service
    return _ingestion


async def get_ask_graph() -> Any:
    global _ask_graph
    if _ask_graph is None:
        from app.graph.graph import build_ask_graph

        store = await get_store()
        try:
            _ask_graph = build_ask_graph(store, RerankingRetriever.from_settings())
        except LLMUnconfiguredError as exc:
            raise HTTPException(
                status_code=503,
                detail={"error": "llm_unconfigured", "message": str(exc)},
            ) from exc
    return _ask_graph