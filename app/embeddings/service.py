from __future__ import annotations

import asyncio
import math
from typing import Protocol

from app.config import get_settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]


class EmbeddingService:
    """Dense embeddings via Ollama (qwen3-embedding:4b), reused across calls.

    `embed_documents` and `embed_query` run the (sync) LangChain client in a
    thread so the ingest pipeline stays async. Embeddings are L2-normalized so
    cosine distance is directly comparable.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder
        self._client: Embedder | None = None

    def _client_impl(self) -> Embedder:
        if self._client is None:
            from langchain_ollama import OllamaEmbeddings

            settings = get_settings()
            self._client = OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.ollama_base_url,
                dimensions=settings.embedding_dimension,
            )
        return self._client

    def _get(self) -> Embedder:
        if self._embedder is not None:
            return self._embedder
        return self._client_impl()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = await asyncio.to_thread(self._get().embed_documents, texts)
        return [l2_normalize(v) for v in vectors]

    async def embed_query(self, query: str) -> list[float]:
        vector = await asyncio.to_thread(self._get().embed_query, query)
        return l2_normalize(vector)


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()