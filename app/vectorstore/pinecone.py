from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from app.config import Settings, get_settings
from app.embeddings.service import EmbeddingService, get_embedding_service

SparseVector = dict[str, list]  # {"indices": [...], "values": [...]}

ScoredChunk = dict[str, Any]


class SparseEncoder(Protocol):
    """Any object exposing a pinecone-text style sparse interface."""

    def encode_documents(self, texts: list[str]) -> list[SparseVector]: ...

    def encode_queries(self, text: str) -> SparseVector: ...


class PineconeVectorStore:
    """Hybrid (dense + BM25 sparse) store on a single Pinecone serverless index.

    Tenant isolation follows the PDF's B.3 "do both" rule: every record lives
    under the tenant's *namespace* AND carries a ``tenant_id`` metadata field
    that is applied as a `filter` on every query.
    """

    def __init__(
        self,
        *,
        client: Pinecone | None = None,
        index: Any | None = None,
        index_name: str | None = None,
        dimension: int | None = None,
        embedder: EmbeddingService | None = None,
        sparse_encoder: SparseEncoder | None = None,
        alpha: float | None = None,
        cloud: str | None = None,
        region: str | None = None,
    ) -> None:
        settings = get_settings()
        self._client = client
        self._index_name = index_name or settings.pinecone_index
        self._dimension = dimension or settings.embedding_dimension
        self._embedder = embedder or get_embedding_service()
        self._alpha = settings.pinecone_alpha if alpha is None else alpha
        self._cloud = cloud or settings.pinecone_cloud
        self._region = region or settings.pinecone_region
        self._sparse_encoder = sparse_encoder
        # Resolved lazily in ensure_index(); an injected `index` is used as-is.
        self._index = index

    # ------------------------------------------------------------------
    # index lifecycle
    # ------------------------------------------------------------------

    def ensure_index(self) -> Any:
        if self._index is not None:
            return self._index

        if self._client is None:
            settings = get_settings()
            self._client = Pinecone(api_key=settings.pinecone_api_key or None)

        index_name = self._index_name
        if not self._client.has_index(index_name):
            self._client.create_index(
                name=index_name,
                dimension=self._dimension,
                metric="dotproduct",  # only dotproduct accepts sparse vectors
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )
            while not self._client.describe_index(index_name).status["ready"]:
                time.sleep(1)

        self._index = self._client.Index(index_name)
        return self._index

    def delete_index(self) -> None:
        if self._client is None or not self._client.has_index(self._index_name):
            return
        self._client.delete_index(self._index_name)
        self._index = None

    # ------------------------------------------------------------------
    # sparse encoder (fitted lazily on each persist batch)
    # ------------------------------------------------------------------

    def _encoder(self) -> SparseEncoder:
        if self._sparse_encoder is None:
            # Fit-free, deterministic sparse encoder: document and query term
            # ids always align, so no persisted corpus vocabulary is required.
            from app.retrieval.bm25 import Bm25SparseVectorizer

            self._sparse_encoder = Bm25SparseVectorizer()
        return self._sparse_encoder

    @staticmethod
    def _is_fitted(encoder: SparseEncoder) -> bool:
        # Encoders without a `fit` step (e.g. Bm25SparseVectorizer) are always
        # ready; a pinecone-text BM25Encoder only is once it has seen a corpus.
        if not hasattr(encoder, "fit"):
            return True
        return getattr(encoder, "doc_freq", None) is not None

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------

    async def persist_chunks(self, chunks: list[Document]) -> int:
        if not chunks:
            return 0
        index = await asyncio.to_thread(self.ensure_index)

        texts = [chunk.page_content for chunk in chunks]
        dense_vectors: list[list[float]] = await self._embedder.embed_documents(texts)
        sparse_vectors: list[SparseVector] = await asyncio.to_thread(
            self._fit_and_encode, texts
        )

        grouped: dict[str, list[dict[str, Any]]] = {}
        for chunk, dense_vector, sparse_vector in zip(chunks, dense_vectors, sparse_vectors):
            metadata = dict(chunk.metadata or {})
            tenant_id = metadata["tenant_id"]
            doc_id = metadata["doc_id"]
            chunk_index = metadata["chunk_index"]
            grouped.setdefault(tenant_id, []).append(
                {
                    "id": f"{doc_id}-{chunk_index}",
                    "values": dense_vector,
                    "sparse_values": sparse_vector,
                    "metadata": {**metadata, "text": chunk.page_content},
                }
            )

        for tenant_id, records in grouped.items():
            index.upsert(vectors=records, namespace=tenant_id, batch_size=100)
        return len(chunks)

    def _fit_and_encode(self, texts: list[str]) -> list[SparseVector]:
        encoder = self._encoder()
        if hasattr(encoder, "fit"):
            encoder.fit(texts)  # type: ignore[attr-defined]
        return encoder.encode_documents(texts)

    def _encode_query_sparse(self, text: str) -> SparseVector:
        encoder = self._encoder()
        if not self._is_fitted(encoder):
            # An injected fit-based encoder with no corpus yet: degrade to
            # dense-only rather than raising on the query path.
            return {"indices": [], "values": []}
        return encoder.encode_queries(text)

    def delete_document(self, tenant_id: str, doc_id: str) -> None:
        index = self.ensure_index()
        listing = index.list(prefix=f"{doc_id}-", namespace=tenant_id)
        ids = self._list_ids(listing)
        if ids:
            index.delete(ids=ids, namespace=tenant_id)

    @staticmethod
    def _list_ids(listing: Any) -> list[str]:
        # index.list() yields pages that may be dicts or SDK model objects.
        ids: list[str] = []
        for page in listing:
            vectors = page["vectors"] if hasattr(page, "get") or isinstance(page, dict) else getattr(page, "vectors")
            for vector in vectors:
                if isinstance(vector, dict):
                    ids.append(vector["id"])
                else:
                    ids.append(getattr(vector, "id"))
        return ids

    def count(self, tenant_id: str | None = None) -> int:
        stats = self.ensure_index().describe_index_stats()
        if tenant_id is not None:
            return int(stats["namespaces"].get(tenant_id, {}).get("vector_count", 0))
        return int(stats["total_vector_count"])

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    async def dense_search(self, query_vector: list[float], tenant_id: str, limit: int = 10) -> list[ScoredChunk]:
        return await asyncio.to_thread(
            self._query, vector=query_vector, tenant_id=tenant_id, limit=limit
        )

    async def sparse_search(self, query_vector: SparseVector, tenant_id: str, limit: int = 10) -> list[ScoredChunk]:
        return await asyncio.to_thread(
            self._query, sparse_vector=query_vector, tenant_id=tenant_id, limit=limit
        )

    async def hybrid_search(
        self,
        query_vector: list[float],
        sparse_vector: SparseVector,
        tenant_id: str,
        limit: int = 10,
        alpha: float | None = None,
    ) -> list[ScoredChunk]:
        alpha = self._alpha if alpha is None else alpha
        has_sparse = bool(sparse_vector.get("values"))
        return await asyncio.to_thread(
            self._query,
            vector=query_vector,
            sparse_vector=sparse_vector if has_sparse else None,
            tenant_id=tenant_id,
            limit=limit,
            alpha=alpha if has_sparse else None,
        )

    def _query(
        self,
        *,
        tenant_id: str,
        limit: int,
        vector: list[float] | None = None,
        sparse_vector: SparseVector | None = None,
        alpha: float | None = None,
    ) -> list[ScoredChunk]:
        index = self.ensure_index()
        kwargs: dict[str, Any] = {
            "namespace": tenant_id,
            "top_k": limit,
            "include_metadata": True,
            "filter": self._tenant_filter(tenant_id),
        }
        if alpha is not None:
            from pinecone_text.hybrid import hybrid_convex_scale

            vector, sparse_vector = hybrid_convex_scale(vector, sparse_vector, alpha)  # type: ignore[arg-type]
        if vector is not None:
            kwargs["vector"] = vector
        if sparse_vector is not None:
            kwargs["sparse_vector"] = sparse_vector

        response = index.query(**kwargs)
        return [
            {
                "id": match["id"],
                "score": match["score"],
                "payload": dict(match.get("metadata") or {}),
            }
            for match in response["matches"]
        ]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    async def embed_query(self, text: str) -> list[float]:
        return await self._embedder.embed_query(text)

    def encode_query(self, text: str) -> SparseVector:
        return self._encode_query_sparse(text)

    @staticmethod
    def _tenant_filter(tenant_id: str) -> dict[str, dict[str, str]]:
        return {"tenant_id": {"$eq": tenant_id}}


def get_vector_store(settings: Settings | None = None) -> PineconeVectorStore:
    settings = settings or get_settings()
    return PineconeVectorStore(
        index_name=settings.pinecone_index,
        dimension=settings.embedding_dimension,
        alpha=settings.pinecone_alpha,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )