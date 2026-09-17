from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from langchain_core.documents import Document

from app.config import get_settings
from app.llm.service import NOT_IN_CORPUS_SENTINEL, Generator, Grader, Rewriter
from app.retrieval.reranker import RerankingRetriever
from app.vectorstore.pinecone import PineconeVectorStore, ScoredChunk

Node = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

#: A deterministic message for the "not in corpus" terminal branch.
NOT_IN_CORPUS_ANSWER = (
    "I could not find information about that in the corpus for this tenant."
)


def create_nodes(
    store: PineconeVectorStore,
    retriever: RerankingRetriever,
    grader: Grader,
    rewriter: Rewriter,
    generator: Generator,
) -> dict[str, Node]:
    """Build the graph's node set around injected dependencies (testable)."""

    settings = get_settings()

    async def retrieve(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("active_query") or state["question"]
        vector = await store.embed_query(query)
        sparse = await asyncio.to_thread(store.encode_query, query)
        candidates: list[ScoredChunk] = await store.hybrid_search(
            vector,
            sparse,
            state["tenant_id"],
            limit=settings.hybrid_top_k,
        )
        ranked = (
            await asyncio.to_thread(retriever.rerank, query, candidates)
            if candidates
            else []
        )
        docs = [
            Document(page_content=item["payload"].get("text", ""), metadata=item["payload"])
            for item in ranked
        ]
        return {
            "active_query": query,
            "retrieved_chunks": docs,
            "trace": [{"node": "retrieve", "detail": f"retrieved {len(docs)} chunk(s)"}],
        }

    async def grade(state: dict[str, Any]) -> dict[str, Any]:
        chunks = state.get("retrieved_chunks") or []
        if not chunks:
            return {
                "relevant_chunks": [],
                "trace": [{"node": "grade", "detail": "no retrieved chunks to grade"}],
            }
        indices = await grader(state["question"], chunks)
        relevant = [chunks[i] for i in indices]
        return {
            "relevant_chunks": relevant,
            "trace": [
                {
                    "node": "grade",
                    "detail": f"graded {len(chunks)} chunk(s); {len(relevant)} relevant",
                }
            ],
        }

    async def rewrite(state: dict[str, Any]) -> dict[str, Any]:
        attempt = (state.get("retry_count") or 0) + 1
        improved = await rewriter(state["question"])
        return {
            "active_query": improved,
            "retry_count": attempt,
            "trace": [{"node": "rewrite", "detail": f"rewrote query (attempt {attempt})"}],
        }

    async def answer(state: dict[str, Any]) -> dict[str, Any]:
        relevant = state.get("relevant_chunks") or []
        context = "\n\n".join(chunk.page_content for chunk in relevant)
        text = (await generator(state["question"], context)).strip()
        if text.strip("`\"' ").upper() == NOT_IN_CORPUS_SENTINEL:
            return {
                "answer": NOT_IN_CORPUS_ANSWER,
                "found_in_corpus": False,
                "trace": [
                    {
                        "node": "answer",
                        "detail": (
                            f"generator declined: {len(relevant)} chunk(s) judged insufficient"
                        ),
                    }
                ],
            }
        return {
            "answer": text,
            "found_in_corpus": True,
            "trace": [
                {"node": "answer", "detail": f"answered from {len(relevant)} relevant chunk(s)"}
            ],
        }

    async def not_in_corpus(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": NOT_IN_CORPUS_ANSWER,
            "found_in_corpus": False,
            "trace": [
                {
                    "node": "not_in_corpus",
                    "detail": "no relevant chunks after mitigated retrieval; not in corpus",
                }
            ],
        }

    return {
        "retrieve": retrieve,
        "grade": grade,
        "rewrite": rewrite,
        "answer": answer,
        "not_in_corpus": not_in_corpus,
    }