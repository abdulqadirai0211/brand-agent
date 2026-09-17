from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.graph.edges import route_after_grade
from app.graph.nodes import create_nodes
from app.graph.state import QAState
from app.llm.service import build_generator, build_grader, build_rewriter, get_chat_model
from app.retrieval.reranker import RerankingRetriever
from app.vectorstore.pinecone import PineconeVectorStore


def build_ask_graph(
    store: PineconeVectorStore,
    retriever: RerankingRetriever | None = None,
    *,
    grader: Any | None = None,
    rewriter: Any | None = None,
    generator: Any | None = None,
) -> Any:
    """Assemble the compiled ask graph.

    ``retrieve -> grade -> (answer | rewrite -> retrieve -> grade -> answer | not_in_corpus)``
    """
    retriever = retriever or RerankingRetriever.from_settings()
    needs_chat = grader is None or rewriter is None or generator is None
    if needs_chat:
        chat = get_chat_model()  # raises LLMUnconfiguredError; caught by the route
    else:
        chat = None
    nodes = create_nodes(
        store,
        retriever,
        grader=grader
        or build_grader(chat, method=get_settings().llm_structured_output_method),
        rewriter=rewriter or build_rewriter(chat),
        generator=generator or build_generator(chat),
    )

    workflow = StateGraph(QAState)
    for name, node in nodes.items():
        workflow.add_node(name, node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "answer": "answer",
            "rewrite": "rewrite",
            "not_in_corpus": "not_in_corpus",
        },
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("answer", END)
    workflow.add_edge("not_in_corpus", END)
    return workflow.compile()