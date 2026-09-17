from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class QAState(TypedDict):
    """Shared state for the question-answering graph.

    `tenant_id` comes from the URL path and is never defaulted; every retrieval
    is scoped to it (namespace + metadata filter), which is what keeps tenants
    isolated across the whole graph.
    """

    tenant_id: str
    question: str
    active_query: str  # original question, or the rewritten query after a retry
    retrieved_chunks: list[Document]
    relevant_chunks: list[Document]
    retry_count: int
    answer: str
    found_in_corpus: bool
    trace: Annotated[list[dict], operator.add]  # one entry per node, appended