from __future__ import annotations

from typing import Literal

from app.graph.state import QAState


def route_after_grade(state: QAState) -> Literal["answer", "rewrite", "not_in_corpus"]:
    """Conditional edge after grading.

    - relevant chunks found -> answer
    - none found and no rewrite attempt yet -> rewrite and retrieve again
    - none found and already retried (max 1) -> not in corpus
    """
    if state.get("relevant_chunks"):
        return "answer"
    if (state.get("retry_count") or 0) < 1:
        return "rewrite"
    return "not_in_corpus"