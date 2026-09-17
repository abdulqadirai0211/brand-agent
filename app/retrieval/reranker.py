from __future__ import annotations

import os
from typing import Any, Callable

from app.config import Settings, get_settings


def _traceable() -> Callable:
    if os.environ.get("LANGSMITH_API_KEY"):
        from langsmith import traceable

        return traceable
    return lambda **kwargs: (lambda fn: fn)


class RerankingRetriever:
    """Reranks hybrid candidates with a cross-encoder: score (query, chunk)
    pairs together, sort descending, keep the top_k survivors.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        device: str = "cpu",
        top_k: int = 5,
        encoder: Any = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self._encoder = encoder

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import CrossEncoder

            self._encoder = CrossEncoder(self.model_name, device=self.device or None)
        return self._encoder

    @_traceable()(run_type="retriever", name="RerankingRetriever")
    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pairs = [(query, candidate["payload"]["text"]) for candidate in candidates]
        scores = self._get_encoder().predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [candidate for candidate, _ in ranked[: self.top_k]]

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RerankingRetriever:
        settings = settings or get_settings()
        return cls(
            model_name=settings.reranker_model,
            device=settings.reranker_device,
            top_k=settings.final_top_k,
        )