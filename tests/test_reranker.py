from app.config import Settings
from app.retrieval.reranker import RerankingRetriever


class FakeEncoder:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def predict(self, pairs: list, **kwargs) -> list[float]:
        return self._scores[: len(pairs)]


def make_candidates(n: int = 3) -> list[dict]:
    return [
        {
            "id": f"c-{i}",
            "score": 0.2 * (n - i),
            "payload": {"text": f"chunk number {i}", "tenant_id": "acme"},
        }
        for i in range(n)
    ]


def test_reranker_sorts_by_score_and_keeps_top_k():
    encoder = FakeEncoder([0.3, 0.9, 0.6])
    reranker = RerankingRetriever(top_k=2, encoder=encoder)

    out = reranker.rerank("gantt timeline", make_candidates())

    assert [c["id"] for c in out] == ["c-1", "c-2"]


def test_reranker_scores_each_query_chunk_pair():
    encoder = FakeEncoder([0.8, 0.1])
    reranker = RerankingRetriever(top_k=5, encoder=encoder)

    out = reranker.rerank("q", make_candidates(2))

    assert encoder._scores[:2] == [0.8, 0.1]
    assert len(out) == 2


def test_reranker_returns_all_when_fewer_than_top_k():
    encoder = FakeEncoder([0.2, 0.7, 0.4])
    reranker = RerankingRetriever(top_k=10, encoder=encoder)

    out = reranker.rerank("q", make_candidates(3))

    assert len(out) == 3


def test_reranker_empty_candidates_returns_empty():
    reranker = RerankingRetriever(top_k=5, encoder=FakeEncoder([]))

    assert reranker.rerank("q", []) == []


def test_reranker_stable_order_on_ties():
    encoder = FakeEncoder([0.5, 0.5, 0.5])
    reranker = RerankingRetriever(top_k=3, encoder=encoder)

    out = reranker.rerank("q", make_candidates())

    assert [c["id"] for c in out] == ["c-0", "c-1", "c-2"]


def test_from_settings_reads_model_and_final_top_k(settings: Settings):
    settings.reranker_model = "cross-encoder/ms-marco-MiniLM-L6-v2"
    settings.final_top_k = 4

    reranker = RerankingRetriever.from_settings(settings)

    assert reranker.model_name == settings.reranker_model
    assert reranker.device == settings.reranker_device
    assert reranker.top_k == 4