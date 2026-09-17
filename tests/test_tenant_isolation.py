import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_ask_graph
from app.graph.nodes import NOT_IN_CORPUS_ANSWER
from app.llm.service import NOT_IN_CORPUS_SENTINEL
from app.main import app
from app.vectorstore.pinecone import PineconeVectorStore
from tests.fakes import FakeLLMs, FakeRetrievalService, build_test_graph, make_chunk

ACME_CHUNKS = [
    make_chunk("Asana offers list and board project views.", "acme", "asana-1", brand="Asana"),
    make_chunk("Asana pricing starts with a free plan.", "acme", "asana-2", 1, brand="Asana"),
]
BETA_CHUNKS = [
    make_chunk("Trello is a board-based tool from Atlassian.", "beta", "trello-1", brand="Trello"),
    make_chunk("Trello cards move left to right across columns.", "beta", "trello-2", 1, brand="Trello"),
]


class _FakeSparseEncoder:
    def encode_documents(self, texts: list[str]) -> list[dict[str, list]]:
        return [{"indices": [0], "values": [1.0]} for _ in texts]

    def encode_queries(self, text: str) -> dict[str, list]:
        return {"indices": [0], "values": [1.0]}


class _FakeIndex:
    def __init__(self) -> None:
        self.queries: list[dict[str, Any]] = []

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.queries.append(kwargs)
        return {"matches": []}


def test_hybrid_search_scopes_query_to_tenant() -> None:
    index = _FakeIndex()
    store = PineconeVectorStore(
        client=object(),
        index=index,
        sparse_encoder=_FakeSparseEncoder(),
        alpha=0.5,
    )

    asyncio.run(
        store.hybrid_search(
            [0.1, 0.2, 0.3, 0.4],
            {"indices": [0, 1], "values": [0.5, 0.5]},
            "acme",
            limit=7,
        )
    )

    kwargs = index.queries[0]
    assert kwargs["namespace"] == "acme"
    assert kwargs["filter"] == {"tenant_id": {"$eq": "acme"}}
    assert kwargs["top_k"] == 7
    assert "vector" in kwargs and "sparse_vector" in kwargs


def test_dense_search_scopes_query_to_tenant() -> None:
    index = _FakeIndex()
    store = PineconeVectorStore(
        client=object(),
        index=index,
        sparse_encoder=_FakeSparseEncoder(),
        alpha=0.5,
    )

    asyncio.run(store.dense_search([0.1, 0.2, 0.3, 0.4], "acme", limit=7))

    kwargs = index.queries[0]
    assert kwargs["namespace"] == "acme"
    assert kwargs["filter"] == {"tenant_id": {"$eq": "acme"}}
    assert kwargs["top_k"] == 7
    assert "vector" in kwargs and "sparse_vector" not in kwargs


def test_graph_only_retrieves_scoped_tenant_and_never_leaks() -> None:
    retrieval = FakeRetrievalService({"acme": ACME_CHUNKS, "beta": BETA_CHUNKS})
    llms = FakeLLMs(relevant_by_call=[[0], [1]])
    graph = build_test_graph(retrieval, llms)

    acme = asyncio.run(
        graph.ainvoke(
            {
                "tenant_id": "acme",
                "question": "Does Asana have boards?",
                "active_query": "Does Asana have boards?",
                "retrieved_chunks": [],
                "relevant_chunks": [],
                "retry_count": 0,
                "answer": "",
                "found_in_corpus": False,
                "trace": [],
            }
        )
    )
    beta = asyncio.run(
        graph.ainvoke(
            {
                "tenant_id": "beta",
                "question": "What is Trello?",
                "active_query": "What is Trello?",
                "retrieved_chunks": [],
                "relevant_chunks": [],
                "retry_count": 0,
                "answer": "",
                "found_in_corpus": False,
                "trace": [],
            }
        )
    )

    assert [q["tenant_id"] for q in retrieval.queries] == ["acme", "beta"]
    assert {c.metadata["tenant_id"] for c in acme["relevant_chunks"]} == {"acme"}
    assert {c.metadata["tenant_id"] for c in beta["relevant_chunks"]} == {"beta"}
    assert acme["relevant_chunks"][0].metadata["brand"] == "Asana"
    assert beta["relevant_chunks"][0].metadata["brand"] == "Trello"


@pytest.fixture()
def client() -> None:
    retrieval = FakeRetrievalService({"acme": ACME_CHUNKS, "beta": BETA_CHUNKS})
    llms = FakeLLMs(relevant_by_call=[[0], [0]])
    graph = build_test_graph(retrieval, llms)
    app.dependency_overrides[get_ask_graph] = lambda: graph
    with TestClient(app) as test_client:
        test_client.retrieval = retrieval  # type: ignore[attr-defined]
        yield test_client
    app.dependency_overrides.pop(get_ask_graph, None)


def test_ask_route_uses_path_tenant_and_returns_citations(client: TestClient) -> None:
    acme = client.post("/tenants/acme/ask", json={"question": "Does Asana have boards?"})
    assert acme.status_code == 200
    body = acme.json()
    assert body["found_in_corpus"] is True
    assert body["answer"] != ""
    assert len(body["citations"]) == 1
    assert body["citations"][0]["brand"] == "Asana"
    assert [step["node"] for step in body["trace"]] == ["retrieve", "grade", "answer"]

    beta = client.post("/tenants/beta/ask", json={"question": "What is Trello?"})
    assert beta.status_code == 200
    assert beta.json()["citations"][0]["brand"] == "Trello"

    assert [q["tenant_id"] for q in client.retrieval.queries] == ["acme", "beta"]  # type: ignore[attr-defined]


def test_ask_rejects_empty_question(client: TestClient) -> None:
    response = client.post("/tenants/acme/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_empty_tenant_returns_not_in_corpus(client: TestClient) -> None:
    response = client.post("/tenants/globex/ask", json={"question": "Asana pricing?"})
    assert response.status_code == 200
    body = response.json()
    assert body["found_in_corpus"] is False
    assert body["citations"] == []
    assert body["answer"] != ""


def test_ask_citations_empty_when_generator_declines() -> None:
    retrieval = FakeRetrievalService({"acme": ACME_CHUNKS})
    llms = FakeLLMs(relevant_by_call=[[0]], answer=NOT_IN_CORPUS_SENTINEL)
    graph = build_test_graph(retrieval, llms)
    app.dependency_overrides[get_ask_graph] = lambda: graph
    try:
        with TestClient(app) as test_client:
            response = test_client.post(
                "/tenants/acme/ask", json={"question": "Does Asana have boards?"}
            )
    finally:
        app.dependency_overrides.pop(get_ask_graph, None)

    assert response.status_code == 200
    body = response.json()
    assert body["found_in_corpus"] is False
    assert body["citations"] == []
    assert body["answer"] == NOT_IN_CORPUS_ANSWER


def test_ask_502_on_llm_timeout() -> None:
    import httpx
    from groq import APITimeoutError

    class _TimingOutGraph:
        async def ainvoke(self, state: dict) -> dict:
            raise APITimeoutError(request=httpx.Request("POST", "https://api.groq.com"))

    app.dependency_overrides[get_ask_graph] = lambda: _TimingOutGraph()
    try:
        with TestClient(app) as test_client:
            response = test_client.post("/tenants/acme/ask", json={"question": "hi"})
    finally:
        app.dependency_overrides.pop(get_ask_graph, None)

    assert response.status_code == 502
    assert response.json()["detail"]["error"] == "llm_unavailable"


def test_ask_503_when_no_llm_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.graph.graph as graph_module
    from app.llm.service import LLMUnconfiguredError

    def _unconfigured() -> None:
        raise LLMUnconfiguredError(
            "No LLM configured: set GROQ_API_KEY (or OPENAI_API_KEY and optionally "
            "OPENAI_BASE_URL for an OpenAI-compatible endpoint) in .env"
        )

    monkeypatch.setattr(graph_module, "get_chat_model", _unconfigured)
    with TestClient(app) as test_client:
        response = test_client.post("/tenants/acme/ask", json={"question": "hi"})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error"] == "llm_unconfigured"
    assert "GROQ_API_KEY" in detail["message"]