import asyncio

import pytest

from app.graph.graph import build_ask_graph
from app.graph.nodes import NOT_IN_CORPUS_ANSWER
from app.llm.service import NOT_IN_CORPUS_SENTINEL
from tests.fakes import FakeLLMs, FakeRetriever, FakeStore, build_test_graph, make_chunk

ACME_CHUNKS = [
    make_chunk("Asana offers list and board views for project management.", "acme", "doc-a"),
    make_chunk("Asana pricing starts with a free plan for individuals.", "acme", "doc-b", 1),
]


@pytest.fixture()
def llms() -> FakeLLMs:
    return FakeLLMs()


def _run(graph, question: str = "Does Asana have board views?") -> dict:
    return asyncio.run(
        graph.ainvoke(
            {
                "tenant_id": "acme",
                "question": question,
                "active_query": question,
                "retrieved_chunks": [],
                "relevant_chunks": [],
                "retry_count": 0,
                "answer": "",
                "found_in_corpus": False,
                "trace": [],
            }
        )
    )


def _trace_nodes(result: dict) -> list[str]:
    return [step["node"] for step in result["trace"]]


def test_relevant_on_first_pass_answers_without_rewrite() -> None:
    llms = FakeLLMs(relevant_by_call=[[0]])
    graph = build_test_graph(FakeStore({"acme": ACME_CHUNKS}), llms)

    result = _run(graph)

    assert result["answer"] == llms.answer
    assert result["found_in_corpus"] is True
    assert len(result["relevant_chunks"]) == 1
    assert result["relevant_chunks"][0].metadata["tenant_id"] == "acme"
    assert _trace_nodes(result) == ["retrieve", "grade", "answer"]
    assert llms.rewrite_calls == 0
    assert llms.generate_calls == 1
    assert llms.grade_calls == 1


def test_rewrites_once_then_answers() -> None:
    llms = FakeLLMs(relevant_by_call=[[], [0]])
    graph = build_test_graph(FakeStore({"acme": ACME_CHUNKS}), llms)

    result = _run(graph, question="What does Asana do?")

    assert result["found_in_corpus"] is True
    assert result["retry_count"] == 1
    assert _trace_nodes(result) == [
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "answer",
    ]
    assert llms.rewrite_calls == 1
    assert llms.grade_calls == 2
    assert llms.generate_calls == 1


def test_never_relevant_retries_at_most_once_then_not_in_corpus() -> None:
    llms = FakeLLMs(relevant_by_call=[[], []])
    graph = build_test_graph(FakeStore({"acme": ACME_CHUNKS}), llms)

    result = _run(graph, question="What does the yurt market look like?")

    assert result["found_in_corpus"] is False
    assert result["answer"] == NOT_IN_CORPUS_ANSWER
    assert result["relevant_chunks"] == []
    assert result["retry_count"] == 1
    assert _trace_nodes(result) == [
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "not_in_corpus",
    ]
    assert llms.rewrite_calls == 1  # exactly one mitigated retry, never two
    assert llms.grade_calls == 2


def test_generator_declining_context_marks_not_in_corpus() -> None:
    llms = FakeLLMs(relevant_by_call=[[0]], answer=NOT_IN_CORPUS_SENTINEL)
    graph = build_test_graph(FakeStore({"acme": ACME_CHUNKS}), llms)

    result = _run(graph)

    assert result["found_in_corpus"] is False
    assert result["answer"] == NOT_IN_CORPUS_ANSWER
    assert result["retry_count"] == 0  # decline happens at generation, not grading
    assert _trace_nodes(result) == ["retrieve", "grade", "answer"]
    assert llms.generate_calls == 1


def test_empty_corpus_short_circuits_grading() -> None:
    llms = FakeLLMs(relevant_by_call=[[0]])
    graph = build_test_graph(FakeStore({"acme": []}), llms)

    result = _run(graph, question="Anything here?")

    assert result["found_in_corpus"] is False
    assert llms.grade_calls == 0  # no LLM call with zero candidates
    assert llms.rewrite_calls == 1
    assert _trace_nodes(result)[0:2] == ["retrieve", "grade"]


def test_graph_build_accepts_injected_fakes_without_touching_real_llm() -> None:
    llms = FakeLLMs(relevant_by_call=[[0]])
    graph = build_ask_graph(
        FakeStore({"acme": ACME_CHUNKS}),
        FakeRetriever(),
        grader=llms.grader,
        rewriter=llms.rewriter,
        generator=llms.generator,
    )
    result = _run(graph)
    assert result["found_in_corpus"] is True
    assert llms.generate_calls == 1