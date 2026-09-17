from app.retrieval.bm25_index import BM25Index, chunk_id, tokenize
from tests.fakes import make_chunk


def test_tokenize_matches_sparse_encoder_semantics() -> None:
    assert tokenize("Asana Workspace pricing!") == ["asana", "workspace", "pricing"]
    assert tokenize("the and a to i for asana") == ["asana"]
    assert tokenize("integration gantt calendar integration") == [
        "integration",
        "gantt",
        "calendar",
        "integration",
    ]
    assert tokenize("") == []


def test_chunk_id_matches_pinecone_vector_id_format() -> None:
    doc = make_chunk("text", "acme", "doc-xyz", 3)
    assert chunk_id(doc) == "doc-xyz-3"


def test_retrieve_returns_top_k_scoredchunk_shape() -> None:
    index = BM25Index()
    docs = [
        make_chunk("Asana workload and gantt reporting", "acme", "a"),
        make_chunk("Asana gantt boards overview", "acme", "b"),
        make_chunk("Trello cards move across columns", "acme", "c"),
    ]
    index.add_documents("acme", docs)

    results = index.retrieve("acme", "gantt", top_k=2)

    assert len(results) == 2
    for result in results:
        assert set(result.keys()) == {"id", "score", "payload"}
        assert "text" in result["payload"]


def test_retrieve_is_scoped_to_tenant_corpus() -> None:
    index = BM25Index()
    index.add_documents("acme", [make_chunk("Asana gantt workload", "acme", "a")])
    index.add_documents(
        "globex", [make_chunk("monday.com boards and workloads", "globex", "m")]
    )

    acme_hits = index.retrieve("acme", "asana gantt", top_k=5)
    other_hits = index.retrieve("globex", "asana gantt", top_k=5)

    assert [hit["id"] for hit in acme_hits] == ["a-0"]
    assert acme_hits[0]["payload"]["tenant_id"] == "acme"
    assert other_hits == []  # no lexical overlap AND a different tenant's corpus


def test_zero_score_matches_are_dropped() -> None:
    index = BM25Index()
    index.add_documents("acme", [make_chunk("monday.com boards", "acme", "m")])

    # "workload" shares no term with the corpus -> no zero-score noise returned
    assert index.retrieve("acme", "workload", top_k=5) == []


def test_unknown_tenant_returns_empty() -> None:
    index = BM25Index()
    index.add_documents("acme", [make_chunk("asana", "acme", "a")])
    assert index.retrieve("does-not-exist", "asana", top_k=5) == []


def test_lazy_rebuild_after_add() -> None:
    index = BM25Index()
    index.add_documents("acme", [make_chunk("asana boards", "acme", "a")])

    assert index.is_built("acme") is False
    hits = index.retrieve("acme", "asana")
    assert index.is_built("acme") is True
    assert [hit["id"] for hit in hits] == ["a-0"]