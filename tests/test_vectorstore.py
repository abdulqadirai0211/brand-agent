import asyncio
import hashlib
import math

from langchain_core.documents import Document

from app.embeddings.service import EmbeddingService, l2_normalize
from app.retrieval.bm25 import Bm25SparseVectorizer
from app.vectorstore.pinecone import PineconeVectorStore

DIM = 1024


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeEmbedder:
    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, query):
        return self._vec(query)

    @staticmethod
    def _vec(text):
        return [(int(hashlib.sha256(f"{text}:{i}".encode()).hexdigest()[0], 16) / 15.0) for i in range(DIM)]


class FakeIndex:
    """In-memory Pinecone index: records keyed by vector id within a namespace."""

    def __init__(self, name):
        self.name = name
        self._records = {}  # namespace -> {vector_id: record}

    def upsert(self, vectors, namespace, batch_size=None):
        self._records.setdefault(namespace, {})
        for v in vectors:
            self._records[namespace][v["id"]] = v

    def delete(self, ids, namespace):
        if namespace in self._records:
            for vid in ids:
                self._records[namespace].pop(vid, None)

    def list(self, prefix, namespace):
        recs = sorted(vid for vid in self._records.get(namespace, {}) if vid.startswith(prefix))
        yield {"vectors": [{"id": vid} for vid in recs]}

    def query(self, *, namespace, top_k, vector=None, sparse_vector=None, include_metadata=True, filter=None):
        recs = self._records.get(namespace, {})
        scored = []
        for vid, rec in recs.items():
            if filter and rec["metadata"].get("tenant_id") != filter.get("tenant_id", {}).get("$eq"):
                continue
            score = 0.0
            if vector is not None:
                score += sum(a * b for a, b in zip(vector, rec["values"]))
            if sparse_vector is not None:
                sparse = dict(zip(rec["sparse_values"]["indices"], rec["sparse_values"]["values"]))
                score += sum(sparse.get(i, 0.0) * v for i, v in zip(sparse_vector["indices"], sparse_vector["values"]))
            scored.append({"id": vid, "score": score, "metadata": rec["metadata"]})
        scored.sort(key=lambda m: m["score"], reverse=True)
        return {"matches": scored[:top_k]}

    def describe_index_stats(self):
        namespaces = {ns: {"vector_count": len(recs)} for ns, recs in self._records.items()}
        return {"namespaces": namespaces, "total_vector_count": sum(ns["vector_count"] for ns in namespaces.values())}


class FakeClient:
    def __init__(self):
        self._indexes = {}
        self.created = []

    def has_index(self, name):
        return name in self._indexes

    def create_index(self, name, dimension, metric, spec):
        self.created.append({"name": name, "dimension": dimension, "metric": metric, "spec": spec})
        self._indexes[name] = FakeIndex(name)

    def describe_index(self, name):
        class _Status:
            status = {"ready": True}

        return _Status()

    def Index(self, name):
        return self._indexes[name]

    def delete_index(self, name):
        self._indexes.pop(name, None)


def chunk(doc_id, index, tenant_id="acme", brand="Acme", source="manual-x", text="some content words"):
    return Document(
        page_content=text,
        metadata={
            "tenant_id": tenant_id,
            "doc_id": doc_id,
            "chunk_index": index,
            "source": source,
            "brand": brand,
        },
    )


def make_store(**overrides):
    client = FakeClient()
    store = PineconeVectorStore(
        client=client,
        index_name="brand-chunks",
        dimension=DIM,
        embedder=EmbeddingService(embedder=FakeEmbedder()),
        sparse_encoder=Bm25SparseVectorizer(),
        alpha=0.5,
        **overrides,
    )
    return client, store


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------


def test_embed_documents_normalized():
    service = EmbeddingService(embedder=FakeEmbedder())
    vectors = asyncio.run(service.embed_documents(["asana workspace", "monday pricing"]))

    assert len(vectors) == 2
    assert all(len(v) == DIM for v in vectors)
    for v in vectors:
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-6)


def test_embed_query_normalized():
    service = EmbeddingService(embedder=FakeEmbedder())
    vector = asyncio.run(service.embed_query("pricing"))

    assert len(vector) == DIM
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-6)


def test_embed_documents_empty():
    service = EmbeddingService(embedder=FakeEmbedder())
    assert asyncio.run(service.embed_documents([])) == []


def test_l2_normalize_zero_vector_unchanged():
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


# ---------------------------------------------------------------------------
# bm25 sparse vectorizer
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_filters():
    v = Bm25SparseVectorizer()
    assert v.tokenize("Asana Workspace pricing!") == ["asana", "workspace", "pricing"]


def test_tokenize_drops_stopwords_and_single_chars():
    v = Bm25SparseVectorizer()
    assert v.tokenize("the and a to i for asana") == ["asana"]


def test_vectorize_deterministic():
    v1 = Bm25SparseVectorizer().vectorize("integration gantt calendar integration")
    v2 = Bm25SparseVectorizer().vectorize("integration gantt calendar integration")

    assert v1["indices"] == v2["indices"]
    assert v1["values"] == v2["values"]
    # term frequency: integration appears twice
    values = dict(zip(v1["indices"], v1["values"]))
    assert set(values.values()) == {1.0, 2.0}


def test_vectorize_repeated_term_matches_query():
    vectorizer = Bm25SparseVectorizer()
    doc = vectorizer.vectorize("asana gantt")
    query = vectorizer.vectorize("asana")
    assert set(query["indices"]).issubset(set(doc["indices"]))


def test_vectorizer_exposes_encode_interface():
    v = Bm25SparseVectorizer()
    docs = v.encode_documents(["asana gantt", "trello boards"])
    q = v.encode_queries("asana")
    assert len(docs) == 2
    assert set(q["indices"]).issubset(set(docs[0]["indices"]))


# ---------------------------------------------------------------------------
# vectorstore: index lifecycle
# ---------------------------------------------------------------------------


def test_ensure_index_creates_serverless_dotproduct_index():
    client, store = make_store()

    index = store.ensure_index()

    assert index.name == "brand-chunks"
    assert client.created[0]["dimension"] == DIM
    assert client.created[0]["metric"] == "dotproduct"
    assert client.created[0]["spec"].cloud == "aws"


def test_ensure_index_reuses_existing_index():
    client, store = make_store()
    store.ensure_index()
    store.ensure_index()

    assert len(client.created) == 1


def test_injected_index_is_used_directly():
    index = FakeIndex("premade")
    client = FakeClient()

    store = PineconeVectorStore(client=client, index=index)

    assert store.ensure_index() is index
    assert client.created == []


# ---------------------------------------------------------------------------
# vectorstore: put
# ---------------------------------------------------------------------------


def test_persist_chunks_writes_payload_contract():
    client, store = make_store()
    index = store.ensure_index()
    c = chunk("d1", 0)

    n = asyncio.run(store.persist_chunks([c]))

    assert n == 1
    record = index._records["acme"]["d1-0"]
    assert record["metadata"]["tenant_id"] == "acme"
    assert record["metadata"]["doc_id"] == "d1"
    assert record["metadata"]["chunk_index"] == 0
    assert record["metadata"]["source"] == "manual-x"
    assert record["metadata"]["brand"] == "Acme"
    assert record["metadata"]["text"] == "some content words"
    assert len(record["values"]) == DIM
    assert record["id"] == "d1-0"


def test_persist_same_chunks_idempotent():
    client, store = make_store()
    store.ensure_index()
    c = chunk("d1", 0)

    assert asyncio.run(store.persist_chunks([c])) == 1
    assert asyncio.run(store.persist_chunks([c])) == 1

    assert store.count("acme") == 1


def test_persist_multiple_docs_reuses_vector_ids():
    client, store = make_store()
    store.ensure_index()
    chunks = [chunk("d1", 0), chunk("d1", 1), chunk("d2", 0)]

    n = asyncio.run(store.persist_chunks(chunks))

    assert n == 3
    assert store.count("acme") == 3


def test_delete_document_removes_only_that_doc():
    client, store = make_store()
    store.ensure_index()
    asyncio.run(store.persist_chunks([chunk("d1", 0), chunk("d1", 1), chunk("d2", 0)]))

    store.delete_document("acme", "d1")

    assert store.count("acme") == 1
    assert store.count() == 1


def test_delete_document_tenant_scoped():
    client, store = make_store()
    store.ensure_index()
    asyncio.run(store.persist_chunks([chunk("d1", 0), chunk("d2", 0, tenant_id="beta")]))

    store.delete_document("acme", "d1")

    assert store.count("acme") == 0
    assert store.count("beta") == 1


def test_count_by_tenant():
    _, store = make_store()
    store.ensure_index()
    asyncio.run(store.persist_chunks([chunk("d1", 0), chunk("d1", 0, tenant_id="beta")]))

    assert store.count("acme") == 1
    assert store.count() == 2


# ---------------------------------------------------------------------------
# vectorstore: search + tenant isolation
# ---------------------------------------------------------------------------


def test_dense_search_round_trip():
    client, store = make_store()
    store.ensure_index()
    asyncio.run(store.persist_chunks([chunk("d1", 0, text="asana gantt pricing")]))

    vector = asyncio.run(store.embed_query("asana gantt pricing"))
    results = asyncio.run(store.dense_search(vector, "acme", limit=5))

    assert len(results) == 1
    assert results[0]["id"] == "d1-0"
    assert results[0]["score"] > 0.0
    assert results[0]["payload"]["doc_id"] == "d1"


def test_sparse_search_round_trip():
    client, store = make_store()
    store.ensure_index()
    asyncio.run(store.persist_chunks([chunk("d1", 0, text="gantt calendar integration")]))

    query_vector = store.encode_query("gantt calendar integration")
    results = asyncio.run(store.sparse_search(query_vector, "acme", limit=5))

    assert len(results) == 1
    assert results[0]["payload"]["doc_id"] == "d1"
    assert results[0]["score"] > 0.0


def test_hybrid_search_round_trip():
    client, store = make_store()
    store.ensure_index()
    asyncio.run(store.persist_chunks([chunk("d1", 0, text="gantt calendar integration")]))

    dense = asyncio.run(store.embed_query("gantt calendar integration"))
    sparse = store.encode_query("gantt calendar integration")
    results = asyncio.run(store.hybrid_search(dense, sparse, "acme", limit=5))

    assert len(results) == 1
    assert results[0]["payload"]["doc_id"] == "d1"


def test_hybrid_search_alpha_scales_query():
    client, store = make_store()
    store.ensure_index()
    asyncio.run(store.persist_chunks([chunk("d1", 0, text="gantt calendar integration")]))

    dense = asyncio.run(store.embed_query("gantt calendar integration"))
    sparse = store.encode_query("gantt calendar integration")

    pure_dense = asyncio.run(store.hybrid_search(dense, sparse, "acme", limit=5, alpha=1.0))
    pure_sparse = asyncio.run(store.hybrid_search(dense, sparse, "acme", limit=5, alpha=0.0))
    half = asyncio.run(store.hybrid_search(dense, sparse, "acme", limit=5, alpha=0.5))

    # In this deterministic corpus the identical-text dense dot is 1.0 while the
    # sparse dot over 3 overlapping terms is 3.0, so alpha=0.0 (sparse only) must
    # score higher than alpha=1.0 (dense only) and alpha=0.5 must sit between.
    assert pure_dense[0]["id"] == "d1-0"
    assert pure_sparse[0]["score"] > pure_dense[0]["score"]
    assert pure_dense[0]["score"] < half[0]["score"] < pure_sparse[0]["score"]


def test_search_isolated_by_tenant_via_namespace_and_filter():
    client, store = make_store()
    store.ensure_index()
    acme = chunk("a1", 0, tenant_id="acme", text="same identical content")
    beta = chunk("b1", 0, tenant_id="beta", text="same identical content")
    asyncio.run(store.persist_chunks([acme, beta]))

    vector = asyncio.run(store.embed_query("same identical content"))

    acme_results = asyncio.run(store.dense_search(vector, "acme", limit=5))
    beta_results = asyncio.run(store.dense_search(vector, "beta", limit=5))

    assert {r["payload"]["tenant_id"] for r in acme_results} == {"acme"}
    assert {r["payload"]["tenant_id"] for r in beta_results} == {"beta"}


def test_sparse_search_isolated_by_tenant():
    client, store = make_store()
    store.ensure_index()
    acme = chunk("a1", 0, tenant_id="acme", text="identical overlapping terms")
    beta = chunk("b1", 0, tenant_id="beta", text="identical overlapping terms")
    asyncio.run(store.persist_chunks([acme, beta]))

    query_vector = store.encode_query("identical overlapping terms")
    results = asyncio.run(store.sparse_search(query_vector, "acme", limit=5))

    assert all(r["payload"]["tenant_id"] == "acme" for r in results)


def test_tenant_filter_blocks_leakage_even_within_namespace():
    client, store = make_store()
    store.ensure_index()
    asyncio.run(store.persist_chunks([chunk("d1", 0, text="overlapping terms")]))

    # Simulate a query that lands in acme's namespace but a record whose
    # metadata tenant_id does not match; the metadata filter must drop it.
    index = client.Index("brand-chunks")
    index._records["acme"]["evil-0"] = {
        "id": "evil-0",
        "values": [1.0] * DIM,
        "sparse_values": {"indices": [], "values": []},
        "metadata": {"tenant_id": "evil"},
    }

    vector = asyncio.run(store.embed_query("overlapping terms"))
    results = asyncio.run(store.dense_search(vector, "acme", limit=5))

    assert all(r["payload"]["tenant_id"] == "acme" for r in results)


def test_get_vector_store_defaults():
    from app.vectorstore.pinecone import get_vector_store

    store = get_vector_store()
    assert store._index_name == "brand-chunks"
    assert store._alpha == 0.5


# ---------------------------------------------------------------------------
# vectorstore: default sparse encoder (fit-free BM25)
# ---------------------------------------------------------------------------


def test_default_sparse_encoder_is_fit_free_and_queryable():
    client = FakeClient()
    store = PineconeVectorStore(
        client=client,
        index_name="brand-chunks",
        dimension=DIM,
        embedder=EmbeddingService(embedder=FakeEmbedder()),
    )

    encoder = store._encoder()
    assert store._is_fitted(encoder) is True

    sparse = store.encode_query("asana gantt pricing")  # must not raise
    assert sparse["indices"]
    assert len(sparse["indices"]) == len(sparse["values"])


def test_default_encoder_document_and_query_vectors_align():
    client = FakeClient()
    store = PineconeVectorStore(
        client=client,
        index_name="brand-chunks",
        dimension=DIM,
        embedder=EmbeddingService(embedder=FakeEmbedder()),
    )
    store.ensure_index()
    asyncio.run(store.persist_chunks([chunk("d1", 0, text="gantt calendar integration")]))

    dense = asyncio.run(store.embed_query("gantt calendar integration"))
    sparse = store.encode_query("gantt calendar integration")
    results = asyncio.run(store.hybrid_search(dense, sparse, "acme", limit=5))

    assert results and results[0]["payload"]["doc_id"] == "d1"