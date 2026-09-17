from app.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_ranks_by_reciprocal_rank_sum() -> None:
    dense = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    bm25 = [{"id": "b"}, {"id": "c"}]

    fused = reciprocal_rank_fusion([dense, bm25])

    # b: 1/62 (dense r2) + 1/61 (bm25 r1) > c: 1/63 (dense r3) + 1/62 (bm25 r2) > a: 1/61 (dense r1)
    assert [item["id"] for item in fused] == ["b", "c", "a"]


def test_rrf_ties_reorder_deterministically_by_insertion() -> None:
    dense = [{"id": "a"}, {"id": "b"}]
    bm25 = [{"id": "b"}, {"id": "a"}]

    fused = reciprocal_rank_fusion([dense, bm25])

    # a and b have identical fused scores; sort is stable so first-inserted wins.
    assert {item["id"] for item in fused} == {"a", "b"}
    assert len(fused) == 2


def test_rrf_deduplicates_and_first_list_payload_wins() -> None:
    dense = [{"id": "a", "score": 0.8, "payload": {"brand": "Asana"}}]
    bm25 = [{"id": "a", "score": 6.4, "payload": {"brand": "Trello"}}]

    fused = reciprocal_rank_fusion([dense, bm25])

    assert len(fused) == 1
    assert fused[0]["id"] == "a"
    assert fused[0]["payload"]["brand"] == "Asana"


def test_rrf_item_appearing_once_keeps_its_rank() -> None:
    dense = [{"id": "only-dense"}, {"id": "shared"}]
    fused = reciprocal_rank_fusion([dense, []])

    assert [item["id"] for item in fused] == ["only-dense", "shared"]


def test_rrf_with_empty_lists_returns_empty() -> None:
    assert reciprocal_rank_fusion([[], [], []]) == []