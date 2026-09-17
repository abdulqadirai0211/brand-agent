from __future__ import annotations

from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
    id_key: str = "id",
) -> list[dict[str, Any]]:
    """Fuse multiple ranked lists by aggregate reciprocal rank.

    ``RRF(item) = sum over lists of 1 / (k + rank(item))``. Rank-based fusion is
    robust to the incomparable score scales of dense (cosine-normed) and BM25
    (unbounded) similarity. Items are deduplicated by ``id_key``; the first list
    in which an item appears wins its payload. Returns items in descending
    fused score.
    """
    scores: defaultdict[str, float] = defaultdict(float)
    documents: dict[str, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item[id_key]
            scores[chunk_id] += 1.0 / (k + rank)
            if chunk_id not in documents:
                documents[chunk_id] = item

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [documents[chunk_id] for chunk_id, _ in ranked]