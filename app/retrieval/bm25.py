from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "he", "her", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "of", "on", "or", "our", "she", "so", "than", "that", "the",
    "their", "them", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "will", "with", "you", "your",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

SparseVector = dict[str, list]


@dataclass
class Bm25SparseVectorizer:
    """Deterministic per-term feature ids so document and query vectors align.

    ``term -> 24-bit feature id`` via a stable hash; stored values are the raw
    term frequencies. No global vocabulary or hidden state is required: the same
    term always maps to the same id. Exposes the same ``indices``/``values``
    shape pinecone-text uses, so it is a drop-in sparse encoder for the hybrid
    store (and a dependency-free stand-in for tests).
    """

    def tokenize(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    def vectorize(self, text: str) -> SparseVector:
        counts = Counter(self.tokenize(text))
        indices: list[int] = []
        values: list[float] = []
        for token, count in sorted(counts.items()):
            indices.append(self._feature_id(token))
            values.append(float(count))
        return {"indices": indices, "values": values}

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self.vectorize(text) for text in texts]

    def encode_queries(self, text: str) -> SparseVector:
        return self.vectorize(text)

    @staticmethod
    def _feature_id(token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=3).digest()
        return int.from_bytes(digest, "big")