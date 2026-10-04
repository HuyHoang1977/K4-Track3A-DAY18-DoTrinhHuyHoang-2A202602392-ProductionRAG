from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize

    segmented = word_tokenize(text, format="text")
    # ⚠️ LƯU Ý: underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép").
    # BM25 tokenize bằng split(" ") → "nghỉ_phép" thành 1 token,
    # nhưng query "nghỉ phép" thành 2 token → KHÔNG khớp.
    # Phải replace("_", " ") để BM25 hoạt động đúng.
    return segmented.replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(chunk.get("text", "")).split() for chunk in chunks
        ]
        # BM25Okapi không nhận corpus rỗng (doc rỗng làm lỗi division by zero).
        self.corpus_tokens = [toks if toks else ["\u0000empty"] for toks in self.corpus_tokens]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []

        tokenized_query = segment_vietnamese(query).split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for i in ranked[:top_k]:
            # Lọc scores[i] > 0 để bỏ docs không liên quan.
            if scores[i] <= 0:
                continue
            chunk = self.documents[i]
            results.append(SearchResult(
                text=chunk.get("text", ""),
                score=float(scores[i]),
                metadata=chunk.get("metadata", {}),
                method="bm25",
            ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        if not chunks:
            return

        # 1. Collection 1024 chiều, cosine similarity (bge-m3).
        self.client.recreate_collection(
            collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        # 2. Encode toàn bộ chunks thành vector.
        texts = [c.get("text", "") for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True,
                                            normalize_embeddings=True)

        # 3. Upsert, text nằm trong payload để search trả về được nội dung.
        points = [
            PointStruct(id=i, vector=v.tolist(),
                        payload={**c.get("metadata", {}), "text": c.get("text", "")})
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        # ⚠️ LƯU Ý: qdrant-client >= 1.9 dùng query_points(), KHÔNG phải search().
        try:
            query_vector = self._get_encoder().encode(
                query, normalize_embeddings=True).tolist()
        except Exception:
            return []

        # Collection chưa tồn tại (chưa index) → không có kết quả.
        try:
            response = self.client.query_points(collection, query=query_vector, limit=top_k)
        except Exception:
            return []

        points = getattr(response, "points", response)  # tương thích mọi phiên bản
        return [
            SearchResult(
                text=(pt.payload or {}).get("text", ""),
                score=float(pt.score),
                metadata=dict(pt.payload or {}),
                method="dense",
            )
            for pt in points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}  # text → {"score": float, "result": SearchResult}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            entry = rrf_scores.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.values(), key=lambda e: e["score"], reverse=True)

    merged: list[SearchResult] = []
    for entry in ranked[:top_k]:
        original: SearchResult = entry["result"]
        merged.append(SearchResult(
            text=original.text,
            score=entry["score"],
            metadata=original.metadata,
            method="hybrid",
        ))
    return merged


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
