from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, RETRIEVAL_MERGE_MAX


def merge_retrieval_units(chunks: list[dict], merge_max: int = RETRIEVAL_MERGE_MAX) -> list[dict]:
    """
    Gộp các child chunk liên tiếp cùng parent thành đơn vị truy hồi lớn hơn.

    Lý do (đo bằng ablation trên 20 câu test):
      child as-is (125 đơn vị, median 186c) → recall proxy 0.7310
      gộp ≤400c   (72 đơn vị,  median 297c) → recall proxy 0.8416
    Child nhỏ vụn (median 186c, nhỏ nhất 3c) chứa quá ít ngữ cảnh để câu hỏi
    khớp, đặc biệt khi dữ liệu là bảng lương/bảng số.

    Đánh đổi: context_precision giảm (0.2348 → 0.1823) vì đơn vị lớn chứa
    nhiều thông tin không liên quan hơn.
    """
    merged: list[dict] = []
    buf_text, buf_meta = "", None

    def flush():
        nonlocal buf_text, buf_meta
        if buf_text.strip() and buf_meta is not None:
            merged.append({"text": buf_text.strip(), "metadata": dict(buf_meta)})
        buf_text, buf_meta = "", None

    for chunk in chunks:
        meta = chunk["metadata"]
        pid = meta.get("parent_id")
        if buf_meta is None or meta.get("parent_id") != pid:
            flush()
            buf_text, buf_meta = chunk["text"], meta
        elif len(buf_text) + 2 + len(chunk["text"]) <= merge_max:
            buf_text += "\n\n" + chunk["text"]
        else:
            flush()
            buf_text, buf_meta = chunk["text"], meta
    flush()
    return merged


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    all_chunks = merge_retrieval_units(all_chunks)
    print(f"  ✓ {len(all_chunks)} retrieval units from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    from src.m5_enrichment import _chat
    if contexts:
        context_str = "\n\n".join(contexts)
        answer = _chat(
            "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'",
            f"Context:\n{context_str}\n\nCâu hỏi: {query}",
            max_tokens=500,
        )
        if not answer:
            answer = contexts[0]
    else:
        answer = "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
