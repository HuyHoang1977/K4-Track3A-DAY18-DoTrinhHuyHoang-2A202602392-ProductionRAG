from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    from sentence_transformers import SentenceTransformer
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}

    # 1. Tách thành câu hoàn chỉnh, bỏ qua đoạn rỗng.
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n\n', text)
    sentences = [s.strip() for s in raw_sentences if s and s.strip()]
    if not sentences:
        return []

    # 2. Encode từng câu bằng all-MiniLM-L6-v2.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    def cosine_sim(a, b) -> float:
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    # 3. Gom nhóm câu liên tiếp có similarity >= threshold.
    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        if cosine_sim(embeddings[i - 1], embeddings[i]) < threshold:
            groups.append([sentences[i]])       # lệch chủ đề → chunk mới
        else:
            groups[-1].append(sentences[i])     # cùng chủ đề → gộp

    return [
        Chunk(
            text=" ".join(group).strip(),
            metadata={**metadata, "chunk_index": i, "strategy": "semantic",
                      "sentence_count": len(group)},
        )
        for i, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}

    def _hard_split(piece: str, size: int) -> list[str]:
        """Cắt cứng theo từ khi một đoạn đơn lẻ vẫn dài hơn size."""
        words, out, cur = piece.split(), [], ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > size:
                out.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            out.append(cur)
        return out

    def _split_units(text: str, size: int) -> list[str]:
        """Chia text thành các đơn vị ≤ size chars, ưu tiên ranh giới câu rồi từ."""
        units: list[str] = []
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(para) <= size:
                units.append(para)
                continue
            # Đoạn quá dài → chia theo câu, câu quá dài → cắt theo từ.
            buf = ""
            for sent in re.split(r'(?<=[.!?])\s+', para):
                sent = sent.strip()
                if not sent:
                    continue
                if len(sent) > size:
                    if buf:
                        units.append(buf)
                        buf = ""
                    units.extend(_hard_split(sent, size))
                elif buf and len(buf) + 1 + len(sent) > size:
                    units.append(buf)
                    buf = sent
                else:
                    buf = f"{buf} {sent}".strip()
            if buf:
                units.append(buf)
        return units

    # 0. Tiền tố id theo nguồn tài liệu.
    # ⚠️ Bắt buộc: pipeline gọi hàm này riêng cho TỪNG tài liệu. Nếu không có
    # tiền tố, mọi tài liệu đều sinh "parent_0" → tra parent theo parent_id sẽ
    # trả về parent của tài liệu khác.
    source = metadata.get("source") or "doc"
    prefix = re.sub(r"[^\w\-]+", "_", str(source)).strip("_") or "doc"

    # 1. Ghép các đơn vị nhỏ thành parent (≤ parent_size), không cắt qua size.
    parents: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    for unit in _split_units(text, parent_size):
        if current and current_len + 2 + len(unit) > parent_size:
            pid = f"{prefix}_parent_{len(parents)}"
            parents.append(Chunk(
                text="\n\n".join(current),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid,
                          "parent_index": len(parents)},
                parent_id=pid,
            ))
            current, current_len = [], 0
        current.append(unit)
        current_len += len(unit) + 2
    if current:
        pid = f"{prefix}_parent_{len(parents)}"
        parents.append(Chunk(
            text="\n\n".join(current),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid,
                      "parent_index": len(parents)},
            parent_id=pid,
        ))

    # 2. Mỗi parent → children (≤ child_size), mỗi child mang parent_id.
    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        for j, unit in enumerate(_split_units(parent.text, child_size)):
            children.append(Chunk(
                text=unit,
                metadata={**metadata, "chunk_type": "child", "parent_id": pid,
                          "child_index": j, "parent_index": parent.metadata["parent_index"]},
                parent_id=pid,
            ))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}

    chunks: list[Chunk] = []
    current_header = ""
    current_content: list[str] = []

    def _flush():
        nonlocal current_header, current_content
        body = "\n\n".join(p for p in current_content if p.strip()).strip()
        if body:
            chunks.append(Chunk(
                text=f"{current_header}\n\n{body}".strip() if current_header else body,
                metadata={**metadata, "chunk_index": len(chunks), "section": current_header,
                          "strategy": "structure"},
            ))
        current_content = []

    # re.split với capturing group → xen kẳng: [content, header, content, header, ...]
    for part in re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE):
        if re.match(r'^#{1,3}\s+\S', part):
            _flush()                       # đóng section trước, rồi mở section mới
            current_header = part.strip()
        elif part.strip():
            current_content.append(part.strip())

    _flush()                               # flush section cuối cùng
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
