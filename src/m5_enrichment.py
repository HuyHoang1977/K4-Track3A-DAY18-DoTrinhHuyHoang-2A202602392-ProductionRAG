from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, re, json, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (GEMINI_API_KEY, GEMINI_MODEL, GEMINI_RPM,
                    GEMINI_MAX_RETRIES, GEMINI_QUOTA_GIVE_UP_AFTER,
                    ENRICH_PREFIX_MIN_CHARS)

_QUOTA_GIVE_UP_AFTER = GEMINI_QUOTA_GIVE_UP_AFTER   # ví dụ 60s → hết quota ngày

# Gemini (OpenAI-compatible endpoint) — fallback sang OpenAI nếu không có Gemini key.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

_client = None
_last_call_at = 0.0
# Đặt True khi phát hiện quota ngày đã cạn → mọi call sau trả None ngay,
# không retry (retry vô nghĩa khi hết quota, chỉ tống thêm phút giây).
_quota_exhausted = False
_MIN_INTERVAL = 60.0 / max(GEMINI_RPM, 1)   # 12 RPM → 5s/request


def _throttle() -> None:
    """Chặn tối thiểu _MIN_INTERVAL giữa 2 lần gọi Gemini (free tier = 15 RPM)."""
    global _last_call_at
    wait = _MIN_INTERVAL - (time.monotonic() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


def _retry_delay(exc: Exception, default: float = 15.0) -> float:
    """Lấy thời gian chờ server chỉ định.

    Gemini trả delay ở 2 dạng khác nhau:
      - trong message: "Please retry in 23.4s."
      - trong details:  "'retryDelay': '52s'"
    Phải đọc cả hai, nếu không sẽ rơi về default và retry quá sớm → 429 lặp lại.
    """
    text = str(exc)
    for pattern in (r"retry in ([\d.]+)\s*s", r"retryDelay'?:\s*'?([\d.]+)\s*s"):
        match = re.search(pattern, text, re.I)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return default


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def quota_was_exhausted() -> bool:
    """True nếu quota đã cạn trong lần chạy này (để caller báo cáo)."""
    return _quota_exhausted


def _get_client():
    """Lazy-init Gemini client, fallback OpenAI client nếu không có GEMINI_API_KEY."""
    global _client
    if _client is None:
        if GEMINI_API_KEY:
            from google import genai
            _client = ("gemini", genai.Client(api_key=GEMINI_API_KEY), GEMINI_MODEL)
        elif OPENAI_API_KEY:
            from openai import OpenAI
            _client = ("openai", OpenAI(api_key=OPENAI_API_KEY), "gpt-4o-mini")
        else:
            _client = ("none", None, None)
    return _client


def _chat(system: str, user: str, max_tokens: int) -> str | None:
    """Gọi LLM (ưu tiên Gemini). Trả None nếu thiếu API key hoặc call lỗi.

    Gemini được throttle theo RPM và retry khi gặp 429, tôn trọng retryDelay
    do server trả về. Sau GEMINI_MAX_RETRIES lần thì trả None để caller dùng fallback.
    """
    provider, client, model = _get_client()
    if provider == "none":
        return None

    global _quota_exhausted
    if _quota_exhausted:
        return None      # quota đã cạn → bỏ qua im lặng, dùng fallback

    attempts = GEMINI_MAX_RETRIES + 1 if provider == "gemini" else 1
    for attempt in range(1, attempts + 1):
        try:
            if provider == "gemini":
                from google.genai import types
                _throttle()
                resp = client.models.generate_content(
                    model=model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=max_tokens,
                        temperature=0.3,
                    ),
                )
                return (resp.text or "").strip()

            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()

        except Exception as e:
            if not _is_rate_limit(e):
                print(f"  ⚠️  {provider} call failed: {e}")
                return None

            delay = _retry_delay(e)

            # Retry lần cuối: hoặc delay quá lớn (hết quota cả ngày) thì dừng hẳn.
            if attempt == attempts or delay > _QUOTA_GIVE_UP_AFTER:
                _quota_exhausted = True
                print(f"  ⛔  Gemini quota đã cạn (server yêu cầu chờ {delay:.0f}s). "
                      f"Dừng gọi API, chuyển sang fallback cho toàn bộ phần còn lại.\n"
                      f"     → kiểm tra quota: https://aistudio.google.com/rate-limit", flush=True)
                return None

            print(f"  ⚠️  429 — chờ {delay:.0f}s (theo server) rồi thử lại "
                  f"({attempt}/{attempts - 1})", flush=True)
            time.sleep(delay)

    return None


def _parse_json(content: str | None) -> dict:
    """Parse JSON từ response của LLM, chịu được ```json fences."""
    if not content:
        return {}
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.S)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


# ─── Technique 1: Chunk Summarization ────────────────────


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    summary = _chat("Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. "
                    "Tóm tắt phải ngắn hơn hoặc bằng độ dài đoạn văn gốc.",
                    text, max_tokens=150)
    # Guard: summary dài hơn gốc là tín hiệu LLM bị "nở" nội dung (đặc biệt với
    # chunk ngắn). Khi đó dùng bản extractive — luôn ≤ độ dài gốc.
    if summary and len(summary) <= len(text):
        return summary

    # Extractive fallback (không cần API):
    return _summary_extractive(text)


def _summary_extractive(text: str) -> str:
    """Fallback không cần API: lấy 2 câu đầu làm summary."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    if not sentences:
        return text
    joined = ". ".join(sentences[:2])
    return joined if joined.endswith((".", "!", "?")) else joined + "."


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    content = _chat(
        f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
        "Trả về mỗi câu hỏi trên 1 dòng.",
        text, max_tokens=200)

    if content:
        questions = [q.strip().lstrip("0123456789.-) ") for q in content.split("\n")]
        questions = [q for q in questions if q][:n_questions]
        if questions:
            return questions

    # Extractive fallback: biến câu khẳng định thành câu hỏi.
    return _questions_extractive(text, n_questions)


def _questions_extractive(text: str, n_questions: int = 3) -> list[str]:
    """Fallback không cần API: đổi câu khẳng định thành câu hỏi."""
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    context = _chat(
        "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. "
        "Chỉ trả về 1 câu.",
        f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", max_tokens=80)

    if context:
        return f"{context}\n\n{text}"

    # Simple fallback:
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    content = _chat(
        'Trích xuất metadata từ đoạn văn. Trả về JSON với đúng các key: '
        '{"topic": "...", "entities": ["..."], '
        '"category": "policy|hr|it|finance", "language": "vi|en"}',
        text, max_tokens=150)

    parsed = _parse_json(content)
    if parsed:
        parsed.setdefault("topic", "general")
        parsed.setdefault("entities", [])
        parsed.setdefault("category", "policy")
        parsed.setdefault("language", "vi")
        return parsed

    return _metadata_heuristic(text)


def _metadata_heuristic(text: str) -> dict:
    """Fallback không cần API: đoán category theo từ khóa."""
    lowered = text.lower()
    keywords = {
        "hr": ["nghỉ phép", "lương", "nhân viên", "thử việc", "hợp đồng", "bảo hiểm"],
        "it": ["mật khẩu", "vpn", "máy tính", "hệ thống", "bảo mật", "wifi", "email"],
        "finance": ["ngân sách", "chi phí", "thuế", "hóa đơn", "tài chính"],
    }
    category = next((c for c, words in keywords.items()
                     if any(w in lowered for w in words)), "policy")
    return {"topic": "general", "entities": [], "category": category, "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    content = _chat(
        """Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}""",
        f"Tài liệu: {source}\n\nĐoạn văn:\n{text}", max_tokens=400)

    parsed = _parse_json(content)
    if parsed:
        return {
            "summary": parsed.get("summary", "") or "",
            "questions": parsed.get("questions", []) or [],
            "context": parsed.get("context", "") or "",
            "metadata": parsed.get("metadata", {}) or {},
        }

    # Fallback: gọi thẳng bản extractive (không gọi lại API cho từng kỹ thuật).
    return {
        "summary": _summary_extractive(text),
        "questions": _questions_extractive(text),
        "context": f"Trích từ {source}. " if source else "",
        "metadata": _metadata_heuristic(text),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        # Chỉ nối tiền tố context cho đơn vị đủ lớn. Đo được: với đơn vị 3-25
        # chars, tiền tố chiếm tới 89% nội dung và làm nhiễu vector embedding.
        # (Đo trên proxy recall: ép prefix mọi đơn vị = 0.6804, chỉ ≥200c = 0.7310)
        allow_prefix = len(text) >= ENRICH_PREFIX_MIN_CHARS

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "") if allow_prefix else ""
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = (contextual_prepend(text, source)
                             if "contextual" in methods and allow_prefix else text)
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
