"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Gemini model dùng cho LLM calls.
# gemini-3.5-flash-lite: rẻ/nhất tier Free, đủ cho summary + HyQA + metadata.
# Docs khuyến nghị 3.5 Flash-Lite cho project mới (2.5 series đã giới hạn truy cập).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Model embedding riêng — KHÔNG dùng GEMINI_MODEL ở đây.
# Gemini trả 501 UNIMPLEMENTED khi gọi embeddings endpoint với model chat.
# gemini-embedding-001 là bản stable; gemini-embedding-2-preview là bản mới hơn.
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# --- Gemini rate limit (Free Tier) ---
# Free tier: 15 request/phút / project / model (GenerateRequestsPerProjectPerModel).
# Đặt 12 để có headroom cho request đến từ RAGAS / nhiều process cùng lúc.
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "12"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
# Nếu server yêu cầu chờ > ngưỡng này, coi như quota ngày đã cạn → dừng gọi API
# thay vì retry tiếp. Thực tế quan sát được trong log:
#   retryDelay 15-29s  = hết RPM  → chờ rồi thử lại được
#   retryDelay 31-52s  = hết RPD  → retry vô nghĩa, chỉ tốn thời gian
GEMINI_QUOTA_GIVE_UP_AFTER = int(os.getenv("GEMINI_QUOTA_GIVE_UP_AFTER", "30"))

# Base URL tương thích OpenAI của Gemini — RAGAS cần client OpenAI nên dùng endpoint này.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# Đơn vị truy hồi = gộp nhiều child chunk cùng parent.
# Đo bằng ablation: 125 child (median 186c) → recall 0.7310
#                    72 đơn vị (median 297c) → recall 0.8416
RETRIEVAL_MERGE_MAX = int(os.getenv("RETRIEVAL_MERGE_MAX", "400"))

# Chỉ nối tiền tố context vào đơn vị đủ lớn. Với đơn vị 3-25 chars, tiền tố
# chiếm tới 89% nội dung → làm nhiễu embedding.
ENRICH_PREFIX_MIN_CHARS = int(os.getenv("ENRICH_PREFIX_MIN_CHARS", "200"))

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
