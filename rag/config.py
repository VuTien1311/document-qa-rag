"""Cấu hình chung. Mọi giá trị đều ghi đè được bằng biến môi trường cùng tên hoặc trong file .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# "gemini": gọi Gemini API (có gói miễn phí, cần GEMINI_API_KEY)
# "ollama": chạy model trên máy (miễn phí hoàn toàn, cần cài Ollama)
PROVIDER = os.getenv("PROVIDER", "gemini")
_DEFAULT_MODELS = {
    "gemini": ("gemini-embedding-2", "gemini-3.5-flash-lite"),
    "ollama": ("bge-m3", "qwen2.5:7b"),
}
EMBED_MODEL = os.getenv("EMBED_MODEL", _DEFAULT_MODELS[PROVIDER][0])
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODELS[PROVIDER][1])

# Vector của các model embedding khác nhau không dùng chung được, nên mỗi model có thư mục dữ liệu riêng
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data")) / EMBED_MODEL.replace(":", "-")
FILES_DIR = DATA_DIR / "files"  # bản gốc các file đã upload
CHROMA_DIR = DATA_DIR / "chroma"  # Vector DB
REGISTRY_PATH = DATA_DIR / "documents.json"  # danh sách tài liệu đã nạp

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))  # số ký tự mỗi chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))  # số ký tự chồng lấn giữa 2 chunk liền nhau
TOP_K = int(os.getenv("TOP_K", 5))  # số chunk đưa cho LLM mỗi câu hỏi
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))  # thấp để bám sát tài liệu
NUM_CTX = int(os.getenv("NUM_CTX", 8192))  # context window khi chạy Ollama (token)

FILES_DIR.mkdir(parents=True, exist_ok=True)
