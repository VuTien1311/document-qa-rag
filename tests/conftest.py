"""Test chạy với embedding/LLM giả: không cần API key, không tốn request."""

import hashlib
import math
import os
import re
import tempfile

# Phải đặt trước khi import rag, vì rag.config đọc DATA_DIR lúc import
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="rag_test_")

import pytest  # noqa: E402

from rag import llm  # noqa: E402


def fake_embed(texts: list[str], kind: str) -> list[list[float]]:
    """Vector bag-of-words băm vào 256 chiều: câu nào chung nhiều từ thì gần nhau."""
    vectors = []
    for text in texts:
        v = [0.0] * 256
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
        norm = math.sqrt(sum(x * x for x in v)) or 1
        vectors.append([x / norm for x in v])
    return vectors


def fake_chat_stream(system: str, prompt: str):
    yield from ["Lỗi E2 do cảm biến bẩn [1]. ", "Xem [2, 3] và [9]."]


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    monkeypatch.setattr(llm, "embed", fake_embed)
    monkeypatch.setattr(llm, "chat_stream", fake_chat_stream)
