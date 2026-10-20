"""Gemini API. Có gói miễn phí (giới hạn số request/phút và /ngày); lấy key tại https://aistudio.google.com/apikey"""

import os
from collections.abc import Iterator

from google import genai
from google.genai import types

from .. import config

EMBED_DIM = 768  # gemini-embedding-2 mặc định 3072 chiều; 768 đủ tốt mà nhẹ hơn 4 lần
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        # Tự thử lại khi bị giới hạn tần suất (429) hoặc server bận, hay gặp với gói free
        retry = types.HttpRetryOptions(attempts=6, initial_delay=2, max_delay=60, http_status_codes=[429, 500, 503])
        _client = genai.Client(http_options=types.HttpOptions(retry_options=retry))
    return _client


def embed(texts: list[str], kind: str) -> list[list[float]]:
    # gemini-embedding-2 không có tham số task_type: loại nhiệm vụ được ghi ngay trong nội dung
    if kind == "query":
        texts = [f"task: search result | query: {t}" for t in texts]
    else:
        texts = [f"title: none | text: {t}" for t in texts]
    result = _get_client().models.embed_content(
        model=config.EMBED_MODEL,
        # mỗi text bọc trong một Content riêng để nhận về một vector cho mỗi text
        contents=[types.Content(parts=[types.Part.from_text(text=t)]) for t in texts],
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    return [e.values for e in result.embeddings]


def chat_stream(system: str, prompt: str) -> Iterator[str]:
    stream = _get_client().models.generate_content_stream(
        model=config.LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system, temperature=config.TEMPERATURE),
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def check_ready() -> list[str]:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return [
            "Chưa có `GEMINI_API_KEY`. Tạo key miễn phí tại https://aistudio.google.com/apikey "
            "rồi ghi vào file `.env` (copy từ `.env.example`)."
        ]
    problems = []
    for name in (config.EMBED_MODEL, config.LLM_MODEL):
        try:
            _get_client().models.get(model=name)
        except Exception as e:
            problems.append(f"Không dùng được model `{name}`: {e}")
    return problems
