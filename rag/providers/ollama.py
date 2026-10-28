"""Ollama: chạy model ngay trên máy, miễn phí hoàn toàn, dữ liệu không rời khỏi máy. Tải tại https://ollama.com"""

from collections.abc import Iterator

import ollama

from .. import config

_client = ollama.Client()  # mặc định http://localhost:11434, đổi bằng biến môi trường OLLAMA_HOST


def embed(texts: list[str], kind: str) -> list[list[float]]:
    return _client.embed(model=config.EMBED_MODEL, input=texts).embeddings  # bge-m3 không phân biệt kind


def chat_stream(system: str, prompt: str) -> Iterator[str]:
    stream = _client.chat(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        stream=True,
        options={"num_ctx": config.NUM_CTX, "temperature": config.TEMPERATURE},
    )
    for chunk in stream:
        yield chunk.message.content


def check_ready() -> list[str]:
    try:
        installed = {m.model for m in _client.list().models}
    except Exception:
        return ["Cannot reach Ollama. Open the Ollama app (or run `ollama serve`)."]
    return [
        f"Model `{name}` is missing. Run `ollama pull {name}`"
        for name in (config.EMBED_MODEL, config.LLM_MODEL)
        if name not in installed and f"{name}:latest" not in installed
    ]
