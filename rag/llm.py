"""Cửa ngõ duy nhất tới model AI. Đổi nhà cung cấp bằng biến PROVIDER, phần còn lại không cần sửa.

Mỗi provider cung cấp 3 hàm:
    embed(texts, kind)          -> list vector; kind = "document" (chunk) hoặc "query" (câu hỏi)
    chat_stream(system, prompt) -> sinh câu trả lời, trả về từng mẩu chữ
    check_ready()               -> danh sách vấn đề cần sửa (rỗng = sẵn sàng)
"""

from . import config

if config.PROVIDER == "ollama":
    from .providers.ollama import chat_stream, check_ready, embed
elif config.PROVIDER == "gemini":
    from .providers.gemini import chat_stream, check_ready, embed
else:
    raise ValueError(f"Invalid PROVIDER {config.PROVIDER!r}: use 'gemini' or 'ollama'")

__all__ = ["chat_stream", "check_ready", "embed"]
