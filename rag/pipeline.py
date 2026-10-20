"""Hai luồng của RAG: (1) nạp tài liệu, (2) hỏi đáp = tìm kiếm ngữ nghĩa + sinh câu trả lời."""

import hashlib
import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from . import config, llm, store
from .chunker import split_text
from .loaders import load_document

EMBED_BATCH_SIZE = 16

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu của người dùng.
Quy tắc:
1. Chỉ dùng thông tin trong các đoạn trích được cung cấp, không dùng kiến thức bên ngoài.
2. Sau mỗi ý, ghi số của đoạn trích làm nguồn trong ngoặc vuông, ví dụ: [1] hoặc [2][3].
3. Nếu các đoạn trích không có thông tin để trả lời, chỉ trả lời:
   "Mình không tìm thấy thông tin này trong tài liệu." Không được bịa.
4. Trả lời ngắn gọn, rõ ràng, bằng cùng ngôn ngữ với câu hỏi. Không liệt kê lại danh sách nguồn ở cuối."""


# ---------- 1. Nạp tài liệu (Document Ingestion) ----------


def ingest(filename: str, data: bytes, on_progress: Callable[[float], None] | None = None) -> dict:
    """Đọc file → chia chunk → embedding → lưu Vector DB. File trùng nội dung đã nạp thì bỏ qua."""
    doc_id = hashlib.sha256(data).hexdigest()[:16]
    if doc := store.get_document(doc_id):
        return doc

    path = config.FILES_DIR / f"{doc_id}{Path(filename).suffix.lower()}"
    path.write_bytes(data)
    try:
        sections, pages = load_document(path)
        chunks = [
            {"text": text, "location": section.location}
            for section in sections
            for text in split_text(section.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        ]
        if not chunks:
            raise ValueError("Không đọc được nội dung chữ nào trong tài liệu.")
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            store.add_chunks(doc_id, filename, batch, llm.embed([c["text"] for c in batch], "document"), start)
            if on_progress:
                on_progress(min(start + EMBED_BATCH_SIZE, len(chunks)) / len(chunks))
    except Exception:
        store.delete_chunks(doc_id)
        path.unlink(missing_ok=True)
        raise

    doc = {
        "id": doc_id,
        "filename": filename,
        "file": path.name,
        "size": len(data),
        "pages": pages,
        "chunks": len(chunks),
        "added_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    store.save_document(doc)
    return doc


def remove(doc_id: str) -> None:
    if doc := store.get_document(doc_id):
        (config.FILES_DIR / doc["file"]).unlink(missing_ok=True)
    store.delete_chunks(doc_id)
    store.remove_document(doc_id)


# ---------- 2. Hỏi đáp (Question Answering) ----------


def retrieve(question: str, doc_ids: list[str] | None = None) -> list[dict]:
    """Tìm kiếm ngữ nghĩa: embed câu hỏi rồi lấy các chunk có vector gần nhất."""
    [query_vector] = llm.embed([question], "query")
    return store.search(query_vector, config.TOP_K, doc_ids)


def generate(question: str, hits: list[dict]) -> Iterator[str]:
    """Đánh số các chunk tìm được thành [1], [2]... rồi nhờ LLM trả lời kèm số nguồn."""
    context = "\n\n".join(f"[{i}] ({source_label(hit)})\n{hit['text']}" for i, hit in enumerate(hits, start=1))
    return llm.chat_stream(SYSTEM_PROMPT, f"Các đoạn trích:\n\n{context}\n\nCâu hỏi: {question}")


def cited_sources(answer: str, hits: list[dict]) -> list[tuple[int, dict]]:
    """Các chunk mà câu trả lời thực sự trích dẫn: [1], [2][3], [1, 3]..."""
    groups = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer)
    numbers = {int(n) for group in groups for n in re.findall(r"\d+", group)}
    return [(n, hits[n - 1]) for n in sorted(numbers) if 1 <= n <= len(hits)]


def source_label(hit: dict) -> str:
    return f"{hit['filename']}, {hit['location']}" if hit["location"] else hit["filename"]
