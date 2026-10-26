"""Hai luồng của RAG: (1) nạp tài liệu, (2) hỏi đáp = tìm kiếm ngữ nghĩa + sinh câu trả lời."""

import hashlib
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

from . import config, llm, store
from .chunker import split_text
from .loaders import load_document

EMBED_BATCH_SIZE = 16
CANDIDATE_POOL = 4  # lấy TOP_K * 4 chunk ứng viên rồi mới chọn TOP_K chunk đa dạng nguồn

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
        document = load_document(path)
        doc = {"id": doc_id, "filename": filename, "title": document.title or Path(filename).stem}
        chunks = [
            {"text": text, "location": section.location}
            for section in document.sections
            for text in split_text(section.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        ]
        if not chunks:
            raise ValueError("Không đọc được nội dung chữ nào trong tài liệu.")
        store.delete_chunks(doc_id)  # dọn chunk còn sót nếu lần nạp trước bị ngắt giữa chừng
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            vectors = llm.embed([with_context(doc["title"], c) for c in batch], "document")
            store.add_chunks(doc, batch, vectors, start)
            if on_progress:
                on_progress(min(start + EMBED_BATCH_SIZE, len(chunks)) / len(chunks))
    except Exception:
        store.delete_chunks(doc_id)
        path.unlink(missing_ok=True)
        raise

    doc |= {
        "file": path.name,
        "size": len(data),
        "pages": document.pages,
        "chunks": len(chunks),
        "added_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    store.save_document(doc)
    return doc


def with_context(title: str, chunk: dict) -> str:
    """Văn bản đem đi embedding: chunk kèm tiêu đề tài liệu và vị trí (chỉ dùng để tạo vector, không hiển thị).

    Chunk cắt giữa một bảng học phí có thể không còn chữ "Quy Nhơn" nào; thiếu dòng tiêu đề này thì
    vector của nó không biết mình thuộc trang nào và bị xếp hạng thấp với câu hỏi về Quy Nhơn.
    """
    header = " — ".join(part for part in (title, chunk["location"]) if part)
    return f"{header}\n\n{chunk['text']}"


def remove(doc_id: str) -> None:
    if doc := store.get_document(doc_id):
        (config.FILES_DIR / doc["file"]).unlink(missing_ok=True)
    store.delete_chunks(doc_id)
    store.remove_document(doc_id)


# ---------- 2. Hỏi đáp (Question Answering) ----------


def retrieve(question: str, doc_ids: list[str] | None = None) -> list[dict]:
    """Tìm kiếm ngữ nghĩa: embed câu hỏi, lấy các chunk có vector gần nhất rồi chọn TOP_K chunk đưa cho LLM."""
    [query_vector] = llm.embed([question], "query")
    candidates = store.search(query_vector, config.TOP_K * CANDIDATE_POOL, doc_ids)
    return select_chunks(candidates, config.TOP_K, config.MAX_CHUNKS_PER_DOC)


def select_chunks(candidates: list[dict], k: int, max_per_doc: int) -> list[dict]:
    """k chunk liên quan nhất, mỗi tài liệu góp tối đa max_per_doc chunk (0 = không giới hạn).

    Vì chunk được embed kèm tiêu đề, mọi chunk của một trang khớp tiêu đề câu hỏi đều được xếp hạng cao và có thể
    chiếm hết k chỗ; câu hỏi liệt kê cần thông tin rải trên nhiều trang thì bị trả lời thiếu. Giới hạn quá chặt
    (1–2) lại cắt mất đoạn chứa đáp án nằm sâu trong một trang dài; 3 là mức đo được tốt nhất (xem README).
    """
    if not max_per_doc:
        return candidates[:k]
    chosen, per_doc = [], Counter()
    for hit in candidates:
        if per_doc[hit["doc_id"]] < max_per_doc:
            chosen.append(hit)
            per_doc[hit["doc_id"]] += 1
            if len(chosen) == k:
                break
    return chosen


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
    location = hit["location"]
    if location.startswith("http"):  # trang web: bản thân URL đã đủ định danh
        return location
    return f"{hit['filename']}, {location}" if location else hit["filename"]
