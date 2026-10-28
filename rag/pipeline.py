"""Hai luồng của RAG: (1) nạp tài liệu, (2) hỏi đáp = tìm kiếm ngữ nghĩa + sinh câu trả lời."""

import hashlib
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

from . import config, keyword_search, llm, store
from .chunker import split_text
from .loaders import load_document

EMBED_BATCH_SIZE = 16
CANDIDATE_POOL = 4  # lấy TOP_K * 4 chunk ứng viên rồi mới chọn TOP_K chunk đa dạng nguồn

SYSTEM_PROMPT = """You answer questions using only the user's documents.
Rules:
1. Use only information from the provided excerpts, never outside knowledge.
2. After each claim, cite the excerpt number in square brackets, e.g. [1] or [2][3].
3. If the excerpts do not contain the answer, reply only with the "not found" sentence given after the question.
   Never make anything up.
4. Answer concisely and clearly, in the language requested after the question, even if the excerpts are in another
   language. Do not list the sources again at the end."""

REWRITE_PROMPT = """Rewrite the user's last question so that it can be understood without the conversation.
Fill in what it refers to (the program, campus, scholarship...) from the earlier messages, keep its meaning and its
language. If it is already self-contained, return it unchanged. Output only the question."""
HISTORY_MESSAGES = 6  # số tin nhắn gần nhất đưa vào khi viết lại câu hỏi nối tiếp

NOT_FOUND = {
    "Vietnamese": "Mình không tìm thấy thông tin này trong tài liệu.",
    "English": "I could not find this information in the documents.",
}
_VIETNAMESE_LETTERS = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]")
# Từ tiếng Việt hay gặp khi gõ không dấu; chọn những từ không trùng từ tiếng Anh thông dụng
_VIETNAMESE_WORDS = {
    "bao", "nhieu", "khong", "nhung", "cua", "nao", "hoc", "phi", "nganh", "truong", "duoc", "thi", "sao", "voi",
    "minh", "toi", "lam", "nhu", "nay", "gi", "cac", "mot", "trong", "dang", "ky",
}  # fmt: skip
_ENGLISH_WORDS = {
    "the", "what", "which", "who", "when", "where", "why", "how", "is", "are", "does", "do", "can", "of", "for",
    "in", "at", "a", "an", "to", "many", "much", "and", "or", "there", "have", "has",
}  # fmt: skip


def question_language(question: str) -> str:
    """ "Vietnamese" hoặc "English", tuỳ bên nào có nhiều từ đặc trưng hơn.

    Từ tiếng Việt là từ có dấu hoặc từ gõ không dấu quen thuộc ("hoc phi bao nhieu"); từ tiếng Anh là các từ chức năng
    (what, the, does...). Đếm thay vì chỉ tìm chữ có dấu, vì câu tiếng Anh hay nhắc tên riêng tiếng Việt
    ("the Chiến binh công nghệ scholarship"). Chỉ dặn LLM "trả lời cùng ngôn ngữ với câu hỏi" là chưa đủ: đoạn trích
    tiếng Việt kéo câu trả lời cho câu hỏi tiếng Anh sang tiếng Việt (đo được 2/6 câu).
    """
    words = re.findall(r"\w+", question.lower())
    vietnamese = sum(bool(_VIETNAMESE_LETTERS.search(w)) or w in _VIETNAMESE_WORDS for w in words)
    english = sum(w in _ENGLISH_WORDS for w in words)
    return "Vietnamese" if vietnamese > english else "English"


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
            raise ValueError("No text could be extracted from this document.")
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
    """Lấy chunk ứng viên bằng vector (và từ khoá nếu bật HYBRID_SEARCH) rồi chọn TOP_K chunk đưa cho LLM."""
    [query_vector] = llm.embed([question], "query")
    pool = config.TOP_K * CANDIDATE_POOL
    candidates = store.search(query_vector, pool, doc_ids)
    if config.HYBRID_SEARCH:
        candidates = reciprocal_rank_fusion(candidates, keyword_search.search(question, pool, doc_ids))
    return select_chunks(candidates, config.TOP_K, config.MAX_CHUNKS_PER_DOC)


def reciprocal_rank_fusion(*rankings: list[dict], k: int = 60) -> list[dict]:
    """Trộn nhiều bảng xếp hạng: mỗi chunk được cộng 1/(k + hạng) ở từng bảng có nó.

    Chỉ dùng thứ hạng nên không cần quy đổi điểm cosine và điểm BM25 (hai thang đo khác nhau) về cùng thang.
    """
    scores: dict[str, float] = {}
    hits: dict[str, dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1 / (k + rank)
            if hits.get(hit["id"], {}).get("score") is None:
                hits[hit["id"]] = hit  # ưu tiên bản có điểm cosine (từ tìm kiếm vector) để hiển thị
    return [hits[chunk_id] for chunk_id in sorted(scores, key=scores.get, reverse=True)]


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


def standalone_question(question: str, history: list[dict]) -> str:
    """Câu hỏi nối tiếp ("Còn ở Đà Nẵng thì sao?") được viết lại thành câu hỏi đầy đủ trước khi tìm kiếm.

    `history` là các tin nhắn trước đó dạng {"role": "user" | "assistant", "content": ...}. Không có lịch sử thì trả
    nguyên câu hỏi, không tốn lượt gọi LLM.
    """
    if not history:
        return question
    turns = "\n".join(f"{m['role']}: {m['content'][:600]}" for m in history[-HISTORY_MESSAGES:])
    rewritten = "".join(llm.chat_stream(REWRITE_PROMPT, f"Conversation:\n{turns}\n\nLast question: {question}"))
    return rewritten.strip() or question


def generate(question: str, hits: list[dict]) -> Iterator[str]:
    """Đánh số các chunk tìm được thành [1], [2]... rồi nhờ LLM trả lời kèm số nguồn."""
    context = "\n\n".join(f"[{i}] ({source_label(hit)})\n{hit['text']}" for i, hit in enumerate(hits, start=1))
    language = question_language(question)
    instructions = f'Answer in {language}. "Not found" sentence: "{NOT_FOUND[language]}"'
    return llm.chat_stream(SYSTEM_PROMPT, f"Excerpts:\n\n{context}\n\nQuestion: {question}\n\n{instructions}")


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
