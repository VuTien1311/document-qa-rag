"""Tìm kiếm theo từ khoá (BM25), bổ sung cho tìm kiếm vector: bắt chính xác mã số, con số, tên riêng."""

import re
import unicodedata

from rank_bm25 import BM25Okapi

from . import store

_index: tuple[int, list[dict], BM25Okapi] | None = None  # (số chunk lúc dựng, các chunk, BM25)


def tokenize(text: str) -> list[str]:
    """Âm tiết tiếng Việt / từ tiếng Anh viết thường; số giữ nguyên dấu phân cách ("22.120.000")."""
    return re.findall(r"\d[\d.,]*\d|\w+", unicodedata.normalize("NFC", text).lower())


def search(query: str, top_k: int, doc_ids: list[str] | None = None) -> list[dict]:
    chunks, bm25 = _get_index()
    if not chunks:
        return []
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(
        (i for i, chunk in enumerate(chunks) if scores[i] > 0 and (not doc_ids or chunk["doc_id"] in doc_ids)),
        key=lambda i: scores[i],
        reverse=True,
    )
    return [chunks[i] for i in ranked[:top_k]]


def _get_index() -> tuple[list[dict], BM25Okapi | None]:
    """Dựng lại chỉ mục khi số chunk thay đổi (nạp hoặc xoá tài liệu); vài trăm chunk dựng trong tích tắc."""
    global _index
    count = store.chunk_count()
    if count == 0:
        return [], None
    if _index is None or _index[0] != count:
        chunks = store.all_chunks()
        _index = (count, chunks, BM25Okapi([tokenize(c["text"]) for c in chunks]))
    return _index[1], _index[2]
