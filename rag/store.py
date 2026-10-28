"""Vector DB (ChromaDB) chứa các chunk, cộng một file JSON ghi danh sách tài liệu đã nạp."""

import json

import chromadb
from chromadb.config import Settings

from . import config

_db = chromadb.PersistentClient(path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
# Không dùng embedding function có sẵn của Chroma: vector được tạo bằng provider (Gemini/Ollama) rồi truyền vào.
_collection = _db.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"}, embedding_function=None)


# ---------- Chunk + vector ----------


def add_chunks(doc: dict, chunks: list[dict], embeddings: list[list[float]], start: int = 0) -> None:
    """`doc` cần id, filename, title; mỗi chunk lưu kèm để hiển thị nguồn mà không phải tra lại danh sách tài liệu."""
    meta = {"doc_id": doc["id"], "filename": doc["filename"], "title": doc["title"]}
    _collection.add(
        ids=[f"{doc['id']}:{start + i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[meta | {"location": c["location"]} for c in chunks],
    )


def search(query_embedding: list[float], top_k: int, doc_ids: list[str] | None = None) -> list[dict]:
    """Lấy top_k chunk gần câu hỏi nhất theo cosine. doc_ids: chỉ tìm trong các tài liệu này."""
    if _collection.count() == 0:
        return []
    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    res = _collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
    return [
        {"id": chunk_id, "text": text, **meta, "score": 1 - distance}
        for chunk_id, text, meta, distance in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
        )
    ]


def all_chunks() -> list[dict]:
    """Mọi chunk (không kèm vector), cho tìm kiếm từ khoá."""
    res = _collection.get(include=["documents", "metadatas"])
    return [
        {"id": chunk_id, "text": text, **meta, "score": None}
        for chunk_id, text, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
    ]


def get_chunk(chunk_id: str) -> dict | None:
    res = _collection.get(ids=[chunk_id], include=["documents", "metadatas"])
    if not res["ids"]:
        return None
    return {"id": chunk_id, "text": res["documents"][0], **res["metadatas"][0], "score": None}


def chunk_count() -> int:
    return _collection.count()


def delete_chunks(doc_id: str) -> None:
    _collection.delete(where={"doc_id": doc_id})


# ---------- Danh sách tài liệu ----------


def list_documents() -> list[dict]:
    if not config.REGISTRY_PATH.exists():
        return []
    return json.loads(config.REGISTRY_PATH.read_text(encoding="utf-8"))


def get_document(doc_id: str) -> dict | None:
    return next((d for d in list_documents() if d["id"] == doc_id), None)


def save_document(doc: dict) -> None:
    _write([d for d in list_documents() if d["id"] != doc["id"]] + [doc])


def remove_document(doc_id: str) -> None:
    _write([d for d in list_documents() if d["id"] != doc_id])


def _write(docs: list[dict]) -> None:
    config.REGISTRY_PATH.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
