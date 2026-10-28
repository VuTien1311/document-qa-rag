"""API tests with fake models (see conftest.py): no API key and no network needed."""

import json

import pytest
from fastapi.testclient import TestClient

from api import app
from rag import store
from rag.config import BASE_DIR

SAMPLE_DOCS = BASE_DIR / "sample_docs"
client = TestClient(app)


@pytest.fixture
def docs():
    files = [("files", (p.name, p.read_bytes())) for p in sorted(SAMPLE_DOCS.iterdir())]
    response = client.post("/documents", files=files)
    assert response.status_code == 201 and response.json()["errors"] == []
    yield {d["filename"]: d for d in response.json()["added"]}
    for doc in store.list_documents():
        client.delete(f"/documents/{doc['id']}")


def test_health_and_document_list(docs):
    assert client.get("/health").json()["documents"] == 3
    assert {d["filename"] for d in client.get("/documents").json()} == set(docs)


def test_bad_file_is_reported_without_stopping_the_others(docs):
    files = [("files", ("notes.xlsx", b"abc")), ("files", ("a.txt", "Ghi chú".encode()))]
    body = client.post("/documents", files=files).json()
    assert [a["filename"] for a in body["added"]] == ["a.txt"]
    assert body["errors"][0]["file"] == "notes.xlsx" and "Unsupported" in body["errors"][0]["error"]


def test_ask_returns_answer_with_cited_sources(docs):
    body = client.post("/ask", json={"question": "Máy báo lỗi E2 thì phải làm gì?"}).json()
    assert "[1]" in body["answer"] and body["query"] == "Máy báo lỗi E2 thì phải làm gì?"
    first = body["sources"][0]
    assert first["n"] == 1 and first["location"] == "page 3" and first["page_image"].startswith("/chunks/")

    image = client.get(first["page_image"])
    assert image.status_code == 200 and image.content.startswith(b"\x89PNG")


def test_ask_stream_sends_tokens_then_sources(docs):
    with client.stream("POST", "/ask/stream", json={"question": "Máy báo lỗi E2 thì phải làm gì?"}) as response:
        events = [e for e in response.read().decode().split("\n\n") if e]
    kinds = [e.split("\n")[0] for e in events]
    assert kinds[:-1] == ["event: token"] * (len(kinds) - 1) and kinds[-1] == "event: sources"
    answer = "".join(json.loads(e.split("data: ", 1)[1]) for e in events[:-1])
    sources = json.loads(events[-1].split("data: ", 1)[1])["sources"]
    assert "[1]" in answer and sources[0]["n"] == 1


def test_follow_up_is_rewritten_with_history(docs):
    history = [{"role": "user", "content": "Lỗi E1 là gì?"}, {"role": "assistant", "content": "Bộ lọc lắp sai [1]."}]
    body = client.post("/ask", json={"question": "Còn E2?", "history": history}).json()
    assert body["query"] != "Còn E2?"  # câu hỏi nối tiếp được viết lại trước khi tìm


def test_errors(docs):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.delete("/documents/khong-ton-tai").status_code == 404
    assert client.get("/chunks/khong-ton-tai/page.png").status_code == 404
