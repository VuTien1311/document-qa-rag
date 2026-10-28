"""HTTP API for the RAG pipeline, so other apps (chatbots, websites) can use it.

Run: uvicorn api:app --reload        Interactive docs: http://localhost:8000/docs
"""

import json
from collections.abc import Iterator
from typing import Annotated, Literal
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from rag import config, pipeline, store
from rag.evidence import claim_for, evidence_units, render_pdf_page

app = FastAPI(
    title="Document QA RAG",
    version="0.1.0",
    description="Question answering over your documents. Every answer cites the passages it is based on.",
)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1, examples=["What is the tuition for Artificial Intelligence at the Hanoi campus?"]
    )
    doc_ids: list[str] | None = Field(None, description="Search only these documents (default: all)")
    history: list[Message] = Field([], description="Earlier messages, used to understand follow-up questions")


class Source(BaseModel):
    n: int = Field(description="Number used in the answer, e.g. [1]")
    chunk_id: str
    title: str
    location: str = Field(description="Page, section or URL")
    text: str
    evidence: list[str] = Field(description="Lines of the passage that match the claims citing it")
    score: float | None = Field(description="Cosine similarity to the question (None for keyword-only matches)")
    page_image: str | None = Field(description="URL of the highlighted PDF page, for PDF sources")


class AskResponse(BaseModel):
    answer: str
    query: str = Field(description="The question actually searched (follow-ups are rewritten into full questions)")
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "documents": len(store.list_documents()),
        "provider": config.PROVIDER,
        "llm": config.LLM_MODEL,
        "embedding": config.EMBED_MODEL,
    }


@app.get("/documents")
def list_documents() -> list[dict]:
    return store.list_documents()


@app.post("/documents", status_code=201)
async def add_documents(files: list[UploadFile]) -> dict:
    """Add PDF, DOCX, TXT or Markdown files. Files already added are skipped; one bad file does not stop the rest."""
    added, errors = [], []
    for file in files:
        try:
            added.append(pipeline.ingest(file.filename, await file.read()))
        except Exception as e:
            errors.append({"file": file.filename, "error": str(e)})
    return {"added": added, "errors": errors}


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str) -> None:
    if not store.get_document(doc_id):
        raise HTTPException(404, "Document not found")
    pipeline.remove(doc_id)


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    query, hits = _search(request)
    answer = "".join(pipeline.generate(query, hits))
    return AskResponse(answer=answer, query=query, sources=_sources(answer, hits))


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    """Server-Sent Events: `token` events carry pieces of the answer as they are generated, then one `sources`
    event carries the query and the cited sources."""
    query, hits = _search(request)  # before streaming starts, so search errors are normal HTTP errors

    def events() -> Iterator[str]:
        answer = ""
        for piece in pipeline.generate(query, hits):
            answer += piece
            yield f"event: token\ndata: {json.dumps(piece, ensure_ascii=False)}\n\n"
        payload = {"query": query, "sources": [s.model_dump() for s in _sources(answer, hits)]}
        yield f"event: sources\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/chunks/{chunk_id}/page.png", responses={200: {"content": {"image/png": {}}}})
def page_image(chunk_id: str, evidence: Annotated[list[str] | None, Query()] = None) -> Response:
    """The PDF page a chunk comes from, with the chunk highlighted in yellow and the `evidence` lines in orange."""
    chunk = store.get_chunk(chunk_id)
    doc = store.get_document(chunk["doc_id"]) if chunk else None
    if not doc or not doc["file"].endswith(".pdf") or not chunk["location"].startswith("page "):
        raise HTTPException(404, "No PDF page for this chunk")
    page = int(chunk["location"].split()[1])
    return Response(
        render_pdf_page(config.FILES_DIR / doc["file"], page, chunk["text"], evidence or []), media_type="image/png"
    )


def _search(request: AskRequest) -> tuple[str, list[dict]]:
    query = pipeline.standalone_question(request.question, [m.model_dump() for m in request.history])
    return query, pipeline.retrieve(query, request.doc_ids)


def _sources(answer: str, hits: list[dict]) -> list[Source]:
    sources = []
    for n, hit in pipeline.cited_sources(answer, hits):
        evidence = evidence_units(hit["text"], claim_for(answer, n))
        is_pdf_page = hit["filename"].lower().endswith(".pdf") and hit["location"].startswith("page ")
        image = f"/chunks/{hit['id']}/page.png?{urlencode({'evidence': evidence}, doseq=True)}" if is_pdf_page else None
        sources.append(
            Source(
                n=n,
                chunk_id=hit["id"],
                title=hit.get("title") or hit["filename"],
                location=hit["location"],
                text=hit["text"],
                evidence=evidence,
                score=hit.get("score"),
                page_image=image,
            )
        )
    return sources
