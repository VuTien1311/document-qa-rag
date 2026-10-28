"""Chat page: documents in the sidebar, answers with cited and verifiable sources in the main area."""

import json
import re
import time
from pathlib import Path

import streamlit as st

from rag import config, llm, pipeline, store
from rag.evidence import claim_for, evidence_units, render_pdf_page
from rag.loaders import SUPPORTED_TYPES

FILE_ICONS = {"pdf": "📕", "docx": "📘", "txt": "📄", "md": "📄"}
SAMPLE_DOCS = config.BASE_DIR / "sample_docs"
FEEDBACK_LOG = config.DATA_DIR / "feedback.jsonl"
# Suggested questions for the demo data (sample_docs + FPT admissions pages); change them to fit your documents
EXAMPLE_QUESTIONS = [
    "What is the tuition for Artificial Intelligence at the Hanoi campus?",
    "Which scholarships does the university offer?",
    "What should I do when the air purifier shows error E2?",
]

for key, default in {"messages": [], "uploader_key": 0, "notices": []}.items():
    st.session_state.setdefault(key, default)


def human_size(size: int) -> str:
    return f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"


def doc_name(doc: dict) -> str:
    """Readable title ("Học phí tại Campus Quy Nhơn") instead of the file name when there is one."""
    return (doc.get("title") or doc["filename"]).replace("_", " ")  # "Huong_dan" cannot wrap in a narrow sidebar


def plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def friendly_error(error: Exception) -> str:
    """Plain-language explanation for errors common on the free tier; other errors are shown as they are."""
    text = str(error)
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return (
            "The free Gemini quota is used up (per minute or per day). Wait a moment and try again; "
            "the daily quota resets at midnight Pacific time."
        )
    if "API_KEY_INVALID" in text or "API key not valid" in text:
        return "The API key is invalid. Check `GEMINI_API_KEY` in your `.env` file."
    return text[:300]


def ingest_files(files: list[tuple[str, bytes]]) -> None:
    for name, data in files:
        progress = st.progress(0.0, text=f"Adding {name}...")
        try:
            pipeline.ingest(name, data, on_progress=progress.progress)
        except Exception as e:
            st.session_state.notices.append(f"Could not add **{name}**: {friendly_error(e)}")
    st.session_state.uploader_key += 1  # new key empties the upload box
    st.rerun()


def log_feedback(msg: dict, rating: int) -> None:
    """Keeps 👍/👎 with the question and answer, so poor answers can be added to the evaluation set."""
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rating": "up" if rating else "down",
        "question": msg["question"],
        "searched_for": msg.get("query", msg["question"]),
        "answer": msg["content"],
        "sources": [pipeline.source_label(hit) for _, hit in msg["sources"]],
    }
    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def excerpt_markdown(text: str, evidence: list[str]) -> str:
    """Compact excerpt: evidence highlighted, Markdown headings (# ...) shown as bold text, line breaks kept."""
    for unit in evidence:
        text = text.replace(unit, highlight(unit), 1)
    text = re.sub(r"^#{1,6}\s*(.+)$", r"**\1**", text, flags=re.MULTILINE)
    return text.replace("\n", "  \n")


def highlight(unit: str) -> str:
    unit = unit.replace("[", r"\[").replace("]", r"\]")
    if unit.startswith("|"):  # table row: highlight each cell so the table still renders
        return "|".join(f" :orange-background[{cell.strip()}] " if cell.strip() else cell for cell in unit.split("|"))
    return f":orange-background[{unit}]"


def pdf_page(hit: dict) -> tuple[str, int] | None:
    """(stored file, page number) when the source is a PDF page, so the original page can be shown."""
    if not (hit["filename"].lower().endswith(".pdf") and hit["location"].startswith("page ")):
        return None
    doc = store.get_document(hit["doc_id"])
    return (str(config.FILES_DIR / doc["file"]), int(hit["location"].split()[1])) if doc else None


@st.cache_data(show_spinner=False, max_entries=64)
def page_image(path: str, page: int, passage: str, evidence: tuple[str, ...]) -> bytes:
    return render_pdf_page(Path(path), page, passage, list(evidence))


def show_sources(sources: list, answer: str) -> None:
    for n, hit in sources:
        location = hit["location"]
        label = f"[{n}] {doc_name(hit)}"
        if location and not location.startswith("http"):
            label += f" · {location}"
        evidence = evidence_units(hit["text"], claim_for(answer, n))
        with st.expander(label):
            if location.startswith("http"):
                st.markdown(f"[{location}]({location})")
            st.caption(f"Relevance {hit['score']:.2f}" if hit.get("score") is not None else "Found by keyword match")
            excerpt = excerpt_markdown(hit["text"], evidence)
            if page := pdf_page(hit):
                excerpt_tab, page_tab = st.tabs(["Excerpt", "Original page"])
                excerpt_tab.markdown(excerpt)
                page_tab.image(page_image(*page, hit["text"], tuple(evidence)))
                page_tab.caption("Yellow: the cited passage · orange: the lines that match the answer")
            else:
                st.markdown(excerpt)


def show_message(msg: dict, index: int) -> None:
    with st.chat_message(msg["role"]):
        if msg.get("error"):
            st.error(msg["content"])
            return
        st.markdown(msg["content"])
        show_sources(msg.get("sources", []), msg["content"])
        if msg.get("query"):
            st.caption(f"🔎 Searched for: {msg['query']}")
        if msg.get("meta"):
            st.caption(msg["meta"])
        if msg["role"] == "assistant":
            rating = st.feedback("thumbs", key=f"feedback_{index}")
            if rating is not None and msg.get("rating") != rating:
                msg["rating"] = rating
                log_feedback(msg, rating)
                st.toast("Thanks for the feedback!")


st.title("📚 Document QA")
st.caption("Ask questions about your documents. Every answer cites its sources so you can check it.")

if not st.session_state.get("ready"):  # check once per session to save requests
    with st.spinner("Connecting to the model..."):
        problems = llm.check_ready()
    if problems:
        st.error("**The AI model is not ready**\n\n" + "\n".join(f"- {p}" for p in problems))
        st.info("Fix this, then reload the page.")
        st.stop()
    st.session_state.ready = True

# ---------- Sidebar: documents ----------
with st.sidebar:
    st.header("📁 Documents")
    uploads = st.file_uploader(
        "Upload PDF, DOCX, TXT or Markdown",
        type=list(SUPPORTED_TYPES),
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        help="Each file is split into passages and embedded for search. Files already added are skipped.",
    )
    if uploads:
        ingest_files([(f.name, f.getvalue()) for f in uploads])

    for notice in st.session_state.notices:
        st.warning(notice)
    st.session_state.notices = []

    docs = store.list_documents()
    if not docs:
        st.info("No documents yet. Upload files above, or load the sample documents from the main page.")
    else:
        # Many documents (e.g. a crawled site): collapse the list and allow searching by name
        with st.expander(f"{plural(len(docs), 'document')} added", expanded=len(docs) <= 10):
            shown = docs
            if len(docs) > 10:
                keyword = st.text_input(
                    "Search documents", placeholder="Search by name...", label_visibility="collapsed"
                )
                shown = [d for d in docs if keyword.lower() in doc_name(d).lower()]
            for doc in shown:
                ext = doc["file"].rsplit(".", 1)[-1]
                size_info = plural(doc["pages"], "page") if doc["pages"] else plural(doc["chunks"], "chunk")
                left, right = st.columns([3, 1], vertical_alignment="center")  # đủ rộng cho icon thùng rác
                left.markdown(
                    f"{FILE_ICONS.get(ext, '📄')} **{doc_name(doc)}**  \n:gray[{human_size(doc['size'])} · {size_info}]"
                )
                with right.popover(":material/delete:", help="Remove document"):
                    st.markdown(f"Remove **{doc_name(doc)}**?")
                    if st.button("Remove", key=f"delete_{doc['id']}", type="primary"):
                        pipeline.remove(doc["id"])
                        st.rerun()

    scope = []
    if docs:
        st.divider()
        names = {d["id"]: doc_name(d) for d in docs}
        scope = st.multiselect(
            "Search only in",
            options=list(names),
            format_func=names.get,
            placeholder="All documents",
            help="Leave empty to search every document.",
        )

    st.divider()
    if st.session_state.messages and st.button("Clear conversation", icon=":material/delete_sweep:"):
        st.session_state.messages = []
        for key in [k for k in st.session_state if k.startswith("feedback_")]:
            del st.session_state[key]  # otherwise a new answer at the same position inherits the old rating
        st.rerun()
    st.caption(f"{config.PROVIDER} · LLM `{config.LLM_MODEL}` · Embedding `{config.EMBED_MODEL}`")

# ---------- Chat ----------
# The chat input always sits at the bottom; read it first to know whether to show the suggestions
typed = st.chat_input("Ask a question about your documents..." if docs else "Add documents first", disabled=not docs)
question = typed or st.session_state.pop("pending_question", None)

for i, msg in enumerate(st.session_state.messages):
    show_message(msg, i)

if not docs:
    # Nút nằm giữa trang chứ không ở thanh bên: màn hình hẹp tự thu gọn thanh bên, người mới sẽ không thấy nút
    st.markdown(
        "#### Get started\n"
        "1. Upload PDF, DOCX, TXT or Markdown files in the sidebar, or try the sample documents.\n"
        "2. Ask a question below.\n"
        "3. Expand a source under the answer to read the passage it came from."
    )
    if SAMPLE_DOCS.is_dir() and st.button("Load sample documents", icon=":material/library_add:", type="primary"):
        ingest_files([(p.name, p.read_bytes()) for p in sorted(SAMPLE_DOCS.iterdir())])
elif scope:
    st.caption(f"🔎 Searching only the {len(scope)} document(s) selected in the sidebar.")

if docs and not st.session_state.messages and not question:
    st.markdown("**Try asking:**")
    for i, example in enumerate(EXAMPLE_QUESTIONS):  # one per line so the full question is readable
        if st.button(example, key=f"example_{i}", icon=":material/chat_bubble_outline:"):
            st.session_state.pending_question = example
            st.rerun()

if question:
    # Earlier turns let a follow-up such as "and in Da Nang?" be rewritten into a complete question
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if not m.get("error")]
    st.session_state.messages.append({"role": "user", "content": question})
    show_message(st.session_state.messages[-1], len(st.session_state.messages) - 1)

    with st.chat_message("assistant"):
        started = time.perf_counter()
        try:
            with st.spinner("Searching your documents..."):
                query = pipeline.standalone_question(question, history)
                hits = pipeline.retrieve(query, scope or None)
                answer = st.write_stream(pipeline.generate(query, hits))
            sources = pipeline.cited_sources(answer, hits)
            meta = f"{plural(len(sources), 'source')} · {time.perf_counter() - started:.1f} s"
            reply = {"content": answer, "sources": sources, "meta": meta, "question": question}
            if query != question:
                reply["query"] = query
        except Exception as e:
            reply = {"content": f"Could not get an answer: {friendly_error(e)}", "error": True}
    st.session_state.messages.append({"role": "assistant"} | reply)
    st.rerun()  # redraw from history so the new answer also gets its sources and 👍/👎
