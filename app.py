"""Giao diện web: danh sách tài liệu ở thanh bên + khung chat có trích dẫn nguồn.

Chạy: streamlit run app.py
"""

import streamlit as st

from rag import config, llm, pipeline, store
from rag.loaders import SUPPORTED_TYPES

FILE_ICONS = {"pdf": "📕", "docx": "📘", "txt": "📄", "md": "📄"}

st.set_page_config(page_title="AI Document Assistant", page_icon="📚", layout="wide")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("notices", [])


def human_size(size: int) -> str:
    return f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"


def show_sources(sources: list) -> None:
    if not sources:
        return
    st.caption("Nguồn trích dẫn")
    for n, hit in sources:
        with st.expander(f"[{n}] {pipeline.source_label(hit)}"):
            st.caption(f"Độ liên quan: {hit['score']:.2f}")
            if hit["location"].startswith("http"):
                st.markdown(f"[Mở trang gốc]({hit['location']})")
            st.markdown(hit["text"].replace("\n", "  \n"))  # giữ nguyên xuống dòng như tài liệu gốc


st.title("📚 AI Document Assistant")
st.caption("Tải tài liệu lên rồi đặt câu hỏi. Câu trả lời luôn kèm nguồn trích dẫn.")

if not st.session_state.get("ready"):  # chỉ kiểm tra một lần mỗi phiên cho đỡ tốn request
    if problems := llm.check_ready():
        st.error("**Model AI chưa sẵn sàng**\n\n" + "\n".join(f"- {p}" for p in problems))
        st.info("Sửa xong thì tải lại trang.")
        st.stop()
    st.session_state.ready = True

# ---------- Thanh bên: tài liệu ----------
with st.sidebar:
    st.header("📁 Tài liệu của bạn")
    uploads = st.file_uploader(
        "Tải lên PDF, DOCX, TXT",
        type=list(SUPPORTED_TYPES),
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploads:
        for file in uploads:
            progress = st.progress(0.0, text=f"Đang nạp {file.name}...")
            try:
                pipeline.ingest(file.name, file.getvalue(), on_progress=progress.progress)
            except Exception as e:
                st.session_state.notices.append(f"Không nạp được **{file.name}**: {e}")
        st.session_state.uploader_key += 1  # đổi key để làm trống ô upload sau khi nạp xong
        st.rerun()

    for notice in st.session_state.notices:
        st.warning(notice)
    st.session_state.notices = []

    docs = store.list_documents()
    if not docs:
        st.caption("Chưa có tài liệu nào. Tải lên để bắt đầu hỏi đáp.")
    else:
        # Nhiều tài liệu (ví dụ cả site đã crawl) thì thu gọn để thanh bên không quá dài
        with st.expander(f"Đã nạp {len(docs)} tài liệu", expanded=len(docs) <= 10):
            for doc in docs:
                ext = doc["file"].rsplit(".", 1)[-1]
                size_info = f"{doc['pages']} trang" if doc["pages"] else f"{doc['chunks']} đoạn"
                left, right = st.columns([5, 1], vertical_alignment="center")
                left.markdown(
                    f"{FILE_ICONS.get(ext, '📄')} **{doc['filename']}**  \n"
                    f":gray[{human_size(doc['size'])} · {size_info}]"
                )
                if right.button(":material/delete:", key=f"delete_{doc['id']}", help="Xoá tài liệu", type="tertiary"):
                    pipeline.remove(doc["id"])
                    st.rerun()

    scope = []
    if docs:
        st.divider()
        names = {d["id"]: d["filename"] for d in docs}
        scope = st.multiselect(
            "Chỉ hỏi trong", options=list(names), format_func=names.get, placeholder="Tất cả tài liệu"
        )

    st.divider()
    if st.button("Xoá cuộc trò chuyện"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"{config.PROVIDER} · LLM `{config.LLM_MODEL}` · Embedding `{config.EMBED_MODEL}`")

# ---------- Khung chat ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources"))

if question := st.chat_input("Đặt câu hỏi về tài liệu...", disabled=not docs):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sources = []
        try:
            with st.spinner("Đang tìm trong tài liệu..."):
                hits = pipeline.retrieve(question, scope or None)
                answer = st.write_stream(pipeline.generate(question, hits))
            sources = pipeline.cited_sources(answer, hits)
            show_sources(sources)
        except Exception as e:
            answer = f"⚠️ Lỗi khi gọi model: {e}"
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
