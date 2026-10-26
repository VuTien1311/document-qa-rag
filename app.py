"""Giao diện web: danh sách tài liệu ở thanh bên + khung chat có trích dẫn nguồn.

Chạy: streamlit run app.py
"""

import re
import time

import streamlit as st

from rag import config, llm, pipeline, store
from rag.loaders import SUPPORTED_TYPES

FILE_ICONS = {"pdf": "📕", "docx": "📘", "txt": "📄", "md": "📄"}
# Câu hỏi gợi ý cho dữ liệu demo (sample_docs + trang tuyển sinh FPT); đổi theo tài liệu của bạn
EXAMPLE_QUESTIONS = [
    "Học phí ngành Trí tuệ nhân tạo ở campus Hà Nội là bao nhiêu?",
    "Trường có những loại học bổng nào?",
    "Máy lọc không khí báo lỗi E2 thì phải làm gì?",
]

st.set_page_config(page_title="Document QA", page_icon="📚", layout="wide")
for key, default in {"messages": [], "uploader_key": 0, "notices": []}.items():
    st.session_state.setdefault(key, default)


def human_size(size: int) -> str:
    return f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"


def doc_name(doc: dict) -> str:
    """Tiêu đề dễ đọc ("Học phí tại Campus Quy Nhơn") thay cho tên file nếu có."""
    return doc.get("title") or doc["filename"]


def friendly_error(error: Exception) -> str:
    """Giải thích các lỗi hay gặp với gói free thay vì in nguyên thông báo kỹ thuật."""
    text = str(error)
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return (
            "Đã hết hạn mức miễn phí của Gemini (theo phút hoặc theo ngày). Đợi một lát rồi thử lại; "
            "hạn mức theo ngày được làm mới lúc nửa đêm giờ Mỹ (khoảng 14–15h giờ Việt Nam)."
        )
    if "API_KEY_INVALID" in text or "API key not valid" in text:
        return "API key không hợp lệ. Kiểm tra `GEMINI_API_KEY` trong file `.env`."
    return f"Lỗi khi gọi model: {text[:300]}"


def excerpt_markdown(text: str) -> str:
    """Đoạn trích hiển thị gọn: tiêu đề markdown (# ...) thành chữ đậm thay vì chữ to, giữ nguyên xuống dòng."""
    text = re.sub(r"^#{1,6}\s*(.+)$", r"**\1**", text, flags=re.MULTILINE)
    return text.replace("\n", "  \n")


def show_sources(sources: list) -> None:
    for n, hit in sources:
        location = hit["location"]
        label = f"[{n}] {doc_name(hit)}"
        if location and not location.startswith("http"):
            label += f" · {location}"
        with st.expander(label):
            if location.startswith("http"):
                st.markdown(f"[{location}]({location})")
            st.caption(f"Độ liên quan {hit['score']:.2f}")
            st.markdown(excerpt_markdown(hit["text"]))


def show_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources", []))
        if msg.get("meta"):
            st.caption(msg["meta"])


st.title("📚 Document QA")
st.caption("Hỏi đáp trên tài liệu của bạn. Mỗi câu trả lời kèm nguồn trích dẫn để tự kiểm chứng.")

if not st.session_state.get("ready"):  # chỉ kiểm tra một lần mỗi phiên cho đỡ tốn request
    with st.spinner("Đang kết nối tới model..."):
        problems = llm.check_ready()
    if problems:
        st.error("**Model AI chưa sẵn sàng**\n\n" + "\n".join(f"- {p}" for p in problems))
        st.info("Sửa xong thì tải lại trang.")
        st.stop()
    st.session_state.ready = True

# ---------- Thanh bên: tài liệu ----------
with st.sidebar:
    st.header("📁 Tài liệu")
    uploads = st.file_uploader(
        "Tải lên PDF, DOCX, TXT, MD",
        type=list(SUPPORTED_TYPES),
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        help="Mỗi file được chia thành đoạn nhỏ và tạo vector để tìm kiếm. File trùng nội dung sẽ được bỏ qua.",
    )
    if uploads:
        for file in uploads:
            progress = st.progress(0.0, text=f"Đang nạp {file.name}...")
            try:
                pipeline.ingest(file.name, file.getvalue(), on_progress=progress.progress)
            except Exception as e:
                st.session_state.notices.append(f"Không nạp được **{file.name}**: {friendly_error(e)}")
        st.session_state.uploader_key += 1  # đổi key để làm trống ô upload sau khi nạp xong
        st.rerun()

    for notice in st.session_state.notices:
        st.warning(notice)
    st.session_state.notices = []

    docs = store.list_documents()
    if not docs:
        st.info("Chưa có tài liệu nào. Tải lên ở ô phía trên để bắt đầu hỏi đáp.")
    else:
        # Nhiều tài liệu (ví dụ cả site đã crawl) thì thu gọn và cho tìm theo tên
        with st.expander(f"Đã nạp {len(docs)} tài liệu", expanded=len(docs) <= 10):
            shown = docs
            if len(docs) > 10:
                keyword = st.text_input("Tìm tài liệu", placeholder="Tìm theo tên...", label_visibility="collapsed")
                shown = [d for d in docs if keyword.lower() in doc_name(d).lower()]
            for doc in shown:
                ext = doc["file"].rsplit(".", 1)[-1]
                size_info = f"{doc['pages']} trang" if doc["pages"] else f"{doc['chunks']} đoạn"
                left, right = st.columns([5, 1], vertical_alignment="center")
                left.markdown(
                    f"{FILE_ICONS.get(ext, '📄')} **{doc_name(doc)}**  \n:gray[{human_size(doc['size'])} · {size_info}]"
                )
                with right.popover(":material/delete:", help="Xoá tài liệu"):
                    st.markdown(f"Xoá **{doc_name(doc)}** khỏi hệ thống?")
                    if st.button("Xoá", key=f"delete_{doc['id']}", type="primary"):
                        pipeline.remove(doc["id"])
                        st.rerun()

    scope = []
    if docs:
        st.divider()
        names = {d["id"]: doc_name(d) for d in docs}
        scope = st.multiselect(
            "Chỉ hỏi trong",
            options=list(names),
            format_func=names.get,
            placeholder="Tất cả tài liệu",
            help="Để trống thì tìm trong mọi tài liệu đã nạp.",
        )

    st.divider()
    if st.session_state.messages and st.button("Xoá cuộc trò chuyện", icon=":material/delete_sweep:"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"{config.PROVIDER} · LLM `{config.LLM_MODEL}` · Embedding `{config.EMBED_MODEL}`")

# ---------- Khung chat ----------
# Ô chat luôn nằm cuối trang dù gọi ở đâu; đọc câu hỏi trước để biết có cần hiện câu hỏi gợi ý không
typed = st.chat_input(
    "Đặt câu hỏi về tài liệu..." if docs else "Tải tài liệu lên trước để đặt câu hỏi", disabled=not docs
)
question = typed or st.session_state.pop("pending_question", None)

for msg in st.session_state.messages:
    show_message(msg)

if scope:
    st.caption(f"🔎 Đang chỉ tìm trong {len(scope)} tài liệu đã chọn ở thanh bên.")

if docs and not st.session_state.messages and not question:
    st.markdown("**Thử hỏi:**")
    for i, example in enumerate(EXAMPLE_QUESTIONS):  # mỗi câu một dòng để đọc được trọn câu
        if st.button(example, key=f"example_{i}", icon=":material/chat_bubble_outline:"):
            st.session_state.pending_question = example
            st.rerun()

if question:
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    show_message(user_msg)

    with st.chat_message("assistant"):
        started = time.perf_counter()
        sources, meta = [], ""
        try:
            with st.spinner("Đang tìm trong tài liệu..."):
                hits = pipeline.retrieve(question, scope or None)
                answer = st.write_stream(pipeline.generate(question, hits))
            sources = pipeline.cited_sources(answer, hits)
            show_sources(sources)
            meta = f"{len(sources)} nguồn · {time.perf_counter() - started:.1f} giây"
            st.caption(meta)
        except Exception as e:
            answer = f"⚠️ {friendly_error(e)}"
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "meta": meta})
