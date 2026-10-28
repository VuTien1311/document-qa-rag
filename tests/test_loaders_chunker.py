import unicodedata
from itertools import pairwise

import pytest

from rag.chunker import split_text
from rag.config import BASE_DIR
from rag.loaders import load_document

SAMPLE_DOCS = BASE_DIR / "sample_docs"


def test_pdf_split_by_page():
    doc = load_document(SAMPLE_DOCS / "Huong_dan_su_dung.pdf")
    assert doc.pages == 4
    sections = doc.sections
    assert sections[2].location == "page 3"
    assert "Vệ sinh và xử lý lỗi" in sections[2].text
    assert "\xa0" not in sections[2].text


def test_docx_split_by_heading_and_reads_tables():
    doc = load_document(SAMPLE_DOCS / "Chinh_sach_cong_ty.docx")
    assert doc.pages is None
    sections = doc.sections
    assert [s.location for s in sections][1:] == [
        "section 1. Chính sách hoàn tiền",
        "section 2. Chính sách bảo hành",
        "section 3. Kênh hỗ trợ",
    ]
    assert "Bộ lọc HEPA thay thế | 6 tháng" in sections[2].text


def test_text_is_normalized_to_nfc(tmp_path):
    path = tmp_path / "nfd.txt"
    path.write_text(unicodedata.normalize("NFD", "Bảo hành"), encoding="utf-8")
    [section] = load_document(path).sections
    assert section.text == unicodedata.normalize("NFC", "Bảo hành")


def test_unsupported_type(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(path)


def test_chunks_respect_size_and_keep_all_content():
    text = "\n\n".join(f"Đoạn {i}. " + " ".join(f"câu {i}.{j} nói về RAG." for j in range(15)) for i in range(20))
    chunks = split_text(text, 1000, 200)
    assert all(len(c) <= 1000 for c in chunks)
    assert all(f"Đoạn {i}." in "".join(chunks) for i in range(20))
    # chunk sau lặp lại phần đuôi chunk trước
    assert all(set(a.split()) & set(b.split()) for a, b in pairwise(chunks))


def test_chunk_edge_cases():
    assert split_text("ngắn thôi", 1000, 200) == ["ngắn thôi"]
    assert split_text("x" * 2500, 1000, 200) == ["x" * 1000, "x" * 1000, "x" * 500]
