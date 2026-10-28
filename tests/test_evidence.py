import pymupdf

from rag.config import BASE_DIR
from rag.evidence import claim_for, evidence_units, render_pdf_page
from rag.loaders import load_document

TUITION = """# Học phí tại Campus Hà Nội
| Ngành | KV1 | Các KV khác |
|---|---|---|
| Ngôn ngữ Anh | 15.480.000 | 22.120.000 |
| Trí tuệ nhân tạo | 22.120.000 | 31.600.000 |"""


def test_claim_for_picks_the_sentences_citing_that_source():
    answer = "Học phí ngành AI là 22.120.000 (KV1) [1]. Ngành Luật rẻ hơn [2]. Cả hai đều thu theo kỳ [1, 2]."
    assert claim_for(answer, 1) == "Học phí ngành AI là 22.120.000 (KV1) [1]. Cả hai đều thu theo kỳ [1, 2]."
    assert claim_for(answer, 3) == answer  # không có [3] thì dùng cả câu trả lời


def test_evidence_prefers_rows_sharing_numbers_with_the_claim():
    claim = "Học phí ngành Trí tuệ nhân tạo là 22,120,000 cho KV1 và 31.600.000 cho khu vực khác [1]."
    assert evidence_units(TUITION, claim)[0] == "| Trí tuệ nhân tạo | 22.120.000 | 31.600.000 |"
    assert evidence_units(TUITION, "Trường có ký túc xá [1].") == []  # không đủ khớp thì không tô gì


def test_codes_find_evidence_across_languages():
    excerpt = "Bảng mã lỗi:\nE1: Bộ lọc chưa được lắp đúng.\nE2: Cảm biến bụi bị bẩn."
    assert evidence_units(excerpt, "Error E2 means the dust sensor is dirty [1].") == ["E2: Cảm biến bụi bị bẩn."]


def test_render_pdf_page_highlights_the_cited_lines():
    path = BASE_DIR / "sample_docs" / "Huong_dan_su_dung.pdf"
    [*_, page3, _] = load_document(path).sections
    evidence = [u for u in page3.text.splitlines() if "E2" in u]
    png = render_pdf_page(path, 3, page3.text, evidence)
    assert png.startswith(b"\x89PNG")

    plain = pymupdf.open(path)[2].get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6)).tobytes("png")
    assert png != plain  # có tô sáng
    with pymupdf.open(path) as pdf:
        assert not list(pdf[2].annots())  # file gốc không bị sửa
