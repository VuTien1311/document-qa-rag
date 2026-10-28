from rag import keyword_search, pipeline
from rag.config import BASE_DIR

SAMPLE_DOCS = BASE_DIR / "sample_docs"


def test_tokenize_keeps_numbers_whole():
    assert keyword_search.tokenize("Học phí 22.120.000 VNĐ, mã 7480201") == [
        "học",
        "phí",
        "22.120.000",
        "vnđ",
        "mã",
        "7480201",
    ]


def test_keyword_search_finds_exact_codes_and_respects_scope():
    docs = {f.name: pipeline.ingest(f.name, f.read_bytes()) for f in sorted(SAMPLE_DOCS.iterdir())}
    try:
        [top, *_] = keyword_search.search("lỗi E2", 5)
        assert top["location"] == "page 3" and "E2" in top["text"]

        docx_id = docs["Chinh_sach_cong_ty.docx"]["id"]
        assert all(h["doc_id"] == docx_id for h in keyword_search.search("bảo hành", 5, [docx_id]))
        assert keyword_search.search("từkhoákhôngtồntại", 5) == []
    finally:
        for doc in docs.values():
            pipeline.remove(doc["id"])
    assert keyword_search.search("lỗi E2", 5) == []  # chỉ mục được dựng lại sau khi xoá


def test_reciprocal_rank_fusion():
    vector = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}, {"id": "c", "score": 0.7}]
    keyword = [{"id": "c", "score": None}, {"id": "d", "score": None}]
    fused = pipeline.reciprocal_rank_fusion(vector, keyword)
    assert [h["id"] for h in fused][:1] == ["c"]  # có mặt ở cả hai bảng nên đứng đầu
    assert {h["id"] for h in fused} == {"a", "b", "c", "d"}
    assert next(h for h in fused if h["id"] == "c")["score"] == 0.7  # giữ điểm cosine để hiển thị
