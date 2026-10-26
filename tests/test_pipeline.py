import pymupdf
import pytest

from rag import config, llm, pipeline, store

SAMPLE_DOCS = config.BASE_DIR / "sample_docs"


@pytest.fixture
def docs():
    loaded = {f.name: pipeline.ingest(f.name, f.read_bytes()) for f in sorted(SAMPLE_DOCS.iterdir())}
    yield loaded
    for doc in store.list_documents():
        pipeline.remove(doc["id"])


def test_ingest_registers_documents(docs):
    pdf = docs["Huong_dan_su_dung.pdf"]
    assert pdf["pages"] == 4 and pdf["chunks"] > 0
    assert len(store.list_documents()) == 3


def test_same_content_is_not_ingested_twice(docs):
    again = pipeline.ingest("ten_khac.pdf", (SAMPLE_DOCS / "Huong_dan_su_dung.pdf").read_bytes())
    assert again["id"] == docs["Huong_dan_su_dung.pdf"]["id"]
    assert len(store.list_documents()) == 3


def test_chunks_are_embedded_with_document_context(monkeypatch):
    embedded, original_embed = [], llm.embed
    monkeypatch.setattr(llm, "embed", lambda texts, kind: embedded.extend(texts) or original_embed(texts, kind))
    doc = pipeline.ingest("Bang_gia.txt", "Ngôn ngữ Anh | 11.060.000".encode())
    try:
        assert embedded == ["Bang_gia\n\nNgôn ngữ Anh | 11.060.000"]  # vector biết chunk thuộc tài liệu nào
        [hit] = [h for h in pipeline.retrieve("Ngôn ngữ Anh") if h["doc_id"] == doc["id"]]
        assert hit["text"] == "Ngôn ngữ Anh | 11.060.000"  # nội dung lưu và hiển thị không kèm tiêu đề
        assert hit["title"] == doc["title"] == "Bang_gia"  # tiêu đề có sẵn để giao diện hiển thị
    finally:
        pipeline.remove(doc["id"])


def test_select_chunks_caps_chunks_per_document():
    hits = [{"doc_id": d} for d in "aaaabbc"]
    assert [h["doc_id"] for h in pipeline.select_chunks(hits, 4, max_per_doc=0)] == list("aaaa")
    assert [h["doc_id"] for h in pipeline.select_chunks(hits, 4, max_per_doc=3)] == list("aaab")
    assert [h["doc_id"] for h in pipeline.select_chunks(hits, 4, max_per_doc=1)] == list("abc")


def test_retrieve_finds_the_right_page(docs):
    hits = pipeline.retrieve("Máy báo lỗi E2 thì phải làm gì?")
    assert pipeline.source_label(hits[0]) == "Huong_dan_su_dung.pdf, trang 3"


def test_retrieve_can_be_scoped_to_documents(docs):
    docx_id = docs["Chinh_sach_cong_ty.docx"]["id"]
    hits = pipeline.retrieve("Máy báo lỗi E2 thì phải làm gì?", [docx_id])
    assert hits and all(h["doc_id"] == docx_id for h in hits)


def test_cited_sources_ignores_out_of_range_numbers(docs):
    hits = pipeline.retrieve("lỗi E2")
    answer = "".join(pipeline.generate("lỗi E2", hits))
    assert [n for n, _ in pipeline.cited_sources(answer, hits)] == [1, 2, 3]  # [9] bị bỏ


def test_remove_deletes_chunks_and_file(docs):
    doc = docs["Chinh_sach_cong_ty.docx"]
    pipeline.remove(doc["id"])
    assert store.get_document(doc["id"]) is None
    assert not (config.FILES_DIR / doc["file"]).exists()
    assert all(h["doc_id"] != doc["id"] for h in pipeline.retrieve("bảo hành hoàn tiền"))


def test_scanned_pdf_is_rejected_and_cleaned_up():
    blank = pymupdf.open()
    blank.new_page()
    with pytest.raises(ValueError, match="OCR"):
        pipeline.ingest("scan.pdf", blank.tobytes())
    assert not list(config.FILES_DIR.iterdir())
