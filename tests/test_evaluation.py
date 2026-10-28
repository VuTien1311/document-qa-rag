import pytest

from rag.evaluation import coverage, evaluate, failures, is_refusal, retrieval_scores, source_of, summarize

QUESTIONS = [
    {"id": "f1", "type": "fact", "question": "học phí?", "relevant": ["https://a/"], "expected": ["31.600.000"]},
    {
        "id": "l1",
        "type": "list",
        "question": "học bổng?",
        "relevant": ["https://b/", "https://c/"],
        "expected": ["X", "Y"],
    },
    {"id": "n1", "type": "negative", "question": "ngành Y?", "relevant": [], "expected": []},
]


def hit(url: str) -> dict:
    return {"location": url, "filename": "page.md", "text": "..."}


def test_source_of_uses_url_for_web_pages_and_filename_otherwise():
    assert source_of(hit("https://a/")) == "https://a/"
    assert source_of({"location": "page 3", "filename": "doc.pdf"}) == "doc.pdf"


def test_retrieval_scores():
    assert retrieval_scores(["x", "https://a/", "https://a/"], ["https://a/"]) == {"hit": 1.0, "rr": 0.5, "recall": 1.0}
    assert retrieval_scores(["https://b/", "x"], ["https://b/", "https://c/"]) == {"hit": 1.0, "rr": 1.0, "recall": 0.5}
    assert retrieval_scores(["x", "y"], ["https://a/"]) == {"hit": 0.0, "rr": 0.0, "recall": 0.0}


def test_coverage_is_case_insensitive_and_supports_alternatives():
    assert coverage("Có học bổng TINH HOA mở lối", ["Tinh hoa mở lối", "Chiến binh"]) == 0.5
    assert coverage("GPA từ 7,0 trở lên", ["7.0|7,0"]) == 1.0


def test_refusal():
    assert is_refusal("Mình KHÔNG TÌM THẤY thông tin này trong tài liệu.")
    assert is_refusal("I could not find this information in the documents.")
    assert not is_refusal("Có, trường đào tạo ngành Y.")


def test_evaluate_and_summarize():
    retrieved = {"học phí?": ["https://a/"], "học bổng?": ["https://b/", "https://x/"], "ngành Y?": ["https://x/"]}
    answers = {"học phí?": "31.600.000 đồng [1].", "học bổng?": "Có X", "ngành Y?": "Mình không tìm thấy."}
    results = evaluate(
        QUESTIONS,
        retrieve=lambda q: [hit(u) for u in retrieved[q]],
        answer=lambda q, hits: answers[q],
    )
    summary = summarize(results)
    assert summary["fact"] == {"n": 1, "hit": 1.0, "rr": 1.0, "recall": 1.0, "coverage": 1.0, "cited": 1.0}
    assert summary["list"] == {"n": 1, "hit": 1.0, "rr": 1.0, "recall": 0.5, "coverage": 0.5, "cited": 0.0}
    assert summary["negative"] == {"n": 1, "refusal": 1.0}
    assert summary["all"]["n"] == 3 and summary["all"]["hit"] == 1.0


def test_evaluate_retries_transient_errors_and_survives_permanent_ones():
    calls = {"học phí?": 0}

    def flaky_answer(question, hits):
        if question == "ngành Y?":
            raise ConnectionError("peer closed connection")  # lỗi mãi
        calls[question] = calls.get(question, 0) + 1
        if calls[question] == 1:
            raise ConnectionError("peer closed connection")  # lỗi lần đầu, lần sau được
        return "31.600.000 đồng"

    results = evaluate(QUESTIONS, retrieve=lambda q: [hit("https://a/")], answer=flaky_answer, retry_delay=0)
    by_id = {r["id"]: r for r in results}
    assert by_id["f1"]["coverage"] == 1.0 and calls["học phí?"] == 2
    assert "error" in by_id["n1"] and "refusal" not in by_id["n1"]
    assert len(results) == 3  # câu lỗi không làm dừng các câu khác
    assert ("n1", "error, not scored: ConnectionError: peer closed connection") in failures(results)


def test_evaluate_without_answers_skips_llm():
    results = evaluate(QUESTIONS[:1], retrieve=lambda q: [hit("https://z/")])
    assert "answer" not in results[0] and results[0]["hit"] == 0.0


@pytest.mark.parametrize("bad", ["", "   "])
def test_coverage_of_empty_answer(bad):
    assert coverage(bad, ["X"]) == 0.0
