"""Đánh giá chất lượng RAG trên bộ câu hỏi có nhãn (JSONL, mỗi dòng một câu).

Mỗi câu: {"id", "type": "fact" | "list" | "negative", "question",
          "relevant": [URL hoặc tên file chứa đáp án], "expected": [ý phải có trong câu trả lời; "a|b" = a hoặc b]}

Phần tìm kiếm (chỉ tốn lượt gọi embedding):
    Hit@K     tỉ lệ câu có ít nhất một nguồn đúng trong K chunk đưa cho LLM
    MRR       trung bình 1/hạng của chunk đúng đầu tiên (0 nếu không có)
    Recall@K  tỉ lệ nguồn đúng xuất hiện trong K chunk, quan trọng với câu hỏi liệt kê
Phần câu trả lời (mỗi câu một lượt gọi LLM):
    Coverage  tỉ lệ ý mong đợi xuất hiện trong câu trả lời
    Refusal   câu hỏi ngoài dữ liệu được trả lời "không tìm thấy" thay vì bịa
"""

import json
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import TypeVar

T = TypeVar("T")
QUESTION_TYPES = ("fact", "list", "negative")
METRICS = ("hit", "rr", "recall", "coverage", "refusal")
REFUSAL_PHRASE = "không tìm thấy"


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_of(hit: dict) -> str:
    """Định danh nguồn của một chunk, cùng dạng với nhãn: URL với trang web, tên file với tài liệu upload."""
    return hit["location"] if hit["location"].startswith("http") else hit["filename"]


def retrieval_scores(sources: list[str], relevant: list[str]) -> dict[str, float]:
    """Điểm tìm kiếm của một câu; `sources` là nguồn của các chunk theo thứ tự xếp hạng."""
    rank = next((i for i, source in enumerate(sources, start=1) if source in relevant), None)
    return {
        "hit": float(rank is not None),
        "rr": 1 / rank if rank else 0.0,
        "recall": len(set(sources) & set(relevant)) / len(relevant),
    }


def coverage(answer: str, expected: list[str]) -> float:
    """Tỉ lệ ý mong đợi có trong câu trả lời, không phân biệt hoa thường; "a|b" khớp a hoặc b."""
    text = _normalize(answer)
    return mean(any(_normalize(option) in text for option in item.split("|")) for item in expected) if expected else 1.0


def is_refusal(answer: str) -> bool:
    return REFUSAL_PHRASE in _normalize(answer)


def evaluate(
    questions: list[dict],
    retrieve: Callable[[str], list[dict]],
    answer: Callable[[str, list[dict]], str] | None = None,
    on_result: Callable[[dict], None] | None = None,
    attempts: int = 3,
    retry_delay: float = 2.0,
) -> list[dict]:
    """Kết quả từng câu (để biết câu nào sai): điểm tìm kiếm, và điểm câu trả lời nếu truyền `answer`.

    Một câu lỗi (mạng chập chờn, hết hạn mức) được thử lại `attempts` lần; vẫn lỗi thì ghi vào `error`
    và chấm tiếp các câu khác, thay vì làm hỏng cả lượt chạy.
    """
    results = []
    for q in questions:
        row = {"id": q["id"], "type": q["type"]}
        try:
            hits = _retry(lambda q=q: retrieve(q["question"]), attempts, retry_delay)
            row["sources"] = [source_of(h) for h in hits]
            if q["relevant"]:
                row |= retrieval_scores(row["sources"], q["relevant"])
            if answer:
                row["answer"] = _retry(lambda q=q, hits=hits: answer(q["question"], hits), attempts, retry_delay)
                if q["type"] == "negative":
                    row["refusal"] = float(is_refusal(row["answer"]))
                else:
                    row["coverage"] = coverage(row["answer"], q["expected"])
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:300]
        results.append(row)
        if on_result:
            on_result(row)
    return results


def failures(results: list[dict]) -> list[tuple[str, str]]:
    """Các câu chưa đạt kèm lý do, để biết cần cải thiện ở đâu."""
    reasons = []
    for r in results:
        if "error" in r:
            reasons.append((r["id"], f"lỗi khi chạy, không chấm được: {r['error'][:120]}"))
        elif r.get("hit") == 0.0:
            reasons.append((r["id"], "không tìm được nguồn đúng"))
        elif r.get("coverage", 1.0) < 1.0:
            reasons.append((r["id"], f"câu trả lời thiếu ý (coverage {r['coverage']:.0%})"))
        elif r.get("refusal") == 0.0:
            reasons.append((r["id"], "trả lời câu ngoài dữ liệu thay vì từ chối"))
    return reasons


def summarize(results: list[dict]) -> dict[str, dict[str, float]]:
    """Trung bình từng chỉ số theo loại câu hỏi và trên toàn bộ."""
    groups = {"all": results} | {t: [r for r in results if r["type"] == t] for t in QUESTION_TYPES}
    return {
        name: {"n": len(rows)} | {m: mean(r[m] for r in rows if m in r) for m in METRICS if any(m in r for r in rows)}
        for name, rows in groups.items()
        if rows
    }


def _retry(call: Callable[[], T], attempts: int, delay: float) -> T:
    """Gọi lại khi lỗi, chờ delay, 2*delay, 4*delay... giữa các lần."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * 2**attempt)
    raise ValueError("attempts phải >= 1")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()
