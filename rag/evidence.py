"""Giúp người dùng kiểm chứng câu trả lời: tìm câu bằng chứng trong đoạn trích và dựng ảnh trang PDF có tô sáng."""

import re
import unicodedata
from pathlib import Path

import pymupdf

PASSAGE_COLOR = (1.0, 0.93, 0.55)  # vàng nhạt: cả đoạn được trích
EVIDENCE_COLOR = (1.0, 0.65, 0.25)  # cam: câu khớp với ý trong câu trả lời
_STOPWORDS = {
    "và", "là", "của", "các", "có", "cho", "được", "trong", "với", "này", "những", "một", "theo", "từ", "khi", "thì",
    "the", "and", "for", "are", "is", "of", "to", "in", "at", "a", "an", "on", "with", "by", "or",
}  # fmt: skip


def claim_for(answer: str, n: int) -> str:
    """Các câu/dòng của câu trả lời có trích dẫn [n] (kể cả dạng [1, 3]); không có thì trả cả câu trả lời."""
    citation = re.compile(rf"\[(?:\d+\s*,\s*)*{n}(?:\s*,\s*\d+)*\]")
    parts = re.split(r"(?<=[.!?])\s+|\n+", answer)
    return " ".join(p for p in parts if citation.search(p)) or answer


def evidence_units(excerpt: str, claim: str, max_units: int = 3) -> list[str]:
    """Những câu (hoặc dòng bảng) của đoạn trích khớp nhiều nhất với ý được trích dẫn.

    Con số và mã ("22.120.000", "E2", "7480201") khớp nặng gấp 3 lần chữ khớp: chúng là bằng chứng mạnh hơn chữ
    "học phí", và giống nhau ở mọi ngôn ngữ nên vẫn tìm được bằng chứng khi trả lời tiếng Anh từ tài liệu tiếng Việt.
    Chỉ lấy câu có điểm từ 3 trở lên (ít nhất một con số/mã, hoặc ba chữ) để không tô nhầm.
    """
    claim_tokens = _tokens(claim)
    numbers = {t for t in claim_tokens if any(c.isdigit() for c in t)}
    scored = []
    for unit in _units(excerpt):
        tokens = _tokens(unit)
        score = 3 * len(tokens & numbers) + len(tokens & (claim_tokens - numbers))
        if score >= 3:
            scored.append((score, unit))
    return [unit for _, unit in sorted(scored, key=lambda s: s[0], reverse=True)[:max_units]]


def render_pdf_page(path: Path, page_number: int, passage: str, evidence: list[str], zoom: float = 1.6) -> bytes:
    """Ảnh PNG của một trang PDF: dòng thuộc đoạn được trích tô vàng, dòng chứa câu bằng chứng tô cam.

    Tô bằng annotation trên bản mở trong bộ nhớ, không ghi lại file gốc.
    """
    with pymupdf.open(path) as pdf:
        page = pdf[page_number - 1]
        lines = _page_lines(page)
        for target, color in [(passage, PASSAGE_COLOR), *((e, EVIDENCE_COLOR) for e in evidence)]:
            for rect in _lines_inside(lines, target):
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=color)
                annot.update()
        return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")


def _units(excerpt: str) -> list[str]:
    """Dòng bảng giữ nguyên; dòng văn bản dài tách thành từng câu để tô cho gọn."""
    units = []
    for line in excerpt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("|") or len(line) <= 200:
            units.append(line)
        else:
            units.extend(s for s in re.split(r"(?<=[.!?;])\s+", line) if s)
    return units


def _tokens(text: str) -> set[str]:
    words = re.findall(r"\d[\d.,]*\d|\w+", _normalize(text))
    # "22.120.000" và "22,120,000" là cùng một số
    words = [re.sub(r"[.,]", "", w) if w[0].isdigit() else w for w in words]
    return {w for w in words if w not in _STOPWORDS and (len(w) > 1 or w.isdigit())}


def _page_lines(page: pymupdf.Page) -> list[tuple[str, pymupdf.Rect]]:
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = _normalize("".join(span["text"] for span in line["spans"]))
            if len(text) >= 4:  # bỏ dòng quá ngắn ("-", "E1:") vì khớp nhầm ở khắp nơi
                lines.append((text, pymupdf.Rect(line["bbox"])))
    return lines


def _lines_inside(lines: list[tuple[str, pymupdf.Rect]], target: str) -> list[pymupdf.Rect]:
    target = _normalize(target)
    return [rect for text, rect in lines if text in target]


def _normalize(text: str) -> str:
    """Chuẩn hoá giống loaders (NFC, bỏ dấu cách không ngắt và gạch nối mềm) để chữ trên trang khớp với đoạn trích."""
    text = unicodedata.normalize("NFC", text).replace("\xa0", " ").replace("\xad", "").lower()
    return re.sub(r"\s+", " ", text).strip()
