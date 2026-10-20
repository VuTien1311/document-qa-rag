"""Chia văn bản thành các chunk khoảng chunk_size ký tự, chunk sau lặp lại phần đuôi chunk trước.

Ưu tiên cắt ở ranh giới lớn trước: đoạn văn > dòng > câu > từ, để mỗi chunk giữ trọn ý.
"""

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    pieces = _split(text, chunk_size, SEPARATORS)
    return _merge(pieces, chunk_size, overlap)


def _split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Tách text thành các mảnh không dài quá chunk_size."""
    if len(text) <= chunk_size:
        return [text]
    sep, *rest = separators
    if sep == "":  # hết cách cắt "đẹp": cắt cứng theo số ký tự
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split(text, chunk_size, rest)

    pieces = []
    for i, part in enumerate(parts):
        piece = part + sep if i < len(parts) - 1 else part  # giữ lại dấu phân cách
        if piece.strip():
            pieces.extend(_split(piece, chunk_size, rest))
    return pieces


def _merge(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Ghép các mảnh liền nhau thành chunk; giữ lại tối đa `overlap` ký tự cuối làm phần chồng lấn."""
    chunks, window, length = [], [], 0
    for piece in pieces:
        if window and length + len(piece) > chunk_size:
            chunks.append("".join(window).strip())
            while window and (length > overlap or length + len(piece) > chunk_size):
                length -= len(window.pop(0))
        window.append(piece)
        length += len(piece)
    if window:
        chunks.append("".join(window).strip())
    return [c for c in chunks if c]
