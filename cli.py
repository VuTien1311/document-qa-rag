"""Dùng RAG trên terminal, không cần giao diện.

python cli.py add sample_docs                     # nạp cả thư mục (hoặc từng file)
python cli.py list
python cli.py ask "Chính sách hoàn tiền như thế nào?"
python cli.py crawl @examples/fpt_education.args  # tải trang web công khai về web_docs/
"""

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from rag import llm, pipeline, store
from rag.crawler import Fetcher, crawl
from rag.loaders import SUPPORTED_TYPES

sys.stdout.reconfigure(encoding="utf-8")


def expand(paths: list[Path]) -> Iterator[Path]:
    """Thư mục được thay bằng mọi file hỗ trợ bên trong (kể cả thư mục con)."""
    for path in paths:
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.suffix.lower().lstrip(".") in SUPPORTED_TYPES)
        else:
            yield path


def main() -> None:
    # fromfile_prefix_chars: "@file.args" đọc tham số từ file, mỗi dòng một tham số
    parser = argparse.ArgumentParser(description="Document QA RAG (CLI)", fromfile_prefix_chars="@")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("add", help="nạp tài liệu (file hoặc thư mục)").add_argument("paths", nargs="+", type=Path)
    commands.add_parser("list", help="liệt kê tài liệu đã nạp")
    ask = commands.add_parser("ask", help="đặt câu hỏi")
    ask.add_argument("question")
    ask.add_argument("--show-context", action="store_true", help="in ra các chunk tìm được")
    web = commands.add_parser("crawl", help="tải trang web công khai về thành Markdown")
    web.add_argument("--sitemap", nargs="*", default=[], help="URL sitemap, lọc bằng --include/--exclude")
    web.add_argument("--url", nargs="*", default=[], help="URL trang lẻ, luôn được tải")
    web.add_argument("--include", help="regex: chỉ lấy URL sitemap khớp mẫu này")
    web.add_argument("--exclude", help="regex: bỏ URL sitemap khớp mẫu này")
    web.add_argument("--limit", type=int, help="số trang tối đa")
    web.add_argument("--delay", type=float, default=1.0, help="số giây chờ giữa 2 request (mặc định 1)")
    web.add_argument("--out", type=Path, default=Path("web_docs"), help="thư mục lưu (mặc định web_docs)")
    args = parser.parse_args()

    if args.command == "list":
        for doc in store.list_documents():
            print(f"{doc['id']}  {doc['filename']}  ({doc['chunks']} chunk)")
        return

    if args.command == "crawl":
        saved = crawl(
            args.out,
            sitemaps=args.sitemap,
            urls=args.url,
            include=args.include,
            exclude=args.exclude,
            limit=args.limit,
            fetcher=Fetcher(args.delay),
            on_page=lambda url, path, status: print(f"{'✓' if path else '✗'} {url}  {status}"),
        )
        print(f"\nĐã lưu {len(saved)} trang vào {args.out}")
        return

    if problems := llm.check_ready():
        sys.exit("\n".join(problems))

    if args.command == "add":
        for path in expand(args.paths):
            try:
                doc = pipeline.ingest(path.name, path.read_bytes())
                print(f"✓ {doc['filename']}: {doc['chunks']} chunk")
            except Exception as e:  # một file lỗi không làm dừng cả lượt nạp
                print(f"✗ {path.name}: {e}")
        return

    hits = pipeline.retrieve(args.question)
    if not hits:
        sys.exit("Chưa nạp tài liệu nào.")
    if args.show_context:
        for i, hit in enumerate(hits, start=1):
            print(f"--- [{i}] {pipeline.source_label(hit)} (độ liên quan {hit['score']:.2f})\n{hit['text']}\n")

    answer = ""
    for piece in pipeline.generate(args.question, hits):
        print(piece, end="", flush=True)
        answer += piece
    print("\n\nNguồn:")
    for n, hit in pipeline.cited_sources(answer, hits):
        print(f"  [{n}] {pipeline.source_label(hit)}  (độ liên quan {hit['score']:.2f})")


if __name__ == "__main__":
    main()
