"""Dùng thử RAG ngay trên terminal, không cần giao diện.

python cli.py add sample_docs/Chinh_sach_cong_ty.docx
python cli.py list
python cli.py ask "Chính sách hoàn tiền như thế nào?"
"""

import argparse
import sys
from pathlib import Path

from rag import llm, pipeline, store

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Document Assistant (CLI)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("add", help="nạp tài liệu").add_argument("files", nargs="+", type=Path)
    commands.add_parser("list", help="liệt kê tài liệu đã nạp")
    ask = commands.add_parser("ask", help="đặt câu hỏi")
    ask.add_argument("question")
    ask.add_argument("--show-context", action="store_true", help="in ra các chunk tìm được")
    args = parser.parse_args()

    if args.command == "list":
        for doc in store.list_documents():
            print(f"{doc['id']}  {doc['filename']}  ({doc['chunks']} chunk)")
        return

    if problems := llm.check_ready():
        sys.exit("\n".join(problems))

    if args.command == "add":
        for path in args.files:
            doc = pipeline.ingest(path.name, path.read_bytes())
            print(f"✓ {doc['filename']}: {doc['chunks']} chunk")
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
