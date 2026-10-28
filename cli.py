"""Command-line interface.

python cli.py add sample_docs                          # add a folder (or individual files)
python cli.py list
python cli.py ask "Chính sách hoàn tiền như thế nào?"
python cli.py crawl @examples/fpt_education.args       # download public web pages into web_docs/
python cli.py eval eval/fpt_questions.jsonl --answers  # score quality on a labeled question set
"""

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from rag import config, evaluation, llm, pipeline, store
from rag.crawler import Fetcher, crawl
from rag.loaders import SUPPORTED_TYPES

sys.stdout.reconfigure(encoding="utf-8")


def expand(paths: list[Path]) -> Iterator[Path]:
    """Folders are replaced by every supported file inside them, including subfolders."""
    for path in paths:
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.suffix.lower().lstrip(".") in SUPPORTED_TYPES)
        else:
            yield path


def run_eval(questions_path: Path, with_answers: bool, types: list[str] | None, out: Path | None) -> None:
    questions = [q for q in evaluation.load_questions(questions_path) if not types or q["type"] in types]
    answer = (lambda question, hits: "".join(pipeline.generate(question, hits))) if with_answers else None
    results = evaluation.evaluate(
        questions,
        pipeline.retrieve,
        answer,
        on_result=lambda r: print(r["id"], end=" ", flush=True),  # progress
    )

    k = config.TOP_K
    headers = ["type", "n", f"Hit@{k}", "MRR", f"Recall@{k}", "Coverage", "Cited", "Refusal"]
    print("\n\n" + "".join(f"{h:>11}" for h in headers))
    for name, scores in evaluation.summarize(results).items():
        cells = [name, scores["n"], *(f"{scores[m]:.2f}" if m in scores else "-" for m in evaluation.METRICS)]
        print("".join(f"{c:>11}" for c in cells))

    if misses := evaluation.failures(results):
        print("\nNot passed:")
        for question_id, reason in misses:
            print(f"  {question_id}: {reason}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nPer-question results: {out}")


def main() -> None:
    # fromfile_prefix_chars: "@file.args" reads arguments from a file, one per line
    parser = argparse.ArgumentParser(description="Document QA RAG (CLI)", fromfile_prefix_chars="@")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("add", help="add documents (files or folders)").add_argument("paths", nargs="+", type=Path)
    commands.add_parser("list", help="list added documents")
    ask = commands.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--show-context", action="store_true", help="print the retrieved chunks")
    web = commands.add_parser("crawl", help="download public web pages as Markdown")
    web.add_argument("--sitemap", nargs="*", default=[], help="sitemap URLs, filtered by --include/--exclude")
    web.add_argument("--url", nargs="*", default=[], help="individual page URLs, always downloaded")
    web.add_argument("--include", help="regex: keep only sitemap URLs matching it")
    web.add_argument("--exclude", help="regex: drop sitemap URLs matching it")
    web.add_argument("--limit", type=int, help="maximum number of pages")
    web.add_argument("--delay", type=float, default=1.0, help="seconds to wait between requests (default 1)")
    web.add_argument("--out", type=Path, default=Path("web_docs"), help="output folder (default web_docs)")
    ev = commands.add_parser("eval", help="score quality on a labeled question set")
    ev.add_argument("questions", type=Path, help="JSONL file, e.g. eval/fpt_questions.jsonl")
    ev.add_argument("--answers", action="store_true", help="also score answers (one LLM call per question)")
    ev.add_argument("--type", nargs="*", choices=evaluation.QUESTION_TYPES, help="only run these question types")
    ev.add_argument("--out", type=Path, help="save per-question results to a JSON file")
    args = parser.parse_args()

    if args.command == "list":
        for doc in store.list_documents():
            print(f"{doc['id']}  {doc['filename']}  ({doc['chunks']} chunks)")
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
        print(f"\nSaved {len(saved)} pages to {args.out}")
        return

    if problems := llm.check_ready():
        sys.exit("\n".join(problems))

    if args.command == "add":
        for path in expand(args.paths):
            try:
                doc = pipeline.ingest(path.name, path.read_bytes())
                print(f"✓ {doc['filename']}: {doc['chunks']} chunks")
            except Exception as e:  # one bad file should not stop the rest
                print(f"✗ {path.name}: {e}")
        return

    if args.command == "eval":
        run_eval(args.questions, args.answers, args.type, args.out)
        return

    hits = pipeline.retrieve(args.question)
    if not hits:
        sys.exit("No documents added yet.")
    if args.show_context:
        for i, hit in enumerate(hits, start=1):
            print(f"--- [{i}] {pipeline.source_label(hit)} (relevance {hit['score']:.2f})\n{hit['text']}\n")

    answer = ""
    for piece in pipeline.generate(args.question, hits):
        print(piece, end="", flush=True)
        answer += piece
    print("\n\nSources:")
    for n, hit in pipeline.cited_sources(answer, hits):
        print(f"  [{n}] {pipeline.source_label(hit)}  (relevance {hit['score']:.2f})")


if __name__ == "__main__":
    main()
