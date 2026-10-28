"""Evaluation page: scores of every configuration tried, read from eval/reports/*.json."""

import json

import pandas as pd
import streamlit as st

from rag.config import BASE_DIR

REPORTS = BASE_DIR / "eval" / "reports"
QUESTION_SETS = {
    "fpt_questions.jsonl": "51 Vietnamese questions",
    "fpt_questions_en.jsonl": "6 English questions over Vietnamese pages",
    "fpt_followups.jsonl": "6 follow-up questions with conversation history",
}


def percent(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}" if value in (0, 1) else f"{value:.1%}"


def scores_table(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for r in reports:
        s = r["summary"]
        rows.append(
            {
                "Configuration": r["name"],
                "Hit@K": s["all"].get("hit"),
                "MRR": s["all"].get("rr"),
                "Recall@K (lists)": s.get("list", {}).get("recall"),
                "Fact coverage": s.get("fact", {}).get("coverage"),
                "List coverage": s.get("list", {}).get("coverage"),
                "Cited": s["all"].get("cited"),
                "Refusals": s.get("negative", {}).get("refusal"),
                "What changed": r["description"],
            }
        )
    return pd.DataFrame(rows).set_index("Configuration").dropna(axis=1, how="all")


def show_failures(reports: list[dict]) -> None:
    for r in reports:
        with st.expander(f"{r['name']}: {len(r['failures'])} question(s) not passed"):
            if not r["failures"]:
                st.write("All questions passed.")
            for question_id, reason in r["failures"]:
                st.markdown(f"- `{question_id}` {reason}")


st.title("📊 Evaluation")
st.caption(
    "Every setting in this app was chosen by measuring it on labeled questions over the crawled FPT University "
    "admissions pages, not by guesswork. Each row is one configuration that was tried."
)

reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(REPORTS.glob("*.json"))]
if not reports:
    st.info(
        "No reports yet. Run `python cli.py eval eval/fpt_questions.jsonl --answers --report eval/reports/name.json`."
    )
    st.stop()
by_set = {name: [r for r in reports if r["questions"] == name] for name in QUESTION_SETS}

main = by_set["fpt_questions.jsonl"]
if main:
    baseline, default = main[0], next((r for r in main if "default" in r["name"]), main[-1])
    tiles = [
        ("Fact coverage", "fact", "coverage"),
        ("List coverage", "list", "coverage"),
        ("Answers citing a source", "all", "cited"),
        ("Correct refusals", "negative", "refusal"),
    ]
    for column, (label, group, metric) in zip(st.columns(len(tiles)), tiles, strict=True):
        now, before = default["summary"][group].get(metric), baseline["summary"][group].get(metric)
        changed = now is not None and before is not None and now != before
        delta = f"{(now - before) * 100:+.1f} pts vs baseline" if changed else None  # no green arrow for "no change"
        column.metric(label, percent(now), delta=delta)

    st.subheader(QUESTION_SETS["fpt_questions.jsonl"])
    table = scores_table(main)
    st.dataframe(
        table,
        column_config={
            col: st.column_config.NumberColumn(format="%.2f")
            for col in table.columns
            if col in ("Hit@K", "MRR", "Recall@K (lists)")
        }
        | {
            col: st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)
            for col in table.columns
            if col in ("Fact coverage", "List coverage", "Cited", "Refusals")
        },
    )
    st.markdown("**Answer coverage by configuration**")
    st.bar_chart(table[["Fact coverage", "List coverage"]], stack=False, horizontal=True)

for name in ("fpt_followups.jsonl", "fpt_questions_en.jsonl"):
    if by_set[name]:
        st.subheader(QUESTION_SETS[name])
        st.dataframe(scores_table(by_set[name]))

st.subheader("Questions not passed")
show_failures(reports)

st.subheader("Reproduce")
st.code(
    "python cli.py eval eval/fpt_questions.jsonl --answers \\\n"
    "    --report eval/reports/my-run.json --description 'what this run tests'",
    language="bash",
)
