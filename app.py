"""Web UI entry point with two pages: Chat and Evaluation.

Run: streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Document QA", page_icon="📚", layout="wide")
st.navigation(
    [
        st.Page("ui/chat.py", title="Chat", icon=":material/chat:", default=True),
        st.Page("ui/evaluation.py", title="Evaluation", icon=":material/monitoring:"),
    ]
).run()
