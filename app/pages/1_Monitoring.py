"""
Monitoring dashboard — reads logs/sessions.jsonl + logs/feedback.jsonl.
Falls back to the committed *.sample.jsonl files so the dashboard is never
empty on a fresh clone. 7 charts + 4 headline metrics.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="HomeBarista — Monitoring", page_icon="📊", layout="wide")
st.title("📊 Monitoring")


def load_jsonl(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


sessions = load_jsonl("logs/sessions.jsonl")
using_sample = False
if sessions.empty:
    sessions = load_jsonl("logs/sessions.sample.jsonl")
    using_sample = True

feedback = load_jsonl("logs/feedback.jsonl")
if feedback.empty:
    feedback = load_jsonl("logs/feedback.sample.jsonl")

if sessions.empty:
    st.warning("No session logs found. Use the app first, or commit the sample logs.")
    st.stop()

if using_sample:
    st.info("Showing committed sample logs (no real sessions logged yet on this machine).")

sessions["date"] = pd.to_datetime(sessions["timestamp"], format="ISO8601").dt.date

# ------------------------------------------------------------------
# Headline metrics
# ------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total sessions", len(sessions))
pass_rate = (sessions["verdict"] == "pass").mean() if "verdict" in sessions else 0.0
col2.metric("Quality pass rate", f"{pass_rate:.0%}")
oos_rate = (sessions["status"] == "out_of_scope").mean()
col3.metric("Out-of-scope rate", f"{oos_rate:.0%}")
if not feedback.empty:
    satisfaction = (feedback["rating"] == "up").mean()
    col4.metric("👍 Satisfaction", f"{satisfaction:.0%}")
else:
    col4.metric("👍 Satisfaction", "no data")

st.divider()

# ------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("1 · Sessions per day")
    st.bar_chart(sessions.groupby("date").size())

    st.subheader("3 · Machines detected")
    st.bar_chart(sessions["machine_type"].fillna("unknown").value_counts())

    st.subheader("5 · Session status")
    st.caption("coaching · clarification_needed · out_of_scope · error — "
               "out_of_scope = requests refused by the ScopeGuard at zero token cost")
    st.bar_chart(sessions["status"].value_counts())

    st.subheader("7 · Agent iterations per session")
    st.bar_chart(sessions["iterations"].value_counts().sort_index())

with right:
    st.subheader("2 · Quality verdicts (pass / warn / fail)")
    st.bar_chart(sessions["verdict"].fillna("n/a").value_counts())

    st.subheader("4 · Top symptoms")
    symptoms = sessions["symptoms"].explode().dropna()
    if symptoms.empty:
        st.caption("No symptoms logged yet.")
    else:
        st.bar_chart(symptoms.value_counts())

    st.subheader("6 · User feedback 👍/👎")
    if feedback.empty:
        st.caption("No feedback logged yet — rate a coaching in the main app.")
    else:
        st.bar_chart(feedback["rating"].value_counts())
        feedback["date"] = pd.to_datetime(feedback["timestamp"], format="ISO8601").dt.date
        st.caption("Feedback over time")
        st.bar_chart(feedback.groupby(["date", "rating"]).size().unstack(fill_value=0))
