"""
Monitoring dashboard — reads logs/sessions.jsonl + logs/feedback.jsonl.
Falls back to the committed *.sample.jsonl files so the dashboard is never
empty on a fresh clone. 7 charts + 4 headline metrics.

Presentation notes (no data is created or transformed here beyond display):
- headline numbers are compact metric cards, not st.metric tiles;
- chart height scales with the number of points, so 2 sessions don't get the
  same canvas as 200;
- chart colors come from the espresso palette in .streamlit/config.toml.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app import ui

st.set_page_config(page_title="HomeBarista — Monitoring", page_icon="☕", layout="wide")
ui.inject_css()

ACCENT = "#6F4E37"
SUCCESS = "#397057"
MUTED = "#B0A79C"


def load_jsonl(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def chart_height(n_points: int) -> int:
    """Small datasets get a small canvas — a near-empty chart should not own
    a third of the screen."""
    return int(min(300, max(150, 60 + 30 * max(n_points, 1))))


def bar(data, color=ACCENT) -> None:
    st.bar_chart(data, color=color, height=chart_height(len(data)))


ui.page_header("HomeBarista Coach", "Monitoring", "Session quality, coverage and user feedback.")

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
n_sessions = len(sessions)
if "verdict" in sessions:
    n_pass = int((sessions["verdict"] == "pass").sum())
    pass_rate = n_pass / n_sessions if n_sessions else 0.0
else:
    n_pass, pass_rate = 0, 0.0
oos_rate = (sessions["status"] == "out_of_scope").mean()

cards = [
    {"label": "Sessions", "value": f"{n_sessions}"},
    {"label": "Quality pass", "value": f"{pass_rate:.0%}", "sub": f"{n_pass} of {n_sessions}"},
    {"label": "Out of scope", "value": f"{oos_rate:.0%}", "sub": "refused at zero token cost"},
]
if not feedback.empty:
    satisfaction = (feedback["rating"] == "up").mean()
    cards.append({
        "label": "Satisfaction",
        "value": f"{satisfaction:.0%}",
        "sub": f"{len(feedback)} rating{'s' if len(feedback) != 1 else ''}",
    })
else:
    cards.append({"label": "Satisfaction", "value": "—", "sub": "no ratings yet"})

ui.metric_cards(cards)

# ------------------------------------------------------------------
# Recent sessions — display-only reshaping of columns already logged
# ------------------------------------------------------------------
ui.section_label("Recent sessions")
recent = sessions.sort_values("timestamp", ascending=False).head(8)
recent_view = pd.DataFrame({
    "When": pd.to_datetime(recent["timestamp"], format="ISO8601").dt.strftime("%d %b %H:%M"),
    "Problem": recent["raw_problem"].fillna("").str.slice(0, 70),
    "Machine": recent["machine_type"].map(lambda v: ui.humanize_label(v, "Unknown")),
    "Status": recent["status"].map(lambda v: ui.humanize_label(v, "—")),
    "Quality": recent["verdict"].map(lambda v: ui.VERDICT_LABELS.get(str(v).lower(), "—")),
})
st.dataframe(recent_view, hide_index=True)  # width defaults to "stretch"

# ------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------
ui.rule()
left, right = st.columns(2, gap="large")

with left:
    ui.section_label("Sessions per day")
    bar(sessions.groupby("date").size())

    ui.section_label("Machines detected")
    bar(sessions["machine_type"].fillna("unknown").map(ui.humanize_label).value_counts())

    ui.section_label("Session status")
    st.caption(
        "Coaching · clarification needed · out of scope · error — out of scope "
        "requests are refused by the ScopeGuard at zero token cost."
    )
    bar(sessions["status"].map(ui.humanize_label).value_counts())

    ui.section_label("Agent iterations per session")
    bar(sessions["iterations"].value_counts().sort_index())

with right:
    ui.section_label("Quality verdicts")
    verdicts = (
        sessions["verdict"].fillna("n/a")
        .map(lambda v: ui.VERDICT_LABELS.get(str(v).lower(), ui.humanize_label(v)))
        .value_counts()
    )
    bar(verdicts, color=SUCCESS)

    ui.section_label("Top symptoms")
    symptoms = sessions["symptoms"].explode().dropna()
    if symptoms.empty:
        st.caption("No symptoms logged yet.")
    else:
        bar(symptoms.map(ui.humanize_label).value_counts())

    ui.section_label("User feedback")
    if feedback.empty:
        st.caption("No feedback logged yet — rate a coaching in the main app.")
    else:
        ratings = feedback["rating"].map({"up": "Helpful", "down": "Not helpful"}).value_counts()
        bar(ratings, color=SUCCESS)

        feedback["date"] = pd.to_datetime(feedback["timestamp"], format="ISO8601").dt.date
        st.caption("Feedback over time")
        over_time = feedback.groupby(["date", "rating"]).size().unstack(fill_value=0)
        over_time = over_time.rename(columns={"up": "Helpful", "down": "Not helpful"})
        st.bar_chart(
            over_time,
            color=[SUCCESS if c == "Helpful" else MUTED for c in over_time.columns],
            height=chart_height(len(over_time)),
        )
