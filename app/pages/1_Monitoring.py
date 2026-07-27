"""
Monitoring dashboard: reads logs/sessions.jsonl + logs/feedback.jsonl.
Falls back to the committed *.sample.jsonl files so the dashboard is never
empty on a fresh clone.

Presentation notes:
- headline numbers are rendered as a compact table for screenshots;
- recent sessions stay as the detailed table;
- charts are rendered row by row, two per line, for visual consistency.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app import ui

st.set_page_config(page_title="HomeBarista - Monitoring", page_icon="☕", layout="wide")
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
    """Keep all chart panels visually comparable while avoiding empty canvases."""
    return int(min(260, max(180, 70 + 28 * max(n_points, 1))))


def bar(data, color=ACCENT) -> None:
    st.bar_chart(data, color=color, height=chart_height(len(data)))


def chart_row(left_title: str, left_render, right_title: str, right_render) -> None:
    """Render charts as explicit rows: two panels, then a new row."""
    left, right = st.columns(2, gap="large")
    with left:
        ui.section_label(left_title)
        left_render()
    with right:
        ui.section_label(right_title)
        right_render()


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

summary_rows = [
    {"Metric": "Sessions", "Value": f"{n_sessions}", "Detail": "Logged sessions"},
    {"Metric": "Quality pass", "Value": f"{pass_rate:.0%}", "Detail": f"{n_pass} of {n_sessions} passed"},
    {"Metric": "Out of scope", "Value": f"{oos_rate:.0%}", "Detail": "Refused at zero token cost"},
]
if not feedback.empty:
    satisfaction = (feedback["rating"] == "up").mean()
    summary_rows.append({
        "Metric": "Satisfaction",
        "Value": f"{satisfaction:.0%}",
        "Detail": f"{len(feedback)} rating{'s' if len(feedback) != 1 else ''}",
    })
else:
    summary_rows.append({"Metric": "Satisfaction", "Value": "-", "Detail": "No ratings yet"})

ui.section_label("Summary")
st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------
# Recent sessions
# ------------------------------------------------------------------
ui.section_label("Recent sessions")
recent = sessions.sort_values("timestamp", ascending=False).head(8)
recent_view = pd.DataFrame({
    "When": pd.to_datetime(recent["timestamp"], format="ISO8601").dt.strftime("%d %b %H:%M"),
    "Problem": recent["raw_problem"].fillna("").str.slice(0, 70),
    "Machine": recent["machine_type"].map(lambda v: ui.humanize_label(v, "Unknown")),
    "Status": recent["status"].map(lambda v: ui.humanize_label(v, "-")),
    "Quality": recent["verdict"].map(lambda v: ui.VERDICT_LABELS.get(str(v).lower(), "-")),
})
st.dataframe(recent_view, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------
ui.rule()


def render_sessions_per_day() -> None:
    bar(sessions.groupby("date").size())


def render_quality_verdicts() -> None:
    verdicts = (
        sessions["verdict"].fillna("n/a")
        .map(lambda v: ui.VERDICT_LABELS.get(str(v).lower(), ui.humanize_label(v)))
        .value_counts()
    )
    bar(verdicts, color=SUCCESS)


def render_machines_detected() -> None:
    bar(sessions["machine_type"].fillna("unknown").map(ui.humanize_label).value_counts())


def render_top_symptoms() -> None:
    symptoms = sessions["symptoms"].explode().dropna()
    if symptoms.empty:
        st.caption("No symptoms logged yet.")
    else:
        bar(symptoms.map(ui.humanize_label).value_counts())


def render_session_status() -> None:
    st.caption(
        "Coaching, clarification needed, out of scope, error. "
        "Out of scope requests are refused by the ScopeGuard at zero token cost."
    )
    bar(sessions["status"].map(ui.humanize_label).value_counts())


def render_user_feedback() -> None:
    if feedback.empty:
        st.caption("No feedback logged yet - rate a coaching in the main app.")
    else:
        ratings = feedback["rating"].map({"up": "Helpful", "down": "Not helpful"}).value_counts()
        bar(ratings, color=SUCCESS)


def render_agent_iterations() -> None:
    bar(sessions["iterations"].value_counts().sort_index())


def render_feedback_over_time() -> None:
    if feedback.empty:
        st.caption("No feedback logged yet.")
    else:
        feedback["date"] = pd.to_datetime(feedback["timestamp"], format="ISO8601").dt.date
        over_time = feedback.groupby(["date", "rating"]).size().unstack(fill_value=0)
        over_time = over_time.rename(columns={"up": "Helpful", "down": "Not helpful"})
        st.bar_chart(
            over_time,
            color=[SUCCESS if c == "Helpful" else MUTED for c in over_time.columns],
            height=chart_height(len(over_time)),
        )


chart_row("Sessions per day", render_sessions_per_day, "Quality verdicts", render_quality_verdicts)
chart_row("Machines detected", render_machines_detected, "Top symptoms", render_top_symptoms)
chart_row("Session status", render_session_status, "User feedback", render_user_feedback)
chart_row("Agent iterations per session", render_agent_iterations, "Feedback over time", render_feedback_over_time)
