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

import altair as alt
import pandas as pd
import streamlit as st

from app import ui

st.set_page_config(page_title="HomeBarista - Monitoring", page_icon="☕", layout="wide")
ui.inject_css()

ACCENT = "#6F4E37"
SUCCESS = "#397057"
MUTED = "#B0A79C"
RECENT_SESSION_LIMIT = 10
FEEDBACK_DAYS = 7


def load_jsonl(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def chart_height(n_points: int) -> int:
    """Keep all chart panels visually comparable while avoiding empty canvases."""
    return int(min(260, max(180, 70 + 28 * max(n_points, 1))))


def bar(data, color=ACCENT, label_size: int = 13) -> None:
    """Bar chart with readable horizontal x labels."""
    if data.empty:
        st.caption("No data yet.")
        return

    frame = data.reset_index()
    frame.columns = ["label", "value"]
    frame["label"] = frame["label"].astype(str)

    chart = (
        alt.Chart(frame)
        .mark_bar(color=color)
        .encode(
            x=alt.X(
                "label:N",
                title=None,
                sort=None,
                axis=alt.Axis(labelAngle=0, labelFontSize=label_size, labelLimit=160),
            ),
            y=alt.Y(
                "value:Q",
                title=None,
                axis=alt.Axis(labelFontSize=12, grid=True),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Label"),
                alt.Tooltip("value:Q", title="Value"),
            ],
        )
        .properties(height=chart_height(len(frame)))
    )
    st.altair_chart(chart, use_container_width=True)


def grouped_bar(data: pd.DataFrame, colors: list[str], label_size: int = 13) -> None:
    """Grouped bar chart for small time-series tables."""
    if data.empty:
        st.caption("No data yet.")
        return

    frame = (
        data.reset_index()
        .melt(id_vars=data.index.name or "date", var_name="series", value_name="value")
        .rename(columns={data.index.name or "date": "label"})
    )
    frame["label"] = frame["label"].astype(str)

    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(
                "label:N",
                title=None,
                sort=None,
                axis=alt.Axis(labelAngle=0, labelFontSize=label_size, labelLimit=160),
            ),
            xOffset=alt.XOffset("series:N"),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(labelFontSize=12, grid=True)),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(range=colors),
                legend=alt.Legend(orient="bottom", labelFontSize=13, titleFontSize=13),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title="Value"),
            ],
        )
        .properties(height=chart_height(data.shape[0]))
    )
    st.altair_chart(chart, use_container_width=True)


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
st.caption(f"Showing the {RECENT_SESSION_LIMIT} most recent sessions.")
recent = sessions.sort_values("timestamp", ascending=False).head(RECENT_SESSION_LIMIT)
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
        st.caption(f"Showing feedback from the last {FEEDBACK_DAYS} days.")
        feedback_window = feedback.copy()
        feedback_window["timestamp"] = pd.to_datetime(feedback_window["timestamp"], format="ISO8601")
        latest_day = feedback_window["timestamp"].dt.normalize().max()
        start_day = latest_day - pd.Timedelta(days=FEEDBACK_DAYS - 1)
        feedback_window = feedback_window[feedback_window["timestamp"].dt.normalize() >= start_day]
        feedback_window["date"] = feedback_window["timestamp"].dt.date
        over_time = feedback_window.groupby(["date", "rating"]).size().unstack(fill_value=0)
        over_time = over_time.rename(columns={"up": "Helpful", "down": "Not helpful"})
        grouped_bar(over_time, colors=[SUCCESS if c == "Helpful" else MUTED for c in over_time.columns])


chart_row("Sessions per day", render_sessions_per_day, "Quality verdicts", render_quality_verdicts)
chart_row("Machines detected", render_machines_detected, "Top symptoms", render_top_symptoms)
chart_row("Session status", render_session_status, "User feedback", render_user_feedback)
chart_row("Agent iterations per session", render_agent_iterations, "Feedback over time", render_feedback_over_time)
