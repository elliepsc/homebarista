"""
HomeBarista Coach — Streamlit UI
================================
Multi-turn chat over the coaching pipeline.

- DEMO mode (default): free, no API key, deterministic engine + mock coaching.
- LIVE mode: locked behind a password (st.secrets["LIVE_PASSWORD"]) OR a
  user-provided API key for the configured LLM provider (kept in this
  user's session state only — never written to os.environ, never stored,
  never logged). Budget: BYO key is unlimited (visitor's own cost); the
  shared password is capped per session AND by a global daily budget
  shared across all visitors (module-level counter — resets on process
  restart, e.g. Streamlit Cloud sleep/redeploy).
- LLM provider/model come from .env (Groq by default — see .env.example
  and engine/llm_client.py).
- Off-topic input is refused by the deterministic ScopeGuard before any
  model loads — zero tokens spent.
- 👍/👎 feedback after each coaching → logs/feedback.jsonl (joined to
  sessions.jsonl via session_id) — feeds the Monitoring dashboard page.

Run: streamlit run app/streamlit_app.py
"""

import asyncio
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # LLM_*, DEMO_MODE, CHROMA_PERSIST_DIR from .env (see .env.example)

from engine.llm_client import DEFAULT_MODELS, get_provider
from pipeline.pipeline import run_pipeline

# Default style — winner of the Phase E RAG evaluation (evals/run_rag_eval.py,
# rag_eval_20260722T195411Z.json): technical, pass_rate 0.452 / mean 0.891.
WINNER_STYLE = "technical"
STYLES = ["detailed", "concise", "technical"]
FEEDBACK_FILE = Path("logs/feedback.jsonl")


def _build_marker() -> str:
    """Short git SHA of the running checkout, read straight from .git (no git
    binary needed). Lets anyone see WHICH commit is live — compare it to the
    latest push to confirm a redeploy actually landed. Falls back to 'local'."""
    try:
        head = Path(".git/HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head[4:].strip()
            loose = Path(".git") / ref
            if loose.exists():
                return loose.read_text(encoding="utf-8").strip()[:7]
            packed = Path(".git/packed-refs")
            if packed.exists():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.endswith(ref):
                        return line.split()[0][:7]
        else:
            return head[:7]
    except Exception:
        pass
    return "local"

# Live-mode budgets. Two unlock paths, two different cost owners:
# - BYO key (visitor's own): their cost, not capped here.
# - Shared password (owner's key): capped per session AND by a global
#   daily budget shared across every visitor, so a session reset (new
#   tab, cleared cookies) can't be used to bypass the per-session cap.
# The global counter MUST live behind st.cache_resource, not a plain
# module-level variable: Streamlit re-executes the whole main script on
# every rerun, so a bare global gets reset every single time (verified
# empirically) — it would never actually accumulate. cache_resource is
# the one construct that survives reruns AND is shared across every
# session for the life of the process. It resets on process restart
# (Cloud sleep/redeploy) — an acceptable soft limit for a portfolio
# project, not a hard guarantee.
MAX_LIVE_RUNS_PER_SESSION = 10
MAX_LIVE_RUNS_GLOBAL_DAILY = 25


@st.cache_resource
def _shared_key_budget_store() -> dict:
    return {"date": None, "count": 0}


def _todays_shared_key_budget() -> dict:
    """The cache_resource-backed budget dict, rolled over for the current
    UTC day. Both read and write go through this so the rollover happens
    consistently regardless of call order."""
    budget = _shared_key_budget_store()
    today = datetime.now(timezone.utc).date().isoformat()
    if budget["date"] != today:
        budget["date"] = today
        budget["count"] = 0
    return budget


def _shared_key_runs_remaining_today() -> int:
    return MAX_LIVE_RUNS_GLOBAL_DAILY - _todays_shared_key_budget()["count"]


def _consume_shared_key_run() -> None:
    _todays_shared_key_budget()["count"] += 1

EXAMPLE_PROBLEMS = [
    "My DeLonghi Dinamica makes bitter espresso every morning",
    "Sour, thin shots from my Gaggia Classic — grind or temperature?",
    "My moka pot coffee tastes burnt and harsh",
    "V60 pour over comes out weak and watery, 30g coffee to 500g water",
    "How do I get more crema from my super-automatic machine?",
]

st.set_page_config(page_title="HomeBarista Coach", page_icon="☕", layout="wide")

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
st.session_state.setdefault("messages", [])       # [{role, content}]
st.session_state.setdefault("last_result", None)  # last run_pipeline output
st.session_state.setdefault("live_runs", 0)
st.session_state.setdefault("live_unlocked", False)
st.session_state.setdefault("byo_api_key", None)  # per-session BYO key, never in os.environ
st.session_state.setdefault("feedback_given", set())  # session_ids already rated


def _write_feedback(session_id: str, rating: str, comment: str = "") -> None:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rating": rating,
        "comment": comment,
    }
    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.title("☕ HomeBarista Coach")
    st.caption("Science-backed coffee coaching from YouTube barista expertise")

    style = st.radio("Coach style", STYLES, index=STYLES.index(WINNER_STYLE))

    llm_provider = get_provider()
    mode = st.radio(
        "Mode",
        ["Demo (free, no key)", "Live (real AI coaching)"],
        help="Demo mode uses the deterministic diagnostic engine with mock "
             "coaching text — no API key, no cost. Live mode calls the LLM "
             f"provider configured in .env (currently: {llm_provider}).",
    )

    if mode.startswith("Live"):
        if not st.session_state["live_unlocked"]:
            password = st.text_input("Live password", type="password")
            try:
                expected_password = st.secrets.get("LIVE_PASSWORD")
            except FileNotFoundError:  # no secrets.toml on a fresh clone
                expected_password = None
            if password and expected_password and hmac.compare_digest(
                password, expected_password
            ):
                st.session_state["live_unlocked"] = True
                st.success("Live mode unlocked.")
            else:
                byo = st.text_input(
                    f"…or paste YOUR {llm_provider} API key",
                    type="password",
                    help="Kept in your session only — never stored, never "
                         "logged, never shared with other sessions.",
                )
                if byo:
                    st.session_state["byo_api_key"] = byo.strip()
                    st.session_state["live_unlocked"] = True
                    st.success("Live mode unlocked with your key.")
        if st.session_state["live_unlocked"]:
            if st.session_state["byo_api_key"]:
                st.caption("Live budget: unlimited — you're using your own API key.")
            else:
                session_remaining = MAX_LIVE_RUNS_PER_SESSION - st.session_state["live_runs"]
                global_remaining = _shared_key_runs_remaining_today()
                remaining = min(session_remaining, global_remaining)
                st.caption(
                    f"Live budget: {max(remaining, 0)} runs left "
                    f"(session cap {MAX_LIVE_RUNS_PER_SESSION}, "
                    f"shared daily cap {MAX_LIVE_RUNS_GLOBAL_DAILY})."
                )
                if remaining <= 0:
                    st.warning("Live budget exhausted — falling back to demo mode.")

    st.divider()
    st.subheader("Try an example")
    for i, example in enumerate(EXAMPLE_PROBLEMS):
        if st.button(example, key=f"example_{i}", use_container_width=True):
            st.session_state["pending_input"] = example

    st.divider()
    with st.expander("Technical details"):
        st.markdown(
            "- **Retrieval**: MiniLM-L6-v2 bi-encoder + cross-encoder re-ranking "
            "(optional hybrid BM25+vector via RRF)\n"
            "- **Diagnostic engine**: deterministic (12 symptoms, machine capability map)\n"
            f"- **LLM**: {llm_provider} · {DEFAULT_MODELS[llm_provider]} by default — "
            "provider/model configured in `.env` (live mode, agentic tool-use loop)\n"
            "- **Guardrails**: deterministic ScopeGuard — off-topic requests "
            "cost 0 tokens\n"
            "- [Source on GitHub](https://github.com/elliepsc/homebarista)"
        )

    st.divider()
    st.caption(f"build `{_build_marker()}`")

# ------------------------------------------------------------------
# Chat history
# ------------------------------------------------------------------
st.title("Describe your coffee problem")

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------
# Input handling
# ------------------------------------------------------------------
user_input = st.chat_input("Describe your coffee problem...", max_chars=1500)
if not user_input:
    user_input = st.session_state.pop("pending_input", None)

if user_input:
    using_byo_key = bool(st.session_state["byo_api_key"])
    shared_key_budget_left = using_byo_key or (
        st.session_state["live_runs"] < MAX_LIVE_RUNS_PER_SESSION
        and _shared_key_runs_remaining_today() > 0
    )
    demo = (
        mode.startswith("Demo")
        or not st.session_state["live_unlocked"]
        or not shared_key_budget_left
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Diagnosing your coffee problem..."):
            # Agent mode (~4 LLM calls/run) only for visitors on their OWN
            # key — their budget, their cost. The shared free key runs the
            # linear pipeline (1 LLM call/run) so a single request can't
            # swallow the whole free-tier token/minute budget and 429.
            result = asyncio.run(run_pipeline(
                user_input,
                coach_style=style,
                demo_mode=demo,
                use_agent=not demo and using_byo_key,
                conversation_history=st.session_state["messages"],
                api_key=st.session_state["byo_api_key"],
            ))

    if not demo:
        st.session_state["live_runs"] += 1
        if not using_byo_key:
            _consume_shared_key_run()

    status = result["status"]
    if status == "coaching":
        reply = result["coaching_text"]
    elif status == "clarification_needed":
        reply = f"Before I can diagnose: **{result['clarification_question']}**"
    elif status == "out_of_scope":
        reply = result["clarification_question"]  # fixed ScopeGuard message, no LLM involved
    else:  # error
        reply = "⚠️ " + result["clarification_question"]

    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.session_state["last_result"] = result
    st.rerun()

# ------------------------------------------------------------------
# Diagnostic details + feedback for the last coaching
# ------------------------------------------------------------------
result = st.session_state["last_result"]
if result and result["status"] == "coaching":
    context = result.get("context", {})
    diagnostic = result.get("diagnostic", {})
    evaluation = result.get("evaluation", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Machine", (context.get("machine_type") or "unknown").replace("_", " "))
    col2.metric("Diagnostic confidence", f"{diagnostic.get('diagnostic_confidence', 0):.0%}")
    col3.metric("Quality verdict", evaluation.get("overall_verdict", "n/a"))

    with st.expander("🔬 Diagnostic details"):
        st.write("**Symptoms:**", ", ".join(diagnostic.get("symptoms", [])) or "none detected")
        for rc in diagnostic.get("root_causes", []):
            st.write(f"- {rc.get('hypothesis', '?')} (confidence {rc.get('confidence', 0):.0%})")

    if result.get("retrieved_chunks"):
        with st.expander("📚 Knowledge sources"):
            for chunk in result["retrieved_chunks"][:5]:
                st.markdown(
                    f"**{chunk.get('channel', '')} — {chunk.get('title', '')}**  \n"
                    f"{chunk.get('text', '')[:300]}…"
                )

    post_eval = evaluation.get("post", {})
    if post_eval:
        with st.expander("✅ Quality checks"):
            for check, ok in post_eval.get("checks", {}).items():
                st.write(("✅" if ok else "❌") + f" {check}")

    # Feedback — one rating per session_id
    session_id = result["session_id"]
    if session_id not in st.session_state["feedback_given"]:
        st.caption(f"Was this coaching helpful? (session {session_id})")
        comment = st.text_input("Optional comment", key=f"fb_comment_{session_id}")
        col_up, col_down, _ = st.columns([1, 1, 6])
        if col_up.button("👍", key=f"fb_up_{session_id}"):
            _write_feedback(session_id, "up", comment)
            st.session_state["feedback_given"].add(session_id)
            st.toast("Thanks for the feedback!")
            st.rerun()
        if col_down.button("👎", key=f"fb_down_{session_id}"):
            _write_feedback(session_id, "down", comment)
            st.session_state["feedback_given"].add(session_id)
            st.toast("Thanks — noted for improvement.")
            st.rerun()
    else:
        st.caption("✔️ Feedback recorded for this session — thank you!")
