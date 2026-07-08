"""
Regression tests for the Live-mode budget in app/streamlit_app.py.

Streamlit re-executes the whole main script on every rerun, so a plain
module-level variable is reset every time (verified empirically — see
PR discussion). The global daily budget for the shared password MUST
live behind st.cache_resource, which is the one construct that survives
reruns and is shared across every session in the same process. These
tests drive the real app via AppTest and would fail if that budget were
ever reverted to a plain module-level dict.
"""

import pytest
from streamlit.testing.v1 import AppTest

import app.streamlit_app as streamlit_app

APP_PATH = "app/streamlit_app.py"


@pytest.fixture(autouse=True)
def _reset_shared_budget():
    """The global budget lives in a cache_resource singleton shared by
    the whole process (that's the point) — reset it between tests so
    they don't leak state into each other."""
    streamlit_app._shared_key_budget_store().update(date=None, count=0)
    yield


async def _fake_run_pipeline(raw_problem, coach_style="detailed", demo_mode=True,
                              use_agent=True, conversation_history=None, api_key=None):
    return {
        "status": "coaching",
        "coaching_text": "mock coaching",
        "session_id": f"sid-{raw_problem}",
        "context": {},
        "diagnostic": {},
        "evaluation": {},
        "retrieved_chunks": [],
        "demo_mode_used": demo_mode,
    }


def _patch_pipeline(monkeypatch):
    monkeypatch.setattr("pipeline.pipeline.run_pipeline", _fake_run_pipeline)


def _unlock_shared_password(at: AppTest) -> AppTest:
    """Skip the password widget/secrets.toml entirely — pre-seed the
    session state it would have produced, since only the *budget*
    behaviour after unlock is under test here."""
    at.session_state["live_unlocked"] = True
    at.session_state["byo_api_key"] = None
    return at


def _drive_shared_key_turns(at: AppTest, n: int) -> AppTest:
    """Submit n chat turns unlocked via the shared password (one session)."""
    at = _unlock_shared_password(at)
    at.run()
    at.sidebar.radio[1].set_value("Live (real AI coaching)").run()
    for i in range(n):
        at.chat_input[0].set_value(f"problem {i}").run()
    return at


def test_shared_budget_persists_across_sessions(monkeypatch):
    _patch_pipeline(monkeypatch)
    # MAX_LIVE_RUNS_PER_SESSION=10, MAX_LIVE_RUNS_GLOBAL_DAILY=25: the
    # per-session cap alone can never prove cross-session accumulation
    # (a single session always shows <=10 regardless of the global pool).
    # Two sessions must together burn past 15 for the global cap to
    # become the binding constraint on a third session.
    session_a = _drive_shared_key_turns(AppTest.from_file(APP_PATH), 9)
    assert session_a.session_state["live_runs"] == 9

    session_b = _drive_shared_key_turns(AppTest.from_file(APP_PATH), 9)
    assert session_b.session_state["live_runs"] == 9

    # 18 runs consumed globally (9 + 9) -> 7 left of the 25/day pool,
    # which is now tighter than session_c's own fresh 10/session cap.
    session_c = _unlock_shared_password(AppTest.from_file(APP_PATH))
    session_c.run()
    session_c.sidebar.radio[1].set_value("Live (real AI coaching)").run()

    caption_texts = " ".join(c.value for c in session_c.sidebar.caption)
    assert "7 runs left" in caption_texts, (
        "the global daily budget (25) must reflect the 18 runs already "
        f"consumed by sessions A and B — got: {caption_texts!r}. If this "
        "fails, the budget counter was reset on rerun/new-session instead "
        "of persisting via st.cache_resource."
    )


def test_session_cap_falls_back_to_demo(monkeypatch):
    _patch_pipeline(monkeypatch)

    at = _unlock_shared_password(AppTest.from_file(APP_PATH))
    at.run()
    at.sidebar.radio[1].set_value("Live (real AI coaching)").run()

    at.session_state["live_runs"] = 10  # MAX_LIVE_RUNS_PER_SESSION already reached
    at.chat_input[0].set_value("sour shots").run()

    # Falls back to demo: the session counter must NOT increment past the cap.
    assert at.session_state["live_runs"] == 10


def test_byo_key_is_never_capped(monkeypatch):
    _patch_pipeline(monkeypatch)

    at = AppTest.from_file(APP_PATH)
    at.session_state["live_unlocked"] = True
    at.session_state["byo_api_key"] = "sk-visitor-own-key"
    at.run()
    at.sidebar.radio[1].set_value("Live (real AI coaching)").run()

    at.session_state["live_runs"] = 10  # would exhaust the shared-key cap
    at.chat_input[0].set_value("weak pour over").run()

    assert at.session_state["live_runs"] == 11, (
        "a bring-your-own key must never be blocked by the shared-key budget"
    )
    caption_texts = " ".join(c.value for c in at.sidebar.caption)
    assert "unlimited" in caption_texts
