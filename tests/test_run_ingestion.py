"""
Tests for ingestion/run_ingestion.py — offline only (no network, no LLM).
"""

from ingestion.run_ingestion import (
    DEFAULT_MAX_VIDEOS_PER_RUN,
    DEFAULT_TRANSCRIPT_DELAY_MIN_S,
    DEFAULT_TRANSCRIPT_DELAY_MAX_S,
    _pacing_delay,
)


# ------------------------------------------------------------------
# E2 — volume cap + randomized pacing
# ------------------------------------------------------------------

def test_default_cap_is_12():
    assert DEFAULT_MAX_VIDEOS_PER_RUN == 12


def test_pacing_jitter_range():
    for _ in range(200):
        delay = _pacing_delay(DEFAULT_TRANSCRIPT_DELAY_MIN_S, DEFAULT_TRANSCRIPT_DELAY_MAX_S)
        assert DEFAULT_TRANSCRIPT_DELAY_MIN_S <= delay <= DEFAULT_TRANSCRIPT_DELAY_MAX_S

    # bounds are configurable, not hardcoded
    assert _pacing_delay(5.0, 5.0) == 5.0
    for _ in range(50):
        delay = _pacing_delay(1.0, 2.0)
        assert 1.0 <= delay <= 2.0
