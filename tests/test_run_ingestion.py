"""
Tests for ingestion/run_ingestion.py — offline only (no network, no LLM).
"""

from ingestion.content_classifier import ContentClassifier
from ingestion.run_ingestion import (
    DEFAULT_MAX_VIDEOS_PER_RUN,
    DEFAULT_TRANSCRIPT_DELAY_MIN_S,
    DEFAULT_TRANSCRIPT_DELAY_MAX_S,
    _pacing_delay,
    _rank_by_relevance,
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


# ------------------------------------------------------------------
# E3 — relevance pre-filtering before any transcript fetch
# ------------------------------------------------------------------

HIGH_RELEVANCE_DOC = {
    "source_id": "high",
    "title": "Espresso Extraction: Fixing Bitter and Sour Shots",
    "description": "Grind size, yield, TDS and channeling — dialing in extraction.",
    "tags": ["espresso", "extraction", "grind", "troubleshooting"],
}

LOW_RELEVANCE_DOC = {
    "source_id": "low",
    "title": "V60 basics",
    "description": "A quick pour over intro.",
    "tags": ["v60"],
}

IRRELEVANT_DOC = {
    "source_id": "irrelevant",
    "title": "Best Hiking Trails in Patagonia",
    "description": "Mountain travel vlog, no coffee content.",
    "tags": ["hiking", "travel"],
}


def test_fetch_order_by_relevance():
    classifier = ContentClassifier()
    raw_docs = [LOW_RELEVANCE_DOC, IRRELEVANT_DOC, HIGH_RELEVANCE_DOC]

    ranked = _rank_by_relevance(raw_docs, classifier)

    assert [d["source_id"] for d in ranked] == ["high", "low"]
    scores = [classifier.score_relevance(d) for d in ranked]
    assert scores == sorted(scores, reverse=True)


def test_irrelevant_never_fetched():
    classifier = ContentClassifier()
    raw_docs = [HIGH_RELEVANCE_DOC, IRRELEVANT_DOC]

    ranked = _rank_by_relevance(raw_docs, classifier)

    assert "irrelevant" not in [d["source_id"] for d in ranked]
    assert classifier.score_relevance(IRRELEVANT_DOC) == 0
