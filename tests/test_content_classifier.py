"""Tests for ContentClassifier."""

from unittest.mock import patch
import pytest
from ingestion.content_classifier import ContentClassifier


@pytest.fixture
def classifier():
    return ContentClassifier()


def test_troubleshooting_bitter(classifier):
    doc = {
        "title": "Why Your Espresso Tastes Bitter",
        "description": "Fix bitter espresso by adjusting grind and extraction.",
        "tags": ["bitter", "espresso", "troubleshooting"],
        "transcript_text": (
            "Your espresso is bitter because of over-extraction. "
            "The sour notes come first, then the bitter finish. "
            "Channeling also causes bitter spots in the puck."
        ),
    }
    result = classifier.classify(doc)
    assert result["domain"] == "troubleshooting"


def test_super_auto_priority(classifier):
    """When a doc mentions both espresso and DeLonghi, method must be super_automatic."""
    doc = {
        "title": "DeLonghi Espresso Setup Guide",
        "description": "Get the perfect espresso from your DeLonghi machine.",
        "tags": ["DeLonghi", "espresso", "super automatic"],
        "transcript_text": (
            "The DeLonghi Dinamica is a super automatic espresso machine. "
            "Unlike manual espresso, the DeLonghi grinds and tamps automatically."
        ),
    }
    with patch.object(classifier, "_classify_with_llm", return_value=None):
        result = classifier.classify(doc)
    assert result["method"] == "super_automatic"


def test_low_confidence_flagged(classifier):
    """An off-topic document must have classification_confidence < 0.4."""
    doc = {
        "title": "Best hiking trails in Patagonia",
        "description": "Explore the mountains of South America with this guide.",
        "tags": ["hiking", "travel", "Patagonia"],
        "transcript_text": (
            "The trail winds through valleys and over mountain passes. "
            "Pack enough water and warm clothes for the journey."
        ),
    }
    with patch.object(classifier, "_classify_with_llm", return_value=None):
        result = classifier.classify(doc)
    assert result["classification_confidence"] < 0.4


def test_confidence_range(classifier):
    """classification_confidence must always be in [0.0, 1.0]."""
    docs = [
        {"title": "", "description": "", "tags": [], "transcript_text": ""},
        {
            "title": "Espresso extraction ratio TDS EY refractometer",
            "description": "extraction yield barista championship grind tamping bloom",
            "tags": ["espresso", "extraction", "grind", "technique"],
            "transcript_text": "bitter sour channeling v60 moka aeropress natural washed",
        },
        {
            "title": "Hiking in mountains",
            "description": "Travel vlog with stunning scenery.",
            "tags": ["travel", "vlog"],
            "transcript_text": "Beautiful scenery and great views.",
        },
    ]
    with patch.object(classifier, "_classify_with_llm", return_value=None):
        for doc in docs:
            result = classifier.classify(doc)
            conf = result["classification_confidence"]
            assert 0.0 <= conf <= 1.0, (
                f"Confidence {conf} out of range for: {doc['title']!r}"
            )
