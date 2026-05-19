import pytest
from unittest.mock import patch, MagicMock


_TRANSCRIPT_API = "ingestion.transcript_fetcher.YouTubeTranscriptApi"


# ------------------------------------------------------------------
# YouTubeClient tests
# ------------------------------------------------------------------

def test_demo_mode_returns_empty(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    from ingestion.youtube_client import YouTubeClient
    client = YouTubeClient()
    videos, quota = client.get_channel_videos("UCfake123", max_results=10)
    assert videos == []
    assert quota == 0


def test_demo_mode_playlist_returns_empty(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    from ingestion.youtube_client import YouTubeClient
    client = YouTubeClient()
    videos, quota = client.get_playlist_videos("PLfake123", max_results=10)
    assert videos == []
    assert quota == 0


def test_build_document_object():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("DEMO_MODE", "true")
    from ingestion.youtube_client import YouTubeClient
    client = YouTubeClient()
    monkeypatch.undo()

    raw = {
        "videoId": "abc123",
        "title": "How to dial in espresso",
        "description": "Tips for espresso extraction",
        "tags": ["espresso", "dialing"],
        "channelTitle": "James Hoffmann",
        "duration_seconds": 720,
    }
    doc = client.build_document_object(raw, ["filter", "grind"])

    assert doc["source_id"] == "abc123"
    assert doc["title"] == "How to dial in espresso"
    assert doc["channel"] == "James Hoffmann"
    assert doc["url"] == "https://www.youtube.com/watch?v=abc123"
    assert doc["description"] == "Tips for espresso extraction"
    assert doc["duration_seconds"] == 720
    assert doc["tags"][:2] == ["filter", "grind"]
    assert "espresso" in doc["tags"]
    assert "dialing" in doc["tags"]
    assert len(doc["tags"]) == len(set(doc["tags"]))


def test_build_document_object_deduplicates_tags():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("DEMO_MODE", "true")
    from ingestion.youtube_client import YouTubeClient
    client = YouTubeClient()
    monkeypatch.undo()

    raw = {
        "videoId": "xyz",
        "title": "T",
        "description": "",
        "tags": ["espresso", "grind"],
        "channelTitle": "C",
        "duration_seconds": 0,
    }
    doc = client.build_document_object(raw, ["grind", "filter"])
    assert doc["tags"].count("grind") == 1


# ------------------------------------------------------------------
# TranscriptFetcher tests
# ------------------------------------------------------------------

def test_transcript_fetch_missing():
    """Video without a transcript returns (None, False)."""
    from ingestion.transcript_fetcher import TranscriptFetcher
    from youtube_transcript_api import NoTranscriptFound

    fetcher = TranscriptFetcher()

    with patch(_TRANSCRIPT_API) as mock_api:
        mock_api.list_transcripts.side_effect = NoTranscriptFound(
            "vid", [], {}
        )
        result_text, available = fetcher.fetch_transcript(
            "no_transcript_video"
        )

    assert available is False
    assert result_text is None


def test_transcript_fetch_disabled():
    """TranscriptsDisabled also returns (None, False)."""
    from ingestion.transcript_fetcher import TranscriptFetcher
    from youtube_transcript_api import TranscriptsDisabled

    fetcher = TranscriptFetcher()

    with patch(_TRANSCRIPT_API) as mock_api:
        mock_api.list_transcripts.side_effect = TranscriptsDisabled("vid")
        result_text, available = fetcher.fetch_transcript("disabled_video")

    assert available is False
    assert result_text is None


def test_transcript_fetch_success():
    from ingestion.transcript_fetcher import TranscriptFetcher

    fetcher = TranscriptFetcher()

    mock_transcript = MagicMock()
    mock_transcript.fetch.return_value = [
        {"text": "Hello coffee world", "start": 0.0, "duration": 2.0},
        {"text": "grind size matters", "start": 2.0, "duration": 2.0},
    ]
    mock_list = MagicMock()
    mock_list.find_transcript.return_value = mock_transcript

    with patch(_TRANSCRIPT_API) as mock_api:
        mock_api.list_transcripts.return_value = mock_list
        text, available = fetcher.fetch_transcript("good_video")

    assert available is True
    assert "Hello coffee world" in text
    assert "grind size matters" in text


def test_demo_mode_fetch_batch_returns_empty(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    from ingestion.transcript_fetcher import TranscriptFetcher
    fetcher = TranscriptFetcher()
    result = fetcher.fetch_batch(["vid1", "vid2"])
    assert result == {}
