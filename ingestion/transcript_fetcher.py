"""
YouTube transcript fetcher using youtube-transcript-api.
"""

import os
import time
import logging

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
)
from youtube_transcript_api._errors import RequestBlocked

logger = logging.getLogger(__name__)


class TranscriptFetchBlocked(Exception):
    """
    YouTube is IP-blocking transcript requests (RequestBlocked/IpBlocked).
    Distinct from a normal "no transcript for this video" skip: callers must
    abort the run immediately (checkpoint + stop), never keep iterating —
    retrying blocked requests just extends the block.
    """

    def __init__(self, video_id: str, cause: Exception):
        self.video_id = video_id
        self.cause = cause
        super().__init__(f"Transcript fetch blocked by YouTube for video {video_id}: {cause}")


class TranscriptFetcher:
    def __init__(self, demo_mode: bool | None = None):
        """
        demo_mode: explicit override (e.g. from run_ingestion.py's --demo CLI
        flag). When None, falls back to the DEMO_MODE env var — but an
        explicit caller-supplied value always wins, so a live ingestion run
        can't be silently forced into demo mode by .env's DEMO_MODE=true
        (the Streamlit app's default).
        """
        if demo_mode is None:
            demo_mode = os.getenv("DEMO_MODE", "").lower() == "true"
        self.demo_mode = demo_mode

    def fetch_transcript(self, video_id: str) -> tuple[str | None, bool]:
        """
        Returns (transcript_text, is_available).
        Prefers English transcripts. Returns the full, untruncated text.

        Raises TranscriptFetchBlocked if YouTube is IP-blocking requests —
        this is NOT the same as "no transcript available" and must not be
        swallowed into (None, False).
        """
        if self.demo_mode:
            return None, False

        try:
            transcript_list = YouTubeTranscriptApi().list(video_id)
            try:
                transcript = transcript_list.find_transcript(["en"])
            except NoTranscriptFound:
                try:
                    transcript = transcript_list.find_generated_transcript(["en"])
                except NoTranscriptFound:
                    return None, False

            snippets = transcript.fetch()
            text = " ".join(s.text for s in snippets)
            return text, True

        except RequestBlocked as exc:
            raise TranscriptFetchBlocked(video_id, exc) from exc
        except Exception:
            return None, False

    def fetch_batch(self, video_ids: list[str]) -> dict[str, str]:
        """
        Fetches transcripts for multiple videos.
        Rate limiting: 0.3s between calls.
        Skips videos without a transcript silently.
        Logs progress every 10 videos.
        Returns {video_id: transcript_text} for successful fetches only.

        Propagates TranscriptFetchBlocked instead of catching it — an IP
        block must stop the batch immediately, not be treated as a per-video skip.
        """
        if self.demo_mode:
            return {}

        results: dict[str, str] = {}

        for i, video_id in enumerate(video_ids):
            if i > 0:
                time.sleep(0.3)

            transcript, available = self.fetch_transcript(video_id)
            if available and transcript:
                results[video_id] = transcript

            if (i + 1) % 10 == 0:
                logger.info(
                    "Transcript fetch progress: %d/%d (fetched: %d)",
                    i + 1, len(video_ids), len(results),
                )

        return results
