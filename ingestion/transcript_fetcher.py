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

logger = logging.getLogger(__name__)


class TranscriptFetcher:
    def fetch_transcript(self, video_id: str) -> tuple[str | None, bool]:
        """
        Returns (transcript_text, is_available).
        Prefers English transcripts. Returns the full, untruncated text.
        """
        if os.getenv("DEMO_MODE", "").lower() == "true":
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

        except Exception:
            return None, False

    def fetch_batch(self, video_ids: list[str]) -> dict[str, str]:
        """
        Fetches transcripts for multiple videos.
        Rate limiting: 0.3s between calls.
        Skips videos without a transcript silently.
        Logs progress every 10 videos.
        Returns {video_id: transcript_text} for successful fetches only.
        """
        if os.getenv("DEMO_MODE", "").lower() == "true":
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
