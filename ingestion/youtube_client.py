"""
YouTube Data API v3 client for fetching channel and playlist videos.
"""

import os
import logging

logger = logging.getLogger(__name__)

# ISO 8601 duration → seconds (e.g. PT4M13S → 253)
import re

def _iso8601_duration_to_seconds(duration: str) -> int:
    match = re.fullmatch(
        r"P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?",
        duration or "",
    )
    if not match:
        return 0
    days, hours, minutes, seconds = (int(x or 0) for x in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


class YouTubeClient:
    def __init__(self):
        if os.getenv("DEMO_MODE", "").lower() == "true":
            self.service = None
        else:
            from googleapiclient.discovery import build
            self.service = build(
                "youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY")
            )
        self.quota_used = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_channel_videos(
        self, channel_id: str, max_results: int = 50
    ) -> tuple[list[dict], int]:
        """
        Returns (videos, quota_used) for a channel.
        Each video: {videoId, title, description, tags, duration_seconds, channelTitle}
        Handles pagination up to max_results. Logs quota.
        Quota cost: ~100 units per search page + 1 unit per video (batched by 50).
        """
        if os.getenv("DEMO_MODE", "").lower() == "true":
            logger.info("DEMO_MODE: skipping YouTube API call for channel %s", channel_id)
            return [], 0

        quota_used = 0
        video_ids: list[str] = []
        page_token = None

        while len(video_ids) < max_results:
            batch = min(50, max_results - len(video_ids))
            request = self.service.search().list(
                part="id",
                channelId=channel_id,
                type="video",
                maxResults=batch,
                pageToken=page_token,
            )
            response = request.execute()
            quota_used += 100  # search.list costs 100 units per page

            for item in response.get("items", []):
                vid = item.get("id", {}).get("videoId")
                if vid:
                    video_ids.append(vid)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        videos, detail_quota = self._fetch_video_details(video_ids)
        quota_used += detail_quota

        self.quota_used += quota_used
        logger.info(
            "Channel %s: %d videos fetched, %d quota units used",
            channel_id, len(videos), quota_used,
        )
        return videos, quota_used

    def get_playlist_videos(
        self, playlist_id: str, max_results: int = 50
    ) -> tuple[list[dict], int]:
        """Same format as get_channel_videos."""
        if os.getenv("DEMO_MODE", "").lower() == "true":
            logger.info("DEMO_MODE: skipping YouTube API call for playlist %s", playlist_id)
            return [], 0

        quota_used = 0
        video_ids: list[str] = []
        page_token = None

        while len(video_ids) < max_results:
            batch = min(50, max_results - len(video_ids))
            request = self.service.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=batch,
                pageToken=page_token,
            )
            response = request.execute()
            quota_used += 1  # playlistItems.list costs 1 unit per page

            for item in response.get("items", []):
                vid = item.get("contentDetails", {}).get("videoId")
                if vid:
                    video_ids.append(vid)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        videos, detail_quota = self._fetch_video_details(video_ids)
        quota_used += detail_quota

        self.quota_used += quota_used
        logger.info(
            "Playlist %s: %d videos fetched, %d quota units used",
            playlist_id, len(videos), quota_used,
        )
        return videos, quota_used

    def build_document_object(self, raw: dict, channel_tags: list[str]) -> dict:
        """
        Maps a raw video dict (from get_channel_videos / get_playlist_videos)
        to the document schema used by the ingestion pipeline.

        Returns: source_id, title, channel, url, description, tags, duration_seconds
        """
        video_id = raw["videoId"]
        api_tags = raw.get("tags") or []
        merged_tags = list(dict.fromkeys(channel_tags + api_tags))  # deduplicate, preserve order

        return {
            "source_id": video_id,
            "title": raw.get("title", ""),
            "channel": raw.get("channelTitle", ""),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "description": raw.get("description", ""),
            "tags": merged_tags,
            "duration_seconds": raw.get("duration_seconds", 0),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_video_details(self, video_ids: list[str]) -> tuple[list[dict], int]:
        """
        Batch-fetch video details via videos.list (50 per call = 1 quota unit each).
        Returns (videos, quota_used).
        """
        if not video_ids:
            return [], 0

        quota_used = 0
        videos: list[dict] = []

        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i : i + 50]
            request = self.service.videos().list(
                part="snippet,contentDetails",
                id=",".join(batch_ids),
                maxResults=50,
            )
            response = request.execute()
            quota_used += len(batch_ids)  # 1 unit per video

            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                content_details = item.get("contentDetails", {})
                videos.append({
                    "videoId": item["id"],
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "tags": snippet.get("tags", []),
                    "channelTitle": snippet.get("channelTitle", ""),
                    "duration_seconds": _iso8601_duration_to_seconds(
                        content_details.get("duration", "")
                    ),
                })

        return videos, quota_used
