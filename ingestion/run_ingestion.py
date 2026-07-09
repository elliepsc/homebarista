"""
Ingestion Pipeline
==================
Offline script: YouTube → transcripts → clean → chunk → embed → ChromaDB.

Fixes vs. original plan:
1. Checkpointing per video: progress saved to ingestion/progress.json.
   Re-runs resume from last processed video — no more full restarts on failure.
2. TranscriptPreprocessor integrated: transcripts are cleaned and
   sentence-tokenized BEFORE chunking.
3. Quality filter: videos with classification_confidence < 0.4 are skipped.
4. is_informative() filter: non-coffee content skipped before indexing.
5. Quota tracking: YouTube API quota estimated and logged per channel.
6. Idempotent upserts: ChromaDB upserts by chunk_id (hash-based = stable).

Usage:
  python -m ingestion.run_ingestion           # full run
  python -m ingestion.run_ingestion --demo    # mock data, no API calls
  python -m ingestion.run_ingestion --dry-run # full pipeline, skip DB write
  python -m ingestion.run_ingestion --channel UCMb0O2CdPBNi-QqPk5T3gsQ
  python -m ingestion.run_ingestion --reset   # wipe progress + DB, start fresh
  python -m ingestion.run_ingestion --metadata-only
      # Data API only (channel/playlist video lists) — never touches the
      # transcript endpoint, so it's safe to run even while transcripts are
      # IP-blocked. Verifies playlist IDs and caches metadata for later runs.

Env overrides (see ultraplan_v4.md A.7 G1/G3):
  TRANSCRIPT_DELAY_S=10     pacing between transcript fetches (+/- jitter)
  TRANSCRIPT_JITTER_S=3
  MAX_VIDEOS_PER_RUN=30     hard cap on transcript fetches per run
  YOUTUBE_QUOTA_BUDGET=8000 stop before exhausting the 10k/day free quota
"""

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ingestion.transcript_preprocessor import TranscriptPreprocessor
from ingestion.embedder import Embedder
from pipeline.vector_store import VectorStore


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

CHANNELS_YAML       = Path("ingestion/channels.yaml")
PROGRESS_FILE       = Path("ingestion/progress.json")
REPORT_FILE         = Path("ingestion/ingestion_report.json")
MOCK_DOCS_FILE      = Path("data/mock_documents.json")
METADATA_CACHE_FILE = Path("ingestion/metadata_cache.json")

# YouTube API quota cost estimates (units)
QUOTA_SEARCH_PAGE   = 100   # search.list per page
QUOTA_VIDEOS_BATCH  = 1     # videos.list per item (batched by 50)
QUOTA_PLAYLIST_PAGE = 1     # playlistItems.list per page

# ------------------------------------------------------------------
# Transcript pacing / run caps — see ultraplan_v4.md A.7 G3.
# The 0.3s pacing below was calibrated for the official Data API, not the
# unofficial transcript endpoint; bursts on it triggered a real IpBlocked
# after ~30 fetches even from a residential IP. Defaults here are the
# normative recovery values (10s pacing + jitter, 30 videos/run cap).
# Read via os.getenv() inside run_ingestion(), not at import time — main()
# calls load_dotenv() lazily (not at module scope) to avoid leaking .env
# into the shared pytest process, same reasoning as DEMO_MODE.
# ------------------------------------------------------------------

DEFAULT_TRANSCRIPT_DELAY_S  = 10.0
DEFAULT_TRANSCRIPT_JITTER_S = 3.0
DEFAULT_MAX_VIDEOS_PER_RUN  = 30

# YouTube Data API free tier = 10,000 units/day. Budget stays comfortably
# under that so a single run can never exhaust the free daily quota (G1).
DEFAULT_YOUTUBE_QUOTA_BUDGET = 8000


# ------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------

def load_progress() -> dict:
    """Load checkpointing state. Returns empty dict if no progress file."""
    if PROGRESS_FILE.exists():
        with PROGRESS_FILE.open() as f:
            return json.load(f)
    return {"processed_video_ids": [], "last_updated": None}


def save_progress(progress: dict) -> None:
    """Persist checkpointing state after each video."""
    progress["last_updated"] = datetime.now(timezone.utc).isoformat()
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_FILE.open("w") as f:
        json.dump(progress, f, indent=2)


def reset_progress() -> None:
    """Wipe checkpointing state (use with --reset flag)."""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("Progress file cleared.")


# ------------------------------------------------------------------
# Metadata cache (decouples metadata fetch from transcript fetch — G3-c)
# ------------------------------------------------------------------

def load_metadata_cache() -> dict:
    """Cache of {channel_or_playlist_id: [raw_document, ...]} from the Data
    API, built by --metadata-only. Never touches the transcript endpoint, so
    it's safe to (re)build even while transcripts are IP-blocked."""
    if METADATA_CACHE_FILE.exists():
        with METADATA_CACHE_FILE.open() as f:
            return json.load(f)
    return {}


def save_metadata_cache(cache: dict) -> None:
    METADATA_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_CACHE_FILE.open("w") as f:
        json.dump(cache, f, indent=2)


# ------------------------------------------------------------------
# Mock data loader (demo mode)
# ------------------------------------------------------------------

def load_mock_documents() -> list[dict]:
    """Load pre-built mock documents for demo/CI mode (no API calls)."""
    if not MOCK_DOCS_FILE.exists():
        raise FileNotFoundError(
            f"Mock documents not found at {MOCK_DOCS_FILE}. "
            "Run Phase 0 setup first."
        )
    with MOCK_DOCS_FILE.open() as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} mock documents from {MOCK_DOCS_FILE}")
    return docs


# ------------------------------------------------------------------
# Metadata-only fetch (Data API, never IP-blocked) — decoupled from
# transcript fetching so metadata can be refreshed/verified even while
# the transcript endpoint is blocked (G3-c).
# ------------------------------------------------------------------

def fetch_metadata_only(
    channels_config: list[dict],
    playlists_config: list[dict],
    quota_budget: int = DEFAULT_YOUTUBE_QUOTA_BUDGET,
    demo_mode: bool = False,
) -> tuple[dict, int]:
    """
    Fetches raw video metadata for every channel + playlist via the official
    Data API only (search/videos/playlistItems — never IP-blocked) and
    returns a cache dict keyed by channel/playlist id. Also serves to verify
    a playlist ID actually resolves (a bad ID surfaces as an API error here,
    not silently during a later transcript run).
    """
    from ingestion.youtube_client import YouTubeClient

    yt_client = YouTubeClient(demo_mode=demo_mode)
    cache: dict = {}
    quota_used = 0

    for channel in channels_config:
        if quota_used + QUOTA_SEARCH_PAGE > quota_budget:
            print(f"  [QUOTA] Budget {quota_budget} would be exceeded — stopping before channel {channel['name']}.")
            break
        try:
            videos, q = yt_client.get_channel_videos(
                channel["id"], max_results=channel.get("max_videos", 50)
            )
            quota_used += q
            cache[channel["id"]] = [
                yt_client.build_document_object(v, channel.get("tags", []))
                for v in videos
            ]
            print(f"  [OK] {channel['name']}: {len(videos)} videos cached (quota: {q})")
        except Exception as exc:
            print(f"  [ERROR] {channel['name']} ({channel['id']}): {exc}")

    for playlist in playlists_config:
        if quota_used + QUOTA_PLAYLIST_PAGE > quota_budget:
            print(f"  [QUOTA] Budget {quota_budget} would be exceeded — stopping before playlist {playlist['name']}.")
            break
        try:
            videos, q = yt_client.get_playlist_videos(playlist["id"], max_results=50)
            quota_used += q
            cache[playlist["id"]] = [
                yt_client.build_document_object(v, playlist.get("tags", []))
                for v in videos
            ]
            print(f"  [OK] playlist {playlist['name']}: {len(videos)} videos cached (quota: {q}) — ID verified valid")
        except Exception as exc:
            print(f"  [ERROR] playlist {playlist['name']} ({playlist['id']}) — likely an invalid ID: {exc}")

    return cache, quota_used


# ------------------------------------------------------------------
# Real YouTube ingestion (requires API keys)
# ------------------------------------------------------------------

def fetch_real_documents(
    channels_config: list[dict],
    playlists_config: list[dict],
    progress: dict,
    channel_filter: str = None,
    demo_mode: bool = False,
    transcript_delay_s: float = DEFAULT_TRANSCRIPT_DELAY_S,
    transcript_jitter_s: float = DEFAULT_TRANSCRIPT_JITTER_S,
    max_videos_per_run: int = DEFAULT_MAX_VIDEOS_PER_RUN,
    quota_budget: int = DEFAULT_YOUTUBE_QUOTA_BUDGET,
) -> tuple[list[dict], int, dict | None]:
    """
    Fetch video metadata + transcripts from YouTube.
    Returns (documents, quota_used, stop_reason).

    stop_reason is None on a clean full pass, or a dict describing why the
    run stopped early: {"reason": "ip_blocked"|"max_videos_per_run"|
    "quota_budget", "channel": ..., "resume_video_id": ...}. On ip_blocked,
    the run aborts immediately — remaining videos are NEVER iterated over,
    since retrying while blocked just extends the block (G3-b).

    Documents have: source_id, title, channel, url, tags, transcript_text.
    """
    from ingestion.youtube_client import YouTubeClient
    from ingestion.transcript_fetcher import TranscriptFetcher, TranscriptFetchBlocked
    from ingestion.content_classifier import ContentClassifier

    yt_client = YouTubeClient(demo_mode=demo_mode)
    transcript_fetcher = TranscriptFetcher(demo_mode=demo_mode)
    classifier = ContentClassifier()
    metadata_cache = load_metadata_cache()

    processed_ids: set = set(progress.get("processed_video_ids", []))
    documents = []
    quota_used = 0
    videos_fetched_this_run = 0
    stop_reason: dict | None = None

    def process_video(doc: dict) -> str:
        """Returns 'ok' | 'skip' | 'blocked' | 'capped'. Mutates documents/processed_ids/quota via closure."""
        nonlocal videos_fetched_this_run, stop_reason
        video_id = doc["source_id"]

        if video_id in processed_ids:
            print(f"  [SKIP] Already processed: {video_id}")
            return "skip"

        if videos_fetched_this_run >= max_videos_per_run:
            stop_reason = {
                "reason": "max_videos_per_run",
                "resume_video_id": video_id,
            }
            print(f"  [CAP] MAX_VIDEOS_PER_RUN={max_videos_per_run} reached — stopping run cleanly.")
            return "capped"

        try:
            transcript, available = transcript_fetcher.fetch_transcript(video_id)
        except TranscriptFetchBlocked as exc:
            stop_reason = {
                "reason": "ip_blocked",
                "resume_video_id": video_id,
                "detail": str(exc),
            }
            print(f"  [BLOCKED] YouTube IP-blocked transcript requests at {video_id} — aborting run, checkpoint saved.")
            return "blocked"

        videos_fetched_this_run += 1

        if not available:
            print(f"  [SKIP] No transcript: {doc['title'][:50]}")
            processed_ids.add(video_id)
            save_progress({"processed_video_ids": list(processed_ids)})
            return "skip"

        doc["transcript_text"] = transcript

        # Quality filter: classify content domain
        classification = classifier.classify(doc)
        doc.update(classification)

        if classification.get("classification_confidence", 0) < 0.4:
            print(
                f"  [SKIP] Low confidence ({classification.get('classification_confidence', 0):.2f}): "
                f"{doc['title'][:50]}"
            )
            processed_ids.add(video_id)
            save_progress({"processed_video_ids": list(processed_ids)})
            return "skip"

        documents.append(doc)
        print(
            f"  [OK] {doc['title'][:50]} "
            f"domain={doc.get('domain')} method={doc.get('method')} "
            f"conf={classification.get('classification_confidence', 0):.2f}"
        )

        # Checkpoint after each successful video
        processed_ids.add(video_id)
        save_progress({"processed_video_ids": list(processed_ids)})
        return "ok"

    def pace() -> None:
        delay = transcript_delay_s + random.uniform(-transcript_jitter_s, transcript_jitter_s)
        time.sleep(max(0.0, delay))

    for channel in channels_config:
        if channel_filter and channel["id"] != channel_filter:
            continue
        if stop_reason:
            break

        print(f"\n{'='*50}")
        print(f"Processing channel: {channel['name']} (priority={channel['priority']})")
        print(f"{'='*50}")

        cached = metadata_cache.get(channel["id"])
        if cached is not None:
            raw_docs = cached
            print(f"  Using cached metadata: {len(raw_docs)} videos (0 quota — see --metadata-only)")
        else:
            if quota_used + QUOTA_SEARCH_PAGE > quota_budget:
                stop_reason = {"reason": "quota_budget", "channel": channel["id"]}
                print(f"  [QUOTA] Budget {quota_budget} would be exceeded — stopping before {channel['name']}.")
                break
            videos, q = yt_client.get_channel_videos(
                channel["id"], max_results=channel.get("max_videos", 50)
            )
            quota_used += q
            print(f"  Found {len(videos)} videos (quota used: {q})")
            raw_docs = [
                yt_client.build_document_object(v, channel.get("tags", []))
                for v in videos
            ]

        for doc in raw_docs:
            outcome = process_video(doc)
            if outcome in ("blocked", "capped"):
                stop_reason["channel"] = channel["id"]
                break
            if outcome == "ok":
                pace()  # rate limiting for the transcript endpoint only

        if stop_reason:
            break

    # Also process priority playlists
    if not stop_reason:
        for playlist in playlists_config:
            if channel_filter:
                break  # skip playlists in single-channel mode

            print(f"\nProcessing playlist: {playlist['name']}")
            cached = metadata_cache.get(playlist["id"])
            if cached is not None:
                raw_docs = cached
                print(f"  Using cached metadata: {len(raw_docs)} videos (0 quota — see --metadata-only)")
            else:
                if quota_used + QUOTA_PLAYLIST_PAGE > quota_budget:
                    stop_reason = {"reason": "quota_budget", "playlist": playlist["id"]}
                    print(f"  [QUOTA] Budget {quota_budget} would be exceeded — stopping before playlist {playlist['name']}.")
                    break
                videos, q = yt_client.get_playlist_videos(playlist["id"], max_results=50)
                quota_used += q
                raw_docs = [
                    yt_client.build_document_object(v, playlist.get("tags", []))
                    for v in videos
                ]

            for doc in raw_docs:
                outcome = process_video(doc)
                if outcome in ("blocked", "capped"):
                    stop_reason["playlist"] = playlist["id"]
                    break
                if outcome == "ok":
                    pace()

            if stop_reason:
                break

    return documents, quota_used, stop_reason


# ------------------------------------------------------------------
# Core pipeline
# ------------------------------------------------------------------

def run_ingestion(
    demo_mode: bool = False,
    dry_run: bool = False,
    channel_filter: str = None,
    reset: bool = False,
    metadata_only: bool = False,
    transcript_delay_s: float = DEFAULT_TRANSCRIPT_DELAY_S,
    transcript_jitter_s: float = DEFAULT_TRANSCRIPT_JITTER_S,
    max_videos_per_run: int = DEFAULT_MAX_VIDEOS_PER_RUN,
    quota_budget: int = DEFAULT_YOUTUBE_QUOTA_BUDGET,
) -> dict:
    """
    Full ingestion pipeline.
    Returns the ingestion report dict.
    """
    start_time = datetime.now(timezone.utc)

    # Load channels config
    if not CHANNELS_YAML.exists():
        raise FileNotFoundError(f"channels.yaml not found at {CHANNELS_YAML}")
    with CHANNELS_YAML.open() as f:
        config = yaml.safe_load(f)

    channels_config  = config.get("channels", [])
    playlists_config = config.get("playlists", [])

    # --metadata-only: Data API only, never touches the transcript endpoint —
    # safe to run even during an active IpBlocked. Builds/refreshes the
    # metadata cache and verifies playlist IDs resolve. No DB write.
    if metadata_only:
        cache, quota_used = fetch_metadata_only(
            channels_config, playlists_config,
            quota_budget=quota_budget, demo_mode=demo_mode,
        )
        save_metadata_cache(cache)
        print(f"\nMetadata cache written to {METADATA_CACHE_FILE} ({quota_used} quota units used).")
        return {"status": "metadata_only", "quota_used": quota_used, "channels_cached": list(cache.keys())}

    # Handle reset
    if reset:
        reset_progress()
        store = VectorStore(demo_mode=False)
        store.delete_collection()
        print("Full reset complete.")

    # Load progress for checkpointing
    progress = load_progress() if not demo_mode else {"processed_video_ids": []}

    # ------------------------------------------------------------------
    # Step 1: Fetch documents
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1 — Fetching documents")
    print("="*60)

    stop_reason = None
    if demo_mode:
        documents = load_mock_documents()
        quota_used = 0
    else:
        documents, quota_used, stop_reason = fetch_real_documents(
            channels_config, playlists_config, progress, channel_filter,
            demo_mode=demo_mode,
            transcript_delay_s=transcript_delay_s,
            transcript_jitter_s=transcript_jitter_s,
            max_videos_per_run=max_videos_per_run,
            quota_budget=quota_budget,
        )
        if stop_reason:
            print(f"\n[STOPPED] {stop_reason['reason']} — resume point: {stop_reason.get('resume_video_id', 'n/a')}")
            print("Checkpoint already saved to progress.json; re-run the same command later to resume.")
            if stop_reason["reason"] == "ip_blocked":
                print("Do NOT retry immediately — wait 24-72h, or use a fresh IP (see ultraplan_v4.md A.7 G3).")

    print(f"\nDocuments to process: {len(documents)}")
    if not documents:
        print("No new documents to index.")
        return {"status": "no_new_documents", "stop_reason": stop_reason, "quota_used": quota_used}

    # ------------------------------------------------------------------
    # Step 2: Clean + chunk
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2 — Cleaning transcripts + chunking")
    print("="*60)

    preprocessor = TranscriptPreprocessor()
    embedder = Embedder()

    all_chunks = []
    skipped_non_informative = 0

    for doc in documents:
        transcript = doc.get("transcript_text", doc.get("content", ""))

        if not transcript:
            print(f"  [SKIP] Empty transcript: {doc.get('title', '?')[:50]}")
            continue

        # is_informative filter catches non-coffee content that slipped through classification
        if not preprocessor.is_informative(transcript):
            print(f"  [SKIP] Non-coffee content: {doc.get('title', '?')[:50]}")
            skipped_non_informative += 1
            continue

        chunks = embedder.chunk_transcript(transcript, doc)
        all_chunks.extend(chunks)

        print(f"  [{len(chunks):3d} chunks] {doc.get('title', '?')[:60]}")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Skipped (non-informative): {skipped_non_informative}")

    if not all_chunks:
        print("No chunks generated. Check transcript quality.")
        return {"status": "no_chunks"}

    # ------------------------------------------------------------------
    # Step 3: Embed
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3 — Embedding chunks")
    print("="*60)

    embedded_chunks = embedder.embed_batch(all_chunks, verbose=True)
    print(f"Embedded {len(embedded_chunks)} chunks.")

    # ------------------------------------------------------------------
    # Step 4: Write to ChromaDB
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"STEP 4 — Writing to ChromaDB {'[DRY RUN - SKIPPED]' if dry_run else ''}")
    print("="*60)

    stats = {}
    if not dry_run:
        store = VectorStore(demo_mode=demo_mode)
        store.add_chunks(embedded_chunks)
        stats = store.get_stats()
        print(f"Collection stats: {json.dumps(stats, indent=2)}")
    else:
        print("Dry run — skipping DB write.")
        stats = {"total_chunks": len(embedded_chunks), "dry_run": True}

    # ------------------------------------------------------------------
    # Step 5: Write ingestion report
    # ------------------------------------------------------------------
    end_time = datetime.now(timezone.utc)
    duration_seconds = (end_time - start_time).total_seconds()

    # Compute distributions from all_chunks (don't need DB for this)
    domain_dist: dict[str, int] = {}
    method_dist: dict[str, int] = {}
    channel_dist: dict[str, int] = {}

    for chunk in all_chunks:
        d = chunk.get("domain", "unknown")
        m = chunk.get("method", "unknown")
        c = chunk.get("channel", "unknown")
        domain_dist[d]   = domain_dist.get(d, 0) + 1
        method_dist[m]   = method_dist.get(m, 0) + 1
        channel_dist[c]  = channel_dist.get(c, 0) + 1

    report = {
        "run_at": start_time.isoformat(),
        "duration_seconds": round(duration_seconds, 1),
        "mode": "demo" if demo_mode else ("dry_run" if dry_run else "live"),
        "channel_filter": channel_filter,
        "videos_fetched": len(documents),
        "videos_skipped_non_informative": skipped_non_informative,
        "total_chunks_generated": len(all_chunks),
        "total_chunks_indexed": 0 if dry_run else stats.get("total_chunks", 0),
        "domain_distribution": domain_dist,
        "method_distribution": method_dist,
        "chunks_per_channel": channel_dist,
        "quota_used": quota_used,
        "stop_reason": stop_reason,
        "embedding_model": embedder.model_name,
        "chunk_size_tokens": embedder.max_tokens,
        "overlap_tokens": embedder.overlap_tokens,
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_FILE.open("w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*60)
    print("INGESTION COMPLETE")
    print("="*60)
    print(f"  Videos processed : {report['videos_fetched']}")
    print(f"  Chunks indexed   : {report['total_chunks_indexed']}")
    print(f"  Duration         : {duration_seconds:.0f}s")
    print(f"  Quota used       : {quota_used} units")
    print(f"  Report saved     : {REPORT_FILE}")

    if not dry_run and not demo_mode:
        print(
            "\nNEXT STEP: Export snapshot for Streamlit Cloud deployment:\n"
            "  python -m pipeline.vector_store --export"
        )

    return report


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    # Entry point only — YOUTUBE_API_KEY, DEMO_MODE, LLM_* (see .env.example)
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="HomeBarista ingestion pipeline — YouTube → ChromaDB"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use mock_documents.json instead of YouTube API (CI-safe)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run full pipeline but skip ChromaDB write"
    )
    parser.add_argument(
        "--channel", type=str, default=None,
        help="Ingest only one channel (by channel ID)"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe checkpointing progress and ChromaDB, then re-ingest from scratch"
    )
    parser.add_argument(
        "--metadata-only", action="store_true",
        help=(
            "Fetch/refresh channel + playlist metadata via the Data API only "
            "(never touches the transcript endpoint, safe during an IP block). "
            "Writes ingestion/metadata_cache.json, no DB write."
        ),
    )
    args = parser.parse_args()

    # Env overrides read here (after load_dotenv), not as function defaults —
    # function defaults are bound at import time and would miss .env values.
    run_ingestion(
        demo_mode=args.demo,
        dry_run=args.dry_run,
        channel_filter=args.channel,
        reset=args.reset,
        metadata_only=args.metadata_only,
        transcript_delay_s=float(os.getenv("TRANSCRIPT_DELAY_S", DEFAULT_TRANSCRIPT_DELAY_S)),
        transcript_jitter_s=float(os.getenv("TRANSCRIPT_JITTER_S", DEFAULT_TRANSCRIPT_JITTER_S)),
        max_videos_per_run=int(os.getenv("MAX_VIDEOS_PER_RUN", DEFAULT_MAX_VIDEOS_PER_RUN)),
        quota_budget=int(os.getenv("YOUTUBE_QUOTA_BUDGET", DEFAULT_YOUTUBE_QUOTA_BUDGET)),
    )


if __name__ == "__main__":
    main()
