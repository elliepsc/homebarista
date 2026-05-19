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
"""

import argparse
import json
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

CHANNELS_YAML    = Path("ingestion/channels.yaml")
PROGRESS_FILE    = Path("ingestion/progress.json")
REPORT_FILE      = Path("ingestion/ingestion_report.json")
MOCK_DOCS_FILE   = Path("data/mock_documents.json")

# YouTube API quota cost estimates (units)
QUOTA_SEARCH_PAGE   = 100   # search.list per page
QUOTA_VIDEOS_BATCH  = 1     # videos.list per item (batched by 50)
QUOTA_PLAYLIST_PAGE = 1     # playlistItems.list per page


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
# Real YouTube ingestion (requires API keys)
# ------------------------------------------------------------------

def fetch_real_documents(
    channels_config: list[dict],
    playlists_config: list[dict],
    progress: dict,
    channel_filter: str = None,
) -> tuple[list[dict], int]:
    """
    Fetch video metadata + transcripts from YouTube.
    Returns (documents, quota_used).

    Documents have: source_id, title, channel, url, tags, transcript_text.
    """
    from ingestion.youtube_client import YouTubeClient
    from ingestion.transcript_fetcher import TranscriptFetcher
    from ingestion.content_classifier import ContentClassifier

    yt_client = YouTubeClient()
    transcript_fetcher = TranscriptFetcher()
    classifier = ContentClassifier()

    processed_ids: set = set(progress.get("processed_video_ids", []))
    documents = []
    quota_used = 0

    for channel in channels_config:
        if channel_filter and channel["id"] != channel_filter:
            continue

        print(f"\n{'='*50}")
        print(f"Processing channel: {channel['name']} (priority={channel['priority']})")
        print(f"{'='*50}")

        # Fetch video list
        videos, q = yt_client.get_channel_videos(
            channel["id"], max_results=channel.get("max_videos", 50)
        )
        quota_used += q
        print(f"  Found {len(videos)} videos (quota used: {q})")

        # Build document objects
        raw_docs = [
            yt_client.build_document_object(v, channel.get("tags", []))
            for v in videos
        ]

        for doc in raw_docs:
            video_id = doc["source_id"]

            # Skip already-processed videos (checkpointing)
            if video_id in processed_ids:
                print(f"  [SKIP] Already processed: {video_id}")
                continue

            # Fetch transcript
            transcript, available = transcript_fetcher.fetch_transcript(video_id)
            if not available:
                print(f"  [SKIP] No transcript: {doc['title'][:50]}")
                processed_ids.add(video_id)
                save_progress({"processed_video_ids": list(processed_ids)})
                continue

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
                continue

            documents.append(doc)
            print(
                f"  [OK] {doc['title'][:50]} "
                f"domain={doc.get('domain')} method={doc.get('method')} "
                f"conf={classification.get('classification_confidence', 0):.2f}"
            )

            # Checkpoint after each successful video
            processed_ids.add(video_id)
            save_progress({"processed_video_ids": list(processed_ids)})

            # Rate limiting for transcript API
            time.sleep(0.3)

    # Also process priority playlists
    for playlist in playlists_config:
        if channel_filter:
            continue  # skip playlists in single-channel mode

        print(f"\nProcessing playlist: {playlist['name']}")
        videos, q = yt_client.get_playlist_videos(
            playlist["id"], max_results=50
        )
        quota_used += q

        for v in videos:
            if v["videoId"] in processed_ids:
                continue
            # Same flow as above (abbreviated — delegate to channel loop next run)
            processed_ids.add(v["videoId"])

    return documents, quota_used


# ------------------------------------------------------------------
# Core pipeline
# ------------------------------------------------------------------

def run_ingestion(
    demo_mode: bool = False,
    dry_run: bool = False,
    channel_filter: str = None,
    reset: bool = False,
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

    if demo_mode:
        documents = load_mock_documents()
        quota_used = 0
    else:
        documents, quota_used = fetch_real_documents(
            channels_config, playlists_config, progress, channel_filter
        )

    print(f"\nDocuments to process: {len(documents)}")
    if not documents:
        print("No new documents to index.")
        return {"status": "no_new_documents"}

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
    args = parser.parse_args()

    run_ingestion(
        demo_mode=args.demo,
        dry_run=args.dry_run,
        channel_filter=args.channel,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
