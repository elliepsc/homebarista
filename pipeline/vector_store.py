"""
Vector Store
============
ChromaDB wrapper with a deployment-safe snapshot strategy.

THE PROBLEM:
Streamlit Cloud has an ephemeral filesystem. A ChromaDB persisted at
data/chroma_db/ is wiped on every app restart, leaving the app with
an empty knowledge base.

THE SOLUTION — two-mode architecture:
1. LOCAL / CI mode: ChromaDB in persist mode (fast, normal dev workflow)
2. DEPLOYED mode: ChromaDB loaded from a pre-built snapshot committed to git.

The snapshot is a single SQLite file (data/chroma_snapshot.db) exported
from ChromaDB's underlying DuckDB/SQLite engine.

HOW IT WORKS:
- After ingestion locally: run `python -m pipeline.vector_store --export`
  → writes data/chroma_snapshot.db (typically 20-80 MB for 300+ docs)
- Commit data/chroma_snapshot.db to git (remove from .gitignore)
- On Streamlit Cloud startup: vector_store detects missing chroma_db/,
  restores from snapshot into a temp directory
- App works immediately without re-ingestion

ALTERNATIVE (if snapshot too large for git):
Use Chroma Cloud free tier — swap CHROMA_MODE=cloud in .env,
set CHROMA_CLOUD_API_KEY + CHROMA_CLOUD_COLLECTION_ID.
"""

import os
import shutil
import json
import tempfile
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings


COLLECTION_NAME = "barista_knowledge"
SNAPSHOT_PATH = Path("data/chroma_snapshot")  # directory snapshot
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db"))


class VectorStore:
    """
    ChromaDB wrapper with deployment-safe initialization.
    Automatically restores from snapshot when persist dir is missing.
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        snapshot_path: Optional[Path] = None,
        demo_mode: bool = False,
    ):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.snapshot_path = snapshot_path or SNAPSHOT_PATH
        self.demo_mode = demo_mode
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = self._init_client()
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ------------------------------------------------------------------
    # Initialization & snapshot restore
    # ------------------------------------------------------------------

    def _init_client(self) -> chromadb.ClientAPI:
        """
        Initialize ChromaDB client.
        If the persist dir doesn't exist but a snapshot does, restore it first.
        This handles the Streamlit Cloud ephemeral filesystem scenario.
        """
        if self.demo_mode:
            # In-memory client for CI / demo mode (no filesystem needed)
            print("VectorStore: running in-memory (demo mode)")
            return chromadb.Client()

        if not self.persist_dir.exists():
            if self.snapshot_path.exists():
                print(f"VectorStore: persist dir missing, restoring from snapshot...")
                self._restore_from_snapshot()
            else:
                print("VectorStore: no persist dir and no snapshot found. Starting empty.")
                self.persist_dir.mkdir(parents=True, exist_ok=True)

        print(f"VectorStore: loading from {self.persist_dir}")
        return chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    def _restore_from_snapshot(self) -> None:
        """
        Copy the committed snapshot directory into the ephemeral persist dir.
        On Streamlit Cloud this runs once per session startup.
        """
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if self.snapshot_path.is_dir():
            shutil.copytree(str(self.snapshot_path), str(self.persist_dir), dirs_exist_ok=True)
            print(f"VectorStore: restored {self.snapshot_path} → {self.persist_dir}")
        else:
            raise FileNotFoundError(
                f"Snapshot not found at {self.snapshot_path}. "
                "Run: python -m pipeline.vector_store --export"
            )

    def export_snapshot(self) -> None:
        """
        Export the current ChromaDB to a snapshot directory for git commit.
        Run this after ingestion to prepare for Streamlit Cloud deployment.

        Usage: python -m pipeline.vector_store --export
        """
        if not self.persist_dir.exists():
            raise RuntimeError("No local ChromaDB to export. Run ingestion first.")

        if self.snapshot_path.exists():
            shutil.rmtree(str(self.snapshot_path))

        shutil.copytree(str(self.persist_dir), str(self.snapshot_path))

        # Report size
        total_size = sum(
            f.stat().st_size for f in self.snapshot_path.rglob("*") if f.is_file()
        )
        size_mb = total_size / (1024 * 1024)
        print(f"Snapshot exported to {self.snapshot_path} ({size_mb:.1f} MB)")
        print(f"Collection stats: {self.get_stats()}")

        if size_mb > 100:
            print(
                "WARNING: Snapshot > 100MB. Consider using Chroma Cloud instead.\n"
                "Set CHROMA_MODE=cloud in .env and CHROMA_CLOUD_API_KEY."
            )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_chunks(self, embedded_chunks: list[dict]) -> None:
        """
        Upsert embedded chunks into ChromaDB.
        Idempotent — safe to call multiple times (deduplicates by chunk_id).
        """
        if not embedded_chunks:
            return

        ids = [c["chunk_id"] for c in embedded_chunks]
        embeddings = [c["embedding"] for c in embedded_chunks]
        documents = [c["text"] for c in embedded_chunks]
        metadatas = [
            {
                "source_id": c["source_id"],
                "title": c["title"],
                "channel": c["channel"],
                "url": c["url"],
                "domain": c["domain"],
                "method": c["method"],
                "difficulty": c["difficulty"],
            }
            for c in embedded_chunks
        ]

        # Upsert in batches of 500 (ChromaDB batch limit)
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
        print(f"Upserted {len(ids)} chunks into '{COLLECTION_NAME}'")

    def delete_collection(self) -> None:
        """Wipe and recreate the collection. Useful for full re-ingestion."""
        self.client.delete_collection(COLLECTION_NAME)
        self._collection = None
        print(f"Collection '{COLLECTION_NAME}' deleted.")

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(
        self,
        embedding: list[float],
        n_results: int = 15,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Query ChromaDB by embedding vector.
        Returns a list of result dicts with text, metadata, and distance.

        Args:
            embedding: query embedding from Embedder.embed_query()
            n_results: number of candidates to retrieve before re-ranking
            where: ChromaDB metadata filter, e.g. {"method": {"$in": ["espresso", "general"]}}
        """
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": min(n_results, self.collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        # Unpack ChromaDB nested list format
        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            output.append(
                {
                    "chunk_id": doc_id,
                    "text": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    # distance in cosine space: 0 = identical, 2 = opposite
                    "semantic_score": max(0.0, 1.0 - results["distances"][0][i]),
                    **results["metadatas"][0][i],
                }
            )
        return output

    def get_stats(self) -> dict:
        """Return collection statistics for monitoring and ingestion reports."""
        count = self.collection.count()
        if count == 0:
            return {"total_chunks": 0}

        # Sample up to 1000 items to compute distributions
        sample = self.collection.get(
            limit=min(1000, count),
            include=["metadatas"],
        )

        domain_dist: dict[str, int] = {}
        method_dist: dict[str, int] = {}
        sources: set[str] = set()
        channels: set[str] = set()

        for meta in sample["metadatas"]:
            domain_dist[meta.get("domain", "unknown")] = (
                domain_dist.get(meta.get("domain", "unknown"), 0) + 1
            )
            method_dist[meta.get("method", "unknown")] = (
                method_dist.get(meta.get("method", "unknown"), 0) + 1
            )
            sources.add(meta.get("source_id", ""))
            channels.add(meta.get("channel", ""))

        return {
            "total_chunks": count,
            "domain_distribution": domain_dist,
            "method_distribution": method_dist,
            "sources_indexed": len(sources),
            "channels_indexed": len(channels),
        }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    store = VectorStore()
    if "--export" in sys.argv:
        store.export_snapshot()
    elif "--stats" in sys.argv:
        stats = store.get_stats()
        print(json.dumps(stats, indent=2))
    else:
        print("Usage: python -m pipeline.vector_store [--export | --stats]")
        print("  --export: snapshot current ChromaDB for Streamlit Cloud deployment")
        print("  --stats:  print collection statistics")
