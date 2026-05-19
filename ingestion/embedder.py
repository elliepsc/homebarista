"""
Embedder
========
Converts cleaned transcript chunks into vector embeddings for ChromaDB.

Key design decisions vs. original plan:
- Chunk IDs are now SHA-256 content hashes (not positional indices).
  This makes the eval dataset stable across re-ingestions.
- Chunking is delegated to TranscriptPreprocessor (sentence-aware),
  not done here with a naive token split.
- Model: all-MiniLM-L6-v2 (English corpus + English queries = optimal).
  Switch to paraphrase-multilingual-MiniLM-L12-v2 if multilingual queries
  are needed in the future.
"""

import hashlib
from typing import Optional
from sentence_transformers import SentenceTransformer

from ingestion.transcript_preprocessor import TranscriptPreprocessor

# ------------------------------------------------------------------
# Model selection
# ------------------------------------------------------------------
# For English-only corpus + English queries: all-MiniLM-L6-v2 (384 dims)
# For multilingual queries: paraphrase-multilingual-MiniLM-L12-v2 (384 dims)
DEFAULT_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


class Embedder:
    """
    Wraps sentence-transformers for chunk-level and query-level embedding.
    Handles chunking via TranscriptPreprocessor (sentence-aware).
    Generates stable, content-based chunk IDs.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_tokens: int = 400,
        overlap_tokens: int = 80,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        self._model: Optional[SentenceTransformer] = None
        self._preprocessor = TranscriptPreprocessor()

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model (avoids loading on import)."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk_transcript(
        self,
        transcript: str,
        source_doc: dict,
    ) -> list[dict]:
        """
        Split a cleaned transcript into sentence-aware chunks.
        Each chunk gets a stable SHA-256-based ID.

        Args:
            transcript: raw transcript text (will be cleaned internally)
            source_doc: document metadata (source_id, title, channel, etc.)

        Returns:
            List of chunk dicts with text + metadata, ready for embed_batch.
        """
        # Step 1: clean the transcript
        cleaned = self._preprocessor.clean(transcript)

        # Step 2: skip non-coffee content entirely
        if not self._preprocessor.is_informative(cleaned):
            print(f"  [SKIP] Non-informative content: {source_doc.get('title', 'unknown')}")
            return []

        # Step 3: sentence-aware split
        chunks_text = self._preprocessor.sentence_aware_chunks(
            cleaned,
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
        )

        # Step 4: build chunk dicts with stable IDs
        chunks = []
        for text in chunks_text:
            chunk_id = self._make_chunk_id(source_doc["source_id"], text)
            chunk = {
                "chunk_id": chunk_id,
                "text": text,
                # Metadata passed through to ChromaDB
                "source_id": source_doc["source_id"],
                "title": source_doc.get("title", ""),
                "channel": source_doc.get("channel", ""),
                "url": source_doc.get("url", ""),
                "domain": source_doc.get("domain", "general"),
                "method": source_doc.get("method", "general"),
                "difficulty": source_doc.get("difficulty", "intermediate"),
            }
            chunks.append(chunk)

        return chunks

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_chunk(self, chunk: dict) -> dict:
        """Embed a single chunk. Returns chunk + embedding vector."""
        embedding = self.model.encode(chunk["text"], normalize_embeddings=True).tolist()
        return {**chunk, "embedding": embedding}

    def embed_query(self, query: str) -> list[float]:
        """Embed a user query for retrieval."""
        return self.model.encode(query, normalize_embeddings=True).tolist()

    def embed_batch(self, chunks: list[dict], verbose: bool = True) -> list[dict]:
        """
        Embed a list of chunks in batches.
        Returns chunks with 'embedding' field added.
        """
        if not chunks:
            return []

        texts = [c["text"] for c in chunks]
        total = len(texts)

        embedded_chunks = []
        for i in range(0, total, BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]
            batch_chunks = chunks[i : i + BATCH_SIZE]

            embeddings = self.model.encode(
                batch_texts,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            for chunk, embedding in zip(batch_chunks, embeddings):
                embedded_chunks.append({**chunk, "embedding": embedding})

            if verbose:
                done = min(i + BATCH_SIZE, total)
                print(f"  Embedded {done}/{total} chunks", end="\r")

        if verbose:
            print(f"  Embedded {total}/{total} chunks — done.")

        return embedded_chunks

    # ------------------------------------------------------------------
    # Stable chunk ID
    # ------------------------------------------------------------------

    @staticmethod
    def _make_chunk_id(source_id: str, text: str) -> str:
        """
        Generate a stable chunk ID from the source document ID and
        the first 200 characters of the chunk text.

        WHY HASH-BASED IDs:
        Positional IDs like "hoffmann_chunk_3" break the eval dataset
        whenever re-ingestion happens (new videos, different chunking params).
        Hash-based IDs are stable as long as the content doesn't change.

        Format: {source_id}_{sha256_hex[:8]}
        Example: UCMb0O2CdPBNi-QqPk5T3gsQ_video123_a3f9b1c2
        """
        content_fingerprint = text[:200].strip().encode("utf-8")
        hash_hex = hashlib.sha256(content_fingerprint).hexdigest()[:8]
        return f"{source_id}_{hash_hex}"


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    embedder = Embedder()

    sample_doc = {
        "source_id": "test_video_001",
        "title": "Why Your Espresso Is Bitter",
        "channel": "James Hoffmann",
        "url": "https://youtube.com/watch?v=test",
        "domain": "troubleshooting",
        "method": "espresso",
        "difficulty": "beginner",
    }

    sample_transcript = """
    [00:00:05] so the most common problem with espresso is bitterness uh and
    the reason this happens is over-extraction. The water has dissolved too many
    compounds from the coffee grounds. Now the fix is almost always the grind size.
    You want to go one notch coarser and see if that helps. [00:01:20] Temperature
    also plays a role. If your machine is running at 96 or 97 degrees celsius that
    is too hot for most light roasts. Try dropping to 93 or 94 degrees.
    """

    chunks = embedder.chunk_transcript(sample_transcript, sample_doc)
    print(f"Generated {len(chunks)} chunks")
    for c in chunks:
        print(f"  ID: {c['chunk_id']} | words: {len(c['text'].split())} | text[:60]: {c['text'][:60]}...")

    if chunks:
        embedded = embedder.embed_batch(chunks)
        print(f"\nEmbedding dim: {len(embedded[0]['embedding'])}")
        print(f"IDs are stable: {chunks[0]['chunk_id'] == embedded[0]['chunk_id']}")

    query_vec = embedder.embed_query("why is my espresso bitter?")
    print(f"\nQuery embedding dim: {len(query_vec)}")
