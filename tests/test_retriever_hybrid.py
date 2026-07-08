"""
Hybrid search tests (ultraplan v3, Phase B) — demo mode, mock data only.
"""

import json
from pathlib import Path

import pytest

from engine.models import BrewingContext, DiagnosticResult, RootCause
from ingestion.embedder import Embedder
from pipeline.retriever import Retriever
from pipeline.vector_store import VectorStore


# ------------------------------------------------------------------
# RRF fusion — pure unit test, no models needed
# ------------------------------------------------------------------

def _chunk(cid: str, **extra) -> dict:
    return {"chunk_id": cid, "text": f"text {cid}", **extra}


def test_rrf_fusion_ranks_shared_docs_higher():
    # "shared" appears in BOTH lists (low ranks), so its RRF score
    # (two contributions) must beat every single-list chunk — even
    # the ones ranked first in their own list.
    vector = [_chunk("v_top", semantic_score=0.9), _chunk("shared", semantic_score=0.8)]
    bm25 = [_chunk("b_top", bm25_score=12.0), _chunk("shared", bm25_score=9.0)]

    fused = Retriever._rrf_fuse(vector, bm25)

    assert fused[0]["chunk_id"] == "shared"
    ids = [c["chunk_id"] for c in fused]
    assert set(ids) == {"shared", "v_top", "b_top"}
    # The fused entry keeps the richest fields from both lists.
    assert fused[0]["semantic_score"] == 0.8
    assert fused[0]["bm25_score"] == 9.0


# ------------------------------------------------------------------
# End-to-end hybrid retrieve on the mock corpus (in-memory store)
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_store():
    docs = json.loads(Path("data/mock_documents.json").read_text(encoding="utf-8"))
    embedder = Embedder()
    store = VectorStore(demo_mode=True)
    chunks = []
    for doc in docs[:10]:  # 10 docs keep the test fast
        chunks.extend(embedder.chunk_transcript(doc["content"], doc))
    store.add_chunks(embedder.embed_batch(chunks, verbose=False))
    return embedder, store


def test_retriever_hybrid_returns_results(demo_store):
    embedder, store = demo_store
    retriever = Retriever(
        embedder, store, use_cross_encoder=False, search_mode="hybrid"
    )

    ctx = BrewingContext(
        machine_type="super_automatic",
        raw_problem="DeLonghi Dinamica makes bitter espresso",
        symptoms_detected=["bitter"],
        goal="troubleshoot",
    )
    diag = DiagnosticResult(
        symptoms=["bitter"],
        root_causes=[RootCause("over-extraction", 0.70, "bitter = over-extracted")],
        intervention_plan=[],
        diagnostic_confidence=0.70,
        method_detected="super_automatic",
    )

    results = retriever.retrieve(ctx, diag)

    assert results, "hybrid retrieve returned no chunks"
    assert all("chunk_id" in c and c["text"] for c in results)
    # RRF ran: every candidate carries a fusion score.
    assert all("rrf_score" in c for c in results)
    # Ranks were attached for transparency.
    assert results[0]["retrieval_rank"] == 1


def test_retriever_vector_default_unchanged(demo_store):
    # Default search_mode must stay "vector" until Phase D picks a winner.
    embedder, store = demo_store
    retriever = Retriever(embedder, store, use_cross_encoder=False)
    assert retriever.search_mode == "vector"
