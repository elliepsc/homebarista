"""
Retrieval evaluation — compares 4 retrieval configs on a synthetic dataset.
===========================================================================
Grading-grid requirement: "Multiple retrieval approaches are evaluated, and
the best one is used." This script compares:

  C1_vector_only   bi-encoder vector search only (baseline)
  C2_vector_ce     vector + cross-encoder re-ranking
  C3_hybrid_ce     hybrid BM25+vector (RRF fusion) + cross-encoder
  C4_raw_query_ce  same as C2 but with the RAW user query instead of the
                   diagnostic-aware rewritten query (_build_query)

C2 vs C4 is the query-rewriting evidence (best practice +1).
C2 vs C3 is the hybrid-search evidence (best practice +1).

Metrics: Hit Rate@5, Hit Rate@10, MRR — valid with a single relevant chunk
per query. Precision@k is deliberately NOT used: with exactly one relevant
chunk per query it is capped at 1/k by construction (0.20 at k=5), which
makes any meaningful threshold unreachable.

Usage:
  python -m evals.run_retrieval_eval --generate       # once, needs live corpus + an LLM key (.env)
  python -m evals.run_retrieval_eval --eval           # 4 configs, live corpus, no API key
  python -m evals.run_retrieval_eval --demo           # CI-safe: mock corpus, no API key
  python -m evals.run_retrieval_eval --generate-demo  # regenerate the committed demo dataset
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

DATASET = Path("data/eval_dataset.json")
DEMO_DATASET = Path("data/eval_dataset_demo.json")
RESULTS_DIR = Path("evals/results")

THRESHOLDS = {"hit_rate_5": 0.55, "mrr": 0.45}

GEN_PROMPT = """Given this barista knowledge passage:
"{passage}"
Write a realistic home user question that this passage directly answers.
The question should sound like a real user describing their coffee problem.
Respond with just the question, nothing else."""

# (retriever kwargs, use_rewritten_query)
# use_rewritten_query=True  → query_override=None → Retriever._build_query
#   builds a diagnostic-aware query (machine + symptom + root cause) = REWRITING.
# use_rewritten_query=False → the raw synthetic user query is passed verbatim.
CONFIGS = {
    "C1_vector_only":  (dict(use_cross_encoder=False, search_mode="vector"), True),
    "C2_vector_ce":    (dict(use_cross_encoder=True,  search_mode="vector"), True),
    "C3_hybrid_ce":    (dict(use_cross_encoder=True,  search_mode="hybrid"), True),
    "C4_raw_query_ce": (dict(use_cross_encoder=True,  search_mode="vector"), False),
}


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def build_demo_store():
    """In-memory store built from the 40 mock docs — used by --demo (CI)."""
    from ingestion.embedder import Embedder
    from pipeline.vector_store import VectorStore

    docs = json.loads(Path("data/mock_documents.json").read_text(encoding="utf-8"))
    store, embedder = VectorStore(demo_mode=True), Embedder()
    chunks = []
    for doc in docs:
        transcript = doc.get("transcript_text", doc.get("content", ""))
        chunks.extend(embedder.chunk_transcript(transcript, doc))
    store.add_chunks(embedder.embed_batch(chunks, verbose=False))
    return store, embedder


def build_live_store():
    """Persistent store from data/chroma_db (requires prior live ingestion)."""
    from ingestion.embedder import Embedder
    from pipeline.vector_store import VectorStore

    store = VectorStore()
    count = store.collection.count()
    if count == 0:
        raise SystemExit(
            "Live corpus is empty. Run the ingestion first "
            "(python -m ingestion.run_ingestion) or use --demo."
        )
    print(f"Live corpus: {count} chunks")
    return store, Embedder()


def make_context(query: str):
    """BrewingContext + DiagnosticResult from a raw query — rules only, no LLM."""
    from engine.diagnostic_planner import DiagnosticPlanner
    from engine.symptom_extractor import SymptomExtractor

    ctx = SymptomExtractor(demo_mode=True).extract(query)
    return ctx, DiagnosticPlanner().diagnose(ctx)


# ------------------------------------------------------------------
# Metrics (single relevant chunk per query)
# ------------------------------------------------------------------

def hit_rate_at(k: int, ranked_ids: list[str], relevant_ids: list[str]) -> float:
    return 1.0 if any(r in ranked_ids[:k] for r in relevant_ids) else 0.0


def mrr(ranked_ids: list[str], relevant_ids: list[str]) -> float:
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


# ------------------------------------------------------------------
# --generate : synthetic dataset from the LIVE corpus (needs API key)
# ------------------------------------------------------------------

def generate_dataset(n_queries: int = 50) -> None:
    from engine.llm_client import LLMClient

    store, _ = build_live_store()
    data = store.collection.get(include=["documents", "metadatas"])

    random.seed(42)
    indices = random.sample(range(len(data["ids"])), min(n_queries, len(data["ids"])))

    client = LLMClient()
    dataset = []
    for n, i in enumerate(indices, start=1):
        text = data["documents"][i]
        meta = data["metadatas"][i]
        # 300 not 100: on Groq, reasoning models (gpt-oss default, qwen3
        # fallback via LLM_BASE_URL) count hidden chain-of-thought tokens
        # against max_tokens — reasoning_effort="low" keeps that budget small.
        response = client.create(
            messages=[{"role": "user", "content": GEN_PROMPT.format(passage=text[:400])}],
            max_tokens=300,
            reasoning_effort="low",
        )
        query = response.text.strip()
        dataset.append({
            "query_id": f"q{n:03d}",
            "synthetic_query": query,
            "relevant_chunk_ids": [data["ids"][i]],
            "machine_type": meta.get("method", "unknown"),
            "domain": meta.get("domain", "general"),
        })
        print(f"  [{n}/{len(indices)}] {query[:70]}")
        time.sleep(0.5)

    DATASET.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(dataset)} queries to {DATASET}")


# ------------------------------------------------------------------
# --generate-demo : 10 handwritten queries against the mock corpus
# ------------------------------------------------------------------

# source_id → handwritten realistic user query. Chunk IDs are computed from
# the actual mock-corpus chunking so they always match the demo store.
DEMO_QUERIES = {
    "mock_001": "My espresso tastes really bitter every morning, what am I doing wrong?",
    "mock_002": "The shots from my machine come out sour and sharp, how do I fix that?",
    "mock_003": "I barely get any crema on my espresso and it disappears right away.",
    "mock_011": "How much does grind size actually change the taste of my coffee?",
    "mock_018": "I just bought a DeLonghi bean-to-cup machine, how should I set it up?",
    "mock_022": "How often do I need to descale my espresso machine and why?",
    "mock_027": "Coffee from my moka pot is bitter and harsh, what should I change?",
    "mock_032": "Can you give me a simple V60 pour over recipe for a beginner?",
    "mock_035": "My Aeropress coffee turns out bitter, which settings should I adjust?",
    "mock_038": "What is the real difference between light roast and dark roast coffee?",
}


def generate_demo_dataset() -> None:
    from ingestion.embedder import Embedder

    docs = {d["source_id"]: d for d in json.loads(
        Path("data/mock_documents.json").read_text(encoding="utf-8")
    )}
    embedder = Embedder()
    dataset = []
    for n, (source_id, query) in enumerate(sorted(DEMO_QUERIES.items()), start=1):
        doc = docs[source_id]
        chunks = embedder.chunk_transcript(doc["content"], doc)
        dataset.append({
            "query_id": f"demo_q{n:03d}",
            "synthetic_query": query,
            # All chunks of the source doc count as relevant: with the short
            # mock docs (1-2 chunks) every chunk answers the question.
            "relevant_chunk_ids": [c["chunk_id"] for c in chunks],
            "machine_type": doc.get("method", "unknown"),
            "domain": doc.get("domain", "general"),
        })

    DEMO_DATASET.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(dataset)} demo queries to {DEMO_DATASET}")


# ------------------------------------------------------------------
# --eval / --demo : run the 4 configs
# ------------------------------------------------------------------

def run_config(name, retriever_kwargs, use_rewritten_query, dataset, store, embedder):
    from pipeline.retriever import Retriever

    retriever = Retriever(embedder, store, **retriever_kwargs)
    scores = {"hit_rate_5": [], "hit_rate_10": [], "mrr": []}
    for item in dataset:
        ctx, diag = make_context(item["synthetic_query"])
        # Rewritten: query_override=None → _build_query (diagnostic-aware).
        # Raw (C4): the synthetic user query is passed through verbatim.
        override = None if use_rewritten_query else item["synthetic_query"]
        chunks = retriever.retrieve(ctx, diag, query_override=override)
        ranked = [c["chunk_id"] for c in chunks]
        rel = item["relevant_chunk_ids"]
        scores["hit_rate_5"].append(hit_rate_at(5, ranked, rel))
        scores["hit_rate_10"].append(hit_rate_at(10, ranked, rel))
        scores["mrr"].append(mrr(ranked, rel))
    return {m: round(sum(v) / len(v), 3) for m, v in scores.items()}


def run_eval(demo: bool) -> None:
    dataset_path = DEMO_DATASET if demo else DATASET
    if not dataset_path.exists():
        raise SystemExit(
            f"{dataset_path} not found. "
            + ("Run --generate-demo first." if demo else "Run --generate first.")
        )
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    store, embedder = build_demo_store() if demo else build_live_store()

    results = {}
    for name, (kwargs, use_rewritten) in CONFIGS.items():
        print(f"\nEvaluating {name} ...")
        results[name] = run_config(name, kwargs, use_rewritten, dataset, store, embedder)
        print(f"  {results[name]}")

    winner = max(results, key=lambda n: (results[n]["hit_rate_5"], results[n]["mrr"]))
    verdict = (
        "PASS"
        if results[winner]["hit_rate_5"] >= THRESHOLDS["hit_rate_5"]
        and results[winner]["mrr"] >= THRESHOLDS["mrr"]
        else "FAIL"
    )

    output = {
        "dataset": str(dataset_path),
        "n_queries": len(dataset),
        "mode": "demo" if demo else "live",
        "configs": results,
        "winner": winner,
        "thresholds": THRESHOLDS,
        "verdict": verdict,
        "notes": (
            "C2 (rewritten diagnostic-aware query) vs C4 (raw user query) is the "
            "query-rewriting comparison. C3 vs C2 is the hybrid-search comparison. "
            "Precision@k dropped: single relevant chunk per query caps it at 1/k."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"retrieval_{stamp}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown table ready to paste in the README
    print("\n| Config | Hit Rate@5 | Hit Rate@10 | MRR |")
    print("|---|---|---|---|")
    for name, m in results.items():
        mark = " **(winner)**" if name == winner else ""
        print(f"| {name}{mark} | {m['hit_rate_5']} | {m['hit_rate_10']} | {m['mrr']} |")
    print(f"\nWinner: {winner} — verdict {verdict}")
    print(f"Results written to {out_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # LLM_* config from .env (see .env.example)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="generate the live synthetic dataset (needs API key)")
    parser.add_argument("--generate-demo", action="store_true", help="regenerate the committed demo dataset")
    parser.add_argument("--eval", action="store_true", help="evaluate the 4 configs on the live corpus")
    parser.add_argument("--demo", action="store_true", help="evaluate the 4 configs on the mock corpus (CI-safe)")
    args = parser.parse_args()

    if args.generate:
        generate_dataset()
    elif args.generate_demo:
        generate_demo_dataset()
    elif args.eval:
        run_eval(demo=False)
    elif args.demo:
        run_eval(demo=True)
    else:
        parser.print_help()
