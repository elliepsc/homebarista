"""
Retriever
=========
Retrieves and re-ranks passages from ChromaDB for a given diagnostic context.

Fixes vs. original plan:
1. Cross-encoder re-ranking replaces the arbitrary composite score formula.
   Model: cross-encoder/ms-marco-MiniLM-L-12-v2 (local, free, standard in literature).
   Fallback: semantic score only if cross-encoder not installed.
2. query_override parameter: the agentic loop can pass a custom query string,
   overriding the auto-generated query. Required by agent.py.
3. Machine-to-method mapping is explicit and documented.
4. Re-ranking justification: cross-encoders evaluate (query, passage) joint
   relevance — more accurate than bi-encoder cosine similarity alone.
"""

from typing import Optional

from engine.models import BrewingContext, DiagnosticResult
from ingestion.embedder import Embedder
from pipeline.vector_store import VectorStore


# ------------------------------------------------------------------
# Machine → allowed ChromaDB method tags
# ------------------------------------------------------------------

MACHINE_TO_METHODS: dict[str, list[str]] = {
    "super_automatic": ["super_automatic", "espresso", "general"],
    "semi_automatic":  ["espresso", "semi_automatic", "general"],
    "manual_lever":    ["espresso", "general"],
    "nespresso":       ["espresso", "general"],
    "moka":            ["moka", "general"],
    "v60":             ["v60", "filter", "general"],
    "aeropress":       ["aeropress", "general"],
    "french_press":    ["french_press", "filter", "general"],
    "unknown":         ["espresso", "moka", "v60", "general"],  # cast wide
}


# ------------------------------------------------------------------
# Retriever
# ------------------------------------------------------------------

class Retriever:
    """
    Two-stage retrieval:
    Stage 1 — ChromaDB bi-encoder search (top-k candidates, fast)
    Stage 2 — Cross-encoder re-ranking (precise, slower, local)

    The cross-encoder evaluates the full (query, passage) pair jointly,
    which is significantly more accurate than cosine similarity alone
    on technical domain text. (Thakur et al., BEIR benchmark, 2021)
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        use_cross_encoder: bool = True,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    ):
        self.embedder = embedder
        self.store = vector_store
        self.use_cross_encoder = use_cross_encoder
        self._cross_encoder = None

        if use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(cross_encoder_model)
                print(f"Cross-encoder loaded: {cross_encoder_model}")
            except ImportError:
                print(
                    "WARNING: sentence_transformers not installed or CrossEncoder unavailable. "
                    "Falling back to semantic-score-only ranking."
                )
                self.use_cross_encoder = False
            except Exception as e:
                print(f"WARNING: Cross-encoder load failed ({e}). Using semantic score only.")
                self.use_cross_encoder = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        brewing_context: BrewingContext,
        diagnostic: DiagnosticResult,
        n_candidates: int = 15,
        n_final: int = 10,
        query_override: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve and re-rank passages for a given diagnostic context.

        Args:
            brewing_context: parsed user context (machine, beans, etc.)
            diagnostic: output of DiagnosticPlanner
            n_candidates: how many candidates to pull from ChromaDB (Stage 1)
            n_final: how many to return after re-ranking (Stage 2)
            query_override: if provided, use this query instead of the auto-built one.
                           Used by the agentic loop for targeted second-pass retrieval.

        Returns:
            Ranked list of chunk dicts with relevance scores.
        """
        # Build or use the provided retrieval query
        query = query_override or self._build_query(brewing_context, diagnostic)

        # Build metadata filter
        where = self._build_filter(brewing_context.machine_type)

        # Stage 1 — bi-encoder search
        query_embedding = self.embedder.embed_query(query)
        candidates = self.store.query(
            embedding=query_embedding,
            n_results=n_candidates,
            where=where,
        )

        if not candidates:
            return []

        # Stage 2 — cross-encoder re-ranking (if available)
        if self.use_cross_encoder and self._cross_encoder and len(candidates) > 1:
            ranked = self._cross_encode_rank(query, candidates, n_final)
        else:
            # Fallback: sort by semantic score (bi-encoder cosine similarity)
            ranked = sorted(candidates, key=lambda x: x["semantic_score"], reverse=True)[:n_final]

        # Attach retrieval position for transparency
        for i, chunk in enumerate(ranked):
            chunk["retrieval_rank"] = i + 1

        return ranked

    # ------------------------------------------------------------------
    # Query builder
    # ------------------------------------------------------------------

    def _build_query(self, context: BrewingContext, diagnostic: DiagnosticResult) -> str:
        """
        Build a focused retrieval query from diagnostic results.
        More specific = better retrieval precision.
        """
        parts = []

        # Machine type
        parts.append(context.machine_type.replace("_", " "))

        # Primary symptom
        if diagnostic.symptoms:
            parts.append(diagnostic.symptoms[0].replace("_", " "))

        # Root cause hypothesis (most informative signal)
        if diagnostic.root_causes:
            parts.append(diagnostic.root_causes[0].hypothesis.replace("-", " ").replace("_", " "))

        # Brewing method
        if diagnostic.method_detected and diagnostic.method_detected != "unknown":
            parts.append(f"{diagnostic.method_detected.replace('_', ' ')} coffee extraction")

        # Bean context (helps for origin-specific advice)
        if context.bean_origin:
            parts.append(context.bean_origin)
        if context.roast_level:
            parts.append(f"{context.roast_level} roast")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Metadata filter
    # ------------------------------------------------------------------

    def _build_filter(self, machine_type: str) -> Optional[dict]:
        """
        Build ChromaDB metadata filter for method-relevant chunks only.
        Returns None if machine_type maps to all methods (avoid empty results).
        """
        allowed = MACHINE_TO_METHODS.get(machine_type, MACHINE_TO_METHODS["unknown"])

        # If allowed covers most methods, don't filter (avoid too-narrow results)
        all_methods = {"espresso", "super_automatic", "moka", "v60", "aeropress",
                       "french_press", "filter", "general"}
        if len(allowed) >= len(all_methods) - 1:
            return None

        return {"method": {"$in": allowed}}

    # ------------------------------------------------------------------
    # Cross-encoder re-ranking
    # ------------------------------------------------------------------

    def _cross_encode_rank(
        self, query: str, candidates: list[dict], n_final: int
    ) -> list[dict]:
        """
        Re-rank candidates using a cross-encoder.

        The cross-encoder scores (query, passage) pairs jointly — it reads
        both texts together and outputs a relevance score. This is much more
        accurate than cosine similarity between independently encoded vectors,
        especially for technical domain text where wording matters.

        We run it only on the top-k candidates (not the full corpus) to keep
        it fast. Cross-encoders are O(n) not O(1) like bi-encoder lookup.
        """
        pairs = [(query, chunk["text"]) for chunk in candidates]
        scores = self._cross_encoder.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(candidates, scores):
            chunk["cross_encoder_score"] = float(score)

        ranked = sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)
        return ranked[:n_final]


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    from engine.models import BrewingContext, DiagnosticResult, RootCause, Intervention

    # Minimal stubs for testing without full pipeline
    embedder = Embedder()
    store = VectorStore(demo_mode=True)

    retriever = Retriever(embedder, store, use_cross_encoder=False)  # demo: no cross-encoder needed

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

    query = retriever._build_query(ctx, diag)
    print(f"Generated query: {query}")

    where = retriever._build_filter("super_automatic")
    print(f"Metadata filter: {where}")
