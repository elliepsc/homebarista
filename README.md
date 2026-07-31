# ☕ HomeBarista Coach

[![ci](https://github.com/elliepsc/homebarista/actions/workflows/ci.yml/badge.svg)](https://github.com/elliepsc/homebarista/actions/workflows/ci.yml)

**AI barista coaching that diagnoses your bad coffee — built on RAG over YouTube barista expertise, with a deterministic diagnostic engine and an agentic tool-use loop.**

> 📋 **Reviewing this project?** [`PEER_REVIEW.md`](PEER_REVIEW.md) maps every Zoomcamp criterion to where it lives and how to verify it in ~30 seconds.

## 1. Problem description

Diagnosing bad coffee at home is an expertise-dispersion problem. The knowledge needed to fix a bitter espresso or a weak V60 exists — but it is scattered across hundreds of hours of YouTube videos from expert baristas (James Hoffmann, Lance Hedrick, and others). A home user with a DeLonghi super-automatic who gets sour shots doesn't know *which* video answers *their* problem, and generic LLM answers ignore the constraints of their machine (you can't adjust tamping pressure on a super-automatic).

HomeBarista Coach turns that dispersed video expertise into an interactive coach: it extracts symptoms from a free-form problem description, runs a **deterministic diagnostic engine** (12 symptoms × machine capability map — diagnosis is rule-based on purpose, so it is testable and never hallucinated), retrieves the most relevant transcript passages with a **two-stage retrieval pipeline** (bi-encoder + cross-encoder re-ranking, optional hybrid BM25+vector), and generates machine-aware coaching through an **agentic tool-use loop** (6 tools; LLM provider configurable in `.env` — Groq free tier by default, Anthropic/OpenAI switchable). A deterministic **ScopeGuard** refuses off-topic requests before any model loads — zero tokens spent on them.

**Who it's for:** home coffee enthusiasts with any setup — super-automatic, semi-automatic, moka pot, V60, Aeropress, French press, Nespresso.

## 2. Architecture

```
OFFLINE — ingestion pipeline (ingestion/)
  YouTube channels → YouTubeClient → TranscriptFetcher → ContentClassifier
  (quality filter, confidence ≥ 0.4) → TranscriptPreprocessor (sentence-aware
  chunking, nltk) → Embedder (MiniLM-L6-v2, SHA-256 chunk IDs) → ChromaDB

ONLINE — coaching loop (pipeline/ + engine/ + orchestration/)
  user message
    → ScopeGuard (deterministic, 0 tokens; off-topic → fixed refusal)
    → SymptomExtractor (rules + optional LLM)
    → DiagnosticPlanner (deterministic capability map)
    → Retriever (vector or hybrid BM25+RRF → cross-encoder re-rank)
    → Agent tool-use loop (LLM from .env — default Groq openai/gpt-oss-120b,
      6 tools, max 8 iterations; see engine/llm_client.py)
      [demo mode / CI: linear deterministic pipeline instead]
    → CoachingEvaluator (deterministic quality checks)
    → sessions.jsonl + feedback.jsonl → Monitoring dashboard
```

Packages: `engine/` (models, extractor, planner, evaluator, scope guard) · `ingestion/` (YouTube → chunks pipeline) · `pipeline/` (vector store, retriever, orchestrator) · `orchestration/` (agent) · `evals/` (retrieval + RAG evaluations) · `app/` (Streamlit UI + monitoring) · `tests/`.

## 3. Evaluation results

### Retrieval — 4 configurations compared (`evals/run_retrieval_eval.py`)

Metrics: Hit Rate@5, Hit Rate@10, MRR on a synthetic dataset (one relevant chunk per query). **Precision@k was deliberately dropped**: with a single relevant chunk per query it is capped at 1/k by construction, making any threshold above 0.20 unreachable at k=5.

Demo corpus (40 mock docs, 10 queries — reproducible in CI with `--demo`):

| Config | Hit Rate@5 | Hit Rate@10 | MRR |
|---|---|---|---|
| C1_vector_only | 0.7 | 0.7 | 0.517 |
| C2_vector_ce (rewritten query + cross-encoder) | 0.8 | 0.8 | 0.733 |
| C3_hybrid_ce (BM25+vector RRF + cross-encoder) | 0.8 | 0.8 | 0.733 |
| C4_raw_query_ce **(winner on demo corpus)** | 0.9 | 0.9 | 0.783 |

Live corpus (50 synthetic queries, generated from the real ingested corpus — 55 chunks / 24 sources / 2 channels at eval time, 2026-07-17; the corpus was subsequently completed to its **final size of 725 chunks / 95 sources / 5 channels** — see the control re-run below):

| Config | Hit Rate@5 | Hit Rate@10 | MRR |
|---|---|---|---|
| C1_vector_only | 0.22 | 0.38 | 0.154 |
| C2_vector_ce (rewritten query + cross-encoder) | 0.20 | 0.34 | 0.137 |
| C3_hybrid_ce (BM25+vector RRF + cross-encoder) | 0.26 | 0.40 | 0.170 |
| C4_raw_query_ce **(winner)** | **0.86** | **0.86** | **0.835** |

**Winner: C4_raw_query_ce**, by a wide margin. **Retriever default aligned**:
the linear pipeline (`pipeline/pipeline.py`, used in demo mode/CI and as the
non-agentic fallback) now passes the raw user message as `query_override`
instead of letting the Retriever build a diagnostic-aware query. The
Retriever's own defaults (`search_mode="vector"`, `use_cross_encoder=True`)
already matched the winning config — no change needed there.

- **C2 vs C4 is the query-rewriting evidence**: the Retriever rewrites the raw user message into a diagnostic-aware query (machine + symptom + root-cause hypothesis) via `_build_query`; the agent can also rewrite queries via `query_override`. On the live corpus, rewriting *hurts* retrieval — it abstracts away the specific vocabulary present in the transcript chunks.
- **C2 vs C3 is the hybrid-search evidence**: BM25 keyword search fused with vector search via Reciprocal Rank Fusion (k=60) before cross-encoder re-ranking. Hybrid helps a little over vector-only, but not nearly as much as dropping the query rewrite.

**Control re-run on the final corpus** (`evals/results/retrieval_20260723T074731Z.json`, same 50-query dataset, now 725 chunks / 95 sources / 5 channels instead of 55/24/2): absolute scores drop for every config as expected with ~13x more distractor chunks (C4 hit_rate@5 0.86 → 0.72), but the ranking is unchanged and C4_raw_query_ce still wins by a wide margin (next best: C3 at 0.12) — the 07-17 table above stands.

### RAG / LLM — 3 prompts compared (`evals/run_rag_eval.py`)

The 3 coach styles (detailed / concise / technical) are 3 different prompts, each evaluated on the full dataset with deterministic structural checks (CoachingEvaluator pass rate + mean score) and an optional LLM judge (specificity / science / actionability / completeness, 1–5).

Live run on the coaching dataset (`data/eval_dataset_coaching.json`),
`evals/results/rag_eval_20260726T175608Z.json`. The dataset holds 33 queries;
**n=13** produce diagnostic coaching live (a taste symptom is extracted), **18**
are factual/how-to questions the live pipeline correctly routes to its
general-answer path (no diagnosis to score), and **2** are out-of-scope. The
pass rate is computed over the 13 genuine diagnostic cases:

| Style | Pass rate | Mean score | Scored (n) | Unscored | Errors |
|---|---|---|---|---|---|
| detailed | 0.231 | 0.869 | 13 | 18 | 2 |
| concise | 0.923 | 0.974 | 13 | 18 | 2 |
| technical **(winner)** | **1.0** | **0.987** | 13 | 18 | 2 |

**Winner: technical** (verdict PASS). Default style in `app/streamlit_app.py`
(`WINNER_STYLE`) and `pipeline/pipeline.py`/`orchestration/agent.py`
(`coach_style`/`style` defaults) aligned accordingly.

**What the pass rate measures — and its limits.** The verdict holds three
critical checks strict (specific measurements, a validation test, appropriate
length) and tolerates at most one non-critical miss; `mentions_root_cause` is a
fuzzy keyword match, not a literal-string match (the earlier literal check was
the main driver of false failures — mean score was already ~0.89 while pass
rate had collapsed to ~0.45). The pass rate at **n=13 is high-variance**; the
decision-grade signal is the **clear, reproducible separation of the three
prompts** (technical 1.0 > concise 0.92 > detailed 0.23) — which is exactly what
the grid asks for ("several approaches compared, the best one retained").

**Threats to validity (retrieval + RAG).** Queries are synthetic and generated
from the corpus itself (favourable-bias risk); the retrieval dataset has a
single relevant chunk per query, so Hit Rate/MRR measure "is the source chunk
retrieved", not "is the answer good"; and the coaching eval scores only the 13
diagnostic queries. Treat every number here as a **relative model-selection
signal, not a decision-grade product metric.**

*LLM judge omitted from the final run: the structural checks plus the clear
style separation are the decision signal; a same-model judge (generator =
judge) would add only a biased relative score. If run, it should use a model
different from the generator to remove that bias.*

## 4. How to run

### Local (uv)

```bash
uv sync --extra dev
uv run python -c "import nltk; nltk.download('punkt_tab')"
cp .env.example .env          # optional — demo mode needs no key

uv run pytest -v                                   # test suite (demo-safe)
uv run python -m ingestion.run_ingestion --demo    # smoke the ingestion pipeline
uv run streamlit run app/streamlit_app.py          # the app (demo mode, no key needed)
```

> WSL note: if the repo lives under `/mnt/c`, use `export UV_LINK_MODE=copy` before `uv sync` to avoid hardlink issues.

**LLM configuration lives only in `.env`** (see `.env.example`): `LLM_PROVIDER` (groq — free tier, default — | anthropic | openai), `LLM_MODEL` (optional override) and the matching API key. `engine/llm_client.py` is the single gateway — no model name is hardcoded anywhere else.

### Docker

```bash
docker compose up --build     # → http://localhost:8501 (demo mode, no key needed)
```

Everything runs inside compose: ChromaDB is embedded in the app container in persistent mode, with `./data` and `./logs` mounted as volumes.

### Cloud

**Live app: [homebarista-coach.streamlit.app](https://homebarista-coach.streamlit.app/)** (main file `app/streamlit_app.py`).

- **Demo mode is the default** — open the link and try it in seconds, no key, no login. It runs the deterministic diagnostic engine on taste problems and gives mock coaching text.
- **Live mode** (real RAG retrieval + LLM coaching) is opt-in per visitor: unlock with the shared password *or* paste your own Groq/Anthropic/OpenAI key (session-only, never stored). The key field links straight to the provider's key page (Groq: [console.groq.com/keys](https://console.groq.com/keys)) for visitors who don't have one yet — the link follows whatever `LLM_PROVIDER` is configured. To keep the Groq free tier (8 000 tokens/min) from throttling, the shared key runs the **linear pipeline** (1 LLM call/request); a bring-your-own key gets the full **agentic** loop.
- The sidebar shows a `build <sha>` marker — the exact commit running, so you can confirm which version you're testing.

To deploy your own instance:

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), "New app" → pick the repo/branch → main file path `app/streamlit_app.py`.
3. In the app's **Settings → Secrets**, paste the content of [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) with real values (`LLM_API_KEY`, `LIVE_PASSWORD`, and the `SNAPSHOT_GITHUB_*` keys that let the app pull the knowledge-base snapshot at boot). Keep `DEMO_MODE = "true"` so the app is free to browse by default.
4. Deploy. Dependencies come from `uv.lock` (uv-sync); Python version from `pyproject.toml` (`>=3.11`).

### Ingestion (live corpus)

```bash
uv run python -m ingestion.run_ingestion --channel <CHANNEL_ID>  # one channel first
uv run python -m ingestion.run_ingestion                         # full run (YOUTUBE_API_KEY)
uv run python -m pipeline.vector_store --stats                   # verify corpus
uv run python -m pipeline.vector_store --export                  # snapshot for deployment
```

Live ingestion must run from a residential machine (YouTube blocks transcript fetching from cloud IPs); `ingestion/progress.json` checkpointing allows resuming after quota/bans. The demo corpus (40 mock docs) remains the reproducible evaluation path for reviewers.

## 5. Interface, monitoring & cost guardrails

- **Streamlit chat app** (`app/streamlit_app.py`): multi-turn conversation ("*Enjoy better coffee at home* — tell me what your coffee tastes like, what machine you use, or what you'd like to improve"), 5 one-click example problems, diagnostic/sources/quality-check expanders. **Demo mode is the default and needs zero API keys** — a reviewer can test in 10 seconds.
- **Feedback loop**: 👍/👎 + optional comment after each coaching → `logs/feedback.jsonl`, joined to `logs/sessions.jsonl` by session id.
- **Monitoring dashboard** (`app/pages/1_Monitoring.py`): 7 charts (sessions/day, quality verdicts, machines, symptoms, status incl. out-of-scope rate, feedback, agent iterations) + 4 headline metrics. Committed sample logs keep it populated on a fresh clone.
- **Cost guardrails**: deterministic ScopeGuard refuses off-topic requests at zero token cost; input capped at 1500 chars; live mode locked behind a password or bring-your-own API key (session-only, never stored); agent loop capped at 8 iterations. Budget: a bring-your-own key is unlimited (visitor's own cost) and runs the full agentic loop; the shared password runs the **linear pipeline (1 LLM call/request)** to stay under the Groq free-tier 8 000 tokens/min limit, and is capped at 10 runs/session **and** a 25-run/day global budget shared across all visitors, so a session reset can't be used to bypass the cap.

### Live coaching example (real retrieval, verbatim)

A taste problem triggers the full RAG path — deterministic diagnosis, then LLM
coaching grounded in retrieved transcript passages, sources cited. Verbatim from
the deployed app:

> **User:** *my Gaggia makes sour espresso*
>
> **Diagnosis (60% confidence):** most likely cause — **under-extraction**.
> Sour = extraction stopped before the acids were balanced by sugars and bitters.
>
> **Coaching (technical style):**
> **Root cause** — under-extraction: too few soluble compounds (mainly acids and
> sugars) are pulled before the flow stops, so the shot is dominated by bright,
> acidic notes.
> 1. **Grind −1 notch (≈10 µm finer)** — raises the surface-area-to-mass ratio, so
>    the water dissolves more solids in the same brew time.
> 2. **Temperature +2 °C (target 94 °C)** — extraction kinetics are
>    temperature-dependent (Arrhenius); more heat accelerates dissolution of the
>    larger, less-soluble compounds that add body and caramel sweetness.
> 3. **Extend brew time +3–5 s (target 27–30 s for 18 g)** — each second past ~20 s
>    extracts a higher proportion of sugars and lipids that counterbalance the acids.
>
> **Validation:** *you should notice a smoother, less acidic cup with a subtle
> caramel sweetness and a crema that holds its colour for at least 30 seconds.*
>
> **Knowledge sources retrieved** (bi-encoder → cross-encoder re-rank):
> - Lance Hedrick — *HOW TO FIX SOUR ESPRESSO: 4 Easy Tips* (2 passages)
> - Lance Hedrick — *Why Light Roast Espresso Makes No Sense (And How To Fix It)* (2 passages)
> - Lance Hedrick — *HOW TO DIAL IN ESPRESSO: Quick and Easy Guide*

By contrast, a question with **no taste symptom** (e.g. *"how do I choose coffee —
what brand, what country?"*) is routed to the general-answer path with **no
retrieval** and no sources, by design — the RAG diagnostic flow is reserved for
taste troubleshooting. *(The exact wording and figures of the LLM coaching vary
run-to-run — sampling is non-deterministic — but the deterministic diagnosis and
adjustment directions are stable.)*

## 6. Honest limitations

- **LLM judge bias**: judge and generator are the same configured model; judge scores are relative, not absolute.
- **Diagnostic weights are heuristics**, not calibrated probabilities — they encode expert rules, and are tested as such.
- **Precision@k dropped** from the retrieval eval (single-relevant-chunk dataset caps it at 1/k); Hit Rate and MRR are the valid metrics in this setting.
- **Demo-corpus eval numbers** (10 queries, 40 docs) are a smoke-level signal; the live 50-query eval is the stronger one (see the threats-to-validity note in §3 for how far to trust either).
- **Transcripts**: fetched for research/educational use; sources are always cited (channel + video URL) in retrieved passages.

## 7. Zoomcamp criteria mapping

| Criterion | Where |
|---|---|
| Problem description | This README §1 |
| Retrieval flow (KB + LLM) | ChromaDB + provider-agnostic LLM (Groq default), `pipeline/` |
| Retrieval evaluation (multiple approaches) | §3, `evals/run_retrieval_eval.py` (4 configs) |
| LLM evaluation (multiple prompts) | §3, `evals/run_rag_eval.py` (3 styles) |
| Interface | Streamlit app, `app/` |
| Ingestion pipeline (automated) | `ingestion/run_ingestion.py` (checkpointing, quality filter, report) |
| Monitoring (feedback + dashboard) | 👍/👎 → `feedback.jsonl`, dashboard 7 charts |
| Containerization | `Dockerfile` + `docker-compose.yml` |
| Reproducibility | `uv.lock`, demo mode without keys, CI |
| Hybrid search (bonus) | BM25+vector RRF, evaluated in §3 |
| Re-ranking (bonus) | cross-encoder ms-marco-MiniLM-L-12-v2 |
| Query rewriting (bonus) | `_build_query` + `query_override`, evaluated C2 vs C4 |
| Cloud deployment (bonus) | [homebarista-coach.streamlit.app](https://homebarista-coach.streamlit.app/) (Streamlit Cloud) |
| Extra: agentic loop, deterministic engine, CI, ScopeGuard | `orchestration/agent.py`, `engine/`, `.github/workflows/ci.yml` |
