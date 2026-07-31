# 👋 Peer-review guide — HomeBarista Coach

Thanks for reviewing! This file maps the **LLM Zoomcamp evaluation criteria** to
exactly where each is satisfied and how to verify it in ~30 seconds. Full detail
lives in [`README.md`](README.md); this is the fast path for grading.

**What the app does (one line):** diagnoses your bad home coffee — a deterministic
symptom→cause engine, a two-stage retrieval pipeline over YouTube barista
transcripts (James Hoffmann, Lance Hedrick…), and an agentic tool-use loop that
writes machine-aware coaching. Knowledge base ≠ the course FAQ.

---

## 1. Fastest way to try it (no setup, no key — 30 seconds)

Open the deployed app: **https://homebarista-coach.streamlit.app/**
It boots in **Demo mode** (free, no login). Click an example problem (e.g.
*"My DeLonghi Dinamica makes bitter espresso"*) → you get a deterministic
diagnosis + mock coaching. Demo mode is intentionally ML-free so it never OOMs.

**To see the real RAG (retrieval + LLM), pick either:**
- In the sidebar, switch **Mode → Live**, then **paste your own free Groq key**
  (the field itself links to https://console.groq.com/keys — 30 s to create one;
  the key stays session-only, never stored), **or**
- use the **shared password** provided in my Zoomcamp submission notes (kept out
  of this public repo on purpose, to protect the shared free-tier quota).

Then ask a **taste problem** (e.g. *"my Gaggia makes sour espresso"*) and open the
**"Knowledge sources"** expander — you'll see real transcript passages cited.
A verbatim example (with the 5 retrieved sources) is in **README §5** if you'd
rather not run it live.

> **Demo vs Live — why two modes:** the graded reproducible path (retrieval eval,
> tests, CI) runs entirely in **demo mode with no key**. Live mode adds the real
> LLM + retrieval and is opt-in per visitor. Both are documented.

---

## 2. Run it locally (if you cloned)

```bash
# clone at the reviewed commit
git clone https://github.com/elliepsc/homebarista.git && cd homebarista
git reset --hard <commit-hash>

uv sync --extra dev                                      # pinned via uv.lock
uv run python -c "import nltk; nltk.download('punkt_tab')"

uv run pytest -v                                         # test suite (no key needed)
uv run python -m ingestion.run_ingestion --demo         # ingestion pipeline (demo)
uv run python -m evals.run_retrieval_eval --demo        # retrieval eval (no key)
uv run streamlit run app/streamlit_app.py               # the app, demo mode
```
> WSL note: if the repo is under `/mnt/c`, `export UV_LINK_MODE=copy` before `uv sync`.

Or everything in Docker: `docker compose up --build` → http://localhost:8501

---

## 3. Criteria → where → 30-second check

| Criterion (max) | Where in the repo | Verify in 30 s |
|---|---|---|
| **Problem description** (2) | `README.md` §1 | Reads clearly for someone who didn't take the course: expertise-dispersion problem, who it's for. |
| **Retrieval flow** (2) | `pipeline/` + `engine/llm_client.py`; README §2 | Knowledge base (**ChromaDB**, 725 chunks) **and** an LLM are both in the flow — not the LLM alone. |
| **Retrieval evaluation** (2) | `evals/run_retrieval_eval.py`; results in `evals/results/retrieval_*.json`; README §3 | **4 configs compared** (vector / +cross-encoder / hybrid / raw-query). Winner **C4** is wired as the default (README §3 explains the alignment). |
| **LLM evaluation** (2) | `evals/run_rag_eval.py`; `evals/results/rag_eval_20260726T175608Z.json`; README §3 | **3 prompts compared** (detailed / concise / technical). Winner **technical** is the default (`WINNER_STYLE`). |
| **Interface** (2) | `app/streamlit_app.py` | Streamlit **UI** (multi-turn chat + expanders). Live URL above. |
| **Ingestion pipeline** (2 or 1 — see note) | `ingestion/run_ingestion.py`; `ingestion/ingestion_report.json` | **Automated** single-command pipeline: fetch → quality-filter → chunk → embed → ChromaDB, with **checkpointing** (`progress.json`) and a **run report**. Smoke it: `... run_ingestion --demo`. |
| **Monitoring** (2) | `app/pages/1_Monitoring.py`; `logs/*.sample.jsonl` | **User feedback** (👍/👎 + comment → `feedback.jsonl`) **and** a dashboard with **7 charts** (≥5). Committed sample logs populate it on a fresh clone. |
| **Containerization** (2) | `Dockerfile` + `docker-compose.yml` | `docker compose up --build` runs **everything** (app + embedded ChromaDB). |
| **Reproducibility** (2) | `uv.lock`, `pyproject.toml`, `.github/workflows/ci.yml`, README §4 | All dependency **versions pinned** (`uv.lock`); demo path runs **without any key**; **CI** runs the tests + ingestion + retrieval-eval smoke on every push. |

### Best-practice points
| Bonus | Where | Check |
|---|---|---|
| **Hybrid search** (+1) | `pipeline/retriever.py` (BM25 + vector via RRF); evaluated as **C3** in README §3 | Config C3 in the retrieval table. |
| **Document re-ranking** (+1) | `pipeline/retriever.py` — cross-encoder `ms-marco-MiniLM-L-12-v2` | Applied in every config except C1; evidence C1 vs C2 in §3. |
| **Query rewriting** (+1) | `Retriever._build_query` + agent `query_override`; evaluated **C2 vs C4** | §3 shows rewriting *hurts* on this corpus — a real negative result, and the reason C4 (raw query) wins. |

### Cloud deployment bonus
| Bonus | Where |
|---|---|
| **Cloud deployment** (+2) | Live on **Streamlit Cloud**: https://homebarista-coach.streamlit.app/ — the sidebar shows a `build <sha>` marker so you can confirm the running commit. Snapshot of the KB is pulled at boot (see `docs-notes/.../runbook_c13_snapshot_deploy.md`). |

### Suggested "up to 3 extra" (reviewer's discretion — the rubric invites this)
If you'd like to award the optional extra points, here's the "something extra":
1. **True agentic tool-use loop** — `orchestration/agent.py`: the LLM decides which
   of 6 tools to call and when to stop (not a fixed pipeline). Guaranteed by tests
   in `tests/test_agent_orchestration.py` (tool ordering; a super-automatic never
   gets impossible advice like tamping).
2. **Deterministic, non-hallucinated diagnostic engine** — `engine/diagnostic_planner.py`
   (12 symptoms × machine-capability map). The *diagnosis* is rule-based and unit-tested,
   so it's testable and can't be hallucinated; the LLM only writes the prose around it.
3. **`ScopeGuard`** (`engine/scope_guard.py`) — deterministic off-topic filter that
   refuses non-coffee requests **before any model loads = 0 tokens**, bilingual (EN/FR),
   with tests.

---

## 4. Honest notes (so nothing surprises you)

- **RAG/LLM eval `n`:** the coaching eval scores **13** genuine taste-problem queries;
  the other 18 in the dataset are factual/how-to questions the pipeline correctly
  routes to a general-answer path (no diagnosis to score), and 2 are out-of-scope.
  Small `n`, so the absolute pass rate is high-variance — the intended signal is the
  **clear separation of the 3 prompts** (technical 1.0 > concise 0.92 > detailed 0.23).
  README §3 states this and the threats-to-validity in full.
- **Ingestion of the *live* corpus** must run from a residential IP (YouTube blocks
  transcript fetching from cloud IPs). The **reproducible** eval/CI path uses the
  committed **demo corpus** (40 mock docs) — no network, no key. The live 725-chunk
  corpus is delivered to the deployed app via a private snapshot (documented).
- **Ingestion score (1 vs 2):** it's a fully **automated** one-command Python pipeline
  (checkpointing, quality filter, report), not an orchestrator like Airflow/Prefect.
  Score per your reading of "special tool" — I've documented the automation so you can
  judge.
- **LLM-judge:** intentionally omitted from the final eval (a same-model judge is
  biased); the deterministic structural checks are the decision signal. README §3
  explains, and how to run it if you want.

Questions or anything unclear? Everything is cross-referenced in `README.md`.
Thanks again for taking the time — enjoy the coffee science. ☕
