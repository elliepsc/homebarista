"""
RAG evaluation — compares the 3 coach styles. LIVE ONLY (real haiku generation).
================================================================================
Grading-grid requirement: "Multiple RAG approaches (e.g. prompts) are
evaluated, and the best one is used." The 3 existing coach styles
(detailed / concise / technical) are 3 different prompts: each is run over
the full eval dataset and scored two ways:

  PART A — structural (deterministic): CoachingEvaluator checks
           (pass_rate, mean overall_score, failed-check distribution).
  PART B — LLM judge (--llm-judge, optional): the configured LLM (see
           .env / engine/llm_client.py) rates specificity / science /
           actionability / completeness (1-5) on a sub-sample per style.

The winning style (structural first, judge as tie-break) becomes the app
default. BIAS NOTE: the judge is the same model as the generator —
scores may be inflated; treat them as relative, not absolute.

Usage:
  python -m evals.run_rag_eval --build-dataset  # offline, no API calls, run once first
  python -m evals.run_rag_eval                  # structural, 3 styles x N queries
  python -m evals.run_rag_eval --llm-judge      # + LLM judge (extra API calls)
Cost: ~150 LLM calls (~1200 tokens out each) — free on the Groq free tier,
a few cents on paid providers. Never in CI.

C10: data/eval_dataset.json was generated for the RETRIEVAL eval and
includes factual/how-to queries with no symptom (origin, technique...).
The linear pipeline's deterministic guard correctly rejects those with
"Could not diagnose" — that's a dataset/eval-scope mismatch, not a
generation-quality signal, and it was inflating the error count on the
2026-07-20 pilot run. --build-dataset filters down to symptom-bearing
queries only (data/eval_dataset_coaching.json) by replaying the exact same
guard offline via SymptomExtractor(demo_mode=True) + DiagnosticPlanner —
no LLM call, safe to run anytime, deterministic.
"""

import argparse
import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATASET = Path("data/eval_dataset.json")
COACHING_DATASET = Path("data/eval_dataset_coaching.json")
RESULTS_DIR = Path("evals/results")

STYLES = ["detailed", "concise", "technical"]
THRESHOLDS = {"pass_rate": 0.70, "mean_score": 0.65}
JUDGE_SAMPLE_SIZE = 15

BIAS_NOTE = (
    "LLM judge uses the same configured model as the generator. "
    "Scores may be inflated."
)

JUDGE_PROMPT = """Rate this barista coaching on 4 criteria (1-5 each):
1. Specificity: precise measurements and adjustments?
2. Science: explains WHY not just WHAT?
3. Actionability: immediately applicable?
4. Completeness: diagnosis + fix + validation test?
Respond in JSON: {{"specificity": n, "science": n, "actionability": n,
"completeness": n, "comment": "..."}}

COACHING TO RATE:
{coaching}"""


# ------------------------------------------------------------------
# Dataset filtering (C10) — offline, no LLM call
# ------------------------------------------------------------------

def build_coaching_dataset() -> None:
    """
    Filter the retrieval dataset down to symptom-bearing queries only, by
    replaying pipeline.py's Step 3 guard (diagnostic_confidence < 0.15 and
    goal == "troubleshoot" → "Could not diagnose") entirely offline. Writes
    COACHING_DATASET. See the module docstring (C10) for why this exists.
    """
    from engine.diagnostic_planner import DiagnosticPlanner
    from engine.symptom_extractor import SymptomExtractor

    if not DATASET.exists():
        raise SystemExit(
            f"{DATASET} not found. Generate it first: "
            "python -m evals.run_retrieval_eval --generate"
        )

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    extractor = SymptomExtractor(demo_mode=True)
    planner = DiagnosticPlanner()

    kept = []
    for item in dataset:
        context = extractor.extract(item["synthetic_query"])
        diagnostic = planner.diagnose(context)
        blocked = diagnostic.diagnostic_confidence < 0.15 and context.goal == "troubleshoot"
        if not blocked:
            kept.append(item)

    COACHING_DATASET.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(kept)}/{len(dataset)} queries kept (symptom-bearing) → {COACHING_DATASET}")
    print(f"{len(dataset) - len(kept)} filtered out — would hit 'Could not diagnose' (factual/no-symptom).")


# ------------------------------------------------------------------
# Part A — structural evaluation (deterministic checks)
# ------------------------------------------------------------------

def eval_style(style: str, dataset: list[dict]) -> tuple[dict, list[str]]:
    """Run the linear pipeline live for one style. Returns (stats, coachings)."""
    from pipeline.pipeline import run_pipeline

    stats = {"pass": 0, "scores": [], "failed_checks": Counter(), "errors": 0, "unscored": 0}
    coachings: list[str] = []

    for n, item in enumerate(dataset, start=1):
        # PITFALL (ultraplan 8.0.1): demo_mode=True would short-circuit both
        # retrieval and generation — this eval MUST run live, linear mode.
        result = asyncio.run(run_pipeline(
            item["synthetic_query"], coach_style=style,
            demo_mode=False, use_agent=False,
        ))
        if result["status"] != "coaching":
            stats["errors"] += 1
            print(f"  [{n}/{len(dataset)}] {result['status']} — skipped")
            continue

        # A "coaching" status without evaluation.post means the LIVE pipeline
        # routed this query to the general-question path (_answer_general_linear
        # → _info_result, evaluation={}), which produces no diagnostic coaching
        # to score. build_coaching_dataset can't perfectly pre-filter these: it
        # replays the guard with the demo-mode (rule) extractor, whereas the live
        # eval uses the LLM extractor, so the two can disagree on a handful of
        # borderline queries. Count them as "unscored" (not errors) and skip the
        # structural scoring — never crash the whole run on one of them.
        post = result.get("evaluation", {}).get("post")
        if not post:
            stats["unscored"] += 1
            print(f"  [{n}/{len(dataset)}] general answer (no diagnostic) — unscored")
            continue

        stats["pass"] += post["verdict"] == "pass"
        stats["scores"].append(post["overall_score"])
        for check, ok in post["checks"].items():
            if not ok:
                stats["failed_checks"][check] += 1
        coachings.append(result["coaching_text"])
        print(f"  [{n}/{len(dataset)}] verdict={post['verdict']} score={post['overall_score']}")
        time.sleep(0.3)

    n_scored = max(len(stats["scores"]), 1)
    return (
        {
            "pass_rate": round(stats["pass"] / n_scored, 3),
            "mean_score": round(sum(stats["scores"]) / n_scored, 3),
            "failed_checks": dict(stats["failed_checks"]),
            "errors": stats["errors"],
            "unscored": stats["unscored"],
            "n_scored": len(stats["scores"]),
        },
        coachings,
    )


# ------------------------------------------------------------------
# Part B — LLM judge (optional)
# ------------------------------------------------------------------

def judge_style(coachings: list[str]) -> dict:
    """Judge a sub-sample of coachings with the configured LLM. Mean per criterion."""
    from engine.llm_client import LLMClient, extract_json

    client = LLMClient()
    criteria = ["specificity", "science", "actionability", "completeness"]
    totals = {c: [] for c in criteria}
    parse_failures = 0

    for text in coachings[:JUDGE_SAMPLE_SIZE]:
        response = client.create(
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(coaching=text[:3000])}],
            max_tokens=300,
            reasoning_effort="low",
        )
        scores = extract_json(response.text)
        try:
            for c in criteria:
                totals[c].append(float(scores[c]))
        except (KeyError, TypeError, ValueError):
            parse_failures += 1
        time.sleep(0.3)

    return {
        **{c: round(sum(v) / len(v), 2) if v else None for c, v in totals.items()},
        "n_judged": len(totals["specificity"]),
        "parse_failures": parse_failures,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(llm_judge: bool) -> None:
    if not COACHING_DATASET.exists():
        raise SystemExit(
            f"{COACHING_DATASET} not found. Build it first (offline, no API calls): "
            "python -m evals.run_rag_eval --build-dataset"
        )
    dataset = json.loads(COACHING_DATASET.read_text(encoding="utf-8"))
    print(f"RAG eval on {len(dataset)} queries × {len(STYLES)} styles (live, linear mode)")

    per_style, coachings_by_style = {}, {}
    for style in STYLES:
        print(f"\n=== Style: {style} ===")
        per_style[style], coachings_by_style[style] = eval_style(style, dataset)
        print(f"  → {per_style[style]}")

    judge_results = None
    if llm_judge:
        judge_results = {}
        for style in STYLES:
            print(f"\nJudging {style} ({min(JUDGE_SAMPLE_SIZE, len(coachings_by_style[style]))} samples)...")
            judge_results[style] = judge_style(coachings_by_style[style])
            print(f"  → {judge_results[style]}")

    winner = max(per_style, key=lambda s: (per_style[s]["pass_rate"], per_style[s]["mean_score"]))
    verdict = (
        "PASS"
        if per_style[winner]["pass_rate"] >= THRESHOLDS["pass_rate"]
        and per_style[winner]["mean_score"] >= THRESHOLDS["mean_score"]
        else "FAIL"
    )

    output = {
        "dataset": str(COACHING_DATASET),
        "n_queries": len(dataset),
        "per_style": per_style,
        "llm_judge": judge_results,
        "winner": winner,
        "bias_note": BIAS_NOTE,
        "thresholds": THRESHOLDS,
        "verdict": verdict,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"rag_eval_{stamp}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown table ready to paste in the README. n_scored is shown on purpose:
    # a pass_rate is only meaningful next to the number of queries it was computed
    # over. "unscored" = queries the LIVE pipeline routed to the general-answer
    # path (factual/how-to, no taste symptom) — not applicable to style scoring.
    print("\n| Style | Pass rate | Mean score | Scored (n) | Unscored | Errors |")
    print("|---|---|---|---|---|---|")
    for style, stats in per_style.items():
        mark = " **(winner)**" if style == winner else ""
        print(
            f"| {style}{mark} | {stats['pass_rate']} | {stats['mean_score']} | "
            f"{stats.get('n_scored', '?')} | {stats.get('unscored', 0)} | {stats['errors']} |"
        )
    print(f"\nWinner: {winner} — verdict {verdict}")
    n_scored_winner = per_style[winner].get("n_scored", 0)
    if n_scored_winner < 20:
        print(f"NOTE: winner scored on only n={n_scored_winner} diagnostic queries — "
              f"report this n in the README; the pass_rate is high-variance at this size.")
    print("Reminder: make the winner the default coach_style in run_pipeline and the app.")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # LLM_* config from .env (see .env.example)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dataset", action="store_true",
                         help="filter to symptom-bearing queries only (offline, no API calls), then exit")
    parser.add_argument("--llm-judge", action="store_true", help="also run the LLM judge (extra API calls)")
    args = parser.parse_args()
    if args.build_dataset:
        build_coaching_dataset()
    else:
        main(llm_judge=args.llm_judge)
