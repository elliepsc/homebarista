"""
Pipeline — Orchestrator
========================
Single entry point for HomeBarista Coach.

Two modes:
1. AGENT mode (default): uses HomeBaristaAgent with tool-use loop.
   The LLM decides the flow — true agentic architecture.
   Requires an LLM API key (GROQ_API_KEY by default — see .env.example
   and engine/llm_client.py for the provider/model configuration).

2. LINEAR mode (fallback/CI): deterministic pipeline without LLM agency.
   Used in demo mode and tests to avoid API calls.
   Same components, fixed order.

Usage from Streamlit or CLI:
    from pipeline.pipeline import run_pipeline
    result = await run_pipeline("my DeLonghi makes bitter espresso")
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from engine.models import BrewingContext, CoachingSession
from engine.scope_guard import ScopeGuard
from engine.symptom_extractor import SymptomExtractor
from engine.diagnostic_planner import DiagnosticPlanner
from engine.coaching_evaluator import CoachingEvaluator

# Heavy imports (sentence-transformers, chromadb, LLM SDKs) are deferred to
# _get_components: an out-of-scope request refused by the ScopeGuard must not
# pay for loading embedding models or SDK clients.
if TYPE_CHECKING:
    from ingestion.embedder import Embedder
    from orchestration.agent import HomeBaristaAgent
    from pipeline.retriever import Retriever
    from pipeline.vector_store import VectorStore


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

LOG_DIR  = Path("logs")
LOG_FILE = LOG_DIR / "sessions.jsonl"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("pipeline")


# ------------------------------------------------------------------
# Component initialisation (lazy singletons)
# ------------------------------------------------------------------

_embedder:   Optional["Embedder"]   = None
_store:      Optional["VectorStore"] = None
_retriever:  Optional["Retriever"]  = None
_extractor:  Optional[SymptomExtractor]  = None
_planner:    Optional[DiagnosticPlanner] = None
_evaluator:  Optional[CoachingEvaluator] = None
_agent:      Optional["HomeBaristaAgent"]  = None
_llm_api_key: Optional[str] = None   # key the LLM-bound singletons were built with
_components_demo: Optional[bool] = None  # demo_mode the singletons were built with


def _get_components(
    demo_mode: bool,
    use_cross_encoder: bool = True,
    api_key: Optional[str] = None,
):
    """Lazy-initialise all components. Reuse across requests.

    Demo mode deliberately builds NOTHING from the ML stack (torch,
    transformers, sentence-transformers, chromadb): a demo request never
    retrieves (it uses _mock_chunks) and never calls the agent, so importing
    those would only add cold-start latency and memory pressure — enough to
    OOM the app on Streamlit Cloud's small free tier. Keeping demo purely
    deterministic (ScopeGuard + regex extractor + planner + mock coaching)
    is what makes demo mode reliable even when live mode is under load.
    """
    global _embedder, _store, _retriever, _extractor, _planner, _evaluator
    global _agent, _llm_api_key, _components_demo

    # Mode-dependent singletons must not survive a demo↔live switch:
    # a store built in demo mode is in-memory and EMPTY — reusing it in
    # live mode would silently retrieve nothing.
    if demo_mode != _components_demo:
        _store = None
        _retriever = None
        _extractor = None
        _agent = None
        _components_demo = demo_mode

    # BYO key (Streamlit live mode): the key is passed per request, NEVER
    # written to os.environ — an env-level key would leak to every other
    # user of the same process. Rebuild LLM-bound components on key change.
    if api_key != _llm_api_key:
        _extractor = None
        _agent = None
        _llm_api_key = api_key

    # Deterministic components — pure Python, no ML imports, always needed.
    if _planner is None:
        _planner = DiagnosticPlanner()
    if _evaluator is None:
        _evaluator = CoachingEvaluator()

    # --------------------------------------------------------------
    # DEMO MODE — no ML stack. retriever/agent stay None (unused).
    # --------------------------------------------------------------
    if demo_mode:
        if _extractor is None:
            _extractor = SymptomExtractor(llm_client=None, demo_mode=True)
        return _extractor, _planner, _retriever, _evaluator, _agent

    # --------------------------------------------------------------
    # LIVE MODE — heavy imports happen here, lazily, only when needed.
    # --------------------------------------------------------------
    from ingestion.embedder import Embedder
    from orchestration.agent import HomeBaristaAgent
    from pipeline.retriever import Retriever
    from pipeline.vector_store import VectorStore
    from engine.llm_client import LLMClient

    if _embedder is None:
        _embedder = Embedder()

    if _store is None:
        _store = VectorStore(demo_mode=False)

    if _retriever is None:
        _retriever = Retriever(
            _embedder,
            _store,
            use_cross_encoder=use_cross_encoder,
        )

    if _extractor is None:
        _extractor = SymptomExtractor(
            llm_client=LLMClient(api_key=api_key), demo_mode=False
        )

    if _agent is None:
        _agent = HomeBaristaAgent(
            symptom_extractor=_extractor,
            diagnostic_planner=_planner,
            retriever=_retriever,
            coaching_evaluator=_evaluator,
            llm_client=LLMClient(api_key=api_key),
            demo_mode=False,
        )

    return _extractor, _planner, _retriever, _evaluator, _agent


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

async def run_pipeline(
    raw_problem: str,
    coach_style: Literal["detailed", "concise", "technical"] = "technical",
    demo_mode: bool = True,
    use_agent: bool = True,
    conversation_history: Optional[list[dict]] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    Run the HomeBarista coaching pipeline.

    Args:
        raw_problem:           latest user message (free-form).
        coach_style:           "detailed" | "concise" | "technical"
        demo_mode:             skip API calls, use mock data.
        use_agent:             True → agentic tool-use loop (default).
                               False → linear deterministic pipeline (CI/demo).
        conversation_history:  prior turns as [{role, content}, ...].
                               Passed to the agent so it remembers machine,
                               beans, and adjustments already tried.
        api_key:               optional per-request LLM key (Streamlit BYO
                               key). Kept out of os.environ on purpose.

    Returns:
        {
            "status":                 "coaching" | "clarification_needed" | "error" | "out_of_scope"
            "coaching_text":          str
            "clarification_question": str
            "context":                dict
            "diagnostic":             dict
            "retrieved_chunks":       list
            "evaluation":             dict
            "session_id":             str
            "tool_call_log":          list
            "iterations":             int
        }
    """
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"Session {session_id} | mode={'agent' if use_agent and not demo_mode else 'linear'}")

    raw_problem = raw_problem.strip()
    if not raw_problem:
        return _error_result(session_id, "Please describe your coffee problem.")

    # ScopeGuard FIRST — before _get_components, so an off-topic request
    # never loads embedding models and never spends a single LLM token.
    verdict = ScopeGuard().check(raw_problem, conversation_history)
    if not verdict["in_scope"]:
        result = _out_of_scope_result(session_id, verdict["message"])
        _log_session(session_id, raw_problem, result)
        return result

    extractor, planner, retriever, evaluator, agent = _get_components(
        demo_mode, api_key=api_key
    )

    try:
        if use_agent and not demo_mode and agent is not None:
            # --------------------------------------------------------
            # AGENT MODE — LLM tool-use loop with conversation memory
            # --------------------------------------------------------
            result = await asyncio.to_thread(
                agent.run, raw_problem, coach_style, conversation_history or []
            )
            result["session_id"] = session_id

        else:
            # --------------------------------------------------------
            # LINEAR MODE — deterministic fixed-order pipeline
            # --------------------------------------------------------
            result = await _run_linear(
                raw_problem, coach_style, session_id,
                extractor, planner, retriever, evaluator, demo_mode,
                api_key=api_key,
            )

    except ValueError as e:
        # ValueError carries user-facing guidance (too vague, blocked, ...)
        return _error_result(session_id, str(e))
    except Exception as e:
        # A 429 that outlived the client's retries means the free-tier
        # token/minute budget is momentarily exhausted (e.g. several visitors
        # sharing the key). That's not a server fault — tell the user plainly
        # how to proceed instead of a scary generic error.
        from engine.llm_client import _is_rate_limit_error
        if _is_rate_limit_error(e):
            logger.warning(f"Session {session_id} hit provider rate limit")
            return _error_result(
                session_id,
                "The free coaching budget is busy right now (shared rate "
                "limit). Please wait about a minute and try again — or unlock "
                "Live mode with your own API key for unlimited use.",
            )
        # Never surface internals (paths, config, provider errors) to the UI.
        logger.exception(f"Session {session_id} failed")
        return _error_result(
            session_id,
            "Something went wrong on our side. Please try again — "
            f"reference: session {session_id}.",
        )

    # Log session
    _log_session(session_id, raw_problem, result)

    return result


async def _run_linear(
    raw_problem: str,
    coach_style: str,
    session_id: str,
    extractor: SymptomExtractor,
    planner: DiagnosticPlanner,
    retriever: "Retriever",
    evaluator: CoachingEvaluator,
    demo_mode: bool,
    api_key: Optional[str] = None,
) -> dict:
    """
    Linear pipeline: fixed order, no LLM agency.
    Used for demo mode, CI, and as a fallback.
    """
    from dataclasses import asdict

    # Step 1 — Extract symptoms
    context: BrewingContext = extractor.extract(raw_problem)

    # Step 2 — Diagnose
    diagnostic = planner.diagnose(context)

    # Step 3 — Route by what kind of coffee question this is.
    # The linear diagnostic pipeline only handles TASTE problems (a detected
    # symptom + enough confidence). Anything else is a legitimate coffee
    # question the diagnostic engine can't answer — never a hard error.
    diagnosable = bool(diagnostic.symptoms) and diagnostic.diagnostic_confidence >= 0.15
    if not diagnosable:
        if context.goal == "troubleshoot":
            # A taste complaint too vague to diagnose — ask for specifics.
            raise ValueError(
                "Could not diagnose the problem. Please describe: your machine, "
                "what tastes wrong (bitter/sour/weak), and any parameters you know."
            )
        # A general/informational coffee question (choosing beans, recipes,
        # theory, buying advice). Demo has no LLM → point to Live; Live
        # answers it directly with one LLM call.
        if demo_mode:
            return _info_result(session_id, _demo_general_text(), context)
        answer = _answer_general_linear(raw_problem, context, api_key=api_key)
        return _info_result(session_id, answer, context)

    # Step 4 — Retrieve
    # query_override=raw_problem: the retrieval eval (README §3) found the raw
    # user query beats the diagnostic-aware rewritten query (_build_query) by
    # a wide margin (hit_rate@5 0.86 vs 0.2-0.26 on the live corpus).
    chunks = retriever.retrieve(context, diagnostic, query_override=raw_problem) if not demo_mode else _mock_chunks()
    retrieval_metadata = {
        "chunks_retrieved": len(chunks),
        "avg_semantic_score": (
            sum(c.get("semantic_score", 0) for c in chunks) / len(chunks)
            if chunks else 0.0
        ),
    }

    # Step 5 — Pre-generation eval
    pre_eval = evaluator.evaluate_diagnostic(diagnostic, context, retrieval_metadata)
    if pre_eval["verdict"] == "blocked":
        raise ValueError(
            "Cannot generate coaching: " + "; ".join(pre_eval.get("warnings", ["vague input"]))
        )

    # Step 6 — Generate coaching (linear: direct prompt, no tool loop)
    coaching_text = _generate_coaching_linear(
        context, diagnostic, chunks, coach_style, demo_mode, api_key=api_key
    )

    # Step 7 — Post-generation eval
    post_eval = evaluator.evaluate_coaching(coaching_text, diagnostic)

    return {
        "status": "coaching",
        "coaching_text": coaching_text,
        "clarification_question": "",
        "context": asdict(context),
        "diagnostic": asdict(diagnostic),
        "retrieved_chunks": chunks,
        "evaluation": {
            "pre": pre_eval,
            "post": post_eval,
            "overall_verdict": post_eval["verdict"],
        },
        "session_id": session_id,
        "tool_call_log": [],
        "iterations": 1,
    }


# Per-style word targets, all inside CoachingEvaluator's [120, 700] window
# (MIN/MAX_COACHING_WORDS) with margin. Without an explicit target, the 3
# styles were only differentiated by the single word "STYLE: {style}" in
# the prompt — the model's output length landed inside/outside the eval
# window essentially at random, which drove ~60% of "appropriate_length"
# failures on the 2026-07-21 live run (C10 post-fix) regardless of style.
STYLE_GUIDANCE = {
    "detailed": (
        "Write a thorough, step-by-step explanation, roughly 400-600 words. "
        "Cover the science behind the root cause, walk through each "
        "adjustment one at a time, and anticipate a likely follow-up question."
    ),
    "concise": (
        "Be brief and to the point, roughly 150-250 words. Give only the "
        "essential fix and the single most important reason why — no filler."
    ),
    "technical": (
        "Use precise technical language, roughly 300-450 words: extraction "
        "chemistry, exact parameter deltas, and the mechanism behind the fix."
    ),
}


def _generate_coaching_linear(
    context: BrewingContext,
    diagnostic,
    chunks: list[dict],
    style: str,
    demo_mode: bool,
    api_key: Optional[str] = None,
) -> str:
    """Generate coaching text via a direct LLM call (no agent loop)."""
    if demo_mode:
        return _demo_coaching(context, diagnostic)

    from engine.llm_client import LLMClient

    client = LLMClient(api_key=api_key)
    chunks_text = "\n\n".join(
        f"[{c.get('channel', '')} — {c.get('title', '')}]\n{c.get('text', '')[:400]}"
        for c in chunks[:3]
    )
    root_cause_name = (
        diagnostic.root_causes[0].hypothesis.replace("_", " ").replace("-", " ")
        if diagnostic.root_causes else "the underlying cause"
    )

    prompt = f"""Generate a barista coaching response.

MACHINE: {context.machine_type} {context.machine_model or ''}
PROBLEM: {context.raw_problem}
PRIMARY ROOT CAUSE: {diagnostic.root_causes[0].hypothesis if diagnostic.root_causes else 'unclear'}
ACTION PLAN:
{chr(10).join(f'{iv.step}. {iv.action}' for iv in diagnostic.intervention_plan)}

EXPERT KNOWLEDGE:
{chunks_text or 'Use general expertise.'}

STYLE: {style} — {STYLE_GUIDANCE[style]}
RULES:
- Give specific measurements (grams, seconds, degrees Celsius, grind notches).
- Name the root cause explicitly and early, using the phrase "{root_cause_name}" naturally in a sentence.
- Explain WHY the fix works with a clear "because"/"due to" statement.
- End with an explicit validation test starting with a phrase like "You should notice...",
  "You should taste...", "Test by...", or "Verify by...", so the user knows how to confirm the fix worked.
"""

    response = client.create(
        messages=[{"role": "user", "content": prompt}],
        system=(
            "You are HomeBarista Coach. Generate specific, science-backed barista coaching. "
            "Always include measurements, name and explain the root cause, end with a validation test. "
            "Follow the style's word-count target closely."
        ),
        max_tokens=1200,
        reasoning_effort="low",
    )
    return response.text


def _answer_general_linear(
    raw_problem: str, context: BrewingContext, api_key: Optional[str] = None
) -> str:
    """Answer a general (non-troubleshoot) coffee question with one LLM call.
    Used by the linear path (shared-key Live mode) for questions the taste-
    diagnostic pipeline can't handle: choosing beans, recipes, theory, buying.
    Scope-disciplined so it never gets pulled off coffee (see ScopeGuard)."""
    from engine.llm_client import LLMClient

    machine = ""
    if context.machine_type and context.machine_type != "unknown":
        machine = f" (their machine: {context.machine_type})"
    response = LLMClient(api_key=api_key).create(
        messages=[{"role": "user",
                   "content": f"Coffee question{machine}: {raw_problem}"}],
        system=(
            "You are HomeBarista Coach, an expert barista. Answer ONLY the "
            "coffee-related part of the question, accurately and practically. "
            "If any part is not about coffee, do not answer that part — briefly "
            "say you only help with coffee. Be specific and concise."
        ),
        max_tokens=700,
        reasoning_effort="low",
    )
    return response.text


def _demo_general_text() -> str:
    """Demo has no LLM, so it can't answer open coffee questions — point the
    user to Live mode instead of erroring."""
    return (
        "[DEMO MODE] That's a great coffee question — but demo mode only runs "
        "the deterministic **diagnostic engine**, which handles taste problems "
        "(bitter, sour, weak, thin crema...). For open questions like choosing "
        "beans, recipes, or brewing theory, switch to **Live (real AI "
        "coaching)** in the sidebar for a full AI-powered answer."
    )


def _info_result(session_id: str, text: str, context: BrewingContext) -> dict:
    """Standard result envelope for a non-diagnostic answer (general question
    or demo redirect) — status 'coaching' so it renders as a normal reply."""
    from dataclasses import asdict
    return {
        "status": "coaching",
        "coaching_text": text,
        "clarification_question": "",
        "context": asdict(context),
        "diagnostic": {},
        "retrieved_chunks": [],
        "evaluation": {},
        "session_id": session_id,
        "tool_call_log": [],
        "iterations": 1,
    }


def _demo_coaching(context: BrewingContext, diagnostic) -> str:
    """Return a mock coaching response for demo mode (no API call)."""
    symptom = diagnostic.symptoms[0] if diagnostic.symptoms else "coffee issue"
    cause = diagnostic.root_causes[0].hypothesis if diagnostic.root_causes else "unclear cause"
    action = diagnostic.intervention_plan[0].action if diagnostic.intervention_plan else "check your setup"

    return (
        f"[DEMO MODE] Based on your {context.machine_type} coffee problem ({symptom}), "
        f"the most likely cause is {cause}.\n\n"
        f"Recommended first step: {action}\n\n"
        f"This is a free demo preview. For full science-backed AI coaching, "
        f"switch to **Live (real AI coaching)** in the sidebar — unlock it with "
        f"the shared password or paste your own API key."
    )


def _mock_chunks() -> list[dict]:
    """Return empty chunk list for demo mode."""
    return []


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def _log_session(session_id: str, raw_problem: str, result: dict) -> None:
    """Append session to JSONL log file."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_problem": raw_problem[:200],
            "status": result.get("status"),
            "machine_type": result.get("context", {}).get("machine_type"),
            "symptoms": result.get("diagnostic", {}).get("symptoms", []),
            "verdict": result.get("evaluation", {}).get("overall_verdict"),
            "iterations": result.get("iterations", 0),
        }
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log session: {e}")


def _out_of_scope_result(session_id: str, message: str) -> dict:
    return {
        "status": "out_of_scope",
        "coaching_text": "",
        "clarification_question": message,
        "context": {},
        "diagnostic": {},
        "retrieved_chunks": [],
        "evaluation": {},
        "session_id": session_id,
        "tool_call_log": [],
        "iterations": 0,
    }


def _error_result(session_id: str, message: str) -> dict:
    return {
        "status": "error",
        "coaching_text": "",
        "clarification_question": message,
        "context": {},
        "diagnostic": {},
        "retrieved_chunks": [],
        "evaluation": {},
        "session_id": session_id,
        "tool_call_log": [],
        "iterations": 0,
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv
    load_dotenv()  # LLM_* config from .env (see .env.example)

    problem = " ".join(sys.argv[1:]) or "my DeLonghi Dinamica makes bitter espresso"
    result = asyncio.run(run_pipeline(problem, demo_mode=True))

    print(f"\nStatus   : {result['status']}")
    print(f"Session  : {result['session_id']}")
    if result.get("coaching_text"):
        print(f"\nCoaching :\n{result['coaching_text']}")
    if result.get("clarification_question"):
        print(f"\nNeeds clarification: {result['clarification_question']}")
