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
    """Lazy-initialise all components. Reuse across requests."""
    global _embedder, _store, _retriever, _extractor, _planner, _evaluator
    global _agent, _llm_api_key, _components_demo

    from ingestion.embedder import Embedder
    from orchestration.agent import HomeBaristaAgent
    from pipeline.retriever import Retriever
    from pipeline.vector_store import VectorStore

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

    if _embedder is None:
        _embedder = Embedder()

    if _store is None:
        _store = VectorStore(demo_mode=demo_mode)

    if _retriever is None:
        _retriever = Retriever(
            _embedder,
            _store,
            use_cross_encoder=use_cross_encoder and not demo_mode,
        )

    if _extractor is None:
        llm_client = None
        if not demo_mode:
            from engine.llm_client import LLMClient
            llm_client = LLMClient(api_key=api_key)
        _extractor = SymptomExtractor(llm_client=llm_client, demo_mode=demo_mode)

    if _planner is None:
        _planner = DiagnosticPlanner()

    if _evaluator is None:
        _evaluator = CoachingEvaluator()

    if _agent is None and not demo_mode:
        from engine.llm_client import LLMClient
        _agent = HomeBaristaAgent(
            symptom_extractor=_extractor,
            diagnostic_planner=_planner,
            retriever=_retriever,
            coaching_evaluator=_evaluator,
            llm_client=LLMClient(api_key=api_key),
            demo_mode=demo_mode,
        )

    return _extractor, _planner, _retriever, _evaluator, _agent


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

async def run_pipeline(
    raw_problem: str,
    coach_style: Literal["detailed", "concise", "technical"] = "detailed",
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
    except Exception:
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

    # Step 3 — Guard: too vague?
    if diagnostic.diagnostic_confidence < 0.15 and context.goal == "troubleshoot":
        raise ValueError(
            "Could not diagnose the problem. Please describe: "
            "your machine, what tastes wrong (bitter/sour/weak), and any parameters you know."
        )

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

    prompt = f"""Generate a barista coaching response.

MACHINE: {context.machine_type} {context.machine_model or ''}
PROBLEM: {context.raw_problem}
PRIMARY ROOT CAUSE: {diagnostic.root_causes[0].hypothesis if diagnostic.root_causes else 'unclear'}
ACTION PLAN:
{chr(10).join(f'{iv.step}. {iv.action}' for iv in diagnostic.intervention_plan)}

EXPERT KNOWLEDGE:
{chunks_text or 'Use general expertise.'}

STYLE: {style}
RULES: Give specific measurements. Explain WHY. End with a validation test.
"""

    response = client.create(
        messages=[{"role": "user", "content": prompt}],
        system=(
            "You are HomeBarista Coach. Generate specific, science-backed barista coaching. "
            "Always include measurements, explain the root cause, end with a validation test."
        ),
        max_tokens=1200,
        reasoning_effort="low",
    )
    return response.text


def _demo_coaching(context: BrewingContext, diagnostic) -> str:
    """Return a mock coaching response for demo mode (no API call)."""
    symptom = diagnostic.symptoms[0] if diagnostic.symptoms else "coffee issue"
    cause = diagnostic.root_causes[0].hypothesis if diagnostic.root_causes else "unclear cause"
    action = diagnostic.intervention_plan[0].action if diagnostic.intervention_plan else "check your setup"

    return (
        f"[DEMO MODE] Based on your {context.machine_type} coffee problem ({symptom}), "
        f"the most likely cause is {cause}.\n\n"
        f"Recommended first step: {action}\n\n"
        f"To see real AI-powered coaching, set DEMO_MODE=false and configure an LLM key "
        f"in .env (GROQ_API_KEY by default — see .env.example)."
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
