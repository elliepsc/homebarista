"""
Pipeline — Orchestrator
========================
Single entry point for HomeBarista Coach.

Two modes:
1. AGENT mode (default): uses HomeBaristaAgent with tool-use loop.
   The LLM decides the flow — true agentic architecture.
   Requires ANTHROPIC_API_KEY.

2. LINEAR mode (fallback/CI): deterministic pipeline without LLM agency.
   Used in demo mode and tests to avoid API calls.
   Same components, fixed order.

Usage from Streamlit or CLI:
    from homebarista.pipeline import run_pipeline
    result = await run_pipeline("my DeLonghi makes bitter espresso")
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import anthropic

from homebarista.models import BrewingContext, CoachingSession
from homebarista.symptom_extractor import SymptomExtractor
from homebarista.diagnostic_planner import DiagnosticPlanner
from homebarista.embedder import Embedder
from homebarista.vector_store import VectorStore
from homebarista.retriever import Retriever
from homebarista.coaching_evaluator import CoachingEvaluator
from homebarista.agent import HomeBaristaAgent


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

LOG_DIR  = Path("logs")
LOG_FILE = LOG_DIR / "sessions.jsonl"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("homebarista")


# ------------------------------------------------------------------
# Component initialisation (lazy singletons)
# ------------------------------------------------------------------

_embedder:   Optional[Embedder]   = None
_store:      Optional[VectorStore] = None
_retriever:  Optional[Retriever]  = None
_extractor:  Optional[SymptomExtractor]  = None
_planner:    Optional[DiagnosticPlanner] = None
_evaluator:  Optional[CoachingEvaluator] = None
_agent:      Optional[HomeBaristaAgent]  = None


def _get_components(demo_mode: bool, use_cross_encoder: bool = True):
    """Lazy-initialise all components. Reuse across requests."""
    global _embedder, _store, _retriever, _extractor, _planner, _evaluator, _agent

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
        _extractor = SymptomExtractor(demo_mode=demo_mode)

    if _planner is None:
        _planner = DiagnosticPlanner()

    if _evaluator is None:
        _evaluator = CoachingEvaluator()

    if _agent is None and not demo_mode:
        _agent = HomeBaristaAgent(
            symptom_extractor=_extractor,
            diagnostic_planner=_planner,
            retriever=_retriever,
            coaching_evaluator=_evaluator,
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

    Returns:
        {
            "status":                 "coaching" | "clarification_needed" | "error"
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

    extractor, planner, retriever, evaluator, agent = _get_components(demo_mode)

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
            )

    except ValueError as e:
        return _error_result(session_id, str(e))
    except Exception as e:
        logger.exception(f"Session {session_id} failed")
        return _error_result(session_id, f"Internal error: {e}")

    # Log session
    _log_session(session_id, raw_problem, result)

    return result


async def _run_linear(
    raw_problem: str,
    coach_style: str,
    session_id: str,
    extractor: SymptomExtractor,
    planner: DiagnosticPlanner,
    retriever: Retriever,
    evaluator: CoachingEvaluator,
    demo_mode: bool,
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
    chunks = retriever.retrieve(context, diagnostic) if not demo_mode else _mock_chunks()
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
        context, diagnostic, chunks, coach_style, demo_mode
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
) -> str:
    """Generate coaching text via direct Anthropic call (no agent loop)."""
    if demo_mode:
        return _demo_coaching(context, diagnostic)

    client = anthropic.Anthropic()
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

    response = client.messages.create(
        model="claude-haiku-3-5",
        max_tokens=1200,
        system=(
            "You are HomeBarista Coach. Generate specific, science-backed barista coaching. "
            "Always include measurements, explain the root cause, end with a validation test."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _demo_coaching(context: BrewingContext, diagnostic) -> str:
    """Return a mock coaching response for demo mode (no API call)."""
    symptom = diagnostic.symptoms[0] if diagnostic.symptoms else "coffee issue"
    cause = diagnostic.root_causes[0].hypothesis if diagnostic.root_causes else "unclear cause"
    action = diagnostic.intervention_plan[0].action if diagnostic.intervention_plan else "check your setup"

    return (
        f"[DEMO MODE] Based on your {context.machine_type} coffee problem ({symptom}), "
        f"the most likely cause is {cause}.\n\n"
        f"Recommended first step: {action}\n\n"
        f"To see real AI-powered coaching, set DEMO_MODE=false and provide your ANTHROPIC_API_KEY."
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

    problem = " ".join(sys.argv[1:]) or "my DeLonghi Dinamica makes bitter espresso"
    result = asyncio.run(run_pipeline(problem, demo_mode=True))

    print(f"\nStatus   : {result['status']}")
    print(f"Session  : {result['session_id']}")
    if result.get("coaching_text"):
        print(f"\nCoaching :\n{result['coaching_text']}")
    if result.get("clarification_question"):
        print(f"\nNeeds clarification: {result['clarification_question']}")
