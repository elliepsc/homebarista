"""
Agent orchestration tests — assert the AGENTIC guarantees, not just the text.
================================================================================
The README's headline claim is "true agency, not a fixed pipeline". Nothing
tested that until now. These tests pin the two properties that actually matter
for the claim to be defensible, WITHOUT any API key or ML stack:

  1. Tool ordering on the happy path: extract -> retrieve -> generate -> validate.
  2. A super-automatic NEVER gets a physically impossible intervention
     (tamping / distribution) surfaced through the FULL agent path — not just
     via the planner in isolation.
  3. ask_clarification halts the loop before any coaching is generated.

The LLM is a scripted fake (loop calls with tools=... get scripted tool_use
responses; the inner generation call with no tools returns coaching text).
The retriever is a stub. Everything else (extractor/planner/evaluator) is real.
"""

import importlib.util
import hashlib
import sys
import types

# --- Import guard -----------------------------------------------------------
# orchestration.agent -> pipeline.retriever -> ingestion.embedder / vector_store
# import sentence_transformers and chromadb AT MODULE TOP LEVEL (a break from
# the lazy-import discipline used elsewhere, e.g. pipeline.pipeline). This test
# needs neither — it injects a stub retriever and a scripted client. To keep the
# test runnable WITHOUT the ML stack, we stub those modules ONLY when they are
# genuinely absent. In CI (deps installed) find_spec() is not None, so we do NOT
# stub and the real modules are used — no sys.modules pollution for other tests.
class _PermissiveModule(types.ModuleType):
    """Stand-in module: any attribute access returns a harmless placeholder, so
    top-level `chromadb.ClientAPI` annotations etc. resolve at import time. None
    of it is ever called — the agent path under test uses a stub retriever."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return object


class _FakeEmbedding(list):
    def tolist(self):
        return list(self)


class _FakeSentenceTransformer:
    """Tiny deterministic fallback for environments without sentence-transformers.

    It is good enough for tests that only need stable vectors, while keeping the
    orchestration tests runnable on a light local install.
    """

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vectors = [_FakeEmbedding(self._vector(text)) for text in items]
        return vectors[0] if single else _FakeEmbedding(vectors)

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
        return [((byte / 255.0) * 2.0) - 1.0 for byte in digest[:8]]


def _cosine_distance(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if not norm_a or not norm_b:
        return 1.0
    return 1.0 - (dot / (norm_a * norm_b))


class _FakeChromaCollection:
    def __init__(self):
        self._rows = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for item_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            self._rows[item_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            }

    def count(self):
        return len(self._rows)

    def query(self, query_embeddings, n_results, include=None, where=None):
        query_embedding = query_embeddings[0]
        rows = []
        for item_id, row in self._rows.items():
            metadata = row["metadata"]
            if where and not _metadata_matches(metadata, where):
                continue
            rows.append((item_id, row, _cosine_distance(query_embedding, row["embedding"])))
        rows.sort(key=lambda item: item[2])
        rows = rows[:n_results]
        return {
            "ids": [[item_id for item_id, _, _ in rows]],
            "documents": [[row["document"] for _, row, _ in rows]],
            "metadatas": [[row["metadata"] for _, row, _ in rows]],
            "distances": [[distance for _, _, distance in rows]],
        }

    def get(self, limit=None, include=None):
        rows = list(self._rows.values())[:limit]
        return {"metadatas": [row["metadata"] for row in rows]}


def _metadata_matches(metadata, where):
    for key, expected in where.items():
        value = metadata.get(key)
        if isinstance(expected, dict) and "$in" in expected:
            if value not in expected["$in"]:
                return False
        elif value != expected:
            return False
    return True


class _FakeChromaClient:
    def __init__(self, *args, **kwargs):
        self._collections = {}

    def get_or_create_collection(self, name, metadata=None):
        return self._collections.setdefault(name, _FakeChromaCollection())

    def delete_collection(self, name):
        self._collections.pop(name, None)


def _stub_if_absent(name, attrs=None, package=False):
    if importlib.util.find_spec(name) is None and name not in sys.modules:
        mod = _PermissiveModule(name)
        if package:
            mod.__path__ = []
        for key, value in (attrs or {}).items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod
    return None

_stub_if_absent("sentence_transformers",
                {"SentenceTransformer": _FakeSentenceTransformer, "CrossEncoder": object})
if importlib.util.find_spec("chromadb") is None and "chromadb" not in sys.modules:
    _chromadb = _stub_if_absent("chromadb", package=True)
    _chromadb.Client = _FakeChromaClient
    _chromadb.PersistentClient = _FakeChromaClient
    _chromadb.ClientAPI = _FakeChromaClient
    _cfg = _PermissiveModule("chromadb.config")
    _cfg.Settings = type("Settings", (), {"__init__": lambda self, *args, **kwargs: None})
    _chromadb.config = _cfg
    sys.modules["chromadb.config"] = _cfg
# ---------------------------------------------------------------------------

from engine.llm_client import LLMResponse, ToolCall
from engine.symptom_extractor import SymptomExtractor
from engine.diagnostic_planner import DiagnosticPlanner
from engine.coaching_evaluator import CoachingEvaluator
from orchestration.agent import HomeBaristaAgent


# ------------------------------------------------------------------
# Doubles
# ------------------------------------------------------------------

class StubRetriever:
    """Returns fixed, high-scoring chunks — no embeddings, no ChromaDB."""

    def retrieve(self, context, diagnostic, n_candidates=15, query_override=None):
        return [
            {"semantic_score": 0.82, "title": "Dialing in espresso",
             "channel": "James Hoffmann",
             "text": "Adjust grind to tune extraction and balance. " * 6},
            {"semantic_score": 0.61, "title": "Fixing bitterness",
             "channel": "Lance Hedrick",
             "text": "Bitter espresso is usually over-extraction. " * 6},
        ]


class ScriptedClient:
    """Fake LLMClient.

    - Loop calls (tools passed) return the next scripted tool_use / end_turn.
    - The inner generate_coaching call (no tools) returns coaching text.
    """

    def __init__(self, script):
        self._script = script
        self._i = 0
        self.loop_calls = 0
        self.generation_calls = 0

    def create(self, messages, system=None, max_tokens=1024,
               tools=None, reasoning_effort=None):
        if not tools:                       # inner generation (no tool loop)
            self.generation_calls += 1
            return LLMResponse(
                text=("Your espresso is bitter because of over-extraction: the grind is "
                      "too fine, so contact time runs long. Go 1 notch coarser and lower "
                      "the temperature by 2 degrees C. You should notice less bitterness "
                      "within 2-3 shots."),
            )
        resp = self._script[self._i]
        self._i += 1
        self.loop_calls += 1
        return resp


def _tool_use(idx, name, inp):
    return LLMResponse(
        text="",
        tool_calls=[ToolCall(id=f"call_{idx}", name=name, input=inp)],
        stop_reason="tool_use",
    )


def _make_agent(script):
    return HomeBaristaAgent(
        symptom_extractor=SymptomExtractor(demo_mode=True),
        diagnostic_planner=DiagnosticPlanner(),
        retriever=StubRetriever(),
        coaching_evaluator=CoachingEvaluator(),
        llm_client=ScriptedClient(script),
        demo_mode=False,
    )


def _executed_tools(result):
    order = []
    for entry in result["tool_call_log"]:
        for tc in entry.get("tool_calls", []):
            order.append(tc["tool"])
    return order


SUPER_AUTO_MSG = "My DeLonghi super-automatic makes bitter espresso every morning"

HAPPY_PATH = [
    _tool_use(1, "extract_symptoms", {"user_description": SUPER_AUTO_MSG}),
    _tool_use(2, "retrieve_knowledge",
              {"query": "super automatic bitter over-extraction grind",
               "machine_type": "super_automatic"}),
    _tool_use(3, "generate_coaching", {"style": "technical"}),
    _tool_use(4, "validate_coaching",
              {"coaching_text": "Go 1 notch coarser at 93 degrees C. "
                                "You should notice less bitterness."}),
    LLMResponse(text="Here is your coaching plan.", stop_reason="end_turn"),
]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_tool_ordering_happy_path():
    agent = _make_agent(list(HAPPY_PATH))
    result = agent.run(SUPER_AUTO_MSG, style="technical")

    assert result["status"] == "coaching"
    assert result["coaching_text"]
    assert _executed_tools(result) == [
        "extract_symptoms", "retrieve_knowledge",
        "generate_coaching", "validate_coaching",
    ]


def test_retrieve_and_generate_are_ordered():
    agent = _make_agent(list(HAPPY_PATH))
    order = _executed_tools(agent.run(SUPER_AUTO_MSG))
    assert order.index("extract_symptoms") < order.index("retrieve_knowledge")
    assert order.index("retrieve_knowledge") < order.index("generate_coaching")
    assert order.index("generate_coaching") < order.index("validate_coaching")


def test_generate_uses_the_inner_llm_call():
    client = ScriptedClient(list(HAPPY_PATH))
    agent = HomeBaristaAgent(
        symptom_extractor=SymptomExtractor(demo_mode=True),
        diagnostic_planner=DiagnosticPlanner(),
        retriever=StubRetriever(),
        coaching_evaluator=CoachingEvaluator(),
        llm_client=client,
        demo_mode=False,
    )
    agent.run(SUPER_AUTO_MSG)
    # exactly one inner (tool-less) generation call happened
    assert client.generation_calls == 1


def test_super_automatic_never_gets_impossible_intervention():
    agent = _make_agent(list(HAPPY_PATH))
    result = agent.run(SUPER_AUTO_MSG)

    # sanity: the machine was actually detected as super-automatic
    assert result["context"]["machine_type"] == "super_automatic"

    # params surfaced to the LLM exclude what the machine can't do
    params = agent._get_adjustable_params()
    assert "tamping" not in params
    assert "distribution" not in params

    # and the diagnostic plan itself never proposes them
    for iv in result["diagnostic"]["intervention_plan"]:
        assert iv["parameter"] not in ("tamping", "distribution"), \
            f"Impossible intervention surfaced for super_automatic: {iv}"


def test_ask_clarification_halts_before_coaching():
    script = [
        _tool_use(1, "extract_symptoms", {"user_description": "my coffee is off somehow"}),
        _tool_use(2, "ask_clarification",
                  {"question": "What machine do you use?", "reason": "machine unknown"}),
        LLMResponse(text="unreachable", stop_reason="end_turn"),
    ]
    agent = _make_agent(script)
    result = agent.run("my coffee tastes a bit off lately")

    assert result["status"] == "clarification_needed"
    assert "machine" in result["clarification_question"].lower()
    assert "generate_coaching" not in _executed_tools(result)
