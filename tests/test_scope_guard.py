"""
ScopeGuard tests — the guard must refuse off-topic input deterministically,
with zero LLM calls, and let coffee questions (EN + FR) through.
"""

from pathlib import Path

import engine.scope_guard
from engine.scope_guard import ScopeGuard, REFUSAL_MESSAGE, TOO_LONG_MESSAGE


guard = ScopeGuard()


def test_refuses_off_topic_english():
    verdict = guard.check("write me a poem about Python")
    assert verdict["in_scope"] is False
    assert verdict["reason"] == "off_topic"
    assert verdict["message"] == REFUSAL_MESSAGE


def test_refuses_off_topic_french():
    verdict = guard.check("aide-moi à remplir mes impôts")
    assert verdict["in_scope"] is False
    assert verdict["reason"] == "off_topic"


def test_allows_french_coffee():
    verdict = guard.check("ma cafetière fait un café amer")
    assert verdict["in_scope"] is True
    assert verdict["reason"] == "ok"


def test_allows_followup_with_history():
    history = [
        {"role": "user", "content": "my espresso tastes bitter"},
        {"role": "assistant", "content": "How long is your shot running?"},
    ]
    verdict = guard.check("yes, 20 seconds", conversation_history=history)
    assert verdict["in_scope"] is True

    # Same follow-up without in-scope history must be refused.
    verdict_no_history = guard.check("yes, 20 seconds")
    assert verdict_no_history["in_scope"] is False


def test_refuses_oversized_input():
    verdict = guard.check("x" * 2000)
    assert verdict["in_scope"] is False
    assert verdict["reason"] == "too_long"
    assert verdict["message"] == TOO_LONG_MESSAGE


def test_zero_llm_calls():
    # The guard's whole reason to exist is costing 0 tokens: its module
    # must never import anthropic (nor any HTTP client). Parse the real
    # import statements rather than grepping text (docstrings mention them).
    import ast

    source = Path(engine.scope_guard.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"anthropic", "httpx", "requests", "urllib"})
