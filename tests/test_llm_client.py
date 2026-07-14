"""
LLMClient — offline tests (no network, no SDK required).
Verifies: provider/model/key resolution from env, JSON extraction, and the
neutral→provider message/tool conversion for both backends using fake SDK
clients injected into the lazy _sdk_client slot.
"""

import json
from types import SimpleNamespace

import pytest

from engine.llm_client import (
    DEFAULT_MODELS,
    LLMClient,
    LLMResponse,
    ToolCall,
    extract_json,
    get_api_key,
    strip_reasoning,
)


# ------------------------------------------------------------------
# Config resolution
# ------------------------------------------------------------------

def test_default_provider_is_groq(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = LLMClient()
    assert client.provider == "groq"
    assert client.model == DEFAULT_MODELS["groq"]


def test_env_overrides_provider_and_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5")
    client = LLMClient()
    assert client.provider == "anthropic"
    assert client.model == "claude-sonnet-5"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "yolo")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        LLMClient()


def test_generic_key_wins_over_provider_key(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_specific")
    monkeypatch.setenv("LLM_API_KEY", "generic")
    assert get_api_key() == "generic"
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert get_api_key() == "gsk_specific"


def test_explicit_api_key_beats_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from_env")
    client = LLMClient(api_key="byo_key")
    assert client.api_key == "byo_key"


# ------------------------------------------------------------------
# extract_json
# ------------------------------------------------------------------

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_and_prosed():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": 1}') == {"a": 1}


def test_extract_json_invalid_returns_none():
    assert extract_json("no json here") is None
    assert extract_json("{broken") is None


def test_strip_reasoning_removes_think_blocks():
    assert strip_reasoning("<think>hmm\nlines</think>\nOK") == "OK"
    # unclosed tag (max_tokens truncated mid-reasoning) → everything stripped
    assert strip_reasoning("<think>never closed...") == ""
    assert strip_reasoning("plain answer") == "plain answer"


# ------------------------------------------------------------------
# Rate-limit retry (Groq free tier: 429s must be retried, not fatal)
# ------------------------------------------------------------------

class FakeRateLimitError(Exception):
    """Stands in for openai.RateLimitError / anthropic.RateLimitError (duck-typed by name)."""
    __name__ = "RateLimitError"


FakeRateLimitError.__name__ = "RateLimitError"


def test_retries_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    sleeps = []
    monkeypatch.setattr("engine.llm_client.time.sleep", lambda s: sleeps.append(s))

    client = LLMClient(api_key="k")
    fake = FakeOpenAI(SimpleNamespace(content="ok", tool_calls=None))
    calls = {"n": 0}
    real_create = fake._create

    def flaky_create(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeRateLimitError("429")
        return real_create(**kwargs)

    fake._create = flaky_create
    fake.chat.completions.create = flaky_create
    client._sdk_client = fake

    response = client.create([{"role": "user", "content": "hi"}])
    assert response.text == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2  # retried twice before succeeding


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr("engine.llm_client.time.sleep", lambda s: None)

    client = LLMClient(api_key="k")
    fake = FakeOpenAI(SimpleNamespace(content="ok", tool_calls=None))

    def always_rate_limited(**kwargs):
        raise FakeRateLimitError("429")

    fake.chat.completions.create = always_rate_limited
    client._sdk_client = fake

    with pytest.raises(FakeRateLimitError):
        client.create([{"role": "user", "content": "hi"}])


def test_non_rate_limit_error_is_not_retried(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    sleeps = []
    monkeypatch.setattr("engine.llm_client.time.sleep", lambda s: sleeps.append(s))

    client = LLMClient(api_key="k")
    fake = FakeOpenAI(SimpleNamespace(content="ok", tool_calls=None))

    def broken_create(**kwargs):
        raise ValueError("not a rate limit")

    fake.chat.completions.create = broken_create
    client._sdk_client = fake

    with pytest.raises(ValueError):
        client.create([{"role": "user", "content": "hi"}])
    assert sleeps == []


def test_openai_response_strips_reasoning(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    client = LLMClient(api_key="k")
    client._sdk_client = FakeOpenAI(SimpleNamespace(
        content="<think>reasoning</think>final answer", tool_calls=None))
    response = client.create([{"role": "user", "content": "hi"}])
    assert response.text == "final answer"


# ------------------------------------------------------------------
# OpenAI-compatible backend conversion (Groq/OpenAI)
# ------------------------------------------------------------------

TOOLS = [{
    "name": "get_thing",
    "description": "Gets the thing.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}]


class FakeOpenAI:
    def __init__(self, message):
        self.captured = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._message = message

    def _create(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=self._message)])


def test_openai_conversion_system_tools_and_tool_history(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    client = LLMClient(api_key="k")
    fake = FakeOpenAI(SimpleNamespace(content="done", tool_calls=None))
    client._sdk_client = fake

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "name": "get_thing", "input": {"x": 1}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
    ]
    response = client.create(messages, system="sys", max_tokens=64, tools=TOOLS)

    sent = fake.captured["messages"]
    assert sent[0] == {"role": "system", "content": "sys"}
    assert sent[2]["tool_calls"][0]["function"]["name"] == "get_thing"
    assert json.loads(sent[2]["tool_calls"][0]["function"]["arguments"]) == {"x": 1}
    assert sent[2]["content"] is None  # empty text must become None
    assert sent[3] == {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'}
    assert fake.captured["tools"][0]["function"]["parameters"] == TOOLS[0]["input_schema"]
    assert response.stop_reason == "end_turn"
    assert response.text == "done"


def test_openai_tool_call_response_parsed(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    client = LLMClient(api_key="k")
    tc = SimpleNamespace(id="c9", function=SimpleNamespace(
        name="get_thing", arguments='{"y": 2}'))
    client._sdk_client = FakeOpenAI(SimpleNamespace(content=None, tool_calls=[tc]))

    response = client.create([{"role": "user", "content": "hi"}], tools=TOOLS)
    assert response.stop_reason == "tool_use"
    assert response.tool_calls == [ToolCall(id="c9", name="get_thing", input={"y": 2})]
    assert response.text == ""


# ------------------------------------------------------------------
# Anthropic backend conversion
# ------------------------------------------------------------------

class FakeAnthropic:
    def __init__(self, response):
        self.captured = None
        self.messages = SimpleNamespace(create=self._create)
        self._response = response

    def _create(self, **kwargs):
        self.captured = kwargs
        return self._response


def test_anthropic_conversion_merges_tool_results(monkeypatch):
    client = LLMClient(provider="anthropic", api_key="k")
    fake = FakeAnthropic(SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")], stop_reason="end_turn"))
    client._sdk_client = fake

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "thinking", "tool_calls": [
            {"id": "a1", "name": "get_thing", "input": {}},
            {"id": "a2", "name": "get_thing", "input": {}},
        ]},
        {"role": "tool", "tool_call_id": "a1", "content": "r1"},
        {"role": "tool", "tool_call_id": "a2", "content": "r2"},
    ]
    response = client.create(messages, system="sys", tools=TOOLS)

    sent = fake.captured["messages"]
    assert fake.captured["system"] == "sys"
    assert fake.captured["tools"] == TOOLS
    # assistant turn: text block + 2 tool_use blocks
    assert [b["type"] for b in sent[1]["content"]] == ["text", "tool_use", "tool_use"]
    # both tool results merged into ONE user turn (Anthropic requirement)
    assert sent[2]["role"] == "user"
    assert [b["tool_use_id"] for b in sent[2]["content"]] == ["a1", "a2"]
    assert len(sent) == 3
    assert response.text == "ok"


# ------------------------------------------------------------------
# reasoning_effort passthrough (OpenAI-compatible only, e.g. Groq gpt-oss)
# ------------------------------------------------------------------

def test_reasoning_effort_passthrough(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    client = LLMClient(api_key="k")
    fake = FakeOpenAI(SimpleNamespace(content="ok", tool_calls=None))
    client._sdk_client = fake

    client.create([{"role": "user", "content": "hi"}], reasoning_effort="low")
    assert fake.captured["reasoning_effort"] == "low"


def test_reasoning_effort_absent_when_not_passed(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    client = LLMClient(api_key="k")
    fake = FakeOpenAI(SimpleNamespace(content="ok", tool_calls=None))
    client._sdk_client = fake

    client.create([{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in fake.captured


def test_reasoning_effort_ignored_for_anthropic():
    client = LLMClient(provider="anthropic", api_key="k")
    fake = FakeAnthropic(SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")], stop_reason="end_turn"))
    client._sdk_client = fake

    client.create([{"role": "user", "content": "hi"}], reasoning_effort="low")
    assert "reasoning_effort" not in fake.captured


# ------------------------------------------------------------------
# Neutral history helpers
# ------------------------------------------------------------------

def test_assistant_and_tool_result_helpers():
    response = LLMResponse(
        text="t", tool_calls=[ToolCall(id="i", name="n", input={"a": 1})],
        stop_reason="tool_use",
    )
    msg = LLMClient.assistant_message(response)
    assert msg == {"role": "assistant", "content": "t",
                   "tool_calls": [{"id": "i", "name": "n", "input": {"a": 1}}]}
    assert LLMClient.tool_result_message("i", "r") == {
        "role": "tool", "tool_call_id": "i", "content": "r"}
