"""
LLMClient — single provider-agnostic gateway for every LLM call.
================================================================
ALL model/provider settings live in .env (see .env.example). Nothing
else in the codebase hardcodes a model name or imports a provider SDK.

    LLM_PROVIDER   groq (default, free tier) | anthropic | openai
    LLM_MODEL      overrides the per-provider default (DEFAULT_MODELS)
    LLM_API_KEY    generic key — wins over the provider-specific one
    GROQ_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY

Call sites:
    orchestration/agent.py        — agent tool-use loop
    engine/symptom_extractor.py   — LLM fallback extraction
    pipeline/pipeline.py          — linear-mode generation
    ingestion/content_classifier.py — low-confidence classification
    evals/run_rag_eval.py         — LLM judge
    evals/run_retrieval_eval.py   — synthetic query generation

Provider-neutral message format used by callers:
    {"role": "user"|"assistant", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{"id","name","input"}]}
    {"role": "tool", "tool_call_id": str, "content": str}
Tools are declared Anthropic-style ({"name","description","input_schema"})
and converted to the OpenAI function format when needed.

SDKs are imported lazily and the network client is only built on the first
call — demo mode and tests never import groq/anthropic/openai SDKs.
Groq and OpenAI share the OpenAI-compatible chat.completions API (`openai`
package, base_url switch); Anthropic uses the `anthropic` package.
"""

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional

# NOTE: .env loading is done by the entry points (app/streamlit_app.py,
# ingestion/run_ingestion.py, evals/*, pipeline CLI) — never here at import
# time, so importing this module has zero environment side effects.

DEFAULT_PROVIDER = "groq"

DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
}

KEY_ENV_VARS = {
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Free-tier providers (Groq default) rate-limit on tokens/minute, not just
# requests/minute — a single generation call can burn a large chunk of that
# budget. Retry 429s with backoff instead of failing the whole eval/pipeline run.
RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BASE_DELAY = 1.0


def _is_rate_limit_error(exc: Exception) -> bool:
    """Duck-typed on purpose — avoids importing openai/anthropic just to catch their errors."""
    if type(exc).__name__ == "RateLimitError":
        return True
    return getattr(exc, "status_code", None) == 429


def _rate_limit_delay(exc: Exception, attempt: int) -> float:
    """Honor the API's Retry-After header when present, else exponential backoff + jitter."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    retry_after = headers.get("retry-after") if headers else None
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return RATE_LIMIT_BASE_DELAY * (2**attempt) + random.uniform(0, 0.5)


def _with_rate_limit_retry(call):
    """Run `call()`, retrying on 429s with backoff. Re-raises other errors immediately."""
    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return call()
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == RATE_LIMIT_MAX_RETRIES:
                raise
            time.sleep(_rate_limit_delay(exc, attempt))


def get_provider() -> str:
    """Active provider from LLM_PROVIDER (default: groq)."""
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Supported: {', '.join(DEFAULT_MODELS)} (see .env.example)."
        )
    return provider


def get_api_key(provider: Optional[str] = None) -> Optional[str]:
    """Resolve the API key: LLM_API_KEY wins, else the provider-specific var."""
    provider = provider or get_provider()
    return os.environ.get("LLM_API_KEY") or os.environ.get(KEY_ENV_VARS[provider])


# Reasoning models that inline their chain of thought as <think>...</think>
# before the answer (qwen3, deepseek-r1...) need it stripped — users must
# never see it, and JSON parsing must not trip on it. Handles the unclosed
# tag too (when max_tokens truncates mid-reasoning). No-op for gpt-oss (the
# default Groq model): it returns reasoning in a separate response field,
# not inline. Kept as a safety net for a qwen3 fallback via LLM_BASE_URL
# (e.g. a self-hosted/OpenRouter endpoint).
_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def strip_reasoning(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def extract_json(text: str) -> Optional[dict]:
    """
    Best-effort JSON extraction from an LLM reply: tolerates markdown fences
    and surrounding prose. Returns None if no valid JSON object is found.
    """
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn" | "tool_use"


class LLMClient:
    """
    Thin chat wrapper over Groq / Anthropic / OpenAI.
    Construction is free of network calls; the SDK client is built lazily.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = (provider or get_provider()).strip().lower()
        if self.provider not in DEFAULT_MODELS:
            raise ValueError(
                f"Unknown provider '{self.provider}'. "
                f"Supported: {', '.join(DEFAULT_MODELS)}."
            )
        self.model = (
            model or os.environ.get("LLM_MODEL") or DEFAULT_MODELS[self.provider]
        )
        self.api_key = api_key or get_api_key(self.provider)
        self._sdk_client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 1024,
        tools: Optional[list[dict]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> LLMResponse:
        """One chat completion in the neutral format described above.

        reasoning_effort ("low"|"medium"|"high") is forwarded only to the
        OpenAI-compatible backend (e.g. Groq's gpt-oss reasoning control) —
        the Anthropic backend ignores it.
        """
        if self.provider == "anthropic":
            return self._create_anthropic(messages, system, max_tokens, tools)
        return self._create_openai_compatible(
            messages, system, max_tokens, tools, reasoning_effort
        )

    @staticmethod
    def assistant_message(response: LLMResponse) -> dict:
        """Neutral assistant turn to append to the message history."""
        msg: dict = {"role": "assistant", "content": response.text}
        if response.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "input": tc.input}
                for tc in response.tool_calls
            ]
        return msg

    @staticmethod
    def tool_result_message(tool_call_id: str, content: str) -> dict:
        """Neutral tool-result turn to append to the message history."""
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    # ------------------------------------------------------------------
    # OpenAI-compatible backend (Groq + OpenAI)
    # ------------------------------------------------------------------

    def _openai_client(self):
        if self._sdk_client is None:
            from openai import OpenAI

            kwargs: dict = {"api_key": self.api_key}
            # LLM_BASE_URL overrides the endpoint — lets any OpenAI-compatible
            # server (Ollama, OpenRouter, vLLM...) be used without code change.
            base_url = os.environ.get("LLM_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
            elif self.provider == "groq":
                kwargs["base_url"] = GROQ_BASE_URL
            self._sdk_client = OpenAI(**kwargs)
        return self._sdk_client

    def _create_openai_compatible(
        self, messages, system, max_tokens, tools, reasoning_effort=None
    ):
        oa_messages: list[dict] = []
        if system:
            oa_messages.append({"role": "system", "content": system})
        for msg in messages:
            if msg["role"] == "tool":
                oa_messages.append({
                    "role": "tool",
                    "tool_call_id": msg["tool_call_id"],
                    "content": msg["content"],
                })
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                oa_messages.append({
                    "role": "assistant",
                    "content": msg.get("content") or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["input"]),
                            },
                        }
                        for tc in msg["tool_calls"]
                    ],
                })
            else:
                oa_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs: dict = {}
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        completion = _with_rate_limit_retry(lambda: self._openai_client().chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=oa_messages,
            **kwargs,
        ))
        choice = completion.choices[0]
        tool_calls = []
        for tc in (choice.message.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

        return LLMResponse(
            text=strip_reasoning(choice.message.content or ""),
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )

    # ------------------------------------------------------------------
    # Anthropic backend
    # ------------------------------------------------------------------

    def _anthropic_client(self):
        if self._sdk_client is None:
            import anthropic

            self._sdk_client = anthropic.Anthropic(api_key=self.api_key)
        return self._sdk_client

    def _create_anthropic(self, messages, system, max_tokens, tools):
        an_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": msg["content"],
                }
                # Consecutive tool results must share a single user turn.
                if (
                    an_messages
                    and an_messages[-1]["role"] == "user"
                    and isinstance(an_messages[-1]["content"], list)
                ):
                    an_messages[-1]["content"].append(block)
                else:
                    an_messages.append({"role": "user", "content": [block]})
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                content: list[dict] = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                content.extend(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["input"],
                    }
                    for tc in msg["tool_calls"]
                )
                an_messages.append({"role": "assistant", "content": content})
            else:
                an_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs: dict = {}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = _with_rate_limit_retry(lambda: self._anthropic_client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=an_messages,
            **kwargs,
        ))
        text = strip_reasoning(
            "".join(b.text for b in response.content if b.type == "text")
        )
        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=b.input)
            for b in response.content
            if b.type == "tool_use"
        ]
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason="tool_use" if response.stop_reason == "tool_use" else "end_turn",
        )
