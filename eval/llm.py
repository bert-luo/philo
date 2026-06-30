"""Thin OpenAI-compatible chat client that works for both OpenAI and OpenRouter.

Exposes one function, `chat`, that handles the per-provider parameter quirks
(GPT-5 reasoning models reject `temperature`/`max_tokens`) and records spend.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI, BadRequestError

from . import config
from .config import CostTracker
from .models import Model

_CLIENTS: dict[str, OpenAI] = {}


# --- tool-call hygiene ------------------------------------------------------
# Some OpenRouter providers (seen with qwen3.6, mimo) emit malformed `arguments`
# JSON — trailing junk after the object ("Extra data") or a truncated string
# ("Unterminated string"). Echoing that verbatim in the next request makes the
# strict provider 400 the whole call. We parse tolerantly and ALWAYS re-serialize.

def parse_tool_args(raw: str | None) -> dict:
    """Best-effort parse of a tool call's `arguments` string into a dict.

    Recovers the leading object from 'Extra data' cases via raw_decode; returns {}
    for anything unrecoverable (e.g. a truncated/unterminated string)."""
    s = (raw or "").strip()
    if not s:
        return {}
    try:
        v = json.loads(s)
    except json.JSONDecodeError:
        try:
            v, _ = json.JSONDecoder().raw_decode(s)
        except Exception:
            return {}
    return v if isinstance(v, dict) else {}


def assistant_tool_calls(tcs) -> tuple[list[dict] | None, dict[str, dict]]:
    """Turn raw model tool_calls into a replay-safe assistant payload.

    Every echoed call gets a re-serialized (guaranteed-valid) `arguments` string;
    nameless calls are dropped. Returns (payload_or_None, {tool_call_id: parsed_args})
    so callers reuse the exact args we will replay."""
    payload, parsed = [], {}
    for tc in tcs:
        name = getattr(getattr(tc, "function", None), "name", None)
        if not name:
            continue
        args = parse_tool_args(tc.function.arguments)
        parsed[tc.id] = args
        payload.append({"id": tc.id, "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}})
    return (payload or None), parsed


def _client(model: Model) -> OpenAI:
    if model.provider not in _CLIENTS:
        if model.provider == "openai":
            _CLIENTS["openai"] = OpenAI(api_key=config.OPENAI_API_KEY)
        elif model.provider == "openrouter":
            _CLIENTS["openrouter"] = OpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            raise ValueError(f"unknown provider {model.provider}")
    return _CLIENTS[model.provider]


def chat(
    model: Model,
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_choice: Any = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    retries: int = 6,
):
    """Return a raw chat completion `message` object (with .content / .tool_calls)."""
    client = _client(model)
    kwargs: dict[str, Any] = {"model": model.model_id, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

    if model.is_reasoning:
        kwargs["max_completion_tokens"] = max_tokens
        if model.reasoning_effort:
            kwargs["reasoning_effort"] = model.reasoning_effort
        # reasoning models only accept the default temperature; omit it
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature

    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            u = getattr(resp, "usage", None)
            if u:
                CostTracker.add(
                    getattr(u, "prompt_tokens", 0) or 0,
                    getattr(u, "completion_tokens", 0) or 0,
                    model.in_price, model.out_price,
                )
            return resp.choices[0].message
        except BadRequestError as e:
            # Drop a param the model dislikes and retry once before giving up.
            msg = str(e).lower()
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature"); continue
            if "max_tokens" in msg and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens"); continue
            # gpt-5.5 rejects reasoning_effort WITH function tools on chat.completions
            # (it wants /v1/responses). Drop the effort flag and run at default reasoning.
            if "reasoning_effort" in msg and "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort"); continue
            # OpenRouter "provider returned error" 400s are often a single flaky
            # upstream provider; a retry may route to a healthy one.
            if ("provider returned error" in msg or "extra data" in msg) \
                    and attempt < retries - 1:
                last_err = e
                time.sleep(2 * (attempt + 1)); continue
            last_err = e
            break
        except Exception as e:  # rate limits, transient 5xx
            last_err = e
            msg = str(e).lower()
            if "rate limit" in msg or "rate_limit" in msg or "429" in msg:
                # TPM windows are per-minute, so a short backoff just re-hits the wall.
                # Honor an explicit "try again in Xs" hint, else escalate toward ~1 min.
                m = re.search(r"try again in ([\d.]+)\s*(ms|s)", msg)
                if m:
                    secs = float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)
                    wait = min(60.0, max(5.0, secs + 1.0))
                else:
                    wait = min(60.0, 8.0 * (attempt + 1))
            else:
                wait = 2.0 * (attempt + 1)
            time.sleep(wait)
    raise RuntimeError(f"chat failed for {model.key}: {last_err}")
