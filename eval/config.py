"""Shared paths, env loading, and cost tracking for the rubric-grader eval suite."""
from __future__ import annotations

import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VAR = ROOT / "eval" / "var"  # generated/runtime data, gitignored
RESULTS = VAR / "results"
NULLS = VAR / "null_inputs"
ATTACKS = VAR / "attacks"  # mechanically-generated reward-hacking deliverables
CACHE = VAR / ".cache"  # transcripts, extracted frames, etc.

for _d in (RESULTS, NULLS, ATTACKS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)


def _load_env() -> None:
    """Minimal .env loader (avoids a python-dotenv dependency)."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


class CostTracker:
    """Cheap, approximate spend tracker so a run never silently burns credits."""

    _lock = threading.Lock()
    prompt_tokens = 0
    completion_tokens = 0
    usd = 0.0
    calls = 0

    @classmethod
    def add(cls, prompt_tokens: int, completion_tokens: int, in_price: float, out_price: float) -> None:
        with cls._lock:
            cls.prompt_tokens += prompt_tokens
            cls.completion_tokens += completion_tokens
            cls.usd += prompt_tokens * in_price + completion_tokens * out_price
            cls.calls += 1

    @classmethod
    def summary(cls) -> str:
        return (
            f"[cost] calls={cls.calls} "
            f"prompt={cls.prompt_tokens:,} completion={cls.completion_tokens:,} "
            f"~${cls.usd:.4f}"
        )
