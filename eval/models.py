"""Judge-model registry.

The suite is chosen to vary three things at once, cheaply:

  * provider          OpenAI vs OpenRouter  -> genuine cross-vendor consistency check
  * capability tier   nano/mini, small/large MoE -> "does a stronger judge agree?"
  * modality access   text / +image / +audio -> the axis the take-home calls out

Prices are USD per *token* (not per million) so CostTracker can multiply directly.
They are rough catalogue figures (June 2026); update if billing changes.

Modalities the harness understands:
  "text"  - always; reads extracted text / transcripts
  "image" - can see rendered PDF pages, photos, and sampled video frames
  "audio" - can natively listen to an attached audio clip (vs. reading a transcript)

A model that lacks "audio" still gets at audio *content* through the
transcribe_audio tool (Whisper). That degraded path is exactly what makes the
modality axis measurable: native-audio judges vs. transcript-only judges scoring
the same deliverable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set

from . import config


@dataclass(frozen=True)
class Model:
    key: str                      # short handle used on the CLI / in result files
    provider: str                 # "openai" | "openrouter"
    model_id: str                 # provider-native id
    modalities: Set[str]
    tier: str                     # human label: strong / mid / weak / free
    in_price: float = 0.0         # $ per prompt token
    out_price: float = 0.0        # $ per completion token
    reasoning_effort: str = ""    # "minimal"|"low"|"medium"|"high" for GPT-5.x thinking models

    @property
    def is_reasoning(self) -> bool:
        # GPT-5.x reasoning models ignore temperature and want max_completion_tokens.
        return self.provider == "openai" and self.model_id.startswith("gpt-5")

    def supports(self, modality: str) -> bool:
        return modality in self.modalities


# ---------------------------------------------------------------------------
# The suite. Keep it small: every model here earns its place on an axis.
# ---------------------------------------------------------------------------
REGISTRY = {
    # --- OpenAI vision-tier, two capability rungs (cheap capability ablation) ---
    "gpt-mini": Model(
        "gpt-mini", "openai", "gpt-5.4-mini", {"text", "image"}, "strong",
        in_price=0.25e-6, out_price=2.0e-6,
    ),
    "gpt-nano": Model(
        "gpt-nano", "openai", "gpt-5.4-nano", {"text", "image"}, "weak",
        in_price=0.05e-6, out_price=0.40e-6,
    ),
    # Frontier reasoning model in low-thinking mode — used to DRIVE the adversarial
    # agent (it must rebuild a rubric-satisfying surface from scratch, which the mini
    # tier can't). Prices are rough June-2026 catalogue estimates.
    "gpt55": Model(
        "gpt55", "openai", "gpt-5.5", {"text", "image"}, "strong",
        in_price=1.25e-6, out_price=10.0e-6, reasoning_effort="low",
    ),

    # --- Qwen, same family, with vs. without vision: the modality ablation pair ---
    "qwen-vl": Model(
        "qwen-vl", "openrouter", "qwen/qwen3.5-35b-a3b", {"text", "image"}, "mid",
        in_price=0.14e-6, out_price=0.55e-6,
    ),
    "qwen-text": Model(
        "qwen-text", "openrouter", "qwen/qwen3-30b-a3b-instruct-2507", {"text"}, "mid",
        in_price=0.05e-6, out_price=0.20e-6,
    ),

    # --- Full-modality judges: the only ones that natively HEAR audio ---
    "gemini-lite": Model(
        "gemini-lite", "openrouter", "google/gemini-2.5-flash-lite",
        {"text", "image", "audio"}, "mid",
        in_price=0.10e-6, out_price=0.40e-6,
    ),
    "nemotron-omni": Model(   # free; second native-audio judge for cross-model on A/V tasks
        "nemotron-omni", "openrouter", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        {"text", "image", "audio"}, "free",
        in_price=0.0, out_price=0.0,
    ),

    # --- rollout policies: a small panel of recent open models to attempt the tasks ---
    # (also usable as judges). Modalities map OpenRouter's image/audio to ours; video
    # is reached via frame sampling, not a native "video" modality.
    "gemma4": Model(
        "gemma4", "openrouter", "google/gemma-4-26b-a4b-it", {"text", "image"}, "mid",
        in_price=0.06e-6, out_price=0.33e-6,
    ),
    "qwen36": Model(
        "qwen36", "openrouter", "qwen/qwen3.6-35b-a3b", {"text", "image"}, "mid",
        in_price=0.14e-6, out_price=1.0e-6,
    ),
    "mimo": Model(   # natively hears audio -> strong policy for the A/V edit tasks
        "mimo", "openrouter", "xiaomi/mimo-v2.5", {"text", "image", "audio"}, "mid",
        in_price=0.105e-6, out_price=0.28e-6,
    ),
}

# Sensible defaults if the caller doesn't pass --models.
# Cross-model panel: one per provider/modality corner, all cheap.
DEFAULT_CROSS_MODEL = ["gpt-mini", "qwen-vl", "gemini-lite"]
# Self-consistency: cheapest capable judge, repeated.
DEFAULT_SELF = "qwen-vl"
# Modality-ablation triple, for the "access to modalities" experiment.
MODALITY_LADDER = ["qwen-text", "qwen-vl", "gemini-lite"]


def resolve(keys) -> list[Model]:
    out = []
    for k in keys:
        if k not in REGISTRY:
            raise SystemExit(f"unknown model '{k}'. known: {', '.join(REGISTRY)}")
        out.append(REGISTRY[k])
    return out
