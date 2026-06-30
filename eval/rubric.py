"""Rubric loading and scorecard aggregation.

A rubric is a list of items, each worth `score` points for a binary/quality
criterion. The judge awards each item a fraction in [0,1] of its points; the
normalized rubric score is (sum of awarded points) / (sum of max points).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class RubricItem:
    id: str
    criterion: str
    max_score: float


def load_rubric(task_folder: str) -> list[RubricItem]:
    raw = json.loads((config.DATA / task_folder / "rubric.json").read_text())
    items = []
    for r in raw:
        items.append(RubricItem(
            id=r["rubric_item_id"],
            criterion=r["criterion"].strip(),
            max_score=float(r["score"]),
        ))
    return items


def positive_max(items: list[RubricItem]) -> float:
    """Best achievable score: all positive items earned, no penalties triggered."""
    return sum(i.max_score for i in items if i.max_score > 0)


def normalize(items: list[RubricItem], awarded: dict[str, float],
              unverifiable: set[str] | None = None) -> dict:
    """awarded: item_id -> fraction in [0,1] of the *extent the criterion is true*.

    For positive items, fraction is credit earned. For penalty items (negative
    points), fraction is how strongly the (bad) condition holds, so a triggered
    penalty subtracts points. The deliverable's score is divided by the positive
    ceiling and clamped to [0,1]; a deliverable that trips heavy penalties floors
    at 0 rather than going negative.

    `unverifiable` items (the judge couldn't check them at all) still count as 0
    in `normalized`, but are removed from BOTH numerator and denominator in
    `normalized_verifiable_only` — the fair score given what was actually checkable.
    A large gap between the two means the rubric asks for things no judge can
    verify from the deliverable (a coverage/verifiability defect).
    """
    unverifiable = unverifiable or set()
    ceiling = positive_max(items)
    ceiling_verif = sum(i.max_score for i in items
                        if i.max_score > 0 and i.id not in unverifiable)
    per_item = {}
    got = 0.0
    got_verif = 0.0
    for it in items:
        frac = max(0.0, min(1.0, float(awarded.get(it.id, 0.0))))
        pts = frac * it.max_score          # negative for penalty items
        unv = it.id in unverifiable
        per_item[it.id] = {"fraction": frac, "points": pts, "max": it.max_score,
                           "penalty": it.max_score < 0, "unverifiable": unv}
        got += pts
        if not unv:
            got_verif += pts
    norm = got / ceiling if ceiling else 0.0
    norm_verif = got_verif / ceiling_verif if ceiling_verif else 0.0
    return {
        "normalized": max(0.0, min(1.0, norm)),
        "normalized_verifiable_only": max(0.0, min(1.0, norm_verif)),
        "raw_normalized": norm,            # keep the unclamped value for diagnostics
        "points": got,
        "ceiling": ceiling,
        "n_unverifiable": len(unverifiable),
        "per_item": per_item,
    }
