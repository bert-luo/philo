"""The rubric iterator (Part 3): a loop that improves a rubric automatically.

For one task it runs:

    evaluate  -> grade gold + nulls with the judge matrix and read off axis signals
    diagnose  -> turn those signals into per-item DEFECTS, each tagged with the axis
                 it fails (boundary leak / inconsistency / unverifiable criterion)
    rewrite   -> ask a model to rewrite ONLY the flagged criteria to close that gap
    evaluate  -> grade the rewritten rubric and report what moved

v0 scope (deliberately small):
  * Axes wired in: Axis 1 (boundary calibration), Axis 2 (reward-hackability, via the
    mechanical attack suite in attacks.py — a positive item a cheap attack games is a
    rewrite target, and the leaking attack names the hack pattern), Axis 3
    (consistency), and verifiability of criteria.
  * Rewrites edit criterion TEXT in place; item ids and point values are held
    fixed so every version is graded on the same scale and before/after is a
    clean comparison. (A richer iterator could split or add items — left for v1.)

Each task gets a self-contained package under eval/var/rubric_packages/<task>/ with
the original rubric, every rewritten version, the test cases (subjects) used, and
the per-version diagnosis. A single tsv (results/rubric_scorecard.tsv) holds one
row per (task x rubric version) with a column per axis.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import config, llm, run_common
from .config import CostTracker
from .harness import judge_safe as judge
from .models import Model, resolve
from .rubric import RubricItem

PACKAGES = config.VAR / "rubric_packages"

# How each attack, when it leaks an item, characterizes the hack pattern at fault.
_HACK_KIND = {
    "restate_input": "the fact is already in the brief, so restating an input earns "
                     "the point (brief-echo, not work done)",
    "keyword_stuff": "a document that merely transcribes the rubric's own wording earns "
                     "the point (rubric-surface match / self-certification)",
    "presence_max": "an oversized title/caption covering the subject earns the point "
                    "(presence with no matching absence condition)",
    "stems_named": "a correctly-named but silent/empty stem earns the point "
                   "(presence-only, no quality check)",
}

# --- diagnosis thresholds ---------------------------------------------------
# A criterion is "contested" when judges disagree this much on its [0,1] fraction.
CONTESTED_STDEV = 0.30
# Flag a criterion unverifiable when at least this fraction of judges said so on gold.
UNVERIFIABLE_FRAC = 0.50
# Don't bother rewriting more than this many items per round (focus the model).
MAX_REWRITES_PER_ROUND = 8
# Loop objective = (gold - worst_attack) margin, penalized for self-inconsistency.
# Cross-model agreement is NOT in the objective (it doesn't drive RL the way a single
# judge's wander does). A round is accepted only if it raises this objective, so a
# "hardening" rewrite that drops gold below an attack (a backfire) is rejected.
SELF_PENALTY = 0.5


# --- raw rubric <-> items (preserve ids/scores; only criterion text changes) -
def load_raw_rubric(task_folder: str) -> list[dict]:
    return json.loads((config.DATA / task_folder / "rubric.json").read_text())


def raw_to_items(raw: list[dict]) -> list[RubricItem]:
    return [RubricItem(id=r["rubric_item_id"], criterion=r["criterion"].strip(),
                       max_score=float(r["score"])) for r in raw]


def save_rubric(raw: list[dict], path) -> None:
    path.write_text(json.dumps(raw, indent=2))


# --- evaluate: grade a rubric version and read off the axis signals ----------
def evaluate_rubric(task_folder: str, items: list[RubricItem], judges: list[Model],
                    self_model: Model | None, repeats: int, temperature: float,
                    subjects: list[str] | None = None, verbose: bool = False,
                    protocol: str = "") -> dict:
    """Run the judge matrix for one (task, rubric version) and derive Axis 1 / 2 / 3
    / verifiability. `protocol` is the version's grading protocol (part of the reward
    function), applied by the judge to every subject. Returns a dict `diagnose` consumes."""
    task = run_common.load_task(task_folder)
    prompt = task["prompt"]
    # Default subjects: gold + nulls (Axis 1) + mechanical attacks (Axis 2).
    subjects = subjects or (run_common.available_subjects(task_folder)
                            + run_common.attack_subjects(task_folder))

    # matrix[subject][judge_key] = base run (T=0); plus self-repeats for one model.
    base: dict[str, dict[str, dict]] = {}
    self_runs: dict[str, list[dict]] = {}
    for subj in subjects:
        try:
            bundle = run_common.resolve_subject(task_folder, subj)
        except SystemExit as e:
            print(f"    skip subject {subj}: {e}"); continue
        base[subj] = {}
        for m in judges:
            base[subj][m.key] = judge(m, bundle, items, prompt,
                                      temperature=0.0, verbose=verbose, protocol=protocol)
        if repeats > 1 and self_model is not None:
            self_runs[subj] = [judge(self_model, bundle, items, prompt,
                                     temperature=temperature, verbose=verbose, protocol=protocol)
                               for _ in range(repeats)]

    nulls = [s for s in base if run_common.is_null_subject(task_folder, s)]
    attack_subs = [s for s in base if run_common.is_attack_subject(task_folder, s)]
    ev: dict = {"subjects": list(base), "nulls": nulls, "attacks": attack_subs,
                "base": base, "self_runs": self_runs}

    # Gold score, available whenever gold is graded (the boundary block below adds
    # the null comparison; hackability only needs gold to show reward stays high).
    if "gold" in base:
        gruns = list(base["gold"].values())
        ev["gold_score"] = round(sum(r["normalized"] for r in gruns) / len(gruns), 3)

    # ---- Axis 2: hackability (mechanical attacks game positive items) ----
    if attack_subs:
        all_attack_runs = [base[s][m.key] for s in attack_subs for m in judges]
        ev["hackability"] = {
            "attacks": attack_subs,
            # union leak across all attacks/judges: any item a cheap attack passes.
            "leaks": run_common.leaks(items, all_attack_runs),
            # per-attack leaks, to attribute WHICH hack pattern each leak is.
            "by_attack": {s: run_common.leaks(items, list(base[s].values()))
                          for s in attack_subs},
            "attack_scores": {s: round(sum(r["normalized"] for r in base[s].values())
                                       / len(base[s]), 3) for s in attack_subs},
        }

    # ---- Axis 1: boundary calibration (needs gold + a null) ----
    if "gold" in base and nulls:
        gold_runs = list(base["gold"].values())
        all_null_runs = [base[s][m.key] for s in nulls for m in judges]
        gold_score = sum(r["normalized"] for r in gold_runs) / len(gold_runs)
        gold_verif = sum(r.get("normalized_verifiable_only", r["normalized"])
                         for r in gold_runs) / len(gold_runs)
        worst_null = max((r["normalized"] for r in all_null_runs), default=0.0)
        ev["boundary"] = {
            "gold_score": round(gold_score, 3),
            "gold_verifiable_only": round(gold_verif, 3),
            "worst_null": round(worst_null, 3),
            "gap": round(gold_score - worst_null, 3),
            # leaks pooled across ALL judges' null runs: any judge that lets a
            # null pass a positive item is evidence of a presence/form reward.
            "leaks": run_common.leaks(items, all_null_runs),
        }

    # ---- Axis 3: cross-model + self-consistency (per subject) ----
    if len(judges) > 1:
        ev["cross_model"] = {s: run_common.agreement(items, list(base[s].values()))
                             for s in base}
    if self_runs:
        ev["self_consistency"] = {s: run_common.agreement(items, runs)
                                  for s, runs in self_runs.items()}

    # ---- verifiability: how many criteria can no judge check on the gold ----
    if "gold" in base:
        gold_runs = list(base["gold"].values())
        unv_counts: dict[str, int] = {}
        for r in gold_runs:
            for iid in r.get("unverifiable", []):
                unv_counts[iid] = unv_counts.get(iid, 0) + 1
        ev["verifiability"] = {
            "frac_unverifiable_by_item": {k: v / len(gold_runs)
                                          for k, v in unv_counts.items()},
            # gap between the headline score and the score over checkable items
            # only: large => the rubric asks for things no judge can verify.
            "gap": round(ev.get("boundary", {}).get("gold_verifiable_only", 0.0)
                         - ev.get("boundary", {}).get("gold_score", 0.0), 3),
        }
    return ev


# --- diagnose: axis signals -> per-item defects ------------------------------
@dataclass
class Defect:
    item_id: str
    criterion: str
    axis: str          # "boundary" | "consistency" | "verifiability"
    kind: str          # short machine label
    detail: str        # human explanation handed to the rewriter
    severity: float    # points at stake x signal strength, for ranking

    def as_dict(self) -> dict:
        return {"item_id": self.item_id, "axis": self.axis, "kind": self.kind,
                "detail": self.detail, "severity": round(self.severity, 2),
                "criterion": self.criterion}


def diagnose(items: list[RubricItem], ev: dict) -> list[Defect]:
    """Turn one evaluation into a ranked list of per-item defects to fix."""
    by_id = {it.id: it for it in items}
    pts = {it.id: abs(it.max_score) for it in items}
    defects: dict[str, Defect] = {}   # item_id -> highest-severity defect

    def add(d: Defect):
        if d.item_id not in defects or d.severity > defects[d.item_id].severity:
            defects[d.item_id] = d

    # Axis 2: attack leaks = positive items a cheap mechanical attack games. The
    # leaking attack names the hack pattern, so the rewriter gets specific guidance.
    hack = ev.get("hackability", {})
    by_attack = hack.get("by_attack", {})
    attack_flagged: set[str] = set()
    for lk in hack.get("leaks", []):
        it = by_id.get(lk["id"])
        if not it:
            continue
        # which attack leaks this item hardest -> which hack pattern to cite
        culprit = max((s for s in by_attack
                       if any(x["id"] == it.id for x in by_attack[s])),
                      key=lambda s: next(x["null_fraction"] for x in by_attack[s]
                                         if x["id"] == it.id),
                      default=hack.get("attacks", ["an attack"])[0])
        add(Defect(it.id, it.criterion, "hackability", "attack_leak",
                   f"The '{culprit}' attack earns {lk['null_fraction']:.0%} of this "
                   f"item's points: {_HACK_KIND.get(culprit, 'cheap output games it')}. "
                   f"Close it — pair the presence condition with the missing absence/"
                   f"quality condition, or require something the attack cannot fake.",
                   severity=pts[it.id] * lk["null_fraction"]))
        attack_flagged.add(it.id)

    # Axis 1: boundary leaks = positive items a do-nothing null passes. Skip items
    # already flagged by an attack (the attack signal is stronger and more specific).
    for lk in ev.get("boundary", {}).get("leaks", []):
        it = by_id.get(lk["id"])
        if not it or it.id in attack_flagged:
            continue
        add(Defect(it.id, it.criterion, "boundary", "null_leak",
                   f"A do-nothing null still earns {lk['null_fraction']:.0%} of this "
                   f"item's points. It rewards mere presence/form, not the work — a "
                   f"reward-hacking surface. Add the missing absence/quality condition.",
                   severity=pts[it.id] * lk["null_fraction"]))

    # Verifiability: judges couldn't check the criterion on the gold deliverable.
    for iid, frac in ev.get("verifiability", {}).get("frac_unverifiable_by_item", {}).items():
        it = by_id.get(iid)
        if not it or frac < UNVERIFIABLE_FRAC:
            continue
        add(Defect(it.id, it.criterion, "verifiability", "unverifiable",
                   f"{frac:.0%} of judges could not verify this from the deliverable "
                   f"or any tool. Rewrite it to be checkable from the artifact itself, "
                   f"or replace the external dependency with an in-deliverable proxy.",
                   severity=pts[it.id] * frac))

    # Axis 3: criteria the SAME judge scores inconsistently across repeats on the gold.
    # Self-consistency is the RL-relevant signal — the policy optimizes against one
    # judge, so what breaks training is that judge wandering, not two judges differing.
    # (Cross-model stdev is kept in the scorecard as a diagnostic but does NOT drive
    # rewrites.)
    sc = ev.get("self_consistency", {}).get("gold", {})
    for p in sc.get("per_item", []):
        if p["stdev"] < CONTESTED_STDEV:
            continue
        it = by_id.get(p["id"])
        if not it:
            continue
        add(Defect(it.id, it.criterion, "consistency", "contested",
                   f"The same judge scores this item inconsistently across repeats "
                   f"(stdev {p['stdev']:.2f} on the gold) — an unstable reward. The "
                   f"wording leaves room for the judge to wander; pin down an explicit, "
                   f"objective test (a threshold, a concrete definition) so one judge "
                   f"reads it the same every time.",
                   severity=pts[it.id] * p["stdev"]))

    return sorted(defects.values(), key=lambda d: -d.severity)


# --- rewrite: ask a model to fix the flagged criteria ------------------------
REWRITE_SYSTEM = """You repair grading rubrics used as RL reward signals. A rubric here has
TWO parts you can edit, and they fix different kinds of defect:

  1. CRITERIA — the per-item checks (you edit their text).
  2. GRADING PROTOCOL — task-specific verification rules the judge applies to EVERY item
     (how rigorously to check a claim). This is part of the reward function too.

You are given the current rubric, the current protocol, and a list of DEFECTS. Choose the
RIGHT lever for each defect:

- hackability/attack_leak: a cheap attack already earns the item's points without the work
  (the detail names which attack). Pick the lever by attack:
  * brief-echo (restate_input): the fact is already in the inputs. EDIT THE CRITERION to
    require synthesis the inputs don't pre-package (correct ordering, cross-referencing,
    judgement that follows from observations), not a fact that can be copied.
  * transcribe-the-rubric (keyword_stuff): NO criterion edit can fix this — the attack just
    copies whatever the criterion says, and a judge that accepts a present assertion passes
    it. This is a PROTOCOL fix: add a rule that the judge must VERIFY the asserted fact
    against the reference inputs / deliverable evidence and must NOT credit a bare assertion
    or restated criterion text. The protocol is the only lever that reaches this hack.
  * padding/presence (presence_max, null_leak): EDIT THE CRITERION to pair the presence
    condition with the matching absence condition, e.g. "the caption is visible" ->
    "the caption is visible AND does not cover the speaker".
- verifiability/unverifiable: rephrase the criterion to be checkable from the deliverable
  itself, or add a protocol rule pointing the judge at the evidence that settles it.
- consistency/contested: the SAME judge wanders on this item. Remove ambiguity with an
  explicit, objective test (a number, a threshold, a concrete definition) so one judge
  reads it identically every time.

Hard rules:
- Keep each item's INTENT and point value. Make it a tighter test of the SAME thing.
- The GRADING PROTOCOL may only make the judge MORE skeptical / more verifying. Never write
  a protocol rule that lets the judge award points more easily — that would just inflate
  every output (gold AND the attacks) and is rejected by the objective.
- Do not weaken a criterion to make gold pass; keep it strict.
- Use the protocol for cross-cutting hacks (claims that should be verified everywhere);
  use criterion edits for item-specific problems.
- Return the rewritten criteria (rubric_item_id unchanged) AND, if you changed it, the FULL
  revised grading protocol text (not a diff)."""


def _rewrite_schema() -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": "propose_rewrites",
            "description": "Submit the rewritten criteria for the flagged rubric items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rewrites": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "rubric_item_id": {"type": "string"},
                                "new_criterion": {"type": "string",
                                                  "description": "The rewritten criterion text."},
                                "rationale": {"type": "string",
                                              "description": "How this closes the defect."},
                            },
                            "required": ["rubric_item_id", "new_criterion"],
                        },
                    },
                    "grading_protocol": {
                        "type": "string",
                        "description": "The FULL revised grading protocol (task-specific "
                                       "verification rules the judge applies to every item). "
                                       "Return the complete text, not a diff. Omit or leave "
                                       "empty to keep the current protocol unchanged.",
                    },
                },
                "required": ["rewrites"],
            },
        },
    }]


def rewrite(task_folder: str, raw: list[dict], defects: list[Defect], model: Model,
            protocol: str = "", verbose: bool = False) -> tuple[list[dict], str, list[dict]]:
    """Rewrite flagged criteria and/or the grading protocol. Returns
    (new_raw, new_protocol, changelog).

    new_raw is a deep copy of `raw` with criterion text replaced for changed items;
    ids and scores are untouched so the next evaluation is a clean comparison. The
    protocol is the other lever — the only one that reaches transcribe-the-rubric hacks."""
    targets = defects[:MAX_REWRITES_PER_ROUND]
    if not targets:
        return [dict(r) for r in raw], protocol, []
    task = run_common.load_task(task_folder)
    by_id = {r["rubric_item_id"]: r for r in raw}
    defect_block = "\n".join(
        f"[{d.item_id}] ({d.axis}/{d.kind}) current: {d.criterion}\n   defect: {d.detail}"
        for d in targets)
    user = (
        f"TASK THE RUBRIC GRADES:\n{task['prompt'].strip()[:1500]}\n\n"
        f"CURRENT GRADING PROTOCOL:\n{protocol.strip() or '(none yet)'}\n\n"
        f"DEFECTS TO FIX ({len(targets)} items):\n{defect_block}\n\n"
        f"Fix each defect with the right lever (criterion edit and/or protocol rule). "
        f"Call propose_rewrites.")
    msg = llm.chat(
        model,
        [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": user}],
        tools=_rewrite_schema(),
        tool_choice={"type": "function", "function": {"name": "propose_rewrites"}},
        temperature=0.3, max_tokens=4096,
    )
    _, parsed = llm.assistant_tool_calls(msg.tool_calls or [])
    proposals, new_protocol = [], protocol
    for args in parsed.values():
        proposals.extend(args.get("rewrites", []))
        gp = (args.get("grading_protocol") or "").strip()
        if gp:
            new_protocol = gp

    new_raw = [dict(r) for r in raw]
    new_by_id = {r["rubric_item_id"]: r for r in new_raw}
    changelog = []
    for p in proposals:
        iid = str(p.get("rubric_item_id", "")).strip().strip("[]").strip()
        new_text = (p.get("new_criterion") or "").strip()
        if iid not in new_by_id or not new_text:
            continue
        old_text = by_id[iid]["criterion"].strip()
        if new_text == old_text:
            continue
        new_by_id[iid]["criterion"] = new_text
        changelog.append({"rubric_item_id": iid, "old": old_text, "new": new_text,
                          "rationale": p.get("rationale", "")})
        if verbose:
            print(f"    rewrote [{iid[:8]}]: {old_text[:50]!r} -> {new_text[:50]!r}")
    if new_protocol.strip() != protocol.strip():
        changelog.append({"rubric_item_id": "__protocol__", "old": protocol,
                          "new": new_protocol, "rationale": "grading protocol updated"})
        if verbose:
            print(f"    updated grading protocol ({len(new_protocol)} chars)")
    return new_raw, new_protocol, changelog


# --- the loop ---------------------------------------------------------------
def _headline(ev: dict, defects: list[Defect]) -> dict:
    """Flat one-row-per-version summary for the scorecard tsv / console."""
    b = ev.get("boundary", {})
    hk = ev.get("hackability", {})
    cm = ev.get("cross_model", {}).get("gold", {})
    sc = ev.get("self_consistency", {}).get("gold", {})
    n_unv = sum(1 for f in ev.get("verifiability", {})
                .get("frac_unverifiable_by_item", {}).values() if f >= UNVERIFIABLE_FRAC)
    # Headline hackability number: the best score any cheap attack reaches. A high
    # value next to a high gold means most of the reward is buyable without the work.
    worst_attack = max(hk.get("attack_scores", {}).values(), default="")
    gold = b.get("gold_score", ev.get("gold_score", ""))
    self_std = sc.get("total_stdev", "")
    # The loop objective: separation of real work from the best attack, penalized for
    # the judge's own wander. "" when gold/attacks weren't both graded this version.
    margin = objective = ""
    if isinstance(gold, (int, float)) and isinstance(worst_attack, (int, float)):
        margin = round(gold - worst_attack, 3)
        objective = round(margin - SELF_PENALTY * (self_std if isinstance(self_std, (int, float)) else 0.0), 3)
    return {
        "gold_score": gold,
        "null_score": b.get("worst_null", ""),
        "gold_gap": b.get("gap", ""),
        "n_leaks": len(b.get("leaks", [])),
        "n_attack_leaks": len(hk.get("leaks", [])),
        "worst_attack_score": worst_attack,
        "margin": margin,
        "objective": objective,
        "self_consistency_stdev": self_std,
        "cross_model_stdev": cm.get("total_stdev", ""),  # diagnostic only, not optimized
        "n_unverifiable": n_unv,
        "verifiability_gap": ev.get("verifiability", {}).get("gap", ""),
        "n_defects": len(defects),
        "reward_hack_flag": "attack-leaks" if hk.get("leaks") else
                            ("null-leaks" if b.get("leaks") else
                             ("null-high" if (b.get("worst_null") or 0) > run_common.NULL_HACK_SCORE else "")),
    }


def iterate_task(task_folder: str, rounds: int, judges: list[Model],
                 self_model: Model | None, repeats: int, temperature: float,
                 rewrite_model: Model, subjects: list[str] | None,
                 verbose: bool = False) -> list[list]:
    """Run the improve loop for one task. Writes its package; returns scorecard rows."""
    print(f"\n=== {task_folder} ===")
    pkg = PACKAGES / task_folder
    pkg.mkdir(parents=True, exist_ok=True)

    raw = load_raw_rubric(task_folder)
    protocol = ""                          # v0 = the original reward function, no protocol
    rows: list[list] = []
    versions_meta = []
    # Best accepted version by the objective (gold-attack margin, self-consistency-penalized).
    best = {"version": 0, "objective": None, "raw": raw, "protocol": protocol}

    v = 0
    while True:
        items = raw_to_items(raw)
        save_rubric(raw, pkg / f"rubric_v{v}.json")
        (pkg / f"protocol_v{v}.txt").write_text(protocol)
        print(f"  -- v{v}: evaluating {len(items)} items "
              f"({len(judges)} judge(s), repeats={repeats}, protocol={'yes' if protocol.strip() else 'none'})")
        ev = evaluate_rubric(task_folder, items, judges, self_model, repeats,
                             temperature, subjects, verbose, protocol=protocol)
        defects = diagnose(items, ev)
        head = _headline(ev, defects)
        (pkg / f"diagnosis_v{v}.json").write_text(json.dumps(
            {"headline": head, "defects": [d.as_dict() for d in defects],
             "boundary": ev.get("boundary"), "hackability": ev.get("hackability"),
             "verifiability": ev.get("verifiability"),
             "self_consistency_gold": ev.get("self_consistency", {}).get("gold", {}).get("most_contested"),
             "cross_model_gold": ev.get("cross_model", {}).get("gold", {}).get("most_contested"),
             }, indent=2))
        print(f"     gold={head['gold_score']} worst_attack={head['worst_attack_score']} "
              f"margin={head['margin']} objective={head['objective']} "
              f"attack_leaks={head['n_attack_leaks']} self_stdev={head['self_consistency_stdev']} "
              f"-> {len(defects)} defects")
        rows.append([task_folder, f"v{v}"] + [head[k] for k in HEAD_COLS])
        versions_meta.append({"version": v, "headline": head, "n_defects": len(defects),
                              "top_defects": [d.as_dict() for d in defects[:5]]})

        # --- accept / reject this version against the best so far (hill-climb) ---
        obj = head["objective"]
        if v == 0:
            best = {"version": 0, "objective": obj, "raw": raw, "protocol": protocol}
        elif isinstance(obj, (int, float)) and isinstance(best["objective"], (int, float)) \
                and obj <= best["objective"]:
            print(f"     REJECT v{v}: objective {obj} <= best v{best['version']} "
                  f"({best['objective']}) — this rewrite backfired; reverting and stopping.")
            raw, protocol = best["raw"], best["protocol"]
            break
        else:
            best = {"version": v, "objective": obj, "raw": raw, "protocol": protocol}

        if v == rounds or not defects or not run_common.has_gold(task_folder):
            if not defects:
                print(f"     no defects left — stopping at v{v}")
            elif not run_common.has_gold(task_folder):
                print("     no gold deliverable — stopping (can't measure iteration improvement)")
            break

        new_raw, new_protocol, changelog = rewrite(
            task_folder, raw, defects, rewrite_model, protocol, verbose)
        (pkg / f"changelog_v{v}_to_v{v+1}.json").write_text(json.dumps(changelog, indent=2))
        crit_changes = sum(1 for c in changelog if c["rubric_item_id"] != "__protocol__")
        proto_changed = any(c["rubric_item_id"] == "__protocol__" for c in changelog)
        print(f"     rewrote {crit_changes} criteria"
              f"{' + grading protocol' if proto_changed else ''} -> v{v+1}")
        if not changelog:
            print("     rewriter changed nothing — stopping"); break
        raw, protocol = new_raw, new_protocol
        v += 1

    # The chosen reward function = the best accepted version's rubric + protocol.
    save_rubric(best["raw"], pkg / "rubric_best.json")
    (pkg / "protocol_best.txt").write_text(best["protocol"])

    # package manifest: before/after at a glance
    (pkg / "report.json").write_text(json.dumps({
        "task": task_folder,
        "subjects_tested": subjects or (run_common.available_subjects(task_folder)
                                        + run_common.attack_subjects(task_folder)),
        "judges": [m.key for m in judges],
        "rounds_run": len(versions_meta) - 1,
        "best_version": best["version"],
        "best_objective": best["objective"],
        "versions": versions_meta,
    }, indent=2))
    print(f"  wrote package {pkg.relative_to(config.ROOT)}/ "
          f"({len(versions_meta)} versions; best=v{best['version']})")
    return rows


HEAD_COLS = ["gold_score", "null_score", "gold_gap", "n_leaks",
             "n_attack_leaks", "worst_attack_score", "margin", "objective",
             "self_consistency_stdev", "cross_model_stdev", "n_unverifiable",
             "verifiability_gap", "n_defects", "reward_hack_flag"]


def run_iterate(tasks: list[str], rounds: int, judge_keys: list[str],
                self_key: str | None, repeats: int, temperature: float,
                rewrite_key: str, subjects: list[str] | None, verbose: bool) -> None:
    judges = resolve(judge_keys)
    self_model = resolve([self_key])[0] if (self_key and repeats > 1) else \
                 (judges[0] if repeats > 1 else None)
    rewrite_model = resolve([rewrite_key])[0]
    all_rows: list[list] = []
    for tf in tasks:
        try:
            all_rows += iterate_task(tf, rounds, judges, self_model, repeats,
                                     temperature, rewrite_model, subjects, verbose)
        except Exception as e:   # one task must not abort the batch
            print(f"  {tf}: ERROR -> {str(e)[:160]}")
    tsv_path = config.RESULTS / "rubric_scorecard.tsv"
    if len(tasks) == 1:
        tsv_path = config.RESULTS / f"rubric_scorecard.{tasks[0]}.tsv"
    run_common.write_tsv(tsv_path, ["task", "rubric_version"] + HEAD_COLS, all_rows)
    print(CostTracker.summary())
