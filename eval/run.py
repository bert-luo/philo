"""Unified eval runner — one entrypoint for the rubric scorecard.

Two modes, selected with --mode:

  rollout   Have policy model(s) attempt the real GDPval tasks and produce actual
            deliverables (eval.policy). These become gradeable subjects.

  judge     Grade one or more SUBJECTS per task against the rubric, then derive the
            axes from a single judging matrix:
              * Boundary calibration (Axis 1): gold should score ~1, nulls ~0; we
                report the gold-null gap and "leaks" (null passing a positive item).
              * Cross-model consistency (Axis 3): when >=2 --judge-models grade the
                same subject, do they agree?
              * Self-consistency (Axis 3): when --repeats > 1, one judge re-grades the
                same subject; do the scores reproduce?

A subject is named on the CLI: `gold`, a null variant (`blank`, `unedited_input`),
`rollout:<policy_key>`, or `preexisting:<path>`. Default subjects are gold + nulls
(the boundary anchors).

Examples:
  # boundary + cross-model in one shot (3 judges, all gold tasks):
  python -m eval.run --mode judge --judge-models gpt-mini qwen-vl gemini-lite

  # add self-consistency (one judge, 5 repeats) on a single task:
  python -m eval.run --mode judge --tasks private_detectives_and_investigators__57b2cdf2 \
      --judge-models qwen-vl --repeats 5 --temperature 0.7

  # consistency OFF (single judge, single pass) on a rollout you generated:
  python -m eval.run --mode judge --subjects rollout:gpt-mini --judge-models gpt-mini

  # generate rollouts first:
  python -m eval.run --mode rollout --policy-models gpt-mini --tasks <folder>
"""
from __future__ import annotations

import argparse
import itertools
import json

from . import config, policy, run_common
from .config import CostTracker
from .harness import judge_safe as judge  # never let one flaky judgement abort the sweep
from .models import resolve
from .rubric import RubricItem, load_rubric
from .run_common import NULL_HACK_SCORE, agreement as _agreement, leaks as _leaks


# --- judge mode -------------------------------------------------------------
def run_judge(tasks, subjects_arg, judge_keys, repeats, self_key, temperature, verbose):
    judges = resolve(judge_keys)
    self_model = resolve([self_key])[0] if self_key else judges[0]
    report: dict = {}
    score_rows, cons_rows = [], []

    for tf in tasks:
        subjects = subjects_arg or run_common.available_subjects(tf)
        if not subjects:
            print(f"skip {tf}: no subjects (no gold/nulls; pass --subjects)"); continue
        print(f"\n=== {tf} ===  subjects={subjects}")
        task = run_common.load_task(tf)
        items = load_rubric(tf)
        report[tf] = {"subjects": {}}

        # matrix[subject][judge_key] = base run (T=0); matrix[subject]['_self'] = repeats
        matrix: dict[str, dict] = {}
        for subj in subjects:
            try:
                bundle = run_common.resolve_subject(tf, subj)
            except SystemExit as e:
                print(f"  skip subject {subj}: {e}"); continue
            print(f"  -- subject '{subj}' ({len(bundle.files)} files)")
            base = {}
            for m in judges:
                r = judge(m, bundle, items, task["prompt"], temperature=0.0, verbose=verbose)
                base[m.key] = r
                vo = r.get("normalized_verifiable_only", r["normalized"])
                flags = ""
                if not r.get("valid", True):
                    flags += f" INVALID(scored {r.get('n_items_scored', 0)}/{len(items)})"
                if r.get("penalty_floored"):
                    flags += " PENALTY-FLOORED"
                if r.get("error"):
                    flags += f" ERROR({r['error'][:40]})"
                print(f"     [{m.key}] score={r['normalized']:.2f} "
                      f"(verifiable-only={vo:.2f}, unverifiable={r.get('n_unverifiable', 0)}){flags}")
                score_rows.append([tf, subj, m.key, round(r["normalized"], 3),
                                   round(vo, 3), r.get("n_unverifiable", 0), len(items),
                                   r.get("valid", True), r.get("penalty_floored", False)])
            self_runs = []
            if repeats > 1:
                self_runs = [judge(self_model, bundle, items, task["prompt"],
                                   temperature=temperature, verbose=verbose)
                             for _ in range(repeats)]
            matrix[subj] = {"base": base, "self": self_runs}
            report[tf]["subjects"][subj] = {
                "is_null": run_common.is_null_subject(tf, subj),
                "base": base,
            }

        # ---- derive: self-consistency (per subject) ----
        if repeats > 1:
            for subj, cell in matrix.items():
                if not cell["self"]:
                    continue
                ag = _agreement(items, cell["self"])
                report[tf]["subjects"][subj]["self_consistency"] = {
                    "model": self_model.key, "repeats": repeats, "temperature": temperature, **ag}
                print(f"  self-consistency [{self_model.key} x{repeats}] on '{subj}': "
                      f"scores={ag['total_scores']} stdev={ag['total_stdev']} "
                      f"unanimous={ag['frac_items_unanimous']:.0%}")
                cons_rows.append([tf, subj, "self", f"{self_model.key}x{repeats}",
                                  ag["total_mean"], ag["total_stdev"],
                                  ag["frac_items_unanimous"], ag["mean_item_stdev"]])

        # ---- derive: cross-model consistency (per subject) ----
        if len(judges) > 1:
            for subj, cell in matrix.items():
                runs = list(cell["base"].values())
                ag = _agreement(items, runs)
                scores = {k: round(r["normalized"], 3) for k, r in cell["base"].items()}
                pair = {f"{a}-{b}": round(abs(scores[a] - scores[b]), 3)
                        for a, b in itertools.combinations(scores, 2)}
                report[tf]["subjects"][subj]["cross_model"] = {
                    "models": list(scores), "scores": scores, "pairwise_gap": pair, **ag}
                print(f"  cross-model on '{subj}': scores={scores} "
                      f"stdev={ag['total_stdev']} unanimous={ag['frac_items_unanimous']:.0%}")
                cons_rows.append([tf, subj, "cross", "+".join(scores),
                                  ag["total_mean"], ag["total_stdev"],
                                  ag["frac_items_unanimous"], ag["mean_item_stdev"]])

        # ---- derive: boundary calibration (per judge, needs gold + a null) ----
        nulls = [s for s in matrix if report[tf]["subjects"][s]["is_null"]]
        if "gold" in matrix and nulls:
            report[tf]["boundary"] = {}
            for m in judges:
                gold_r = matrix["gold"]["base"][m.key]
                null_runs = [matrix[s]["base"][m.key] for s in nulls]
                worst_null = max((r["normalized"] for r in null_runs), default=0.0)
                gap = gold_r["normalized"] - worst_null
                leaks = _leaks(items, null_runs)
                report[tf]["boundary"][m.key] = {
                    "gold": round(gold_r["normalized"], 3),
                    "worst_null": round(worst_null, 3), "gap": round(gap, 3),
                    "leaks": leaks}
                print(f"  boundary [{m.key}]: gold={gold_r['normalized']:.2f} "
                      f"worst_null={worst_null:.2f} gap={gap:+.2f} leaks={len(leaks)}")

    _write_outputs(report, score_rows, cons_rows)
    print(CostTracker.summary())


def _hack_flag(report, tf, subj, judge_key, norm) -> str:
    """Mark reward-hacking suspects: leaks on gold, or a null scoring too high."""
    is_null = report[tf]["subjects"][subj]["is_null"]
    if is_null and norm > NULL_HACK_SCORE:
        return "null-scores-high"
    if subj == "gold":
        n = len(report.get(tf, {}).get("boundary", {}).get(judge_key, {}).get("leaks", []))
        if n:
            return f"{n}-leaks"
    return ""


def _write_outputs(report, score_rows, cons_rows):
    out_json = config.RESULTS / "scorecard.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_json.relative_to(config.ROOT)}")

    # annotate score rows with boundary gap + reward-hack flag
    full_rows = []
    for r in score_rows:
        tf, subj, jk, norm = r[0], r[1], r[2], r[3]
        gap = ""
        if subj == "gold":
            b = report.get(tf, {}).get("boundary", {}).get(jk)
            if b:
                gap = b["gap"]
        full_rows.append(r + [gap, _hack_flag(report, tf, subj, jk, norm)])

    run_common.write_tsv(
        config.RESULTS / "scorecard.tsv",
        ["task", "subject", "judge", "score", "verifiable_only", "n_unverifiable",
         "n_items", "valid", "penalty_floored", "gold_gap", "reward_hack_flag"],
        full_rows)
    if cons_rows:
        run_common.write_tsv(
            config.RESULTS / "consistency.tsv",
            ["task", "subject", "kind", "judges", "mean_score", "score_stdev",
             "frac_unanimous", "mean_item_stdev"],
            cons_rows)


# --- rollout mode -----------------------------------------------------------
def run_rollout(tasks, policy_keys, verbose, force=False):
    policies = resolve(policy_keys)
    rows = []
    for tf in tasks:
        print(f"\n=== {tf} ===")
        for m in policies:
            try:
                res = policy.rollout(m, tf, verbose=verbose, force=force)
            except Exception as e:  # one model/task failing must not abort the batch
                print(f"  [{m.key}] {tf}: ERROR -> {str(e)[:160]}")
                rows.append([tf, m.key, False, 0, f"(error: {str(e)[:80]})"])
                continue
            rows.append([tf, m.key, res["submitted"], res["rounds"],
                         "|".join(res["deliverable_files"]) or "(none)"])
    run_common.write_tsv(
        config.RESULTS / "rollouts.tsv",
        ["task", "policy", "submitted", "rounds", "deliverable_files"],
        rows)
    print(CostTracker.summary())
    print("\nGrade them with, e.g.:")
    print(f"  python -m eval.run --mode judge --subjects rollout:{policies[0].key} "
          f"--judge-models {' '.join(policy_keys)}")


# --- CLI --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Unified rubric-scorecard eval runner.")
    ap.add_argument("--mode", choices=["judge", "rollout", "iterate"], default="judge")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="task folders; default = all (rollout) / all with gold (judge, iterate)")
    ap.add_argument("--verbose", action="store_true")

    # judge mode
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="what to grade: gold | blank | unedited_input | rollout:<key> | "
                         "preexisting:<path>. Default = gold + available nulls.")
    ap.add_argument("--judge-models", nargs="*", default=["gpt-mini"],
                    help=">=2 models turns ON cross-model consistency.")
    ap.add_argument("--repeats", type=int, default=1,
                    help=">1 turns ON self-consistency (one judge, repeated).")
    ap.add_argument("--self-model", default=None,
                    help="judge used for self-consistency repeats (default: first --judge-model).")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="temperature for self-consistency repeats.")

    # rollout mode
    ap.add_argument("--policy-models", nargs="*", default=["gpt-mini"],
                    help="model(s) that attempt the tasks to generate deliverables.")
    ap.add_argument("--force", action="store_true",
                    help="rollout mode: regenerate even if a deliverable already exists.")

    # iterate mode (the rubric-improvement loop; reuses --judge-models / --repeats /
    # --self-model / --temperature / --subjects as the per-version evaluation config)
    ap.add_argument("--rounds", type=int, default=1,
                    help="iterate mode: max rewrite rounds per task (default 1).")
    ap.add_argument("--rewrite-model", default="gpt-mini",
                    help="iterate mode: model that rewrites flagged criteria.")
    a = ap.parse_args()

    if a.mode == "rollout":
        tasks = a.tasks or run_common.all_tasks()
        run_rollout(tasks, a.policy_models, a.verbose, a.force)
    elif a.mode == "iterate":
        from . import iterate
        tasks = a.tasks or run_common.tasks_with_gold()
        iterate.run_iterate(tasks, a.rounds, a.judge_models, a.self_model,
                            a.repeats, a.temperature, a.rewrite_model,
                            a.subjects, a.verbose)
    else:
        tasks = a.tasks or run_common.tasks_with_gold()
        run_judge(tasks, a.subjects, a.judge_models, a.repeats,
                  a.self_model, a.temperature, a.verbose)


if __name__ == "__main__":
    main()
