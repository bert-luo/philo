"""The judge harness: one model grades one deliverable against one rubric.

It is a small tool-calling agent. The model inspects the deliverable with the
modality tools it's allowed, then calls submit_scorecard. We normalize the
result to a [0,1] rubric score.
"""
from __future__ import annotations

import json

from . import llm, tools
from .deliverable import Bundle
from .models import Model
from .rubric import RubricItem, normalize

MAX_ROUNDS = 12
MAX_TOOL_CALLS = 30          # hard budget across the whole judgement
MAX_CALLS_PER_TURN = 8       # ignore parallel-call floods within one turn

SYSTEM = """You are a meticulous, skeptical grader. You score a work deliverable against a fixed rubric.

Rules:
- Grade the files marked role=deliverable (the worker's output). Files marked
  role=reference are the INPUTS the worker was given; open them only to check
  comparison criteria (e.g. "matches the script"), never grade them as output.
- INSPECT before you score. Do not guess.
- Use the DETERMINISTIC tools for objective facts instead of eyeballing:
  * file_facts  -> extension/type, PDF page count, word count (e.g. "is a PDF",
                   "<= 2 pages", word limits)
  * probe_media -> codec, exact resolution, fps, runtime in seconds (e.g.
                   "H.264", "1920x1080", "runtime 29.9-30.1s", "music fits 30s")
- For visual criteria ("includes a shot of X", identifiable faces, on-screen
  text), call view_video_frames; it returns one frame per scene cut. To read a
  text card exactly, re-sample that time window with start_seconds/end_seconds.
- For anything about sound (music style, tone, mix, audible VO) you MUST consult
  the audio: listen_audio if you have it, otherwise transcribe_audio. Do not score
  audio criteria from visuals alone.
- Judge ONLY what the rubric asks. Give each item a verdict:
  * met          criterion clearly TRUE of the deliverable (full points)
  * unmet        criterion clearly FALSE (0)
  * partial      genuinely in between (give `awarded` in [0,1])
  * unverifiable the criterion cannot be checked from the deliverable or ANY tool
                 (external provenance like "footage is royalty-free / from Pexels",
                 "music from a stock provider") and there is no source log in the
                 deliverable. Use it for exactly these cases — not as a cop-out.
- PENALTY items have negative points and name a bad thing: 'met' means the bad
  thing IS present (incurs the penalty). A clean gold deliverable is 'unmet' on penalties.
- Be strict: a heading being present is not the same as it being correct.
- A blank / do-nothing deliverable should be 'unmet' on almost everything.
- When done, call submit_scorecard exactly once with one entry per rubric item id."""


def _rubric_block(items: list[RubricItem]) -> str:
    lines = []
    for it in items:
        tag = "PENALTY " if it.max_score < 0 else ""
        lines.append(f"[{it.id}] ({tag}{it.max_score:g} pts) {it.criterion}")
    return "\n".join(lines)


def _apply_scores(args: dict, awarded: dict, unverifiable: set, notes: dict) -> None:
    """Fold one submit_scorecard payload into the running tallies (verdict -> fraction)."""
    for e in args.get("scores", []):
        if "id" not in e:
            continue
        # Some judges echo the id with the surrounding brackets from "[<id>]" in the
        # rubric block; strip them so the verdict matches the real rubric_item_id.
        iid = str(e["id"]).strip().strip("[]").strip()
        verdict = (e.get("verdict") or "").lower()
        if verdict == "met":
            awarded[iid] = 1.0
        elif verdict == "unmet":
            awarded[iid] = 0.0
        elif verdict == "unverifiable":
            awarded[iid] = 0.0
            unverifiable.add(iid)
        else:  # partial, or a model that ignored the enum
            awarded[iid] = max(0.0, min(1.0, float(e.get("awarded", 0.0))))
        if e.get("note"):
            notes[iid] = e["note"]


def judge(
    model: Model,
    bundle: Bundle,
    items: list[RubricItem],
    task_prompt: str,
    temperature: float = 0.0,
    verbose: bool = False,
    protocol: str = "",
) -> dict:
    """Run the agent loop. Returns normalize(...) dict plus raw notes/meta.

    `protocol` is a per-rubric grading protocol: task-specific verification rules
    that travel WITH the rubric version (the iterator can evolve them). They are
    part of the reward function — e.g. "for any stated fact, confirm it against the
    reference log; do not credit a bare assertion" — and are how the loop closes
    hacks that no criterion-text edit can reach (a transcribe-the-rubric attack)."""
    del_list = "\n".join(f"- {f['name']} [{f['type']}, {f['size_kb']}KB]"
                         for f in bundle.listing(role="deliverable")) or "(empty — a do-nothing output)"
    ref_files = bundle.listing(role="reference")
    ref_block = ""
    if ref_files:
        ref_list = "\n".join(f"- {f['name']} [{f['type']}, {f['size_kb']}KB]" for f in ref_files)
        ref_block = (
            "\n\nREFERENCE INPUTS (what the worker was GIVEN — NOT the deliverable, never "
            "gradeable; consult only for comparison criteria via read_reference/view_reference/"
            "compare_audio_sync):\n" + ref_list)
    user = (
        f"TASK GIVEN TO THE WORKER:\n{task_prompt.strip()}\n\n"
        f"DELIVERABLE TO GRADE (bundle '{bundle.label}') — grade ONLY these files. Award credit "
        f"only for what is present IN the deliverable itself, never for content that appears only "
        f"in a reference input:\n{del_list}{ref_block}\n\n"
        f"Inspect with the tools, then grade every rubric item below.\n\n"
        f"RUBRIC ({len(items)} items):\n{_rubric_block(items)}"
    )
    if protocol.strip():
        user += (
            "\n\nGRADING PROTOCOL — task-specific verification rules that apply to "
            "EVERY item above. They tighten how you must check a criterion; follow "
            "them even when a criterion looks superficially satisfied. They only ever "
            "make you MORE skeptical, never a reason to award points more freely:\n"
            + protocol.strip())
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    schemas = tools.tool_schemas(model)

    awarded: dict[str, float] = {}
    unverifiable: set[str] = set()
    notes: dict[str, str] = {}
    seen: dict[str, str] = {}     # tool-call signature -> prior text (dedupe loops)
    rounds = 0
    tool_calls_used = 0
    submitted = False

    while rounds < MAX_ROUNDS and not submitted:
        rounds += 1
        # last chance, or out of inspection budget: force the scorecard
        force = rounds == MAX_ROUNDS or tool_calls_used >= MAX_TOOL_CALLS
        msg = llm.chat(
            model, messages, tools=schemas,
            tool_choice=({"type": "function", "function": {"name": "submit_scorecard"}}
                         if force else "auto"),
            temperature=temperature, max_tokens=4096,
        )
        tcs = msg.tool_calls or []
        # Record the assistant turn with replay-safe (re-serialized) tool calls.
        replay, parsed_args = llm.assistant_tool_calls(tcs)
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": replay,
        })
        if not tcs:
            # model answered in prose; nudge it to use the tool
            messages.append({"role": "user",
                             "content": "Call submit_scorecard now with one entry per item id."})
            continue

        media_followups: list[dict] = []
        for i, tc in enumerate(tcs):
            args = parsed_args.get(tc.id, {})
            # Cap parallel-call floods: answer the first few, refuse the rest.
            if tc.function.name != "submit_scorecard" and i >= MAX_CALLS_PER_TURN:
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": "skipped: too many tool calls in one turn"})
                continue
            if tc.function.name != "submit_scorecard":
                tool_calls_used += 1
                sig = tc.function.name + "|" + json.dumps(args, sort_keys=True)
                if sig in seen:
                    # Repeat call: acknowledge without re-attaching media (saves tokens, breaks loops).
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": "(already provided earlier) " + seen[sig][:200]})
                    continue
            if tc.function.name == "submit_scorecard":
                _apply_scores(args, awarded, unverifiable, notes)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": "scorecard received"})
                submitted = True
                continue
            text, parts = tools.dispatch(tc.function.name, args, bundle)
            seen[sig] = text
            if verbose:
                print(f"  [{model.key}] {tc.function.name}({args}) -> {text[:60]!r}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
            media_followups.extend(parts)

        if media_followups and not submitted:
            messages.append({"role": "user", "content":
                             [{"type": "text", "text": "Attached media from your tool call(s):"},
                              *media_followups]})

    # Top-up: some models (esp. on big rubrics / heavy media) submit a partial or
    # empty scorecard. Re-ask, listing ONLY the missing items, before giving up.
    # Without this, an under-filled scorecard is indistinguishable from a real 0.
    topups = 0
    while submitted and topups < 2:
        missing = [it for it in items if it.id not in awarded]
        if len(missing) <= len(items) * 0.05:   # essentially complete
            break
        topups += 1
        messages.append({"role": "user", "content":
            (f"You scored only {len(awarded)} of {len(items)} items. "
             f"Call submit_scorecard NOW with verdicts for the {len(missing)} REMAINING items:\n"
             + _rubric_block(missing))})
        try:
            msg = llm.chat(model, messages, tools=schemas,
                           tool_choice={"type": "function", "function": {"name": "submit_scorecard"}},
                           temperature=temperature, max_tokens=4096)
        except Exception as e:
            # Best-effort: a flaky top-up must not discard the scorecard we already have.
            if verbose:
                print(f"  [{model.key}] top-up failed ({str(e)[:60]}); keeping {len(awarded)} items")
            break
        replay, parsed_args = llm.assistant_tool_calls(msg.tool_calls or [])
        for tc in (msg.tool_calls or []):
            if tc.function.name == "submit_scorecard":
                _apply_scores(parsed_args.get(tc.id, {}), awarded, unverifiable, notes)
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": replay})
        for tc in (msg.tool_calls or []):
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ok"})

    result = normalize(items, awarded, unverifiable)
    result["notes"] = notes
    result["unverifiable"] = sorted(unverifiable)
    result["rounds"] = rounds
    result["submitted"] = submitted
    result["model"] = model.key
    result["n_items_scored"] = len(awarded)
    # A judgement is only trustworthy if (near) every item got a verdict.
    result["valid"] = submitted and len(awarded) >= 0.9 * len(items)
    # Did penalties exceed the positive ceiling (score floored at 0)? -> rubric defect signal.
    result["penalty_floored"] = result["raw_normalized"] < 0
    return result


def judge_safe(model: Model, bundle: Bundle, items: list[RubricItem], task_prompt: str,
               **kw) -> dict:
    """judge() that never raises: on failure returns an error-marked result so a
    single bad call can't abort a whole sweep."""
    try:
        return judge(model, bundle, items, task_prompt, **kw)
    except Exception as e:
        print(f"  [{model.key}] ERROR on bundle '{bundle.label}': {str(e)[:120]}")
        res = normalize(items, {})
        res.update({"error": str(e)[:300], "notes": {}, "unverifiable": [],
                    "model": model.key, "submitted": False, "n_items_scored": 0,
                    "valid": False, "penalty_floored": False})
        return res
