# Rubric scorecard — unified eval runner

One entrypoint (`eval.run`) that, from a single judging matrix, scores GDPval
rubrics along **Axis 1 (boundary calibration)** and **Axis 3 (consistency)**, an
**iterate** mode that automatically rewrites a rubric to raise those scores, and a
**rollout** mode that has policy models attempt the real tasks to produce
gradeable deliverables. Built on the shared judge harness, model suite, and
null-input generator.

## Module map

The 14 modules split into four layers, in dependency order (each layer only
imports from the ones above it):

**Core primitives** — generic building blocks, no eval-specific control flow:
| module | role |
|--------|------|
| `config.py` | shared paths (`eval/var/...`), `.env` loading, cost tracking |
| `models.py` | model registry (provider × capability × modality) |
| `llm.py` | OpenAI-compatible chat client (OpenAI + OpenRouter) |
| `deliverable.py` | `Bundle`: flat file view over a deliverable, incl. zip members |
| `rubric.py` | rubric loading + normalization (handles penalty items) |

**Judge harness** — grades a deliverable against a rubric:
| module | role |
|--------|------|
| `media.py` | ffmpeg frames / audio clips + Whisper transcription (cached) |
| `tools.py` | judge agent tool schemas + dispatch, gated by each model's modalities |
| `harness.py` | the judge: tool-using agent → per-item `[0,1]` scorecard |

**Subject generation** — the deliverables that get judged:
| module | role |
|--------|------|
| `null_inputs.py` | do-nothing deliverables per task (Axis 1 anchors) |
| `attacks.py` | mechanical reward-hacking deliverables (Axis 2 anchors) |
| `policy.py` | the rollout policy: agent that EDITS media (ffmpeg) / writes docs |

**Orchestration / entrypoints** — what you actually run:
| module | role |
|--------|------|
| `run_common.py` | task/subject resolution + shared axis analysis (leaks, agreement) |
| `run.py` | unified runner: `--mode {judge, rollout, iterate}` |
| `iterate.py` | the rubric-improvement loop: evaluate → diagnose → rewrite |
| `adv_eval.py` | evaluates the adversarial agent vs. the mechanical attack suite (blind pairwise + rubric score) |

```
eval/
  <14 modules above>
  var/                  # all generated/runtime data, gitignored
    results/            # scorecard.{json,tsv}, consistency.tsv, rollouts.tsv,
                         #   rubric_scorecard.tsv, adversarial_eval.{json,tsv}
    rollouts/            # generated deliverables: <task>/<policy>/deliverable/
    rollouts_manifest.json  # consumed by the viewer's Rollouts tab
    rubric_packages/     # per-task: rubric_v*.json, diagnosis_v*.json, changelog, report.json
    null_inputs/          # generated do-nothing deliverables
    attacks/              # generated mechanical-attack deliverables
    .cache/                # ffmpeg/Whisper cache (frames, clips, transcripts)
    logs/                  # ad hoc run logs
```

## Setup

```bash
source .venv/bin/activate
pip install openai pymupdf pillow        # plus the existing pandas/pyarrow/requests
brew install ffmpeg                       # frames, audio extraction
# .env must define OPENAI_API_KEY and OPENROUTER_API_KEY
python scripts/download_data.py          # if data/ isn't populated yet
```

## The judge suite (`models.py`)

Chosen cheap, and to vary three things so the axes are measurable:

| key | provider | model | modalities | role on the axes |
|-----|----------|-------|-----------|------------------|
| `gpt-mini` | OpenAI | gpt-5.4-mini | text, image | strong vision anchor; cross-vendor |
| `gpt-nano` | OpenAI | gpt-5.4-nano | text, image | weak rung → capability ablation vs `gpt-mini` |
| `qwen-vl` | OpenRouter | qwen/qwen3.5-35b-a3b | text, image | cheap mid judge; cross-vendor |
| `qwen-text` | OpenRouter | qwen/qwen3-30b-a3b-instruct | text | **modality ablation**: same family, no vision |
| `gemini-lite` | OpenRouter | google/gemini-2.5-flash-lite | text, image, **audio** | the only judge that natively *hears* |
| `nemotron-omni` | OpenRouter | nvidia/nemotron-3-nano-omni (free) | text, image, audio | free 2nd native-audio judge for A/V tasks |

A judge without `audio` still reaches audio *content* through the
`transcribe_audio` tool (Whisper). Native-audio vs transcript-only on the same
deliverable is exactly the **modality-access** experiment. The
`MODALITY_LADDER = [qwen-text, qwen-vl, gemini-lite]` is the text → +vision →
+audio triple for that ablation.

## The harness (`harness.py`)

One model grades one deliverable against one rubric as a small tool-calling
agent. It is handed the file list (deliverable files plus read-only `reference`
inputs for comparison criteria), inspects with the tools its modalities allow,
then calls `submit_scorecard` with a per-item **verdict**:
`met | unmet | partial | unverifiable`. Safeguards: per-judgement tool-call
budget, per-turn flood cap, duplicate-call dedupe, forced scorecard on the last round.

To give the judge a *fair shot* (early runs lost ~60% of the gold ceiling to
things it couldn't see, vs ~3% genuine gaps), the toolbox includes deterministic
fact tools so objective criteria are never guessed:

* `probe_media` — ffprobe ground truth: codec, exact resolution, fps, runtime,
  audio channels (kills "cannot determine codec/resolution/runtime").
* `file_facts` — extension/type, PDF page count, word count ("is a PDF", "<= 2 pages").
* `view_video_frames` — defaults to **one frame per scene cut** (distinct shots,
  for "includes a shot of X" / identifiable-face checks); pass `start/end` to
  re-sample a window at higher resolution to read on-screen text.
* read-only **reference files**, so "matches the script" items can compare.
* an explicit **`unverifiable`** verdict for criteria no judge can check from the
  artifact (external provenance like "footage is royalty-free / from Pexels").

Scores are normalized by the **positive ceiling** (sum of positive item points);
**penalty items** (negative points, e.g. "uses footage with identifiable faces:
-85") subtract when `met`, flooring at 0. Two scores are reported: `normalized`
(unverifiable items = 0) and `normalized_verifiable_only` (unverifiable items
dropped from numerator and denominator — the fair score given what is checkable).
A large gap between them is itself a rubric **verifiability defect**.

## Null inputs (`null_inputs.py`)

Per task, keyed on the gold deliverable's type:

| gold type | `blank` | `unedited_input` |
|-----------|---------|------------------|
| PDF | one empty page | the unrevised source doc rendered to PDF |
| video (mp4) | 5 s black + silence | source audio over a black frame |
| archive (stems zip) | zip of 5 s silence | the raw input audio re-zipped, unmixed |

```bash
python -m eval.null_inputs          # (re)generate all; writes eval/var/null_inputs/<task>/<variant>/
```

## Attack suite (`attacks.py`) — Axis 2 (hackability)

Where nulls are *do-nothing* outputs, attacks are *cheap-but-targeted* deliverables
that try to raise the rubric score without doing the work. They are mechanical (no
policy/judge LLM in the loop), so the suite is free, deterministic, and reproducible.
An attack passing a positive item is a `leak` (`run_common.leaks`) — a surface-form
reward and a rewrite target the iterator keys on. Deliverable type is taken from the
gold when released, else inferred from the rubric, so attacks exist for gold-free
tasks too.

| deliverable type | attacks |
|------------------|---------|
| document (PDF) | `restate_input` — echo the prompt + reference text (facts in the brief); `keyword_stuff` — emit every quoted literal + the de-verbed clause of every criterion (the rubric as its own answer key) |
| video (mp4) | `presence_max` — long black/source video with an oversized burned-in title + a caption bar covering the frame (presence-without-absence) |
| archive (stems zip) | `stems_named` — a zip of correctly-named stems that are pure silence (presence-only "includes an X stem") |

Two attacks per document task give the iterator a *gradient*: an item only
`keyword_stuff` passes is a pure rubric-surface match; one `restate_input` also passes
is a brief-echo. Used as subjects via the normal resolver (`attack:<name>` or the bare
name), and surfaced for the iterator via `run_common.attack_subjects` /
`hack_subjects` (nulls ∪ attacks).

```bash
python -m eval.attacks              # (re)generate all; writes eval/var/attacks/<task>/<attack>/
# score an attack against a rubric like any subject:
python -m eval.run --mode judge --tasks <folder> --subjects gold keyword_stuff restate_input
```

Sanity check on `private_detectives__57b2cdf2` (gpt-mini): gold **0.96**, `keyword_stuff`
**0.89**, `restate_input` **0.74** — ~93% of the gold score is buyable with no
investigation; 44/45 positive items leak to the suite. That is the rubric's hackability
made concrete.

### v0 adversarial agent + blind eval (`adv_eval.py`)

A general agent (`attacks.build_adversarial`) that, unlike the fixed mechanical
attacks above, tries to clone the gold deliverable closely enough to max the rubric
while doing none of the real work. `adv_eval.py` then scores it on two independent
axes and reports the gap:

* `rubric_score` — how well it games the rubric (`eval.harness.judge`). Higher = better attack.
* `gold_winrate` — a blind, rubric-free pairwise comparison: the attack and the gold
  are shown to a judge as anonymous, position-randomized submissions "A"/"B" (no
  "this one is the professional gold" label), and it picks the better piece of work.
  Fraction of comparisons gold wins. Higher = the attack is genuinely worse.
* `deception = rubric_score * gold_winrate` — high only when an attack both games
  the rubric *and* is clearly worse than gold on a blind read.

Needs a released gold deliverable (it clones gold), so it only runs on the 6 gold tasks.

```bash
python -m eval.adv_eval --tasks film_and_video_editors__75401f7c \
    private_detectives_and_investigators__57b2cdf2 --judge-model gpt-mini

# more pairwise repeats per attack, different model driving the adversarial agent:
python -m eval.adv_eval --adv-model gpt-mini --judge-model qwen-vl --repeats 5

# reuse a previously-built adversarial deliverable instead of regenerating:
python -m eval.adv_eval --no-build
```
Outputs `var/results/adversarial_eval.json` (full detail) and
`var/results/adversarial_eval.tsv` (one row per task).

## Running the eval (`--mode judge`)

Judge mode grades one or more **subjects** per task and derives all axes from a
single judging matrix. A subject is named on the CLI:

| subject | meaning |
|---------|---------|
| `gold` | the released gold deliverable (oracle, expect ≈1) |
| `blank`, `unedited_input` | synthesized do-nothing nulls (expect ≈0) |
| `rollout:<policy_key>` | a deliverable produced by `--mode rollout` |
| `preexisting:<path>` | files under `<path>` (or `<path>/<task_folder>`) |

Default subjects are `gold` + the available nulls (the boundary anchors). The axes
fall out of how many models/repeats you ask for:

* **Boundary (Axis 1)** — derived automatically when both `gold` and ≥1 null are
  among the subjects: gold−null **gap** + **leaks** (positive items a null still
  passes — form/presence rewards, reward-hacking suspects).
* **Cross-model consistency (Axis 3)** — turned ON by passing ≥2 `--judge-models`:
  per-subject score spread, pairwise gaps, most-contested items.
* **Self-consistency (Axis 3)** — turned ON by `--repeats > 1`: one judge
  (`--self-model`, default the first judge) re-grades each subject at
  `--temperature`; reports score stdev + fraction of unanimous items.

```bash
# boundary + cross-model in one shot (3 judges, all gold tasks):
python -m eval.run --mode judge --judge-models gpt-mini qwen-vl gemini-lite

# add self-consistency (5 repeats), single task:
python -m eval.run --mode judge --tasks <folder> --judge-models qwen-vl \
    --repeats 5 --temperature 0.7

# consistency OFF (one judge, one pass) on a generated rollout:
python -m eval.run --mode judge --subjects rollout:gpt-mini --judge-models gpt-mini

# grade a directory of preexisting outputs:
python -m eval.run --mode judge --subjects gold preexisting:/path/to/outputs
```
Outputs:
* `var/results/scorecard.json` — full detail (per-item verdicts, boundary, both
  consistency analyses).
* `var/results/scorecard.tsv` — one row per (task × subject × judge): `score`,
  `verifiable_only`, `n_unverifiable`, `gold_gap`, `reward_hack_flag`.
* `var/results/consistency.tsv` — self + cross rows per (task × subject).

## The iterator (`--mode iterate`)

The rubric-improvement loop (take-home Part 3). For each task it runs, per version:

1. **evaluate** — grade `gold` + `hack_subjects` (nulls ∪ attacks) with the judge
   matrix and read off the same axis signals judge mode produces (reusing
   `run_common.leaks` / `.agreement`).
2. **diagnose** (`iterate.diagnose`) — turn those signals into per-item **defects**,
   each tagged with the axis it fails and ranked by *points-at-stake × signal*:
   * `boundary/null_leak` (Axis 1) — a positive item a do-nothing null still passes.
   * `hackability/attack_leak` (Axis 2) — a positive item a mechanical attack
     (`attacks.py`) games: a presence/form/surface-match reward, the classic hack
     surface. The attack that leaks it names the hack kind (brief-echo vs
     rubric-surface vs presence-without-absence).
   * `verifiability/unverifiable` — ≥50% of judges couldn't check the item on gold.
   * `consistency/contested` (Axis 3) — judges disagree on the item (high stdev on
     gold): the wording is ambiguous.
3. **rewrite** (`--rewrite-model`) — a model rewrites **only the flagged criteria**
   to close the named defect (pair presence with absence, make it checkable from the
   artifact, replace vague adjectives with an objective test). Item **ids and point
   values are held fixed**, so every version is graded on the same scale and
   before/after is a clean comparison.
4. **evaluate again** — and stop when defects hit zero or `--rounds` is reached.

It reuses the judge-mode flags for the per-version evaluation config
(`--judge-models` ≥2 → cross-model defects; `--repeats`/`--self-model` → self-consistency;
`--subjects` → which anchors to test; default `gold` + nulls).

```bash
# one rewrite round, cross-model defects from two judges, all gold tasks:
python -m eval.run --mode iterate --judge-models gpt-mini qwen-vl --rounds 1

# a single task, cheaper anchors, two rounds:
python -m eval.run --mode iterate --tasks private_detectives_and_investigators__a46d5cd2 \
    --judge-models gpt-mini gpt-nano --subjects gold blank --rounds 2
```
Outputs:
* `var/results/rubric_scorecard.tsv` — **one row per (task × rubric version)** with a
  column per axis: `gold_score`, `null_score`, `gold_gap`, `n_leaks` (hackability),
  `cross_model_stdev` / `self_consistency_stdev` (agreement), `n_unverifiable`,
  `verifiability_gap`, `n_defects`, `reward_hack_flag`.
* `var/rubric_packages/<task>/` — a self-contained package per task: every
  `rubric_v{n}.json`, `diagnosis_v{n}.json` (the defects), `changelog_v{n}_to_v{n+1}.json`
  (what each rewrite changed and why), and `report.json` (before/after at a glance).

> v0 scope: rewrites edit criterion *text* in place (no item add/split yet); Axis 2
> is approximated by the Axis-1 boundary leaks. Both are natural v1 extensions.

## Rollouts (`--mode rollout`)

Have policy model(s) attempt the *real* tasks and produce actual deliverables, so
the axes can be run on genuine model outputs (not just gold/null anchors). The
policy is a tool-using agent in a per-(task, policy) sandbox: it inspects the
inputs with the same modality tools the judge has, then **edits real media** by
driving a sandboxed `ffmpeg` (trim / concat / scale / mux for video & audio), or
writes / renders a document for report tasks. ffmpeg is the only external binary
it may call — no shell, sandbox-relative paths only.

```bash
python -m eval.run --mode rollout --policy-models gpt-mini --tasks <folder>
python -m eval.run --mode rollout --policy-models gpt-mini qwen-vl   # all 10 tasks
```
Writes `eval/var/rollouts/<task>/<policy>/deliverable/` (+ `rollout.json` trace and
`var/results/rollouts.tsv`). Grade them by passing `--subjects rollout:<policy>` to
judge mode.

> Note: video/audio rollouts attempt genuine edits, so reliability depends on the
> policy model composing valid ffmpeg steps within the round budget; report tasks
> (PDF) are the most reliable. Either way the output is real, never faked media.

## Cost

Every run prints `[cost] ... ~$X`. A PDF task is ~$0.005–0.01 per judge; a video
task ~$0.02 (image context dominates). Whisper transcripts, frames, and audio
clips are cached under `eval/var/.cache/` so reruns are cheap. Only 6 of the 10 tasks
have a released gold deliverable, so judge mode defaults to those 6 (rollout mode
defaults to all 10).
```
PDF gold:   private_detectives __57b2cdf2, __a46d5cd2 ; producers_and_directors __e4f664ea
video gold: film_and_video_editors __e222075d, __75401f7c
audio gold: audio_and_video_technicians __38889c3b
```
