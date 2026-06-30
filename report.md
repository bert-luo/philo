# Progress Report — Rubric Scorecard for GDPval

## Overview

Building a rubric evaluator and iterator for the 10 video-centric GDPval tasks (Philo Labs take-home, Track R). The evaluator scores rubrics on **Axis 1 (boundary calibration)**, **Axis 2 (hackability)**, and **Axis 3 (consistency)**, plus derived diagnostics (leaks, attack leaks, unverifiable items). All four pieces now exist end to end: data, viewer, eval framework (judge + attacks + iterator + rollouts), and a first pass of real results across all axes.

## What Exists

### Data Layer
- 10 tasks downloaded into `data/` from the GDPval gold subset, covering 4 occupations across 2 capability families (editing: film/video editors, audio/video technicians, producers/directors; understanding: private detectives).
- 6 of 10 have released gold deliverables; 4 are gold-free (no ground-truth output — these get attacks and rollouts but no Axis 1 gap).
- Manifest at `data/manifest.json` drives the viewer.

### Viewer (`viewer/`)
- Static HTML/CSS/JS single-page app: sidebar task list → tabs (Prompt, Rubric, Reference Files, Gold Deliverable, Rollouts).
- Inline video/audio/image/PDF/text preview; rubric parsed from JSON into a sortable table with score pills and tags.
- Rollouts tab reads `eval/var/rollouts_manifest.json` and shows each policy's submit status, round count, and produced files.

### Eval Framework (`eval/`)

One entrypoint, `eval.run --mode {judge, rollout, iterate}`, built on a shared judge harness, model registry, and null/attack generators.

**Judge harness (`harness.py` + `tools.py`)**: the judge is a tool-calling agent. It gets the task prompt, the rubric, and a role-labelled file manifest (`deliverable` vs read-only `reference`), then calls tools until it submits a per-item verdict (`met` / `unmet` / `partial` / `unverifiable`). Deterministic fact tools (`probe_media` via ffprobe, `file_facts`, `analyze_audio`, `compare_audio_sync`, scene-cut frame sampling) exist so the judge never has to guess codec/resolution/runtime/loudness/sync — early runs lost ~60% of the gold ceiling to exactly that gap.

**Judge / policy model registry (`models.py`)** now spans 9 models chosen to vary three things at once:

| key | provider | modalities | role |
|-----|----------|-----------|------|
| `gpt-mini` / `gpt-nano` | OpenAI | text+image | capability ablation (strong vs weak, same vendor) |
| `qwen-vl` / `qwen-text` | OpenRouter | text+image / text-only | modality ablation (same family, vision on/off) |
| `gemini-lite` / `nemotron-omni` | OpenRouter | text+image+**audio** | native-audio judges (free 2nd one for A/V cross-checks) |
| `gemma4` / `qwen36` / `mimo` | OpenRouter | text+image(+audio for `mimo`) | rollout **policies** — agents that attempt the real tasks |

**Null inputs (`null_inputs.py`)**: per task, a `blank` (5s black+silence / empty PDF / silent stem zip) and `unedited_input` (source material delivered as-is, unrevised) — two cheap anchors for the bottom of Axis 1.

**Attack suite (`attacks.py`) — Axis 2, new this pass**: cheap, deterministic, no-LLM-in-the-loop deliverables designed to raise the rubric score without doing the work:
- `restate_input` (PDF) — echo the prompt + reference text back as the "report."
- `keyword_stuff` (PDF) — emit every quoted literal and de-verbed clause from the rubric itself, i.e. use the rubric as its own answer key.
- `presence_max` (video) — black/source video with an oversized burned-in title and a caption bar covering the frame: presence without the matching absence condition.
- `stems_named` (audio archive) — correctly-named stems that are pure silence: presence-only "includes an X stem" criteria pass for free.

A positive item any attack passes is a **leak**, surfaced the same way as null-leaks. Sanity check on `private_detectives__57b2cdf2`: gold scores 0.92–0.96 across judge runs, `keyword_stuff` scores 0.89–0.92, and `restate_input` scores 0.74–0.80 — **44 of 45 positive items leak to mechanical attacks**, i.e. ~90%+ of the score is buyable with zero investigative work.

**Adversarial agent eval (`adv_eval.py`) — new this pass**: goes one step further than mechanical attacks. An LLM agent is told the rubric and asked to game it (not do the work) while a separate **blind pairwise judge** — shown only anonymous "A"/"B" submissions, position-randomized, never told which is gold — independently rates which is actually better work. Two numbers per attack: `rubric_score` (how well it games the rubric) and `gold_winrate` (how often the blind judge still prefers gold). Their product, `deception`, is high only when an attack both games the rubric **and** is empirically worse:

| task | attack | rubric_score | gold_winrate | deception |
|------|--------|-------------:|-------------:|----------:|
| `film_and_video_editors__75401f7c` | `presence_max` (mechanical) | 0.22 | 1.00 | 0.22 |
| `film_and_video_editors__75401f7c` | adversarial agent | 0.93 | 0.50 | 0.466 |
| `private_detectives__57b2cdf2` | `restate_input` (mechanical) | 0.80 | 1.00 | 0.80 |
| `private_detectives__57b2cdf2` | `keyword_stuff` (mechanical) | 0.92 | 1.00 | 0.92 |
| `private_detectives__57b2cdf2` | adversarial agent | 1.00 | 1.00 | **1.00** |

The detectives rubric is the worst case found so far: an adversarial agent reaches a **perfect rubric score while the blind judge still unanimously prefers gold** — the rubric is fully satisfiable without doing the actual investigative work. The editors rubric is harder to game with a purely mechanical attack (0.22), but a smarter agent now reaches 0.93 on the rubric while the blind judge calls it only a coin-flip tie against gold (not yet a clear loss) — still a meaningful gap between "passes the checklist" and "is better than gold," and worth tightening further.

## Part 3: The Iterator

**The iterator (`iterate.py`, `--mode iterate`)**: an evaluate → diagnose → rewrite → re-evaluate loop. Diagnosis turns the signals above into per-item **defects**, each tagged with the axis it fails (`boundary/null_leak`, `hackability/attack_leak`, `verifiability/unverifiable`, `consistency/contested`) and ranked by points-at-stake × signal strength. A rewrite model (`gpt-5.4-mini`) edits only the flagged criteria (item ids/points held fixed, so before/after is comparable on the same scale). The accept/reject objective is `(gold − worst_attack)` margin, penalized for self-inconsistency — a round is kept **only if it raises this objective**; otherwise the loop reverts to the prior version and stops. Cross-model stdev is recorded but does not gate acceptance (rationale: a policy trains against one judge, so that judge's own wander is the RL-relevant signal, not disagreement between judges).

**First run (4 tasks × 2 rounds, parallel, ~24 min, $1.63)**: ran on `38889c3b`, `a46d5cd2`, `e4f664ea`, and `75401f7c` simultaneously using `gpt-mini` with 2 self-consistency repeats. Only `38889c3b` produced an accepted rewrite. Rate-limit 429 errors corrupted several gold evaluations (producing `gold=0.0` headlines), confirming the iterator needs a gold-score sanity check before trusting an objective computed from a transient judge failure.

**Second run (6 gold tasks × 3 rounds cap, 2 parallel batches, ~5 min total, $4.07)**: extended to all 6 gold tasks with the TSV-clobber fix (per-task `.tsv` files) and a gold-free guard (`iterate_task` breaks after v0 for tasks without a gold deliverable). Each task got up to 3 rewrite rounds; the loop ran 3 tasks per batch in parallel. Nothing was packaged before this run except e222075d (first-time pack), so the results are a clean snapshot of the iterator's current state across the full gold-task set.

#### Second-run results (all 6 gold tasks)

`eval/var/rubric_packages/<task>/report.json` is the source of truth.

| task | state | gold | worst attack | margin | objective | attack leaks | self-stdev | defects | outcome |
|------|---|-----:|------:|------:|------:|------:|------:|------:|:--:|
| `audio_technicians__38889c3b` | **v0 (best)** | **0.597** | **0.161** | **0.436** | **0.436** | 5 | 0.000 | **10** | ✅ kept |
| | v1 (tried, rejected) | 0.484 | 0.210 | 0.274 | 0.258 | 7 | 0.032 | 14 | ❌ backfire |
| `film_editors__75401f7c` | **v0 (best)** | **0.907** | **0.559** | **0.348** | **0.299** | 22 | 0.097 | **32** | ✅ kept |
| | v1 (tried, rejected) | 0.941 | 0.915 | 0.026 | 0.024 | 36 | 0.004 | 37 | ❌ backfire |
| `film_editors__e222075d` | **v0 (best)** | **0.721** | **0.131** | **0.590** | **0.566** | 4 | 0.049 | **10** | ✅ kept |
| | v1 (tried, rejected) | 0.000 ⚠ | 0.066 | −0.066 | −0.066 | 2 | 0.000 | 3 | ❌ collapsed |
| `private_detectives__57b2cdf2` | **v0 (best)** | **0.983** | **0.907** | **0.076** | **0.072** | 45 | 0.008 | **45** | ✅ kept |
| | v1 (tried, rejected) | 0.898 | 1.000 | −0.102 | −0.112 | 45 | 0.021 | 45 | ❌ backfire |
| `private_detectives__a46d5cd2` | **v0 (best)** | **0.884** | **0.637** | **0.247** | **0.238** | 43 | 0.017 | **48** | ✅ kept |
| | v1 (tried, rejected) | 0.712 | 0.712 | 0.000 | −0.013 | 49 | 0.027 | 51 | ❌ backfire |
| `producers_and_directors__e4f664ea` | v0 (initial) | 0.921 | 0.789 | 0.132 | 0.131 | 34 | 0.003 | 38 | — |
| | v1 (accepted) | 0.942 | 0.766 | 0.176 | 0.175 | 35 | 0.002 | 38 | ✅ improved |
| | **v2 (best)** | **0.899** | **0.655** | **0.244** | **0.237** | **28** | 0.013 | **32** | ✅ **+81%** |
| | v3 (tried, rejected) | 0.937 | 0.749 | 0.188 | 0.184 | 36 | 0.008 | 37 | ❌ backfire |

**Reliability note**: the `e222075d` v1 gold score (0.000) follows the same pattern as `75401f7c` and `e4f664ea` v1 in the first run: a real deliverable does not score 0.00, and the most likely explanation is a transient judge failure (429 rate limit or similar) that wasn't caught before the headline was written. The iterate loop still needs its own retry/sanity check on the gold score before trusting an objective computed from it — a 0.000 gold should trigger an automatic re-evaluate, not an accepted "improvement" or rejection. Logged as a harness fix.

**Earlier 4-judge run for context** (separate, pre-rerun pass, since superseded by the table above but useful as a second data point on `75401f7c`): gold 0.78→0.92, gap 0.51→0.65, but attack leaks rose 27→37 and worst-attack rose 0.70→0.91 — i.e. in that pass the rewrite *did* get accepted and *did* improve the gold-null gap while making hackability measurably worse, the same single-axis-tradeoff failure mode the current run's reject logic is designed to catch. Taken together, the two runs say the same thing about this rubric: it is easy to move the gold-null gap, hard to move gold and hackability in the same direction, and the loop now refuses to accept a rewrite unless both move together — which on this task currently means refusing to change anything (v0 kept, 0/11 defects fixed across two attempts).

**Honest read across all 6 gold tasks**:

- **`producers_and_directors__e4f664ea` is the second clean win** (and the strongest improvement yet): objective 0.131 → 0.237 (+81%), attack leaks 34→28, worst-attack 0.789→0.655. Three rounds of rewrites each independently improved either gold or worst-attack, and the hill-climb correctly rejected v3 when it regressed. This task's rubric appears genuinely amenable to the wording-level fixes the iterator produces.
- **`38889c3b` kept v0 this time** (previous run accepted v1). The v0 start differs (gold 0.597 vs 0.548) due to stochastic self-consistency from 2 repeats; the rewrite still backfired (gold ↓ 0.484, attack ↑ 0.210). With only 2 repeats, the objective is noisy enough that the accept/reject threshold is unreliable — the earlier accepted v1 may have crossed that threshold by luck of the draw.
- **`a46d5cd2`, `57b2cdf2`, `75401f7c` kept v0** — every rewrite backfired. `57b2cdf2` remains genuinely resistant: 45/45 attack leaks unchanged between versions, structural form-over-substance problem. `75401f7c` has 22 attack leaks at v0 — the highest leak count of any non-detective gold task — and worst-attack (0.559) is dangerously close to gold (0.907).
- **`e222075d` first run**: v1 gold collapsed to 0.000 (same rate-limit transient failure mode as earlier runs), so the real assessment is v0-only: gold=0.721, worst_attack=0.131, objective=0.566 — actually one of the stronger starting rubrics. Needs a gold-score sanity check before its next iteration attempt.
- **Signal-to-noise problem**: with only 2 self-consistency repeats, the objective has ~0.01–0.05 variance from stochastic judge wandering. This is enough to flip accept/reject decisions near the margin (as with 38889c3b, which was accepted in one run and rejected in the next). Increasing to 5 repeats would stabilize the objective but 5× the cost per evaluate call.
- **Overall: 2/6 tasks have shown an accepted improvement over at least one run, 4/6 remain stuck at v0.** The producer task (e4f664ea) is the first multi-round success story. The iterator's hill-climb works correctly (rejects backfires), but the rewrite model rarely finds a change that improves the multi-axis objective — the bottleneck is the rewrite, not the evaluation.

#### Gold-free tasks (4) — initial-state only, no gold means no iterator yet

The iterator's objective is gold-vs-attack margin, so it needs a gold deliverable; for the 4 gold-free tasks only the attack-only initial state has been measured (mechanical attacks, `gpt-mini`, no rewrite attempted):

| task | attack | score | leaked criteria |
|---|---|---:|---|
| `film_and_video_editors__a941b6d8` | `presence_max` | 0.229 (16/70 pts) | 10 leaks, all format/presence: exact resolution, fps, codec, container, pixel format, "no letterboxing," "no encoding glitches." |
| `film_and_video_editors__c94452e4` | `presence_max` | 0.143 (8/56 pts) | 4 leaks: extension, codec, "no on-screen text in opening shot," "no shot change while super remains." |
| `audio_technicians__4b894ae3` | `stems_named` | 0.000 (0/31 pts) | 0 leaks — every criterion requires real audio content; the most hack-resistant rubric found in the whole slice. |
| `audio_technicians__ff85ee58` | `stems_named` | 0.000, penalty-floored | 3 leaks (filename match, "is an audio file," "is a WAV") plus it tripped a −10pt loudness penalty — the floor saved the headline score but the underlying leaks are real. |

`e222075d` sits between these two groups: it has a gold deliverable and now a rubric package (v0 best, objective 0.566, 4 attack leaks). Its gold stdev (0.349 from the 5-repeat run) is the highest in the set — the −85pt "identifiable faces" penalty against a +61pt ceiling means missing/hitting that one criterion swings the whole floored score. This is a rubric defect, not harness noise, flagged as `consistency/contested` for a future iteration attempt (once the gold-score sanity check is in place).

**Rollouts (`--mode rollout`)**: policy agents (`gpt-mini`, `qwen36`, `gemma4`, `mimo`) attempted real deliverables for all 10 tasks by driving sandboxed `ffmpeg` (video/audio) or writing/rendering documents (PDF tasks), producing genuine edited media rather than faked outputs. All 10 task folders now have at least one policy attempt; PDF/report tasks are the most reliable (multiple full deliverables per task), video/audio tasks vary by whether the policy composed valid ffmpeg pipelines within its round budget. Two `a46d5cd2` rollouts (`qwen36`, `mimo`) have been graded end to end and appear in `results/rollouts.tsv`; grading the remaining rollouts against gold/attack subjects is the next step.

### Results (`eval/var/results/`)

Axis 1 + Axis 3 numbers below are from the clean re-run (single judge `gpt-mini`, 5 self-consistency repeats at T=0.7) after fixing the original harness bugs (reference contamination, macOS zip junk, bracketed-id echo, penalty normalization) and adding the audio DSP tools.

**Axis 1 (Boundary Calibration)**

| Task | Gold | Worst Null | Gap | Leaks | Notes |
|------|-----:|----------:|----:|------:|-------|
| `producers_and_directors__e4f664ea` | 0.94 | 0.01 | **+0.93** | 1 | Tightest boundary |
| `private_detectives__a46d5cd2` | 0.80 | 0.03 | **+0.65** | 1 | Clean separation, but see hackability below |
| `film_and_video_editors__75401f7c` | 0.92 | 0.21–0.26 | **+0.65** | 10–11 | Unedited input still earns ~0.2 (form leaks) |
| `film_and_video_editors__e222075d` | 0.43–0.70 | 0.10–0.13 | **+0.33–0.57** | 4 | ⚠ penalty-floored / bimodal, see consistency below |
| `audio_technicians__38889c3b` | 0.55–0.61 | 0.03 | **+0.52–0.58** | 1 | 7–8 unverifiable provenance items |
| `private_detectives__57b2cdf2` | 0.92–1.00 | 0.59–0.64 (unedited) | **+0.3–0.4** | 33–44 | ⚠ unedited null scores >0.5; rubric rewards form |

**Axis 3 (Self-Consistency)**, gpt-mini ×5 at T=0.7:

| Task | Gold mean | Gold stdev | Blank stdev | Unedited stdev |
|------|----------:|-----------:|-------------:|----------------:|
| `producers_and_directors__e4f664ea` | 0.937 | **0.009** | 0.005 | 0.007 |
| `audio_technicians__38889c3b` | 0.539 | **0.028** | 0.024 | 0.056 |
| `private_detectives__57b2cdf2` | 0.963 | **0.024** | 0.000 | 0.066 |
| `private_detectives__a46d5cd2` | 0.795 | **0.045** | 0.000 | 0.013 |
| `film_and_video_editors__75401f7c` | 0.920 | **0.039** | 0.014 | 0.052 |
| `film_and_video_editors__e222075d` | 0.426 | **0.349** ⚠ | 0.013 | — |

`e222075d`'s gold stdev (0.349, bimodal: ~0.7 vs 0.0) directly tracks whether the judge's sampled scene-cut frames happen to catch a visible face — the rubric's −85pt "identifiable faces" penalty against a +61pt ceiling means missing/hitting that one criterion swings the whole floored score. This is a rubric defect, not harness noise, and is now formally flagged as a `consistency/contested` + penalty-floor defect for the iterator (not yet run on this task).

**Axis 2 (Hackability)** — see attack suite and adversarial-agent results above; `rubric_scorecard.tsv` carries `n_attack_leaks`, `worst_attack_score`, and a `reward_hack_flag` column per (task × rubric version).

## What's Next

1. **Fix the iterate loop's gold-score reliability bug** — add a sanity check (reject/retry any version whose gold score is implausibly far from a quick repeat) before trusting any more headlines; the `e222075d` v1 0.000 result is blocking real iterator progress on that task.
2. **Increase self-consistency repeats to 5** — the 2-repeat objective is noisy enough to flip accept/reject decisions (38889c3b accepted in one run, rejected in another). Higher repeat count stabilizes the hill-climb at ~5× cost per evaluate.
3. **Re-run the iterator on `e222075d`** after (1) is fixed, and on the 4 gold-free tasks via a gold-free variant of the objective (e.g. attack-score-only, since margin needs gold).
4. **Push past round 1 with a smarter rewrite prompt** for the tasks stuck at v0 — the rewrite model currently trades single-axis improvements for multi-axis regressions; it needs to see the attack suite's constraint *while* drafting, not just be graded against it after.
5. **`57b2cdf2` likely needs a new criterion, not a rewritten one** — 45/45 attack leaks survived two independent iteration attempts; wording-level fixes cannot close a structural form-over-substance gap.
6. **Grade the remaining rollouts** (8 of 10 tasks have ungraded policy deliverables) so Axis 1/2/3 can be reported on genuine model outputs, not just gold/null/attack anchors.
7. **Cross-model consistency at scale**: still only spot-checked; extend the modality-ladder comparisons across all 6 gold tasks.
8. **Report (Part 4)**: 1,500–2,500 word LaTeX PDF — framing, results, and an honest verdict on which rubrics would be trusted as RL reward. Current leaning: `e4f664ea` yes after v2 improvement (objective +81%); `38889c3b` and `a46d5cd2` lean yes (strong v0, attempted rewrites correctly rejected); `57b2cdf2` and `75401f7c` no (hackable and/or resistant to repair); `e222075d` no (unstable, sign-flipping penalty); the 2 gold-free video tasks lean yes-with-caveats; the 2 gold-free audio tasks split (`4b894ae3` looks strong, `ff85ee58` has filename-leak pattern).
9. **Package deliverables**: finalize all rubric packages and export the agent session transcript.
