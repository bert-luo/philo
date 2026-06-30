CONFIDENTIAL - DO NOT DISTRIBUTE
PHILO LABS
Research Engineer / Scientist Take-Home
Track R: Evaluating Rubrics for Multimodal Agentic RL
~4 hours | 2 calendar days | Jun 2026
Context
Math has formal proof. Code has unit tests. They are trustworthy because people built them on
purpose.
For knowledge work, creative work, and most tasks that matter, we have no such thing. We write
rubrics instead: short lists of criteria, each scored by a person or by a model acting as a judge.
The problem: a rubric can look like it grades quality while it really grades form. A memo can pass
every line and still say nothing. A strong analysis can fail lines that have nothing to do with whether it
is strong.
On a leaderboard, that is a small problem. As an RL reward, it compounds. The policy learns to
satisfy the checklist without doing the work, and the rubric quietly teaches the model the wrong thing.
So before we trust a rubric as training signal, we have to grade the rubric itself. This take-home asks
you to build that grader.
Why GDPval
You will work from GDPval (Patwardhan et al., 2025), OpenAI's open benchmark of real,
economically valuable knowledge work. Dataset: https://huggingface.co/datasets/openai/gdpval
•
•
•
•
Scale. 1,320 tasks across 44 occupations and 9 sectors. A gold subset of 220 is public.
The right shape. Tasks are multimodal. Many are exactly our kind: video understanding
(detectives, private investigators), video and image editing (film directors, VFX artists), plus
documents, slides, and spreadsheets.
What ships with each task. A prompt, reference files, a gold deliverable made by an
experienced professional, and a grading rubric.
Why that helps. The gold deliverable is a free oracle. It is exactly what you need to test
whether a rubric is calibrated.
The Task
Build two things and run them end to end on a slice of GDPval:
•
a rubric evaluator that scores how good a rubric is along the axes below, and
•
an automatic iterator that rewrites a rubric to raise those scores.
Use your coding agent (Claude Code, Codex, Gemini CLI, OpenCode, or similar) heavily throughout.
Part 1: Get familiar with 10 tasks
We have chosen the 10 tasks for you. They are all video-centric on purpose. Video is where rubrics
are weakest, where we work, and the part candidates tend to avoid. Use these exact tasks. Do not
swap in document or slide tasks.
Occupation GDPval task folder
Film and video editors film_and_video_editors__75401f7c
Film and video editors film_and_video_editors__a941b6d8
Film and video editors film_and_video_editors__c94452e4
Film and video editors film_and_video_editors__e222075d
Audio and video technicians audio_and_video_technicians__38889c3b
Audio and video technicians audio_and_video_technicians__4b894ae3
Audio and video technicians audio_and_video_technicians__ff85ee58
Private detectives and
investigators private_detectives_and_investigators__57b2cdf2
Private detectives and
investigators private_detectives_and_investigators__a46d5cd2
Producers and directors producers_and_directors__e4f664ea
The slice spans both capability families we care about:
•
•
Editing. Film and video editors, audio and video technicians, and producers and directors. The
output is an edited or produced video.
Video understanding. Private detectives and investigators. The work is to review footage and
find what matters.
Part 2: The rubric scorecard
This is the heart of the take-home, and the part we will read most closely. For each rubric, score it
along the axes below.
Treat them as a standing test suite. A rubric that fails any one is suspect, no matter how reasonable
it looks on the page. The list is not complete on purpose: finding the axes we have not named is part
of the work.
Axis 1: Boundary calibration
The cheap checks, run first. Two sanity anchors:
•
Gold scores near 1. Run the rubric on the gold deliverable.
•
Null scores near 0. Run it on a do-nothing output: an empty file, the unedited input, or a black
frame.
Axis 2: Reward hackability
This is the axis that matters most for RL. A rubric is hackable if a policy can raise its score without
improving the work. The policy will optimize for exactly what you reward, including the parts you did
not mean.
The classic trap is a criterion that only checks that something is present. Take a captioning rubric
with the line "the caption is visible."
A policy can satisfy it with a huge caption that covers the speaker's face. Visible, yes, but worse, not
better. The line states a presence condition with no matching absence condition.
The fix is to pair them: "the caption is visible AND does not cover the main subject." Presence plus
absence closes the gap.
That is one pattern. There are more, and we left them for you to find.
Axis 3: Consistency
A rubric is only useful if it returns the same verdict on the same work.
•
Self-consistency: run the same rubric on the same output several times with the same judge.
Do the scores reproduce, or wander?
•
Cross-model consistency: score the same output with two or more judge models. Do they
agree?
When two judges disagree, ask why. If one judge is simply more capable, fine. If a criterion is so
vague that two models read it differently, that is a rubric defect: fix the wording, do not blame the
judge.
And more (open on purpose)
We expect you to add axes we have not named. Strong candidates:
•
Discrimination: does it separate excellent work from merely adequate, not just good from
broken?
•
Coverage: does it grade what the task is actually about, or only the easy-to-name surface?
•
Saturation: how does it behave once every strong model passes it?
Part 3: The iterator
Build a loop that improves a rubric automatically:
•
evaluate the rubric on the axes above,
•
diagnose which axis it fails, and why,
•
rewrite the rubric to fix that failure,
•
evaluate again.
Run it on your 10 tasks. For each, report before and after: did hackability drop, did cross-model
agreement rise where it should, did the gold move toward 1 and the null toward 0?
Be frank about what moved and what did not. A rubric that resists improvement is itself a finding.
Part 4: Report
1,500-2,500 words in LaTeX, compiled to PDF with your coding agent. Cover:
•
the framing of your scorecard and your design choices,
•
what you learned running it on GDPval,
•
most importantly, an honest account of which rubrics you would trust as RL reward and which
you would not, and which axes you believe generalize to rubrics you never saw.
Deliverables
Deliverable Details
Report (PDF) LaTeX compiled to PDF via your coding agent. 1,500-2,500 words.
Code GitHub repo (preferred) or zip. Clean README with run instructions
and the agent prompts you used.
rubric_scorecard.tsv
Auto-generated by the pipeline. One row per (task x rubric version),
with a column for each axis (gold score, null score, hackability labels,
self-consistency variance, cross-model agreement) and a column
flagging suspected reward hacking. Reproducible with a single
command.
10 rubric packages For each task: the original rubric, the rewritten rubric, the test cases
used, and the hackability analysis. Released as one folder per task.
Agent session
Exported coding agent session. For Claude Code, run /export, or use
claude-code-transcripts to export as HTML. For other agents, export
the full session log per your tool's docs.
Evaluation
•
Systematic thinking (40%): a coherent framework, first-principles reasoning about what
makes a rubric trustworthy as training signal, and clear thinking about reward hacking and
about what a rubric fails to cover.
•
•
•
Rubric-evaluation quality (25%): axes that are precise, measurable, implementable from your
spec alone, and honest about which ones would survive being used as reward.
Implementation (20%): clean, modular, end to end. The evaluator and iterator run with a single
command, and the rubric packages are self-contained.
Agent fluency (15%): effective use of coding agents across the full workflow.
Logistics
•
•
•
•
Time: ~4 focused hours, 2 calendar days from receipt.
Resources: anything publicly available. The GDPval gold subset is on Hugging Face. API and
compute costs reimbursed up to $200 (note your spend).
Questions: email us. Good questions are a positive signal.
Submit: email a GitHub link or a zip containing all deliverables.
References
•
•
•
GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks.
Patwardhan et al., 2025 (OpenAI). The benchmark and gold task subset you will work from.
arXiv:2510.04374
Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains. Gunjal et al.,
2025. Uses rubric feedback as an on-policy RL reward and studies how to turn rubric scores
into a reward. Precedent for treating rubrics as the reward signal. arXiv:2507.17746
AdvancedIF and RIFL: Rubric-Based Benchmarking and RL for Instruction Following. He
et al., 2025 (Meta Superintelligence Labs). An expert-curated rubric benchmark plus a pipeline
of rubric generation, a rubric verifier, and reward shaping. Precedent for the generator, verifier,
and updater shape. arXiv:2511.10507
This document is confidential. Do not share or post publicly.
