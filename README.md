# Philo take-home — GDPval data + viewer

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas pyarrow requests
```

## Download the 10 target tasks

Fetches `openai/gdpval` from Hugging Face, filters to the 10 task folders
specified in `instructions.md` (Part 1), and writes each task's prompt,
rubric, reference files, and gold deliverable files into `data/<task_folder>/`.

```bash
python scripts/download_data.py
```

Re-running is safe/idempotent — it skips files that already exist on disk.

Output layout per task:

```
data/<occupation>__<task_short_id>/
  task.json           # prompt, sector, occupation, file lists
  rubric.json          # structured rubric items (score, criterion, tags, ids)
  rubric_pretty.txt     # human-readable rubric text as published
  reference_files/      # input files given to the worker
  deliverable_files/    # the gold deliverable (when released for that task)
data/manifest.json       # all 10 tasks' metadata in one file (used by the viewer)
```

## Browse the data in a UI

```bash
# Optional: index our custom model rollouts (eval/var/rollouts/) for the viewer.
python scripts/build_rollouts_manifest.py

python scripts/serve.py
```

Opens `http://localhost:8765/viewer/index.html` — a sidebar lists the 10
tasks; each task has tabs for:

- **Prompt** — the task instructions
- **Rubric** — scored criteria table (+ toggle for the raw rubric text)
- **Reference Files** — the inputs given to the worker
- **Gold Deliverable** — the expert reference output
- **Rollouts** — our custom model outputs per policy (from `eval/var/rollouts/`):
  each policy's submit status, round count, notes, the steps it took, and the
  deliverable files it produced

Files preview inline for video/audio/image/PDF/text and fall back to a
download link for other types (zip, docx, psd). The server supports HTTP
Range requests so large video files can be scrubbed/seeked.

The Rollouts tab reads `eval/var/rollouts_manifest.json`. Re-run
`python scripts/build_rollouts_manifest.py` whenever `eval/var/rollouts/` changes.
