"""Scan eval/var/rollouts/ and emit eval/var/rollouts_manifest.json for the viewer.

Layout consumed:
  eval/var/rollouts/<task_folder>/<policy>/rollout.json   (metadata, optional)
  eval/var/rollouts/<task_folder>/<policy>/deliverable/*   (the model's output files)

Each policy that has no rollout.json and no deliverable/ folder is still listed
with status "no_submission" so failed/empty rollouts stay visible.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROLLOUTS_DIR = ROOT / "eval" / "var" / "rollouts"
OUT = ROOT / "eval" / "var" / "rollouts_manifest.json"


def file_entries(folder: Path):
    entries = []
    if not folder.is_dir():
        return entries
    for f in sorted(folder.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            entries.append(
                {
                    "name": f.name,
                    "path": str(f.relative_to(ROOT)),
                    "size_bytes": f.stat().st_size,
                }
            )
    return entries


def main():
    manifest = {}  # task_folder -> [policy records]
    if not ROLLOUTS_DIR.is_dir():
        print(f"No rollouts dir at {ROLLOUTS_DIR}")
        OUT.write_text(json.dumps(manifest, indent=2))
        return

    for task_dir in sorted(p for p in ROLLOUTS_DIR.iterdir() if p.is_dir()):
        task = task_dir.name
        policies = []
        for policy_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            meta = {}
            rj = policy_dir / "rollout.json"
            if rj.is_file():
                try:
                    meta = json.loads(rj.read_text())
                except json.JSONDecodeError:
                    meta = {}

            deliverables = file_entries(policy_dir / "deliverable")
            submitted = bool(meta.get("submitted")) and len(deliverables) > 0
            status = "submitted" if submitted else (
                "no_deliverable" if meta else "no_submission"
            )

            policies.append(
                {
                    "policy": policy_dir.name,
                    "status": status,
                    "submitted": submitted,
                    "rounds": meta.get("rounds"),
                    "notes": meta.get("notes", ""),
                    "steps": meta.get("steps", []),
                    "requested_files": meta.get("requested_files", []),
                    "deliverable_files": deliverables,
                }
            )
        manifest[task] = policies
        n_sub = sum(1 for p in policies if p["submitted"])
        print(f"{task}: {len(policies)} policies ({n_sub} submitted)")

    OUT.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
