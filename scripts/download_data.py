"""Download the 10 GDPval tasks specified in instructions.md (Part 1).

Fetches the single parquet split of openai/gdpval, filters down to the
10 target task folders, and writes each task's metadata + rubric +
reference/deliverable files into data/<task_folder>/.
"""
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

PARQUET_URL = "https://huggingface.co/datasets/openai/gdpval/resolve/main/data/train-00000-of-00001.parquet"

TARGET_FOLDERS = [
    "film_and_video_editors__75401f7c",
    "film_and_video_editors__a941b6d8",
    "film_and_video_editors__c94452e4",
    "film_and_video_editors__e222075d",
    "audio_and_video_technicians__38889c3b",
    "audio_and_video_technicians__4b894ae3",
    "audio_and_video_technicians__ff85ee58",
    "private_detectives_and_investigators__57b2cdf2",
    "private_detectives_and_investigators__a46d5cd2",
    "producers_and_directors__e4f664ea",
]


def slugify(occupation: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", occupation.lower()).strip("_")


def task_folder(row) -> str:
    short_id = row["task_id"].split("-")[0]
    return f"{slugify(row['occupation'])}__{short_id}"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists) {dest.relative_to(ROOT)}")
        return
    print(f"  downloading {url} -> {dest.relative_to(ROOT)}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def main() -> None:
    print(f"Fetching parquet from {PARQUET_URL} ...")
    resp = requests.get(PARQUET_URL, timeout=120)
    resp.raise_for_status()
    df = pd.read_parquet(io.BytesIO(resp.content))
    print(f"Loaded {len(df)} total GDPval tasks")

    df["folder"] = df.apply(task_folder, axis=1)
    found = df[df["folder"].isin(TARGET_FOLDERS)]
    print(f"Matched {len(found)}/{len(TARGET_FOLDERS)} target tasks")

    missing = set(TARGET_FOLDERS) - set(found["folder"])
    if missing:
        print(f"WARNING: could not find folders: {sorted(missing)}", file=sys.stderr)

    DATA_DIR.mkdir(exist_ok=True)
    manifest = []

    for _, row in found.iterrows():
        folder = row["folder"]
        task_dir = DATA_DIR / folder
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {folder} ({row['task_id']}) ===")

        rubric_items = json.loads(row["rubric_json"]) if row["rubric_json"] else []

        def local_file_entries(rel_paths, urls, subdir):
            entries = []
            for rel_path, url in zip(rel_paths, urls):
                name = Path(rel_path).name
                local = task_dir / subdir / name
                download_file(url, local)
                entries.append(
                    {
                        "name": name,
                        "path": f"data/{folder}/{subdir}/{name}",
                        "size_bytes": local.stat().st_size if local.exists() else None,
                    }
                )
            return entries

        reference_entries = local_file_entries(
            row["reference_files"], row["reference_file_urls"], "reference_files"
        )
        deliverable_entries = local_file_entries(
            row["deliverable_files"], row["deliverable_file_urls"], "deliverable_files"
        )

        meta = {
            "task_id": row["task_id"],
            "folder": folder,
            "sector": row["sector"],
            "occupation": row["occupation"],
            "prompt": row["prompt"],
            "reference_files": reference_entries,
            "deliverable_files": deliverable_entries,
            "num_rubric_items": len(rubric_items),
            "rubric_max_score": sum(item.get("score") or 0 for item in rubric_items),
        }
        (task_dir / "task.json").write_text(json.dumps(meta, indent=2))
        (task_dir / "rubric.json").write_text(json.dumps(rubric_items, indent=2))
        (task_dir / "rubric_pretty.txt").write_text(row["rubric_pretty"] or "")

        manifest.append(meta)

    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest for {len(manifest)} tasks to {DATA_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
