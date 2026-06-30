"""Generate sensible 'do-nothing' deliverables for Axis 1 (boundary calibration).

For each task with a gold deliverable we synthesize null variants that a lazy or
degenerate policy might emit. A well-calibrated rubric should score these near 0:

  blank           a structurally-valid but empty artifact of the right TYPE
                  (blank PDF / black silent video / zip of silence)
  unedited_input  the raw input handed back with no work done
                  (the unrevised source doc as a PDF; the source audio re-zipped;
                   a black video carrying only the source audio track)

These are task-aware but generated from generic rules keyed on the gold file
type, so they stay sensible without hand-coding every task.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import fitz

from . import config
from .deliverable import extract_text, kind


def _run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _blank_pdf(out: Path):
    doc = fitz.open()
    doc.new_page()  # one empty page
    doc.save(out)


def _text_pdf(text: str, out: Path):
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
    page.insert_textbox(rect, text[:4000], fontsize=9)
    doc.save(out)


def _black_video(out: Path, dur=5, audio: Path | None = None):
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=640x360:d={dur}"]
    if audio is not None:
        cmd += ["-i", str(audio), "-shortest"]
    else:
        cmd += ["-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono", "-t", str(dur)]
    cmd += ["-pix_fmt", "yuv420p", str(out)]
    _run(cmd)


def _silent_wav(out: Path, dur=5):
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
          "-t", str(dur), str(out)])


def _zip_of(files: list[Path], out: Path):
    with zipfile.ZipFile(out, "w") as z:
        for f in files:
            z.write(f, arcname=f.name)


def _task(task_folder: str) -> dict:
    return json.loads((config.DATA / task_folder / "task.json").read_text())


def build_nulls(task_folder: str) -> dict[str, Path]:
    """Create null bundles on disk; return {variant_name: directory}."""
    meta = _task(task_folder)
    dels = meta.get("deliverable_files") or []
    if not dels:
        return {}  # no gold -> Axis 1 not applicable

    gold_type = kind(dels[0]["name"])
    refs = [config.DATA / task_folder / "reference_files" / Path(r["name"]).name
            for r in meta.get("reference_files", [])]
    refs = [r for r in refs if r.exists()]

    base = config.NULLS / task_folder
    variants: dict[str, Path] = {}

    def mkdir(name: str) -> Path:
        d = base / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        return d

    if gold_type == "document":  # gold is a PDF
        d = mkdir("blank"); _blank_pdf(d / "blank.pdf"); variants["blank"] = d
        src = next((r for r in refs if r.suffix.lower() in (".docx", ".pdf", ".txt")), None)
        if src:
            d = mkdir("unedited_input")
            _text_pdf(extract_text(src), d / "unedited_input.pdf")
            variants["unedited_input"] = d

    elif gold_type == "video":  # gold is an mp4
        d = mkdir("blank"); _black_video(d / "black.mp4"); variants["blank"] = d
        aud = next((r for r in refs if r.suffix.lower() in (".mp3", ".wav", ".m4a")), None)
        vid = next((r for r in refs if r.suffix.lower() in (".mp4", ".mov")), None)
        if aud:
            d = mkdir("unedited_input")
            _black_video(d / "raw_audio_over_black.mp4", dur=10, audio=aud)
            variants["unedited_input"] = d
        elif vid:
            d = mkdir("unedited_input")
            shutil.copy(vid, d / Path(vid).name)
            variants["unedited_input"] = d

    elif gold_type == "archive":  # gold is a zip (e.g. audio stems)
        d = mkdir("blank")
        _silent_wav(d / "silence.wav")
        _zip_of([d / "silence.wav"], d / "blank_stems.zip")
        (d / "silence.wav").unlink()
        variants["blank"] = d
        src_audio = [r for r in refs if r.suffix.lower() in (".wav", ".mp3")]
        if src_audio:
            d = mkdir("unedited_input")
            _zip_of(src_audio, d / "raw_input_stems.zip")
            variants["unedited_input"] = d

    return variants


if __name__ == "__main__":
    import sys
    for tf in sys.argv[1:] or [p.name for p in config.DATA.iterdir() if p.is_dir()]:
        v = build_nulls(tf)
        print(f"{tf}: {list(v) or 'no gold -> skipped'}")
