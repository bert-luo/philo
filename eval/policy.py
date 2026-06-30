"""The rollout policy: a model attempts the actual GDPval task to produce a real deliverable.

This is the generation counterpart to the judge harness. Where the judge grades a
finished deliverable, the policy *makes* one — so we can score genuine model outputs
(not just gold/null anchors) along the same axes, and feed the iterator real rollouts.

It is a tool-using agent working inside a per-(task, policy) sandbox:

  rollouts/<task>/<policy>/
    refs/         the task's reference/input files, staged under safe names
    <work files>  intermediate + output files the agent creates
    deliverable/  the file(s) the agent finally submits (what the judge grades)

The agent inspects the inputs with the same modality tools the judge has, then
*edits real media* by driving a sandboxed `ffmpeg` (trim / concat / scale / mux for
video and audio tasks) or writes/renders a document (report tasks). ffmpeg is the
only external binary it may call, with no shell and only sandbox-relative paths, so
"attempt real A/V edits" stays contained.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import fitz  # pymupdf

from . import config, llm, media, run_common
from .config import CostTracker
from .deliverable import Bundle, extract_text, kind
from .models import Model
from .tools import _img, _safe_image

MAX_ROUNDS = 16
MAX_TOOL_CALLS = 40
MAX_CALLS_PER_TURN = 6
FFMPEG_TIMEOUT = 600  # a 1080p re-encode of a long reel is slow but legitimate

SYSTEM = """You are an expert practitioner doing real, economically valuable work. \
You are given a task and its input/reference files, and you must PRODUCE the actual \
deliverable the task asks for — not a description of it.

How you work:
- INSPECT the inputs first with the inspection tools. Don't guess what footage,
  audio, or documents contain.
- Then BUILD the real artifact:
  * Video / audio editing tasks: use the `ffmpeg` tool to trim, concatenate, scale,
    re-encode, and mux. Inputs live in the `refs/` folder (reference them as
    `refs/<name>`); write your outputs into the current working directory (e.g.
    `output.mp4`). You may run ffmpeg as many times as needed (build intermediates,
    then a final concat). Honor every hard spec in the prompt (exact resolution,
    codec, max duration, which clip opens/closes, which sound effect goes where).
  * Document / report tasks: use `write_text_file` for plain text, or `render_pdf`
    to produce a PDF deliverable. Write the real content, structured as asked.
- When the deliverable file(s) exist in the working directory, call `finish` with
  their names. Submit ONLY the final deliverable file(s), not intermediates or inputs.

Rules:
- ffmpeg is the only program you can run, via the `ffmpeg` tool. Pass its arguments
  as a list WITHOUT the word "ffmpeg" (e.g. ["-i","refs/a.mp4","-t","5","out.mp4"]).
  Use only sandbox-relative paths; never absolute paths.
- Be decisive and finish within the round budget. A real, slightly-imperfect
  deliverable is far better than none. Don't stop until `finish` is called."""


# --- sandbox + staging ------------------------------------------------------
def _sandbox(task_folder: str, policy_key: str) -> Path:
    return run_common.ROLLOUTS / task_folder / policy_key


def _sanitize(name: str) -> str:
    """A short, shell-safe filename that keeps the original extension."""
    stem = Path(name).name
    ext = Path(stem).suffix.lower()
    base = re.sub(r"[^a-z0-9]+", "_", Path(stem).stem.lower()).strip("_") or "file"
    return f"{base[:48]}{ext}"


def _stage_refs(task_folder: str, workdir: Path) -> dict[str, Path]:
    """Copy every reference file (incl. media inside archives) into refs/ under
    safe names. Returns {staged_name: path}. Uses Bundle's zip expansion."""
    refs = workdir / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    src = Bundle.from_dir("refs", run_common.reference_dir(task_folder), role="reference")
    staged: dict[str, Path] = {}
    used: set[str] = set()
    for logical, real in src.files.items():
        if logical.lower().endswith(".zip"):
            continue  # we stage the expanded members, not the container
        nm = _sanitize(logical)
        i = 1
        while nm in used:  # disambiguate collisions
            p = Path(nm)
            nm = f"{p.stem}_{i}{p.suffix}"
            i += 1
        used.add(nm)
        dst = refs / nm
        if not dst.exists():
            shutil.copy(real, dst)
        staged[nm] = dst
    return staged


def _resolve_input(staged: dict[str, Path], name: str) -> Path:
    """Map an inspection-tool `name` (with or without a refs/ prefix) to a real path."""
    key = Path(name).name
    if key in staged:
        return staged[key]
    raise FileNotFoundError(f"{name} not staged. inputs: {sorted(staged)}")


# --- ffmpeg sandbox ---------------------------------------------------------
def _unsafe(token: str) -> bool:
    return token.startswith(("/", "~")) or ".." in token


def _run_ffmpeg(args: list[str], workdir: Path) -> str:
    bad = [a for a in args if _unsafe(a)]
    if bad:
        return f"REJECTED: absolute paths / '..' not allowed in sandbox: {bad}"
    before = {p.name for p in workdir.iterdir() if p.is_file()}
    cmd = ["ffmpeg", "-nostdin", "-y", *args]
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                              timeout=FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"ffmpeg timed out after {FFMPEG_TIMEOUT}s (command too heavy?)"
    after = {p.name for p in workdir.iterdir() if p.is_file()}
    new = sorted(after - before)
    tail = (proc.stderr or "")[-1200:]
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    return f"ffmpeg {status}. new files: {new or '(none)'}\n--- ffmpeg log (tail) ---\n{tail}"


def _render_pdf(text: str, out: Path) -> None:
    """Paginate plain text / markdown into a simple multi-page PDF."""
    doc = fitz.open()
    chunk, lines = [], text.splitlines() or [""]
    pages: list[str] = []
    for ln in lines:
        chunk.append(ln)
        if len(chunk) >= 52:
            pages.append("\n".join(chunk)); chunk = []
    if chunk:
        pages.append("\n".join(chunk))
    for body in pages or [""]:
        page = doc.new_page()
        rect = fitz.Rect(54, 54, page.rect.width - 54, page.rect.height - 54)
        page.insert_textbox(rect, body, fontsize=10, fontname="helv")
    doc.save(out)


# --- tools advertised to the policy -----------------------------------------
def _tool_schemas(model: Model) -> list[dict]:
    t: list[dict] = [
        {"type": "function", "function": {
            "name": "list_inputs",
            "description": "List the staged input/reference files you can edit from (in refs/).",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "read_text",
            "description": "Extract the text of a PDF / DOCX / TXT input.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"}}, "required": ["name"]}}},
        {"type": "function", "function": {
            "name": "probe_media",
            "description": "ffprobe facts for an input (container, codec, resolution, fps, duration).",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"}}, "required": ["name"]}}},
        {"type": "function", "function": {
            "name": "transcribe_audio",
            "description": "Speech-to-text transcript of an input's audio track.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"}}, "required": ["name"]}}},
        {"type": "function", "function": {
            "name": "ffmpeg",
            "description": ("Run ffmpeg in the sandbox to edit media. Pass args as a list WITHOUT "
                            "the word 'ffmpeg'. Inputs are in refs/ (e.g. 'refs/clip.mp4'); write "
                            "outputs to the working dir. Use only relative paths."),
            "parameters": {"type": "object", "properties": {
                "args": {"type": "array", "items": {"type": "string"}},
                "purpose": {"type": "string", "description": "<=12 words: what this step does"}},
                "required": ["args"]}}},
        {"type": "function", "function": {
            "name": "write_text_file",
            "description": "Write a plain-text deliverable file in the working dir.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"}, "content": {"type": "string"}},
                "required": ["name", "content"]}}},
        {"type": "function", "function": {
            "name": "render_pdf",
            "description": "Render text/markdown into a multi-page PDF deliverable in the working dir.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"}, "content": {"type": "string"}},
                "required": ["name", "content"]}}},
        {"type": "function", "function": {
            "name": "list_workdir",
            "description": "List files you have produced so far in the working dir.",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "finish",
            "description": "Submit the final deliverable file(s) (names in the working dir). Call once when done.",
            "parameters": {"type": "object", "properties": {
                "deliverable_files": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string", "description": "<=20 words on what you produced"}},
                "required": ["deliverable_files"]}}},
    ]
    if model.supports("image"):
        t.append({"type": "function", "function": {
            "name": "view_video_frames",
            "description": ("See frames from an input video: one per scene cut by default, or pass "
                            "start_seconds/end_seconds to sample a window."),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"}, "count": {"type": "integer"},
                "start_seconds": {"type": "number"}, "end_seconds": {"type": "number"}},
                "required": ["name"]}}})
        t.append({"type": "function", "function": {
            "name": "view_image",
            "description": "See an image input.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"}}, "required": ["name"]}}})
    if model.supports("audio"):
        t.append({"type": "function", "function": {
            "name": "listen_audio",
            "description": "Listen to a clip of an audio/video input (judge music style, mix, tone).",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"}, "start_seconds": {"type": "number"},
                "duration_seconds": {"type": "number"}}, "required": ["name"]}}})
    return t


def _dispatch(name: str, args: dict, staged: dict[str, Path], workdir: Path):
    """Returns (text, media_parts, finished_files_or_None)."""
    try:
        if name == "list_inputs":
            lines = [f"- refs/{n}  [{kind(n)}, {p.stat().st_size // 1024}KB]"
                     for n, p in sorted(staged.items())]
            return "Input files (in refs/):\n" + "\n".join(lines), [], None
        if name == "read_text":
            return extract_text(_resolve_input(staged, args["name"]))[:20000], [], None
        if name == "probe_media":
            return json.dumps(media.probe(_resolve_input(staged, args["name"])), indent=2), [], None
        if name == "transcribe_audio":
            return "Transcript:\n" + media.transcribe(_resolve_input(staged, args["name"]))[:20000], [], None
        if name == "view_video_frames":
            path = _resolve_input(staged, args["name"])
            count = max(1, min(14, int(args.get("count", 10))))
            s, e = args.get("start_seconds"), args.get("end_seconds")
            if s is None and e is None:
                frames, how = media.scene_frames(path, max_frames=count), "scene-cut"
            else:
                frames, how = media.video_frames(path, n=count, start=s, end=e), f"{s}-{e}s"
            parts = []
            for fp in frames:
                b64, mime = _safe_image(fp)
                if b64 is not None:
                    parts.append(_img(b64, mime))
            if not parts:
                return "No readable frames could be extracted.", [], None
            return f"{len(parts)} {how} frames attached.", parts, None
        if name == "view_image":
            path = _resolve_input(staged, args["name"])
            b64, mime = _safe_image(path)
            if b64 is None:
                return f"ERROR: could not decode image ({mime})", [], None
            return "Image attached.", [_img(b64, mime)], None
        if name == "listen_audio":
            path = _resolve_input(staged, args["name"])
            s = float(args.get("start_seconds", 0.0))
            d = min(90.0, float(args.get("duration_seconds", 60.0)))
            clip = media.audio_clip_mp3(path, s, d)
            return (f"Audio clip {s:.0f}-{s + d:.0f}s attached.",
                    [{"type": "input_audio", "input_audio": {"data": clip, "format": "mp3"}}], None)
        if name == "ffmpeg":
            return _run_ffmpeg([str(a) for a in args.get("args", [])], workdir), [], None
        if name == "write_text_file":
            out = workdir / Path(args["name"]).name
            out.write_text(args.get("content", ""))
            return f"wrote {out.name} ({out.stat().st_size} bytes)", [], None
        if name == "render_pdf":
            out = workdir / Path(args["name"]).name
            _render_pdf(args.get("content", ""), out)
            return f"rendered {out.name} ({out.stat().st_size // 1024}KB)", [], None
        if name == "list_workdir":
            files = [p.name for p in sorted(workdir.iterdir())
                     if p.is_file() and p.name != "rollout.json"]
            return "Working-dir files: " + (", ".join(files) or "(none)"), [], None
        if name == "finish":
            return "finishing", [], list(args.get("deliverable_files", []))
        return f"unknown tool {name}", [], None
    except FileNotFoundError as e:
        return f"ERROR: {e}", [], None
    except Exception as e:
        return f"ERROR running {name}: {e}", [], None


# --- the rollout loop -------------------------------------------------------
def rollout(model: Model, task_folder: str, verbose: bool = False,
            force: bool = False) -> dict:
    """Generate one deliverable for `task_folder` with `model`. Returns a result dict.

    Idempotent: if a deliverable already exists for this (task, policy) it is reused
    (skipped) unless `force=True`, so a batch can be resumed after an interruption."""
    task = run_common.load_task(task_folder)
    workdir = _sandbox(task_folder, model.key)
    existing = workdir / "rollout.json"
    if not force and (workdir / "deliverable").exists() and \
            any((workdir / "deliverable").iterdir()) and existing.exists():
        res = json.loads(existing.read_text())
        print(f"  [{model.key}] {task_folder}: cached -> {res.get('deliverable_files')}")
        return res
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    staged = _stage_refs(task_folder, workdir)

    listing = "\n".join(f"- refs/{n}  [{kind(n)}]" for n in sorted(staged)) or "(none)"
    user = (
        f"TASK:\n{task['prompt'].strip()}\n\n"
        f"INPUT FILES staged in refs/ (reference them as refs/<name>):\n{listing}\n\n"
        f"Produce the deliverable, then call finish with its file name(s)."
    )
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
    schemas = _tool_schemas(model)

    seen: dict[str, str] = {}
    rounds = tool_calls = 0
    delivered: list[str] | None = None
    notes = ""
    steps: list[str] = []
    nudged = False

    def _has_output() -> bool:
        return any(p.is_file() and p.name != "rollout.json" for p in workdir.iterdir())

    while rounds < MAX_ROUNDS and delivered is None:
        rounds += 1
        # Running low on rounds with nothing produced: push the model to act NOW.
        if not nudged and rounds >= MAX_ROUNDS - 4 and not _has_output():
            messages.append({"role": "user", "content":
                "You have not produced any output file yet and are almost out of steps. "
                "IMMEDIATELY create the deliverable now with ffmpeg / write_text_file / "
                "render_pdf, then call finish. Do not inspect further."})
            nudged = True
        force = rounds == MAX_ROUNDS or tool_calls >= MAX_TOOL_CALLS
        msg = llm.chat(model, messages, tools=schemas,
                       tool_choice=({"type": "function", "function": {"name": "finish"}}
                                    if force else "auto"),
                       temperature=0.4, max_tokens=8000)
        tcs = msg.tool_calls or []
        replay, parsed_args = llm.assistant_tool_calls(tcs)
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": replay})
        if not tcs:
            messages.append({"role": "user",
                             "content": "Keep going: build the deliverable with the tools, then call finish."})
            continue

        media_followups: list[dict] = []
        for i, tc in enumerate(tcs):
            args = parsed_args.get(tc.id, {})
            fn = tc.function.name
            if fn != "finish" and i >= MAX_CALLS_PER_TURN:
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": "skipped: too many tool calls in one turn"})
                continue
            if fn not in ("finish", "ffmpeg", "write_text_file", "render_pdf"):
                tool_calls += 1
                sig = fn + "|" + json.dumps(args, sort_keys=True)
                if sig in seen:
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": "(already provided) " + seen[sig][:200]})
                    continue
            else:
                tool_calls += 1
                sig = None

            text, parts, fin = _dispatch(fn, args, staged, workdir)
            if sig is not None:
                seen[sig] = text
            if fn in ("ffmpeg", "write_text_file", "render_pdf"):
                steps.append(f"{fn}: {args.get('purpose', '') or args.get('name', '')}".strip())
            if verbose:
                print(f"  [{model.key}] {fn}({str(args)[:80]}) -> {text[:80]!r}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
            media_followups.extend(parts)
            if fin is not None:
                delivered, notes = fin, args.get("notes", "")

        if media_followups and delivered is None:
            messages.append({"role": "user", "content":
                             [{"type": "text", "text": "Attached media from your tool call(s):"},
                              *media_followups]})

    # collect the submitted deliverable files
    out = workdir / "deliverable"
    out.mkdir(exist_ok=True)
    produced: list[str] = []
    for nm in (delivered or []):
        src = workdir / Path(nm).name
        if src.exists() and src.is_file():
            shutil.copy(src, out / src.name)
            produced.append(src.name)

    result = {
        "task": task_folder, "policy": model.key, "rounds": rounds,
        "submitted": delivered is not None, "deliverable_files": produced,
        "requested_files": delivered or [], "notes": notes, "steps": steps,
    }
    (workdir / "rollout.json").write_text(json.dumps(result, indent=2))
    status = "ok" if produced else "NO DELIVERABLE"
    print(f"  [{model.key}] {task_folder}: {status} -> {produced} ({rounds} rounds)")
    return result
