"""Agent tool schemas + dispatch, gated by the judge model's modalities.

A tool returns (text_summary, media_parts):
  * text_summary  goes back as the tool-role message (OpenAI tool messages are text only)
  * media_parts   if non-empty, the harness appends them as a follow-up user message,
                  which is how images/audio actually reach a multimodal model.
"""
from __future__ import annotations

from pathlib import Path

from . import deliverable as D
from . import media
from .models import Model


def tool_schemas(model: Model) -> list[dict]:
    """The toolbox advertised to this model (only what its modalities can use)."""
    t: list[dict] = [
        {"type": "function", "function": {
            "name": "list_files",
            "description": "List every file in the deliverable (and media inside archives).",
            "parameters": {"type": "object", "properties": {}},
        }},
        {"type": "function", "function": {
            "name": "read_text",
            "description": "Extract the text of a PDF / DOCX / TXT file.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"}}, "required": ["name"]},
        }},
        {"type": "function", "function": {
            "name": "transcribe_audio",
            "description": "Speech-to-text transcript of an audio or video file's audio track.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"}}, "required": ["name"]},
        }},
        {"type": "function", "function": {
            "name": "file_facts",
            "description": ("Deterministic facts about a file: exact extension/type, size, "
                            "PDF page count, and word count. Use this for objective criteria "
                            "(is it a PDF? is it <=2 pages? word count) instead of guessing."),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"}}, "required": ["name"]},
        }},
        {"type": "function", "function": {
            "name": "probe_media",
            "description": ("Ground-truth technical facts (ffprobe) for an audio/video file: "
                            "container, codec, exact resolution, fps, duration in seconds, "
                            "audio channels/sample rate AND bit depth (pcm_s24le = 24-bit). "
                            "Use for codec/resolution/runtime/sample-rate/bit-depth criteria."),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"}}, "required": ["name"]},
        }},
        {"type": "function", "function": {
            "name": "analyze_audio",
            "description": ("Measured audio/music facts (DSP) you must NOT guess: integrated "
                            "loudness (LUFS), true peak (dBFS), tempo (BPM), estimated musical "
                            "key, and harmonic fraction. Use for tempo/key/loudness/fidelity/"
                            "instrumental criteria. Numbers are estimates; combine with listening."),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"}}, "required": ["name"]},
        }},
        {"type": "function", "function": {
            "name": "compare_audio_sync",
            "description": ("Measure timing alignment between a deliverable audio file and a "
                            "REFERENCE track (e.g. a drum reference): start offset (ms), tail "
                            "offset (ms), and end-to-end drift (ms). Use for 'synchronized to' "
                            "and 'drift <= N ms' criteria."),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "deliverable audio file"},
                "reference_name": {"type": "string", "description": "reference track to align against"}},
                "required": ["name", "reference_name"]},
        }},
        {"type": "function", "function": {
            "name": "list_reference_files",
            "description": ("List the INPUT/reference files the worker was given. These are NOT "
                            "the deliverable and are never gradeable — consult them only for "
                            "comparison criteria (e.g. 'matches the script')."),
            "parameters": {"type": "object", "properties": {}},
        }},
        {"type": "function", "function": {
            "name": "read_reference",
            "description": "Read the text of a reference/input file (for comparison only, never as the graded output).",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"}}, "required": ["name"]},
        }},
    ]
    if model.supports("image"):
        t += [
            {"type": "function", "function": {
                "name": "view_pdf_page",
                "description": "See a rendered image of one PDF page (0-indexed). Use for layout/formatting.",
                "parameters": {"type": "object", "properties": {
                    "name": {"type": "string"}, "page": {"type": "integer"}},
                    "required": ["name", "page"]},
            }},
            {"type": "function", "function": {
                "name": "view_image",
                "description": "See an image file from the deliverable.",
                "parameters": {"type": "object", "properties": {
                    "name": {"type": "string"}}, "required": ["name"]},
            }},
            {"type": "function", "function": {
                "name": "view_reference",
                "description": ("See a reference/input image or PDF page (for comparison only, "
                                "e.g. do the deliverable's photos match the provided ones)."),
                "parameters": {"type": "object", "properties": {
                    "name": {"type": "string"}, "page": {"type": "integer"}},
                    "required": ["name"]},
            }},
            {"type": "function", "function": {
                "name": "view_video_frames",
                "description": ("See frames from a video. By default returns one frame per "
                                "detected scene cut (distinct shots) — best for 'includes a "
                                "shot of X' and identifiable-face checks. Pass start/end "
                                "seconds to sample a specific window uniformly (e.g. to read "
                                "an on-screen text card)."),
                "parameters": {"type": "object", "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer", "description": "max frames, 1-14"},
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"}},
                    "required": ["name"]},
            }},
        ]
    if model.supports("audio"):
        t.append({"type": "function", "function": {
            "name": "listen_audio",
            "description": "Natively listen to a clip of an audio/video file (judge timbre, mix, music, tone).",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"},
                "start_seconds": {"type": "number"},
                "duration_seconds": {"type": "number", "description": "<=90"}},
                "required": ["name"]},
        }})

    t.append({"type": "function", "function": {
        "name": "submit_scorecard",
        "description": "Submit final per-item verdicts. Call exactly once when done inspecting.",
        "parameters": {"type": "object", "properties": {
            "scores": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string", "description": "rubric_item_id"},
                "verdict": {"type": "string",
                    "enum": ["met", "unmet", "partial", "unverifiable"],
                    "description": (
                        "met = criterion clearly true (full points); unmet = clearly false "
                        "(0); partial = genuinely in between (give `awarded`); unverifiable = "
                        "CANNOT be checked from the deliverable or any tool, e.g. external "
                        "provenance like 'footage is royalty-free' or 'music from a stock "
                        "provider' with no source log in the deliverable.")},
                "awarded": {"type": "number",
                    "description": "ONLY for verdict=partial: fraction of points, 0.0-1.0"},
                "note": {"type": "string", "description": "<=14 words of justification"}},
                "required": ["id", "verdict"]}}},
            "required": ["scores"]},
    }})
    return t


def dispatch(name: str, args: dict, bundle: D.Bundle) -> tuple[str, list[dict]]:
    try:
        return _dispatch(name, args, bundle)
    except FileNotFoundError as e:
        return f"ERROR: {e}", []
    except Exception as e:  # never let a tool crash the loop
        return f"ERROR running {name}: {e}", []


def _dispatch(name: str, args: dict, bundle: D.Bundle) -> tuple[str, list[dict]]:
    DEL = "deliverable"  # grading tools must only ever touch the deliverable

    if name == "list_files":
        lines = [f"- {f['name']} [{f['type']}, {f['size_kb']}KB]"
                 for f in bundle.listing(role=DEL)]
        return "Deliverable files (the graded output):\n" + "\n".join(lines), []

    if name == "list_reference_files":
        refs = bundle.listing(role="reference")
        if not refs:
            return "No reference/input files for this task.", []
        lines = [f"- {f['name']} [{f['type']}, {f['size_kb']}KB]" for f in refs]
        return "Reference INPUT files (for comparison only, NOT graded):\n" + "\n".join(lines), []

    if name == "read_text":
        text = D.extract_text(bundle.resolve(args["name"], DEL))
        return text[:20000], []

    if name == "read_reference":
        text = D.extract_text(bundle.resolve(args["name"], "reference"))
        return "[REFERENCE INPUT — compare only, do not grade as output]\n" + text[:20000], []

    if name == "transcribe_audio":
        path = bundle.resolve(args["name"], DEL)
        return "Transcript:\n" + media.transcribe(path)[:20000], []

    if name == "file_facts":
        import json as _json
        return _json.dumps(D.file_facts(bundle.resolve(args["name"], DEL)), indent=2), []

    if name == "probe_media":
        import json as _json
        return _json.dumps(media.probe(bundle.resolve(args["name"], DEL)), indent=2), []

    if name == "analyze_audio":
        import json as _json
        return _json.dumps(media.analyze_audio(bundle.resolve(args["name"], DEL)), indent=2), []

    if name == "compare_audio_sync":
        import json as _json
        d = bundle.resolve(args["name"], DEL)
        ref = bundle.resolve(args["reference_name"], "reference")
        return _json.dumps(media.audio_sync(d, ref), indent=2), []

    if name == "view_pdf_page":
        path = bundle.resolve(args["name"], DEL)
        n = D.pdf_page_count(path)
        b64 = D.pdf_page_png(path, int(args.get("page", 0)))
        return (f"Rendered page {args.get('page', 0)} of {n} (attached).",
                [_img(b64, "image/png")])

    if name == "view_image":
        path = bundle.resolve(args["name"], DEL)
        b64, mime = _safe_image(path)
        if b64 is None:
            return (f"ERROR: could not decode image {args['name']} ({mime})", [])
        return ("Image attached.", [_img(b64, mime)])

    if name == "view_reference":
        path = bundle.resolve(args["name"], "reference")
        if path.suffix.lower() == ".pdf":
            b64 = D.pdf_page_png(path, int(args.get("page", 0)))
            return ("[REFERENCE INPUT] page attached.", [_img(b64, "image/png")])
        b64, mime = _safe_image(path)
        if b64 is None:
            return (f"ERROR: could not decode reference image {args['name']} ({mime})", [])
        return ("[REFERENCE INPUT] image attached.", [_img(b64, mime)])

    if name == "view_video_frames":
        path = bundle.resolve(args["name"], DEL)
        count = max(1, min(14, int(args.get("count", 10))))
        start, end = args.get("start_seconds"), args.get("end_seconds")
        if start is None and end is None:
            frame_paths = media.scene_frames(path, max_frames=count)
            how = "scene-cut"
        else:
            frame_paths = media.video_frames(path, n=count, start=start, end=end)
            how = f"window {start}-{end}s"
        parts = []
        for fp in frame_paths:
            b64, mime = _safe_image(fp)
            if b64 is not None:
                parts.append(_img(b64, mime))
        if not parts:
            return ("No readable frames could be extracted from this video.", [])
        return (f"{len(parts)} {how} frames attached.", parts)

    if name == "listen_audio":
        path = bundle.resolve(args["name"], "deliverable")
        start = float(args.get("start_seconds", 0.0))
        dur = min(90.0, float(args.get("duration_seconds", 60.0)))
        clip = media.audio_clip_mp3(path, start, dur)
        return (f"Audio clip {start:.0f}-{start + dur:.0f}s attached.",
                [{"type": "input_audio", "input_audio": {"data": clip, "format": "mp3"}}])

    return f"unknown tool {name}", []


def _img(b64: str, mime: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _safe_image(path: Path) -> tuple[str | None, str]:
    """Return (base64, mime) for an image the APIs accept (png/jpeg/gif/webp).

    Anything else (tiff, bmp, psd, cmyk jpeg, ...) is transcoded to JPEG via PIL,
    and large images are downscaled. A single odd reference image must never 400
    the whole run."""
    import base64 as _b64
    import io
    from PIL import Image
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((1536, 1536))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            return _b64.b64encode(buf.getvalue()).decode(), "image/jpeg"
    except Exception as e:
        return None, str(e)
