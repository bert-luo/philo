"""ffmpeg / transcription helpers, with on-disk caching to avoid repeat spend."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path

from openai import OpenAI

from . import config

_oai = None


def _openai() -> OpenAI:
    global _oai
    if _oai is None:
        _oai = OpenAI(api_key=config.OPENAI_API_KEY)
    return _oai


def _hash(path: Path, *extra) -> str:
    h = hashlib.sha1()
    h.update(str(path).encode())
    h.update(str(path.stat().st_size).encode())
    for e in extra:
        h.update(str(e).encode())
    return h.hexdigest()[:16]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def duration_seconds(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            check=True, capture_output=True, text=True,
        ).stdout
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return 0.0


def probe(path: Path) -> dict:
    """Deterministic media facts via ffprobe (container, codecs, resolution, fps, ...).

    This exists so the judge never has to *guess* objective technical criteria
    ('codec is H.264', 'resolution is 1920x1080', 'runtime 29.9-30.1s')."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams",
             "-of", "json", str(path)],
            check=True, capture_output=True, text=True,
        ).stdout
        raw = json.loads(out)
    except Exception as e:
        return {"error": f"ffprobe failed: {e}"}
    fmt = raw.get("format", {})
    facts: dict = {
        "container": fmt.get("format_name"),
        "duration_s": round(float(fmt.get("duration", 0) or 0), 3),
        "size_bytes": int(fmt.get("size", 0) or 0),
        "bit_rate": fmt.get("bit_rate"),
        "video": [], "audio": [],
    }
    for s in raw.get("streams", []):
        if s.get("codec_type") == "video":
            num, den = (s.get("r_frame_rate", "0/1").split("/") + ["1"])[:2]
            fps = round(float(num) / float(den), 2) if float(den) else None
            facts["video"].append({
                "codec": s.get("codec_name"),
                "width": s.get("width"), "height": s.get("height"),
                "fps": fps, "pix_fmt": s.get("pix_fmt"),
            })
        elif s.get("codec_type") == "audio":
            facts["audio"].append({
                "codec": s.get("codec_name"),
                "sample_rate": s.get("sample_rate"),
                "channels": s.get("channels"),
            })
    return facts


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


# --- frames ---------------------------------------------------------------
def _nonempty(paths: list[Path]) -> list[Path]:
    return [p for p in paths if p.exists() and p.stat().st_size > 0]


def video_frames(path: Path, n: int = 10, width: int = 768,
                 start: float | None = None, end: float | None = None) -> list[Path]:
    """n evenly-spaced JPEG frame files, optionally within a [start,end] window.

    Higher default resolution + count than before so on-screen text and distinct
    shots are actually legible/visible. Returns paths; the caller validates/encodes."""
    dur = duration_seconds(path) or 1.0
    a = 0.0 if start is None else max(0.0, start)
    b = dur if end is None else min(dur, end)
    if b <= a:
        b = dur
    outdir = config.CACHE / f"frames_{_hash(path, n, width, a, b)}"
    frames = sorted(outdir.glob("*.jpg")) if outdir.exists() else []
    if not frames:
        outdir.mkdir(parents=True, exist_ok=True)
        span = b - a
        for i in range(n):
            ts = a + span * (i + 0.5) / n
            out = outdir / f"{i:03d}.jpg"
            _run(["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", str(path),
                  "-frames:v", "1", "-vf", f"scale={width}:-1", "-q:v", "3", str(out)])
        frames = sorted(outdir.glob("*.jpg"))
    return _nonempty(frames)


def scene_frames(path: Path, max_frames: int = 12, width: int = 768,
                 threshold: float = 0.25) -> list[Path]:
    """One JPEG per detected scene cut (distinct shots), capped at max_frames.

    Covers 'includes a shot of X' / face-penalty items far better than uniform
    sampling, which routinely misses whole shots in a 30s spot."""
    outdir = config.CACHE / f"scene_{_hash(path, max_frames, width, threshold)}"
    frames = sorted(outdir.glob("*.jpg")) if outdir.exists() else []
    if not frames:
        outdir.mkdir(parents=True, exist_ok=True)
        _run(["ffmpeg", "-y", "-i", str(path),
              "-vf", f"select='gt(scene,{threshold})',scale={width}:-1",
              "-vsync", "vfr", "-frames:v", str(max_frames), "-q:v", "3",
              str(outdir / "%03d.jpg")])
        frames = sorted(outdir.glob("*.jpg"))
        if not _nonempty(frames):  # static video: fall back to uniform sampling
            return video_frames(path, n=max_frames, width=width)
    return _nonempty(frames)


# --- audio clip (native listening) ----------------------------------------
def audio_clip_mp3(path: Path, start: float = 0.0, dur: float = 90.0) -> str:
    """Downsampled mono mp3 clip as base64 (small enough to attach)."""
    out = config.CACHE / f"clip_{_hash(path, start, dur)}.mp3"
    if not out.exists():
        _run(["ffmpeg", "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
              "-i", str(path), "-ac", "1", "-ar", "16000", "-b:a", "48k", str(out)])
    return base64.b64encode(out.read_bytes()).decode()


# --- transcription (the degraded audio path available to every judge) ------
def _extract_wav(path: Path) -> Path:
    out = config.CACHE / f"audio_{_hash(path)}.mp3"
    if not out.exists():
        _run(["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", "16000",
              "-b:a", "48k", out.as_posix()])
    return out


def transcribe(path: Path, max_seconds: float = 600.0) -> str:
    """Whisper transcript, cached. Caps very long media to bound cost."""
    cache = config.CACHE / f"transcript_{_hash(path, max_seconds)}.txt"
    if cache.exists():
        return cache.read_text()
    audio = _extract_wav(path)
    # Trim to the cap so a 30-min stem set doesn't blow the budget.
    dur = duration_seconds(audio)
    if dur > max_seconds:
        trimmed = config.CACHE / f"trim_{_hash(audio, max_seconds)}.mp3"
        if not trimmed.exists():
            _run(["ffmpeg", "-y", "-t", f"{max_seconds}", "-i", str(audio), str(trimmed)])
        audio = trimmed
    try:
        with open(audio, "rb") as fh:
            r = _openai().audio.transcriptions.create(model="whisper-1", file=fh)
        text = r.text
    except Exception as e:
        text = f"[transcription failed: {e}]"
    cache.write_text(text)
    return text


# --- audio understanding (DSP/MIR) ----------------------------------------
# Deterministic measurements a judge otherwise has to *guess*: loudness, tempo,
# musical key, and sync drift to a reference. Results are cached as JSON.

def _ebur128(path: Path) -> dict:
    """Integrated loudness (LUFS) + true peak via ffmpeg's ebur128 filter."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
             "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True,
        ).stderr
    except Exception as e:
        return {"error": f"ebur128 failed: {e}"}
    import re
    def grab(label):
        m = re.findall(rf"{label}:\s*(-?\d+\.?\d*)", out)
        return float(m[-1]) if m else None
    return {"integrated_lufs": grab("I"), "loudness_range_lu": grab("LRA"),
            "true_peak_dbfs": grab("Peak")}


def _load_mono(path: Path, sr: int = 22050, max_s: float = 240.0):
    import librosa
    y, _sr = librosa.load(str(path), sr=sr, mono=True, duration=max_s)
    return y, sr


_KRUMHANSL_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KRUMHANSL_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimate_key(y, sr) -> dict:
    """Krumhansl-Schmuckler key estimate from mean chroma (approximate)."""
    import numpy as np, librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    if chroma.sum() == 0:
        return {"key": None}
    chroma = chroma / chroma.sum()
    best = (-1, None)
    for i in range(12):
        for prof, mode in ((_KRUMHANSL_MAJ, "major"), (_KRUMHANSL_MIN, "minor")):
            r = float(np.corrcoef(np.roll(prof, i), chroma)[0, 1])
            if r > best[0]:
                best = (r, f"{_NOTES[i]} {mode}")
    return {"key": best[1], "key_confidence": round(best[0], 2)}


def analyze_audio(path: Path) -> dict:
    """Loudness + tempo + key + basic spectral stats for one audio/video file."""
    cache = config.CACHE / f"analyze_{_hash(path)}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    facts: dict = {"duration_s": round(duration_seconds(path), 2)}
    facts.update(_ebur128(path))
    try:
        import numpy as np, librosa
        y, sr = _load_mono(path)
        if y.size:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            facts["tempo_bpm"] = round(float(np.atleast_1d(tempo)[0]), 1)
            facts.update(_estimate_key(y, sr))
            # crude harmonic/percussive split -> "is there sustained pitched (vocal/lead) content"
            yh, yp = librosa.effects.hpss(y)
            facts["harmonic_fraction"] = round(float(
                np.sum(yh ** 2) / (np.sum(y ** 2) + 1e-9)), 2)
            facts["spectral_centroid_hz"] = round(float(
                librosa.feature.spectral_centroid(y=y, sr=sr).mean()), 0)
    except Exception as e:
        facts["analysis_error"] = str(e)[:120]
    cache.write_text(json.dumps(facts))
    return facts


def audio_sync(path: Path, ref: Path, sr: int = 22050) -> dict:
    """Estimate start offset and end-to-end drift between a track and a reference
    via onset-envelope cross-correlation on the head and tail windows."""
    cache = config.CACHE / f"sync_{_hash(path)}_{_hash(ref)}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    try:
        import numpy as np, librosa

        def onset_env(p, offset, dur):
            y, _ = librosa.load(str(p), sr=sr, mono=True, offset=offset, duration=dur)
            if y.size == 0:
                return None
            return librosa.onset.onset_strength(y=y, sr=sr)

        def lag_ms(a, b):
            if a is None or b is None:
                return None
            n = min(len(a), len(b))
            a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
            xc = np.correlate(a, b, mode="full")
            lag_frames = xc.argmax() - (n - 1)
            return round(lag_frames * 512 / sr * 1000, 1)  # hop=512

        da, dref = duration_seconds(path), duration_seconds(ref)
        w = 10.0
        head = lag_ms(onset_env(path, 0, w), onset_env(ref, 0, w))
        tail = lag_ms(onset_env(path, max(0, da - w), w),
                      onset_env(ref, max(0, dref - w), w))
        res = {"start_offset_ms": head, "tail_offset_ms": tail,
               "drift_ms": (round(abs(tail - head), 1) if head is not None and tail is not None else None),
               "deliverable_duration_s": round(da, 2), "reference_duration_s": round(dref, 2)}
    except Exception as e:
        res = {"error": str(e)[:120]}
    cache.write_text(json.dumps(res))
    return res
