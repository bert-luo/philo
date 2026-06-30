"""A Bundle wraps the set of files a judge may inspect for one deliverable.

It abstracts over file type and over zip containers, so the agent tools can
refer to everything by a flat logical name (e.g. "stems.zip/lead_vocal.wav").
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf

from . import config

TEXT_EXT = {".pdf", ".docx", ".txt", ".md", ".srt"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}


def kind(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "video"
    if ext in TEXT_EXT:
        return "document"
    if ext == ".zip":
        return "archive"
    return "other"


@dataclass
class Bundle:
    label: str
    files: dict[str, Path] = field(default_factory=dict)  # logical name -> real path
    roles: dict[str, str] = field(default_factory=dict)   # logical name -> "deliverable"|"reference"
    _extract_dir: Path | None = None

    @classmethod
    def from_dir(cls, label: str, directory: Path, expand_zip_media: bool = True,
                 role: str = "deliverable") -> "Bundle":
        b = cls(label=label)
        b._ingest(directory, role, expand_zip_media)
        return b

    @classmethod
    def with_references(cls, label: str, deliverable_dir: Path, reference_dir: Path | None,
                        expand_zip_media: bool = True) -> "Bundle":
        """Deliverable files (graded) plus read-only reference/input files (for comparison)."""
        b = cls(label=label)
        b._ingest(deliverable_dir, "deliverable", expand_zip_media)
        if reference_dir is not None:
            b._ingest(reference_dir, "reference", expand_zip_media)
        return b

    def _ingest(self, directory: Path, role: str, expand_zip_media: bool) -> None:
        if not directory or not directory.exists():
            return
        for p in sorted(directory.iterdir()):
            if p.is_file():
                self.files[p.name] = p
                self.roles[p.name] = role
                if expand_zip_media and p.suffix.lower() == ".zip":
                    before = set(self.files)
                    self._expand_zip(p)
                    for k in set(self.files) - before:
                        self.roles[k] = role

    def _expand_zip(self, zpath: Path) -> None:
        extract_root = config.CACHE / f"zip_{zpath.stem}"
        try:
            with zipfile.ZipFile(zpath) as zf:
                for member in zf.namelist():
                    if member.endswith("/"):
                        continue
                    base = Path(member).name
                    # Skip macOS archive cruft (AppleDouble resource forks, __MACOSX).
                    if base.startswith("._") or "__MACOSX" in member:
                        continue
                    if kind(member) in ("audio", "video", "image", "document"):
                        target = extract_root / member
                        if not target.exists():
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(member) as src, open(target, "wb") as dst:
                                dst.write(src.read())
                        logical = f"{zpath.name}/{Path(member).name}"
                        self.files[logical] = target
        except zipfile.BadZipFile:
            pass

    # --- introspection -----------------------------------------------------
    def listing(self, role: str | None = None) -> list[dict]:
        out = []
        for name, path in self.files.items():
            r = self.roles.get(name, "deliverable")
            if role is not None and r != role:
                continue
            out.append({
                "name": name,
                "type": kind(name),
                "role": r,
                "size_kb": round(path.stat().st_size / 1024, 1),
            })
        return out

    def resolve(self, name: str, role: str | None = None) -> Path:
        """Resolve a logical name to a path, optionally restricted to one role.

        Restricting to role='deliverable' is how we stop a judge from loading a
        reference INPUT through a grading tool and scoring it as the output."""
        names = {k: p for k, p in self.files.items()
                 if role is None or self.roles.get(k, "deliverable") == role}
        if name in names:
            return names[name]
        for k, p in names.items():  # tolerate basename-only refs from the model
            if Path(k).name == name:
                return p
        # Give a pointed message if it exists but in the other role.
        if role == "deliverable" and (name in self.files or
                any(Path(k).name == name for k in self.files)):
            raise FileNotFoundError(
                f"'{name}' is a REFERENCE input, not part of the graded deliverable. "
                f"Use read_reference/view_reference to consult it for comparison only.")
        raise FileNotFoundError(f"{name} not in bundle {self.label}: {list(names)}")


# --- text extraction --------------------------------------------------------
def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        doc = fitz.open(path)
        parts = []
        for i, page in enumerate(doc):
            parts.append(f"--- page {i + 1} ---\n{page.get_text()}")
        return "\n".join(parts).strip() or "[PDF has no extractable text layer]"
    if ext in (".txt", ".md", ".srt"):
        return path.read_text(errors="replace")
    if ext == ".docx":
        return _docx_text(path)
    return f"[no text extractor for {ext}]"


def _docx_text(path: Path) -> str:
    import re
    import zipfile as zf
    try:
        with zf.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
    except Exception as e:
        return f"[docx read failed: {e}]"
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def pdf_page_png(path: Path, page: int, zoom: float = 1.6) -> str:
    import base64
    doc = fitz.open(path)
    page = max(0, min(page, doc.page_count - 1))
    pix = doc[page].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return base64.b64encode(pix.tobytes("png")).decode()


def pdf_page_count(path: Path) -> int:
    return fitz.open(path).page_count


def file_facts(path: Path) -> dict:
    """Deterministic, non-guessable facts: extension, size, page/word counts.

    Lets the judge answer 'is it a PDF?', 'is it <= 2 pages?', 'word count'
    without inferring from a rendering."""
    facts = {
        "name": path.name,
        "extension": path.suffix.lower(),
        "type": kind(path.name),
        "size_kb": round(path.stat().st_size / 1024, 1),
    }
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        facts["page_count"] = doc.page_count
        facts["word_count"] = sum(len(p.get_text().split()) for p in doc)
    elif path.suffix.lower() in (".docx", ".txt", ".md", ".srt"):
        facts["word_count"] = len(extract_text(path).split())
    return facts
