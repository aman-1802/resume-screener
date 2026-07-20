"""Best-effort candidate name detection from a resume PDF.

A plain "first line that looks like a name" heuristic is fragile: many
resumes put the name and contact details (email, phone, location) on the
same visual line in a header, which makes that line fail a simple regex
and pushes the heuristic onto the next plausible-looking line -- often a
section heading like "Technical Skills". Instead, we use the name's most
reliable visual signal: it's almost always the largest text near the top
of page 1. PyMuPDF exposes per-span font size, so we rank text lines by
font size (restricted to the top of the page) and pick the first one that
still looks name-shaped.
"""

import re

import fitz  # PyMuPDF

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'-]+$")
_NON_NAME_WORDS = {
    "resume", "cv", "curriculum", "vitae", "profile", "summary", "objective",
    "contact", "about", "portfolio", "linkedin", "github",
}


def _looks_like_name(text: str) -> bool:
    words = text.split()
    if not (2 <= len(words) <= 5):
        return False
    if not _NAME_RE.match(text):
        return False
    return not any(w.lower() in _NON_NAME_WORDS for w in words)


def _from_font_size(pdf_path) -> str | None:
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return None

    try:
        if doc.page_count == 0:
            return None
        page = doc[0]
        page_height = page.rect.height
        lines = []

        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                max_size = max(s.get("size", 0) for s in spans)
                y = line.get("bbox", [0, 0, 0, 0])[1]
                lines.append((max_size, y, text))

        # Only consider the top third of the page -- the header area.
        top_lines = [l for l in lines if l[1] <= page_height / 3]
        # Largest font first; topmost first among ties.
        top_lines.sort(key=lambda l: (-l[0], l[1]))

        for _size, _y, text in top_lines:
            if _looks_like_name(text):
                return text
        return None
    finally:
        doc.close()


def _from_plain_text(resume_text: str) -> str | None:
    for line in resume_text.splitlines():
        line = line.strip()
        if _looks_like_name(line):
            return line
    return None


def guess_candidate_name(pdf_path, resume_text: str, fallback: str) -> str:
    """Try font-size detection (digital PDFs), then plain-text scan, then filename."""
    name = _from_font_size(pdf_path)
    if name:
        return name

    name = _from_plain_text(resume_text)
    if name:
        return name

    return fallback
