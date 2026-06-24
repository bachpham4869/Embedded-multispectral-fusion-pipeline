"""Integrity check for docs/thesis_eval/MANIFEST.md.

Asserts:
1. Every artifact_path listed in MANIFEST.md that is marked 'available' exists on disk.
2. No internal code-style labels (E1, P2, phase 2, etc.) appear in:
   - any filename under docs/thesis_eval/
   - the 'caption' column of MANIFEST.md rows
   The regex is narrow and anchored to avoid false positives on normal English.
"""
from __future__ import annotations

import csv
import re
from io import StringIO
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "docs/thesis_eval/MANIFEST.md"

# Anchored to start of token: E/P followed by 1-2 digits, or "phase" + digit
# Matches: E1_, E12-, P2_, phase_2, phase 3, phase-1 etc.
# Does NOT match: "entropy", "phase_type", "preprocessing"
_INTERNAL_CODE_RE = re.compile(
    r"(?<![a-zA-Z])(?:(?:[Ee]\d{1,2}|[Pp]\d{1,2})(?:[_\-]|\b)|phase[_\s\-]\d)",
    re.IGNORECASE,
)


def _parse_manifest_table_rows(text: str) -> list[dict[str, str]]:
    """Extract CSV-like rows from Markdown tables in MANIFEST.md."""
    rows = []
    header: list[str] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            # Separator row — next row after this will be a data row
            continue
        if header is None:
            header = [c.lower().replace(" ", "_") for c in cells]
            continue
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    return rows


@pytest.fixture(scope="module")
def manifest_rows():
    if not MANIFEST.exists():
        pytest.skip(f"MANIFEST.md not found: {MANIFEST}")
    text = MANIFEST.read_text(encoding="utf-8")
    return _parse_manifest_table_rows(text), text


def test_manifest_exists():
    assert MANIFEST.exists(), f"MANIFEST.md missing: {MANIFEST}"


def test_available_artifacts_exist_on_disk(manifest_rows):
    rows, _ = manifest_rows
    missing = []
    for row in rows:
        status = row.get("status", "").strip().strip("`")
        path_raw = row.get("artifact_path", "").strip().strip("`")
        if status == "available" and path_raw:
            full = ROOT / path_raw
            if not full.exists():
                missing.append(path_raw)
    assert not missing, (
        f"{len(missing)} 'available' artifacts missing on disk:\n"
        + "\n".join(f"  {p}" for p in missing[:10])
    )


def test_no_internal_codes_in_captions(manifest_rows):
    rows, _ = manifest_rows
    hits = []
    for row in rows:
        caption = row.get("caption", "")
        m = _INTERNAL_CODE_RE.search(caption)
        if m:
            hits.append((row.get("artifact_path", "?"), caption, m.group()))
    assert not hits, (
        f"Internal code patterns found in MANIFEST captions:\n"
        + "\n".join(f"  {p!r}: matched {m!r} in {c!r}" for p, c, m in hits[:5])
    )


def test_no_internal_codes_in_thesis_filenames():
    if not (ROOT / "docs/thesis_eval").exists():
        pytest.skip("docs/thesis_eval/ does not exist")
    hits = []
    for f in (ROOT / "docs/thesis_eval").rglob("*"):
        if _INTERNAL_CODE_RE.search(f.name):
            hits.append(str(f.relative_to(ROOT)))
    assert not hits, (
        f"Internal code patterns in filenames under docs/thesis_eval/:\n"
        + "\n".join(f"  {h}" for h in hits[:10])
    )
