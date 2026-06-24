#!/usr/bin/env python3
"""Shared utilities for SmartBinocular ML metadata and leakage tooling."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2 as cv
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

METADATA_STATUS_VERIFIED = "verified"
METADATA_STATUS_INFERRED = "inferred_low_confidence"
METADATA_STATUS_UNRESOLVED = "unresolved"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

FEATURE_KEYS = (
    "nir_mean_brightness",
    "nir_std",
    "nir_entropy",
    "nir_p95",
    "nir_glare_score",
    "nir_sharpness",
    "nir_dark_fraction",
    "nir_saturation_mean",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "prev_env_class",
    "nir_blue_mean_ema",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def row_sha256(row: dict[str, Any]) -> str:
    return sha256_json(row)


def optical_feature_hash(row: dict[str, Any]) -> str | None:
    if not all(key in row for key in FEATURE_KEYS):
        return None
    return sha256_json([row.get(key) for key in FEATURE_KEYS])


def effective_label(row: dict[str, Any]) -> str:
    return (row.get("label") or row.get("weak_label") or "unlabeled") or "unlabeled"


def dhash_image(path: Path, hash_size: int = 8) -> str:
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if img is None:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv.imdecode(data, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image for dHash: {path}")
    resized = cv.resize(img, (hash_size + 1, hash_size), interpolation=cv.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bool(bit))
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_distance_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def classify_metadata_status(
    *,
    has_identity: bool,
    has_file_sha256: bool,
    weak_hint_only: bool,
) -> tuple[str, str]:
    if has_identity:
        return METADATA_STATUS_VERIFIED, "path_or_frame_identity"
    if has_file_sha256:
        return METADATA_STATUS_VERIFIED, "file_sha256"
    if weak_hint_only:
        return METADATA_STATUS_INFERRED, "weak_feature_hint"
    return METADATA_STATUS_UNRESOLVED, "unresolved"


def repo_relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(Path(path))


def count_by(rows: Iterable[dict[str, Any]], field: str, *, missing: str = "unknown") -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(field)
        counts[str(value if value not in (None, "") else missing)] += 1
    return dict(sorted(counts.items()))


def label_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter(effective_label(row) for row in rows)
    return dict(sorted(counts.items()))


def manifest_for_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            manifest.append({"path": str(path), "exists": False})
            continue
        manifest.append(
            {
                "path": str(path),
                "exists": True,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return manifest


def manifest_hash(manifest: Any) -> str:
    return sha256_json(manifest)


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def git_branch_name() -> str:
    try:
        out = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except Exception:
        return "unknown"


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except Exception:
        versions["sklearn"] = "unavailable"
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception:
        versions["scipy"] = "unavailable"
    versions["numpy"] = np.__version__
    versions["opencv"] = cv.__version__
    return versions


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        rows = [["" for _ in headers]]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
