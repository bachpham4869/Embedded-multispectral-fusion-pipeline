#!/usr/bin/env python3
"""Artifact inventory and pairing helpers for offline fusion evaluation."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DATA_EXTS = {".npy", ".npz"}
TABLE_EXTS = {".csv", ".json", ".jsonl"}
VIDEO_EXTS = {".mp4", ".avi"}
SCAN_EXTS = IMAGE_EXTS | DATA_EXTS | TABLE_EXTS | VIDEO_EXTS

EVIDENCE_REAL_PAIRED = "real_paired"
EVIDENCE_PROXY = "proxy"
EVIDENCE_UNPAIRED = "unpaired"
EVIDENCE_SYNTHETIC = "synthetic"
EVIDENCE_UNKNOWN = "unknown"

PROCESSING_BUCKET_NAMES = {
    "A": "night_hybrid_enhance",
    "B": "nir_mono_clahe",
    "C": "highlight_tone_map",
    "D": "fog_dehaze_lite",
    "E": "rain_temporal_median",
    "F": "dawn_dusk_blend",
}

FUSION_MODE_NAMES = {
    "nir_only_baseline",
    "thermal_heatmap_only",
    "alpha_blend_baseline",
    "foreground_mask_overlay",
    "mask_weighted_blend",
    "legacy_gradient_overlay",
    "laplacian_pyramid_fusion",
}


@dataclass(frozen=True)
class ArtifactClassification:
    kind: str
    modality: str
    evidence_label: str


@dataclass(frozen=True)
class ArtifactRecord:
    path: Path
    kind: str
    modality: str
    evidence_label: str
    extension: str
    size_bytes: int
    session_id: str
    timestamp_iso: str
    timestamp_source: str
    sidecar_path: Path | None
    source_or_session: str
    bucket_or_condition: str


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    fusion: ArtifactRecord
    nir: ArtifactRecord | None
    thermal: ArtifactRecord | None
    time_gap_nir_sec: float | None
    time_gap_thermal_sec: float | None
    pair_status: str
    evidence_label: str
    source_or_session: str
    bucket_or_condition: str


def classify_artifact(path: Path) -> ArtifactClassification:
    name = path.name.lower()
    suffix = path.suffix.lower()
    text = str(path).lower()

    if suffix in VIDEO_EXTS:
        return ArtifactClassification("video", "unknown", EVIDENCE_UNKNOWN)
    if suffix in DATA_EXTS:
        evidence = EVIDENCE_SYNTHETIC if "surrogate" in text or "synthetic" in text else EVIDENCE_PROXY
        modality = "thermal" if "thermal" in text else "unknown"
        return ArtifactClassification("numeric_sequence", modality, evidence)
    if suffix in {".json", ".jsonl", ".csv"}:
        if name.startswith("session_"):
            return ArtifactClassification("session_metrics", "metrics", EVIDENCE_REAL_PAIRED)
        if name.startswith("manifest_"):
            return ArtifactClassification("run_manifest", "metrics", EVIDENCE_REAL_PAIRED)
        if "metrics" in text:
            return ArtifactClassification("session_metrics", "metrics", EVIDENCE_REAL_PAIRED)
        return ArtifactClassification("sidecar_or_table", "metadata", EVIDENCE_UNKNOWN)
    if suffix not in IMAGE_EXTS:
        return ArtifactClassification("unknown", "unknown", EVIDENCE_UNKNOWN)

    if re.match(r"^(auto_)?fusion_\d{8}-\d{6}\.", name) or re.match(r"^fus[_-]?\d", name):
        return ArtifactClassification("fusion_output", "fusion", EVIDENCE_REAL_PAIRED)
    if re.match(r"^(auto_)?thermal_\d{8}-\d{6}\.", name) or re.match(r"^thm[_-]?\d", name):
        evidence = EVIDENCE_SYNTHETIC if "surrogate" in text or "synthetic" in text else EVIDENCE_REAL_PAIRED
        return ArtifactClassification("thermal_output", "thermal", evidence)
    if re.match(r"^(auto_)?imx_\d{8}-\d{6}\.", name) or re.match(r"^(auto_)?nir_\d", name) or re.match(r"^(auto_)?opt", name):
        return ArtifactClassification("nir_raw", "nir", EVIDENCE_REAL_PAIRED)
    if "weather" in text or "darkface" in text or "exdark" in text:
        return ArtifactClassification("nir_proxy_still", "nir", EVIDENCE_PROXY)
    return ArtifactClassification("unknown_image", "unknown", EVIDENCE_UNKNOWN)


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _timestamp_from_name(path: Path) -> tuple[str, str]:
    patterns = (
        ("%Y%m%d-%H%M%S", r"(\d{8}-\d{6})"),
        ("%Y%m%dT%H%M%S", r"(\d{8}T\d{6})"),
        ("%Y-%m-%d_%H-%M-%S", r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"),
    )
    for fmt, pattern in patterns:
        match = re.search(pattern, path.name)
        if match:
            value = match.group(1)
            parsed = dt.datetime.strptime(value, fmt)
            return parsed.isoformat(), "filename"
    return dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(), "mtime"


def _parse_time(value: str) -> dt.datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def _time_gap(a: str, b: str) -> float | None:
    da = _parse_time(a)
    db = _parse_time(b)
    if da is None or db is None:
        return None
    return abs((da - db).total_seconds())


def _sidecar_for(path: Path) -> Path | None:
    sidecar = path.with_suffix(".json")
    return sidecar if sidecar.exists() else None


def _source_session(path: Path, sidecar: dict[str, Any]) -> str:
    session = str(sidecar.get("session_id") or "")
    if session:
        return session
    parts = path.parts
    for idx, part in enumerate(parts):
        if part in {"fusion_captures", "data", "docs", "artifacts"} and idx + 1 < len(parts):
            return "/".join(parts[idx: idx + 2])
    return str(path.parent)


def _condition_from_path(path: Path, sidecar: dict[str, Any]) -> str:
    for key in ("bucket", "env_class", "mode"):
        if sidecar.get(key):
            value = str(sidecar[key])
            return PROCESSING_BUCKET_NAMES.get(value, value)
    text = str(path).lower()
    for token in ("night_clear", "nir_night", "normal_night", "fog", "glare", "backlight", "rain", "normal_day"):
        if token in text:
            return token
    return ""


def discover_artifacts(paths: Iterable[Path], *, max_depth: int | None = None) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    skip_dirs = {".git", ".gitnexus", ".venv", "__pycache__", ".pytest_cache"}
    roots = [Path(p) for p in paths if Path(p).exists()]
    for root in roots:
        if root.is_file():
            candidates = [root]
        else:
            candidates = []
            base_depth = len(root.parts)
            for current, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d not in skip_dirs]
                cur_path = Path(current)
                if max_depth is not None and len(cur_path.parts) - base_depth >= max_depth:
                    dirs[:] = []
                candidates.extend(cur_path / f for f in files)
        for path in candidates:
            if path.suffix.lower() not in SCAN_EXTS:
                continue
            cls = classify_artifact(path)
            sidecar_path = _sidecar_for(path) if path.suffix.lower() in IMAGE_EXTS else None
            sidecar = _safe_json(sidecar_path) if sidecar_path else (_safe_json(path) if path.suffix.lower() == ".json" else {})
            timestamp = str(sidecar.get("timestamp_iso") or sidecar.get("timestamp") or "")
            source = "sidecar" if timestamp else ""
            if not timestamp:
                timestamp, source = _timestamp_from_name(path)
            session = _source_session(path, sidecar)
            records.append(
                ArtifactRecord(
                    path=path,
                    kind=cls.kind,
                    modality=cls.modality,
                    evidence_label=cls.evidence_label,
                    extension=path.suffix.lower(),
                    size_bytes=path.stat().st_size,
                    session_id=str(sidecar.get("session_id") or ""),
                    timestamp_iso=timestamp,
                    timestamp_source=source,
                    sidecar_path=sidecar_path,
                    source_or_session=session,
                    bucket_or_condition=_condition_from_path(path, sidecar),
                )
            )
    return sorted(records, key=lambda r: str(r.path))


def pair_status_for_gap(
    gap_sec: float | None,
    *,
    strict_window_sec: float = 1.0,
    qualitative_window_sec: float = 20.0,
) -> str:
    if gap_sec is None:
        return "missing_pair"
    if gap_sec <= strict_window_sec:
        return "strict_paired"
    if gap_sec <= qualitative_window_sec:
        return "qualitative_weak"
    return "reject_unpaired"


def _nearest(target: ArtifactRecord, candidates: list[ArtifactRecord]) -> tuple[ArtifactRecord | None, float | None]:
    if not candidates:
        return None, None
    same_session = [
        c for c in candidates
        if target.session_id and c.session_id and c.session_id == target.session_id
    ]
    pool = same_session or candidates
    best = min(pool, key=lambda c: _time_gap(target.timestamp_iso, c.timestamp_iso) if _time_gap(target.timestamp_iso, c.timestamp_iso) is not None else math.inf)
    gap = _time_gap(target.timestamp_iso, best.timestamp_iso)
    return best, gap


def pair_capture_records(
    records: Iterable[ArtifactRecord],
    *,
    strict_window_sec: float = 1.0,
    qualitative_window_sec: float = 20.0,
) -> list[PairRecord]:
    recs = list(records)
    fusions = [r for r in recs if r.kind == "fusion_output" and r.extension in IMAGE_EXTS]
    nirs = [r for r in recs if r.kind in {"nir_raw", "nir_enhanced", "nir_proxy_still"} and r.extension in IMAGE_EXTS]
    thermals = [r for r in recs if r.kind in {"thermal_output", "thermal_raw", "thermal_enhanced"} and r.extension in IMAGE_EXTS]
    pairs: list[PairRecord] = []
    for idx, fusion in enumerate(sorted(fusions, key=lambda r: (r.timestamp_iso, str(r.path))), start=1):
        nir, gap_nir = _nearest(fusion, nirs)
        thermal, gap_thermal = _nearest(fusion, thermals)
        status = pair_status_for_gap(gap_nir, strict_window_sec=strict_window_sec, qualitative_window_sec=qualitative_window_sec)
        if status == "strict_paired" and gap_thermal is not None and gap_thermal > strict_window_sec:
            status = "qualitative_weak" if gap_thermal <= qualitative_window_sec else "reject_unpaired"
        if status == "strict_paired":
            evidence = EVIDENCE_REAL_PAIRED
        elif status == "qualitative_weak":
            evidence = EVIDENCE_PROXY
        elif status == "missing_pair":
            evidence = EVIDENCE_UNPAIRED
        else:
            evidence = EVIDENCE_UNPAIRED
        pairs.append(
            PairRecord(
                pair_id=f"pair_{idx:04d}",
                fusion=fusion,
                nir=nir,
                thermal=thermal,
                time_gap_nir_sec=gap_nir,
                time_gap_thermal_sec=gap_thermal,
                pair_status=status,
                evidence_label=evidence,
                source_or_session=fusion.source_or_session,
                bucket_or_condition=fusion.bucket_or_condition,
            )
        )
    return pairs


def format_pairing_manifest_rows(pairs: Iterable[PairRecord]) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        rows.append(
            {
                "pair_id": pair.pair_id,
                "pair_status": pair.pair_status,
                "evidence_label": pair.evidence_label,
                "source_or_session": pair.source_or_session,
                "bucket_or_condition": pair.bucket_or_condition,
                "fusion_path": str(pair.fusion.path),
                "nir_path": str(pair.nir.path) if pair.nir else "",
                "thermal_path": str(pair.thermal.path) if pair.thermal else "",
                "fusion_timestamp": pair.fusion.timestamp_iso,
                "nir_timestamp": pair.nir.timestamp_iso if pair.nir else "",
                "thermal_timestamp": pair.thermal.timestamp_iso if pair.thermal else "",
                "time_gap_nir_sec": pair.time_gap_nir_sec,
                "time_gap_thermal_sec": pair.time_gap_thermal_sec,
                "timestamp_source": pair.fusion.timestamp_source,
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in row_list:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def markdown_table(rows: Iterable[dict[str, Any]], columns: list[str]) -> str:
    row_list = list(rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in row_list:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def path_sha256(path: Path, *, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        remaining = max_bytes
        while True:
            size = 65536 if remaining is None else min(65536, remaining)
            if size <= 0:
                break
            chunk = fh.read(size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def run_manifest(command: str, inputs: list[str], config: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "timestamp_iso": now_iso(),
        "git_commit": git_commit(),
        "command": command,
        "inputs": inputs,
        "config": config or {},
        "metric_definitions_version": "fusion_eval_metrics_v1",
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
