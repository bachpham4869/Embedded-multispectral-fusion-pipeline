#!/usr/bin/env python3
"""Rebuild/enrich SmartBinocular ML JSONL with non-production metadata."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ml_metadata_utils import (
    IMAGE_EXTS,
    METADATA_STATUS_INFERRED,
    METADATA_STATUS_UNRESOLVED,
    METADATA_STATUS_VERIFIED,
    count_by,
    dhash_image,
    effective_label,
    git_branch_name,
    git_commit_hash,
    label_counts,
    manifest_for_paths,
    manifest_hash,
    optical_feature_hash,
    package_versions,
    read_jsonl,
    repo_relative,
    row_sha256,
    sha256_file,
    write_jsonl,
)


SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "offline_image2weather": {"dataset": "image2weather", "rel": "image2weather", "deterministic": True},
    "offline_weather_time": {"dataset": "weather_time", "rel": "weather_time", "deterministic": True},
    "offline_mwd": {"dataset": "mwd", "rel": "mwd", "deterministic": True},
    "offline_weather11": {"dataset": "weather11", "rel": "weather11", "deterministic": True},
    "offline_darkface": {"dataset": "darkface", "rel": "darkface", "deterministic": False},
    "offline_backlight": {"dataset": "backlight", "rel": "backlight", "deterministic": False},
    "offline_gray_nir": {"dataset": "gray_nir", "rel": "gray", "deterministic": False},
    "offline_exdark_street": {"dataset": "exdark_street", "rel": "ExDark", "deterministic": False},
    "offline_glare_street": {"dataset": "glare_street", "rel": "glare", "deterministic": False},
}

MWD_LABEL_RE = re.compile(r"^([a-z]+)", re.IGNORECASE)
EXDARK_STREET = {"boat", "bus", "car", "motorbike"}


def load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_mwd_root(root: Path) -> Path:
    nested = root / "dataset2" / "dataset2"
    return nested if nested.is_dir() else root


def _resolve_weather11_root(root: Path) -> Path:
    nested = root / "dataset"
    return nested if nested.is_dir() else root


def _mapped_env(mapping: dict[str, Any], key: str, raw_label: str) -> tuple[str | None, float | None, str]:
    entry = (mapping.get(key, {}) or {}).get(raw_label, {}) or {}
    env = entry.get("env")
    confidence = entry.get("confidence")
    if env:
        return str(env), float(confidence if confidence is not None else 0.5), "dataset_original"
    weak = entry.get("weak_label")
    if weak:
        return str(weak), float(confidence if confidence is not None else 0.5), "weak_heuristic"
    return None, float(confidence) if confidence is not None else None, "dataset_original"


def _resolve_weather_time_label(weather: str, period: str, mapping: dict[str, Any]) -> tuple[str | None, float | None, str]:
    section = mapping.get("weather_time", {}) or {}
    for override in section.get("combined_overrides", []) or []:
        match = override.get("match", {}) or {}
        if all((k == "period" and period == v) or (k == "weather" and weather == v) for k, v in match.items()):
            env = override.get("env")
            if env:
                return str(env), float(override.get("confidence", 0.75)), "dataset_original"
            return None, float(override.get("confidence", 0.75)), "dataset_original"
    entry = (section.get("weather", {}) or {}).get(weather, {}) or {}
    env = entry.get("env")
    if env:
        return str(env), float(entry.get("confidence", 0.5)), "dataset_original"
    return None, float(entry.get("confidence", 0.5)) if "confidence" in entry else None, "dataset_original"


def _image_files(root: Path, *, require_decode: bool = False) -> list[Path]:
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if require_decode:
        paths = [p for p in paths if _can_decode(p)]
    return paths


def _can_decode(path: Path) -> bool:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv.imdecode(data, cv.IMREAD_GRAYSCALE)
    if img is None:
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    return img is not None


def _candidate(
    *,
    source_tag: str,
    dataset: str,
    path: Path,
    raw_label: str | None,
    label: str | None,
    label_source: str | None,
    label_confidence: float | None,
    verified_identity: bool,
    source_root: Path,
) -> dict[str, Any]:
    rel = repo_relative(path)
    return {
        "source": source_tag,
        "source_dataset": dataset,
        "path": path,
        "relative_image_id": rel,
        "raw_label": raw_label,
        "label": label,
        "label_source": label_source,
        "label_confidence": label_confidence,
        "verified_identity": verified_identity,
        "split_group_id": f"{source_tag}::{rel}",
    }


def _weather_time_annotations(root: Path) -> list[dict[str, Any]]:
    candidates = [root / "train_dataset" / "train.json", root / "train.json"]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "annotations" in data:
                return list(data["annotations"])
            if isinstance(data, list):
                return data
    return []


def build_source_index(source_root: Path, mapping: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for source_tag, spec in SOURCE_SPECS.items():
        dataset = spec["dataset"]
        root = source_root / spec["rel"]
        verified_identity = bool(spec["deterministic"])
        rows: list[dict[str, Any]] = []
        if not root.exists():
            index[source_tag] = rows
            continue
        if dataset == "mwd":
            for path in sorted(p for p in _resolve_mwd_root(root).iterdir() if p.suffix.lower() in IMAGE_EXTS and _can_decode(p)):
                match = MWD_LABEL_RE.match(path.stem)
                if not match:
                    continue
                raw = match.group(1).lower()
                label, conf, label_source = _mapped_env(mapping, "mwd", raw)
                if label:
                    rows.append(
                        _candidate(
                            source_tag=source_tag,
                            dataset=dataset,
                            path=path,
                            raw_label=raw,
                            label=label,
                            label_source=label_source,
                            label_confidence=conf,
                            verified_identity=verified_identity,
                            source_root=source_root,
                        )
                    )
        elif dataset == "weather11":
            for class_dir in sorted(p for p in _resolve_weather11_root(root).iterdir() if p.is_dir()):
                raw = class_dir.name.lower()
                label, conf, label_source = _mapped_env(mapping, "weather11", raw)
                if not label:
                    continue
                for path in sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and _can_decode(p)):
                    rows.append(
                        _candidate(
                            source_tag=source_tag,
                            dataset=dataset,
                            path=path,
                            raw_label=raw,
                            label=label,
                            label_source=label_source,
                            label_confidence=conf,
                            verified_identity=verified_identity,
                            source_root=source_root,
                        )
                    )
        elif dataset == "weather_time":
            img_base = root / "train_dataset" / "train_images"
            if not img_base.is_dir():
                img_base = root / "train_images"
            for ann in _weather_time_annotations(root):
                fname = Path(str(ann.get("filename", "")).replace("\\", "/")).name
                path = img_base / fname
                if not path.exists() or not _can_decode(path):
                    continue
                weather = str(ann.get("weather", ""))
                period = str(ann.get("period", ""))
                label, conf, label_source = _resolve_weather_time_label(weather, period, mapping)
                if label:
                    rows.append(
                        _candidate(
                            source_tag=source_tag,
                            dataset=dataset,
                            path=path,
                            raw_label=f"{weather}|{period}",
                            label=label,
                            label_source=label_source,
                            label_confidence=conf,
                            verified_identity=verified_identity,
                            source_root=source_root,
                        )
                    )
        elif dataset == "image2weather":
            for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                raw = class_dir.name
                label, conf, label_source = _mapped_env(mapping, "image2weather", raw)
                if not label:
                    continue
                for path in sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
                    rows.append(
                        _candidate(
                            source_tag=source_tag,
                            dataset=dataset,
                            path=path,
                            raw_label=raw,
                            label=label,
                            label_source=label_source,
                            label_confidence=conf,
                            verified_identity=verified_identity,
                            source_root=source_root,
                        )
                    )
        elif dataset == "exdark_street":
            pairs: list[tuple[Path, str]] = []
            for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                raw = class_dir.name.lower()
                if raw not in EXDARK_STREET:
                    continue
                for path in sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and _can_decode(p)):
                    pairs.append((path, raw))
            rng = random.Random(42)
            rng.shuffle(pairs)
            for path, raw in pairs:
                label, conf, label_source = _mapped_env(mapping, "exdark_street", raw)
                if label:
                    rows.append(
                        _candidate(
                            source_tag=source_tag,
                            dataset=dataset,
                            path=path,
                            raw_label=raw,
                            label=label,
                            label_source=label_source,
                            label_confidence=conf,
                            verified_identity=verified_identity,
                            source_root=source_root,
                        )
                    )
        elif dataset in {"darkface", "backlight", "gray_nir", "glare_street"}:
            key = dataset
            if dataset == "darkface":
                image_root = root / "image" if (root / "image").is_dir() else root
                paths = _image_files(image_root)
                raw = "all"
            elif dataset == "glare_street":
                paths = [p for p in _image_files(root) if "mask" not in {part.lower() for part in p.parts}]
                raw = "glare"
            else:
                paths = _image_files(root)
                raw = "all"
            rng = random.Random(42)
            rng.shuffle(paths)
            for path in paths:
                label, conf, label_source = _mapped_env(mapping, key, raw)
                if label:
                    rows.append(
                        _candidate(
                            source_tag=source_tag,
                            dataset=dataset,
                            path=path,
                            raw_label=raw,
                            label=label,
                            label_source=label_source,
                            label_confidence=conf,
                            verified_identity=verified_identity,
                            source_root=source_root,
                        )
                    )
        index[source_tag] = rows
    return index


def _metadata_for_candidate(candidate: dict[str, Any], *, source_root: Path) -> dict[str, Any]:
    path = Path(candidate["path"])
    metadata = {
        "source_dataset": candidate["source_dataset"],
        "metadata_recovery_method": "source_frame_index_replay",
        "metadata_missing": [],
    }
    if candidate["verified_identity"]:
        file_hash = sha256_file(path)
        metadata.update(
            {
                "metadata_status": METADATA_STATUS_VERIFIED,
                "original_image_path": repo_relative(path),
                "relative_image_id": candidate["relative_image_id"],
                "file_sha256": file_hash,
                "dhash": dhash_image(path),
                "split_group_id": f"file_sha256::{file_hash}",
                "capture_device": "offline_dataset",
            }
        )
    else:
        metadata.update(
            {
                "metadata_status": METADATA_STATUS_INFERRED,
                "candidate_original_image_path": repo_relative(path),
                "candidate_relative_image_id": candidate["relative_image_id"],
                "split_group_id": f"inferred::{candidate['source']}::{candidate.get('relative_image_id', row_sha256(candidate))}",
                "capture_device": "offline_dataset",
                "metadata_missing": ["verified original image identity"],
            }
        )
    return metadata


def enrich_records(
    rows: list[dict[str, Any]],
    *,
    source_index: dict[str, list[dict[str, Any]]],
    source_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    hash_coverage = 0
    dhash_coverage = 0
    for row in rows:
        out = dict(row)
        source = str(row.get("source") or "unknown")
        frame_idx = row.get("frame_idx")
        candidate = None
        if isinstance(frame_idx, int) and source in source_index and 0 <= frame_idx < len(source_index[source]):
            candidate = source_index[source][frame_idx]
        if candidate is not None:
            out.update(_metadata_for_candidate(candidate, source_root=source_root))
            if out.get("metadata_status") == METADATA_STATUS_INFERRED and optical_feature_hash(row):
                out["weak_identity_hint"] = "source_frame_index_replay_without_original_path"
        else:
            out.update(
                {
                    "source_dataset": source.removeprefix("offline_") if source != "unknown" else "unknown",
                    "metadata_status": METADATA_STATUS_UNRESOLVED,
                    "metadata_recovery_method": "unresolved",
                    "metadata_missing": ["original image path", "file_sha256", "dhash"],
                    "split_group_id": f"unresolved::{source}::{optical_feature_hash(row) or row_sha256(row)}",
                }
            )
        status = str(out.get("metadata_status", METADATA_STATUS_UNRESOLVED))
        status_counts[status] += 1
        if out.get("file_sha256"):
            hash_coverage += 1
        if out.get("dhash"):
            dhash_coverage += 1
        enriched.append(out)
    summary = {
        "row_count": len(rows),
        "metadata_status_counts": dict(status_counts),
        "file_sha256_coverage": hash_coverage,
        "dhash_coverage": dhash_coverage,
        "label_distribution": label_counts(enriched),
        "source_distribution": count_by(enriched, "source"),
    }
    return enriched, summary


def write_manifest(
    path: Path,
    *,
    command: str,
    inputs: list[Path],
    outputs: list[Path],
    summaries: dict[str, Any],
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "branch": git_branch_name(),
        "command": command,
        "input_manifest": manifest_for_paths(inputs),
        "output_manifest": manifest_for_paths([p for p in outputs if p.exists()]),
        "row_counts": {name: summary["row_count"] for name, summary in summaries.items()},
        "label_distribution": {name: summary["label_distribution"] for name, summary in summaries.items()},
        "source_distribution": {name: summary["source_distribution"] for name, summary in summaries.items()},
        "metadata_coverage": {
            name: {
                "metadata_status_counts": summary["metadata_status_counts"],
                "file_sha256_coverage": summary["file_sha256_coverage"],
                "dhash_coverage": summary["dhash_coverage"],
            }
            for name, summary in summaries.items()
        },
        "feature_set_id": "optical_12_baseline",
        "split_policy": "Phase 3 metadata enrichment only; group-aware split generated separately",
        "versions": package_versions(),
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich existing SmartBinocular JSONL with recoverable image metadata.")
    p.add_argument("--merged-in", type=Path, required=True)
    p.add_argument("--train-in", type=Path, required=True)
    p.add_argument("--test-in", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mapping = load_mapping(args.mapping)
    source_index = build_source_index(args.source_root, mapping)
    specs = [
        ("merged", args.merged_in, args.out_dir / "merged_logs_ml_metadata.jsonl"),
        ("train", args.train_in, args.out_dir / "from_logs_train_metadata.jsonl"),
        ("test", args.test_in, args.out_dir / "from_logs_test_metadata.jsonl"),
    ]
    summaries: dict[str, Any] = {}
    outputs: list[Path] = []
    for name, input_path, output_path in specs:
        rows = read_jsonl(input_path)
        enriched, summary = enrich_records(rows, source_index=source_index, source_root=args.source_root)
        write_jsonl(output_path, enriched)
        summaries[name] = summary
        outputs.append(output_path)
        print(f"{name}: {len(enriched)} rows -> {output_path}")
    write_manifest(
        args.manifest,
        command=" ".join([sys.executable, *sys.argv]),
        inputs=[args.merged_in, args.train_in, args.test_in, args.mapping],
        outputs=outputs,
        summaries=summaries,
    )
    print(f"Wrote manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
