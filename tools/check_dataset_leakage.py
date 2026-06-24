#!/usr/bin/env python3
"""Metadata-level leakage checks for SmartBinocular ENV train/test JSONL files.

The script checks only existing JSONL records. It can optionally inspect image
paths when metadata includes them, but it never downloads or mixes datasets.
If source images are unavailable, the report explicitly marks perceptual-hash
near-duplicate checks as not run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

PATH_FIELDS = ("image_path", "path", "filepath", "file_path", "filename", "image", "source_path")
SESSION_FIELDS = ("session_id", "session", "capture_session", "video_id", "sequence_id")
FRAME_FIELDS = ("frame_idx", "frame_index", "frame_id", "timestamp", "ts")
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


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def effective_label(row: dict[str, Any]) -> str:
    return (row.get("label") or row.get("weak_label") or "unlabeled") or "unlabeled"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_manifest(paths: Iterable[Path]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for path in paths:
        stat = path.stat()
        manifest.append({"path": str(path), "sha256": sha256_file(path), "bytes": stat.st_size})
    return manifest


def manifest_hash(manifest: list[dict[str, Any]]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except Exception:
        versions["sklearn"] = "unavailable"
    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except Exception:
        versions["numpy"] = "unavailable"
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception:
        versions["scipy"] = "unavailable"
    return versions


def _first_present(row: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return None


def _collect_values(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> set[str]:
    values: set[str] = set()
    for row in rows:
        value = _first_present(row, fields)
        if value is not None:
            values.add(str(value))
    return values


def _filename_values(rows: list[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for row in rows:
        value = _first_present(row, PATH_FIELDS)
        if value is not None:
            values.add(Path(str(value)).name)
    return values


def _session_frame_values(rows: list[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for row in rows:
        session = _first_present(row, SESSION_FIELDS)
        frame = _first_present(row, FRAME_FIELDS)
        if session is not None and frame is not None:
            values.add(f"{session}::{frame}")
    return values


def _row_fingerprint(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _feature_fingerprint(row: dict[str, Any]) -> str | None:
    if not all(key in row for key in FEATURE_KEYS):
        return None
    payload = json.dumps([row.get(key) for key in FEATURE_KEYS], separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _feature_values(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in FEATURE_KEYS}


def _missing_metadata_note(train_row: dict[str, Any], test_row: dict[str, Any]) -> str:
    missing: list[str] = []
    if _first_present(train_row, PATH_FIELDS) is None or _first_present(test_row, PATH_FIELDS) is None:
        missing.append("original image path/hash metadata")
    train_session = _first_present(train_row, SESSION_FIELDS)
    test_session = _first_present(test_row, SESSION_FIELDS)
    if train_session is None or test_session is None:
        missing.append("session_id metadata")
    return "; ".join(missing) if missing else "none"


def feature_vector_overlap_details(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return train/test rows with identical optical_12 feature vectors.

    This is a warning signal, not proof of image-level leakage. Exact image
    duplicate review still needs original image path/hash metadata.
    """

    train_by_fp: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(train_rows):
        fp = _feature_fingerprint(row)
        if fp is not None:
            train_by_fp[fp].append((idx, row))

    details: list[dict[str, Any]] = []
    for test_idx, test_row in enumerate(test_rows):
        fp = _feature_fingerprint(test_row)
        if fp is None or fp not in train_by_fp:
            continue
        test_values = _feature_values(test_row)
        for train_idx, train_row in train_by_fp[fp]:
            train_values = _feature_values(train_row)
            diff = {
                key: (None if train_values.get(key) == test_values.get(key) else [train_values.get(key), test_values.get(key)])
                for key in FEATURE_KEYS
            }
            details.append(
                {
                    "train_index": train_idx,
                    "test_index": test_idx,
                    "train_label": effective_label(train_row),
                    "test_label": effective_label(test_row),
                    "train_source": train_row.get("source", "unknown"),
                    "test_source": test_row.get("source", "unknown"),
                    "train_label_source": train_row.get("label_source", "unknown"),
                    "test_label_source": test_row.get("label_source", "unknown"),
                    "train_label_confidence": train_row.get("label_confidence", ""),
                    "test_label_confidence": test_row.get("label_confidence", ""),
                    "train_nir_channel": train_row.get("nir_channel", "unknown"),
                    "test_nir_channel": test_row.get("nir_channel", "unknown"),
                    "train_thermal_channel": train_row.get("thermal_channel", "unknown"),
                    "test_thermal_channel": test_row.get("thermal_channel", "unknown"),
                    "feature_hash": fp,
                    "match_type": "exact_feature_vector",
                    "feature_values_json": json.dumps(test_values, sort_keys=True, separators=(",", ":"), default=str),
                    "feature_diff_json": json.dumps(diff, sort_keys=True, separators=(",", ":"), default=str),
                    "missing_metadata": _missing_metadata_note(train_row, test_row),
                }
            )
    return details


def _overlap_report(train_values: set[str], test_values: set[str], sample_limit: int = 20) -> dict[str, Any]:
    values = sorted(train_values & test_values)
    return {"count": len(values), "examples": values[:sample_limit]}


def _class_counts_for_values(
    rows: list[dict[str, Any]],
    values: set[str],
    fields: tuple[str, ...],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = _first_present(row, fields)
        if value is not None and str(value) in values:
            counts[effective_label(row)] += 1
    return dict(counts)


def check_metadata_overlap(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check train/test overlap using metadata available in JSONL rows."""

    limitations: list[str] = []

    train_sources = {str(row.get("source") or "unknown") for row in train_rows}
    test_sources = {str(row.get("source") or "unknown") for row in test_rows}
    source_overlap = _overlap_report(train_sources, test_sources)

    train_paths = _collect_values(train_rows, PATH_FIELDS)
    test_paths = _collect_values(test_rows, PATH_FIELDS)
    if not train_paths and not test_paths:
        limitations.append("No path-like metadata fields found; image/path overlap cannot be checked.")
    path_overlap = _overlap_report(train_paths, test_paths)

    train_filenames = _filename_values(train_rows)
    test_filenames = _filename_values(test_rows)
    filename_overlap = _overlap_report(train_filenames, test_filenames)

    train_session_frames = _session_frame_values(train_rows)
    test_session_frames = _session_frame_values(test_rows)
    if not train_session_frames and not test_session_frames:
        limitations.append("No session metadata fields found; session/frame overlap cannot be checked.")
    session_frame_overlap = _overlap_report(train_session_frames, test_session_frames)

    train_row_hashes = {_row_fingerprint(row) for row in train_rows}
    test_row_hashes = {_row_fingerprint(row) for row in test_rows}
    row_hash_overlap = _overlap_report(train_row_hashes, test_row_hashes)

    train_feature_hashes = {fp for row in train_rows if (fp := _feature_fingerprint(row)) is not None}
    test_feature_hashes = {fp for row in test_rows if (fp := _feature_fingerprint(row)) is not None}
    feature_hash_overlap = _overlap_report(train_feature_hashes, test_feature_hashes)

    overlapped_sources = set(source_overlap["examples"])
    source_overlap_by_class = {
        "train": _class_counts_for_values(train_rows, overlapped_sources, ("source",)),
        "test": _class_counts_for_values(test_rows, overlapped_sources, ("source",)),
    }

    return {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "source_overlap": source_overlap,
        "source_overlap_by_class": source_overlap_by_class,
        "path_overlap": path_overlap,
        "filename_overlap": filename_overlap,
        "session_frame_overlap": session_frame_overlap,
        "exact_row_hash_overlap": row_hash_overlap,
        "feature_vector_hash_overlap": feature_hash_overlap,
        "metadata_limitations": limitations,
    }


def _resolve_image_path(raw: str, image_root: Path | None) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if image_root is not None:
        return image_root / path
    return REPO_ROOT / path


def _average_hash(path: Path, hash_size: int = 8) -> int | None:
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((hash_size, hash_size))
            pixels = list(img.getdata())
    except Exception:
        return None
    avg = sum(pixels) / len(pixels)
    bits = 0
    for idx, px in enumerate(pixels):
        if px >= avg:
            bits |= 1 << idx
    return bits


def _hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def perceptual_hash_overlap(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    image_root: Path | None,
    max_hamming: int,
) -> dict[str, Any]:
    """Optional near-duplicate check when source image paths exist."""

    def build(rows: list[dict[str, Any]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows:
            raw = _first_present(row, PATH_FIELDS)
            if raw is None:
                continue
            path = _resolve_image_path(str(raw), image_root)
            if not path.is_file():
                continue
            ahash = _average_hash(path)
            if ahash is not None:
                out[str(raw)] = ahash
        return out

    train_hashes = build(train_rows)
    test_hashes = build(test_rows)
    examples: list[str] = []
    count = 0
    for tr_path, tr_hash in train_hashes.items():
        for te_path, te_hash in test_hashes.items():
            if _hamming(tr_hash, te_hash) <= max_hamming:
                count += 1
                if len(examples) < 20:
                    examples.append(f"{tr_path} <-> {te_path}")
    return {
        "count": count,
        "examples": examples,
        "train_images_hashed": len(train_hashes),
        "test_images_hashed": len(test_hashes),
        "max_hamming": max_hamming,
    }


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out) + "\n"


def _write_report(
    path: Path,
    report: dict[str, Any],
    train_manifest: list[dict[str, Any]],
    test_manifest: list[dict[str, Any]],
    command: str,
    phash_report: dict[str, Any] | None,
    overlap_details_path: Path | None = None,
    overlap_detail_count: int | None = None,
) -> None:
    versions = package_versions()
    all_manifest = [*train_manifest, *test_manifest]
    rows = [
        ["source overlap", report["source_overlap"]["count"], ", ".join(report["source_overlap"]["examples"]) or "none"],
        ["path overlap", report["path_overlap"]["count"], ", ".join(report["path_overlap"]["examples"]) or "none"],
        [
            "filename overlap",
            report["filename_overlap"]["count"],
            ", ".join(report["filename_overlap"]["examples"]) or "none",
        ],
        [
            "session/frame overlap",
            report["session_frame_overlap"]["count"],
            ", ".join(report["session_frame_overlap"]["examples"]) or "none",
        ],
        [
            "exact row hash overlap",
            report["exact_row_hash_overlap"]["count"],
            ", ".join(report["exact_row_hash_overlap"]["examples"]) or "none",
        ],
        [
            "feature-vector hash overlap",
            report["feature_vector_hash_overlap"]["count"],
            ", ".join(report["feature_vector_hash_overlap"]["examples"]) or "none",
        ],
    ]
    if phash_report is not None:
        rows.append(
            [
                "perceptual hash near-duplicate",
                phash_report["count"],
                ", ".join(phash_report["examples"]) or "none",
            ]
        )
    else:
        rows.append(["perceptual hash near-duplicate", "not run", "requires resolvable image paths and --phash"])

    body = [
        "# Leakage Check Summary",
        "",
        "Status: metadata-level leakage audit. This does not prove absence of leakage.",
        "",
        "Interpretation: no exact JSON-row overlap was found. Feature-vector overlaps are investigated separately. Image-level and near-duplicate leakage remain unverified unless original image path/hash metadata is present.",
        "",
        "## Reproducibility",
        "",
        _md_table(
            ["Field", "Value"],
            [
                ["command", command],
                ["git_commit", git_commit_hash()],
                ["dataset_manifest_hash", manifest_hash(all_manifest)],
                ["train_rows", report["train_rows"]],
                ["test_rows", report["test_rows"]],
                ["split_method", "existing JSONL train/test files; no split performed by this script"],
                ["random_seed", "not used"],
                ["python", versions["python"]],
                ["sklearn", versions["sklearn"]],
                ["numpy", versions["numpy"]],
                ["scipy", versions["scipy"]],
            ],
        ),
        "## Dataset Files",
        "",
        _md_table(
            ["Split", "Path", "SHA256", "Bytes"],
            [["train", m["path"], m["sha256"], m["bytes"]] for m in train_manifest]
            + [["test", m["path"], m["sha256"], m["bytes"]] for m in test_manifest],
        ),
        "## Overlap Checks",
        "",
        _md_table(["Check", "Overlap Count", "Examples"], rows),
        "## Feature-Vector Overlap Details",
        "",
        (
            f"Detailed overlap rows are written to `{overlap_details_path}`."
            if overlap_details_path is not None
            else "Detailed overlap export was not requested."
        ),
        (
            f"Detailed train/test overlap pairs: {overlap_detail_count}."
            if overlap_detail_count is not None
            else ""
        ),
        "",
        "## Source Overlap By Class",
        "",
        _md_table(
            ["Split", "Class", "Rows From Overlapped Sources"],
            [
                [split, label, count]
                for split, counts in report["source_overlap_by_class"].items()
                for label, count in sorted(counts.items())
            ]
            or [["n/a", "n/a", 0]],
        ),
        "## Limitations",
        "",
    ]
    limitations = list(report["metadata_limitations"])
    if phash_report is None:
        limitations.append(
            "Image-level/near-duplicate leakage remains unverified because JSONL lacks original image path/hash metadata or --phash was not run with resolvable image paths."
        )
    if report["path_overlap"]["count"] or report["filename_overlap"]["count"]:
        limitations.append(
            "Path/filename overlap is metadata-level unless source-image SHA256 or perceptual hash is run."
        )
    for item in limitations:
        body.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")


def _write_overlap_details_csv(path: Path, details: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "train_index",
        "test_index",
        "train_label",
        "test_label",
        "train_source",
        "test_source",
        "train_label_source",
        "test_label_source",
        "train_label_confidence",
        "test_label_confidence",
        "train_nir_channel",
        "test_nir_channel",
        "train_thermal_channel",
        "test_thermal_channel",
        "feature_hash",
        "match_type",
        "feature_values_json",
        "feature_diff_json",
        "missing_metadata",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(details)


def _write_overlap_details_md(path: Path, details: list[dict[str, Any]], csv_path: Path | None) -> None:
    rows = [
        [
            d["train_index"],
            d["test_index"],
            d["train_label"],
            d["test_label"],
            d["train_source"],
            d["test_source"],
            d["train_label_confidence"],
            d["test_label_confidence"],
            d["train_nir_channel"],
            d["train_thermal_channel"],
            d["feature_hash"][:12],
            d["match_type"],
            d["missing_metadata"],
        ]
        for d in details
    ]
    body = [
        "# Feature-Vector Overlap Details",
        "",
        "Status: warning-level leakage evidence. These rows have identical `optical_12_baseline` feature vectors across train/test, but image-level duplicate status cannot be confirmed without original image path/hash metadata.",
        "",
        f"Overlap pairs: {len(details)}",
        "",
    ]
    if csv_path is not None:
        body.extend([f"CSV artifact: `{csv_path}`", ""])
    body.extend(
        [
            _md_table(
                [
                    "Train idx",
                    "Test idx",
                    "Train label",
                    "Test label",
                    "Train source",
                    "Test source",
                    "Train conf",
                    "Test conf",
                    "NIR",
                    "Thermal",
                    "Feature hash",
                    "Match",
                    "Missing metadata",
                ],
                rows or [["n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]],
            ),
            "Feature values and per-feature diffs are stored in the CSV as JSON fields.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check metadata-level train/test leakage for SmartBinocular JSONL datasets."
    )
    p.add_argument("--train", type=Path, nargs="+", required=True, help="Training JSONL file(s)")
    p.add_argument("--test", type=Path, nargs="+", required=True, help="Test JSONL file(s)")
    p.add_argument("--out", type=Path, default=Path("docs/tables/ml/leakage_check_summary.md"), help="Output Markdown path")
    p.add_argument("--image-root", type=Path, default=None, help="Optional base directory for relative image paths")
    p.add_argument("--phash", action="store_true", help="Run optional average-hash near-duplicate check when image paths resolve")
    p.add_argument("--phash-max-hamming", type=int, default=4, help="Near-duplicate threshold for 8x8 average hash")
    p.add_argument("--feature-overlap-details-md", type=Path, default=None, help="Optional Markdown export for feature-vector overlap details")
    p.add_argument("--feature-overlap-details-csv", type=Path, default=None, help="Optional CSV export for feature-vector overlap details")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in [*args.train, *args.test]:
        if not path.is_file():
            print(f"ERROR: JSONL not found: {path}", file=sys.stderr)
            return 2

    train_rows = read_jsonl(args.train)
    test_rows = read_jsonl(args.test)
    report = check_metadata_overlap(train_rows, test_rows)
    phash_report = None
    if args.phash:
        phash_report = perceptual_hash_overlap(
            train_rows,
            test_rows,
            image_root=args.image_root,
            max_hamming=args.phash_max_hamming,
        )
    details = feature_vector_overlap_details(train_rows, test_rows)
    if args.feature_overlap_details_csv is not None:
        _write_overlap_details_csv(args.feature_overlap_details_csv, details)
        print(f"Wrote {args.feature_overlap_details_csv}")
    if args.feature_overlap_details_md is not None:
        _write_overlap_details_md(args.feature_overlap_details_md, details, args.feature_overlap_details_csv)
        print(f"Wrote {args.feature_overlap_details_md}")
    command = " ".join([sys.executable, *sys.argv])
    _write_report(
        args.out,
        report,
        train_manifest=dataset_manifest(args.train),
        test_manifest=dataset_manifest(args.test),
        command=command,
        phash_report=phash_report,
        overlap_details_path=args.feature_overlap_details_md,
        overlap_detail_count=len(details),
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
