#!/usr/bin/env python3
"""Q1 fusion evaluation over captured SmartBinocular screenshots.

Question:
    Does NIR-thermal fusion provide measurable improvement over NIR-only
    processing under low-light conditions?

Requirements:
    opencv-python, numpy, matplotlib
    optional: scipy

Notes:
    - This script intentionally does not import pandas. The local Python
      environment used during implementation segfaulted on pandas import.
    - CSV output uses the Python standard library csv module.
    - Strict quantitative claims require near-same-time FUS/OPT pairs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
EPS = 1e-9
TIMESTAMP_FORMATS = (
    ("%Y%m%d-%H%M%S", r"(\d{8}-\d{6})"),
    ("%Y-%m-%d_%H-%M-%S", r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"),
    ("%Y-%m-%d %H:%M:%S", r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"),
)


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    mode: str
    timestamp: dt.datetime
    timestamp_text: str
    timestamp_confidence: str
    timestamp_source: str
    readable: bool
    sidecar_mode: str
    sidecar_timestamp: str


@dataclass
class PairRecord:
    pair_id: str
    fus: ImageRecord
    opt: ImageRecord | None
    time_gap_sec: float | None
    pair_status: str
    timestamp_confidence: str
    is_unique_pair: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SmartBinocular Q1 fusion-vs-NIR evidence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=Path("fusion_captures"))
    parser.add_argument("--out", type=Path, default=Path("q1_results"))
    parser.add_argument("--strict-window-sec", type=float, default=1.0)
    parser.add_argument("--qual-window-sec", type=float, default=20.0)
    parser.add_argument("--crop-top", type=int, default=70)
    parser.add_argument("--sat-threshold", type=int, default=50)
    parser.add_argument("--min-component-area", type=int, default=100)
    parser.add_argument("--min-alignment-inliers", type=int, default=8)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    return parser.parse_args()


def classify_mode(path: Path) -> str:
    name = path.name.lower()
    if "thermal" in name or "thm" in name:
        return "THERMAL"
    if "fusion" in name or "fus" in name:
        return "FUS"
    if any(token in name for token in ("imx", "opt", "optical", "nir")):
        return "OPT"
    return "OTHER"


def parse_filename_timestamp(path: Path) -> tuple[dt.datetime, str, str, str]:
    import re

    for fmt, pattern in TIMESTAMP_FORMATS:
        match = re.search(pattern, path.name)
        if match:
            parsed = dt.datetime.strptime(match.group(1), fmt)
            return parsed, parsed.isoformat(sep=" "), "high", "filename"

    parsed = dt.datetime.fromtimestamp(path.stat().st_mtime)
    return parsed, parsed.isoformat(sep=" "), "low", "mtime"


def read_sidecar(path: Path) -> tuple[str, str]:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return "", ""
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: could not read sidecar {sidecar}: {exc}", file=sys.stderr)
        return "", ""
    return str(data.get("mode", "")), str(data.get("timestamp_iso", ""))


def discover_images(input_dir: Path) -> tuple[list[ImageRecord], dict[str, int]]:
    images: list[ImageRecord] = []
    counts = {
        "total_images": 0,
        "fusion_images": 0,
        "opt_images": 0,
        "thermal_images": 0,
        "other_images": 0,
        "unreadable_images": 0,
        "high_confidence_timestamps": 0,
        "low_confidence_timestamps": 0,
    }

    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        mode = classify_mode(path)
        timestamp, text, confidence, source = parse_filename_timestamp(path)
        sidecar_mode, sidecar_timestamp = read_sidecar(path)
        readable = cv.imread(str(path), cv.IMREAD_COLOR) is not None

        if not readable:
            counts["unreadable_images"] += 1
            print(f"WARNING: unreadable image skipped for metrics: {path}", file=sys.stderr)

        counts["total_images"] += 1
        if mode == "FUS":
            counts["fusion_images"] += 1
        elif mode == "OPT":
            counts["opt_images"] += 1
        elif mode == "THERMAL":
            counts["thermal_images"] += 1
        else:
            counts["other_images"] += 1

        if confidence == "high":
            counts["high_confidence_timestamps"] += 1
        else:
            counts["low_confidence_timestamps"] += 1

        images.append(
            ImageRecord(
                path=path,
                mode=mode,
                timestamp=timestamp,
                timestamp_text=text,
                timestamp_confidence=confidence,
                timestamp_source=source,
                readable=readable,
                sidecar_mode=sidecar_mode,
                sidecar_timestamp=sidecar_timestamp,
            )
        )

    return images, counts


def pair_status_for_gap(
    gap_sec: float | None,
    strict_window_sec: float,
    qual_window_sec: float,
) -> str:
    if gap_sec is None:
        return "reject"
    if gap_sec <= strict_window_sec:
        return "strict_quantitative"
    if gap_sec <= qual_window_sec:
        return "qualitative_pair"
    return "reject"


def timestamp_pair_confidence(fus: ImageRecord, opt: ImageRecord | None) -> str:
    if opt is None:
        return fus.timestamp_confidence
    if fus.timestamp_confidence == "high" and opt.timestamp_confidence == "high":
        return "high"
    return "low"


def build_pairs(
    images: list[ImageRecord],
    strict_window_sec: float,
    qual_window_sec: float,
) -> tuple[list[PairRecord], list[PairRecord]]:
    fus_images = sorted([img for img in images if img.mode == "FUS"], key=lambda x: (x.timestamp, x.path.name))
    opt_images = sorted([img for img in images if img.mode == "OPT"], key=lambda x: (x.timestamp, x.path.name))

    pairs: list[PairRecord] = []
    for idx, fus in enumerate(fus_images, start=1):
        nearest: ImageRecord | None = None
        gap: float | None = None
        if opt_images:
            nearest = min(opt_images, key=lambda opt: abs((opt.timestamp - fus.timestamp).total_seconds()))
            gap = abs((nearest.timestamp - fus.timestamp).total_seconds())
        status = pair_status_for_gap(gap, strict_window_sec, qual_window_sec)
        pairs.append(
            PairRecord(
                pair_id=f"pair_{idx:03d}",
                fus=fus,
                opt=nearest,
                time_gap_sec=gap,
                pair_status=status,
                timestamp_confidence=timestamp_pair_confidence(fus, nearest),
            )
        )

    accepted = [p for p in pairs if p.pair_status != "reject" and p.opt is not None]
    unique_pairs: list[PairRecord] = []
    used_opts: set[Path] = set()
    used_fus: set[Path] = set()
    for pair in sorted(accepted, key=lambda p: (p.time_gap_sec if p.time_gap_sec is not None else math.inf, p.pair_id)):
        assert pair.opt is not None
        if pair.fus.path in used_fus or pair.opt.path in used_opts:
            continue
        pair.is_unique_pair = True
        unique_pairs.append(pair)
        used_fus.add(pair.fus.path)
        used_opts.add(pair.opt.path)

    return pairs, unique_pairs


def pair_to_row(pair: PairRecord) -> dict[str, Any]:
    opt = pair.opt
    return {
        "pair_id": pair.pair_id,
        "fus_path": str(pair.fus.path),
        "opt_path": "" if opt is None else str(opt.path),
        "fus_timestamp": pair.fus.timestamp_text,
        "opt_timestamp": "" if opt is None else opt.timestamp_text,
        "time_gap_sec": "" if pair.time_gap_sec is None else round(pair.time_gap_sec, 6),
        "pair_status": pair.pair_status,
        "timestamp_confidence": pair.timestamp_confidence,
        "fus_timestamp_confidence": pair.fus.timestamp_confidence,
        "opt_timestamp_confidence": "" if opt is None else opt.timestamp_confidence,
        "fus_timestamp_source": pair.fus.timestamp_source,
        "opt_timestamp_source": "" if opt is None else opt.timestamp_source,
        "fus_readable": pair.fus.readable,
        "opt_readable": "" if opt is None else opt.readable,
        "is_unique_pair": pair.is_unique_pair,
    }


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": out_dir,
        "plots": out_dir / "plots",
        "debug_masks": out_dir / "debug_masks",
        "contact_sheets": out_dir / "contact_sheets",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            safe_row = {key: format_csv_value(row.get(key, "")) for key in fieldnames}
            writer.writerow(safe_row)


def format_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8g}"
    return value


def crop_hud(bgr: np.ndarray, crop_top: int) -> np.ndarray:
    top = max(0, min(int(crop_top), max(0, bgr.shape[0] - 1)))
    return bgr[top:, :, :].copy()


def resize_to_match(src_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    h, w = target_bgr.shape[:2]
    if src_bgr.shape[:2] == (h, w):
        return src_bgr
    return cv.resize(src_bgr, (w, h), interpolation=cv.INTER_LINEAR)


def colorful_mask(
    bgr: np.ndarray,
    sat_threshold: int,
    min_component_area: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + b * b)

    mask = ((sat > sat_threshold) & (chroma > 12.0) & (val > 20)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.zeros(mask.shape, dtype=np.uint8)
    component_count = 0
    largest_area = 0
    for label in range(1, count):
        area = int(stats[label, cv.CC_STAT_AREA])
        if area >= min_component_area:
            kept[labels == label] = 255
            component_count += 1
            largest_area = max(largest_area, area)

    area = int(np.count_nonzero(kept))
    area_ratio = float(area) / float(kept.size) if kept.size else 0.0
    if area < min_component_area or component_count == 0:
        status = "insufficient"
    elif area_ratio > 0.60:
        status = "excessive"
    else:
        status = "sufficient"

    info = {
        "target_mask_area": area,
        "target_mask_area_ratio": area_ratio,
        "target_component_count": component_count,
        "target_largest_component_area": largest_area,
        "target_roi_status": status,
    }
    return kept, info


def feature_mask_for_fus(fus_bgr: np.ndarray, sat_threshold: int) -> np.ndarray:
    mask, _ = colorful_mask(fus_bgr, sat_threshold=sat_threshold, min_component_area=25)
    inverse = cv.bitwise_not(mask)
    kernel = np.ones((5, 5), np.uint8)
    inverse = cv.erode(inverse, kernel, iterations=1)
    return inverse


def transform_reasonable(matrix: np.ndarray, transform_type: str, width: int, height: int) -> bool:
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    try:
        if transform_type == "homography":
            warped = cv.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        else:
            affine = matrix.reshape(2, 3)
            warped = cv.transform(corners.reshape(-1, 1, 2), affine).reshape(-1, 2)
    except Exception:
        return False
    if not np.isfinite(warped).all():
        return False
    bbox_w = float(warped[:, 0].max() - warped[:, 0].min())
    bbox_h = float(warped[:, 1].max() - warped[:, 1].min())
    if bbox_w < width * 0.2 or bbox_w > width * 3.0:
        return False
    if bbox_h < height * 0.2 or bbox_h > height * 3.0:
        return False
    src_center = np.array([width / 2.0, height / 2.0])
    dst_center = warped.mean(axis=0)
    max_shift = max(width, height) * 0.75
    return float(np.linalg.norm(dst_center - src_center)) <= max_shift


def align_opt_to_fus(
    fus_bgr: np.ndarray,
    opt_bgr: np.ndarray,
    sat_threshold: int,
    min_inliers: int,
    min_inlier_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = fus_bgr.shape[:2]
    opt_bgr = resize_to_match(opt_bgr, fus_bgr)
    fus_gray = cv.cvtColor(fus_bgr, cv.COLOR_BGR2GRAY)
    opt_gray = cv.cvtColor(opt_bgr, cv.COLOR_BGR2GRAY)
    fus_mask = feature_mask_for_fus(fus_bgr, sat_threshold)

    orb = cv.ORB_create(nfeatures=1500, fastThreshold=7)
    kp_fus, des_fus = orb.detectAndCompute(fus_gray, fus_mask)
    kp_opt, des_opt = orb.detectAndCompute(opt_gray, None)

    metrics: dict[str, Any] = {
        "alignment_status": "failed_or_unusable",
        "alignment_notes": "",
        "alignment_transform_type": "none",
        "alignment_fus_keypoints": len(kp_fus) if kp_fus is not None else 0,
        "alignment_opt_keypoints": len(kp_opt) if kp_opt is not None else 0,
        "alignment_matches": 0,
        "alignment_inliers": 0,
        "alignment_inlier_ratio": 0.0,
    }

    if des_fus is None or des_opt is None or len(kp_fus) < 8 or len(kp_opt) < 8:
        metrics["alignment_notes"] = "insufficient_keypoints"
        return opt_bgr, metrics

    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_opt, des_fus)
    matches = sorted(matches, key=lambda m: m.distance)[:250]
    metrics["alignment_matches"] = len(matches)
    if len(matches) < 8:
        metrics["alignment_notes"] = "insufficient_matches"
        return opt_bgr, metrics

    src = np.float32([kp_opt[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_fus[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    hmat, hmask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
    if hmat is not None and hmask is not None:
        inliers = int(hmask.ravel().sum())
        ratio = float(inliers) / float(len(matches))
        if (
            inliers >= min_inliers
            and ratio >= min_inlier_ratio
            and transform_reasonable(hmat, "homography", w, h)
        ):
            warped = cv.warpPerspective(opt_bgr, hmat, (w, h), flags=cv.INTER_LINEAR)
            metrics.update(
                {
                    "alignment_status": "usable",
                    "alignment_transform_type": "homography",
                    "alignment_inliers": inliers,
                    "alignment_inlier_ratio": ratio,
                }
            )
            return warped, metrics
        metrics.update({"alignment_inliers": inliers, "alignment_inlier_ratio": ratio})

    amat, amask = cv.estimateAffinePartial2D(src, dst, method=cv.RANSAC, ransacReprojThreshold=5.0)
    if amat is not None and amask is not None:
        inliers = int(amask.ravel().sum())
        ratio = float(inliers) / float(len(matches))
        if (
            inliers >= min_inliers
            and ratio >= min_inlier_ratio
            and transform_reasonable(amat, "affine", w, h)
        ):
            warped = cv.warpAffine(opt_bgr, amat, (w, h), flags=cv.INTER_LINEAR)
            metrics.update(
                {
                    "alignment_status": "usable",
                    "alignment_transform_type": "affine",
                    "alignment_inliers": inliers,
                    "alignment_inlier_ratio": ratio,
                }
            )
            return warped, metrics
        if inliers > int(metrics["alignment_inliers"]):
            metrics.update({"alignment_inliers": inliers, "alignment_inlier_ratio": ratio})

    metrics["alignment_notes"] = "ransac_failed_quality_gate"
    return opt_bgr, metrics


def luma_channel(bgr: np.ndarray) -> np.ndarray:
    return cv.cvtColor(bgr, cv.COLOR_BGR2YCrCb)[:, :, 0]


def entropy_uint8(gray: np.ndarray) -> float:
    hist = cv.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    prob = hist / (hist.sum() + EPS)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def tenengrad(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    score = gx * gx + gy * gy
    if mask is not None:
        valid = mask.astype(bool)
        if not np.any(valid):
            return float("nan")
        return float(np.mean(score[valid]))
    return float(np.mean(score))


def full_frame_metrics(bgr: np.ndarray) -> dict[str, float]:
    luma = luma_channel(bgr)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    mean = float(np.mean(luma))
    std = float(np.std(luma))
    lap = cv.Laplacian(gray, cv.CV_32F)
    return {
        "mean_luma": mean,
        "std_luma": std,
        "rms_contrast": std / (mean + EPS),
        "entropy_luma": entropy_uint8(luma),
        "pct_crushed": float(np.mean(luma < 10.0)) * 100.0,
        "pct_saturated": float(np.mean(luma > 245.0)) * 100.0,
        "sharpness_laplacian_var": float(np.var(lap)),
        "tenengrad_score": tenengrad(gray),
    }


def valid_background_mask(mask: np.ndarray, border: int = 5) -> np.ndarray:
    target = mask.astype(bool)
    background = ~target
    if background.shape[0] > border * 2 and background.shape[1] > border * 2:
        background[:border, :] = False
        background[-border:, :] = False
        background[:, :border] = False
        background[:, -border:] = False
    return background


def target_salience_metrics(bgr: np.ndarray, target_mask: np.ndarray) -> dict[str, float]:
    target = target_mask.astype(bool)
    background = valid_background_mask(target_mask)
    if not np.any(target) or not np.any(background):
        return {
            "target_mean_luma": float("nan"),
            "background_mean_luma": float("nan"),
            "background_std_luma": float("nan"),
            "luma_cnr": float("nan"),
            "target_mean_chroma": float("nan"),
            "background_mean_chroma": float("nan"),
            "background_std_chroma": float("nan"),
            "chroma_cnr": float("nan"),
            "target_tenengrad": float("nan"),
            "non_target_texture_score": float("nan"),
        }

    luma = luma_channel(bgr).astype(np.float32)
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + b * b)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    target_luma = float(np.mean(luma[target]))
    bg_luma = float(np.mean(luma[background]))
    bg_luma_std = float(np.std(luma[background]))
    target_chroma = float(np.mean(chroma[target]))
    bg_chroma = float(np.mean(chroma[background]))
    bg_chroma_std = float(np.std(chroma[background]))

    return {
        "target_mean_luma": target_luma,
        "background_mean_luma": bg_luma,
        "background_std_luma": bg_luma_std,
        "luma_cnr": abs(target_luma - bg_luma) / (bg_luma_std + EPS),
        "target_mean_chroma": target_chroma,
        "background_mean_chroma": bg_chroma,
        "background_std_chroma": bg_chroma_std,
        "chroma_cnr": abs(target_chroma - bg_chroma) / (bg_chroma_std + EPS),
        "target_tenengrad": tenengrad(gray, target),
        "non_target_texture_score": tenengrad(gray, background),
    }


def prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def pct_improvement(new_value: float, old_value: float) -> float:
    if not math.isfinite(new_value) or not math.isfinite(old_value):
        return float("nan")
    if abs(old_value) < 1e-6:
        return float("nan")
    return ((new_value - old_value) / abs(old_value)) * 100.0


def classify_evidence_strength(
    pair: PairRecord,
    readable: bool,
    target_roi_status: str,
    alignment_status: str,
) -> str:
    if pair.pair_status == "reject" or not readable:
        return "not_comparable"
    if target_roi_status != "sufficient":
        return "visual_only"
    if pair.pair_status == "strict_quantitative":
        if pair.timestamp_confidence == "high" and alignment_status == "usable":
            return "strong_quantitative"
        return "moderate_quantitative"
    if pair.pair_status == "qualitative_pair":
        return "weak_qualitative"
    return "not_comparable"


def overlay_mask(fus_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = fus_bgr.copy()
    color = np.zeros_like(fus_bgr)
    color[:, :, 2] = 255
    color[:, :, 1] = 180
    alpha = 0.45
    masked = mask.astype(bool)
    overlay[masked] = cv.addWeighted(fus_bgr, 1.0 - alpha, color, alpha, 0)[masked]
    return overlay


def annotate_panel(image: np.ndarray, lines: list[str]) -> np.ndarray:
    panel = image.copy()
    y = 24
    for line in lines:
        cv.putText(panel, line, (12, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(panel, line, (12, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)
        y += 24
    return panel


def write_contact_sheet(
    path: Path,
    pair: PairRecord,
    fus_bgr: np.ndarray,
    opt_bgr: np.ndarray,
    target_overlay: np.ndarray,
    row: dict[str, Any],
) -> None:
    panels = [
        annotate_panel(opt_bgr, ["OPT / NIR-only"]),
        annotate_panel(fus_bgr, ["FUS / NIR-thermal"]),
        annotate_panel(target_overlay, ["Target mask overlay"]),
    ]
    sheet = np.hstack(panels)
    header_h = 112
    header = np.zeros((header_h, sheet.shape[1], 3), dtype=np.uint8)
    header[:] = (24, 24, 24)
    lines = [
        f"{pair.pair_id}  gap={row.get('time_gap_sec', '')}s  status={pair.pair_status}",
        f"delta_luma_cnr={format_float(row.get('delta_luma_cnr'))}  "
        f"delta_chroma_cnr={format_float(row.get('delta_chroma_cnr'))}",
        f"alignment={row.get('alignment_status', '')}  evidence={row.get('evidence_strength', '')}",
    ]
    y = 28
    for line in lines:
        cv.putText(header, line, (16, y), cv.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv.LINE_AA)
        y += 30
    out = np.vstack([header, sheet])
    cv.imwrite(str(path), out)


def format_float(value: Any, digits: int = 3) -> str:
    try:
        f = float(value)
    except Exception:
        return ""
    if not math.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def process_pair(
    pair: PairRecord,
    args: argparse.Namespace,
    dirs: dict[str, Path],
) -> dict[str, Any]:
    assert pair.opt is not None
    base_row: dict[str, Any] = {
        **pair_to_row(pair),
        "readable_pair": pair.fus.readable and pair.opt.readable,
        "processing_error": "",
    }

    fus_raw = cv.imread(str(pair.fus.path), cv.IMREAD_COLOR)
    opt_raw = cv.imread(str(pair.opt.path), cv.IMREAD_COLOR)
    if fus_raw is None or opt_raw is None:
        base_row.update(
            {
                "readable_pair": False,
                "processing_error": "unreadable_image",
                "alignment_status": "not_attempted",
                "target_roi_status": "not_attempted",
                "evidence_strength": "not_comparable",
            }
        )
        return base_row

    fus = crop_hud(fus_raw, args.crop_top)
    opt = resize_to_match(crop_hud(opt_raw, args.crop_top), fus)
    opt_aligned, alignment = align_opt_to_fus(
        fus,
        opt,
        sat_threshold=args.sat_threshold,
        min_inliers=args.min_alignment_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
    )
    target_mask, target_info = colorful_mask(
        fus,
        sat_threshold=args.sat_threshold,
        min_component_area=args.min_component_area,
    )

    overlay = overlay_mask(fus, target_mask)
    mask_path = dirs["debug_masks"] / f"{pair.pair_id}_target_mask.png"
    overlay_path = dirs["debug_masks"] / f"{pair.pair_id}_overlay.png"
    cv.imwrite(str(mask_path), target_mask)
    cv.imwrite(str(overlay_path), overlay)

    row = {**base_row, **alignment, **target_info}
    row["debug_target_mask_path"] = str(mask_path)
    row["debug_overlay_path"] = str(overlay_path)

    fus_full = full_frame_metrics(fus)
    opt_full = full_frame_metrics(opt_aligned)
    row.update(prefixed("fus", fus_full))
    row.update(prefixed("opt", opt_full))
    row.update(
        {
            "delta_entropy": fus_full["entropy_luma"] - opt_full["entropy_luma"],
            "delta_rms_contrast": fus_full["rms_contrast"] - opt_full["rms_contrast"],
            "delta_pct_crushed": fus_full["pct_crushed"] - opt_full["pct_crushed"],
            "delta_pct_saturated": fus_full["pct_saturated"] - opt_full["pct_saturated"],
            "delta_sharpness": fus_full["sharpness_laplacian_var"] - opt_full["sharpness_laplacian_var"],
            "delta_tenengrad": fus_full["tenengrad_score"] - opt_full["tenengrad_score"],
        }
    )

    if target_info["target_roi_status"] == "sufficient":
        fus_target = target_salience_metrics(fus, target_mask)
        opt_target = target_salience_metrics(opt_aligned, target_mask)
        row.update(prefixed("fus", fus_target))
        row.update(prefixed("opt", opt_target))
        row.update(
            {
                "delta_luma_cnr": fus_target["luma_cnr"] - opt_target["luma_cnr"],
                "delta_chroma_cnr": fus_target["chroma_cnr"] - opt_target["chroma_cnr"],
                "pct_improvement_luma_cnr": pct_improvement(fus_target["luma_cnr"], opt_target["luma_cnr"]),
                "pct_improvement_chroma_cnr": pct_improvement(fus_target["chroma_cnr"], opt_target["chroma_cnr"]),
                "delta_non_target_texture_score": (
                    fus_target["non_target_texture_score"] - opt_target["non_target_texture_score"]
                ),
            }
        )
    else:
        for key in (
            "delta_luma_cnr",
            "delta_chroma_cnr",
            "pct_improvement_luma_cnr",
            "pct_improvement_chroma_cnr",
            "delta_non_target_texture_score",
        ):
            row[key] = float("nan")

    row["evidence_strength"] = classify_evidence_strength(
        pair,
        readable=bool(row["readable_pair"]),
        target_roi_status=str(target_info["target_roi_status"]),
        alignment_status=str(alignment["alignment_status"]),
    )

    contact_path = dirs["contact_sheets"] / f"{pair.pair_id}.png"
    write_contact_sheet(contact_path, pair, fus, opt_aligned, overlay, row)
    row["contact_sheet_path"] = str(contact_path)
    return row


def finite_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except Exception:
            continue
        if math.isfinite(value):
            vals.append(value)
    return vals


def iqr(values: list[float]) -> float:
    if not values:
        return float("nan")
    q75, q25 = np.percentile(np.array(values, dtype=np.float64), [75, 25])
    return float(q75 - q25)


def bootstrap_ci_median(values: list[float], iterations: int, seed: int = 42) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(max(1, iterations)):
        draw = rng.choice(arr, size=len(arr), replace=True)
        samples.append(float(np.median(draw)))
    low, high = np.percentile(np.array(samples), [2.5, 97.5])
    return float(low), float(high)


def sign_test_p(values: list[float]) -> float:
    nonzero = [v for v in values if abs(v) > EPS]
    n = len(nonzero)
    if n == 0:
        return float("nan")
    positives = sum(1 for v in nonzero if v > 0)
    k = min(positives, n - positives)
    p = 2.0 * sum(math.comb(n, i) * (0.5**n) for i in range(0, k + 1))
    return min(1.0, p)


def wilcoxon_p(values: list[float]) -> float:
    nonzero = [v for v in values if abs(v) > EPS]
    if len(nonzero) < 2:
        return float("nan")
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def summarize_group(
    group_name: str,
    rows: list[dict[str, Any]],
    all_pairs: list[PairRecord],
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = [
        "delta_luma_cnr",
        "delta_chroma_cnr",
        "delta_entropy",
        "delta_rms_contrast",
        "delta_pct_saturated",
        "delta_pct_crushed",
        "delta_sharpness",
    ]
    summary: dict[str, Any] = {
        "group": group_name,
        "valid_pair_count": len([p for p in all_pairs if p.pair_status != "reject"]),
        "strict_pair_count": len([p for p in all_pairs if p.pair_status == "strict_quantitative"]),
        "qualitative_pair_count": len([p for p in all_pairs if p.pair_status == "qualitative_pair"]),
        "rejected_pair_count": len([p for p in all_pairs if p.pair_status == "reject"]),
        "metric_row_count": len(rows),
        "median_time_gap_sec": safe_median(finite_values(rows, "time_gap_sec")),
    }
    for metric in metrics:
        vals = finite_values(rows, metric)
        ci_low, ci_high = bootstrap_ci_median(vals, args.bootstrap_iters)
        summary[f"mean_{metric}"] = statistics.fmean(vals) if vals else float("nan")
        summary[f"median_{metric}"] = safe_median(vals)
        summary[f"std_{metric}"] = statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else float("nan")
        summary[f"iqr_{metric}"] = iqr(vals)
        summary[f"median_{metric}_ci95_low"] = ci_low
        summary[f"median_{metric}_ci95_high"] = ci_high
        summary[f"sign_test_p_{metric}"] = sign_test_p(vals)
        summary[f"wilcoxon_p_{metric}"] = wilcoxon_p(vals)
    return summary


def safe_median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def select_rows(rows: list[dict[str, Any]], *, status: str | None = None, unique: bool | None = None) -> list[dict[str, Any]]:
    selected = rows
    if status is not None:
        selected = [row for row in selected if row.get("pair_status") == status]
    if unique is not None:
        selected = [row for row in selected if str(row.get("is_unique_pair")) == str(unique)]
    return selected


def final_evidence_classification(unique_rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> str:
    rows = unique_rows if unique_rows else all_rows
    if not rows:
        return "not enough evidence"

    strict_rows = [row for row in unique_rows if row.get("pair_status") == "strict_quantitative"]
    qual_rows = [row for row in unique_rows if row.get("pair_status") == "qualitative_pair"]
    luma_med = safe_median(finite_values(rows, "delta_luma_cnr"))
    chroma_med = safe_median(finite_values(rows, "delta_chroma_cnr"))
    salience_improves = (math.isfinite(luma_med) and luma_med > 0.0) or (math.isfinite(chroma_med) and chroma_med > 0.0)
    has_target = any(row.get("target_roi_status") == "sufficient" for row in rows)

    if strict_rows and has_target and math.isfinite(luma_med) and luma_med > 0.0:
        return "strict quantitative support"
    if qual_rows and has_target and salience_improves:
        return "weak/qualitative support"
    return "not enough evidence"


def make_boxplot(rows: list[dict[str, Any]], key: str, title: str, path: Path) -> None:
    groups = [
        ("All", finite_values(rows, key)),
        ("Strict", finite_values(select_rows(rows, status="strict_quantitative"), key)),
        ("Qual", finite_values(select_rows(rows, status="qualitative_pair"), key)),
        ("Unique", finite_values(select_rows(rows, unique=True), key)),
    ]
    labels = [name for name, vals in groups if vals]
    data = [vals for _, vals in groups if vals]
    plt.figure(figsize=(7, 4.5))
    if data:
        plt.boxplot(data, tick_labels=labels, showmeans=True)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel(key)
    else:
        plt.text(0.5, 0.5, "No finite data", ha="center", va="center")
        plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_time_gap_histogram(pairs: list[PairRecord], args: argparse.Namespace, path: Path) -> None:
    gaps = [p.time_gap_sec for p in pairs if p.time_gap_sec is not None]
    plt.figure(figsize=(7, 4.5))
    if gaps:
        plt.hist(gaps, bins=min(20, max(5, len(gaps))), color="#4C78A8", edgecolor="white")
        plt.axvline(args.strict_window_sec, color="#54A24B", linestyle="--", label="strict window")
        plt.axvline(args.qual_window_sec, color="#E45756", linestyle="--", label="qual window")
        plt.xlabel("Nearest FUS/OPT time gap (s)")
        plt.ylabel("Pair count")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No pair gaps", ha="center", va="center")
        plt.axis("off")
    plt.title("FUS to nearest OPT time gaps")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_metric_summary_bar(rows: list[dict[str, Any]], path: Path) -> None:
    keys = [
        "delta_luma_cnr",
        "delta_chroma_cnr",
        "delta_entropy",
        "delta_rms_contrast",
        "delta_pct_saturated",
        "delta_pct_crushed",
    ]
    labels = [key.replace("delta_", "") for key in keys]
    medians = [safe_median(finite_values(rows, key)) for key in keys]
    colors = ["#54A24B" if math.isfinite(v) and v >= 0 else "#E45756" for v in medians]
    values = [0.0 if not math.isfinite(v) else v for v in medians]
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(labels, values, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Median FUS - OPT delta")
    plt.title("Median Q1 metric deltas")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_plots(rows: list[dict[str, Any]], pairs: list[PairRecord], args: argparse.Namespace, dirs: dict[str, Path]) -> None:
    make_boxplot(rows, "delta_luma_cnr", "Delta luma CNR (FUS - OPT)", dirs["plots"] / "delta_luma_cnr_boxplot.png")
    make_boxplot(rows, "delta_chroma_cnr", "Delta chroma CNR (FUS - OPT)", dirs["plots"] / "delta_chroma_cnr_boxplot.png")
    make_boxplot(rows, "delta_entropy", "Delta entropy (FUS - OPT)", dirs["plots"] / "delta_entropy_boxplot.png")
    make_time_gap_histogram(pairs, args, dirs["plots"] / "time_gap_histogram.png")
    make_metric_summary_bar(rows, dirs["plots"] / "metric_summary_bar.png")


def metric_table_markdown(summary: dict[str, Any]) -> str:
    metrics = [
        "delta_luma_cnr",
        "delta_chroma_cnr",
        "delta_entropy",
        "delta_rms_contrast",
        "delta_pct_saturated",
        "delta_pct_crushed",
    ]
    lines = [
        "| Metric | Median | 95% bootstrap CI | Sign p | Wilcoxon p |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        median = summary.get(f"median_{metric}", float("nan"))
        low = summary.get(f"median_{metric}_ci95_low", float("nan"))
        high = summary.get(f"median_{metric}_ci95_high", float("nan"))
        sign_p = summary.get(f"sign_test_p_{metric}", float("nan"))
        wil_p = summary.get(f"wilcoxon_p_{metric}", float("nan"))
        lines.append(
            f"| `{metric}` | {format_float(median, 4)} | "
            f"[{format_float(low, 4)}, {format_float(high, 4)}] | "
            f"{format_float(sign_p, 4)} | {format_float(wil_p, 4)} |"
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    counts: dict[str, int],
    pairs: list[PairRecord],
    unique_pairs: list[PairRecord],
    metric_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    final_classification: str,
) -> None:
    strict_count = len([p for p in pairs if p.pair_status == "strict_quantitative"])
    qual_count = len([p for p in pairs if p.pair_status == "qualitative_pair"])
    rejected_count = len([p for p in pairs if p.pair_status == "reject"])
    valid_count = strict_count + qual_count
    unique_strict = len([p for p in unique_pairs if p.pair_status == "strict_quantitative"])
    unique_qual = len([p for p in unique_pairs if p.pair_status == "qualitative_pair"])
    all_summary = next((s for s in summaries if s["group"] == "all_accepted"), {})
    unique_summary = next((s for s in summaries if s["group"] == "unique_accepted"), all_summary)
    table_source = unique_summary if unique_pairs else all_summary

    unique_rows = [row for row in metric_rows if str(row.get("is_unique_pair")) == "True"]
    interpretation_rows = unique_rows if unique_rows else metric_rows
    median_luma = safe_median(finite_values(interpretation_rows, "delta_luma_cnr"))
    median_chroma = safe_median(finite_values(interpretation_rows, "delta_chroma_cnr"))
    median_entropy = safe_median(finite_values(interpretation_rows, "delta_entropy"))
    median_sat = safe_median(finite_values(interpretation_rows, "delta_pct_saturated"))
    median_gap = safe_median(finite_values(interpretation_rows, "time_gap_sec"))

    lines = [
        "# Q1 Fusion Evaluation Report",
        "",
        "## Q1 statement",
        "",
        '"Does NIR\u2013thermal fusion provide measurable improvement over NIR-only processing under low-light conditions?"',
        "",
        "## Dataset summary",
        "",
        f"- Images found: {counts['total_images']}",
        f"- Fusion images: {counts['fusion_images']}",
        f"- OPT/NIR/IMX images: {counts['opt_images']}",
        f"- Thermal-only images: {counts['thermal_images']} (counted, excluded from FUS-vs-OPT pairing)",
        f"- Other images: {counts['other_images']}",
        f"- Unreadable images: {counts['unreadable_images']}",
        f"- Nearest-pair rows: {len(pairs)}",
        f"- Accepted nearest pairs: {valid_count}",
        f"- Strict quantitative pairs: {strict_count}",
        f"- Qualitative pairs: {qual_count}",
        f"- Rejected pairs: {rejected_count}",
        f"- Unique accepted pairs: {len(unique_pairs)} ({unique_strict} strict, {unique_qual} qualitative)",
        "",
        "## Pair validity warning",
        "",
    ]
    if strict_count == 0:
        lines.append(
            "No strict same-time FUS/OPT pairs were found. Quantitative Q1 claims are not valid from this folder alone. "
            "Results below should be treated as qualitative or weakly paired evidence."
        )
    else:
        lines.append("Strict paired outputs are available; quantitative comparison is more reliable.")

    lines.extend(
        [
            "",
            "## Metric summary",
            "",
            "The table below uses the unique accepted subset when available, because it prevents one OPT frame from being reused across many FUS frames.",
            "",
            metric_table_markdown(table_source),
            "",
            "## Interpretation",
            "",
            "### Optical/luminance quality",
            "",
            "Full-frame luminance metrics summarize display brightness, tonal spread, contrast, clipping, and sharpness after the HUD crop. "
            "They are useful quality proxies, but they are not direct evidence that fusion improves target detection.",
            "",
        ]
    )
    if math.isfinite(median_entropy) and median_entropy > 0:
        lines.append(
            f"- Entropy increased in the evaluated subset (median delta {median_entropy:.4f}), "
            "but this may partly reflect the added thermal color overlay rather than purely better optical detail."
        )
    elif math.isfinite(median_entropy):
        lines.append(f"- Entropy did not increase in the evaluated subset (median delta {median_entropy:.4f}).")
    if math.isfinite(median_sat) and median_sat > 0:
        lines.append(
            f"- Luma saturation increased (median delta {median_sat:.4f} percentage points). "
            "Fusion also increases color saturation due to the thermal overlay; this should not be interpreted as optical overexposure unless luma saturation also increases."
        )
    elif math.isfinite(median_sat):
        lines.append(f"- Luma saturation did not increase (median delta {median_sat:.4f} percentage points).")

    lines.extend(
        [
            "",
            "### Fusion-specific target salience",
            "",
            "Target salience metrics use colorful thermal-overlay regions from the FUS image as candidate warm-target regions. "
            "These metrics are more relevant for Q1 than full-frame entropy because fusion is expected to improve warm-object observability.",
            "",
        ]
    )
    if math.isfinite(median_luma) and median_luma > 0:
        lines.append(f"- Fusion improves warm-target luma salience in the evaluated pairs (median delta luma CNR {median_luma:.4f}).")
    elif math.isfinite(median_luma):
        lines.append(f"- Warm-target luma CNR did not improve in the evaluated subset (median delta {median_luma:.4f}).")
    if math.isfinite(median_chroma) and median_chroma > 0:
        lines.append(
            f"- Chroma CNR increased (median delta {median_chroma:.4f}), but chroma CNR can be inflated by the thermal color overlay."
        )
    elif math.isfinite(median_chroma):
        lines.append(f"- Chroma CNR did not increase (median delta {median_chroma:.4f}).")
    if math.isfinite(median_gap) and median_gap > 1.0:
        lines.append(
            f"- The median accepted time gap is {median_gap:.2f} s. The result is not a strict pixel-level comparison because FUS and OPT were captured at different times."
        )

    lines.extend(
        [
            "",
            "## Final Q1 evidence classification",
            "",
            f"`{final_classification}`",
            "",
            "Because this folder has no strict same-time FUS/OPT pairs, it cannot support strict quantitative Q1 claims.",
            "",
            "## Recommended thesis wording",
            "",
        ]
    )
    if strict_count > 0:
        lines.append(
            "For the available strict FUS/OPT pairs, fusion increased warm-target salience metrics relative to NIR-only output. "
            "The conclusion should be limited to the evaluated scenes and should distinguish target salience from full-frame optical quality."
        )
    else:
        lines.append(
            "The available capture folder contains no strict same-time FUS/OPT pairs; therefore it cannot be used as strict quantitative evidence for Q1. "
            "The weakly paired qualitative results can be cited only as visual/diagnostic evidence that the fusion overlay may improve warm-target salience, "
            "with the explicit caveat that time gaps and thermal color overlay confound pixel-level metric comparisons."
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_slide_summary(
    path: Path,
    counts: dict[str, int],
    pairs: list[PairRecord],
    unique_pairs: list[PairRecord],
    metric_rows: list[dict[str, Any]],
    final_classification: str,
) -> None:
    unique_rows = [row for row in metric_rows if str(row.get("is_unique_pair")) == "True"]
    rows = unique_rows if unique_rows else metric_rows
    strict_count = len([p for p in pairs if p.pair_status == "strict_quantitative"])
    qual_count = len([p for p in pairs if p.pair_status == "qualitative_pair"])
    rejected_count = len([p for p in pairs if p.pair_status == "reject"])
    lines = [
        "# Q1 Slide Summary",
        "",
        "## Slide-ready bullets",
        "",
        f"- Dataset: {counts['fusion_images']} fusion, {counts['opt_images']} OPT/NIR/IMX, {counts['thermal_images']} thermal-only images.",
        f"- Pairing: {strict_count} strict, {qual_count} qualitative, {rejected_count} rejected nearest pairs.",
        f"- Unique accepted subset: {len(unique_pairs)} pair(s), used to avoid reusing one OPT frame many times.",
        f"- Median delta luma CNR: {format_float(safe_median(finite_values(rows, 'delta_luma_cnr')), 4)}.",
        f"- Median delta chroma CNR: {format_float(safe_median(finite_values(rows, 'delta_chroma_cnr')), 4)}.",
        f"- Median delta entropy: {format_float(safe_median(finite_values(rows, 'delta_entropy')), 4)}.",
        f"- Evidence classification: `{final_classification}`.",
        "",
        "## Caveat to show on slide",
        "",
        "No strict same-time FUS/OPT pairs were found in this folder, so this is not strict quantitative Q1 evidence. "
        "Entropy and chroma CNR can be inflated by the thermal color overlay; target salience is the more relevant metric.",
        "",
        "## Suggested conclusion",
        "",
        "The current captures provide weak qualitative evidence at most. A defensible quantitative Q1 claim requires synchronized or near-same-time FUS/OPT captures.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_final_summary(
    counts: dict[str, int],
    pairs: list[PairRecord],
    unique_pairs: list[PairRecord],
    metric_rows: list[dict[str, Any]],
    final_classification: str,
) -> None:
    strict_count = len([p for p in pairs if p.pair_status == "strict_quantitative"])
    qual_count = len([p for p in pairs if p.pair_status == "qualitative_pair"])
    rejected_count = len([p for p in pairs if p.pair_status == "reject"])
    unique_rows = [row for row in metric_rows if str(row.get("is_unique_pair")) == "True"]
    rows = unique_rows if unique_rows else metric_rows
    print("\nQ1 final summary")
    print("----------------")
    print(f"Images: total={counts['total_images']} FUS={counts['fusion_images']} OPT={counts['opt_images']} thermal={counts['thermal_images']}")
    print(f"Pairs: strict={strict_count} qualitative={qual_count} rejected={rejected_count} unique_accepted={len(unique_pairs)}")
    print(f"median_delta_luma_cnr={format_float(safe_median(finite_values(rows, 'delta_luma_cnr')), 6)}")
    print(f"median_delta_chroma_cnr={format_float(safe_median(finite_values(rows, 'delta_chroma_cnr')), 6)}")
    print(f"median_delta_entropy={format_float(safe_median(finite_values(rows, 'delta_entropy')), 6)}")
    print(f"median_delta_rms_contrast={format_float(safe_median(finite_values(rows, 'delta_rms_contrast')), 6)}")
    print(f"median_delta_pct_saturated={format_float(safe_median(finite_values(rows, 'delta_pct_saturated')), 6)}")
    print(f"median_delta_pct_crushed={format_float(safe_median(finite_values(rows, 'delta_pct_crushed')), 6)}")
    print(f"evidence_classification={final_classification}")
    if strict_count == 0:
        print("WARNING: no strict same-time FUS/OPT pairs; strict quantitative Q1 claims are not valid from this folder alone.")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input folder not found: {args.input}")

    dirs = ensure_dirs(args.out)
    print(f"Scanning images under {args.input}...")
    images, counts = discover_images(args.input)
    print(
        f"Found {counts['total_images']} images: "
        f"{counts['fusion_images']} FUS, {counts['opt_images']} OPT/NIR/IMX, "
        f"{counts['thermal_images']} thermal-only, {counts['other_images']} other."
    )

    pairs, unique_pairs = build_pairs(images, args.strict_window_sec, args.qual_window_sec)
    write_csv(
        args.out / "q1_pairs.csv",
        [pair_to_row(pair) for pair in pairs],
        fieldnames=[
            "pair_id",
            "fus_path",
            "opt_path",
            "fus_timestamp",
            "opt_timestamp",
            "time_gap_sec",
            "pair_status",
            "timestamp_confidence",
            "fus_timestamp_confidence",
            "opt_timestamp_confidence",
            "fus_timestamp_source",
            "opt_timestamp_source",
            "fus_readable",
            "opt_readable",
            "is_unique_pair",
        ],
    )
    write_csv(
        args.out / "q1_pairs_unique.csv",
        [pair_to_row(pair) for pair in unique_pairs],
        fieldnames=[
            "pair_id",
            "fus_path",
            "opt_path",
            "fus_timestamp",
            "opt_timestamp",
            "time_gap_sec",
            "pair_status",
            "timestamp_confidence",
            "fus_timestamp_confidence",
            "opt_timestamp_confidence",
            "fus_timestamp_source",
            "opt_timestamp_source",
            "fus_readable",
            "opt_readable",
            "is_unique_pair",
        ],
    )

    accepted_pairs = [pair for pair in pairs if pair.pair_status != "reject" and pair.opt is not None]
    print(f"Processing {len(accepted_pairs)} accepted nearest pairs...")
    metric_rows: list[dict[str, Any]] = []
    for index, pair in enumerate(accepted_pairs, start=1):
        print(f"  [{index}/{len(accepted_pairs)}] {pair.pair_id}: {pair.fus.path.name} vs {pair.opt.path.name if pair.opt else ''}")
        metric_rows.append(process_pair(pair, args, dirs))

    metric_field_order = [
        "pair_id",
        "fus_path",
        "opt_path",
        "fus_timestamp",
        "opt_timestamp",
        "time_gap_sec",
        "pair_status",
        "is_unique_pair",
        "timestamp_confidence",
        "readable_pair",
        "evidence_strength",
        "processing_error",
        "alignment_status",
        "alignment_transform_type",
        "alignment_fus_keypoints",
        "alignment_opt_keypoints",
        "alignment_matches",
        "alignment_inliers",
        "alignment_inlier_ratio",
        "alignment_notes",
        "target_mask_area",
        "target_mask_area_ratio",
        "target_component_count",
        "target_largest_component_area",
        "target_roi_status",
    ]
    write_csv(args.out / "q1_metrics_per_pair.csv", metric_rows, fieldnames=None if not metric_rows else metric_field_order + [k for k in metric_rows[0].keys() if k not in metric_field_order])

    summaries = [
        summarize_group("all_accepted", metric_rows, pairs, args),
        summarize_group("strict_quantitative", select_rows(metric_rows, status="strict_quantitative"), pairs, args),
        summarize_group("qualitative_pair", select_rows(metric_rows, status="qualitative_pair"), pairs, args),
        summarize_group("unique_accepted", select_rows(metric_rows, unique=True), pairs, args),
        summarize_group("unique_strict", select_rows(metric_rows, status="strict_quantitative", unique=True), pairs, args),
        summarize_group("unique_qualitative", select_rows(metric_rows, status="qualitative_pair", unique=True), pairs, args),
    ]
    write_csv(args.out / "q1_summary.csv", summaries)

    make_plots(metric_rows, pairs, args, dirs)
    final_classification = final_evidence_classification(select_rows(metric_rows, unique=True), metric_rows)
    write_report(args.out / "q1_report.md", counts, pairs, unique_pairs, metric_rows, summaries, final_classification)
    write_slide_summary(args.out / "q1_slide_summary.md", counts, pairs, unique_pairs, metric_rows, final_classification)

    print_final_summary(counts, pairs, unique_pairs, metric_rows, final_classification)
    print(f"\nSaved Q1 outputs under: {args.out}")


if __name__ == "__main__":
    main()
