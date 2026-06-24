"""Metrics accumulator for offline EI person-in-dark evaluation.

Metric families (per DECISIONS_AND_RISKS.md Q1):
  1. Image-level presence F1 — sanity gate
  2. Centroid-in-GT-box hit — PRIMARY decision metric
  3. Padilla mAP@0.5/@0.75 — secondary, trend-only (via voc_to_padilla + pascalvoc.py)
  4. Calibration / per-image stats (cell activation rate, score histogram, host latency)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._types import GTBox


@dataclass
class PerImageResult:
    stem: str
    gt_positive: bool          # True if any GT box present
    predicted_positive: bool   # True if any detection above threshold
    centroid_hit: bool         # True if any detection centroid falls inside a GT box
    n_detections: int
    n_gt_boxes: int
    inference_ms: float
    max_score: float           # max raw person score in the grid (0 if no detections)
    cell_activation_rate: float  # fraction of cells >= threshold
    raw_scores: np.ndarray     # (gh, gw) float32 for histogram/sweep


class MetricsAccumulator:
    """Collects per-image results and computes aggregate metrics."""

    def __init__(self, img_w: int = 0, img_h: int = 0, fit_mode: str = "crop") -> None:
        self.img_w = img_w
        self.img_h = img_h
        self.fit_mode = fit_mode
        self._results: List[PerImageResult] = []
        self._latencies: List[float] = []

    def add(self, result: PerImageResult) -> None:
        self._results.append(result)
        self._latencies.append(result.inference_ms)

    def centroid_in_any_box(
        self,
        cx_norm: float,
        cy_norm: float,
        img_w: int,
        img_h: int,
        gt_boxes: Tuple[GTBox, ...],
        fit_mode: str,
    ) -> bool:
        """Return True if the detection centroid (in 128-input-frame coords) lands in any GT box."""
        orig_x, orig_y = unmap_centroid(cx_norm, cy_norm, img_w, img_h, fit_mode)
        for box in gt_boxes:
            if box.xmin <= orig_x <= box.xmax and box.ymin <= orig_y <= box.ymax:
                return True
        return False

    def compute(self) -> Dict:
        """Return a dict of all aggregate metrics."""
        n = len(self._results)
        if n == 0:
            return {"n_images": 0}

        # Image-level F1
        tp_img = sum(1 for r in self._results if r.gt_positive and r.predicted_positive)
        fp_img = sum(1 for r in self._results if not r.gt_positive and r.predicted_positive)
        fn_img = sum(1 for r in self._results if r.gt_positive and not r.predicted_positive)
        prec_img = tp_img / (tp_img + fp_img) if (tp_img + fp_img) > 0 else 0.0
        rec_img = tp_img / (tp_img + fn_img) if (tp_img + fn_img) > 0 else 0.0
        f1_img = (
            2 * prec_img * rec_img / (prec_img + rec_img)
            if (prec_img + rec_img) > 0 else 0.0
        )
        f1_ci = _wilson_ci(tp_img, tp_img + fn_img) if (tp_img + fn_img) > 0 else (0.0, 0.0)

        # Centroid-in-GT-box hit
        positives = [r for r in self._results if r.gt_positive]
        tp_hit = sum(1 for r in positives if r.centroid_hit)
        fn_hit = len(positives) - tp_hit
        fp_hit = sum(1 for r in self._results if not r.gt_positive and r.predicted_positive)
        prec_hit = tp_hit / (tp_hit + fp_hit) if (tp_hit + fp_hit) > 0 else 0.0
        rec_hit = tp_hit / (tp_hit + fn_hit) if (tp_hit + fn_hit) > 0 else 0.0
        f1_hit = (
            2 * prec_hit * rec_hit / (prec_hit + rec_hit)
            if (prec_hit + rec_hit) > 0 else 0.0
        )
        hit_ci = _wilson_ci(tp_hit, len(positives)) if positives else (0.0, 0.0)

        # Latency (host-side, NOT RPi-comparable)
        lat = np.array(self._latencies, dtype=np.float64)
        p50 = float(np.percentile(lat, 50)) if len(lat) > 0 else 0.0
        p95 = float(np.percentile(lat, 95)) if len(lat) >= 20 else None

        # Score histogram (10 bins over all images)
        all_scores = np.concatenate([r.raw_scores.ravel() for r in self._results])
        hist, bin_edges = np.histogram(all_scores, bins=10, range=(0.0, 1.0))

        n_pos_gt = sum(1 for r in self._results if r.gt_positive)
        n_neg_gt = n - n_pos_gt

        return {
            "n_images": n,
            "n_gt_positive": n_pos_gt,
            "n_gt_negative": n_neg_gt,
            "image_level_f1": {
                "precision": round(prec_img, 4),
                "recall": round(rec_img, 4),
                "f1": round(f1_img, 4),
                "tp": tp_img,
                "fp": fp_img,
                "fn": fn_img,
                "wilson_ci_95": [round(f1_ci[0], 4), round(f1_ci[1], 4)],
            },
            "centroid_hit": {
                "precision": round(prec_hit, 4),
                "recall": round(rec_hit, 4),
                "f1": round(f1_hit, 4),
                "tp": tp_hit,
                "fp": fp_hit,
                "fn": fn_hit,
                "n_gt_positive_images": len(positives),
                "wilson_ci_95": [round(hit_ci[0], 4), round(hit_ci[1], 4)],
            },
            "latency_host_ms": {
                "NOTE": "Mac-side latency; NOT comparable to RPi4 inference times",
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2) if p95 is not None else None,
                "n_samples": len(lat),
            },
            "score_histogram": {
                "bins": hist.tolist(),
                "bin_edges": [round(float(e), 2) for e in bin_edges.tolist()],
            },
            "mean_cell_activation_rate": round(
                float(np.mean([r.cell_activation_rate for r in self._results])), 4
            ),
        }

    def per_image_rows(self) -> List[Dict]:
        return [
            {
                "stem": r.stem,
                "gt_positive": int(r.gt_positive),
                "predicted_positive": int(r.predicted_positive),
                "centroid_hit": int(r.centroid_hit),
                "n_detections": r.n_detections,
                "n_gt_boxes": r.n_gt_boxes,
                "inference_ms": round(r.inference_ms, 3),
                "max_score": round(r.max_score, 4),
                "cell_activation_rate": round(r.cell_activation_rate, 4),
            }
            for r in self._results
        ]


def unmap_centroid(
    cx_norm: float,
    cy_norm: float,
    img_w: int,
    img_h: int,
    fit_mode: str,
) -> Tuple[float, float]:
    """Map a detection centroid from 128-input-frame normalized coords back to original image pixel coords.

    For fit_mode='crop':
        The 128px frame covers the center-cropped square (side = min(H,W)).
        x_offset = (W - side) // 2,  y_offset = (H - side) // 2
        orig_x = x_offset + cx_norm * side

    For fit_mode='letterbox':
        The 128px frame covers the padded square (side = max(H,W)).
        x_offset = (side - W) // 2,  y_offset = (side - H) // 2 (padding added)
        orig_x = cx_norm * side - x_offset

    For fit_mode='passthrough':
        No squaring; 128px frame is a direct resize of the original.
        orig_x = cx_norm * W
    """
    if fit_mode == "crop":
        side = min(img_w, img_h)
        x_off = (img_w - side) // 2
        y_off = (img_h - side) // 2
        return x_off + cx_norm * side, y_off + cy_norm * side
    elif fit_mode == "letterbox":
        side = max(img_w, img_h)
        x_off = (side - img_w) // 2
        y_off = (side - img_h) // 2
        return cx_norm * side - x_off, cy_norm * side - y_off
    elif fit_mode == "passthrough":
        return cx_norm * img_w, cy_norm * img_h
    else:
        raise ValueError(f"Unknown fit_mode: {fit_mode!r}")


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def voc_to_padilla(
    eval_items,
    out_dir: Path,
    detections_by_stem: Dict[str, List],
) -> None:
    """Write Padilla-format GT and detection text files for secondary mAP.

    GT format per line:  <label> <xmin> <ymin> <xmax> <ymax>
    Det format per line: <label> <score> <xmin> <ymin> <xmax> <ymax>

    Args:
        eval_items: iterable of EvalItem
        out_dir: directory to write gt/ and dt/ subdirs
        detections_by_stem: {stem: list of (label, score, xmin, ymin, xmax, ymax)}
    """
    gt_dir = out_dir / "gt"
    dt_dir = out_dir / "dt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    dt_dir.mkdir(parents=True, exist_ok=True)

    for item in eval_items:
        gt_lines = [
            f"{b.label} {b.xmin:.1f} {b.ymin:.1f} {b.xmax:.1f} {b.ymax:.1f}"
            for b in item.gt_boxes
        ]
        (gt_dir / f"{item.stem}.txt").write_text("\n".join(gt_lines))

        dets = detections_by_stem.get(item.stem, [])
        dt_lines = [
            f"{lbl} {score:.4f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
            for lbl, score, xmin, ymin, xmax, ymax in dets
        ]
        (dt_dir / f"{item.stem}.txt").write_text("\n".join(dt_lines))
