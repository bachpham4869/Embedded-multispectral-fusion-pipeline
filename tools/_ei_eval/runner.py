"""Epoch runner for offline EI person-in-dark evaluation.

Orchestrates one epoch: iterates eval items, runs inference, accumulates metrics,
writes params.yaml / summary.json / per_image.csv to the output directory.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .discover import EvalItem, iter_eval_items
from .metrics import MetricsAccumulator, PerImageResult, voc_to_padilla
from .pipeline_lite import PipelineLiteVariant, apply as pipeline_lite_apply
from .runtime import EIRuntime


def run_epoch(
    *,
    tflite_path: str,
    dataset_root: Path,
    out_dir: Path,
    threshold: float = 0.8,
    fit_mode: str = "crop",
    interp: str = "area",
    pipeline_lite_variant: str = "identity",
    metric_primary: str = "centroid_hit",
    limit: int = 500,
    run_id: str,
    epoch_name: str,
    argv: List[str],
    sanity_dataset_root: Optional[Path] = None,
    sanity_limit: int = 0,
) -> Dict:
    """Run one evaluation epoch and write artefacts to out_dir.

    Returns the summary dict (also written to out_dir/summary.json).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variant = PipelineLiteVariant(pipeline_lite_variant)
    runtime = EIRuntime(
        tflite_path,
        num_threads=4,
        threshold=threshold,
        fit_mode=fit_mode,
        interp=interp,
    )

    params = _build_params(
        tflite_path=tflite_path,
        tflite_sha256=runtime.tflite_sha256,
        dataset_root=str(dataset_root),
        threshold=threshold,
        fit_mode=fit_mode,
        interp=interp,
        pipeline_lite_variant=pipeline_lite_variant,
        metric_primary=metric_primary,
        limit=limit,
        run_id=run_id,
        epoch_name=epoch_name,
        argv=argv,
    )
    _write_yaml(out_dir / "params.yaml", params)

    print(f"[runner] Starting epoch '{epoch_name}' → {out_dir}")
    summary = _run_split(
        runtime=runtime,
        dataset_root=dataset_root,
        limit=limit,
        fit_mode=fit_mode,
        variant=variant,
        out_dir=out_dir,
        split_label="train",
    )

    if sanity_dataset_root and sanity_limit > 0:
        sanity_out = out_dir / "sanity"
        print(f"[runner] Sanity pass ({sanity_limit} images) → {sanity_out}")
        sanity_summary = _run_split(
            runtime=runtime,
            dataset_root=sanity_dataset_root,
            limit=sanity_limit,
            fit_mode=fit_mode,
            variant=variant,
            out_dir=sanity_out,
            split_label="sanity_test",
        )
        summary["sanity"] = sanity_summary

    summary["params"] = params
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[runner] Epoch complete. Primary metric ({metric_primary}): "
          f"{_primary_value(summary, metric_primary)}")
    return summary


def _run_split(
    *,
    runtime: EIRuntime,
    dataset_root: Path,
    limit: int,
    fit_mode: str,
    variant: PipelineLiteVariant,
    out_dir: Path,
    split_label: str,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = MetricsAccumulator(fit_mode=fit_mode)
    detections_by_stem: Dict[str, List] = {}
    all_items: List[EvalItem] = []

    items = list(iter_eval_items(dataset_root, limit=limit))
    all_items = items

    for item in items:
        bgr = cv2.imread(str(item.image_path))
        if bgr is None:
            print(f"[runner] WARNING: could not read {item.image_path} — skipping")
            continue

        img_h, img_w = bgr.shape[:2]
        bgr = pipeline_lite_apply(bgr, variant=variant, fit_mode=fit_mode, interp=runtime.interp)

        detections, inference_ms, raw_scores = runtime.infer(bgr)

        n_dets = len(detections)
        max_score = float(raw_scores.max()) if raw_scores.size > 0 else 0.0
        gh, gw = raw_scores.shape
        act_rate = float((raw_scores >= runtime.threshold).sum()) / (gh * gw) if gh * gw > 0 else 0.0

        gt_positive = len(item.gt_boxes) > 0
        pred_positive = n_dets > 0
        hit = False
        if pred_positive and gt_positive:
            for det in detections:
                if acc.centroid_in_any_box(det.cx, det.cy, img_w, img_h, item.gt_boxes, fit_mode):
                    hit = True
                    break

        acc.add(PerImageResult(
            stem=item.stem,
            gt_positive=gt_positive,
            predicted_positive=pred_positive,
            centroid_hit=hit,
            n_detections=n_dets,
            n_gt_boxes=len(item.gt_boxes),
            inference_ms=inference_ms,
            max_score=max_score,
            cell_activation_rate=act_rate,
            raw_scores=raw_scores,
        ))

        # Build Padilla-format detection for secondary mAP
        if n_dets > 0:
            from .metrics import unmap_centroid
            dets_padilla = []
            for det in detections:
                ox, oy = unmap_centroid(det.cx, det.cy, img_w, img_h, fit_mode)
                cell_w = (1.0 / gw) * (img_w if fit_mode == "passthrough" else min(img_w, img_h))
                cell_h = (1.0 / gh) * (img_h if fit_mode == "passthrough" else min(img_w, img_h))
                dets_padilla.append((
                    det.label, det.score,
                    max(0.0, ox - cell_w / 2), max(0.0, oy - cell_h / 2),
                    min(img_w, ox + cell_w / 2), min(img_h, oy + cell_h / 2),
                ))
            detections_by_stem[item.stem] = dets_padilla

    agg = acc.compute()
    rows = acc.per_image_rows()

    _write_csv(out_dir / "per_image.csv", rows)
    voc_to_padilla(all_items, out_dir / "padilla_gt", detections_by_stem)

    return {split_label: agg}


def _primary_value(summary: Dict, metric_primary: str) -> str:
    try:
        if metric_primary == "centroid_hit":
            return str(summary.get("train", {}).get("centroid_hit", {}).get("f1", "n/a"))
        if metric_primary == "image_f1":
            return str(summary.get("train", {}).get("image_level_f1", {}).get("f1", "n/a"))
    except Exception:
        pass
    return "n/a"


def _build_params(*, tflite_path, tflite_sha256, dataset_root, threshold,
                  fit_mode, interp, pipeline_lite_variant, metric_primary,
                  limit, run_id, epoch_name, argv) -> Dict:
    git_rev = "unknown"
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass

    versions = {}
    for pkg in ("numpy", "tflite-runtime", "opencv-python-headless", "opencv-python"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        versions["cv2"] = cv2.__version__
    except Exception:
        pass

    return {
        "run_id": run_id,
        "epoch_name": epoch_name,
        "tflite_path": tflite_path,
        "tflite_sha256": tflite_sha256,
        "dataset_root": str(dataset_root),
        "git_rev": git_rev,
        "threshold": threshold,
        "fit_mode": fit_mode,
        "interp": interp,
        "pipeline_lite_variant": pipeline_lite_variant,
        "metric_primary": metric_primary,
        "limit": limit,
        "argv": argv,
        "versions": versions,
    }


def _write_yaml(path: Path, data: Dict) -> None:
    lines = []
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            lines.append(f"{k}: {json.dumps(v)}")
        else:
            lines.append(f"{k}: {v!r}")
    path.write_text("\n".join(lines) + "\n")


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
