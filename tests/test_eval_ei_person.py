"""Tests for the offline EI person-in-dark eval harness.

Key acceptance criteria (from DECISIONS_AND_RISKS.md / Phase B spec):
  - runtime._prepare_ei_input resolves to the production module (parity guarantee)
  - --limit 0 without --allow-full-run exits non-zero
  - unmap_centroid round-trips correctly for all fit_mode variants
  - discover.iter_eval_items fails loud on malformed XML
  - MetricsAccumulator.compute returns expected keys for all metric families
"""

from __future__ import annotations

import inspect
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

# Add tools/ to path so _ei_eval is importable
import sys
_TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))


# ── Production parity ─────────────────────────────────────────────────────────

def test_runtime_uses_production_preprocess():
    """The harness _prepare_ei_input must resolve to the production module.

    Acceptance: inspect.getsourcefile returns a path ending in ei_person_in_dark.py,
    not runtime.py.
    """
    from _ei_eval import runtime as harness_runtime
    from smartbinocular.experimental.ei_person_in_dark import _prepare_ei_input as prod_fn

    harness_fn = harness_runtime._prepare_ei_input
    harness_src = inspect.getsourcefile(harness_fn)
    prod_src = inspect.getsourcefile(prod_fn)

    assert harness_src == prod_src, (
        f"Harness _prepare_ei_input resolves to {harness_src}, "
        f"expected production module at {prod_src}"
    )
    assert harness_fn is prod_fn, "harness._prepare_ei_input is not the same object as production"


def test_runtime_uses_production_postprocess():
    from _ei_eval import runtime as harness_runtime
    from smartbinocular.experimental.ei_person_in_dark import _fomo_postprocess as prod_fn

    assert harness_runtime._fomo_postprocess is prod_fn


# ── CLI limit guard ───────────────────────────────────────────────────────────

def test_limit_zero_without_allow_full_run_exits_nonzero(tmp_path):
    """--limit 0 without --allow-full-run must exit with code 2."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

    import importlib
    import tools.eval_ei_person as cli_mod

    result = cli_mod.main([
        "--tflite", "dummy.tflite",
        "--dataset", str(tmp_path),
        "--limit", "0",
    ])
    assert result == 2


def test_limit_zero_with_allow_full_run_does_not_exit_early(tmp_path):
    """--limit 0 --allow-full-run should NOT fail with exit code 2."""
    import tools.eval_ei_person as cli_mod

    # Will fail later (no tflite, no dataset) but must not fail with the limit guard
    # We check that the guard itself passes (result != 2).
    try:
        result = cli_mod.main([
            "--tflite", "dummy_nonexistent.tflite",
            "--dataset", str(tmp_path),
            "--limit", "0",
            "--allow-full-run",
        ])
    except SystemExit as exc:
        result = exc.code
    except Exception:
        result = 1  # Any other error is fine — limit guard was passed
    assert result != 2


# ── unmap_centroid ────────────────────────────────────────────────────────────

def test_unmap_centroid_crop_centered_detection():
    """Centroid at (0.5, 0.5) in 128-frame should map to image center for any crop."""
    from _ei_eval.metrics import unmap_centroid

    img_w, img_h = 640, 480
    ox, oy = unmap_centroid(0.5, 0.5, img_w, img_h, fit_mode="crop")
    # For crop: side=480, x_off=(640-480)//2=80, y_off=0
    # orig_x = 80 + 0.5*480 = 80+240 = 320, orig_y = 0+0.5*480 = 240
    assert abs(ox - 320.0) < 1e-6
    assert abs(oy - 240.0) < 1e-6


def test_unmap_centroid_crop_top_left():
    """Top-left centroid (0,0) in crop mode maps to the top-left of the cropped square."""
    from _ei_eval.metrics import unmap_centroid

    img_w, img_h = 640, 480
    ox, oy = unmap_centroid(0.0, 0.0, img_w, img_h, fit_mode="crop")
    # side=480, x_off=80, y_off=0
    # orig_x=80, orig_y=0
    assert abs(ox - 80.0) < 1e-6
    assert abs(oy - 0.0) < 1e-6


def test_unmap_centroid_letterbox_centered():
    from _ei_eval.metrics import unmap_centroid

    img_w, img_h = 640, 480
    ox, oy = unmap_centroid(0.5, 0.5, img_w, img_h, fit_mode="letterbox")
    # side=640, x_off=(640-640)//2=0, y_off=(640-480)//2=80
    # orig_x=0.5*640-0=320, orig_y=0.5*640-80=240
    assert abs(ox - 320.0) < 1e-6
    assert abs(oy - 240.0) < 1e-6


def test_unmap_centroid_passthrough():
    from _ei_eval.metrics import unmap_centroid

    img_w, img_h = 320, 240
    ox, oy = unmap_centroid(0.25, 0.75, img_w, img_h, fit_mode="passthrough")
    assert abs(ox - 80.0) < 1e-6
    assert abs(oy - 180.0) < 1e-6


# ── discover ─────────────────────────────────────────────────────────────────

def _write_voc_xml(path: Path, boxes=None):
    """Write a minimal Pascal VOC XML file."""
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = path.stem + ".jpg"
    sz = ET.SubElement(root, "size")
    ET.SubElement(sz, "width").text = "640"
    ET.SubElement(sz, "height").text = "480"
    ET.SubElement(sz, "depth").text = "3"
    for xmin, ymin, xmax, ymax in (boxes or [(10, 20, 100, 200)]):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "person"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(xmin)
        ET.SubElement(bnd, "ymin").text = str(ymin)
        ET.SubElement(bnd, "xmax").text = str(xmax)
        ET.SubElement(bnd, "ymax").text = str(ymax)
    tree = ET.ElementTree(root)
    tree.write(str(path))


def _make_fake_dataset(tmp_path: Path, n: int = 5) -> Path:
    """Create a minimal fake dataset under tmp_path/train/train."""
    root = tmp_path / "train" / "train"
    img_dir = root / "train_images"
    ann_dir = root / "train_annotations"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    import numpy as np
    import cv2
    for i in range(1, n + 1):
        stem = f"{i:06d}"
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)
        _write_voc_xml(ann_dir / f"{stem}.xml")
    return root


def test_discover_yields_correct_count(tmp_path):
    from _ei_eval.discover import iter_eval_items

    ds = _make_fake_dataset(tmp_path, n=5)
    items = list(iter_eval_items(ds, limit=3))
    assert len(items) == 3


def test_discover_all_items_have_gt_boxes(tmp_path):
    from _ei_eval.discover import iter_eval_items

    ds = _make_fake_dataset(tmp_path, n=3)
    for item in iter_eval_items(ds):
        assert len(item.gt_boxes) == 1
        assert item.gt_boxes[0].label == "person"


def test_discover_fails_loud_on_malformed_xml(tmp_path):
    from _ei_eval.discover import iter_eval_items
    import cv2

    img_dir = tmp_path / "train_images"
    ann_dir = tmp_path / "train_annotations"
    img_dir.mkdir()
    ann_dir.mkdir()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "000001.jpg"), img)
    (ann_dir / "000001.xml").write_text("this is not valid xml <<<<")

    with pytest.raises((ValueError, ET.ParseError)):
        list(iter_eval_items(tmp_path))


def test_discover_missing_images_dir_raises(tmp_path):
    from _ei_eval.discover import iter_eval_items

    with pytest.raises(FileNotFoundError):
        list(iter_eval_items(tmp_path))


# ── MetricsAccumulator ────────────────────────────────────────────────────────

def test_metrics_compute_has_all_families():
    """compute() must return all four metric family keys."""
    from _ei_eval.metrics import MetricsAccumulator, PerImageResult

    acc = MetricsAccumulator(fit_mode="crop")
    raw = np.zeros((8, 8), dtype=np.float32)
    for i in range(3):
        acc.add(PerImageResult(
            stem=f"{i:06d}",
            gt_positive=(i % 2 == 0),
            predicted_positive=(i % 2 == 0),
            centroid_hit=(i % 2 == 0),
            n_detections=1 if i % 2 == 0 else 0,
            n_gt_boxes=1 if i % 2 == 0 else 0,
            inference_ms=5.0 + i,
            max_score=0.9 if i % 2 == 0 else 0.0,
            cell_activation_rate=0.1,
            raw_scores=raw,
        ))

    result = acc.compute()
    assert "image_level_f1" in result
    assert "centroid_hit" in result
    assert "latency_host_ms" in result
    assert "score_histogram" in result
    assert result["n_images"] == 3


def test_metrics_wilson_ci_valid_range():
    from _ei_eval.metrics import _wilson_ci

    lo, hi = _wilson_ci(80, 100)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi > lo


def test_metrics_wilson_ci_zero_n():
    from _ei_eval.metrics import _wilson_ci

    lo, hi = _wilson_ci(0, 0)
    assert lo == 0.0 and hi == 0.0
