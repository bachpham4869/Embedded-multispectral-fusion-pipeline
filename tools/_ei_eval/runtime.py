"""TFLite runtime wrapper for offline EI person-in-dark evaluation.

Imports _prepare_ei_input and _fomo_postprocess directly from the production module
(smartbinocular.experimental.ei_person_in_dark) to guarantee preprocessing parity.

The local _prepare_ei_input_local function supports letterbox and custom interp
variants for comparison sweeps only — it is NOT in the production helper and must
not be used as the canonical baseline (see DECISIONS_AND_RISKS.md R3).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Production preprocess and postprocess — imported for parity guarantee.
# Test: tests/test_eval_ei_person.py::test_runtime_uses_production_preprocess
# verifies inspect.getsourcefile resolves to the production module.
from smartbinocular.experimental.ei_person_in_dark import (
    EIDetection,
    _fomo_postprocess,
    _prepare_ei_input,
)

__all__ = [
    "EIRuntime",
    "_prepare_ei_input",
    "_fomo_postprocess",
]

_INTERP_FLAGS = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


def _prepare_ei_input_local(
    bgr: np.ndarray,
    scale: float,
    zero_point: int,
    fit_mode: str = "crop",
    interp: str = "area",
) -> np.ndarray:
    """Local variant supporting all fit_mode × interp combinations.

    Adds letterbox (pad-to-square) and arbitrary interp to the production crop/passthrough.
    NOT in the production pipeline — harness comparison only.
    """
    h, w = bgr.shape[:2]
    flag = _INTERP_FLAGS.get(interp, cv2.INTER_AREA)

    if fit_mode == "crop":
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        square = bgr[y0: y0 + s, x0: x0 + s]
    elif fit_mode == "letterbox":
        s = max(h, w)
        canvas = np.zeros((s, s, 3), dtype=bgr.dtype)
        y0, x0 = (s - h) // 2, (s - w) // 2
        canvas[y0: y0 + h, x0: x0 + w] = bgr
        square = canvas
    elif fit_mode == "passthrough":
        square = bgr
    else:
        raise ValueError(f"Unknown fit_mode: {fit_mode!r}")

    resized = cv2.resize(square, (128, 128), interpolation=flag)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if scale > 0.0:
        q = np.round(rgb.astype(np.float32) / 255.0 / scale + zero_point)
    else:
        q = rgb.astype(np.float32) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)[np.newaxis]


class EIRuntime:
    """Wraps tflite_runtime.Interpreter for single-image offline inference.

    Uses the production _prepare_ei_input for the canonical 'crop' fit_mode.
    Uses _prepare_ei_input_local for letterbox and custom interp variants.
    """

    def __init__(
        self,
        tflite_path: str,
        *,
        num_threads: int = 4,
        threshold: float = 0.8,
        fit_mode: str = "crop",
        interp: str = "area",
    ) -> None:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except ImportError:
            try:
                from ai_edge_litert.interpreter import Interpreter  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "No TFLite runtime found. Install one of: "
                    "pip install tflite-runtime  OR  pip install ai-edge-litert"
                ) from exc

        self.tflite_path = str(tflite_path)
        self.threshold = threshold
        self.fit_mode = fit_mode
        self.interp = interp
        self.tflite_sha256 = _sha256(self.tflite_path)

        self._interp = Interpreter(model_path=self.tflite_path, num_threads=num_threads)
        self._interp.allocate_tensors()

        inp_d = self._interp.get_input_details()[0]
        out_d = self._interp.get_output_details()[0]
        self.in_scale: float = float(inp_d["quantization"][0])
        self.in_zp: int = int(inp_d["quantization"][1])
        self.out_scale: float = float(out_d["quantization"][0])
        self.out_zp: int = int(out_d["quantization"][1])
        self._inp_idx: int = inp_d["index"]
        self._out_idx: int = out_d["index"]
        self.output_shape: Tuple = tuple(out_d["shape"].tolist())

    def infer(self, bgr: np.ndarray) -> Tuple[List[EIDetection], float, np.ndarray]:
        """Run one inference.

        Returns:
            detections: EIDetection list (filtered by threshold).
            inference_ms: wall time of set_tensor+invoke+get_tensor.
            raw_scores: float32 array shape (gh, gw) of dequantized person scores
                before threshold — used for threshold-sweep metric.
        """
        use_local = (self.fit_mode in ("letterbox", "passthrough") or self.interp != "area")
        if use_local:
            tensor = _prepare_ei_input_local(
                bgr, self.in_scale, self.in_zp, self.fit_mode, self.interp
            )
        else:
            tensor = _prepare_ei_input(bgr, self.in_scale, self.in_zp, self.fit_mode)

        t0 = time.monotonic()
        self._interp.set_tensor(self._inp_idx, tensor)
        self._interp.invoke()
        raw_out = self._interp.get_tensor(self._out_idx)
        inference_ms = (time.monotonic() - t0) * 1000.0

        detections = _fomo_postprocess(
            raw_out, out_scale=self.out_scale, out_zp=self.out_zp, threshold=self.threshold
        )

        # Raw per-cell scores for sweep metric (dequantized, before threshold)
        probs = (raw_out.astype(np.float32) - self.out_zp) * self.out_scale
        _, gh, gw, C = probs.shape
        if C >= 2:
            shifted = probs - probs.max(axis=-1, keepdims=True)
            exp_ = np.exp(shifted)
            raw_scores: np.ndarray = (exp_ / exp_.sum(axis=-1, keepdims=True))[0, :, :, 1]
        else:
            raw_scores = 1.0 / (1.0 + np.exp(-probs[0, :, :, 0]))

        return detections, inference_ms, raw_scores


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
