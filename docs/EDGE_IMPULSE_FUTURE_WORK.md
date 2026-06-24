# Edge Impulse Integration — Future Work Design Memo

**Status:** Design memo only — no production commits to `src/` until Phases 0–6 complete  
**Draft:** 2026-04-27  
**Scope:** Describes a non-blocking integration path; does not commit to a timeline

---

## Motivation

The current ML pipeline uses a scikit-learn Random Forest (`models/production/env_classifier.joblib`) running in a background daemon thread (`MLInferenceThread`, `src/smartbinocular/ml_inference.py`). Edge Impulse could offer:

1. **On-device optimization** — quantized, CMSIS-NN or NEON-accelerated inference for ARM Cortex-A72 (RPi4B)
2. **Hardware-in-the-loop continuous learning** — incremental labeling from live field data via Edge Impulse Studio
3. **Standardized deployment artifact** — `.tflite` or `.eim` replaces the joblib bundle, reducing dependency on scikit-learn at runtime

---

## Risks (per thesis mentor doc §D.3)

| Risk | Description | Mitigation |
|------|-------------|------------|
| R1 — Latency regression | TFLite inference on CPU-only RPi4B may be slower than the 12-column RF for this feature dimensionality (9–12 float32 features) | Benchmark both on device before committing |
| R2 — Feature pipeline coupling | Edge Impulse DSP blocks assume raw sensor input; our 9/12 optical features are computed by `FeatureExtractor`. Bridging requires a custom DSP block or pre-extracted features path. | Use the "pre-extracted features" input type in Edge Impulse |
| R3 — Versioning | Edge Impulse models are versioned externally; joblib bundles are git-tracked. CI/CD must reconcile both. | Keep joblib as the authoritative production artifact; Edge Impulse is opt-in |
| R4 — Class imbalance | Current dataset has 9 classes with 50–2000 samples each. Edge Impulse's default Keras pipeline may need custom class weighting. | Mirror the `class_weight="balanced"` RF setting in the Keras block |
| R5 — No GPU/NPU | RPi4B has no hardware accelerator. Edge Impulse's `cortex-a` target uses NEON SIMD, which is available on the A72. Expected benefit is modest for a 9-column feature vector. | Benchmark before claiming speedup in thesis |

---

## Proposed integration architecture

```
[FeatureExtractor.compute()]         existing (unchanged)
        ↓ 9 float32 features
[EdgeImpulseInferenceThread]         new daemon thread, mirrors MLInferenceThread
        ↓ post_result(MLSharedResult)
[EnvPresetController / main.py]      existing consumer (unchanged)
```

The `EdgeImpulseInferenceThread` would:
- Load an `.eim` or `.tflite` bundle at startup (path from env var `EI_MODEL_PATH`)
- Expose the same `MLSharedResult` interface as `MLInferenceThread`
- Fall back to the RF bundle if the `.eim` is absent

No changes to the frame loop, feature extraction, or display pipeline would be required.

---

## Concrete implementation steps (future)

1. Export the 9-feature training data from `from_logs_train_a.jsonl` to Edge Impulse CSV format (pre-extracted features mode)
2. Train a Keras MLP in Edge Impulse Studio with:
   - Input: 9 float32 features
   - Architecture: 2 hidden layers (64, 32 neurons), ReLU, dropout 0.3
   - Output: 9-class softmax
   - Class weights: balanced
3. Deploy as `cortex-a` `.eim` target or export `.tflite`
4. Implement `EdgeImpulseInferenceThread` mirroring `MLInferenceThread` (see `src/smartbinocular/ml_inference.py:MLInferenceThread`)
5. Benchmark `.eim` vs RF on-device: measure inference time per frame, abstention rate at τ=0.62, and macro F1 on a held-out sequence
6. Only promote to production if latency ≤ RF and macro F1 ≥ 0.93 (current RF baseline at τ=0.62)

---

## What this memo commits to

- **Nothing.** This is a design memo for future-work framing in the thesis.
- No new imports, no `src/` changes, no new dependencies.
- If a proof-of-concept is pursued, it lives under `tools/ei_export.py` and `models/edge_impulse/` — outside the production pipeline.

---

## Thesis framing

In the thesis future-work chapter, Edge Impulse integration should be described as:

> "A non-blocking extension point is available: `MLInferenceThread` is already isolated behind the `MLSharedResult` interface, allowing a future Edge Impulse `.eim` model to replace the scikit-learn RF bundle without changes to the frame loop. The pre-extracted-features mode eliminates the DSP coupling risk. Benchmarking on-device latency and calibration against the current RF baseline (macro F1 = 0.954 at τ=0.62) would determine whether the quantized MLP offers a practical advantage on the ARM Cortex-A72."

---

## Appendix — image-impulse experimental integration

### Memo-vs-reality delta

This memo was written assuming the Edge Impulse artifact would be a **pre-extracted-features MLP** (9–12 float32 columns → `run_classifier` with a `signal_t` wrapping the feature vector). The artifact that was actually exported is different:

- **Task:** FOMO object detection (not classification)
- **Input:** 128×128 RGB INT8 image (not a 9-column float vector)
- **Output:** `(1, 16, 16, C)` INT8 heatmap — one box per qualifying grid cell
- **Label:** single class `"person"` (threshold 0.8)

The pre-extracted-features path described in the memo does not apply. The integration uses `tflite_runtime` to load the `.tflite` directly and postprocesses the heatmap in ~30 lines of numpy.

### Bundle history

| Bundle | `EI_CLASSIFIER_COMPILED` | `.tflite` present | Status |
|--------|--------------------------|-------------------|--------|
| `person_in_dark-cpp-mcu-v2-impulse-#2/` | 1 (EON-compiled) | No | Superseded |
| `person_in_dark-cpp-linux-v3-impulse-#2/` | 0 (non-EON) | Yes | Integration target |

The MCU/EON bundle had no extractable `.tflite` and triggered a compile-time `#error` when `USE_FULL_TFLITE=1` was applied. The Linux v3 export resolves both issues.

### Integration path

**Primary — Path A (`tflite-runtime` in Python):** `pip install tflite-runtime`, copy the export's INT8 model to `models/ei/person_in_dark_fomo_int8.tflite` (see `models/ei/README.md`), run `interpreter.invoke()` in a daemon worker thread. Smallest blast radius: one optional module, no C++ build, no subprocess. Removal is two files.

**Fallback — Path B (sidecar):** Build EI's `example-standalone-inferencing-linux` against the new bundle with `USE_FULL_TFLITE=1`. Same Python concurrency shape; only the worker's `_invoke()` body changes. Use only if `tflite-runtime` perf is unacceptable on RPi4B.

### Flags

| Env var | Default | Notes |
|---------|---------|-------|
| `EI_PERSON_IN_DARK_ENABLED` | `False` | Master switch |
| `EI_PERSON_TFLITE_PATH` | `models/ei/person_in_dark_fomo_int8.tflite` | Overrides `config ei_person.tflite_path` |
| `EI_PERSON_INFER_INTERVAL` | `10` | Submit every N optical frames (~3 Hz at 30 FPS) |
| `EI_PERSON_FIT_MODE` | `crop` | Center-crop to square before 128×128 resize |
| `EI_PERSON_THRESHOLD` | `0.8` | Matches `EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD` |
| `EI_PERSON_DRAW_BBOX` | `False` | Amber bbox HUD overlay (debug only) |
| `EI_PERSON_NUM_THREADS` | `2` | TFLite thread count; reserves 2 cores for main pipeline |

### Installing `tflite-runtime` (optional `ei` extra)

```bash
pip install -e ".[ei]"          # uses pip install tflite-runtime
```

On older RPi images where the wheel is unavailable:
- **piwheels:** `pip install --index-url https://www.piwheels.org/simple tflite-runtime`
- **apt:** `sudo apt install python3-tflite-runtime`
- **Full runtime:** install `tensorflow` or `tensorflow-lite`; `from tflite_runtime.interpreter import Interpreter` resolves identically for all three.

No code change is required — the import path is the same regardless of which package provides the wheel.

### Status

Experimental; observability-only (HUD debug line + `ei_person` session JSON block). May be deleted at any time without affecting production NIR/thermal/fusion behavior.

### Removal

```bash
git rm -r src/smartbinocular/experimental/ tools/ei_smoke.py
git checkout -- src/smartbinocular/main.py src/smartbinocular/config.py \
                src/smartbinocular/metrics.py pyproject.toml
# optionally: git rm -r person_in_dark-cpp-linux-v3-impulse-#2/ person_in_dark-cpp-mcu-v2-impulse-#2/
pytest tests/   # must pass green
```
