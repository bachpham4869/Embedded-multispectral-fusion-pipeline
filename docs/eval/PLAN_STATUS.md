# Evaluation Campaign — Status

> Last updated: 2026-05-03. Two campaigns tracked below.

---

## Thesis Evaluation Campaign — Status

> Mac-side: **complete**. RPi4-side: pending device sessions and thermal captures.

### Done (Mac)

| Phase | Artifacts | Notes |
|---|---|---|
| Scaffold + MANIFEST | `docs/thesis_eval/MANIFEST.md` | All topics, host/status/caption columns |
| NIR enhancement | IQA round, CLAHE/dehaze/anti-glare sweeps, rain median N sweep (N∈{2,3,5,7}) | Copied from `data/eval/iqa_runs/` and `docs/tables/iqa/` |
| Bucket dispatch | Bucket gallery, dispatch consistency MD (12.6% rule/manifest agreement), per-bucket IQA table | 12.6% reflects cold-start EMA, not a bug |
| ML classifier | Confusion matrix, ROC/PR per night class, F1-τ-abstention, learning curve, feature importance, reliability grid, per-class summary | Optical test set; NIR domain shift UNVERIFIED |
| Fusion (PROXY) | Alpha sweep 1890 rows, A/B 270 rows, thermal contribution hist, per-class alpha, alpha curve | All dummy thermal; labeled PROXY |
| Timing & performance | Stage timing bars, latency histogram, transitions CSV (18 RPi4 sessions) | Short sessions 24-244 s; ≥5-min pending |
| Thermal sweeps | 3DNR sweep — THESIS_SKIP; rain median N — available | 3DNR needs RPi4 `.npy` sequences |
| Open-question mini-studies (×6) | Fusion benefit by class, ML hysteresis stability, train/deploy gap, bucket-F coverage, DCP resolution tradeoff, FPS consistency | All with explicit limitation notes |
| Pytest gates (×6) | `test_fusion_alpha_sweep_consistency`, `test_ab_fusion_vs_nir_consistency`, `test_rain_median_n_sweep`, `test_3dnr_alpha_sweep`, `test_thesis_manifest_integrity`, `test_session_inventory` | 256 passed 11 skipped; RPi4 gates opt-in via `THESIS_RPI_SESSIONS_OK=1` |

### Pending (RPi4)

| Item | Blocker |
|---|---|
| 3DNR alpha sweep | `data/eval/thermal_seq/*.npy` from MI48 recordings |
| Kalman Q/R sweep | Same |
| Session index (≥5-min sessions ×6) | Dedicated RPi4 capture sessions (§E matrix) |
| Cross-platform IQA delta | RPi4 IQA round via `run_nir_iqa_eval.py` |
| `test_session_inventory` green | Session index CSV populated |

---

## EI Person-in-Dark Eval Plan — Status

> Closed: 2026-05-03. All phases executed or explicitly deferred.

---

## Done

### Phase 0 — Decisions and risk register
- `docs/eval/ei_person_find_in_dark/DECISIONS_AND_RISKS.md` — resolved Q1–Q5, risk
  register R1–R8, locked evaluation scope, canonical CLI baseline command.
  Q4 confirmed: `EI_CLASSIFIER_RESIZE_FIT_SHORTEST` = center-crop = production
  `fit_mode="crop"` — **no train/serve resize mismatch**.

### Phase A — Session metrics report
- `docs/eval/sessions/2026-04-30/REPORT.md` — 18-session inventory, 3-session
  per-session brief with stage timing tables (framecache, nir_bucket, jerk, blend,
  hud, display, ml_infer, thermal_proc), EI block cross-check with corrected
  interpretation of `inferences_ok=338 ≠ 338 invocations` (HUD-rate sampling).
- `docs/eval/sessions/2026-04-30/inventory.csv` — 18 rows, all columns.

### Phase A code fixes
| Fix | Status | File | Effect |
|-----|--------|------|--------|
| A1 — decouple EI metric accumulation from debug gate | ✅ Shipped | `main.py` | `mean_inference_ms`, `p95_inference_ms`, `mean_detections_per_frame` now populate for all EI-enabled runs regardless of `debug` flag. Display-thread only; producer budget unaffected. |
| A3 — robust p95 via `numpy.percentile` + n≥20 gate | ✅ Shipped | `metrics.py` | Replaces biased `sorted()[int(n*0.95)]`. Runs only in `finalize()`. Zero hot-path cost. |
| A6 — `draw_bbox` default `True → False` | ✅ Shipped | `config.py` | Matches `EDGE_IMPULSE_FUTURE_WORK.md`; behavior unchanged (HUD gates rendering on `debug`). |

### Phase B — Offline eval harness
- `tools/eval_ei_person.py` — CLI with locked defaults (`--fit-mode crop`, `--limit 500`,
  `--metric-primary centroid_hit`, `--limit 0` guard requiring `--allow-full-run`).
- `tools/_ei_eval/` — internal package: `discover.py` (VOC XML, stem-pair, degenerate-box
  skip), `runtime.py` (imports production `_prepare_ei_input` / `_fomo_postprocess`
  directly; falls back to `ai-edge-litert` when `tflite-runtime` unavailable),
  `pipeline_lite.py` (identity-only v0; contrast variants raise `NotImplementedError`),
  `metrics.py` (image-F1, centroid-hit, Wilson CI, `unmap_centroid`, Padilla GT writer),
  `runner.py` (epoch orchestrator; writes `params.yaml`, `summary.json`, `per_image.csv`).
- **Tests:** 15 passing in `tests/test_eval_ei_person.py`. Full suite: **231 tests green**.
  Key tests: `test_runtime_uses_production_preprocess` (parity via `inspect.getsourcefile`),
  limit-guard, centroid-unmap math, malformed-XML hard-fail, Wilson CI bounds.

### Phase C — Offline eval runs (train split, 500 images)

> **Critical finding — INT8 softmax cap:** The model's INT8 output quantization
> (`scale=0.00390625, zp=-128`) compresses the logit range to `[0.0, 0.9961]`. The
> maximum achievable softmax person probability is **~0.730**
> (`exp(0.9961) / (exp(0.9961) + exp(0.0)) ≈ 0.730`). The EI metadata threshold of
> `0.8` (calibrated on the float model) is **unreachable at INT8 inference**. This
> also means the production `config.py` `threshold: 0.8` produces zero detections.
> See `baseline_train500_crop_t060/README.md` for details and recommended fix.

| run_id | threshold | centroid_hit F1 | image_F1 | P | R | tp | fp | fn |
|--------|-----------|-----------------|----------|---|---|----|----|----|
| baseline_train500_crop_t060 | 0.60 | **0.9701** | 0.9723 | 1.000 | 0.942 | 471 | 0 | 29 |
| baseline_train500_crop_t065 | 0.65 | 0.9572 | — | 1.000 | 0.918 | 459 | 0 | 41 |

All 500 evaluated images are GT-positive (first 500 stems of the train split).
Wilson 95% CI for t=0.60: centroid_hit recall `[0.918, 0.959]`.
Mac-side latency (not RPi-comparable): p50=0.2 ms, p95=0.26 ms.

Artefacts:
- `docs/eval/ei_person_find_in_dark/baseline_train500_crop_t060/README.md`
- `docs/eval/ei_person_find_in_dark/baseline_train500_crop_t060/epoch_00_raw_t060_crop_area/params.yaml`
- `docs/eval/ei_person_find_in_dark/baseline_train500_crop_t060/epoch_00_raw_t060_crop_area/summary.json`
- `docs/eval/ei_person_find_in_dark/baseline_train500_crop_t065/epoch_00_raw_t065_crop_area/summary.json`
- `per_image.csv` and `padilla_gt/` are git-ignored (large per-image files; regenerate with CLI).

---

## Partial / Blocked

| Item | Reason |
|------|--------|
| tflite-runtime on macOS | No macOS wheel on PyPI. Resolved via `ai-edge-litert` fallback in `runtime.py`. **No blocking issue.** |
| Threshold=0.8 baseline | Unreachable: INT8 softmax cap ~0.730. Run at t=0.60 and t=0.65 instead. |
| Sanity-test split eval | `--sanity-test-limit` not exercised (test split at `data/find-person-in-the-dark/test/test/` exists; gitignored). Deferred. |
| Threshold sweep t∈{0.50,0.55,0.60,0.65,0.70} | Only 2 of 5 points run (t=0.60, t=0.65). Sweep stopped early per Q1 — t=0.60 wins. Full sweep deferred. |
| Fit-mode comparison (`letterbox`, `passthrough`) | Deferred. Q4 confirmed `crop` is canonical. Compare only if threshold plateau is observed. |
| `--limit 2000` near-final sweep | Deferred; t=0.60 at n=500 has narrow CI already. |

---

## Follow-ups (separate tickets)

1. **A2 — dedupe `detections_per_frame` / `inferences_ok` by `frame_id`** (DECISIONS Q3 verbatim ticket): counters are HUD-rate-sampled until A2 lands; `inferences_ok ≈ frames_submitted × infer_interval`. Acceptance: 30s debug=False run yields `inferences_ok ≈ frames_submitted ± 2`.
2. **Production threshold fix** — update `config.py` `ei_person.threshold: 0.8 → 0.6` after on-device capture-and-replay confirms t=0.60 generalizes to NIR input. Do not change default before device verification.
3. **On-device capture-and-replay (R4)** — capture raw NIR frames on RPi4, feed offline harness, measure the sRGB/NIR distribution gap.
4. ~~**`fuse_stage_timing_ms` / `thermal_stage_timing_ms` stubs (A7)**~~ — **RESOLVED** in current code. `main.py:717-721` instantiates real `StageProfiler()` objects and wires them to `thesis_metrics.fuse_stage_profiler` / `thesis_metrics.thermal_stage_profiler`; `main.py:1439-1493` actively times 7 fusion sub-stages via `_fuse_sp`; `thermal_proc.stage_profiler = _thermal_sp` at line 722. The `{}` in the 18 existing sessions (2026-04-22) reflects pre-wiring captures. New sessions will emit fully populated keys.
5. **Padilla mAP secondary metric** — `evaluate/evaluate/pascalvoc.py` not yet wired into runner; Padilla GT files are written to `padilla_gt/` but the pascalvoc.py call is deferred.
