# Session Metrics Report — 2026-04-30

> Generated from `fusion_captures/metrics/session_*.json` (18 files total).
> Inventory: [inventory.csv](inventory.csv)
> Code references are to the state of `main.py` / `metrics.py` **before** the A1/A3
> fixes landed in this work.

---

## Summary

| # | Date | Duration (s) | Frames | FPS mean±std | env_mode | debug | EI enabled | Software |
|---|------|-------------|--------|-------------|----------|-------|------------|---------|
| 1 | 2026-04-22 | 178 | 714 | 5.3±3.9 | auto_rule | N | N | fusion_live_optimized |
| 2 | 2026-04-22 | 103 | 401 | 4.4±1.8 | auto_rule | N | N | fusion_live_optimized |
| 3 | 2026-04-22 | 24 | 94 | 4.2±0.6 | auto_rule | N | N | fusion_live_optimized |
| 4 | 2026-04-22 | 70 | 330 | 4.8±0.6 | auto_rule | N | N | fusion_live_optimized |
| 5 | 2026-04-22 | 157 | 907 | 7.2±4.3 | auto_rule | N | N | fusion_live_optimized |
| 6 | 2026-04-22 | 128 | 767 | 7.6±4.0 | auto_rule | N | N | fusion_live_optimized |
| 7 | 2026-04-22 | 62 | 462 | 9.5±4.4 | auto_rule | N | N | fusion_live_optimized |
| 8 | 2026-04-24 | 118 | 1023 | 9.8±3.4 | auto_rule | N | N | fusion_live_optimized |
| 9 | 2026-04-24 | 79 | 637 | 10.1±4.5 | auto_rule | N | N | fusion_live_optimized |
| 10 | 2026-04-24 | 112 | 576 | 7.4±5.9 | auto_rule | N | N | fusion_live_optimized |
| 11 | 2026-04-24 | 101 | 580 | 6.7±2.8 | auto_rule | N | N | fusion_live_optimized |
| 12 | 2026-04-24 | 94 | 536 | 6.7±3.1 | auto_rule | N | N | fusion_live_optimized |
| 13 | 2026-04-24 | 244 | 1338 | 6.5±3.4 | auto_rule | N | N | fusion_live_optimized |
| 14 | 2026-04-24 | 82 | 472 | 7.0±4.9 | auto_rule | N | N | fusion_live_optimized |
| 15 | 2026-04-25 | 226 | 1310 | 6.3±2.0 | auto_rule | **Y** | N | smartbinocular |
| 16 | 2026-04-30 | 193 | 1213 | 6.8±2.4 | auto_rule | N | N | smartbinocular |
| 17 | 2026-04-30 | 64 | 493 | 8.7±4.9 | auto_rule | N | N | smartbinocular |
| 18 | **2026-04-30** | **55** | **347** | **7.8±5.3** | auto_rule | **Y** | **Y** | smartbinocular |

18 sessions, all `env_mode=auto_rule`, all on `aarch64` RPi4. All have `vcgencmd_throttled=0x0` (no CPU throttling). Software migrated from `fusion_live_optimized` → `smartbinocular` (labeled `rpi_p3b`). Only session 18 has the `ei_person` block.

---

## Per-session brief: newest 3 (sessions 16–18)

### Session 16 — `session_20260430-150125.json`

**Config:** debug=False, rpi_throughput_max=True, env_mode=auto_rule, nir_capture_fps=60, nir_brightness_subsample=4, opencv_num_threads=4. No EI.

**Duration / throughput:** 192.7 s, 1213 frames, 6.8±2.4 fps.

**Env distribution:**

| env | frames |
|-----|--------|
| default | 17 |
| normal_night | 529 |
| glare | 550 |
| normal_day | 48 |
| fog | 69 |

**Bucket distribution:** A=530, RAW=683.

**Stage timing (mean ms / std ms / n):**

| Stage | mean ms | std ms | n |
|-------|---------|--------|---|
| framecache | 12.60 | 8.11 | 1213 |
| nir_bucket | 29.04 | 40.27 | 1155 |
| jerk | 5.01 | 9.13 | 1213 |
| blend | 11.79 | 17.16 | 1213 |
| hud | 4.98 | 8.61 | 1213 |
| display | 3.05 | 3.56 | 1213 |
| ml_infer | 17.58 | 29.08 | 80 |
| thermal_proc | 20.71 | 22.31 | 813 |

> `fuse_stage_timing_ms` and `thermal_stage_timing_ms` are both `{}` (empty) in this
> and all recent sessions. This appears to be instrumentation that was either disabled or
> never wired in for the `rpi_p3b` build. These stub keys can be removed from
> `metrics.py finalize()` unless a specific measurement is planned.

**Throttle:** `throttled=0x0`, temp=70.6°C, all 4 cores at 1800 MHz.

**Note on `nir_bucket` std=40.27 ms:** High variance is expected — this stage dispatches
to bucket A (full `HybridNIREnhancer`, expensive) or RAW passthrough (cheap) depending
on ENV_CLASS. With 550 frames in glare/normal_day → RAW and 530 in normal_night/default
→ Bucket A, the bimodal distribution explains the large std.

---

### Session 17 — `session_20260430-150536.json`

**Config:** debug=False, rpi_throughput_max=True. No EI.

**Duration / throughput:** 64.3 s, 493 frames, 8.7±4.9 fps.

**Env distribution:**

| env | frames |
|-----|--------|
| default | 17 |
| normal_night | 476 |

**Bucket distribution:** A=476, RAW=17.

**Stage timing:**

| Stage | mean ms | std ms | n |
|-------|---------|--------|---|
| framecache | 11.37 | 7.27 | 493 |
| nir_bucket | 59.67 | 31.27 | 493 |
| jerk | 7.50 | 8.47 | 493 |
| blend | 1.03 | 5.50 | 493 |
| hud | 4.65 | 6.90 | 493 |
| display | 2.75 | 1.93 | 493 |
| ml_infer | 17.96 | 28.05 | 32 |

**Comparison vs session 16:** `nir_bucket` mean jumps from 29 ms to 60 ms. Here almost
all frames are normal_night → Bucket A (NIR enhancement), vs session 16 where ~half
went to RAW. The additional `thermal_proc` stage (813 entries in session 16) is absent
here — confirmed thermal processing was reduced. `blend=1.03 ms` vs `11.79 ms` in
session 16, likely because thermal was not active for most of this session.

**Throttle:** `throttled=0x0`, temp=70.1°C.

---

### Session 18 — `session_20260430-151717.json` ← EI ENABLED

**Config:** debug=**True**, rpi_throughput_max=True. **EI person-in-dark enabled.**

**Duration / throughput:** 55.3 s, 347 frames, 7.8±5.3 fps.

**Env distribution:**

| env | frames |
|-----|--------|
| default | 17 |
| normal_night | 130 |
| fog | 33 |
| glare | 149 |
| normal_day | 18 |

**Bucket distribution:** A=131, RAW=216.

**Stage timing:**

| Stage | mean ms | std ms | n |
|-------|---------|--------|---|
| framecache | 10.58 | 8.27 | 347 |
| nir_bucket | 23.79 | 37.47 | 347 |
| jerk | 5.84 | 7.07 | 347 |
| blend | 12.52 | 15.79 | 347 |
| hud | 3.72 | 7.23 | 347 |
| display | 2.43 | 2.19 | 347 |
| ml_infer | 7.70 | 13.76 | 23 |

**Throttle:** `throttled=0x0`, temp=67.6°C (cooler than session 16 — shorter run).

---

## EI metrics cross-check (session 18 — read this before the headline numbers)

> **Warning:** Until the A2 fix lands, `inferences_ok` does NOT equal the number of
> actual inference invocations. See the explanation below before drawing any conclusions
> from the headline block.

The raw `ei_person` block from session 18:

```json
{
  "frames_submitted": 34,
  "frames_dropped_overflow": 0,
  "inferences_ok": 338,
  "inferences_err": 9,
  "mean_inference_ms": 10.11,
  "p95_inference_ms": 30.74,
  "mean_detections_per_frame": 0.0
}
```

### Quirk Q1 — counters gated by `debug=True` (A1 fix landed)

**Before A1:** `inferences_ok`, `inferences_err`, `inference_ms_samples`, and
`detections_per_frame` only accumulated when `cfg["debug"]==True`
(`main.py:1741-1761`). In sessions 16 and 17 (debug=False), those fields would have
been `None`. The A1 fix has now landed: counters always accumulate when EI is enabled,
regardless of debug mode. Session 18 was recorded under the old behaviour (debug=True),
so its numbers include the counters.

### Quirk Q2 — `inferences_ok` is HUD-rate, not invoke-rate (A2 not yet fixed)

**`frames_submitted=34`** = actual invocations (submit throttled 1-in-10, 347 frames ÷
10 ≈ 35). **`inferences_ok=338`** does NOT mean 338 separate inferences. The shared
result is read once per display loop iteration (≈30 Hz), not once per inference (≈3 Hz
at infer_interval=10). With 347 frames and approximately 1 read per frame, the shared
result is sampled roughly 347 times total, most of which re-read the same inference
result ~10×. `338 + 9 = 347 = frames_total` — this confirms it: every display frame
incremented exactly one counter.

**`inferences_err=9`** most likely reflects frames where the shared result was still
`stale` (EIWorker hadn't posted the first result yet, i.e., the first ~9 frames at
startup). Not 9 failed invocations.

**`mean_detections_per_frame=0.0`** is plausible: the model detected no persons during
this 55-second run (scene content unknown), or the threshold=0.8 was not met.

### Quirk Q3 — `mean_inference_ms` is honest

`mean_inference_ms=10.11` is the worker thread's own measurement of the TFLite invoke
call wall time. It is sampled per HUD tick but the value is the same measurement —
re-reading it does not bias the mean. This is a reliable signal: **~10 ms per inference
on RPi4 at INT8 quantization** with `num_threads=2`.

### Quirk Q4 — `p95_inference_ms` biased (A3 fix landed)

**Before A3:** p95 used `sorted(samples)[int(n*0.95)]`. With 338 samples this gives
index 321, close to the true p95. But the 338 "samples" are the same ~34 values
repeated ~10× — so the distribution is discrete and biased by the sampling pattern.
`p95_inference_ms=30.74` likely reflects the one or two slowest individual invocations
(maybe first-time model warm-up) rather than the true p95 of the inference distribution.
The A3 fix (now landed) gates p95 reporting on n≥20 unique samples, which will only be
met in debug=False runs (after A2 also lands). Treat this value with caution.

### Summary table (session 18, corrected interpretation)

| Metric | Raw value | Correct interpretation |
|--------|-----------|------------------------|
| `frames_submitted` | 34 | 34 actual inference submissions (1-in-10 of 347 frames) |
| `frames_dropped_overflow` | 0 | Queue never overflowed — good |
| `inferences_ok` | 338 | 338 HUD-loop reads of a non-stale result (~10× per actual invoke); **not 338 invocations** |
| `inferences_err` | 9 | 9 HUD-loop reads of a stale result (startup frames before first result) |
| `mean_inference_ms` | 10.11 ms | Reliable: true TFLite invoke time on RPi4 |
| `p95_inference_ms` | 30.74 ms | Unreliable: biased by display-rate sampling + small unique-sample count (A3+A2 needed) |
| `mean_detections_per_frame` | 0.0 | No persons detected at threshold=0.8 this session |

---

## Optimization critique

### A1 — Decouple EI counter accumulation from debug gate ✅ IMPLEMENTED

**File:** `src/smartbinocular/main.py` (block formerly at L1741-1767).
**Effect:** `mean_inference_ms`, `p95_inference_ms`, `mean_detections_per_frame` now
populate in session JSON for all EI-enabled runs, regardless of `debug` flag. HUD
overlay and bbox drawing remain debug-gated.
**Producer budget:** Unaffected (change runs on display thread).

### A2 — Dedupe `detections_per_frame` by `frame_id` ⬜ DEFERRED (separate ticket)

See verbatim ticket wording in `DECISIONS_AND_RISKS.md Q3`. Until this lands,
`inferences_ok` ≈ `frames_submitted × infer_interval` (≈ frames_total).

### A3 — Robust p95 in `metrics.py` ✅ IMPLEMENTED

**File:** `src/smartbinocular/metrics.py:finalize()`.
**Change:** `sorted(...)[int(n*0.95)]` → `numpy.percentile(..., 95, method="linear")`,
gated on `n ≥ 20`.
**Effect:** p95 is now `None` for short sessions or sessions with fewer than 20 latency
samples. For longer debug-True runs, it will be a less biased estimate.
**Producer budget:** Runs only at `finalize()`, zero hot-path cost.

### A4 — `infer_interval` vs `env_classification_interval`

`ei_person.infer_interval=10` submits at ~3 Hz (at 30 FPS). The ML inference interval
(`ml_inference_interval` or similar config key) is independent. No change recommended
unless profiling shows producer-side coupling. At `mean_inference_ms=10.11 ms`, the
3 Hz cadence is already conservative.

### A5 — EI input source

Live pipeline feeds `nir_raw` (raw NIR before any processing). Correct — do not feed
enhanced or fused frames. The offline harness reads sRGB JPGs; the resulting parity gap
is documented as R4 in `DECISIONS_AND_RISKS.md`.

### A6 — `draw_bbox` default mismatch ✅ IMPLEMENTED

**File:** `src/smartbinocular/config.py:239`.
**Change:** `"draw_bbox": True` → `"draw_bbox": False` to match
`EDGE_IMPULSE_FUTURE_WORK.md`. Behavior unchanged: HUD block is already gated on
`cfg["debug"]`, so bbox rendering only happens with debug=True regardless.

### A7 — `fuse_stage_timing_ms` / `thermal_stage_timing_ms` empty

Both are `{}` in all recent sessions. Either the instrumentation is disabled in the
`rpi_p3b` build or never wired. Recommendation: either re-enable the timer wiring or
remove the stub keys from `metrics.py finalize()` to avoid misleading empty dicts in
the JSON.
