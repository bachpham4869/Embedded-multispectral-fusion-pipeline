# Decisions and Risks — EI Person-in-Dark Eval Harness

> Spec version: 2026-04-30. This document is the contract between the eval harness
> (`tools/eval_ei_person.py`) and the session-metrics report. All tooling defaults
> listed here are authoritative over anything in the plan file.
>
> Producer constraint (non-negotiable): no change made here may make
> `python -m smartbinocular` heavier by default. ≤50 ms/frame budget is preserved.
>
> **Runtime model layout (live pipeline):** The production default is
> `models/ei/person_in_dark_fomo_int8.tflite` (overridable via `EI_PERSON_TFLITE_PATH`).
> Committed eval baselines under `docs/eval/ei_person_find_in_dark/` still record the
> historical Edge Impulse export path in `params.yaml` / `summary.json` for exact replay;
> use the same `.tflite` file copied into `models/ei/` for on-device runs.

---

## Resolved open questions

### Q1 — Primary metric

**Decision (staged):** Primary tuning metric = **centroid-in-GT-box hit rate**
(decision metric), gated by **image-level presence F1** (sanity gate). Padilla
mAP@0.5 / @0.75 = secondary, **trend-only**. Promote mAP to primary only if a future
non-FOMO model emits boxes spanning ≥30% of GT median person area; until then, IoU mAP
is structurally pessimistic by construction.

**Rationale:** FOMO emits single-cell boxes (`w=1/gw≈1/16, h=1/gh≈1/16`). IoU≥0.5 vs
full-person GT is physically unattainable. Image-level F1 answers "is anyone present?";
centroid-in-box answers localization without penalizing the FOMO geometry.

**Tooling defaults:**
- `tools/eval_ei_person.py --metric-primary centroid_hit` (default).
- `summary.json` always reports both metric families. Epoch winner is decided by
  centroid-hit unless `--metric-primary image_f1` overrides.

**Caveat (reproduced in every per-run README):**
> "Padilla mAP@0.5 numbers reported here are computed by synthesizing tiny boxes from
> FOMO cell centroids. They will be near-zero at any threshold and are useful only as a
> stable trend signal across epochs that share the same `box_mode`. Do not compare these
> to a YOLO/SSD baseline."

---

### Q2 — pipeline_lite v0 scope

**Decision:** **Identity-only v0.** Variants restricted to
`{crop, letterbox, passthrough} × {area, linear, nearest}`. No contrast bump, no
gamma, no CLAHE in v0.

**Rationale:** Identity establishes the only credible baseline for skew measurement.
Adding a contrast variant on day 1 conflates resize-policy parity with preprocessing
change. Any variant promoted to live must be sub-1 ms on RPi — that requires device
microbench, not a Mac eval.

**Trigger to enable a contrast variant in a later run:** centroid-hit recall plateaus
across all `{fit_mode × interp}` permutations AND the gap between hit-rate and
image-F1 (suggesting localization rather than detection is the bottleneck) is ≥0.05.

**Tooling defaults:**
- `--pipeline-lite-variant identity` (default). Other variants present as enum but
  raise `NotImplementedError` until the trigger fires.

---

### Q3 — Phase A code fixes (metrics decoupled from debug, p95)

**Decision:** Implement **A1** and **A3** in this work. Defer **A2**.

**A1 — decouple counter accumulation from debug gate** (`main.py:1741-1761`):
Metric counters (`inferences_ok/err`, `inference_ms_samples`, `detections_per_frame`)
are moved outside the `cfg["debug"]` gate. The HUD text overlay and bbox drawing remain
debug-gated. The change runs on the **display thread**, not the producer.
**Producer ≤50 ms/frame: unaffected.**

**A3 — robust p95** (`metrics.py:416`): Replace
`sorted(_inf_ms)[int(len(_inf_ms) * 0.95)]` with
`numpy.percentile(_inf_ms, 95, method="linear")`, gate on `n >= 20`.
Runs only in `finalize()` at session end. Zero live-path cost.

**A2 deferred — per-inference dedupe by `frame_id`:** Changes result-freshness
semantics; higher test-suite impact. Defer.

**Follow-up ticket wording (verbatim):**
> EI metrics: dedupe `detections_per_frame` and `inferences_ok/err` by
> `EIDetectionResult.frame_id`. Today these counters are appended once per HUD-loop
> iteration in `main.py` (currently at the same site as the metric accumulation block),
> repeat-counting each actual inference roughly `infer_interval` (≈10) times.
> Acceptance: in a 30-second `debug=False` run with `infer_interval=10` and
> `nir_capture_fps=30`, the new session JSON has `inferences_ok ≈ frames_submitted ± 2`
> and `len(detections_per_frame) ≈ frames_submitted`. No producer-side cost.

**Note on A2 residual:** After A1, sessions with `debug=False` will have
`detections_per_frame` sampled at display rate — the count reflects HUD reads, not
actual invocations. The report must state this limitation until A2 lands.

---

### Q4 — Train-time resize policy (resolved — definitive)

**Finding:** `person_in_dark-cpp-linux-v3-impulse-#2/model-parameters/model_metadata.h:99`
declares:
```c
#define EI_CLASSIFIER_RESIZE_MODE   EI_CLASSIFIER_RESIZE_FIT_SHORTEST
```
From `edge-impulse-sdk/classifier/ei_constants.h`:
```c
#define EI_CLASSIFIER_RESIZE_FIT_SHORTEST   1
#define EI_CLASSIFIER_RESIZE_FIT_LONGEST    2
#define EI_CLASSIFIER_RESIZE_SQUASH         3
```
`EI_CLASSIFIER_RESIZE_FIT_SHORTEST` = scale so the **shortest** side = 128, then
**center-crop** the longer side to 128. This is equivalent to: (1) center-crop the input
to a square using `min(H,W)` side, (2) resize to 128×128. This is exactly what the
production `_prepare_ei_input(fit_mode="crop")` already does.

`EI_CLASSIFIER_RESIZE_FIT_LONGEST` (not used here) = scale so the longest side = 128,
then pad the shorter side — that would be letterbox.

**There is NO training/serving mismatch on the resize step.** The production
`config.py "fit_mode": "crop"` is correct.

**Canonical baseline `--fit-mode = crop`.**

**Tooling defaults (updated from "no default until Q4 done"):**
- `tools/eval_ei_person.py --fit-mode crop` (default, now locked).
- `letterbox` and `passthrough` remain as optional sweep variants to measure what
  happens if the resize policy were different — useful for a skew study but not the
  primary baseline.
- No separate ticket needed for fit_mode correction (the production code is correct).

---

### Q5 — GT authority

**Decision:** Pascal VOC XML under
`data/find-person-in-the-dark/train/train/train_annotations/` (paired by stem to
`train/train/train_images/*.jpg`) is the **single source of truth** for image-level
positivity and centroid-in-GT-box hit. Padilla `groundtruths/` text is a derived format
consumed only when running `evaluate/evaluate/pascalvoc.py` for the secondary mAP step.

**Tooling defaults:**
- `tools/_ei_eval/discover.py` reads VOC XML only.
- `voc_to_padilla(records, out_dir)` writes the Padilla format on demand into
  `<run_id>/padilla_gt/` for the secondary mAP step; never used as a parallel GT.

---

## Risk register (refined)

| ID | Description | Severity | Mitigation | Residual risk |
|----|-------------|----------|------------|---------------|
| R1 | EI metric sampling bug — counters gated by `debug` in `main.py:1741-1761` | **High** | A1 implemented in this work: counters always accumulate when EI is enabled. Until A2 lands, **no report may claim `inferences_ok` equals actual invocations** (`inferences_ok` ≈ `frames_submitted × infer_interval` until A2). Place R1 cross-check above headline EI numbers in every report. | A1-only: `detections_per_frame` still display-rate sampled; documented. |
| R2 | FOMO geometry vs full-box GT | Med | Centroid-in-GT-box is the primary metric (Q1). Track `gt_count` distribution per epoch. Close-up / single-large-person frames bias recall up — document as distribution, not tooling error. | Not directly comparable to EI Studio confusion-matrix. |
| R3 | Letterbox not in production `_prepare_ei_input` (crop is correct per Q4) | Low | `FIT_SHORTEST` = center-crop, which is what production `fit_mode="crop"` does — no mismatch. Letterbox is available in the harness as a comparison variant only. If a future model switches to `FIT_LONGEST`, add letterbox to the production helper at that time. | None for current model. |
| R4 | Train/serve skew: NIR raw sensor frames vs sRGB JPG offline | **High** | Document parity limits in every per-run README: harness results are upper bounds assuming JPG distribution matches NIR statistics — they do not. Pipeline_lite v0 stays identity to make the gap measurable. | Reported harness metrics overstate real on-device accuracy. Recommend on-device capture-and-replay as a future ticket. |
| R5 | Scale: 15030 train images | Med | **Default `--limit 500`** (~5 min Mac-side, ~50 GT positives for stable F1). `--limit 2000` for near-final sweeps. `--limit 0` requires `--allow-full-run` (rejected by default in v0). Report 95% Wilson CI on F1 alongside point estimate. | Small-N variance on F1; CI addresses this. |
| R6 | VOC schema drift, duplicate stems, orphan XML/JPG | Med | `discover.py` validates first 10 records; fails loud on missing/non-numeric tags; reports duplicate stems and orphan files before sweep starts. | Unknown VOC variants surface via the validator. |
| R7 | Reproducibility | Low | `params.yaml` records `tflite_sha256`, numpy/tflite-runtime/cv2 versions (`importlib.metadata`), dataset root + `git rev-parse HEAD`, full `argv`, deterministic image ordering (sorted by stem). | None; numpy random not used. |
| R8 | Producer hot-path regression from A1+A3 | Low | A1 on display thread; A3 in `finalize()`. After change: `stage_timing_ms.framecache.mean_ms` and `stage_timing_ms.nir_bucket.mean_ms` must be within ±0.5 ms of a pre-change baseline over 30 s. | Negligible. |

---

## Evaluation scope (locked)

- **Source split: train only.** `data/find-person-in-the-dark/train/train/train_images/`
  paired with `data/find-person-in-the-dark/train/train/train_annotations/`.
- **No full-set runs.** Default `--limit 500`. `--limit 0` rejected unless
  `--allow-full-run` is passed (reserved; not enabled in v0).
- **Optional held-out sanity:** `--sanity-test-limit 0` (default off). Set e.g. `200`
  to run an additional pass on the first 200 images of `test/test/test_images/` under
  the same params; results written to `<run_id>/sanity/`.
- **No live device runs.** Mac-only. RPi runs are future work.
- **No model retraining.** Evaluation of the deployed bundle only.

---

## CLI canonical baseline command

```bash
pip install -e ".[ei]"

python tools/eval_ei_person.py \
  --tflite models/ei/person_in_dark_fomo_int8.tflite \
  --dataset data/find-person-in-the-dark/train/train \
  --pass raw \
  --threshold 0.8 \
  --fit-mode crop \
  --interp area \
  --limit 500 \
  --metric-primary centroid_hit \
  --run-id baseline_train500_crop_t080
```

`--fit-mode letterbox` is available as a comparison variant to measure the effect of
changing resize policy, but `crop` is the canonical baseline that matches training
(`EI_CLASSIFIER_RESIZE_FIT_SHORTEST`).

---

## Separate tickets (not in this work)

1. **A2: dedupe `detections_per_frame` and `inferences_ok/err` by `frame_id`** — see
   verbatim ticket wording in Q3 above.
2. **On-device capture-and-replay** — capture raw NIR frames on RPi4, feed them to the
   offline harness to close the R4 sRGB/NIR distribution gap.
