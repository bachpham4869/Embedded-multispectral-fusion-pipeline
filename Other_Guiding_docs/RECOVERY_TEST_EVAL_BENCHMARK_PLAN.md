# SmartBinocular recovery test/evaluation/benchmark plan

This document is a review-only recovery plan. It does not edit the manuscript. Every proposed test maps an input asset to a script or required new script, an expected output artifact, a claim that becomes supportable, and a claim that remains forbidden.

Evidence tags used below:

- `GENERATED_OFFLINE`: generated from paired/offline inputs, not captured by the live pipeline.
- `RUNTIME_CAPTURED`: captured by the running prototype, but may still be weakly paired.
- `TARGET_HW_PROFILED`: measured on target hardware under a named profile/session context.
- `SENSOR_PROXY_UNLABELED`: sensor-domain data exists, but no user-confirmed ground truth labels.
- `SURROGATE_SWEEP`: simulation, pseudo-sequence, or parameter sweep rather than real field validation.

## 1. Evidence asset inventory

| Asset group | Path(s) | What it contains | Reusable for which claim | Status |
|---|---|---|---|---|
| Thesis source/evidence registry | `HK252-DATN-142/thesis.tex`; `HK252-DATN-142/chapters/**`; `HK252-DATN-142/THESIS_EVIDENCE_INDEX.md`; `HK252-DATN-142/THESIS_FIGURE_MANIFEST.md`; `HK252-DATN-142/THESIS_STYLE_RULES.md`; `HK252-DATN-142/THESIS_CHAPTER_GUIDE.md`; `docs/thesis_eval/MANIFEST.md` | Manuscript source, evidence index, figure manifest, style rules, generated-evaluation manifest. | Reproducibility, claim-to-artifact linking, post-evidence manuscript edits. | Present; do not edit in this pass. |
| Q1 runtime fusion captures | `fusion_captures/`; `fusion_captures/metrics/*.json`; `fusion_captures/metrics_rpi_optimized/*.json`; `q1_results/` | Runtime images, sidecars, session metrics, and existing weak-pair Q1 audit output. | `RUNTIME_CAPTURED` visual/runtime feasibility, weak-pair fusion audit, partial timing context. | Present; existing Q1 audit reports 0 strict quantitative pairs. |
| Q1 paired offline inputs | `data/paired_data/imx_paired.mp4`; `data/paired_data/thermal_paired.mp4`; `data/paired_data/timestamps.csv`; `artifacts/paired_eval/*`; `docs/fusion/*` | Strict paired NIR/thermal videos, 584-row manifest, generated fusion metrics, failure cases, and fusion docs. | `GENERATED_OFFLINE` fusion utility and no-reference proxy comparison. | Present; strict manifest has no captured runtime fusion output paths. |
| Q1 homography/alignment | `src/smartbinocular/assets/homography.json`; `fusion_captures/metrics/manifest_*.json` | Static homography and runtime homography quality sidecar fields. | Alignment sanity check and capture-readiness gate. | Present; at least one manifest shows large corner drift and corners outside NIR frame. |
| Q2 still-image bucket eval | `data/eval/nir_val/manifest_v2.csv`; `data/eval/iqa_runs/round_2026-04-28.csv`; `docs/tables/iqa/*`; `docs/thesis_eval/bucket_dispatch/*`; `tools/run_nir_iqa_eval.py`; `tools/check_dispatch_consistency.py` | 270-frame NIR validation manifest, 1,620 forced-bucket rows, dispatch-consistency tables/scripts. | Offline adaptive dispatch proxy, fixed Bucket A comparison, rule/manifest consistency. | Present; still-image only. |
| Q2 sequence/sweep assets | `tools/sweep_rain_median_n.py`; `HK252-DATN-142/tables/ch6_evaluation/sweeps/rain_median_n_sweep.csv`; `docs/thesis_eval/open_questions/**` | Rain median sweep and transition/open-question outputs. | `SURROGATE_SWEEP` for rain/transition sensitivity. | Present as pseudo-sequence/surrogate; MISSING ASSET for real rain sequence. |
| Q3 MAD assets | `data/thermal/scaled_mi48_sequences/auto_clip`; `docs/thesis_eval/open_questions/tables/mad_anomaly_results.json`; `docs/thesis_eval/open_questions/figures/mad_anomaly_examples/**`; `tools/run_mad_anomaly_offline.py` | Scaled MI48 sequence, MAD smoke output, example anomaly thumbnails. | Indicator behavior smoke test and qualitative gallery. | Present; no ground truth labels found. |
| ML offline evidence | `docs/ml/*`; `docs/tables/ml/*`; `artifacts/ml/*`; `models/production/env_classifier.*` | RF200/RF100 offline cluster-aware benchmark, leakage checks, calibration/domain-shift docs, production model artifacts. | Offline duplicate-cluster-aware accuracy and model-readiness caveats. | Present; does not prove sensor-real accuracy or target latency. |
| ML sensor-domain assets | `data/sensor_eval/*`; `artifacts/ml/sensor_domain_shift/*`; `artifacts/ml/final_labeling_package/agent_manual_labels.csv`; `docs/tables/ml/agent_labeled_sensor_eval.md`; `tools/domain_shift_report.py`; `tools/validate_sensor_manual_labels.py`; `tools/evaluate_sensor_labeled_predictions.py` | Sensor feature JSONL/manifest, raw sensor predictions, 120-row user-label template, 24-row agent/manual-reviewed subset, domain-shift and labeled-eval scripts. | `SENSOR_PROXY_UNLABELED` audit now; preliminary agent-label warning evidence; sensor-real accuracy after user-confirmed labels are frozen. | Present; MISSING ASSET for user-confirmed labels. |
| ML/RPi latency assets | `tools/rpi4_model_latency_benchmark.py`; `docs/ml/RPI4_MODEL_LATENCY_PROTOCOL.md` | RPi4 feature+predict latency benchmark protocol. | `TARGET_HW_PROFILED` model deployment latency after running on RPi4. | Script present; MISSING ASSET for completed target-hardware run. |
| Latency/FPS timing assets | `fusion_captures/metrics/*.json`; `fusion_captures/metrics_rpi_optimized/*.json`; `docs/thesis_eval/timing_performance/*`; `tools/measure_stage_timing.py`; `tools/plot_stage_timing.py` | Session-level and stage-level timing tables/figures from target hardware. | Partial `TARGET_HW_PROFILED` runtime timing, not full acceptance. | Present; full mode matrix not complete. |
| Figures/tables | `HK252-DATN-142/figures`; `HK252-DATN-142/tables` | Source images and tables used in the thesis. | Figure rebuild/export plan and caveated captions. | Present; several embedded PDF figures are low DPI or screenshot-like. |
| Tests/manifests | `tests/test_*.py`; `tools/check_thesis_readiness.py` | 51 pytest modules found; manifest/model/session/evidence tests are available in the test tree. | Reproducibility smoke, registry integrity, model artifact integrity. | Present; run after approval because some tests may touch generated outputs/cache. |

## 2. Dangerous-zone recovery matrix

| Dangerous zone | Current weakness | Existing repo assets that help | Proposed test/eval/benchmark | Expected output artifact | Claim supported after running | Remaining caveat | Priority |
|---|---|---|---|---|---|---|---|
| Q1 Fusion utility | Strict evidence is generated offline from paired inputs; runtime fusion captures are not strict paired. | `data/paired_data/*`; `artifacts/paired_eval/*`; `docs/fusion/*`; `fusion_captures/`; `q1_results/`; `tools/evaluate_paired_fusion.py`; `tools/q1_fusion_eval.py` | Run dry-run paired audit and runtime-capture audit into `review_artifacts`. | `review_artifacts/q1_*` reports and updated dry-run logs. | `GENERATED_OFFLINE` fusion utility and `RUNTIME_CAPTURED` feasibility, separated. | Still no strict runtime fusion validation unless new capture includes synchronized NIR/thermal/fusion triples. | R0/R2 |
| Q1 Alignment/homography | Homography metadata exists, but strict manifest has no homography path and runtime manifest shows large drift in at least one session. | `src/smartbinocular/assets/homography.json`; `fusion_captures/metrics/manifest_*.json` | Alignment metadata audit and corner-drift gate. | `review_artifacts/alignment_audit.md` or JSON summary. | Homography-readiness and alignment-risk statement. | Not task-level alignment validation without annotated correspondences or target board capture. | R1/R2 |
| Q2 Adaptive bucket dispatch | Offline still-image cold-start is not full live temporal behavior; dispatch agreement is low. | `data/eval/nir_val/manifest_v2.csv`; `docs/tables/iqa/*`; `tools/run_nir_iqa_eval.py`; `tools/check_dispatch_consistency.py` | Forced-bucket dry-run and dispatch-consistency rerun; adaptive-vs-fixed table. | `review_artifacts/iqa_*` and `review_artifacts/dispatch_consistency_*`. | Offline forced-bucket utility and rule/manifest consistency. | No live sequence validation for rain/transition. | R0 |
| Q2 Bucket E/rain and transition | Rain evidence is pseudo-sequence/surrogate; transition needs hysteresis stability evidence. | `tools/sweep_rain_median_n.py`; `HK252-DATN-142/tables/ch6_evaluation/sweeps/rain_median_n_sweep.csv`; `docs/thesis_eval/open_questions/**` | Sequence audit using existing sweeps, plus new real rain/transition field protocol if missing. | `review_artifacts/q2_sequence_caveat.md`; later real-session CSV/plots. | `SURROGATE_SWEEP` sensitivity only, or live transition stability after new capture. | Current repo cannot prove Bucket E live rain behavior. | R1/R2 |
| Q3 MAD anomaly indicator | No ground-truth labels; cannot be called validated detector. | `data/thermal/scaled_mi48_sequences/auto_clip`; `docs/thesis_eval/open_questions/tables/mad_anomaly_results.json`; `tools/run_mad_anomaly_offline.py` | MAD smoke documentation now; 50-100 frame manual label benchmark next. | Existing MAD JSON/gallery; proposed `review_artifacts/mad_labeled_eval.*`. | Indicator behavior; after labels, preliminary detector metrics on a small labeled subset. | Without labels, no precision/recall/F1 validation. | R0/R1 |
| ML environment classifier | RF200/RF100 metrics are offline duplicate-cluster-aware; sensor-real labels and target latency are pending. | `docs/ml/*`; `docs/tables/ml/*`; `artifacts/ml/*`; `data/sensor_eval/*`; `models/production/env_classifier.*` | Readiness summary, sensor manual-label validation, labeled predictions eval, domain drift, RPi4 latency. | `review_artifacts/ml_*`; after approval, sensor-labeled and RPi4 latency artifacts. | Offline accuracy now; sensor-real accuracy only after labels; deployment latency only after RPi4 run. | Current sensor predictions are `SENSOR_PROXY_UNLABELED`. | R0/R1/R2 |
| Latency/FPS/real-time | Stage timing, FPS, design target, isolated microbench, and full mode-matrix acceptance can be conflated. | `fusion_captures/metrics/*.json`; `docs/thesis_eval/timing_performance/*`; `tools/measure_stage_timing.py`; `tools/plot_stage_timing.py` | Stage timing verification and full mode-matrix target-hardware protocol. | `review_artifacts/stage_timing_*`; later `artifacts/timing/full_mode_matrix/**`. | Partial `TARGET_HW_PROFILED` profiling in named sessions. | No full acceptance unless all modes/profiles/environments are captured. | R0/R2 |
| Reproducibility/evidence registry | Evidence index and manifest may be stale relative to thesis claims and generated artifacts. | `HK252-DATN-142/THESIS_EVIDENCE_INDEX.md`; `docs/thesis_eval/MANIFEST.md`; tests under `tests/` | Evidence freeze audit, manifest integrity tests, checksum/hash verification proposal. | `review_artifacts/evidence_freeze_checklist.md`; pytest log. | Traceable claim-to-artifact registry. | Reproducibility claim needs a passing frozen run and stable checksums. | R0/R1 |
| Figure/table quality | Several figures are low-DPI, screenshot-like, or too small in the compiled PDF. | `HK252-DATN-142/figures/**`; `HK252-DATN-142/tables/**`; compiled PDF image inventory. | Figure rebuild/export audit and prioritized redraw list. | `review_artifacts/figure_quality_audit.md`; rebuilt SVG/PDF/300-DPI raster assets after approval. | Better print readability and caveated captions. | Figure quality does not validate technical claims. | R0/R1 |

## 3. Concrete test specifications

### Q1-R0-STRICT-PAIRED

- Test ID: `Q1-R0-STRICT-PAIRED`
- Purpose: verify and, after approval, rerun strict paired generated-fusion evidence without confusing it with runtime capture.
- Input artifacts: `artifacts/paired_eval/strict_paired_manifest.csv`; `data/paired_data/imx_paired.mp4`; `data/paired_data/thermal_paired.mp4`; `data/paired_data/timestamps.csv`; `artifacts/paired_eval/*`; `docs/fusion/*`.
- Existing script path: `tools/evaluate_paired_fusion.py`.
- Proposed command:

```bash
python3 tools/evaluate_paired_fusion.py \
  --manifest artifacts/paired_eval/strict_paired_manifest.csv \
  --output-dir review_artifacts/q1_strict_paired_dryrun \
  --fusion-tables-dir review_artifacts/q1_strict_paired_tables \
  --paired-docs-dir review_artifacts/q1_strict_paired_docs \
  --fusion-docs-dir review_artifacts/q1_strict_paired_fusion_docs \
  --figures-dir review_artifacts/q1_strict_paired_figures \
  --dry-run
```

- Output file(s): dry-run report/log under `review_artifacts/q1_strict_paired_dryrun/`; after approval, regenerated paired-eval tables and figures under the chosen output directories.
- Metrics: frame pairing count, timestamp skew, entropy/log-RMS/saturation/crush, thermal foreground saliency, no-reference IQA deltas, failure-case rows.
- Acceptance criterion: manifest resolves the expected 584 strict paired rows and regenerated tables match existing row counts within an explained delta.
- Runtime estimate: dry-run under 2 minutes; full run may be longer and requires approval.
- Risk/caveat: this is `GENERATED_OFFLINE`; strict manifest currently has no captured runtime `fusion_output_path`.
- Thesis wording enabled if passed: "On 584 strict offline NIR/thermal pairs, generated fusion was evaluated with no-reference quality and saliency metrics."
- Thesis wording still forbidden even if passed: "The live runtime fusion pipeline was quantitatively validated" or "fusion improves task accuracy in field deployment."

### Q1-R0-RUNTIME-CAPTURE-AUDIT

- Test ID: `Q1-R0-RUNTIME-CAPTURE-AUDIT`
- Purpose: audit whether runtime-captured fusion assets support feasibility/visual demonstration claims.
- Input artifacts: `fusion_captures/`; `fusion_captures/*.json`; `fusion_captures/metrics/*.json`; existing `q1_results/`.
- Existing script path: `tools/q1_fusion_eval.py`.
- Proposed command:

```bash
python3 tools/q1_fusion_eval.py \
  --input fusion_captures \
  --out review_artifacts/q1_runtime_capture_audit \
  --strict-window-sec 1 \
  --qual-window-sec 20 \
  --bootstrap-iters 1000
```

- Output file(s): `review_artifacts/q1_runtime_capture_audit/q1_report.md`, tables, and example grids.
- Metrics: accepted nearest pairs, strict quantitative pair count, weak-pair quality deltas, foreground thermal saliency, saturation/crush, bootstrap intervals if available.
- Acceptance criterion: report explicitly states whether strict quantitative pairs are nonzero; if zero, only weak/runtime feasibility claims are allowed.
- Runtime estimate: under 10 minutes.
- Risk/caveat: existing `q1_results/q1_report.md` already indicates 0 strict quantitative pairs, so a rerun is expected to remain weak evidence unless new captures were added.
- Thesis wording enabled if passed: "Runtime fusion captures demonstrate that the prototype can produce fusion frames, but the available audit remains weakly paired."
- Thesis wording still forbidden even if passed: "Runtime fusion utility was quantitatively proven on synchronized NIR/thermal/fusion triples."

### Q1-R1-ALIGNMENT

- Test ID: `Q1-R1-ALIGNMENT`
- Purpose: determine whether available homography metadata is usable for an alignment sanity check and whether new calibration capture is mandatory.
- Input artifacts: `src/smartbinocular/assets/homography.json`; `fusion_captures/metrics/manifest_*.json`; sidecars in `fusion_captures/`.
- Existing script path: no dedicated alignment audit script found; write a small read-only script if approved, for example `tools/audit_homography_alignment.py`.
- Proposed command:

```bash
python3 tools/audit_homography_alignment.py \
  --homography src/smartbinocular/assets/homography.json \
  --manifests fusion_captures/metrics/manifest_*.json \
  --out review_artifacts/q1_alignment_audit.md
```

- Output file(s): `review_artifacts/q1_alignment_audit.md` and optional `review_artifacts/q1_alignment_audit.json`.
- Metrics: max corner drift, all-corners-within-NIR boolean, inlier count/ratio if present, thermal-to-NIR projected corner bounds.
- Acceptance criterion: all projected corners inside NIR frame and corner drift below a predeclared tolerance, for example less than 5-10 px for calibration-board captures or a justified larger threshold for display-scale sidecars.
- Runtime estimate: under 30 minutes including script creation.
- Risk/caveat: metadata-only audit does not prove object-level alignment without annotated correspondences or calibration-board images.
- Thesis wording enabled if passed: "Homography metadata passed a sanity check under the reviewed runtime sidecars."
- Thesis wording still forbidden even if passed: "Cross-modal alignment was validated at task level."

### Q2-R0-FORCED-BUCKET

- Test ID: `Q2-R0-FORCED-BUCKET`
- Purpose: verify the offline forced-bucket IQA evidence and prepare an adaptive-vs-fixed Bucket A comparison table.
- Input artifacts: `data/eval/nir_val/manifest_v2.csv`; existing `data/eval/iqa_runs/round_2026-04-28.csv`; `docs/tables/iqa/*`.
- Existing script path: `tools/run_nir_iqa_eval.py`.
- Proposed command:

```bash
python3 tools/run_nir_iqa_eval.py \
  --manifest data/eval/nir_val/manifest_v2.csv \
  --out-dir review_artifacts/q2_forced_bucket_dryrun \
  --dry-run
```

- Output file(s): dry-run output under `review_artifacts/q2_forced_bucket_dryrun/`; after approval, full forced-bucket CSV/table in the selected output directory.
- Metrics: per-environment mean quality score by bucket, adaptive-vs-fixed Bucket A delta, per-class sample count, confidence intervals if bootstrapped.
- Acceptance criterion: dry-run resolves manifest paths and the full approved run regenerates the expected 270 x bucket grid or explains any exclusions.
- Runtime estimate: dry-run under 2 minutes; full run likely under 1 hour but should be approved first.
- Risk/caveat: still-image cold-start evidence does not cover live temporal dispatch or rain/transition persistence.
- Thesis wording enabled if passed: "On the offline NIR validation manifest, forced-bucket IQA supports bucket-specific behavior relative to fixed Bucket A."
- Thesis wording still forbidden even if passed: "Adaptive dispatch is validated for live rain/transition sequences."

### Q2-R0-DISPATCH-CONSISTENCY

- Test ID: `Q2-R0-DISPATCH-CONSISTENCY`
- Purpose: rerun the manifest/rule consistency audit and quantify dispatch disagreement as an evidence caveat.
- Input artifacts: `data/eval/nir_val/manifest_v2.csv`; `docs/tables/iqa/dispatch_consistency.md`.
- Existing script path: `tools/check_dispatch_consistency.py`.
- Proposed command:

```bash
python3 tools/check_dispatch_consistency.py \
  --manifest data/eval/nir_val/manifest_v2.csv \
  --out review_artifacts/q2_dispatch_consistency_dryrun.md \
  --dry-run
```

- Output file(s): `review_artifacts/q2_dispatch_consistency_dryrun.md`; after approval, full consistency report/table.
- Metrics: agreement rate, per-class confusion/dispatch consistency, classes with zero agreement, reason codes if available.
- Acceptance criterion: report reproduces the low-agreement caveat and identifies which ENV labels need temporal or sensor-real clarification.
- Runtime estimate: under 5 minutes.
- Risk/caveat: consistency audit checks rules against manifest labels; it is not an accuracy benchmark.
- Thesis wording enabled if passed: "The dispatch rules and manifest labels disagree in specific classes, motivating live sequence validation."
- Thesis wording still forbidden even if passed: "The dispatcher is correct for those classes."

### Q2-R1-SEQUENCE

- Test ID: `Q2-R1-SEQUENCE`
- Purpose: separate true sequence behavior from pseudo-sequence rain/transition sweeps and define what can be salvaged now.
- Input artifacts: `tools/sweep_rain_median_n.py`; `HK252-DATN-142/tables/ch6_evaluation/sweeps/rain_median_n_sweep.csv`; `docs/thesis_eval/open_questions/**`.
- Existing script path: `tools/sweep_rain_median_n.py`; transition/hysteresis tables exist under `docs/thesis_eval/open_questions/**`.
- Proposed command:

```bash
python3 tools/sweep_rain_median_n.py \
  --manifest data/eval/nir_val/manifest_v2.csv \
  --out review_artifacts/q2_rain_median_n_sweep.csv
```

- Output file(s): `review_artifacts/q2_rain_median_n_sweep.csv` and `review_artifacts/q2_sequence_caveat.md`.
- Metrics: dispatch stability, bucket-switch rate, hysteresis transitions per minute, rain median window sensitivity, per-class confusion if real sequence labels exist.
- Acceptance criterion: if input rows remain `pseudo_sequence_noise_jitter`, classify the result as `SURROGATE_SWEEP` only; if real frame-order sequence exists, require timestamp continuity and environment labels.
- Runtime estimate: under 30 minutes for existing sweep; 1 day if a light real-sequence parser has to be written.
- Risk/caveat: current known rain sweep is pseudo-sequence, not field rain validation.
- Thesis wording enabled if passed: "A surrogate sweep explored sensitivity of Bucket E/rain parameters."
- Thesis wording still forbidden even if passed: "Bucket E was validated on real rain video."

### Q3-R0-MAD-SMOKE

- Test ID: `Q3-R0-MAD-SMOKE`
- Purpose: document current MAD behavior as an indicator smoke test and prevent detector-validation overclaim.
- Input artifacts: `data/thermal/scaled_mi48_sequences/auto_clip`; `docs/thesis_eval/open_questions/tables/mad_anomaly_results.json`; `docs/thesis_eval/open_questions/figures/mad_anomaly_examples/**`.
- Existing script path: `tools/run_mad_anomaly_offline.py`.
- Proposed command:

```bash
python3 tools/run_mad_anomaly_offline.py
```

- Output file(s): `docs/thesis_eval/open_questions/tables/mad_anomaly_results.json`; `docs/thesis_eval/open_questions/figures/mad_anomaly_examples/**`.
- Metrics: number of frames processed, number of frames flagged, candidate centroid/area/intensity summary if available, example gallery.
- Acceptance criterion: output is reproducible and clearly marked smoke/proxy/no-label.
- Runtime estimate: under 10 minutes, but run only after approval because the script writes into docs paths and has no dry-run/help guard.
- Risk/caveat: no labels; current output supports indicator behavior only.
- Thesis wording enabled if passed: "The MAD logic executed on a thermal sequence and produced candidate anomalies for qualitative inspection."
- Thesis wording still forbidden even if passed: "The MAD method is a validated detector" or "precision/recall/F1 were demonstrated."

### Q3-R1-MANUAL-LABEL

- Test ID: `Q3-R1-MANUAL-LABEL`
- Purpose: create the minimum labeled subset needed for preliminary MAD detector metrics.
- Input artifacts: `data/thermal/scaled_mi48_sequences/auto_clip`; current MAD results; optional example frames.
- Existing script path: no complete evaluator found; write a light script if approved, for example `tools/evaluate_mad_manual_labels.py`.
- Proposed command:

```bash
python3 tools/evaluate_mad_manual_labels.py \
  --labels review_artifacts/mad_manual_labels.csv \
  --predictions docs/thesis_eval/open_questions/tables/mad_anomaly_results.json \
  --out-json review_artifacts/mad_manual_eval.json \
  --summary review_artifacts/mad_manual_eval.md \
  --gallery review_artifacts/mad_manual_eval_gallery
```

- Output file(s): `review_artifacts/mad_manual_labels.csv`; `review_artifacts/mad_manual_eval.json`; `review_artifacts/mad_manual_eval.md`; qualitative gallery.
- Metrics: precision, recall, F1, false positives per minute, temporal persistence, centroid stability, false-negative examples.
- Acceptance criterion: 50-100 frames labeled with frame-level binary labels at minimum; stronger option adds point or circle annotations for centroid stability.
- Runtime estimate: 1 day including manual labeling.
- Risk/caveat: small labels are preliminary and may not generalize across scenes.
- Thesis wording enabled if passed: "A small manually labeled thermal subset provides preliminary MAD precision/recall/F1."
- Thesis wording still forbidden even if passed: "The detector is validated across operational environments."

### ML-R0-READINESS

- Test ID: `ML-R0-READINESS`
- Purpose: consolidate existing ML evidence and prevent offline metrics from becoming sensor-real deployment claims.
- Input artifacts: `docs/ml/*`; `docs/tables/ml/*`; `artifacts/ml/*`; `models/production/env_classifier.*`; `data/training_v3/cluster_aware_conservative_train.jsonl`; `data/training_v3/cluster_aware_conservative_test.jsonl`; `data/sensor_eval/raw_sensor_features.jsonl`.
- Existing script path: optional figure/table refresh via `tools/render_ml_curves.py` and `tools/domain_shift_report.py`; no rerun required for review-only summary.
- Proposed command:

```bash
python3 tools/domain_shift_report.py \
  --train data/training_v3/cluster_aware_conservative_train.jsonl \
  --test data/training_v3/cluster_aware_conservative_test.jsonl \
  --sensor data/sensor_eval/raw_sensor_features.jsonl \
  --model models/production/env_classifier.joblib \
  --scaler-group rgb \
  --out-json review_artifacts/ml_domain_shift_review.json \
  --summary review_artifacts/ml_domain_shift_review.md
```

- Output file(s): `review_artifacts/ml_readiness_summary.md`; optional domain-shift JSON/summary using the cluster-aware frozen split paths above.
- Metrics: accuracy, balanced accuracy, macro-F1, per-class support, calibration/ECE if probabilities exist, Brier score, domain drift, low-support class risk.
- Acceptance criterion: all reported metrics are tied to frozen artifact paths and explicitly labeled offline duplicate-cluster-aware.
- Runtime estimate: under 30 minutes if paths match; otherwise review-only summary from existing docs.
- Risk/caveat: RF200/RF100 metrics remain offline; sensor frames are unlabeled until user label audit completes. The existing 24-row agent/manual-reviewed subset is useful as warning evidence, not gold sensor accuracy.
- Thesis wording enabled if passed: "RF200/RF100 results are supported as offline duplicate-cluster-aware benchmark results."
- Thesis wording still forbidden even if passed: "The classifier achieved the same accuracy on sensor-real deployment data."

### ML-R1-SENSOR-LABEL

- Test ID: `ML-R1-SENSOR-LABEL`
- Purpose: turn the 120-row sensor manual-label template into a preliminary sensor-real accuracy benchmark.
- Input artifacts: `artifacts/ml/sensor_domain_shift/manual_label_template_autofilled.csv`; `artifacts/ml/sensor_domain_shift/raw_sensor_predictions.csv`; `data/sensor_eval/*`.
- Existing script path: `tools/validate_sensor_manual_labels.py`; `tools/evaluate_sensor_labeled_predictions.py`.
- Proposed command:

```bash
python3 tools/validate_sensor_manual_labels.py \
  --search artifacts/ml/sensor_domain_shift/manual_label_template_autofilled.csv \
  --out-md review_artifacts/ml_sensor_manual_label_validation.md \
  --out-json review_artifacts/ml_sensor_manual_label_validation.json

python3 tools/evaluate_sensor_labeled_predictions.py \
  --label-validation review_artifacts/ml_sensor_manual_label_validation.json \
  --predictions artifacts/ml/sensor_domain_shift/raw_sensor_predictions.csv \
  --out-json review_artifacts/ml_sensor_labeled_eval.json \
  --summary review_artifacts/ml_sensor_labeled_eval.md \
  --per-class review_artifacts/ml_sensor_labeled_per_class.md \
  --errors review_artifacts/ml_sensor_labeled_errors.csv \
  --fig-dir review_artifacts/ml_sensor_labeled_figures
```

- Output file(s): label validation JSON/MD, sensor-labeled eval JSON/MD, per-class table, error cases, figures.
- Metrics: accuracy, balanced accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, low-support class warnings.
- Acceptance criterion: labels are user-confirmed, non-empty, single-label per frame, and sufficient class support is reported.
- Runtime estimate: 1 day including human label review.
- Risk/caveat: 120 rows is small; low-support classes such as glare/backlight/transition may remain underpowered.
- Thesis wording enabled if passed: "A small manually reviewed sensor-real subset provides preliminary accuracy and error analysis."
- Thesis wording still forbidden even if passed: "The classifier is fully validated across all deployment environments."

### ML-R2-RPI4-LATENCY

- Test ID: `ML-R2-RPI4-LATENCY`
- Purpose: measure feature extraction plus prediction latency on target Raspberry Pi 4 hardware.
- Input artifacts: `models/production/env_classifier.joblib`; `data/sensor_eval/raw_sensor_features.jsonl`; `data/sensor_eval/frames_manifest.csv`; `tools/rpi4_model_latency_benchmark.py`; `docs/ml/RPI4_MODEL_LATENCY_PROTOCOL.md`.
- Existing script path: `tools/rpi4_model_latency_benchmark.py`.
- Proposed command:

```bash
python3 tools/rpi4_model_latency_benchmark.py \
  --model models/production/env_classifier.joblib \
  --features data/sensor_eval/raw_sensor_features.jsonl \
  --frames-manifest data/sensor_eval/frames_manifest.csv \
  --scaler-group rgb \
  --repeats 300 \
  --max-frames 300 \
  --hardware-label "Raspberry Pi 4 CPU" \
  --out-json artifacts/ml/rpi4_latency/run_manifest.json \
  --summary docs/tables/ml/rpi4_model_latency_summary.md
```

- Output file(s): `artifacts/ml/rpi4_latency/run_manifest.json`; `docs/tables/ml/rpi4_model_latency_summary.md`.
- Metrics: feature extraction latency, predict latency, total per-frame latency, mean/p50/p95/p99, model load time, CPU temperature/throttle if captured.
- Acceptance criterion: run is executed on Raspberry Pi 4 and records hardware label, repeats, warmup, model version/hash, and p95/p99 latency.
- Runtime estimate: under 1 hour on hardware.
- Risk/caveat: workstation run does not count as deployment latency.
- Thesis wording enabled if passed: "The RF inference path was profiled on Raspberry Pi 4 under the documented benchmark protocol."
- Thesis wording still forbidden even if passed: "The full camera loop meets the 50 ms target in all modes."

### LAT-R0-STAGE

- Test ID: `LAT-R0-STAGE`
- Purpose: verify existing session/stage timing tables and keep target-hardware profiling separate from full acceptance.
- Input artifacts: `fusion_captures/metrics/*.json`; `fusion_captures/metrics_rpi_optimized/*.json`; `docs/thesis_eval/timing_performance/*`.
- Existing script path: `tools/measure_stage_timing.py`; `tools/plot_stage_timing.py`.
- Proposed command:

```bash
python3 tools/measure_stage_timing.py \
  --session-dir fusion_captures/metrics \
  --out-dir review_artifacts/timing_stage_metrics

python3 tools/plot_stage_timing.py \
  --sessions fusion_captures/metrics \
  --out-dir review_artifacts/timing_stage_figures \
  --host RPi4
```

- Output file(s): `review_artifacts/timing_stage_metrics/**`; `review_artifacts/timing_stage_figures/**`.
- Metrics: n frames/sessions, warmup if available, mean, p50, p95, p99 if script supports it, FPS, dropped frames, thermal reuse rate, throttle status, active profile, hardware model.
- Acceptance criterion: tables explicitly state session count, mode/profile, hardware label, and whether p95/p99 are measured or unavailable.
- Runtime estimate: under 20 minutes.
- Risk/caveat: current session set is not a full mode/profile/environment matrix.
- Thesis wording enabled if passed: "Existing Raspberry Pi sessions provide partial target-hardware stage timing under named contexts."
- Thesis wording still forbidden even if passed: "All modes and profiles meet the real-time acceptance target."

### LAT-R2-FULL-MATRIX

- Test ID: `LAT-R2-FULL-MATRIX`
- Purpose: produce a full target-hardware acceptance benchmark for real-time claims.
- Input artifacts: runtime prototype, RPi4 camera/thermal stack, `tools/rpi_field_collect.py`, `tools/capture_paired_data.py`, current logging schema.
- Existing script path: `tools/rpi_field_collect.py`; `tools/capture_paired_data.py`; timing scripts above.
- Proposed command:

```bash
python3 tools/rpi_field_collect.py \
  --mode monitor \
  --outdir artifacts/timing/full_mode_matrix/<session_id> \
  --profile <raw|throughput|quality> \
  --duration-sec 300 \
  --notes "<environment, mode, hardware, camera settings>"
```

- Output file(s): `artifacts/timing/full_mode_matrix/**/session.json`; per-frame timing CSV/JSON; derived `docs/tables/timing/full_mode_matrix_summary.md`.
- Metrics: mode, profile, environment, n frames, warmup, mean, p50, p95, p99, FPS, dropped frames, thermal reuse rate, CPU throttle, temperature, hardware model, software commit/config hash.
- Acceptance criterion: NIR-only, thermal-only, and fusion each run across raw/throughput/quality profiles, with at least 300 seconds per session and no missing timing fields.
- Runtime estimate: 2-3 days including setup and capture.
- Risk/caveat: requires target hardware and field conditions; cannot be synthesized from current assets.
- Thesis wording enabled if passed: "A full mode-matrix benchmark on Raspberry Pi 4 supports the real-time claim under specified profiles."
- Thesis wording still forbidden even if passed: "The system is production-certified or robust in all weather/security conditions."

### REPRO-R0-FREEZE

- Test ID: `REPRO-R0-FREEZE`
- Purpose: freeze evidence registry before manuscript edits and verify manifest/model/session integrity.
- Input artifacts: `HK252-DATN-142/THESIS_EVIDENCE_INDEX.md`; `docs/thesis_eval/MANIFEST.md`; `tests/test_*.py`; `models/production/env_classifier.*`; generated docs/tables.
- Existing script path: `tools/check_thesis_readiness.py`; selected pytest tests.
- Proposed command:

```bash
python3 tools/check_thesis_readiness.py

python3 -m pytest \
  tests/test_thesis_manifest_integrity.py \
  tests/test_model_registry_integrity.py \
  tests/test_session_inventory.py \
  tests/test_fusion_evidence_readiness.py
```

- Output file(s): `review_artifacts/evidence_freeze_checklist.md`; pytest log; missing/stale artifact table.
- Metrics: missing artifact count, stale artifact count, model/config/dataset hash coverage, session JSON schema pass/fail, evidence-index coverage per major claim.
- Acceptance criterion: every thesis claim in the P0/P1 list links to a concrete artifact, and all selected integrity tests pass or have documented expected failures.
- Runtime estimate: under 2 hours after approval.
- Risk/caveat: if `check_thesis_readiness.py` assumes an older path layout, fix or run it only after confirming paths; do not edit manuscript during this pass.
- Thesis wording enabled if passed: "Major quantitative claims are traceable to frozen artifacts."
- Thesis wording still forbidden even if passed: "All experiments are independently reproducible without hardware and raw data."

### FIG-R0-QUALITY

- Test ID: `FIG-R0-QUALITY`
- Purpose: identify figures/tables that should be rebuilt before final submission.
- Input artifacts: compiled PDF image inventory; `HK252-DATN-142/figures/**`; `HK252-DATN-142/tables/**`; `HK252-DATN-142/THESIS_FIGURE_MANIFEST.md`.
- Existing script path: no dedicated figure-quality script found; use `pdfimages -list` and file metadata; write a small audit script only if needed.
- Proposed command:

```bash
pdfimages -list /Users/phongpham/Downloads/HK252-DATN-142_2252614_2252057.pdf \
  > review_artifacts/pdf_image_inventory.txt
```

- Output file(s): `review_artifacts/pdf_image_inventory.txt`; prioritized figure rebuild list.
- Metrics: embedded image DPI, pixel dimensions, file type, screenshot-like raster risk, caption caveat coverage.
- Acceptance criterion: diagrams are vector PDF/SVG where possible; raster images are at least 300 DPI at print size; small thermal/fusion thumbnails are enlarged or split.
- Runtime estimate: under 1 hour.
- Risk/caveat: better figure export improves readability but does not add technical validation.
- Thesis wording enabled if passed: "Figures are print-readable and captions identify generated/offline/surrogate evidence."
- Thesis wording still forbidden even if passed: "Figure quality validates the experiment."

## 4. Minimal emergency evidence plan

### Within 2 hours

- Freeze the evidence/caveat boundary in writing: `GENERATED_OFFLINE` for Q1 strict paired fusion, `RUNTIME_CAPTURED` only for weak-pair captures, `SENSOR_PROXY_UNLABELED` for sensor ML, `SURROGATE_SWEEP` for rain/transition sweeps, and partial `TARGET_HW_PROFILED` for timing sessions.
- Run only dry-run or read-only checks after approval: `Q1-R0-STRICT-PAIRED`, `Q2-R0-FORCED-BUCKET`, `Q2-R0-DISPATCH-CONSISTENCY`, and `FIG-R0-QUALITY`.
- Update the review artifacts, not the manuscript, with a claim-to-artifact ledger and exact forbidden wording.
- Check `THESIS_EVIDENCE_INDEX.md` and `docs/thesis_eval/MANIFEST.md` for stale or missing artifacts; list gaps rather than patching thesis text.

### Within 1 day

- Complete `ML-R1-SENSOR-LABEL` by manually reviewing the 120-row sensor label template, then run label validation and labeled prediction evaluation.
- Create `Q3-R1-MANUAL-LABEL` for 50-100 thermal frames and run a small preliminary MAD precision/recall/F1 benchmark.
- Add a lightweight alignment metadata audit script for `Q1-R1-ALIGNMENT`, then determine whether homography sidecars are defensible or require field recapture.
- Regenerate or rerun selected Q2 forced-bucket/dispatch tables after approval and prepare a per-class metric table.

### Within 2-3 days

- Capture strict runtime fusion triples: synchronized NIR-only, thermal, and live fusion frames with sidecars and homography quality metadata.
- Run `ML-R2-RPI4-LATENCY` on actual Raspberry Pi 4.
- Run `LAT-R2-FULL-MATRIX` for NIR-only, thermal-only, and fusion across raw/throughput/quality profiles.
- Capture real rain and transition sequences for Bucket E and Bucket F/hysteresis stability, or clearly demote those claims to future work.

## 5. Tests that should NOT be claimed as validation

- `Q1-R0-STRICT-PAIRED`: validates neither live runtime fusion nor task-level utility; it supports generated offline fusion utility only.
- `Q1-R0-RUNTIME-CAPTURE-AUDIT`: if strict pairs remain zero, it is a runtime feasibility and visual-demonstration audit, not quantitative fusion validation.
- `Q2-R0-DISPATCH-CONSISTENCY`: this is a rule/manifest consistency check, not dispatcher accuracy.
- `Q2-R1-SEQUENCE` with `pseudo_sequence_noise_jitter`: this is `SURROGATE_SWEEP`, not real rain validation.
- `Q3-R0-MAD-SMOKE`: this is a smoke test and qualitative indicator behavior check, not detector validation.
- `ML-R0-READINESS`: offline cluster-aware RF metrics do not establish sensor-real accuracy.
- `ML-R1-SENSOR-LABEL` on 120 rows: even if passed, this is preliminary small-subset evidence, not broad deployment validation.
- `ML-R2-RPI4-LATENCY`: model latency alone does not prove end-to-end full-loop real-time performance.
- `LAT-R0-STAGE`: existing stage timing supports profiled contexts only; it is not a full mode-matrix acceptance benchmark.
- `FIG-R0-QUALITY`: redrawn or high-resolution figures improve presentation only; they do not validate claims.

## 6. Recommended manuscript changes after evidence is generated

| If test passes | Section to update | New allowed wording | Figures/tables to add/update |
|---|---|---|---|
| `Q1-R0-STRICT-PAIRED` passes | Abstract, Chapter 3 RQ1, Chapter 6 Q1, Chapter 7 limitations | "Generated offline fusion was evaluated on 584 strict paired frames using no-reference quality and saliency metrics." | Add/update paired fusion metric table and failure-case figure with `GENERATED_OFFLINE` caption. |
| `Q1-R0-RUNTIME-CAPTURE-AUDIT` finds only weak pairs | Chapter 6 Q1, Chapter 7 future work | "Runtime fusion captures demonstrate feasibility, but strict synchronized runtime fusion validation remains pending." | Add runtime capture gallery only if caption says weak-pair/runtime feasibility. |
| `Q1-R1-ALIGNMENT` passes | Chapter 4 calibration, Chapter 6 Q1 caveats | "Available homography metadata passed a sanity check for the reviewed captures." | Add homography corner-projection table or alignment sidecar summary. |
| `Q2-R0-FORCED-BUCKET` passes | Chapter 6 Q2 | "Offline forced-bucket IQA supports bucket-specific behavior on the NIR validation manifest." | Add per-class adaptive-vs-fixed Bucket A table. |
| `Q2-R0-DISPATCH-CONSISTENCY` passes | Chapter 6 Q2 limitations | "Rule/manifest disagreement identifies classes requiring live temporal validation." | Add dispatch confusion/consistency table. |
| `Q2-R1-SEQUENCE` passes only as pseudo-sequence | Chapter 6 Q2, Chapter 7 limitations | "Rain and transition sweeps are surrogate sensitivity analyses, not live-sequence validation." | Add sweep table with `SURROGATE_SWEEP` caption. |
| `Q3-R1-MANUAL-LABEL` passes | Chapter 6 Q3, Chapter 7 conclusion | "A small manually labeled thermal subset provides preliminary MAD precision/recall/F1." | Add MAD labeled confusion table and example error gallery. |
| `ML-R1-SENSOR-LABEL` passes | Chapter 6 ML section, Abstract caveat if needed | "A manually reviewed sensor subset provides preliminary sensor-domain accuracy and error analysis." | Add sensor confusion matrix, per-class support table, error-case gallery. |
| `ML-R2-RPI4-LATENCY` passes | Chapter 6 ML latency and Chapter 7 deployment | "RF feature extraction and prediction were profiled on Raspberry Pi 4 under the documented protocol." | Add RF100/RF200 latency trade-off table. |
| `LAT-R2-FULL-MATRIX` passes | Abstract, Chapter 3 NFR, Chapter 6 timing, Chapter 7 conclusion | "Full-loop timing was profiled on Raspberry Pi 4 across the tested modes/profiles/environments." | Add full mode-matrix table with p50/p95/p99/FPS/drops/throttle. |

| If test fails/inconclusive | Section to update | Honest wording |
|---|---|---|
| Q1 strict paired rerun fails or cannot reproduce | Chapter 6 Q1 | "Fusion evidence is limited to existing generated/offline artifacts and cannot be regenerated under the current artifact set." |
| Q1 runtime audit still has 0 strict pairs | Abstract, Chapter 6 Q1, Chapter 7 | "Runtime fusion is demonstrated visually, while quantitative runtime fusion remains future work." |
| Q1 alignment audit fails | Chapter 4 calibration, Chapter 6 Q1 | "Available homography metadata indicates alignment risk; no strict alignment validation is claimed." |
| Q2 forced-bucket rerun fails | Chapter 6 Q2 | "Q2 evidence is limited to previously generated tables and should be framed as preliminary/offline." |
| Q2 sequence evidence remains pseudo-only | Chapter 6 Q2, Chapter 7 | "Bucket E/rain and Bucket F/transition behavior remain unvalidated in real temporal sequences." |
| Q3 label set is absent or too small | Chapter 3 RQ3, Chapter 6 Q3, Chapter 7 | "MAD is an anomaly indicator smoke test, not a validated detector." |
| Sensor labels are not confirmed | Chapter 6 ML, Chapter 7 | "Sensor-domain ML evidence is unlabeled domain-shift audit only." |
| RPi4 model latency is not run | Chapter 6 ML latency | "RF deployment latency is not measured on target hardware." |
| Full mode matrix is not run | Abstract, Chapter 3 NFR, Chapter 6 timing | "Timing evidence is partial target-hardware profiling, not full acceptance." |

## 7. Missing evidence collection protocol

### Strict runtime fusion capture protocol for Q1

- Need capture: synchronized NIR-only frame, thermal frame, and captured runtime fusion output for the same timestamp/frame ID.
- Minimum quantity: at least 300 strict triples per environment class used in Q1 claims; emergency minimum 100 triples across night/low-light and one mixed-thermal scene.
- Environments: night clear, low light, thermal foreground object, fog/haze if available, and at least one negative-control scene.
- Hardware settings: Raspberry Pi 4, fixed software commit, camera resolution/profile, thermal FPS, NIR FPS, active profile, homography file hash.
- File naming convention: `q1_runtime_<session_id>/<frame_id>_{nir,thermal,fusion}.png` plus `q1_runtime_<session_id>/<frame_id>.json`.
- Metadata to save: monotonic timestamp, wall-clock timestamp, mode, profile, exposure/gain if available, thermal scale/range, homography path/hash, homography quality, CPU temperature, throttle status, dropped frames.
- Command/logging to enable: use the runtime capture path that records sidecars; if collecting paired source streams, use `python3 tools/capture_paired_data.py --seconds 300 --outdir artifacts/q1_runtime_capture/<session_id> --save-raw-npz` where supported.
- Minimum acceptance criterion: at least 100 synchronized triples with timestamp skew under 100 ms, fusion output captured by runtime pipeline, and usable homography sidecars.

### Real rain/transition sequence protocol for Q2

- Need capture: frame-ordered NIR sequences with ENV labels for rain, transition, glare/backlight, night clear, and mixed edge conditions.
- Minimum quantity: 5 sessions per critical environment; at least 300 seconds per session or 1,000 frames per class if FPS is low.
- Hardware settings: same RPi/camera profile used for deployment; log active bucket, dispatch score, hysteresis state, selected profile, FPS.
- File naming convention: `q2_sequence_<env>_<session_id>/frame_<frame_id>.png` and `dispatch_<session_id>.csv`.
- Metadata to save: timestamp, ENV label, bucket chosen, rule features, RF probability if used, hysteresis state, rain median window state, manual reviewer note.
- Command/logging to enable: use runtime dispatch logging plus `python3 tools/rpi_field_collect.py --mode monitor --outdir artifacts/q2_sequences/<session_id> --duration-sec 300 --notes "<env>"`.
- Minimum acceptance criterion: per-class confusion/dispatch consistency table, transition switch rate, and hysteresis no-chatter criterion defined before analysis.

### MAD manual-label protocol for Q3

- Need capture: thermal frames with anomalies and non-anomalies, ideally raw/radiometric when available.
- Minimum quantity: emergency 50-100 labeled frames; stronger target 300-500 frames across 3 sessions.
- Label format: frame-level binary label at minimum; preferred point/circle annotation for object center and approximate radius.
- Environments: normal background, hot object, moving warm object, reflective false-positive scene, sensor noise/edge cases.
- File naming convention: `q3_mad_<session_id>/frame_<frame_id>.png` plus `mad_labels_<session_id>.csv`.
- Metadata to save: timestamp, thermal scale/range, sensor mode, object description, labeler ID, confidence, exclusion reason if ambiguous.
- Command/logging to enable: run current MAD script only after labels are frozen, then evaluate with a new `tools/evaluate_mad_manual_labels.py`.
- Minimum acceptance criterion: report precision, recall, F1, false positives per minute, temporal persistence, centroid stability, and failure examples.

### Sensor-real ML label protocol

- Need capture: user-confirmed labels for `artifacts/ml/sensor_domain_shift/manual_label_template_autofilled.csv` and additional frames if low-support classes remain weak.
- Minimum quantity: confirm existing 120-row template first; expand to at least 50 frames per critical class if possible.
- Environments: night clear, fog/haze, glare, backlight, rain, transition, low light, normal daylight.
- Hardware settings: same sensor stack as deployment; record camera settings and model/scaler hash.
- File naming convention: keep current frame IDs; add `label_source=user_manual_label`, `labeler`, `label_confidence`, `review_notes`.
- Metadata to save: ENV label, ambiguity flag, scene note, timestamp, session ID, camera mode, hardware label.
- Command/logging to enable: `python3 tools/validate_sensor_manual_labels.py` followed by `python3 tools/evaluate_sensor_labeled_predictions.py`.
- Minimum acceptance criterion: no missing labels in reviewed subset, per-class support reported, low-support classes explicitly caveated.

### Full latency/FPS mode-matrix protocol

- Need capture: NIR-only, thermal-only, and fusion runtime sessions across raw, throughput, and quality profiles.
- Minimum quantity: at least 300 seconds per mode/profile combination; emergency minimum 60 seconds per cell with explicit caveat.
- Environments: night, fog/haze, glare/backlight, rain if available, transition, and a stable indoor control.
- Hardware settings: Raspberry Pi 4 model/revision, OS version, camera settings, thermal sensor mode, CPU governor, cooling setup, software commit hash.
- File naming convention: `lat_<mode>_<profile>_<env>_<session_id>/session.json` plus per-frame timing CSV/JSON.
- Metadata to save: n frames, warmup count, mean, p50, p95, p99, FPS, dropped frames, thermal reuse rate, CPU throttle status, temperature, active profile, model version/hash.
- Command/logging to enable: `python3 tools/rpi_field_collect.py --mode monitor --outdir artifacts/timing/full_mode_matrix/<session_id> --profile <profile> --duration-sec 300 --notes "<mode/env>"`.
- Minimum acceptance criterion: all cells have complete metadata and p95/p99 timing; missing cells force wording to "profiled context" rather than "full real-time acceptance."
