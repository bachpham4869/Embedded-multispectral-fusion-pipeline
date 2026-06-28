# SmartBinocular — Fusion Vision System

Real-time NIR + thermal fusion pipeline for a Raspberry Pi–based smart binocular. Combines a NIR camera (IMX290 via Picamera2) with a thermal camera (Senxor MI48 via SPI) into a fused display on an 800×480 panel.

| Topic | Context Overview & Target Scope |
| --- | --- |
| **Thesis & evidence register** | Traceability, measured stage timing; **RPi runbook:** Deployment configurations; *(archive)* Follow-up historical work |
| **Thesis & pipeline roadmap** | Active scope, core definitions, and technical limitations guidelines |
| **References (IEEE-style)** | Scientific gaps, cross-index calibration, and evaluation backlog documentation |
| **Offline ML** (datasets → JSONL → train) | Feature extraction logs, training datasets status, and thresholds $\tau_1/\tau_2$ ML gate rationale |
| **Deploy / reproducibility** | Environment dependency synchronization, model production registry, and LAB luma gate controls |

## Hardware Requirements
<img width="500" alt="Prototype_Binocular" src="https://github.com/user-attachments/assets/59feaec9-62fa-4ceb-b475-dd61e63502c0" />

- Raspberry Pi 4B (4GB+ recommended)
- Senxor MI48 thermal sensor (SPI/I2C, 80×62, ~9 FPS)
- NIR camera — IMX290 via Picamera2/libcamera (640×480, up to 60 FPS)
- 800×480 5" HDMI display

## Quick Start (device)

```bash
# Build tooling on the Pi must be recent enough for PEP 517 (setuptools.build_meta).
python3 -m pip install -U pip setuptools wheel
```
# Install the package and runtime deps (no PyPI package named "senxor" — MI48 uses built-in mi48_driver)
```text
pip install -e .
pip uninstall -y opencv-python-headless 2>/dev/null || true   # if present: headless has no imshow/namedWindow
pip install gpiozero smbus spidev
```
# Picamera2 is usually from the OS: sudo apt install -y python3-picamera2

# Run (calibration can be skipped if homography.json already exists — see below)
```text
python -m smartbinocular
```
The active codebase is in src/smartbinocular/. The original monolith is archived at legacy/py/fusion_live_optimized.py — do not import it. Older ML/plan docs and research v1–v3 live under legacy/md/ — see legacy/README.md (includes legacy/md/ML_PER_CLASS_CONFIDENCE_PLAN.md).

Real-time budget (design vs measured): The product targets low-latency fusion on RPi4; per-stage times depend on mode and build. Measured session summaries (when fusion_captures/metrics/session_*.json exist) are aggregated in docs/tables/timing/stage_timing_summary.md and interpreted in docs/PIPELINE_EVIDENCE_REGISTER.md §A.3 — do not treat a single “≤50 ms” line in this README as a guarantee without those artifacts.

Offline ML (Mac/Linux, not on RPi frame loop)
Prepare ENV classifier data and train a baseline Random Forest:

Bash
pip install -e ".[ml-tools]"
# 1) Feature JSONL per dataset — see tools/README.md (offline_pipeline, --by-label-dir)
# 2) Stratified train/test — tools/split_stratified_jsonl.py
# 3) Train (--dataset flag; run separately from eval):
```text
python models/train_classifier.py \
    --mode optical_only \
    --dataset data/training/merged_logs_ml.jsonl \
    --output models/production/env_classifier.joblib
```
# 4) Evaluate on disjoint held-out file (separate run, no --dataset needed):
```text
python models/train_classifier.py \
    --mode optical_only \
    --evaluate-model models/production/env_classifier.joblib \
    --test-dataset data/training/from_logs_test.jsonl
```
Feature set version: FEATURE_SET_OPTICAL_ONLY = 12 optical features (feature #12: nir_blue_mean_ema — B-channel mean EMA of nir_small_bgr; distinct from main.py's brightness nir_b_ema). Existing 11-feature bundles are rejected at load time (safe degrade to rule-based). Regenerate JSONL via tools/offline_pipeline.py before retraining.

ML routing (v4 backlog): MLInferenceThread applies per-class posterior EMA (MLPosteriorEMA, general α=0.55, asymmetric glare α) to stabilize top-1/top-2 before posting to MLSharedResult. compose_env_from_ml_top2 implements the illumination-primary + weather-hint-overlay policy (10-rule compositor): when night or normal_day is top-1, env_class stays with illumination; interference (fog/rain/glare) appears only as a small apply_secondary_hint overlay — never replaces env_class. See docs/CHANGELOG.md, docs/research_report_20260415_env_policy_branching_optical_v4.md, and docs/research_report_20260421_thesis_improvement.md (scientific gaps & calibration / evaluation backlog).

Details: tools/README.md.

Calibration
The homography matrix aligns the thermal sensor's coordinate space onto the NIR camera frame. It is stored in src/smartbinocular/assets/homography.json and is bundled with the package.

If assets/homography.json already exists → the calibration step can be skipped; the pipeline loads it automatically.

To regenerate (e.g. after physical re-mounting of sensors), run:

```bash
python3 -c "from smartbinocular import hardware; hardware.calibrate()"
The output file must be placed (or will be written) at src/smartbinocular/assets/homography.json.
```
Keyboard Shortcuts : 
```text
Key [ 1 ]: Switch to IMX/NIR mode

Key [ 2 ]: Switch to Thermal mode

Key [ 3 ]: Switch to Fusion mode

Key [ R ]: Toggle raw/processed view

Key [ S ]: Save current frame + metadata JSON

Key [ A ]: Auto-capture (5 second countdown)

Key [ + ] / [ = ]: Increase fusion blend alpha (+0.05)

Key [ - ]: Decrease fusion blend alpha (-0.05)

Key [ Q ]: Quit and save session metrics

Left-click on the display to set the A1 bearing measurement probe point.
```
# 5) Mode Descriptions : 
```text
 Mode 1 — IMX (NIR)
Displays the NIR camera feed. Automatic switching between raw and enhanced output based on a Schmitt trigger on the EMA brightness:

Raw mode: when mean brightness ≥ 30.0 (configurable via nir_schmitt_raw_on)

Enhanced mode: when mean brightness ≤ 18.0 (configurable via nir_schmitt_dim_on)

Hysteresis prevents flickering at boundary values

When in enhanced mode, HybridNIREnhancer applies: Dark/Bright Channel → Atmosphere Light estimation → Adaptive CLAHE → detail sharpening → color correction. In very dark conditions (night mode), the output is converted to grayscale for SNR boost.

Anti-glare (C4): when high-percentile pixels exceed threshold, A1-lite tone mapping (gamma + highlight roll-off) is applied.
```
```text
Mode 2 — Thermal (3DNR)

Displays the MI48 thermal feed processed through:

Temporal 3DNR (EMA filter, reduces thermal noise)

Background model (warmup 40 frames, adaptive update)

AGC (percentile stretch)

Edge enhancement (Laplacian)

Colormap (TURBO preferred, JET fallback)

Warmup progress is shown on-screen. The thermal_display_raw_mix parameter blends raw detail back in to prevent over-smoothing.
```
```text
Mode 3 — Fusion (NIR + Thermal)
NIR forms the background. Thermal heat map is:

Processed via ThermalProcessor

Colorized with the thermal colormap

Warped via homography H into NIR coordinate space

Blended only over foreground mask pixels (fg_mask from background model)

The blend weight is alpha * fusion_alpha_boost, controllable via +/- keys. When the background model is warming up, only the NIR feed is shown.

Key Features
Schmitt trigger NIR brightness: EMA-smoothed brightness with hysteresis prevents mode flickering

D2 JerkGate: Motion detection via frame difference on downscaled NIR; consecutive-frame requirement avoids single-frame false positives

S3 Temporal glare blend: IIR smoothing on glare-affected frames; reset on motion

S4 Luminance cap: Hard cap on LAB L channel to protect display brightness (configurable per ENV preset). Throughput-oriented builds may enable display_luma_cap_glare_gate so full BGR↔LAB cap is skipped when NIR/thermal glare flags are both clear — see legacy/md/DEPLOY_HARDENING.md §8 for the product trade-off.

S6 Stream skew gate: EMA-smoothed NIR/thermal timestamp skew with GOOD/DEGRADED/BAD states

A1 Bearing HUD: Pinhole model to compute azimuth/elevation offset for any clicked pixel

ENV presets: 14 named presets (night, fog, glare_heavy, backlight, etc.) applied with hysteresis

E1 detector: Lightweight local-stats anomaly detector on thermal for surveillance alerts

Capture metadata: Every saved PNG gets a companion JSON with full pipeline state

Output Files
Saved to ./fusion_captures/ (or ~/fusion_captures/ if cwd is not writable):

fusion_captures/<mode>_<timestamp>.png — captured frames

fusion_captures/<mode>_<timestamp>.json — per-capture metadata

fusion_captures/metrics/manifest_<session>.json — session manifest

fusion_captures/metrics/session_<timestamp>.json — end-of-session summary (stage_timing_ms per pipeline stage; when fusion sub-profiling is wired, fuse_stage_timing_ms breaks down fusion_composite inside the foreground-mask path)
```

# 6) Module Structure
```text
src/smartbinocular/
├── main.py              # Entry point — pipeline loop, keyboard handling, HUD
├── config.py            # CONFIG dict, optimization profiles, OpenCV runtime setup
├── hardware.py          # ThermalCaptureThread, NIRCaptureThread, homography load
├── mi48_driver.py       # Standalone MI48 SPI driver (low-level register access)
├── thermal_pipeline.py  # ThermalProcessor: 3DNR, AGC, edge enhance, colormap
├── nir_pipeline.py      # HybridNIREnhancer, Schmitt trigger, CLAHE, anti-glare
├── feature_extractor.py # FeatureExtractor: computes FEATURE_SET_OPTICAL_ONLY per frame
├── feature_schema.py    # ENV_CLASSES, FEATURE_SET_OPTICAL_ONLY (12 features), FeatureRecord
├── ml_inference.py      # EnvClassifier, MLInferenceThread, MLSharedResult (background RF)
├── display_pipeline.py  # LAB grade, luminance cap, display_grade_and_cap_bgr()
├── motion.py            # D2 JerkGate, S3 TemporalGlareBlend, S4 shake gate
├── metrics.py           # ThesisRunMetrics, JSON reporting, experiment context
├── env_presets.py       # ENV_PRESETS (14 named scenes), EnvPresetController
└── utils.py             # FPSCounter, StreamSkewQualityGate
```
# 7) Data Flow
```text
[MI48 SPI]                    [IMX290 Picamera2]
    │                                 │
ThermalCaptureThread          NIRCaptureThread
    │                                 │
    ▼                                 ▼
ThermalProcessor              HybridNIREnhancer
  3DNR → AGC → edge            Dark/Bright Ch → CLAHE
  → colormap (TURBO)           → sharpen → anti-glare
    │                                 │
    └──────────┬──────────────────────┘
               ▼
        Fusion blend (mode 3)
          homography warp H
          fg_mask from BG model
          alpha * fusion_alpha_boost
               │
               ▼
        display_grade_and_cap_bgr()
          LAB grade → L-cap
               │
               ▼
        HUD overlay (FPS, A1 bearing, S6 skew)
               │
               ▼
        cv.imshow("SmartBinocular")
               │
               ▼
        metrics finalize → session_<ts>.json
```
ML Integration Points
The RF environment classifier (ml_inference.py) is already live — it runs on a background daemon thread every 15 frames using 12 optical features (FEATURE_SET_OPTICAL_ONLY) and drives env_mode=auto_rule. Enable it with:

```bash
ML_INFERENCE_ENABLED=1 ML_MODEL_PATH=models/production/env_classifier.joblib python -m smartbinocular
Additional future hooks in main.py / pipeline modules:

ENV classification | ml_inference.py | Live — RF classifier, macro night F1=0.98 at τ=0.62

After ThermalProcessor output | thermal_pipeline.py | Future: object detection on thermal (YOLO-nano, MobileNet-SSD)

After HybridNIREnhancer output | nir_pipeline.py | Future: per-bucket parameter tuning

Inside EnvPresetController.tick() | env_presets.py | Future: deeper rule+ML fusion beyond current auto_rule

After fusion blend | main.py loop | Future: semantic segmentation overlay

E1 detector block | main.py loop | Future: replace local-stats detector with learned anomaly model

All pipeline frames are np.ndarray BGR uint8 at native resolution — standard OpenCV format, compatible with cv2.dnn, ONNX Runtime, and TFLite without conversion.
```
# 8) References 
[1] R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, 2nd ed. Cambridge, UK: Cambridge Univ. Press, 2004. (Referenced for homography matrix alignment calibration in hardware.py)

[2] K. He, J. Sun, and X. Tang, "Single Image Haze Removal Using Dark Channel Prior," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 12, pp. 2341-2353, Dec. 2011. (Basis for the HybridNIREnhancer atmosphere calculation in nir_pipeline.py)

[3] L. Breiman, "Random Forests," Machine Learning, vol. 45, no. 1, pp. 5-32, Oct. 2001. (Referenced for the background environment classifier architecture in ml_inference.py)

[4] S. M. Pizer et al., "Adaptive Histogram Equalization and Its Variations," Computer Vision, Graphics, and Image Processing, vol. 39, no. 3, pp. 355-368, Sep. 1987. (Referenced for Contrast Limited Adaptive Histogram Equalization (CLAHE) arrays)

