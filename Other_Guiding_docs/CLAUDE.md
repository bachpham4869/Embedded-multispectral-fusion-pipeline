# SmartBinocular — Agent Context

## What This Project Is

Real-time night-vision binocular system running on **Raspberry Pi 4B** (CPU-only, no GPU/NPU).
Fuses NIR camera (IMX290, 640×480, up to 60 FPS) + thermal sensor (Senxor MI48, 80×62, ~9 FPS)
into a single enhanced display with motion detection, environment classification, and HUD overlay.

## Hardware Constraints

- RPi4B quad-core ARM Cortex-A72 @ 1.8 GHz, NEON SIMD, 4 GB RAM
- Budget: ≤50 ms/frame end-to-end
- OpenCV NEON-accelerated ops (blur, warp, CLAHE) are cheap; memory bandwidth is the bottleneck
- No heavy CNNs; prefer classical CV, signal processing, lightweight ML (Random Forest, small MLP)

## Repository Layout

```
root/
  README.md             ← user-facing setup, hardware, pipeline overview
  CLAUDE.md             ← this file (may be gitignored in some clones)
  pyproject.toml        ← package definition, entry points, dependencies
  src/
    smartbinocular/
      __init__.py
      __main__.py       ← enables: python -m smartbinocular
      main.py           ← pipeline entry point, orchestrates all modules
      config.py         ← runtime config, hardware params, pipeline tuning knobs
      hardware.py       ← camera (Picamera2/libcamera) + thermal (MI48 SPI) drivers
      mi48_driver.py       ← standalone MI48 SPI driver (low-level register access)
      feature_extractor.py ← FeatureExtractor: computes FEATURE_SET_OPTICAL_ONLY from a frame
      thermal_pipeline.py  ← thermal preprocessing, background subtraction, Kalman BG model
      nir_pipeline.py      ← optical buckets A–F + HybridNIREnhancer (night paths)
      feature_schema.py    ← ENV_CLASSES, FEATURE_SET_* for ML training/inference
      ml_inference.py      ← background RF inference, MLSharedResult
      display_pipeline.py  ← fusion blend, display grading
      hud.py               ← HUD rendering (Layer 1 data overlay, always burned into saved frames)
      controls.py          ← soft-button bar hit-testing, mouse routing, GPIO button input (ButtonInput)
      recording.py         ← non-blocking video recorder (daemon thread + bounded queue)
      motion.py            ← sparse LK optical flow, jerk detection
      env_presets.py       ← environment classification FSM (NIGHT, GLARE, FOG, INDOOR, etc.)
      metrics.py           ← FPS, latency, session logging, JSON metrics export
      utils.py             ← FPSCounter, StreamSkewQualityGate, shared helpers
      assets/              ← static assets (icons, fonts)
      experimental/        ← in-progress research code (ei_person_in_dark.py); NOT imported by main pipeline
  scripts/
    fetch_weather_datasets.sh  ← dataset download helper
  tools/
    offline_pipeline.py   ← reads image datasets → FeatureRecord JSONL (ML training input)
    split_stratified_jsonl.py ← stratified train/test split on JSONL
    sweep_clahe_clip.py   ← offline CLAHE clip sweep with proxy IQA metrics
    sweep_kalman_qr.py    ← Q/R sweep on thermal .npy sequences
    threshold_sweep.py    ← ML confidence threshold sweep
    validate_schema.py    ← feature schema and bundle validation
    label_mapping.yaml    ← dataset-name → ENV_CLASS mapping
  docs/
    README.md               ← index of markdown docs in this folder
    research_report_*_optical_v4.md ← current ENV/optical policy reference
  legacy/
    README.md    ← index; archived plans + research v1–v3; what is stale vs worth reading
    md/          ← archived Markdown (moved from docs/ + root legacy): plans, thesis syntheses, ENV research drafts
    py/          ← archived Python: fusion_live_optimized.py (monolith), final_fusion.py, … DO NOT IMPORT
```

## Pipeline Data Flow

```
[MI48 SPI 80×62 ~9 FPS]        [IMX290 Picamera2 640×480 up to 60 FPS]
        ↓                                        ↓
ThermalProcessor                       NIR bucket dispatch (A–F via ENV_CLASS)
  • Kalman BG model (vectorized)         • Bucket A (normal_night / night_clear):
  • Soft Gaussian fg mask                    HybridNIREnhancer — CLAHE + weight map
  • MAD anomaly scoring (E1)               + Dark/Bright Ch (every update_rate frames)
  • colormap → heatmap (TURBO)           • Bucket B (fog / glare): CLAHE-only
        ↓                                • Bucket D: DCP dehaze
        ↓                                • Bucket E: temporal median
        ↓                                • Bucket C / F: anti-glare / passthrough
                    Fusion blend (homography warp + fg_mask alpha)
                    display_grade_and_cap_bgr() → luma_only cap
                              ↓
                         HUD overlay
                       cv2.imshow()
                              ↓
                    metrics.finalize() → session_<ts>.json
```

## Module Responsibilities (Quick Reference)

| Module | Responsibility | Key Classes |
| --- | --- | --- |
| `main.py` | Orchestrates frame loop, owns FrameCache | `main()` |
| `config.py` | All tuning constants, hardware addresses | `CONFIG` dict |
| `hardware.py` | Camera/thermal init and frame capture | `CameraManager`, `ThermalManager` |
| `thermal_pipeline.py` | Thermal preprocessing, bg model | `ThermalProcessor` |
| `nir_pipeline.py` | Optical bucket dispatch (A–F), NIR processors | `OPTICAL_BUCKET_DISPATCH`, `HybridNIREnhancer`, bucket helpers |
| `ml_inference.py` | RF load/infer in daemon thread | `EnvClassifier`, `MLInferenceThread`, `MLSharedResult` |
| `feature_schema.py` | ENV taxonomy + feature set definitions | `ENV_CLASSES`, `FEATURE_SET_OPTICAL_ONLY` |
| `display_pipeline.py` | Fusion and display grading | `display_grade_and_cap_bgr()` |
| `hud.py` | HUD overlay rendering (Layer 1 + chrome) | `HudState`, `ChromeState` |
| `controls.py` | Soft-button bar, mouse routing, GPIO buttons | `MouseRouter`, `ButtonInput` |
| `recording.py` | Non-blocking video capture (daemon thread) | `VideoRecorder` |
| `motion.py` | Sparse optical flow, stabilization | `MotionDetector` |
| `env_presets.py` | Environment FSM, preset switching | `EnvPresetController` |
| `metrics.py` | FPS, latency tracking, session export | `MetricsCollector` |
| `utils.py` | Shared helpers, FPS counter | `FPSCounter`, `StreamSkewQualityGate` |

## Entry Points

```bash
# Install in editable mode (required for src-layout imports)
pip install -e .

# Run
python -m smartbinocular
# or
smartbinocular
```

## Testing

```bash
# Preferred: use uv (uv.lock is present — creates .venv with Python 3.12)
uv sync                            # install all deps including dev/ml-tools
uv run pytest tests/               # always uses .venv Python

# Alternative (pip)
pip install -e ".[ml-tools]"       # optional: sklearn/joblib for ML tests & offline training
pip install pytest
pytest tests/
```

> **Note:** Use `.venv/bin/python` (Python 3.12 via uv) — system Python 3.14 has a scipy
> binary incompatibility (`_PyUnstable_Object_IsUniquelyReferenced`) that breaks ML tests.
> `.venv/bin/python -m pytest tests/` is always safe.

## ENV + ML (current implementation)

**Activate ML inference:**
```bash
ML_INFERENCE_ENABLED=1 ML_MODEL_PATH=models/production/env_classifier.joblib python -m smartbinocular
```
**Switch `env_mode`** (in `config.py` CONFIG dict or env var):
- `env_mode: "off"` — always uses `normal_night` → Bucket A (default, safe)
- `env_mode: "auto_rule"` — rule-based heuristic; if ML enabled + confident, ML drives ENV
- `env_mode: "manual"` — stays on the preset set via keyboard

- **`nir_pipeline.py`**: `OPTICAL_BUCKET_DISPATCH` maps each stabilized `ENV_CLASS` to buckets A–F. Full `HybridNIREnhancer` runs only for `night_clear` / `normal_night`; fog, rain, transition, etc. use lighter fixed paths (see `CHANGELOG.md` and `docs/research_report_*_env_policy_branching_optical_v4.md`).
- **`ml_inference.py`**: loads the RF from `ML_MODEL_PATH` (enable with `ML_INFERENCE_ENABLED`); inference runs on a **background thread** on an interval (`ML_INFERENCE_INTERVAL`, default every 15 frames). Bundle must match **`FEATURE_SET_OPTICAL_ONLY`** — **12** optical features (see `feature_schema.py`; feature #12 is `nir_blue_mean_ema`). `MLInferenceThread` applies `MLPosteriorEMA` (general α=0.55, asymmetric glare α) before posting `MLTop2` to `MLSharedResult`.
- **`main.py`**: with `env_mode=auto_rule`, high-confidence ML predictions drive desired env class (`ml_confidence_threshold=0.62`). `compose_env_from_ml_top2` implements the illumination-primary + weather-hint-overlay policy (10-rule compositor; invariant: when night or `normal_day` is top-1, env_class stays with illumination). Rule-based `infer_env_tags_auto_rule` is fallback when ML confidence is below threshold or ML is disabled.

## ML integration — future hook locations (reference only, NOT baseline wiring)

These are research extension points from `legacy/md/ML_HYBRID_SYSTEM_PLAN.md`. They are **not implemented** in the current pipeline — do not treat them as missing code to fill in:

1. **`thermal_pipeline.py`** — ML-refined anomaly scoring after `ThermalProcessor.process()`
2. **`nir_pipeline.py`** — ML-guided CLAHE clip inside `HybridNIREnhancer` (bucket A only)
3. **`env_presets.py`** — deeper fusion of rule tags + ML inside `EnvPresetController` (beyond current `auto_rule` routing)
4. **`display_pipeline.py`** — ML-driven fusion alpha after blend
5. **`main.py`** — E1 / anomaly gates using ML confidence

## Development Notes for Agents

- All imports are absolute: `from smartbinocular.config import ...`
- `legacy/md/ML_HYBRID_SYSTEM_PLAN.md` — historical Phase 1/2 checklist (Phase 2 complete, 183 tests green); treat as archive only
- The `legacy/` folder is read-only archival (`md/` + `py/` + index README) — never import pipeline code from it
- Target: ≤50 ms/frame total; profile before optimizing

## ML Thread Concurrency Model

The ML classifier is **non-blocking**: each frame extracts the **12-feature** optical vector (`FEATURE_SET_OPTICAL_ONLY`, incl. `nir_blue_mean_ema`) and posts it to a background daemon thread; the pipeline immediately continues using the **last known env label** (stale-safe by design). Never add synchronous waits on the ML thread inside the frame loop.

## NIR path and cost

Optical processing is **bucket-dispatched** by stabilized `ENV_CLASS` — not a single `HybridNIREnhancer` path for every frame. Only bucket **A** uses full `HybridNIREnhancer`; bright/day classes use anti-glare or passthrough (buckets B–F), which is the practical replacement for “skip NIR when bright” heuristics. See `OPTICAL_BUCKET_DISPATCH` in `nir_pipeline.py`.

## Heuristic Layer Status

Per accuracy testing, `normal_night` and `night_clear` ML classification is already reliable
(high accuracy). The heuristic pre-filter layer before ML may be unnecessary for those classes.
`phase_bright` (bright/day confusion) is the current weak point to watch.

## Offline Training Pipeline

- `tools/offline_pipeline.py` — reads image datasets (image2weather, darkface, exdark_street, glare, etc.) → extracts `FeatureRecord` JSONL; uses `_dummy_thermal()` zeros for optical-only datasets
- `data/training/` — training data: `from_logs_train.jsonl`, `from_logs_test.jsonl`, `merged_logs_ml.jsonl`
- `models/production/env_classifier.json` — production RF bundle sidecar (metrics, ECE, `ml_gate_reference`); `.joblib` often gitignored locally
- Retrain on Mac, deploy `.joblib` to RPi4 via rsync:
  ```bash
  rsync -avz models/production/env_classifier.joblib pi@raspberrypi.local:~/smartBinocular/models/production/
  ```

## Key CLI Flags and Workflows

```bash
# Lean mode (rule-based env, no ML, fastest path):
env_mode=off python -m smartbinocular

# Full ML mode (production model, auto_rule compositor):
ML_INFERENCE_ENABLED=1 ML_MODEL_PATH=models/production/env_classifier.joblib \
  env_mode=auto_rule python -m smartbinocular

# Offline feature extraction (run on Mac, not RPi):
python tools/offline_pipeline.py --by-label-dir data/weather/ --output data/training/from_logs_train.jsonl

# CLAHE clip sweep (Bucket B, proxy IQA metrics):
python tools/sweep_clahe_clip.py --input data/training/from_logs_test.jsonl --out-dir sweep_results/

# Retrain RF (Mac only):
python models/train_classifier.py \
    --mode optical_only \
    --dataset data/training/merged_logs_ml.jsonl \
    --output models/production/env_classifier.joblib

# Deploy model to RPi4:
rsync -avz models/production/env_classifier.joblib pi@raspberrypi.local:~/smartBinocular/models/production/
```

> **Performance knob**: `fusion_warp_work_scale` in CONFIG (default `0.5` when `rpi_throughput_max` merges `RPI_THROUGHPUT_MAX_DEFAULTS`).
> Reducing to 0.5 shrinks the blend buffer from 600×360 to 400×240, saving ~15 ms/frame on
> `fuse_blend_math` at the cost of slightly lower fusion resolution. Profile on device before committing.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **smartBinocular** (5694 symbols, 12233 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/smartBinocular/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/smartBinocular/context` | Codebase overview, check index freshness |
| `gitnexus://repo/smartBinocular/clusters` | All functional areas |
| `gitnexus://repo/smartBinocular/processes` | All execution flows |
| `gitnexus://repo/smartBinocular/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
