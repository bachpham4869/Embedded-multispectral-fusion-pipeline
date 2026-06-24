# ML Model Deploy Checklist

*Archived 2026-04+ — RPi field checklist complete; for day-to-day run/sync see [`../docs/RPi_RUNBOOK.md`](../../docs/RPi_RUNBOOK.md).*

Phase 3C.6 — Observe-only deployment (no preset consumer).  
See [`OFFLINE_ML_PLAN.md`](OFFLINE_ML_PLAN.md) Constraint [C6]: do NOT deploy to production until RPi validation passes.

---

## 1. Train and save the model (Mac)

These are two separate commands — training then evaluation:

```bash
# Step 1 — Train
python models/train_classifier.py \
    --mode optical_only \
    --dataset data/training/merged_logs_ml.jsonl \
    --output models/production/env_classifier.joblib

# Step 2 — Evaluate on disjoint held-out file (no training, no --dataset needed)
python models/train_classifier.py \
    --mode optical_only \
    --evaluate-model models/production/env_classifier.joblib \
    --test-dataset data/training/from_logs_test.jsonl
```

Expected: CV balanced accuracy ≥ 0.60 (0.75 target for RPi promotion).  
The `.json` sidecar is written alongside — check `models/production/env_classifier.json` for metrics.

---

## 2. Verify bundle metadata

```python
import joblib
b = joblib.load("models/production/env_classifier.joblib")
print(b["sklearn_version"])        # must match RPi sklearn
print(b["numpy_version"])          # must match RPi numpy
print(b["feature_set"])            # must be FEATURE_SET_OPTICAL_ONLY (12 features; #12 = nir_blue_mean_ema)
print(b["class_int_to_label"])     # {1: "night_clear", 2: "normal_night", ...}
print(b["training_mode"])          # "optical_only"
```

---

## 3. Pin sklearn + numpy versions

In `pyproject.toml`, keep **core** and **ML optional** deps aligned on Mac and RPi. NumPy lives under `[project]` (runtime + training stack); scikit-learn and joblib are under `ml-tools` (install: `pip install -e ".[ml-tools]"`).

```toml
[project]
dependencies = [
    "numpy>=1.26,<2",
    # ... opencv-python, etc.
]

[project.optional-dependencies]
ml-tools = [
    "scikit-learn>=1.6,<1.9",
    "joblib>=1.5,<2",
    # matplotlib, tqdm, pyyaml ...
]
```

Legacy `setup.py` mirrors only `[project]` `install_requires` (numpy, opencv-python) for older pip on Raspberry Pi OS — not the `ml-tools` extras.

Verify on RPi:

```bash
python -c "import sklearn, numpy; print(sklearn.__version__, numpy.__version__)"
```

If versions differ, retrain on Mac with the RPi versions active (via venv or conda).

---

## 4. Rsync model to RPi

```bash
# Adjust RPi_HOST to your device hostname or IP
RPi_HOST=smartbino.local
rsync -avz models/production/env_classifier.joblib \
    ${RPi_HOST}:~/smartBinocular/models/production/env_classifier.joblib
```

---

## 5. Configure on RPi

In `src/smartbinocular/config.py` (or via env vars):

```python
"ML_INFERENCE_ENABLED": True,
"ML_MODEL_PATH": "/home/pi/smartBinocular/models/production/env_classifier.joblib",
"ML_INFERENCE_INTERVAL": 15,   # every 15 frames ≈ once/second at 15 fps
```

Or without editing config:

```bash
export ML_INFERENCE_ENABLED=1
export ML_MODEL_PATH=/home/pi/smartBinocular/models/production/env_classifier.joblib
python -m smartbinocular
```

---

## 6. Validate on RPi (T032 checklist)

Run for 5 minutes in a representative environment:

- [ ] Pipeline starts without error
- [ ] `[ML] inference active` printed at startup
- [ ] FPS ≥ 20 fps sustained (run `debug=True` and watch FPS HUD)
- [ ] HUD shows `ML:<label> <confidence>` (requires `debug=True`)
- [ ] JSONL at `logs/ml/live_*.jsonl` contains `ml_env_label` and `ml_confidence` fields
- [ ] No preset changes caused by ML (env_presets remains rule-driven — confirm by running same scene with inference disabled and checking preset trace matches)
- [ ] No Python exceptions in terminal output
- [ ] Thread joins cleanly on Ctrl+C (no hang)

---

## 7. Notes

- Rationale for **global** `ml_confidence_threshold` / `ml_secondary_confidence_threshold` (calibration, sweeps, trade-offs): [`../docs/tables/ml/ML_GATE_RATIONALE.md`](../../docs/tables/ml/ML_GATE_RATIONALE.md).
- Models in `models/baseline/` are **not production-ready** — use `models/production/` for RPi.
- `models/**/*.joblib` is gitignored; only `models/production/env_classifier.json` (metrics sidecar) may be tracked.
- This deploy enables **observe-only** inference. The ENV→preset consumer wiring is a separate later phase.
- Constraint [C6]: never bypass RPi validation even if Mac accuracy looks good.
