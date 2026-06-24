# RPi4 Model Latency Protocol

Status: protocol only until run on Raspberry Pi 4 CPU. macOS numbers remain
proxy evidence and must not be used to claim the 20 FPS deployment target.

Run on the target Raspberry Pi 4 environment:

```bash
.venv/bin/python tools/rpi4_model_latency_benchmark.py \
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

The benchmark records:

- feature extraction latency,
- model inference latency,
- feature+predict latency,
- model load time,
- model size,
- peak Python memory from `tracemalloc` when feasible.

Interpretation rules:

- A non-RPi run is a proxy benchmark only.
- The 20 FPS deployment claim requires target-hardware feature+predict timing,
  not model-only timing.
- RF100/RF200 or feature-v2 migration is not approved by this protocol alone;
  production migration also requires manual sensor labels, schema versioning,
  and user confirmation.
