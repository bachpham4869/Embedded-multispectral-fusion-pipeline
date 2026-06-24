# ECE — three night classes (OOF, isotonic)

**Source:** `models/baseline/rf_phase1_retrain_optical12.json` / production sidecar — `ece_by_night_class` (subset of `ece_by_class`).

Train: `merged_logs_ml.jsonl` (14094 rows), TimeSeriesSplit CV, `--reliability-scope all` for per-class figures.

| Class | ECE |
| --- | --- |
| night_clear | 0.0176 |
| normal_night | 0.0278 |
| nir_night | 0.0145 |

All &lt; 0.03 — calibration adequate for using scalar probability gates. See [ece_all_env_classes.md](ece_all_env_classes.md) and [threshold_sweep.csv](threshold_sweep.csv). Rationale for global τ₁/τ₂ (incl. night-focused macro-F1): [ML_GATE_RATIONALE.md](ML_GATE_RATIONALE.md).
