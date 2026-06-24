# ECE by ENV class (all `ENV_CLASSES`, OOF)

**Source:** `models/baseline/rf_phase1_retrain_optical12.json` — key `ece_by_class`.

**Train:** `data/training/merged_logs_ml.jsonl`, `optical_only`, isotonic calibration, `--reliability-scope all`.

| Class | ECE |
| --- | --- |
| night_clear | 0.0176 |
| normal_night | 0.0278 |
| normal_day | 0.0427 |
| fog | 0.0163 |
| rain | 0.0330 |
| glare | 0.0185 |
| backlight | 0.0119 |
| transition | 0.0210 |
| nir_night | 0.0145 |

**Worst class:** `normal_day` (0.0427) — still &lt; 0.05. Global gates in `config.py` are supported by this table + [threshold_sweep.csv](threshold_sweep.csv). See archived plan [../../../legacy/md/ML_PER_CLASS_CONFIDENCE_PLAN.md](../../../legacy/md/ML_PER_CLASS_CONFIDENCE_PLAN.md); current thesis traceability: [../../PIPELINE_EVIDENCE_REGISTER.md](../../PIPELINE_EVIDENCE_REGISTER.md).

**See also (why τ₁/τ₂ were chosen, sweep methodology):** [ML_GATE_RATIONALE.md](ML_GATE_RATIONALE.md).
