# ML reliability diagram sources

Per-class **reliability diagram PNGs** for the `rf_phase1_retrain_optical12` training run (OOF isotonic RF). Filenames: `rf_phase1_retrain_optical12_reliability_<ENV_CLASS>.png`.

They are the **input** to `tools/compose_reliability_figure.py` (default `--src` = this directory) to build:

- `reliability_night_classes.png`
- `reliability_all_env_classes.png`

(both in this folder).

Regenerate per-class PNGs with `python models/train_classifier.py --reliability-plots ...` when retraining, then re-copy here if you want to refresh composites.
