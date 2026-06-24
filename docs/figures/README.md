# docs/figures — thesis & ML figures

| Folder | Contents |
|--------|----------|
| [`iqa/`](iqa/) | Offline IQA appendix (stratified examples, links to tables) |
| [`ml/reliability/`](ml/reliability/) | Per-class reliability diagrams (production + phase1 stems) and composed strips |
| [`ml/ablation/`](ml/ablation/) | Reliability PNGs for 9-feature vs 12-feature ablation models (`rf_optical_*`) |

Composed night / all-class figures live under `ml/reliability/` (`reliability_night_classes.png`, `reliability_all_env_classes.png`).

Regenerate composites: `python tools/compose_reliability_figure.py --help` (defaults point at `ml/reliability/`).
