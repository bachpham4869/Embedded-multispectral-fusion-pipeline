# Training Data — Eval-Split Policy

**IMPORTANT: Read before running any sweep or training script.**

## File roles

| File | Role | May be touched during... |
|------|------|--------------------------|
| `from_logs_train.jsonl` | Training set (full) | Phase 4 retraining; Phase 1 sweeps against val split only |
| `from_logs_train_a.jsonl` | Training sub-split (85%) | Phase 4 retraining after val split created |
| `from_logs_val.jsonl` | Validation split (15%) | Phase 3 CLAHE sweep; Phase 3 batch IQA sweep |
| `from_logs_test.jsonl` | **HELD-OUT TEST SET — EVAL ONLY** | Phase 4 final evaluation (ONCE per model) |
| `merged_logs_ml.jsonl` | Merged full dataset | Production model retraining only |

## Quarantine rules

- `from_logs_test.jsonl` is the **held-out evaluation set**. It must be touched **at most once
  per model variant** — at the very end of Phase 4, after all hyperparameter decisions are frozen.
- **Never** run `tools/threshold_sweep.py` or `tools/secondary_threshold_sweep.py` against
  `from_logs_test.jsonl` during development. Use `from_logs_val.jsonl` for any iterative sweeps.
- `tools/sweep_clahe_clip.py` and `tools/batch_nir_enhancer.py` operate on image files, NOT these
  JSONL files. They use the image val set under `data/eval/nir_val/` (see Phase 3).

## Creating the val split

If `from_logs_val.jsonl` does not yet exist, run once:

```bash
python tools/split_stratified_jsonl.py \
    --input data/training/from_logs_train.jsonl \
    --train-ratio 0.85 \
    --seed 42 \
    --train-out data/training/from_logs_train_a.jsonl \
    --test-out data/training/from_logs_val.jsonl
```

The split was run on 2026-04-27 with seed=42. Resulting file hashes:
- `from_logs_val.jsonl` SHA-256: `b45ee0e4c695ca7f9672935aeacdca9e2997a745233b5fc1c0656080b29b4797`
- `from_logs_train_a.jsonl` SHA-256: `bb394196d498ce7e3e6508f8fd67657c9826431276519405724574caf4a6478a`

Val split distribution: 1799 samples across 9 classes (backlight=47, fog=197, glare=51,
night_clear=355, nir_night=307, normal_day=312, normal_night=299, rain=165, transition=66).

## Leakage risk

`tools/threshold_sweep.py` and `tools/secondary_threshold_sweep.py` both default their `--input`
argument to `data/training/from_logs_test.jsonl`. **Always pass `--input data/training/from_logs_val.jsonl`**
during Phase 3 development sweeps. The default is intentionally not changed to avoid breaking
existing usage — this README is the guardrail.
